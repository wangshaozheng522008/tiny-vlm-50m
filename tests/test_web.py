"""Local web server: text-only API, multi-turn history, no image surface."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest
import torch

from minillm.config import MINILLM_NANO
from minillm.model import MiniLLM
from minillm.train import save_model_checkpoint
from minillm.web import serve


@pytest.fixture()
def server(tmp_path, tiny_tokenizer):
    cfg = MINILLM_NANO.copy()
    cfg.vocab_size = tiny_tokenizer.vocab_size
    model = MiniLLM(cfg).eval()
    ckpt = tmp_path / "model.pt"
    save_model_checkpoint(ckpt, model, tiny_tokenizer, step=0, tokens=0, val_loss=None)
    tiny_tokenizer.save(tmp_path / "tokenizer.json")
    httpd = serve(str(ckpt), str(tmp_path / "tokenizer.json"), port=0, device="cpu",
                  name="test-model")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=10) as response:
        return response.status, response.read()


def post(base, path, payload):
    request = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_index_is_text_only_chat(server):
    status, body = get(server, "/")
    assert status == 200
    html = body.decode()
    assert "MiniLLM" in html and "对话" in html
    assert "image" not in html.lower() and "图片上传" not in html
    assert "/static/app.js" in html and "/static/styles.css" in html


def test_static_assets_and_traversal_guard(server):
    status, body = get(server, "/static/app.js")
    assert status == 200 and b"api/chat" in body
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        get(server, "/static/../web.py")
    assert excinfo.value.code == 404


def test_info_reports_params_and_checkpoint(server):
    status, body = get(server, "/api/info")
    info = json.loads(body)
    assert status == 200
    assert info["params"] > 0 and info["trained"] is False
    assert info["checkpoint_bytes"] > 0 and info["quantized"] is None
    assert info["max_context"] == MINILLM_NANO.max_position_embeddings


def test_chat_returns_reply_and_timings(server):
    status, data = post(server, "/api/chat", {
        "message": "你好", "max_new_tokens": 6, "temperature": 0.0,
    })
    assert status == 200
    assert isinstance(data["reply"], str)
    assert data["tokens"] <= 6
    for key in ("ttft_ms", "prefill_tok_s", "tok_s", "total_ms", "prompt_tokens"):
        assert isinstance(data[key], (int, float)) and data[key] >= 0
    assert data["history_turns"] == 1


def test_multi_turn_history_and_reset(server):
    post(server, "/api/chat", {"message": "第一句", "max_new_tokens": 2})
    _, second = post(server, "/api/chat", {"message": "第二句", "max_new_tokens": 2})
    assert second["history_turns"] == 2
    assert second["prompt_tokens"] > 0
    status, _ = post(server, "/api/reset", {})
    assert status == 200
    _, third = post(server, "/api/chat", {"message": "重新开始", "max_new_tokens": 2})
    assert third["history_turns"] == 1


def test_context_limit_drops_old_turns(server):
    for index in range(3):
        post(server, "/api/chat", {"message": f"第{index}轮" + "很长的内容" * 30,
                                   "max_new_tokens": 2, "context_limit": 256})
    _, data = post(server, "/api/chat", {"message": "最后一轮", "max_new_tokens": 2,
                                          "context_limit": 256})
    assert data["context_limit"] <= MINILLM_NANO.max_position_embeddings
    assert data["dropped_turns"] >= 1


def test_empty_message_is_rejected(server):
    status, data = post(server, "/api/chat", {"message": "   "})
    assert status == 400 and "empty" in data["error"]


def test_unknown_route_404(server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        get(server, "/api/does-not-exist")
    assert excinfo.value.code == 404
