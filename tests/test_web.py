import base64
import io
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
import torch
from PIL import Image

from tiny_vlm.config import TextConfig, VisionConfig, VLMConfig
from tiny_vlm.model import TinyVLM
from tiny_vlm.tokenizer import TinyTokenizer
from tiny_vlm.web import MAX_REQUEST_BYTES, ModelRuntime, create_server, decode_image


@pytest.fixture
def image_data():
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture
def web_server(tmp_path):
    tokenizer = TinyTokenizer.train(["hello world what color red blue a circle", "你好，描述图片。"], vocab_size=300)
    cfg = VLMConfig(vision=VisionConfig(img_size=32, d_model=16, depth=1, num_heads=2),
                    text=TextConfig(vocab_size=tokenizer.vocab_size, max_seq_len=96,
                                    d_model=16, depth=1, num_heads=2))
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fixture")
    runtime = ModelRuntime(TinyVLM(cfg).eval(), tokenizer, checkpoint, torch.device("cpu"))
    server = create_server(runtime, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", runtime
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def request(base, path, payload=None, headers=None, raw=None):
    body = json.dumps(payload).encode() if payload is not None else raw
    req = Request(base + path, data=body, headers=headers or ({"Content-Type": "application/json"} if body else {}))
    try:
        with urlopen(req, timeout=10) as response:
            return response.status, response.read(), response.headers
    except HTTPError as exc:
        return exc.code, exc.read(), exc.headers


def test_assets_status_and_no_arbitrary_file_access(web_server):
    base, _ = web_server
    for path, content in (("/", "一个模型，三种输入"), ("/app.js", "api/generate"),
                          ("/styles.css", "@media")):
        code, body, headers = request(base, path)
        assert code == 200 and content in body.decode()
        assert headers["X-Content-Type-Options"] == "nosniff"
    code, body, _ = request(base, "/api/status")
    data = json.loads(body)
    assert code == 200 and data["ready"] and data["modes"] == ["text", "image", "mixed"]
    assert request(base, "/../../pyproject.toml")[0] == 404
    assert request(base, "/api/example-image")[0] == 404
    assert json.loads(request(base, "/api/examples")[1])["examples"][0]["mode"] == "text"


@pytest.mark.parametrize("mode", ["text", "image", "mixed"])
def test_real_model_generation_for_each_mode(web_server, image_data, mode):
    base, _ = web_server
    code, body, _ = request(base, "/api/generate", {
        "mode": mode, "prompt": "hello" if mode != "image" else "",
        "image": image_data if mode != "text" else None,
        "parameters": {"max_new_tokens": 3, "temperature": 0, "top_k": 10},
    })
    data = json.loads(body)
    assert code == 200 and data["mode"] == mode and isinstance(data["text"], str)
    assert 1 <= data["output_tokens"] <= 3 and data["elapsed_ms"] >= 0
    assert data["image_tokens"] == (4 if mode != "text" else 0)


@pytest.mark.parametrize("payload", [
    {}, {"mode": "image"}, {"mode": "mixed", "prompt": "hello"},
    {"mode": "text", "prompt": " "}, {"mode": "wrong", "prompt": "hello"},
    {"prompt": "hello", "parameters": {"max_new_tokens": True}},
    {"prompt": "hello", "parameters": {"temperature": float("nan")}},
    {"prompt": "hello", "parameters": {"top_k": 0}},
    {"mode": "image", "image": "data:image/png;base64,not an image"},
    {"prompt": "hello" * 1000}, [],
])
def test_bad_requests_have_readable_errors(web_server, payload):
    base, _ = web_server
    code, body, _ = request(base, "/api/generate", payload)
    assert code == 400 and json.loads(body)["error"]


def test_concurrent_generation_returns_busy_and_releases_lock(web_server):
    base, runtime = web_server
    runtime.lock.acquire()
    try:
        code, body, _ = request(base, "/api/generate", {"prompt": "hello"})
        assert code == 409 and "稍后" in json.loads(body)["error"]
    finally:
        runtime.lock.release()
    code, _, _ = request(base, "/api/generate", {"prompt": "hello", "parameters": {"max_new_tokens": 1}})
    assert code == 200 and not runtime.lock.locked()


def test_request_size_content_type_and_invalid_json(web_server):
    base, _ = web_server
    assert request(base, "/api/generate", raw=b"{}", headers={"Content-Type": "text/plain"})[0] == 415
    assert request(base, "/api/generate", raw=b"bad json", headers={"Content-Type": "application/json"})[0] == 400
    assert request(base, "/api/generate", raw=b"{}", headers={
        "Content-Type": "application/json", "Content-Length": str(MAX_REQUEST_BYTES + 1)})[0] == 413


def test_phone_photo_orientation_is_applied():
    image = Image.new("RGB", (20, 30), "red")
    exif = image.getexif()
    exif[274] = 6
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    decoded = decode_image("data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode())
    assert decoded.size == (30, 20) and decoded.mode == "RGB"
