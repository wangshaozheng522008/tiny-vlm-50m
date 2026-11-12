"""Local text-only chat UI for MiniLLM (stdlib HTTP server, no extra deps).

    python -m minillm.web --ckpt runs/minillm/model-int4.pt \
        --tokenizer runs/minillm/tokenizer.json --port 8765

The server binds 127.0.0.1 by default and serves a single-page chat client.
Multi-turn history, the system prompt, temperature/top-k/max-token/context-limit
controls, per-reply TTFT/prefill/decode timings, clear and export are handled
here. There is no image input anywhere.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from .config import MiniLLMConfig
from .model import MiniLLM
from .quantize import dequantize_model
from .tokenizer import DEFAULT_SYSTEM, END_OF_TURN_ID, MiniLLMTokenizer
from .utils import describe_device, get_device, human_bytes, set_seed
from .text_policy import checkpoint_text_policy

ASSETS = Path(__file__).with_name("web_assets")
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
}


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[MiniLLM, dict]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("architecture") != "minillm":
        raise SystemExit(
            f"{path}: not a MiniLLM checkpoint. TinyVLM (vision-language) checkpoints are "
            "not supported by the text-only server."
        )
    if "quantized" in payload:
        model = dequantize_model(payload)
    else:
        model = MiniLLM(MiniLLMConfig.from_dict(payload["config"]))
        model.load_state_dict(payload["model"])
    return model.to(device).eval(), payload


class ChatEngine:
    """Single-session chat state plus token-by-token decoding for accurate timings."""

    def __init__(self, model: MiniLLM, tokenizer: MiniLLMTokenizer, device: torch.device,
                 name: str, checkpoint_bytes: int, step: int = 0):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.name = name
        self.checkpoint_bytes = checkpoint_bytes
        self.step = step
        self.history: list[dict] = []
        self.lock = threading.Lock()

    def info(self) -> dict:
        cfg = self.model.cfg
        return {
            "name": self.name,
            "params": self.model.num_parameters(),
            "max_context": cfg.max_position_embeddings,
            "vocab_size": cfg.vocab_size,
            "layers": cfg.num_hidden_layers,
            "hidden_size": cfg.hidden_size,
            "device": describe_device(self.device),
            "checkpoint_bytes": self.checkpoint_bytes,
            "checkpoint_human": human_bytes(self.checkpoint_bytes),
            "quantized": getattr(self, "quantized_bits", None),
            "step": self.step,
            "trained": self.step > 0,
            "default_system": DEFAULT_SYSTEM,
        }

    def reset(self) -> None:
        with self.lock:
            self.history.clear()

    def _prompt_ids(self, message: str, system: str, context_limit: int) -> tuple[list[int], int]:
        """Chat ids for the current history, dropping old turns to fit the limit."""
        turns = [*self.history, {"role": "user", "content": message}]
        dropped = 0
        while True:
            ids = self.tokenizer.apply_chat_template(
                turns, system=system, add_generation_prompt=True
            )
            if len(ids) <= context_limit or len(turns) <= 1:
                if len(ids) > context_limit:
                    # A single turn can still be longer than the window: keep the
                    # tail, which contains the generation prompt and the newest text.
                    ids = ids[-context_limit:]
                return ids, dropped
            # drop the oldest user/assistant pair, never the system prompt
            turns = turns[2:]
            dropped += 1

    @torch.no_grad()
    def chat(self, message: str, *, system: str = DEFAULT_SYSTEM, temperature: float = 0.7,
             top_k: int = 40, max_new_tokens: int = 256, context_limit: int = 2048) -> dict:
        message = (message or "").strip()
        if not message:
            raise ValueError("empty message")
        with self.lock:
            system = (system or "").strip() or DEFAULT_SYSTEM
            context_limit = max(64, min(int(context_limit), self.model.cfg.max_position_embeddings))
            max_new_tokens = max(1, min(int(max_new_tokens),
                                       self.model.cfg.max_position_embeddings - 8))
            ids, dropped = self._prompt_ids(message, system, context_limit - max_new_tokens)
            input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

            started = time.perf_counter()
            out = self.model(input_ids, use_cache=True)
            cache = out.past_key_values
            logits = out.logits[:, -1]
            prompt_len = input_ids.shape[1]
            ttft = time.perf_counter() - started

            # A chat turn ends with EOS or with the SFT closing marker; both must
            # stop decoding, and neither is part of the visible answer or history.
            stop_ids = {int(self.model.cfg.eos_token_id), END_OF_TURN_ID}
            generated: list[int] = []
            decode_started = time.perf_counter()
            for _ in range(max_new_tokens):
                token = self.model._sample(logits, temperature, top_k, None)
                if int(token.item()) in stop_ids:
                    break
                generated.append(int(token.item()))
                if len(generated) >= max_new_tokens:
                    break
                out = self.model(token, past_key_values=cache, use_cache=True,
                                 position_offset=prompt_len + len(generated) - 1)
                cache = out.past_key_values
                logits = out.logits[:, -1]
            decode_seconds = time.perf_counter() - decode_started
            total = time.perf_counter() - started
            reply = self.tokenizer.decode(generated, skip_special_tokens=True)
            self.history.extend([
                {"role": "user", "content": message},
                {"role": "assistant", "content": reply},
            ])
            return {
                "reply": reply,
                "tokens": len(generated),
                "prompt_tokens": prompt_len,
                "ttft_ms": ttft * 1000,
                "prefill_tok_s": prompt_len / max(ttft, 1e-9),
                "decode_ms": decode_seconds * 1000,
                "tok_s": len(generated) / max(decode_seconds, 1e-9) if generated else 0.0,
                "total_ms": total * 1000,
                "dropped_turns": dropped,
                "history_turns": len(self.history) // 2,
                "context_limit": context_limit,
            }


def make_handler(engine: ChatEngine):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quieter logs
            print(f"[web] {self.address_string()} {fmt % args}")

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_asset(self, name: str) -> None:
            path = (ASSETS / name).resolve()
            if not path.is_file() or ASSETS.resolve() not in path.parents:
                self.send_error(404)
                return
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPES.get(path.suffix, "text/plain"))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - http.server API
            route = self.path.split("?", 1)[0]
            if route in ("/", "/index.html"):
                self._send_asset("index.html")
            elif route == "/api/info":
                self._send_json(engine.info())
            elif route.startswith("/static/"):
                self._send_asset(route[len("/static/"):])
            elif route == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            else:
                self.send_error(404)

        def do_POST(self):  # noqa: N802 - http.server API
            route = self.path.split("?", 1)[0]
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send_json({"error": "invalid JSON"}, 400)
                return
            if route == "/api/reset":
                engine.reset()
                self._send_json({"ok": True})
                return
            if route != "/api/chat":
                self.send_error(404)
                return
            try:
                result = engine.chat(
                    body.get("message", ""),
                    system=body.get("system", DEFAULT_SYSTEM),
                    temperature=float(body.get("temperature", 0.7)),
                    top_k=int(body.get("top_k", 40)),
                    max_new_tokens=int(body.get("max_new_tokens", 256)),
                    context_limit=int(body.get("context_limit", 2048)),
                )
            except ValueError as exc:
                self._send_json({"error": str(exc)}, 400)
                return
            except Exception as exc:  # keep the server alive, surface the error
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)
                return
            self._send_json(result)

    return Handler


def serve(ckpt: str, tokenizer: str, host: str = "127.0.0.1", port: int = 8765,
          device: str | None = None, name: str | None = None) -> ThreadingHTTPServer:
    torch_device = get_device(device)
    model, payload = load_checkpoint(ckpt, torch_device)
    tok = MiniLLMTokenizer.load(tokenizer)
    tok.text_policy = checkpoint_text_policy(payload)
    tok.assert_vocab_size(model.cfg.vocab_size)
    if payload.get("tokenizer_fingerprint") not in (None, tok.fingerprint):
        raise SystemExit("checkpoint and tokenizer fingerprints do not match")
    engine = ChatEngine(
        model, tok, torch_device,
        name=name or Path(ckpt).stem,
        checkpoint_bytes=Path(ckpt).stat().st_size,
        step=int(payload.get("step", 0) or 0),
    )
    engine.quantized_bits = payload.get("quantized_bits")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"[warn] binding {host}: this server has no authentication; prefer 127.0.0.1")
    server = ThreadingHTTPServer((host, port), make_handler(engine))
    print(f"[web] {engine.name}: {engine.model.num_parameters():,} params on "
          f"{describe_device(torch_device)}, {engine.info()['checkpoint_human']}")
    print(f"[web] serving http://{host}:{server.server_address[1]}")
    return server


def main() -> None:
    p = argparse.ArgumentParser(description="Local MiniLLM chat UI")
    p.add_argument("--ckpt", required=True, help="fp32 or INT4 MiniLLM checkpoint")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--device", default=None)
    p.add_argument("--name", default=None, help="display name for the model")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    set_seed(args.seed)
    server = serve(args.ckpt, args.tokenizer, args.host, args.port, args.device, args.name)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
