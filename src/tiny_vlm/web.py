"""Serve the local model playground: python -m tiny_vlm.web."""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import math
import mimetypes
import threading
import time
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from .data import build_prompt, build_transform
from .generate import load_model, validate_tokenizer
from .model import count_parameters
from .tokenizer import EOS_ID, TinyTokenizer
from .utils import get_device

ASSETS = Path(__file__).with_name("web_assets")
MAX_IMAGE_BYTES = 8 * 1024**2
MAX_REQUEST_BYTES = 12 * 1024**2
MAX_IMAGE_PIXELS = 16_000_000
DEFAULT_CHECKPOINT = Path("runs/multimodal-tiny-stage2/model-int8.pt")


class InputError(ValueError):
    pass


class ModelBusy(RuntimeError):
    pass


def decode_image(data: str) -> Image.Image:
    if not isinstance(data, str) or not data.startswith("data:image/") or ";base64," not in data:
        raise InputError("请上传 PNG、JPEG、WebP 或 GIF 图片。")
    encoded = data.split(";base64,", 1)[1]
    if len(encoded) > (MAX_IMAGE_BYTES + 2) // 3 * 4:
        raise InputError("图片不能超过 8 MB。")
    try:
        raw = base64.b64decode(encoded, validate=True)
        if not raw or len(raw) > MAX_IMAGE_BYTES:
            raise InputError("图片为空或超过 8 MB。")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format not in {"PNG", "JPEG", "WEBP", "GIF"}:
                    raise InputError("请使用 PNG、JPEG、WebP 或 GIF 图片。")
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise InputError("图片不能超过 1600 万像素，请缩小后重试。")
                return ImageOps.exif_transpose(image).convert("RGB")
    except InputError:
        raise
    except (ValueError, OSError, binascii.Error, UnidentifiedImageError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise InputError("无法读取这张图片，请重新选择有效的图片文件。") from exc


def generation_parameters(payload: dict) -> dict:
    params = payload.get("parameters", {})
    if not isinstance(params, dict):
        raise InputError("生成参数格式不正确。")
    result = {}
    for name, default, low, high in (("max_new_tokens", 32, 1, 256), ("top_k", 50, 1, 200)):
        value = params.get(name, default)
        if type(value) is not int or not low <= value <= high:
            raise InputError(f"{name} 必须是 {low} 到 {high} 之间的整数。")
        result[name] = value
    temperature = params.get("temperature", 0.0)
    if type(temperature) not in (float, int) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise InputError("随机程度必须在 0 到 2 之间。")
    result["temperature"] = float(temperature)
    return result


class ModelRuntime:
    def __init__(self, model, tokenizer, checkpoint: Path, device, examples_dir: Path | None = None):
        validate_tokenizer(model, tokenizer)
        self.model = model
        self.tokenizer = tokenizer
        self.checkpoint = checkpoint
        self.device = device
        self.lock = threading.Lock()
        self.transform = build_transform(model.cfg.vision.img_size)
        self.example_image = None
        self.examples = [{"id": "text", "mode": "text", "title": "试一道文字题",
                          "description": "只输入文字，看看模型怎么回答。",
                          "prompt": "计算 2 + 3，只回答结果。", "reference": "5"}]
        if examples_dir is not None:
            self._load_examples(examples_dir)

    def _load_examples(self, folder):
        path = folder / "test.jsonl"
        if not path.is_file():
            return
        try:
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            caption = next((r for r in records if r.get("image") and not r.get("prompt")
                            and r.get("image", "").endswith("000000.png")), None)
            if caption is None:
                caption = next((r for r in records if r.get("image") and not r.get("prompt")), None)
            if caption is None:
                return
            image_path = (folder / caption["image"]).resolve()
            if not image_path.is_relative_to(folder.resolve()):
                return
            with Image.open(image_path) as image:
                buffer = io.BytesIO()
                image.convert("RGB").save(buffer, format="PNG")
                self.example_image = buffer.getvalue()
            self.examples.append({"id": "image", "mode": "image", "title": "让模型看张图",
                                  "description": "只提供图片，生成一句描述。", "prompt": "",
                                  "image_url": "/api/example-image", "reference": caption.get("answer", caption.get("caption", ""))})
            for key, question, title in (
                ("mixed", "图中的图形是什么颜色？", "图片搭配一个问题"),
                ("count", "图中有几个图形？只回答数字。", "换个问题再试试"),
            ):
                target = next((r for r in records if r.get("image") == caption["image"]
                               and r.get("prompt") == question), None)
                if target:
                    self.examples.append({"id": key, "mode": "mixed", "title": title,
                                          "description": question, "prompt": question,
                                          "image_url": "/api/example-image", "reference": target["answer"]})
        except (OSError, ValueError, KeyError, StopIteration):
            self.example_image = None

    def status(self):
        cfg = self.model.cfg
        return {"ready": True, "busy": self.lock.locked(), "name": "TinyVLM",
                "checkpoint": self.checkpoint.name, "device": str(self.device),
                "parameters_m": round(count_parameters(self.model).total / 1e6, 2),
                "file_mb": round(self.checkpoint.stat().st_size / 1024**2, 2),
                "image_size": cfg.vision.img_size, "max_context": cfg.text.max_seq_len,
                "modes": ["text", "image", "mixed"], "max_image_mb": 8}

    def generate(self, payload):
        if not isinstance(payload, dict):
            raise InputError("请求必须是 JSON 对象。")
        prompt = payload.get("prompt", "")
        if not isinstance(prompt, str) or len(prompt) > 10000:
            raise InputError("请输入文字，最多 10000 个字符。")
        prompt = prompt if prompt.strip() else ""
        image_data = payload.get("image")
        mode = payload.get("mode", "mixed" if image_data and prompt else ("image" if image_data else "text"))
        if mode not in ("text", "image", "mixed"):
            raise InputError("请选择文字、图片或图文模式。")
        if mode == "text" and (not prompt or image_data):
            raise InputError("文字模式需要文字输入；图片请在图片或图文模式上传。")
        if mode == "image" and (not image_data or prompt):
            raise InputError("图片模式只需要图片；提问请切换到图文模式。")
        if mode == "mixed" and (not image_data or not prompt):
            raise InputError("图文模式需要同时上传图片并输入问题。")
        params = generation_parameters(payload)
        ids = build_prompt(self.tokenizer, prompt)
        remaining = self.model.cfg.text.max_seq_len - len(ids)
        if remaining <= 0:
            raise InputError(f"输入超出模型的 {self.model.cfg.text.max_seq_len} token 上限，请缩短文字。")
        image = decode_image(image_data) if image_data is not None else None
        if not self.lock.acquire(blocking=False):
            raise ModelBusy("模型正在处理另一个请求，请稍后重试。")
        try:
            start = time.perf_counter()
            with torch.inference_mode():
                inputs = torch.tensor([ids], device=self.device)
                pixels = self.transform(image).unsqueeze(0).to(self.device) if image is not None else None
                params["max_new_tokens"] = min(params["max_new_tokens"], remaining)
                generated = self.model.generate(inputs, pixel_values=pixels, eos_token_id=EOS_ID, **params)
                new_ids = generated[0, len(ids):].tolist()
                answer = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            elapsed = time.perf_counter() - start
            return {"text": answer, "mode": mode, "input_tokens": len(ids),
                    "image_tokens": self.model.cfg.vision.num_patches if image is not None else 0,
                    "output_tokens": len(new_ids), "elapsed_ms": round(elapsed * 1000),
                    "tokens_per_second": round(len(new_ids) / max(elapsed, 1e-6), 2),
                    "limit_reached": bool(new_ids and new_ids[-1] != EOS_ID
                                          and len(new_ids) >= params["max_new_tokens"]),
                    "parameters": params}
        finally:
            self.lock.release()


def create_server(runtime, host="127.0.0.1", port=8765):
    class Handler(BaseHTTPRequestHandler):
        def send_body(self, body, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def send_json(self, data, status=200):
            self.send_body(json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

        def do_GET(self):
            path = urlsplit(self.path).path
            assets = {"/": "index.html", "/app.js": "app.js", "/styles.css": "styles.css"}
            if path in assets:
                asset = ASSETS / assets[path]
                mime = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
                self.send_body(asset.read_bytes(), mime + "; charset=utf-8")
            elif path == "/api/status":
                self.send_json(runtime.status())
            elif path == "/api/examples":
                self.send_json({"examples": runtime.examples})
            elif path == "/api/example-image" and runtime.example_image is not None:
                self.send_body(runtime.example_image, "image/png")
            elif path == "/favicon.ico":
                self.send_body(b"", "image/x-icon", 204)
            else:
                self.send_json({"error": "页面不存在。"}, 404)

        def do_POST(self):
            if urlsplit(self.path).path != "/api/generate":
                self.send_json({"error": "接口不存在。"}, 404)
                return
            if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
                self.send_json({"error": "请求需要使用 application/json。"}, 415)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    self.send_json({"error": "请求为空或超过 12 MB。"}, 413)
                    return
                payload = json.loads(self.rfile.read(length))
                self.send_json(runtime.generate(payload))
            except (InputError, ValueError, UnicodeError) as exc:
                self.send_json({"error": str(exc) if isinstance(exc, InputError) else "请求格式不正确，请检查输入。"}, 400)
            except ModelBusy as exc:
                self.send_json({"error": str(exc)}, 409)
            except Exception:
                import traceback
                traceback.print_exc()
                self.send_json({"error": "模型生成失败，请查看服务终端日志。"}, 500)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, help="default: tokenizer.json beside checkpoint")
    parser.add_argument("--examples-dir", type=Path, default=Path("data/multimodal"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.threads <= 0:
        parser.error("port must be 1–65535 and threads must be positive")
    tokenizer_path = args.tokenizer or args.ckpt.parent / "tokenizer.json"
    if not args.ckpt.is_file() or not tokenizer_path.is_file():
        parser.error("找不到权重或 tokenizer，请用 --ckpt 和 --tokenizer 指定现有文件。")
    torch.set_num_threads(args.threads)
    device = get_device(args.device)
    print(f"正在加载 {args.ckpt} …", flush=True)
    model, _ = load_model(args.ckpt, device)
    runtime = ModelRuntime(model, TinyTokenizer.load(tokenizer_path), args.ckpt, device, args.examples_dir)
    server = create_server(runtime, args.host, args.port)
    print(f"TinyVLM 模型实验室已启动：http://{args.host}:{server.server_port}", flush=True)
    print("按 Ctrl+C 停止服务。", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
