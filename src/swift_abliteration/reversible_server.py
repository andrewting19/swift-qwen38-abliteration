from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=8192)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int = 3819
    stop: str | list[str] | None = None
    stream: bool = False


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


class RequestMetricWriter:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        if self.path is None:
            return
        line = json.dumps(record, sort_keys=True) + "\n"
        with self.lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()


def apply_text_stops(text: str, stop: str | list[str] | None) -> str:
    if stop is None:
        return text
    values = [stop] if isinstance(stop, str) else stop
    positions = [text.find(value) for value in values if value and value in text]
    return text[: min(positions)] if positions else text


def make_app(
    engine: Any,
    served_model_name: str,
    metric_path: str | Path | None = None,
) -> FastAPI:
    app = FastAPI(title="Swift reversible evaluation server")
    metrics = RequestMetricWriter(metric_path)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": served_model_name}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": served_model_name,
                    "object": "model",
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat(request: ChatRequest) -> dict[str, Any]:
        if request.stream:
            raise HTTPException(status_code=400, detail="Streaming is not supported.")
        if request.model != served_model_name:
            raise HTTPException(status_code=404, detail="Unknown model name.")
        started = time.perf_counter()
        try:
            result: GenerationResult = await asyncio.to_thread(
                engine.generate,
                [message.model_dump() for message in request.messages],
                request.max_tokens,
                request.temperature,
                request.top_p,
                request.seed,
            )
        except Exception as exc:
            metrics.write(
                {
                    "request_id": uuid.uuid4().hex,
                    "status": "error",
                    "elapsed_seconds": time.perf_counter() - started,
                    "error_type": type(exc).__name__,
                }
            )
            raise HTTPException(status_code=500, detail=type(exc).__name__) from exc
        text = apply_text_stops(result.text, request.stop)
        elapsed = time.perf_counter() - started
        request_id = "chatcmpl-" + uuid.uuid4().hex
        metrics.write(
            {
                "request_id": request_id,
                "status": "ok",
                "elapsed_seconds": elapsed,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
            }
        )
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
        }

    return app


class TransformersGenerationEngine:
    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.lock = threading.Lock()
        self.device = model.get_input_embeddings().weight.device

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> GenerationResult:
        import torch

        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        batch = self.tokenizer([rendered], return_tensors="pt")
        batch = {name: value.to(self.device) for name, value in batch.items()}
        do_sample = temperature > 0
        generator = torch.Generator(device=self.device).manual_seed(seed)
        settings: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "generator": generator,
        }
        if do_sample:
            settings.update({"temperature": temperature, "top_p": top_p})
        with self.lock, torch.inference_mode():
            generated = self.model.generate(**batch, **settings)
        prompt_tokens = int(batch["input_ids"].shape[1])
        new_tokens = generated[0, prompt_tokens:]
        return GenerationResult(
            text=self.tokenizer.decode(new_tokens, skip_special_tokens=True),
            prompt_tokens=prompt_tokens,
            completion_tokens=int(new_tokens.shape[0]),
        )
