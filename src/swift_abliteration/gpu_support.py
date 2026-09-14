from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

ACKNOWLEDGEMENT = "I_UNDERSTAND_THIS_LOADS_AND_EDITS_A_55GB_MODEL"


def require_acknowledgement(value: str) -> None:
    if value != ACKNOWLEDGEMENT:
        raise RuntimeError(f"Pass --acknowledge {ACKNOWLEDGEMENT}")


def read_prompt_jsonl(path: str | Path) -> tuple[list[str], str]:
    content = Path(path).read_bytes()
    prompts = []
    for line in content.splitlines():
        if line.strip():
            prompts.append(json.loads(line)["text"])
    return prompts, hashlib.sha256(content).hexdigest()


def system_record() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _command(["git", "rev-parse", "HEAD"]),
    }
    try:
        import torch

        result.update(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count(),
                "gpus": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
            }
        )
    except ImportError:
        result["torch"] = None
    return result


def require_large_gpu(minimum_bytes: int = 75 * 1024**3) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")
    largest = max(
        torch.cuda.get_device_properties(i).total_memory
        for i in range(torch.cuda.device_count())
    )
    if largest < minimum_bytes:
        raise RuntimeError(
            f"At least {minimum_bytes / 1024**3:.0f} GiB on one GPU is required."
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not report BF16 support.")


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _command(arguments: list[str]) -> str | None:
    try:
        return subprocess.run(
            arguments, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
