#!/usr/bin/env python3
"""Compare selected tensors in two remote Safetensors files.

The script uses HTTP range requests. It does not download either complete
checkpoint shard. Scan mode reads small windows. Rank-one mode downloads only
the selected tensors and estimates the dominant singular update.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import requests
except ImportError as exc:  # pragma: no cover - command-line dependency check
    raise SystemExit(
        "This script needs requests. Install the forensics extra with "
        "`python -m pip install -e '.[forensics]'`."
    ) from exc


DEFAULT_BASE_URL = (
    "https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/resolve/"
    "5711b268169967567844e1e560e8a3966da959b1/"
    "model_pp0_ep0_shard0.safetensors"
)
DEFAULT_EDITED_URL = (
    "https://huggingface.co/dealignai/MiMo-V2.6-Flash-RL-UNCENSORED/resolve/"
    "0bdb11d7f37d3f6939e53f822c2bff518e4852fa/"
    "model_pp0_ep0_shard0.safetensors"
)
DEFAULT_TENSOR_PATTERN = "model.layers.{layer}.self_attn.o_proj.weight"


@dataclass
class RemoteSafeTensor:
    url: str
    verify_tls: bool = True

    def __post_init__(self) -> None:
        self.session = requests.Session()
        first = self._request(self.url, 0, 7)
        if len(first.content) < 8:
            raise RuntimeError("Safetensors header-length read returned fewer than 8 bytes")
        self.url = first.url
        self.header_length = struct.unpack("<Q", first.content[:8])[0]
        header_response = self._request(self.url, 8, 7 + self.header_length)
        self.header: dict[str, Any] = json.loads(
            header_response.content[: self.header_length]
        )
        self.data_start = 8 + self.header_length

    def _request(self, url: str, start: int, end: int) -> requests.Response:
        response = self.session.get(
            url,
            headers={"Range": f"bytes={start}-{end}"},
            timeout=180,
            verify=self.verify_tls,
        )
        response.raise_for_status()
        return response

    def tensor_info(self, name: str) -> dict[str, Any]:
        info = self.header.get(name)
        if not isinstance(info, dict) or "data_offsets" not in info:
            raise KeyError(f"Tensor is not present: {name}")
        return info

    def tensor_size(self, name: str) -> int:
        start, end = self.tensor_info(name)["data_offsets"]
        return int(end - start)

    def read_tensor(self, name: str) -> bytes:
        info = self.tensor_info(name)
        start, end = info["data_offsets"]
        response = self._request(
            self.url,
            self.data_start + int(start),
            self.data_start + int(end) - 1,
        )
        expected = int(end - start)
        if len(response.content) != expected:
            raise RuntimeError(
                f"Range read for {name} returned {len(response.content)} bytes; "
                f"expected {expected}"
            )
        return response.content

    def read_probe(self, name: str, probe_bytes: int) -> bytes:
        info = self.tensor_info(name)
        start, end = (int(value) for value in info["data_offsets"])
        size = end - start
        length = min(probe_bytes, size)
        relative_start = max(0, (size - length) // 2)
        absolute_start = self.data_start + start + relative_start
        return self._request(
            self.url,
            absolute_start,
            absolute_start + length - 1,
        ).content


def parse_layers(value: str) -> list[int]:
    layers: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            first, last = (int(item) for item in part.split("-", 1))
            if last < first:
                raise argparse.ArgumentTypeError(f"Invalid layer range: {part}")
            layers.update(range(first, last + 1))
        else:
            layers.add(int(part))
    if not layers:
        raise argparse.ArgumentTypeError("At least one layer is required")
    return sorted(layers)


def bf16_bytes_to_float32(raw: bytes, shape: list[int]) -> np.ndarray:
    words = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    words <<= 16
    return words.view(np.float32).reshape(shape)


def absolute_cosine(left: np.ndarray, right: np.ndarray) -> float:
    numerator = abs(
        float(np.dot(left.astype(np.float64), right.astype(np.float64)))
    )
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return numerator / denominator if denominator else 0.0


def dominant_rank_one_metrics(
    base: np.ndarray,
    edited: np.ndarray,
    *,
    seed: int,
    iterations: int = 10,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    difference = base - edited
    difference_norm = float(np.linalg.norm(difference))
    base_norm = float(np.linalg.norm(base))

    generator = np.random.default_rng(seed)
    left = generator.standard_normal(difference.shape[0]).astype(np.float32)
    left /= np.linalg.norm(left)
    for _ in range(iterations):
        right = difference.T @ left
        right /= np.linalg.norm(right)
        left = difference @ right
        left /= np.linalg.norm(left)

    right_scaled = difference.T @ left
    singular_value = float(np.linalg.norm(right_scaled))
    right = right_scaled / singular_value

    metrics = {
        "relative_frobenius_change": difference_norm / base_norm,
        "top_rank1_energy_fraction": (singular_value / difference_norm) ** 2,
        "left_projection_alignment": absolute_cosine(right, base.T @ left),
        "right_projection_alignment": absolute_cosine(left, base @ right),
        "top_singular_value": singular_value,
    }
    return metrics, left, right


def validate_pair(
    base: RemoteSafeTensor, edited: RemoteSafeTensor, name: str
) -> dict[str, Any]:
    base_info = base.tensor_info(name)
    edited_info = edited.tensor_info(name)
    for field in ("dtype", "shape"):
        if base_info[field] != edited_info[field]:
            raise RuntimeError(
                f"{name} has different {field}: "
                f"{base_info[field]} vs {edited_info[field]}"
            )
    if base.tensor_size(name) != edited.tensor_size(name):
        raise RuntimeError(f"{name} has different byte sizes")
    return base_info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--edited-url", default=DEFAULT_EDITED_URL)
    parser.add_argument("--layers", type=parse_layers, default=parse_layers("0-47"))
    parser.add_argument("--tensor-pattern", default=DEFAULT_TENSOR_PATTERN)
    parser.add_argument("--mode", choices=("scan", "rank1"), default="scan")
    parser.add_argument("--probe-bytes", type=int, default=65_536)
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS verification. Use only when a local CA setup requires it.",
    )
    args = parser.parse_args()

    if args.insecure:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    base = RemoteSafeTensor(args.base_url, verify_tls=not args.insecure)
    edited = RemoteSafeTensor(args.edited_url, verify_tls=not args.insecure)
    result: dict[str, Any] = {
        "base_url": args.base_url,
        "edited_url": args.edited_url,
        "mode": args.mode,
        "tensor_pattern": args.tensor_pattern,
        "layers": [],
    }

    left_directions: dict[int, np.ndarray] = {}
    right_directions: dict[int, np.ndarray] = {}
    for layer in args.layers:
        name = args.tensor_pattern.format(layer=layer)
        info = validate_pair(base, edited, name)
        row: dict[str, Any] = {
            "layer": layer,
            "tensor": name,
            "dtype": info["dtype"],
            "shape": info["shape"],
        }
        if args.mode == "scan":
            base_probe = base.read_probe(name, args.probe_bytes)
            edited_probe = edited.read_probe(name, args.probe_bytes)
            row.update(
                {
                    "probe_bytes": len(base_probe),
                    "probe_changed": base_probe != edited_probe,
                    "base_probe_sha256": hashlib.sha256(base_probe).hexdigest(),
                    "edited_probe_sha256": hashlib.sha256(edited_probe).hexdigest(),
                }
            )
        else:
            if info["dtype"] != "BF16":
                raise RuntimeError(
                    f"Rank-one analysis currently supports BF16, not {info['dtype']}"
                )
            base_array = bf16_bytes_to_float32(base.read_tensor(name), info["shape"])
            edited_array = bf16_bytes_to_float32(
                edited.read_tensor(name), info["shape"]
            )
            metrics, left, right = dominant_rank_one_metrics(
                base_array,
                edited_array,
                seed=20260922 + layer,
            )
            row.update(metrics)
            left_directions[layer] = left
            right_directions[layer] = right
        result["layers"].append(row)

    if args.mode == "rank1":
        result["adjacent_direction_absolute_cosines"] = []
        for first, second in zip(args.layers, args.layers[1:], strict=False):
            result["adjacent_direction_absolute_cosines"].append(
                {
                    "first_layer": first,
                    "second_layer": second,
                    "left": absolute_cosine(
                        left_directions[first], left_directions[second]
                    ),
                    "right": absolute_cosine(
                        right_directions[first], right_directions[second]
                    ),
                }
            )

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
