#!/usr/bin/env bash
set -euo pipefail

root=${1:-$PWD/swift-qwen38-runtime}
mkdir -p "$root"

runtime_name=llama-build-cuda13.0-sm120a-dflash-kvpatch-03c968f0.tar.zst
runtime_url=https://github.com/andrewting19/abliteration-station/releases/download/bootstrap-artifacts-v1/$runtime_name
runtime_sha=ed237650a0c6fd27457fd5ec15d3e2f7fd7ce5822db1a47b655231f2f2316357

target_name=Swift-Qwen3.8-27B-Abliterated-UD-Q3_K_XL.gguf
target_url=https://huggingface.co/andrewting/Swift-Qwen3.8-27B-Abliterated/resolve/main/$target_name?download=true
target_sha=19363c7b7d1e8337a20a7c13eef49fd6f614841f6d36a894ef058f79394eceb4

draft_name=Qwen3.8-27B-DFlash2-Q4_K_M.gguf
draft_url=https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF/resolve/main/$draft_name?download=true
draft_sha=1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd

fetch() {
  local url=$1 output=$2 sha=$3
  if [[ -s "$output" ]] && echo "$sha  $output" | sha256sum -c - >/dev/null 2>&1; then
    return
  fi
  curl -fL --retry 5 --retry-delay 2 --continue-at - --output "$output.part" "$url"
  echo "$sha  $output.part" | sha256sum -c -
  mv "$output.part" "$output"
}

fetch "$runtime_url" "$root/$runtime_name" "$runtime_sha"
fetch "$target_url" "$root/$target_name" "$target_sha"
fetch "$draft_url" "$root/$draft_name" "$draft_sha"

mkdir -p "$root/runtime"
tar --zstd -xf "$root/$runtime_name" -C "$root/runtime"

echo "Runtime prepared at $root"
