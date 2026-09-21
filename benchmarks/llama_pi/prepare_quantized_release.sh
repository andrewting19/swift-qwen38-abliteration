#!/usr/bin/env bash
set -euo pipefail

root=${1:-/workspace/swift-abliterated-q3}
model_repo=${MODEL_REPO:-andrewting/Swift-Qwen3.8-27B-Abliterated}
model_revision=${MODEL_REVISION:-583d5e0442f640fc7c27911e84c0e72cc12029a9}
llama_revision=${LLAMA_REVISION:-5ecbe1ac17ec0484c5b44af0bd580cdc9c428ed4}
type_map=${TYPE_MAP:-$root/ud-q3-k-xl-types.txt}
build_tar=${BUILD_TAR:-$root/llama-build.tar.zst}

hf_dir=$root/hf
llama_dir=$root/llama.cpp
venv=$root/venv
bf16=$root/Swift-Qwen3.8-27B-Abliterated-BF16.gguf
quant=$root/Swift-Qwen3.8-27B-Abliterated-UD-Q3_K_XL.gguf
imatrix=$root/imatrix_unsloth.gguf
draft_q4=$root/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
draft_q4_sha=1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd

for required in "$type_map" "$build_tar"; do
  [[ -s "$required" ]] || { echo "missing required file: $required" >&2; exit 1; }
done

mkdir -p "$root"
df -h "$root"
free -h

if [[ ! -d "$llama_dir/.git" ]]; then
  git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git "$llama_dir"
fi
git -C "$llama_dir" fetch --depth 1 origin "$llama_revision"
git -C "$llama_dir" checkout --detach "$llama_revision"

if [[ ! -x "$venv/bin/python" ]]; then
  python3 -m venv "$venv"
fi
"$venv/bin/pip" install --upgrade pip
"$venv/bin/pip" install -r "$llama_dir/requirements/requirements-convert_hf_to_gguf.txt" huggingface_hub

"$venv/bin/hf" download "$model_repo" \
  --revision "$model_revision" \
  --local-dir "$hf_dir" \
  --exclude '*.mp4' '*.png' '*.md' 'evaluation_results.json' 'checkpoint-*.json' \
  'abliteration_*' 'LICENSE*' 'NOTICE'

if [[ ! -s "$bf16" ]]; then
  "$venv/bin/python" "$llama_dir/convert_hf_to_gguf.py" "$hf_dir" \
    --outfile "$bf16" \
    --outtype bf16
fi

if [[ ! -s "$imatrix" ]]; then
  "$venv/bin/hf" download unsloth/Qwen3.8-27B-GGUF imatrix_unsloth.gguf \
    --local-dir "$root"
fi

runtime_dir=$root/runtime
mkdir -p "$runtime_dir"
tar --zstd -xf "$build_tar" -C "$runtime_dir"
quantize=$runtime_dir/build/bin/llama-quantize
[[ -x "$quantize" ]] || { echo "missing quantizer: $quantize" >&2; exit 1; }
export LD_LIBRARY_PATH="$runtime_dir/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ ! -s "$quant" ]]; then
  "$quantize" \
    --imatrix "$imatrix" \
    --tensor-type-file "$type_map" \
    "$bf16" "$quant" Q3_K_M "$(nproc)"
fi

if ! echo "$draft_q4_sha  $draft_q4" | sha256sum -c - >/dev/null 2>&1; then
  "$venv/bin/hf" download incoai/Qwen3.8-27B-DFlash2-GGUF \
    Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
    --local-dir "$root"
  echo "$draft_q4_sha  $draft_q4" | sha256sum -c -
fi

sha256sum "$quant" > "$quant.sha256"
stat --printf='quant_bytes=%s\n' "$quant"
cat "$quant.sha256"
