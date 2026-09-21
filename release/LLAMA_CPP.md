# llama.cpp RTX 5090 profile

The complete download, server, and Pi configuration bundle is in the
[`replication/`](./replication) directory.

Use the target GGUF in this repository with the public DFlash2 draft:

- `incoai/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf`
- DFlash draft length: 4
- Context: 262,144
- Target and draft KV cache: Q4_0
- Batch: 8,192
- Micro-batch: 2,048
- Full GPU offload and flash attention
- `GGML_CUDA_GRAPH_OPT=1`

The llama.cpp build must include Qwen3.8 DFlash2 support. The validated build
reports source commit `7339054744f109c4cd89b75689dbb8a2c154d60e`.

Example:

```bash
export GGML_CUDA_GRAPH_OPT=1
export LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

/path/to/llama.cpp/build/bin/llama-server \
  --model Swift-Qwen3.8-27B-Abliterated-UD-Q3_K_XL.gguf \
  --spec-draft-model Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --spec-type draft-dflash \
  --spec-draft-ngl all \
  --spec-draft-type-k q4_0 \
  --spec-draft-type-v q4_0 \
  --spec-draft-n-max 4 \
  --gpu-layers all \
  --ctx-size 262144 \
  --parallel 1 \
  --batch-size 8192 \
  --ubatch-size 2048 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --flash-attn on \
  --fit off \
  --threads 16 \
  --threads-batch 16 \
  --poll 50 \
  --poll-draft 1 \
  --ctx-checkpoints 4 \
  --cache-prompt \
  --cache-reuse 0 \
  --no-context-shift \
  --jinja \
  --reasoning-format deepseek \
  --reasoning on \
  --reasoning-effort medium \
  --backend-sampling \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --repeat-penalty 1.0
```

On one RTX 5090, the selected profile measured 83.44 token-weighted decode
tokens per second in a live Pi session that ended at 104,686 total tokens. The
session used 12 shell-tool calls, generated 4,863 tokens, and had no tool
errors. A fixed 120,844-token Pi replay measured 81.04 tokens per second.

The target SHA-256 is
`19363c7b7d1e8337a20a7c13eef49fd6f614841f6d36a894ef058f79394eceb4`.
