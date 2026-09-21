# Reproduce the RTX 5090 Pi profile

This bundle reproduces the validated inference configuration for
`Swift-Qwen3.8-27B-Abliterated-UD-Q3_K_XL.gguf`.

## Requirements

- Linux x86-64
- NVIDIA RTX 5090
- NVIDIA driver and CUDA 13-compatible runtime
- `libnccl.so.2`, `curl`, `zstd`, and `sha256sum`
- At least 32 GB of system RAM and 20 GB of free disk
- Pi coding agent 0.84.4 or a compatible version

The prebuilt llama.cpp runtime uses CUDA architecture `120a`. Its SHA-256 is
checked before extraction. It contains no model weights or credentials.

## Run

```bash
chmod +x fetch.sh serve.sh
./fetch.sh /workspace/swift-qwen38-runtime
./serve.sh /workspace/swift-qwen38-runtime
```

Copy `models.json` to the Pi configuration directory, or merge its provider
entry into your existing file. Then select provider `local-swift` and model
`swift-abliterated-q3`.

## Pinned inputs

- Target SHA-256:
  `19363c7b7d1e8337a20a7c13eef49fd6f614841f6d36a894ef058f79394eceb4`
- DFlash2 Q4_K_M SHA-256:
  `1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd`
- Runtime SHA-256:
  `ed237650a0c6fd27457fd5ec15d3e2f7fd7ce5822db1a47b655231f2f2316357`
- llama.cpp source commit:
  `7339054744f109c4cd89b75689dbb8a2c154d60e`
- Patched source tree:
  `03c968f0fc5470485859ef71d03ba4e665033afa`

The source patch and build script are public in
`andrewting19/abliteration-station`. The exact binary is in its
`bootstrap-artifacts-v1` GitHub release.

## Expected performance

The validated host used an RTX 5090 with a 570 W power limit and an AMD EPYC
7542. A live Pi session ended at 104,686 total tokens and generated 4,863
tokens at 83.44 token-weighted TPS. It made 12 shell-tool calls with no tool
errors. A fixed 120,844-token Pi replay measured 81.04 TPS.

Host CPU, PCIe bandwidth, GPU power limits, driver versions, prompt content,
and draft acceptance can change the result. Treat 80 TPS as the measured gate,
not as a guarantee for every RTX 5090 host.
