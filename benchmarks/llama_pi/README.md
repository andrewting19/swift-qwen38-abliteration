# RTX 5090 Pi performance target

The target is 80 to 100 decode tokens per second on one RTX 5090 in a long,
real Pi coding-agent context.

The experiment uses the published BF16 checkpoint, the tensor-type layout from
the proven Unleashed UD-Q3_K_XL package, and the public Unsloth Qwen3.8
importance matrix. It does not copy Unleashed model weights. The external
DFlash2 draft is unchanged.

Test order:

1. Convert the published checkpoint to GGUF and quantize it.
2. Verify the quantized tensor map and run a deterministic smoke test.
3. Replay one frozen Pi context with normal autoregressive decoding.
4. Replay the same context with the checkpoint MTP head.
5. Replay the same context with the Q4_K_M DFlash2 draft at `n_max=4`.
   This was the best validated setting for the abliterated target.
6. Run one complete Pi coding task with the fastest valid mode.

The speed gate is at least 80 sustained decode tokens per second. The quality
gate requires valid tool calls, task completion, focused tests, and no severe
repetition or malformed output.
