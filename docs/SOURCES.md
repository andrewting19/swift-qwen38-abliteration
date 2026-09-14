# Sources and Frozen Inputs

The code uses the source commits below. The commit IDs protect the experiment from later upstream changes.

- Base model: [ukisai/Swift-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-Qwen3.8-27b), revision `1b30aaaf753fe5c1cb51ada2ea0367a53445359c`
- Harmful direction source: [walledai/AdvBench](https://huggingface.co/datasets/walledai/AdvBench), revision `9d4730540082fa4017450b65ca1c0e1d8d30446e`
- Harmless direction source: [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca), revision `dce01c9b08f87459cf36a430d809084718273017`
- AdvBench source file: [llm-attacks/llm-attacks](https://github.com/llm-attacks/llm-attacks), revision `098262edf85f807224e70ecd87b9d83716bf6b73`
- Alpaca source file: [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca), revision `761dc5bfbdeeffa89b8bff5d038781a4055f796a`
- Full-tensor edit description: [OrcaRouter, How Abliteration Works](https://www.orcarouter.ai/blog/how-abliteration-works)
- Layer-band implementation linked by Huihui: [Sumandora/remove-refusals-with-transformers](https://github.com/Sumandora/remove-refusals-with-transformers)
- Broader method notes: [local method survey](../outputs/abliteration-model-method-survey.md)

The Orca article states that it uses massive-activation masking. It does not give a complete masking algorithm or threshold. This repository does not label its plain difference-of-means direction as an exact reproduction of that private detail.
