# Sources and Frozen Inputs

The code uses the source commits below. The commit IDs protect the experiment from later upstream changes.

- Base model: [ukisai/Swift-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-Qwen3.8-27b), revision `1b30aaaf753fe5c1cb51ada2ea0367a53445359c`
- Harmful direction source: [walledai/AdvBench](https://huggingface.co/datasets/walledai/AdvBench), revision `9d4730540082fa4017450b65ca1c0e1d8d30446e`
- Harmless direction source: [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca), revision `dce01c9b08f87459cf36a430d809084718273017`
- Semantic-matched harmful source: [heretic-org/Semantic-Harmful](https://huggingface.co/datasets/heretic-org/Semantic-Harmful), revision `001ca2ceaef94a748235e0ba1366aee48436e286`
- Semantic-matched harmless source: [heretic-org/Semantic-Harmless](https://huggingface.co/datasets/heretic-org/Semantic-Harmless), revision `7e9f2b01272da85f2be7a3437f31ac46698e8735`
- AdvBench source file: [llm-attacks/llm-attacks](https://github.com/llm-attacks/llm-attacks), revision `098262edf85f807224e70ecd87b9d83716bf6b73`
- Alpaca source file: [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca), revision `761dc5bfbdeeffa89b8bff5d038781a4055f796a`
- Full-tensor edit description: [OrcaRouter, How Abliteration Works](https://www.orcarouter.ai/blog/how-abliteration-works)
- Orca model card: [Qwen3.8-27B-Uncensored-FP8](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8)
- Original research and dataset splits: [andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction)
- FailSpy dataset and split implementation: [FailSpy/abliterator](https://github.com/FailSpy/abliterator/blob/main/abliterator.py)
- Heretic defaults: [p-e-w/heretic](https://github.com/p-e-w/heretic/blob/master/src/heretic/config.py)
- Layer-band implementation linked by Huihui: [Sumandora/remove-refusals-with-transformers](https://github.com/Sumandora/remove-refusals-with-transformers)
- Broader method notes: [local method survey](../outputs/abliteration-model-method-survey.md)
- Final external judge: [OpenAI GPT-5.6 Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
- Initial diagnostic judge: [OpenAI GPT-5 nano](https://developers.openai.com/api/docs/models/gpt-5-nano)
- Judge pricing reference: [OpenAI API pricing](https://developers.openai.com/api/docs/pricing)

The Orca article states that it uses massive-activation masking. It does not give a complete masking algorithm or threshold. This repository does not label its plain difference-of-means direction as an exact reproduction of that private detail.
