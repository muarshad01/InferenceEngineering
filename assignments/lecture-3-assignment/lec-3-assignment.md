Lesson Description
Paper link: https://arxiv.org/pdf/2506.09342

Instructions:

Remove the RoPE part and construct a boilerplate code file (.ipynb) which runs ablations studies on the TinyStories Dataset for the following configurations:

- MQA
- GQA (vary group size as a hyperparameter)
- MLA (vary latent dim as hyperparameter)
- MHA

You should benchmark following:

- Compute metrics: KV Cache size, inter token latency, P99, P50
- Most importantly, add benchmarks on performance quality: grammar fluency of generated outputs
- Also benchmark perplexity and validation loss
