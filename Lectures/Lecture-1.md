#### Lecture-1: Introduction to Inference Engineering


***

* 5:00

<img src="https://github.com/muarshad01/Inference-Engg/blob/main/images/roadmap.png" width="600" height="300" />

***

* 15:00

* [Qwen - Huggging Face](https://huggingface.co/Qwen)

#### Inference
* Time to First Token (TTFT)
* Inter-token latency (ITL)
* Tokens per second (TPS or t/s)

***

* 20:00

#### Qwen Optimizations
* Attention (Flash Attention)
  * MHA
  * MQA
  * MLA
* State Space Models (SSA)
  * Mixture of SSM like Mamba architecture with sliding-window
* Quantization
* Speculative Decoding (Multiple tokens are generated at the same time together, so that, Inference becomes faster.)
* Parallelism (Amount of memory on GPU takes reduces, so that, Inference becomes faster.)
* Disaggregated Serving (Decouple prefill and decode)

***

#### Tooling
* vLLM
* SGLang

*** 

* 25:00

#### Parameter-efficient fine-tuning (PEFT) techniques 
* LoRA (Low-Rank Adaptation)
* QLoRA (Quantized LoRA)

***

* [LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)
* Gemma 3 (27B) - 5:1 (local:global) sliding window and GQA

*** 

* 30:00

* Semantic Layer to route the queries
