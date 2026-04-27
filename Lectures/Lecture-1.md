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

#### Attention Varients
* MHA
* MQA
* GQA
* MLA
* Sliding Window
* Linear State Space
* Mamba

***

#### Memory-traffic Rearrangemtn
* FlashAttention
* PagedAttention
* Prefix Caching
* Chuned Refill

***

#### Quantization
* FP16
* FP8
* INT8
* INT4
* GPTQ
* GGUF
* QAT
* BitNet

***

#### Scheduling Tricks
* Continuous batching
* Speculative decoding (N-gram, EAGLE, Medusa): Multiple tokens are generated at the same time together, so that, Inference becomes faster.

***

#### KV-cache Compression
* MLA absorption
* sliding window cache
* SSM state

***

* Disaggregated Serving: Decouple Prefill(P) / Decode(D)

***

#### Tooling
* __vLLM__: High-througput serving engine, PagedAttention, Continuous batching, Prefix caching.
* __SGLang__: Structured generation, RadixAttention, KV-cache sharing.
* __TensorRT-LLM__: NVIDIA's compiled kernels, FP8, multi-GPU parallelism.
* __Ray Serve__: Distributed orchestration, autoscaling, multi-model routing.

*** 

* 25:00

#### Parameter-efficient fine-tuning (PEFT) techniques 
* LoRA (Low-Rank Adaptation)
* QLoRA (Quantized LoRA)

***

* [LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)
* Gemma 3 (27B) - 5:1 (local:global) sliding window and GQA

*** 

* 35:00

* Semantic Layer to route the queries

* [Grok 4](https://x.ai/news/grok-4)

***

* 40:00

| Model ||
|---|--|
| Qwen3 | Alibaba China|
| Kimi |  China|
| DeepSeek-V3 | DeepSeek China| 
| GLM-5| China|
|---|---|
| Llama 4 | Meta|
| Gemma 3 | Google |
| Phi 4| Microsoft |
| Minstral Large | France|

***

* 50:00

* [GPU-Powered AI Inference | Scale AI Inference | Try Modal Free](https://modal.com/?utm_source=google&utm_medium=search&utm_campaign=ai_models_nonbrand&utm_term=ai%20inference%20infra&gad_source=1&gad_campaignid=22941570665&gbraid=0AAAAA9n1HKiQdbvwZC-U1jox7gml1d4Td&gclid=Cj0KCQjwkrzPBhCqARIsAJN460mX1mS9Y8rkKlcgu5z53Qc58HThSvVSvpywxJ3AejTyTkEJUSRuUjYaAgniEALw_wcB)

* [Jetson Orin Nano Super Developer Kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/)

***

* 1:00:00
