#### Lecture-1: Introduction to Inference Engineering

***

#### What an Inference Engineer Actually Does?
1. Download an open-source model
2. Measure the product's Service Level Objective (SLO) - TTFT, ITL, throughput, cost per token
3. Choose runTime optimizations - attention varient, quantization, KV-Cache tricks
4. Choose Infrastructure layout - parallelism, replication, disaggregation
5. Deploy via a serving engine - vLLM, SGLang, TensorRT-LLM, Ray Serve
6. Benchmark, iterate, re-tune - a new frontier model drops every 6 weeks

***

#### Pre-training versus Inference
* Pre-training: Memory bound
* Pre-training: Compute bound

***

#### Inference Engineer
1. Application
2. Pre-Trating
3. __Inference Engg__: RunTime, Infrastructure, Tooling
4. __GPU Engg__: Kernel coding
5. Build Innovative GPU Chips ([Wafer: AI that makes AI fast](https://www.ycombinator.com/companies/wafer))


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
| Gemma 4 | Google |
| Phi 4| Microsoft |
| Minstral Large | France|

***

* 50:00

* [GPU-Powered AI Inference | Scale AI Inference | Try Modal Free](https://modal.com/?utm_source=google&utm_medium=search&utm_campaign=ai_models_nonbrand&utm_term=ai%20inference%20infra&gad_source=1&gad_campaignid=22941570665&gbraid=0AAAAA9n1HKiQdbvwZC-U1jox7gml1d4Td&gclid=Cj0KCQjwkrzPBhCqARIsAJN460mX1mS9Y8rkKlcgu5z53Qc58HThSvVSvpywxJ3AejTyTkEJUSRuUjYaAgniEALw_wcB)

* [Jetson Orin Nano Super Developer Kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/)

***

* 1:00:00

* [GPT-5.5 API Pricing. Twice as expensive as GPT-5.4](https://www.reddit.com/r/OpenAI/comments/1sts36l/gpt55_api_pricing_twice_as_expensive_as_gpt54/)

*** 

* [Introducing Claude Opus 4.7](https://www.google.com/search?q=opus+4.7&rlz=1C5GCCM_en&oq=opus+4.7&gs_lcrp=EgZjaHJvbWUyDAgAEEUYORixAxiABDIGCAEQABgDMhAIAhAAGIMBGLEDGIAEGIoFMg0IAxAAGIMBGLEDGIAEMhAIBBAAGIMBGLEDGIAEGIoFMgYIBRAAGAMyBwgGEAAYgAQyBggHEEUYQNIBCDI2OTRqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8)

***

* 1:05:00

#### Edge Inference
* Raspberry Pi LLM Inference
* Mobile Phone LLM Inference
* [SmolChat - On-Device Inference of SLMs in Android](https://github.com/shubham0204/SmolChat-Android)
* Gemma 4 model Laptop Inference

***

* [Meet Instant AI - cerebras.ai](https://chat.cerebras.ai/)

* [Taalas | The model is The Computer](https://taalas.com/)
  * 15,000 tps
  * Model burned into silocon

***

* Ablation study
* 500 different experiments
* [karpathy autoresearch](https://github.com/karpathy/autoresearch)
* Heatmap of configurations (choose best configuration)

*** 

* 1:45:00

#### GPU RoofLine
1. Bandwidth
2. FLOPS

***

* 1:55:00

#### Arithmethic Intensity

* $$Arithmethic Intensity = \frac{num ~FLOPS}{Bytes ~transferred}$$

$$
\begin{align}
   output(N,d) &= x(N,d) \times W_q(d,d) \\
\end{align}
$$

Multiplying one-row of dimention ($d$) with one column ofdimention ($d$) has $d$ multi-plication operations. We have to do $d-1$ additions to get one output value. Total (addition+multiplication) operations is then $2d-1$. For output matrix with $Nd$ entries, we thus have a total of $2Nd^2$ opeartions. 
