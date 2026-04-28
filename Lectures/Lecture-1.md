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

* $$\text{Arithmethic ~Intensity} = \frac{\text{num ~FLOPS}}{\text{Bytes ~transferred}}$$

$$
\begin{align}
   output(N,d) &= x(N,d) \times W_q(d,d) \\
\end{align}
$$

```
FLOPS = 2 x N x d x d = 2Nd^2 (two FLOPs per multiply-add)
Total Bytes = 2Nd + 2d^2 + 2Nd = 4Nd + 2d^2
```

$$
\begin{align}
\text{Arithmethic ~Intensity} &= \frac{2Nd^2}{4Nd+2d^2} \\
                              &= \frac{Nd}{2N+d} \\
\end{align}
$$

***

#### GPU Roofline Plot

* 2:10:00

***

#### Inference
* Prefill: Comptue bound
* Decode: Memory bound

*** 

* 2:20:00

#### Journey of Prompty through Inference Pipeline
* Prefill / Decode from Scratch

* [A Bird’s-Eye View of LLM Architecture](https://vizuara.substack.com/p/a-birds-eye-view-of-llm-architecture)

***

* 2:30:00

* Prompt: Inference journey is "a"

1. Tokenization
2. Embeddings:
* Token embedding matrix (vocab_size, dim_of_model (8))
3. Input Matrix
  ```
      Inference  | . . . . . . . . |
 X =  Journey    | . . . . . . . . |
      is         | . . . . . . . . |
      a          | . . . . . . . . |
  ```

X is input embedding matrix of dimention is (4,8).

4. Positional embedding matrix:
```                     
1                                   |
2                                   |            
.                                   |
.                                   |
.                                   |
.                                   |
max-seq-length (context length)     |
```

***

* 3:00:00

| Memory Type | Needed/Not Needed|
|---|---|
| Parameters   | Needed     |
| Gradients    | Not Needed |
| Optimzations | Not Needed |
| Activations  | Needed     |
| KV-cache     | Needed     |

***

* 3:05:00

* [TurboQuant: Redefining AI efficiency with extreme compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

***

#### Speculative Decoding
* LLM
* SLM - produces more tokens, which are verified by LLM


* 3:20:00


***


## 3. LLM Architecture from Scratch
* Vizz
  * Forward Pass (We store Parameters and Activations (Input to Layers))
  * Backward Pass (We store the Gradients; Optimizer States (Momentum ($$\hat{m}$$), Variance($$\hat{v}$$)))
* We'll discuss __Parameters, Gradients, Activation, Optimizer States__

<p align="center">
  <img src="https://github.com/muarshad01/NVIDIA-GPU/blob/main/images/day-1/LLM-architecture.png" width="200" height="400"/>
</p>

***

* 1:20:00

* IE ($$x_0$$) = Token Embedding (TE) + Positional Embedding (PE)
* IE ($$x_0$$) -> __Transformer Block__

***

#### [A Bird’s-Eye View of LLM Architecture](https://vizuara.substack.com/p/a-birds-eye-view-of-llm-architecture)
* __Note__: Within Transfomer block, the first component is __LN1)__.

***

* Input to __LN1__ is $$x_0$$ Matrix (i.e., IE vectors):

$$
x_0=
\begin{bmatrix}
  1 & 0.5 & 1 & 0\\
  2 & 0.1 & 1 & 1\\
  1 & 0 & 0 & 2
\end{bmatrix}
$$

* The dimension $$x_0$$ matrix is $$(s \times h)=(3 \times 4)$$.

* __Input__: $$(b, s, h) = (1, 3, 4)$$
  * Batch size = b = 1
  * Number of token = s = 3
  * Embedding dimension = d = h = 4

* __LN1__: For each IE vector in $$x_0$$, we perform the following operation, i.e., subtract the mean($$\mu$$) and divide by sqrt-of-variance $$(\sqrt{\sigma})$$

$$\hat{x} = \bigg(\frac{x-\mu}{\sqrt{\sigma}}\bigg)\times\alpha + \beta$$

* The output $$x_1$$ is a matrix of dimention $$(s \times h)=(3 \times 4)$$.
* __Note__: $$\alpha$$ (Scale) and $$\beta$$ (Shift) are trainable parameters.

* __Question__: Why we need LN?
  * LN solves the __problem of internal co-variate shift.__
  * Everytime a new input sample comes into a NN, if the distribution of those input samples is different then the training becomes very hard.
  * We want the mean ($$\mu=0$$) and standard-deviation ($$\sigma=1$$) to be same.

***

#### $$(W_q, W_k, W_v)$$ 
* $$(W_q, W_k, W_v)$$ are trainable matrices of dimension $$(h \times h)=(4 \times 4)$$.
* $x_1$ matrix is multiplied with each of $$(W_q, W_k, W_v)$$ matrices to generate $$(Q, K, V)$$ (i.e., set of vectors) of dimension $$(s \times h)=(3 \times 4)$$.
  * $$Q = x_1 \times W_q$$
  * $$K = x_1 \times W_k$$
  * $$V = x_1 \times W_v$$

***

#### Attention Score
* $$Q \times K^{T} = (s \times s)$$ matrix

* __Question__: Why do we divide by $$\sqrt{d_{keys}}$$?
  * When we multiply Queries (Q) with Keys (K) transpose, i.e. $$Q \times K^{T}$$, the variance of that matrix multiplication can grow a lot. This is not good when we take gradients ($$\frac{\partial L}{\partial W}$$).
  * Generally, when we take gradients,  during the back-propagation, we want the variance of $$Q \times K^{T}$$ to be consistent.
  * Therefore, we divie by $$\sqrt{d_{keys}}$$, so that variance $$Q \times K^{T}$$ stays as much close to 1 as possible, so that, the back-propagation will happen in a seamless manner.

***

#### Attention Weights

$$\text{softmax}\bigg(\frac{Q \times K^{T}}{\sqrt{d_{keys}}}\bigg)$$
* which is $$(s \times s)$$ matrix.

***

#### Softmax
* __Question__: Why we need softmax?
* Softmax ensures two things:
1. All values $x_i$ are between $$0 \rightarrow 1$$.
 2. All values sum up to 1, i.e., $$\text{Sum} = \sum \hat{X} \rightarrow 1$$

$$
\begin{align}
   X          &= \\{x_1, x_2, x_3, x_4, x_5, x_6\\}\\
   \text{Sum} &= \sum_{i=1}^{n=6}e^{x_i}\\
   \hat{X}    &=\bigg\\{\frac{e^{x_1}}{\text{Sum}},\frac{e^{x_2}}{\text{Sum}},\frac{e^{x_3}}{\text{Sum}},\frac{e^{x_4}}{\text{Sum}},\frac{e^{x_5}}{\text{Sum}},\frac{e^{x_6}}{\text{Sum}}\bigg\\}
\end{align}
$$

***

#### Context Matrix

$$Z = \text{softmax}\bigg(\frac{Q \times K^{T}}{\sqrt{d_{keys}}}\bigg)\times V$$

* $$Z$$ is a $$(s \times h)$$ dimension matrix.

$$x_{atten} (3 \times 4) = Z \times W_0 = (s \times h) \times (h \times h)= (3 \times 4) \times (4 \times 4)$$

* $$x_{atten}$$ is the output of Transformer block and has dimension $$(s \times h)=(3 \times 4)$$.
* __Note__: $$W_0$$ is the output projection matrix. It brings us back to our orginal dimension and has dimension $$(h \times h)=(4 \times 4)$$

***

#### Masking
* In in auto-regressive task, which is next-token prediction task, we can't peek into the future. We can only look into the past. Therefore, we apply the __causal mask__.

#### Dropout

#### Shortcupt Connection
* $$x_{res1} = x_{attn} + x_0$$.
  * It gives an alternative-path for the gradients to flow.
  * It prevents the __Vanishing Gradient Problem.__

*** 



