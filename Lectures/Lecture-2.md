## Lecture-2: KV-Cache

* What - Need
* Good: Benefits
* Evil: Dark side

***

* [TurboQuant: Redefining AI efficiency with extreme compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
* [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)

*** 

* 5:00:00

#### GPU Chip Level 
* [Cerebras Inference](https://chat.cerebras.ai/)
* [Taalas - Burn the LLM on Chip](https://taalas.com/)

#### Y Combinator
* [Y Combinator - Requests for Startups](https://www.ycombinator.com/rfs)

***

* 30:00


|||
|---|---|
| $X(4,8)$ | $X(4,8)$ |
|---|---|
| $W_Q (8,4)$ | $W_Q_1 (8,4)$ |
| $W_K (8,4)$ | $W_K (8,4)$ |
| $W_V (8,4)$ | $W_V (8,4)$ |
|---|---|
| $Q(4,4)=X \times W_Q$ | $Q(4,4)=X \times W_Q$ |
| $K(4,4)=X \times W_K$ | $K(4,4)=X \times W_K$ |
| $V(4,4)=X \times W_V$ | $V(4,4)=X \times W_V$ |
|---|---|
| $\text{Attention Score}=A(4,4)=Q \times K^T$ ||
| $\text{Attention Weight}=B(4,4)=\text{softmax}\bigg(\text{casual}\bigg(\frac{Q \times K^T}{\sqrt{d_{keys}}}\bigg)\bigg)$ ||
| $\text{Context matrix}=Z(4,4)=\text{softmax}\bigg(\text{casual}\bigg(\frac{Q \times K^T}{\sqrt{d_{keys}}}\bigg)\bigg) \times V$ ||


#### MHA
* Size of the attention-socre-matrices remains exactly the same.
* Number of attention socre matrices is same as number of attention heads.

* 
