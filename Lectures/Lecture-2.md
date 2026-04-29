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


| Single HA | Multi HA ||
|---|---|---|
| $X(4,8)=X(s,d)$ | $X(4,8)=X(s,d)$ ||
|---|---|---|
| $W_Q (8,4)=W_Q (d_{in},d_{out})$ | $W_{Q_1} (8,2)$ <br> $W_{Q_1} (8,2)$ | $$d_{head} = \frac{d_{out}}{n_{head}} = \frac{4}{2} = 2$$|
| $W_K (8,4)$ | $W_{K_1} (8,2)$ <br> $W_{K_2} (8,2)$ ||
| $W_V (8,4)$ | $W_{V_1} (8,2)$ <br> $W_{V_1} (8,2)$ ||
|---|---|---|
| $Q(4,4)=X \times W_Q$ | $Q_1(4,2)=X \times W_{Q_1}$ <br> $Q_2(4,2)=X \times W_{Q_2}$||
| $K(4,4)=X \times W_K$ | $K_1(4,2)=X \times W_{K_1}$ <br> $K_2(4,2)=X \times W_{K_2}$||
| $V(4,4)=X \times W_V$ | $V_1(4,2)=X \times W_{V_1}$ <br> $V_2(4,2)=X \times W_{V_2}$||
|---|---|---|
| $A(4,4)=Q \times K^T$ | $A_1(4,4)=Q_1 \times K_1^T$ <br> $A_2(4,4)=Q_2 \times K_2^T$ | - Attention score matrix <br> - Size of the attention socre matrices remains exactly the same. <br> - Number of attention socre matrices is same as number of attention heads.|
| $B(4,4)=\text{softmax}\bigg(\text{casual}\bigg(\frac{Q \times K^T}{\sqrt{d_{keys}}}\bigg)\bigg)$ | $B_1(4,4)$ <br> $B_2(4,4)$ | Attention weight matrix |
| $Z(4,4)=\text{softmax}\bigg(\text{casual}\bigg(\frac{Q \times K^T}{\sqrt{d_{keys}}}\bigg)\bigg) \times V$ | $Z_1(4,2)=\text{Perspective-1}$ <br> $Z_2(4,2)=\text{Perspective-2}$ | Context matrix |
|| $Z(4,4) = [Z_1(4,2) \parallel Z_2(4,2)]$ |Concatinate Context matrices|

***

#### Perspective

```
The artist painted the portrait of a woman with a brush.
```

1. painting a woman with a "brush"
2. painting of a "woman with a brush"

*** 

* 55:00

#### The problem of redundant computation
* Are we calculating the same thing over and over?

***

* 1:10:00

* 








