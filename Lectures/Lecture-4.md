## Lecture-4

#### Four Paradigmsn for Compressing Attention Across Tokens
* Full Attention
* [Sliding Window Attention (SWA)](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/06_swa/README.md)
* Linear Attention 
* State Space Models (Mamba)
* Mamba Architecture

***

* 25:00

$$l \times b \times \boxed{s} \times \underbrace{n_{heads} \times h}_{\text{embedding dim}} \times 2 \times bytes$$

* Our foucs is now on sequence length (s).
* We have already seen one approach __DeepSeek Sparse Attention (DSA)__ to tackle this.

***

* 30:00

#### [Sliding Window Attention](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/06_swa/README.md)
* N: Sequence length
* W: Sliding window length
* We reduce KV-cache size by a factor of $\frac{W}{N}$

#### Trade-offs
* Long range dependency is completely lost.
* No index, which is bad! (We don't know which tokens are important.)
* Lazy compared to DSA.
* Attend to unimportant tokens. Miss out on important tokens.

* How researchers mitigated drawback of SWA?

***

#### Gemma
* $\to TE \to PE \to T_1(Sliding) \to T_2(Sliding) \to T_3(Sliding) \to T_4(Sliding) \to T_5(Full) \to T_6(Sliding) \to T_7(Sliding) \to T_8(Sliding) \to T_9(Sliding) \to T_{10}(Full) \to T_{11} \to T_{12} \ldots T_{96} \to Logits \to NextToken$

***

* 40:00

#### Effective Receptive Field
* L : num Layers
* W : Window Size
* $W \times L$

* 50:00

***

#### Active Research Area
* Receptive Field of Attention Mechanaism (Promising area of research)
* Dynamic sliding window sizes ($$W=f(L)$$)

***

* 1:20:00

#### FLOPs during Prefill
* Queries: $N$ Queries of dimension $d$
* Keys: $N$ Keys of dimension $d$

| Reduction |   | Prefill (Compute Bound Regime) | Decode (Memory Bound Regime) | 
|---|---|---|---|
|               | MHA | $2 \times N^2 \times d$        | $2 \times N \times d$ |
| $\frac{W}{N}$ | SWA | $2 \times N \times W \times d$ | $2 \times W \times d$ |

***

#### Convolution
* FFT: $O(N^2) \to O(N \log N)$

***

* [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
* [Demo Video "CNN Explainer: Learning Convolutional Neural Networks with Interactive Visualization"
](https://www.youtube.com/watch?v=HnWIHWFbuUQ)

***

* 2:20:00

#### State Space Model (SSM)

***

* 2:30:00

* $Q(N,d)$
* $K(N,d)$
* $V(N,d)$

$$\text{softmax}\bigg(\frac{Q(N,d)\times K^T(d,N)}{\sqrt{d_{Keys}}}\bigg) \times V(N,d)$$

* Cache Size = $2 \times N \times d$

* What if $\text{softmax}$ is removed and we only have $(Q \times K^T) \times V$.
* We can first multiply $A(d,d) = K^T(d,N)\times V(N,d)$
* New token: $B(d,d) = k_t^T(d,1) \times v_t(1,d)$
* Update cache: $A(d,d)+B(d,d)$
* __Observation__:
* Whenever a new token comes, the influence of past is compressed into a small $d \times d$ matrix.
* The whole context is compressed into a small $d \times d$ matrix.
* Context bottlenexk problem.
* Size of KV-Cache would reduce by a huge amount and stay fixed, but contxt is compressed into $d \times d$ values and as $N$ increaes, it puts a lot of pressure on this small context to all past tokens.

* 
***
