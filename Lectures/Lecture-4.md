## Lecture-4

#### Four Paradigmsn for Compressing Attention Across Tokens
* Full Attention
* Sliding Window Attention
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
* $W \times L$

* 50:00

***

#### Active Research Area
* Receptive Field of Attention Mechanaism (Promising area of research)
* Dynamic sliding window sizes ($$W=f(L)$$)

***

* 1:20:00

#### FLOPs during Pre-Fill
* $N_{Queries}$ of dimension $d$
* $N_{Keys}$ of dimension $d$
* For 1-query, number of FLOPs= $2 \times N \times d$
* For N-queries, number of FLOPs= $2 \times N^2 \times d$ (Pre-fill), which is $2 \times N \times d$ during decode, as I look at only one query.


* For SWA, number of FLOPs = $2 \times N \times W \times d$ (Pre-fill)


|   | Pre-Fill | Decode | 
|---|---|---|
| MHA   | $2 \times N^2 \times d$        | $2 \times N \times d$|
| SWA   | $2 \times N \times W \times d$ | $2 \times W \times d$|

***

