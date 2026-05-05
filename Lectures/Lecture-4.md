## Lecture-4

#### Four Paradigmsn for Compressing Attention Across Tokens
* Full Attention
* Sliding Attention
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

#### Sliding Window Attentin
* N: Sequence length
* W: Sliding window length
* Reduction = $\frac{N}{W}$

#### Trade-offs
* Long range dependency is completely lost.
* No index, which is bad! (We don't know which tokens are important.)
* Lazy compared to DSA.
* Attend to unimportant tokens. Miss out on important tokens.

* How researchers mitigated drawback of SWA?

***

#### Gemma
* $TE \to PE \to T_1(Sliding) \to T_2(Sliding) \to T_3(Sliding) \to T_4(Sliding) \to T_5(Full) \to T_6(Sliding) \to T_7(Sliding) \to T_8(Sliding) \to T_9(Sliding) \to \mbox{T_{10}(Full)} \to T_{11} \to T_{12} \ldots T_{96} \to Logits \to NextToken$
