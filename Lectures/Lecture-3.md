## Lecture-3
* MHA
* MQA
* GQA
* MLA
* DeepSeek V3.2 Sparse Attention (DSA) 

* [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models (Dec 2025)](https://arxiv.org/abs/2512.02556)

***

#### MHA
* [Lecture09 - MHA](https://github.com/muarshad01/DeepSeek/blob/main/Notes/lecture09_notes.md)

* 30:00

#### MQA
* [Lecture10 - MQA](https://github.com/muarshad01/DeepSeek/blob/main/Notes/lecture10_notes.md)

***

* 50:00

#### GQA
* [Lecture11 - GQA](https://github.com/muarshad01/DeepSeek/blob/main/Notes/lecture11_notes.md)

***

#### MLA
* Shift focus from reducing the number of heads to compressing the informtion within these heads.
* What if we don't have to cache K & V seperately.
* What if, we could first project our input into a single, combined, much smaller matrix, a latent matrix and cache only that!
* This is the central idea of MLA:
* Instead of caching two large matrices, K & V, we only cache one smaller, lower dimensional matrix $C_{KV}$.
* This single matrix becomes our highly efficient cache.
* When we need the full Keys and Values, we can resonstruct them on the fly from the compressed latent representation.
***
