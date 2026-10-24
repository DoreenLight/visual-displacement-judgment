# Parallelizing Linear Transformers with the Delta Rule over Sequence Length

Songlin Yang<sup>⋄</sup> Bailin Wang<sup>⋄</sup> Yu Zhang<sup>†</sup> Yikang Shen<sup>‡</sup> Yoon Kim<sup>⋄</sup>

<sup>⋄</sup>Massachusetts Institute of Technology <sup>†</sup>Soochow University <sup>‡</sup>MIT-IBM Watson AI Lab

yangsl66@mit.edu

## Abstract

Transformers with linear attention (i.e., linear transformers) and state-space models have recently been suggested as a viable linear-time alternative to transformers with softmax attention. However, these models still underperform transformers especially on tasks that require in-context retrieval. While more expressive variants of linear transformers which replace the additive update in linear transformers with the delta rule [DeltaNet; 101] have been found to be more effective at associative recall, existing algorithms for training such models do not parallelize over sequence length and are thus inefficient to train on modern hardware. This work describes a hardware-efficient algorithm for training linear transformers with the delta rule, which exploits a memory-efficient representation for computing prod ucts of Householder matrices [11]. This algorithm allows us to scale up DeltaNet to standard language modeling settings. We train a 1.3B model for 100B tokens and find that it outperforms recent linear-time baselines such as Mamba [31] and GLA [124] in terms of perplexity and zero-shot performance on downstream tasks. We also experiment with two hybrid models which combine DeltaNet layers with (1) sliding-window attention layers every other layer or (2) two global attention layers, and find that these hybrids outperform strong transformer baselines.

## 1 Introduction

The attention mechanism [8, 116] has been shown to be an important primitive for accurate sequence modeling. Attention is moreover efficient during training as it is rich in matrix multiplications and can thus take advantage of highly parallel processing capabilities and specialized accelerators on modern GPUs. However, the complexity of attention is quadratic in sequence length, and hence it is a fundamentally expensive primitive. And while recent techniques have made it possible to scale attention to longer sequences through hardware-aware restructuring of the intermediate computations [20, 18, 59, 14], these methods still require storing the key/value vectors of previous elements, and this “KV cache” (whose size grows linearly) can be unwieldy to manage for long sequences.

Linear attention transformers [48] replace the exponential kernel in softmax attention with a dotproduct over (possibly transformed) key and query vectors. This makes it possible to formulate linear attention as a linear RNN with matrix-valued hidden states, thus obviating the need for a KV cache and enabling constant-memory inference. While initial variants of linear attention generally underperformed softmax attention on language modeling, gated variants of linear attention which incorporate a data-dependent gating factor have recently been shown to be competitive against strong transformer baselines [124, 92, 9, 79]. These gated linear transformers, along with time-varying state space models such as Mamba [31, 19] (which can be reparameterized as a gated linear transformer [124]), have been suggested as a potential alternative to ordinary transformers. However, despite the competitive language modeling performance, these models have been shown to underperform transformers on recall-intensive tasks [6, 7], which is important for many practical downstream tasks of interest (e.g., in retrieval-augmented generation [53]).

To enhance associative recall over long contexts, Schlag et al. [101] propose DeltaNet, a variant of a linear transformer which uses a delta rule-like update [121] to retrieve and update a value vector that is associated with the current key. DeltaNet was found to be effective on synthetic tasks and small scale language modeling/machine translation. However, the original work used a sequential algorithm that did not parallelize across sequence length, thus resulting in hardwareinefficient training, and it has not been clear how to scale DeltaNet to larger models and datasets.

This work describes a hardware-efficient training algorithm for DeltaNets which parallelizes the forward/backward passes across sequence length. We reparameterize the DeltaNet as a matrix-valued RNN whose recurrence is given by a generalized Householder transformation. This reparameterization enables the use of the compact WY representation [11] for products of Householder matrices, eliminating the need to materialize the hidden states of matrix size at each time step during paral lel training, which would otherwise result in high I/O costs. The memory-efficient representation makes it possible to straightforwardly extend the chunkwise parallel strategy for training linear attention models [34, 108, 124] to the DeltaNet case. We scale DeltaNets to moderate-scale language modeling benchmarks (1.3B models trained on 100B tokens), where DeltaNet is found to obtain better language modeling and zero-shot downstream task performance than strong linear recurrent models such as Mamba [31] and GLA [124]. For in-context retrieval and learning evaluation, we evaluate DeltaNet on synthetic and real benchmarks [4, 2, 84, 6], where it is again found to perform well against linear recurrent baselines. Finally, we experiment with a hybrid approach where we combine DeltaNet layers with sliding attention layers or global attention layers, and find that these hybrid models can improve upon ordinary transformers, as well as the pure DeltaNet transformer.

## 2 Background

## 2.1 Linear Transformer: Transformers with Linear Attention

Given a sequence of d-dimensional input vectors $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { L }$ , transformers use the softmax attention mechanism to attend over the entire past,

$$
\pmb {q} _ {t}, \pmb {k} _ {t}, \pmb {v} _ {t} = \pmb {W} _ {Q} \pmb {x} _ {t}, \pmb {W} _ {K} \pmb {x} _ {t}, \pmb {W} _ {V} \pmb {x} _ {t},
$$

$$
\pmb {o} _ {t} = \sum_ {i = 1} ^ {t} \frac {\exp (\pmb {k} _ {i} ^ {\top} \pmb {q} _ {t})}{\sum_ {j = 1} ^ {t} \exp (\pmb {k} _ {j} ^ {\top} \pmb {q} _ {t})} \pmb {v} _ {i},
$$

where $\mathbf { W } _ { Q } , \mathbf { W } _ { K } , \mathbf { W } _ { V } \in \mathbb { R } ^ { d \times d } , \mathbf { q } _ { t } , \mathbf { k } _ { t } , \mathbf { v } _ { t } , \mathbf { o } _ { t } \in \mathbb { R } ^ { d }$ . (Here we assume a single attention head for simplicity). Linear attention [48] replaces the exponential kernel $\exp ( { k _ { i } ^ { \intercal } { q _ { t } } } )$ with the dot-product $\phi { ( k _ { i } ) } ^ { \mathsf { T } } \phi ( \pmb q _ { t } )$ where $\phi : \mathbb { R } ^ { d }  \mathbb { R } ^ { n }$ is a feature map. This makes it possible to rearrange computations to represent linear attention as a linear RNN with matrix-valued hidden states,

$$
\boldsymbol {o} _ {t} = \sum_ {i = 1} ^ {t} \frac {\phi (\boldsymbol {k} _ {i}) ^ {\top} \phi (\boldsymbol {q} _ {t})}{\sum_ {j = 1} ^ {t} \phi (\boldsymbol {k} _ {j}) ^ {\top} \phi (\boldsymbol {q} _ {t})} \boldsymbol {v} _ {i} = \frac {\left(\sum_ {i = 1} ^ {t} \boldsymbol {v} _ {i} \phi (\boldsymbol {k} _ {i}) ^ {\top}\right) \phi (\boldsymbol {q} _ {t})}{\left(\sum_ {j = 1} ^ {t} \phi (\boldsymbol {k} _ {j}) ^ {\top}\right) \phi (\boldsymbol {q} _ {t})} = \frac {\mathbf {S} _ {t} \phi (\boldsymbol {q} _ {t})}{\mathbf {z} _ {t} ^ {\top} \phi (\boldsymbol {q} _ {t})},
$$

where $\begin{array} { r } { \mathbf { S } _ { t } = \sum _ { i = 1 } ^ { t } { \pmb { v } } _ { i } \phi ( { \pmb { k } } _ { i } ) ^ { \top } \in \mathbb { R } ^ { d \times n } } \end{array}$ and $\begin{array} { r } { \mathbf { z } _ { t } = \sum _ { i = 1 } ^ { t } \phi ( \pmb { k } _ { i } ) \in \mathbb { R } ^ { n } } \end{array}$ . If we allow n to go to infinity, P Plinear attention can use feature maps associated with polynomial kernels to compute a polynomial approximation to the exponential kernel as a dot product, and can thus approximate softmax attention arbitrarily well [6]. The denominator $\mathbf { z } _ { t } ^ { \top } \phi ( \pmb q _ { t } ) \in \mathbb { R }$ can result in numerical instabilities [86] and is removed in recent works [101, 63]. It is also common to use the identity mapping for φ [63, 108], which results in the following simplified linear transformer: $\mathbf { S } _ { t } = \mathbf { S } _ { t - 1 } + v _ { t } k _ { t } ^ { \top } , o _ { t } = \mathbf { S } _ { t } \mathbf { q } _ { t }$

Efficient training. Let $\mathbf { Q } , \mathbf { K } , \mathbf { V } \in \mathbb { R } ^ { L \times d }$ be the stacked query, key, value vectors, e.g., $\mathbf { Q } _ { i } =$ $\pmb q _ { i }$ . We can then compute the output $\mathbf { O } \in \mathbb { R } ^ { L \times d }$ in parallel via $\mathbf { O } = \left( \mathbf { Q } \mathbf { K } ^ { \mathsf { T } } \odot \mathbf { M } _ { L } \right) \mathbf { V } .$ , where $\mathbf { M } _ { L } \in \mathbb { R } ^ { L \times L }$ is the causal mask. This fully “parallel form” and the above “recurrent form” have different FLOPs and parallelization tradeoffs. The parallel form takes $O ( L ^ { 2 } d + L d ^ { 2 } )$ and thus requires more FLOPs than the recurrent form, which takes $O ( L d ^ { 2 } )$ . However, the parallel form is often much faster in practice for moderate-length sequences as it can be done in $O ( 1 )$ steps. This sequence-level parallellism also enables high GPU occupancy. The recurrent form requires fewer

FLOPs but cannot be parallelized across sequence $\mathrm { l e n g t h } ^ { \mathrm { 1 } }$ and the elementwise operations involved in recurrence moreover cannot make use of specialized matmul accelerators (e.g., tensor cores).

Chunkwise parallel form. The chunkwise parallel form [34, 108, 124] strikes a balance between the parallel and recurrent forms, allowing for fewer FLOPs than the parallel form and more sequence level parallelism than the recurrent form. Concretely, suppose the query/key/value vectors are split into $\begin{array} { c } { { \frac { \texttt { L } } { C } } } \end{array}$ chunks where each chunk is of length $C .$ Let $\dot { { \bf Q } } _ { [ t ] } \in \mathbb { R } ^ { \dot { C } \times d }$ be all the query vectors for chunk t, and let ${ \pmb q } _ { [ t ] } ^ { i } = { \pmb q } _ { t C + i }$ be the i-th query vector within the t’th chunk; the key/value chunks are defined similarly. Note that $t \in [ 0 , L / C ) , i \in [ 1 , C ]$ . The state matrices are also re-indexed such that ${ \bf S } _ { [ t ] } ^ { i } = { \bf S } _ { t C + i }$ , and we additionally define ${ \bf S } _ { [ t ] } ^ { 0 ^ { \scriptstyle * } } = { \bf \dot { S } } _ { [ t - 1 ] } ^ { C }$ , i.e., the initial state of a chunk is the last state of the previous chunk. We can then obtain the following identity for the hidden state and output vector for the r-th element within the t-th chunk,

$$
\mathbf {S} _ {[ t ]} ^ {r} = \mathbf {S} _ {[ t ]} ^ {0} + \sum_ {i = 1} ^ {r} \boldsymbol {v} _ {[ t ]} ^ {i} \boldsymbol {k} _ {[ t ]} ^ {i ^ {\top}}, \qquad \boldsymbol {o} _ {[ t ]} ^ {r} = \mathbf {S} _ {[ t ]} ^ {0} \boldsymbol {q} _ {[ t ]} ^ {r} + \sum_ {i = 1} ^ {r} \boldsymbol {v} _ {[ t ]} ^ {i} \left(\boldsymbol {k} _ {[ t ]} ^ {i ^ {\top}} \boldsymbol {q} _ {[ t ]} ^ {r}\right).
$$

By further rewriting the intra-chunk computation based on the parallel form, we obtain following,

$$
\mathbf {S} _ {[ t + 1 ]} = \mathbf {S} _ {[ t ]} + \mathbf {V} _ {[ t ]} ^ {\top} \mathbf {K} _ {[ t ]}
$$

$$
\in \mathbb {R} ^ {d \times d},\tag{1}
$$

$$
\mathbf {O} _ {[ t ]} = \mathbf {Q} _ {[ t ]} \mathbf {S} _ {[ t ]} ^ {\top} + \left(\mathbf {Q} _ {[ t ]} \mathbf {K} _ {[ t ]} ^ {\top} \odot \mathbf {M} _ {C}\right) \mathbf {V} _ {[ t ]}
$$

$$
\in \mathbb {R} ^ {C \times d}\tag{2}
$$

where we let ${ \mathbf S } _ { [ t ] } = { \mathbf S } _ { [ t ] } ^ { 0 }$ to reduce notational clutter. With this “chunkwise parallel form”, information is propagated chunk-to-chunk through ${ \bf S } _ { [ t ] }$ , and the intra-chunk states ${ \bf S } _ { [ t ] } ^ { i }$ for $i \in [ 1 , C ]$ need not be materialized, thus saving memory.

The complexity of the chunkwise parallel form is $O ( L C d + L d ^ { 2 } )$ , and the number of steps (without chunk-level parallel scan) is $O ( \textstyle { \frac { L } { T } } )$ . Hence, $C = L$ recovers the fully parallel form and $C = 1$ recovers the recurrent form. The chunkwise parallel form allows us to interpolate between the two forms, in essence trading off the number of sequential computations against sequence-level parallelism. In practice $C$ is set to a small constant (usually 64 or 128), allowing for subquadratic training. This chunkwise form enables practical speed-ups against parallel-form-only softmax attention even on moderate-length sequences, as demonstrated by FLASHLINEARATTENTION [124, 123]

## 2.2 DeltaNet: Linear Transformers with the Delta Update Rule

The above linear transformer employs a simple linear recurrence: $\mathbf { S } _ { t } = \mathbf { S } _ { t - 1 } + v _ { t } \mathbf { } \mathbf { } k _ { t } ^ { \mathsf { T } }$ . This can be seen as additively updating the memory $\mathbf { S } _ { t - 1 }$ with new key-value associations at each time step. However, a purely additive update rule makes it difficult to deallocate past key-value associations, eventually leading to key “collisions” when $L > d ,$ as pointed out by Schlag et al. [101]. A model should ideally learn to remove less important key-value associations to make room for new ones, and this removal should depend on the interaction between the new key and the memory content.

From a fast weight programming [102] perspective, the recurrent hidden state of linear attention is the fast weight mapping the input $\pmb q _ { t }$ to output $\mathbf { } _ { o _ { t } }$ with a Hessian-like update rule, which has limited memory capacity [68]. DeltaNet, on the other hand, uses the delta update rule or the Widrow-Hoff learning rule [121] for fast weight update: $\mathbf { S } _ { t } = \mathbf { S } _ { t - 1 } - \beta _ { t } \big ( \mathbf { S } _ { t - 1 } k _ { t } - \bar { v } _ { t } \big ) k _ { t } ^ { \top }$ , where $\beta _ { t }$ is the learning rate, $\mathbf { S } _ { t - 1 } \mathbf { \delta } _ { k _ { t } }$ represents the current prediction, ${ \mathbf { } } v _ { t }$ is the target value. The method derives its name from the core principle of updating weights based on the “delta” (difference) between the prediction $\mathbf { S } _ { t - 1 } \mathbf { \delta } _ { k _ { t } }$ and the target ${ \mathbf { } } v _ { t } .$ , which has been shown better memory capacity [28, 85, 56, 101]. This process can also be regarded as optimizing an online regression loss using a single step of SGD,<sup>2</sup>

$$
\mathcal {L} _ {t} (\mathbf {S}) = \frac {1}{2} \| \mathbf {S} \pmb {k} _ {t} - \pmb {v} _ {t} \| ^ {2}, \quad \mathbf {S} _ {t} = \mathbf {S} _ {t - 1} - \beta_ {t} \nabla_ {\mathbf {S} _ {t - 1}} \mathcal {L} _ {t} (\mathbf {S} _ {t - 1}) = \mathbf {S} _ {t - 1} - \beta_ {t} (\mathbf {S} _ {t - 1} \pmb {k} _ {t} - \pmb {v} _ {t}) \pmb {k} _ {t} ^ {\top},
$$

which has been discussed in several recent works [110, 58, 125, 10]. In contrast, linear transformers

can be regarded as optimizing an online linear (negative inner-product) loss $\mathcal { L } _ { t } = - \langle \mathbf { S } \pmb { k } _ { t } , \pmb { v } _ { t } \rangle . ^ { 3 }$

An alternative interpretation for DeltaNet is from the perspective of key-value retrieval. It first retrieves the old value using the current key, $v _ { t } ^ { \mathrm { o l d } } = \mathbf { S } _ { t - 1 } \bar { k _ { t } }$ . It then obtains a new value ${ \boldsymbol { v } } _ { t } ^ { \mathrm { n e w } }$ by interpolating between the old value and the current value ${ \mathbf { } } v _ { t } .$ , which replaces ${ \mathbf { } } { \mathbf { } } v _ { t } ^ { \mathrm { { o l d } } }$ in the memory:

$$
\boldsymbol {v} _ {t} ^ {\text { new }} = \beta_ {t} \boldsymbol {v} _ {t} + (1 - \beta_ {t})   \boldsymbol {v} _ {t} ^ {\text { old }}, \quad \mathbf {S} _ {t} = \mathbf {S} _ {t - 1} \underbrace {- \boldsymbol {v} _ {t} ^ {\text { old }} \boldsymbol {k} _ {t} ^ {\top}} _ {\text { remove }} \underbrace {+ \boldsymbol {v} _ {t} ^ {\text { new }} \boldsymbol {k} _ {t} ^ {\top}} _ {\text { write }}
$$

Here $\beta _ { t } = \sigma ( \mathbf { W } _ { \beta } \pmb { x } _ { t } ) \in ( 0 , 1 )$ is a soft “writing strength”: when $\beta _ { t } = 1$ , the old value is completely removed and $\begin{array} { r } { \pmb { v } _ { t } ^ { \mathrm { n e w } } = \pmb { v } _ { t } ; } \end{array}$ when $\beta _ { t } = 0$ , the memory remains unmodified and we have $\mathbf { S } _ { t } = \mathbf { S } _ { t - 1 }$ The output computation is the same as vanilla linear attention, i.e., $o _ { t } = \mathbf { S } _ { t } \pmb { q } _ { t }$ . The complexity of this recurrent form is the same as that of vanilla linear attention, $\mathrm { i } . \mathrm { e } . , { \mathcal { O } } ( L d ^ { 2 } )$

Schlag et al. [101] demonstrate that DeltaNet outperforms ordinary linear transformers on smallscale language modeling and synthetic in-context retrieval tasks. However, their training algorithm, an extension of the memory-efficient recurrent implementation for linear Transformers [48, §3.3.1], is strictly sequential and thus hardware inefficient, as discussed in [124, §3.2], motivating us to derive an equivalent chunkwise algorithm to train DeltaNet at scale, as introduced below.

## 3 Parallelizing DeltaNet Across the Sequence Dimension

## 3.1 A Memory-efficient Reparameterization

We first observe that $\mathbf { S } _ { t }$ admits a purely additive representation of the form $\begin{array} { r } { \mathbf { S } _ { t } = \sum _ { i = 1 } ^ { t } \mathbf { { u } } _ { i } \mathbf { { k } } _ { i } ^ { \top } } \end{array}$ for $u _ { i } , k _ { i } \in \mathbb { R } ^ { d }$ , since we can simply set $\pmb { u } _ { i } = \pmb { v } _ { i } ^ { \mathrm { n e w } } - \pmb { v } _ { i } ^ { \mathrm { o l d } } = \beta _ { i } ( \pmb { v } _ { i } - \pmb { v } _ { i } ^ { \mathrm { o l d } } )$ . Recall from $\ S 2 . 1$ that simple linear attention has the form $\begin{array} { r } { \mathbf { S } _ { t } = \sum _ { i = 1 } ^ { t } v _ { i } \pmb { k } _ { i } ^ { \top } } \end{array}$ . Thus, DeltaNet simply replaces the value vector ${ \mathbf { } } v _ { i }$ Pin linear attention with the “pseudo” value vector $\mathbf { \Delta } \mathbf { u } _ { i }$ . Once the $\mathbf { \chi } _ { \mathbf { \chi } _ { i } ^ { \prime } \mathbf { \tilde { s } } }$ have been constructed, the rest of computation can proceed as in ordinary linear attention, i.e., $\mathbf { O } = \left( \mathbf { Q } \mathbf { K } ^ { \mathsf { T } } \odot \mathbf { M } \right)$ U where $\mathbf { U } \in \mathbb { R } ^ { L \times d }$ is the row-wise concatenation of the $\mathbf { \Delta } \mathbf { u } _ { i }$ vectors.

However, computing $\mathbf { \Delta } \mathbf { u } _ { t }$ naïvely requires explicitly materializing $\mathbf { S } _ { t - 1 }$ to compute $\pmb { v } _ { t } ^ { \mathrm { o l d } }$ , which would require $\mathcal { O } ( d ^ { 2 } )$ ) memory. We now show that we can obtain the $\mathbf { \chi } _ { \mathbf { \chi } _ { t } } \mathbf { \chi } _ { \mathbf { \chi } }$ s without explicitly materializing $\mathbf { S } _ { t - 1 }$ in $\mathcal O ( d )$ memory. Our simple proof (by induction) relies on an application of the WY represen tation for products of Householder matrices [11]. The base case is clear since we have $\mathbf { S } _ { 1 } = \beta _ { 1 } \pmb { v } _ { 1 } \pmb { k } _ { 1 } ^ { \top }$ so $\pmb { u } _ { 1 } = \beta _ { 1 } \pmb { v } _ { 1 }$ . For the inductive step, we first observe that the DeltaNet update is given by,

$$
\mathbf {S} _ {t} = \mathbf {S} _ {t - 1} - \boldsymbol {v} _ {t} ^ {\text {old}} \boldsymbol {k} _ {t} ^ {\top} + \boldsymbol {v} _ {t} ^ {\text {new}} \boldsymbol {k} _ {t} ^ {\top} = \mathbf {S} _ {t - 1} - \beta_ {t} (\mathbf {S} _ {t - 1} \boldsymbol {k} _ {t}) \boldsymbol {k} _ {t} ^ {\top} + \beta_ {t} \boldsymbol {v} _ {t} \boldsymbol {k} _ {t} ^ {\top} = \mathbf {S} _ {t - 1} (\mathbf {I} - \beta_ {t} \boldsymbol {k} _ {t} \boldsymbol {k} _ {t} ^ {\top}) + \beta_ {t} \boldsymbol {v} _ {t} \boldsymbol {k} _ {t} ^ {\top},
$$

which can be seen as applying a generalized Householder transformation $( \mathrm { i . e . }$ ., matmul with an identity plus rank-one matrix) to the previous state. The inductive step is then given by,

$$
\mathbf {S} _ {t} = \mathbf {S} _ {t - 1} (\mathbf {I} - \beta_ {t} \boldsymbol {k} _ {t} \boldsymbol {k} _ {t} ^ {\top}) + \beta_ {t} \boldsymbol {v} _ {t} \boldsymbol {k} _ {t} ^ {\top} = \sum_ {i = 1} ^ {t - 1} \boldsymbol {u} _ {i} \boldsymbol {k} _ {i} ^ {\top} + \underbrace {\beta_ {t} \left(\boldsymbol {v} _ {t} - \sum_ {i = 1} ^ {t - 1} \boldsymbol {u} _ {i} \left(\boldsymbol {k} _ {i} ^ {\top} \boldsymbol {k} _ {t}\right)\right)} _ {\text { defined   as } \boldsymbol {u} _ {t}} \boldsymbol {k} _ {t} ^ {\top} = \sum_ {i = 1} ^ {t} \boldsymbol {u} _ {i} \boldsymbol {k} _ {i} ^ {\top}\tag{3}
$$

Note that $\mathbf { \Delta } \mathbf { u } _ { t }$ does not require materializing any of the hidden states and requires $\mathcal O ( d )$ memory to compute, thus completing the proof. While we have avoided materializing $\mathbf { S } _ { t } \mathbf { \bar { s } } .$ , computing $\mathbf { \Delta } \mathbf { \em u } _ { t } \mathrm { : \partial }$ ’s for all $L$ (that is, U) takes $\bar { \mathcal { O } ( L ^ { 2 } d ) }$ and moreover cannot be fully parallelized, unlike in linear attention where we can calculate all the value vectors V in parallel in $\mathcal { O } ( 1 )$ steps. We now show that the above trick still enables an efficient chunkwise parallel form for DeltaNet.

## 3.2 Chunkwise Parallel Form for DeltaNet

To derive the chunkwise parallel form, we first unroll the recurrence,

$$
\mathbf {S} _ {t} = \mathbf {S} _ {t - 1} (\mathbf {I} - \beta_ {t} \boldsymbol {k} _ {t} \boldsymbol {k} _ {t} ^ {\top}) + \beta_ {t} \boldsymbol {v} _ {t} \boldsymbol {k} _ {t} ^ {\top} = \sum_ {i = 1} ^ {t} \beta_ {i} (\boldsymbol {v} _ {i} \boldsymbol {k} _ {i} ^ {\top}) \left(\prod_ {j = i + 1} ^ {t} (\mathbf {I} - \beta_ {j} \boldsymbol {k} _ {j} \boldsymbol {k} _ {j} ^ {\top})\right).\tag{4}
$$

We then define the following variables: $\begin{array} { r l r } { \mathbf { P } _ { i } ^ { j } } & { = } & { \prod _ { t = i } ^ { j } ( \mathbf { I } \ - \ \beta _ { t } \pmb { k } _ { t } \pmb { k } _ { t } ^ { \top } ) \in \mathbb { R } ^ { d \times d } , \mathbf { H } _ { i } ^ { j } = } \end{array}$ $\begin{array} { r } { \sum _ { t = i } ^ { j } \beta _ { t } ( \pmb { v } _ { t } \pmb { k } _ { t } ^ { \top } ) \mathbf { P } _ { t + 1 } ^ { j } \in \mathbb { R } ^ { d \times d } } \end{array}$ , where we let $\mathbf { P } _ { i } ^ { j } = \mathbf { I }$ whenever $i > j$ . Intuitively, $\mathbf { P } _ { i } ^ { j }$ is the “decay factor” to be applied to $\mathbf { S } _ { i }$ for obtaining $\mathbf { S } _ { j }$ , and $\mathbf { H } _ { i } ^ { j }$ represents the contributions to $\mathbf { S } _ { j }$ starting from token i. (Hence $\mathbf { S } _ { t } = \mathbf { H } _ { 1 } ^ { t } ,$ . The chunkwise recurrence can then be written as,

$$
\mathbf {S} _ {[ t ]} ^ {r} = \mathbf {S} _ {[ t ]} ^ {0} \mathbf {P} _ {[ t ]} ^ {r} + \mathbf {H} _ {[ t ]} ^ {r}\tag{5}
$$

where we define the chunkwise variables $\mathbf { S } _ { [ t ] } ^ { i } = \mathbf { S } _ { t C + i } , \mathbf { P } _ { [ t ] } ^ { r } = \mathbf { P } _ { t C + 1 } ^ { t C + r } , \mathbf { H } _ { [ t ] } ^ { r } = \mathbf { H } _ { t C + 1 } ^ { t C + r }$ . Here we have $\textstyle { \frac { L } { C } }$ chunks of size $C .$ The trick is to now efficiently represent the $\mathbf { P } _ { [ t ] } ^ { r } , \mathbf { H } _ { [ t ] } ^ { r } \in \mathbb { R } ^ { d \times d }$ matrices using a similar approach described in §3.1, so that these matrices can be stored in $\mathcal O ( d )$ memory,

$$
\mathbf {P} _ {[ t ]} ^ {r} = \mathbf {I} - \sum_ {i = 1} ^ {r} \boldsymbol {w} _ {[ t ]} ^ {i} \boldsymbol {k} _ {[ t ]} ^ {i ^ {\top}}, \qquad \mathbf {H} _ {[ t ]} ^ {r} = \sum_ {i = 1} ^ {r} \boldsymbol {u} _ {[ t ]} ^ {i} \boldsymbol {k} _ {[ t ]} ^ {i ^ {\top}} \quad \in \mathbb {R} ^ {d \times d}\tag{6}
$$

$$
\boldsymbol {w} _ {[ t ]} ^ {r} = \beta_ {[ t ]} ^ {r} \left(\boldsymbol {k} _ {[ t ]} ^ {r} - \sum_ {i = 1} ^ {r - 1} \boldsymbol {w} _ {[ t ]} ^ {i} (\boldsymbol {k} _ {[ t ]} ^ {i ^ {\top}} \boldsymbol {k} _ {[ t ]} ^ {r})\right), \quad \boldsymbol {u} _ {[ t ]} ^ {r} = \beta_ {[ t ]} ^ {r} \left(\boldsymbol {v} _ {[ t ]} ^ {r} - \sum_ {i = 1} ^ {r - 1} \boldsymbol {u} _ {[ t ]} ^ {i} (\boldsymbol {k} _ {[ t ]} ^ {i ^ {\top}} \boldsymbol {k} _ {[ t ]} ^ {r})\right) \quad \in \mathbb {R} ^ {d}\tag{7}
$$

The derivations for the above can be found in the appendix. Subsequently, based on Eq. 5, we can obtain the chunk-level recurrence for hidden states and outputs as,

$$
\mathbf {S} _ {[ t ]} ^ {r} = \mathbf {S} _ {[ t ]} ^ {0} - \left(\mathbf {S} _ {[ t ]} ^ {0} \sum_ {i = 1} ^ {r} \boldsymbol {w} _ {[ t ]} ^ {i} \boldsymbol {k} _ {[ t ]} ^ {i ^ {\intercal}}\right) + \sum_ {i = 1} ^ {r} \boldsymbol {u} _ {[ t ]} ^ {i} \boldsymbol {k} _ {[ t ]} ^ {i ^ {\intercal}} = \mathbf {S} _ {[ t ]} ^ {0} + \sum_ {i = 1} ^ {r} \left(\boldsymbol {u} _ {[ t ]} ^ {i} - \mathbf {S} _ {[ t ]} ^ {0} \boldsymbol {w} _ {[ t ]} ^ {i}\right) \boldsymbol {k} _ {[ t ]} ^ {i ^ {\intercal}},
$$

$$
\boldsymbol {o} _ {[ t ]} ^ {r} = \mathbf {S} _ {[ t ]} ^ {r} \boldsymbol {q} _ {[ t ]} ^ {r} = \mathbf {S} _ {[ t ]} ^ {0} \boldsymbol {q} _ {[ t ]} ^ {r} + \sum_ {i = 1} ^ {r} \left(\boldsymbol {u} _ {[ t ]} ^ {i} - \mathbf {S} _ {[ t ]} ^ {0} \boldsymbol {w} _ {[ t ]} ^ {i}\right) \left(\boldsymbol {k} _ {[ t ]} ^ {i ^ {\intercal}} \boldsymbol {q} _ {[ t ]} ^ {i}\right).
$$

Letting ${ \bf S } _ { [ t ] } = { \bf S } _ { [ t ] } ^ { 0 }$ , the above can be simplified to matrix notations similarly to Eq.1-2,

$$
\mathbf {S} _ {[ t + 1 ]} = \mathbf {S} _ {[ t ]} + \left(\mathbf {U} _ {[ t ]} - \mathbf {W} _ {[ t ]} \mathbf {S} _ {[ t ]} ^ {\top}\right) ^ {\top} \mathbf {K} _ {[ t ]},\tag{8}
$$

$$
\mathbf {O} _ {[ t ]} = \mathbf {Q} _ {[ t ]} \mathbf {S} _ {[ t ]} ^ {\top} + (\mathbf {Q} _ {[ t ]} \mathbf {K} _ {[ t ]} ^ {\top} \odot \mathbf {M}) \left(\mathbf {U} _ {[ t ]} - \mathbf {W} _ {[ t ]} \mathbf {S} _ {[ t ]} ^ {\top}\right)\tag{9}
$$

where $\begin{array} { r } { \bigsqcup _ { [ t ] } = \bigsqcup _ { [ t ] } ^ { 1 : C } \in \mathbb { R } ^ { C \times d } \mathrm { f o r } \bigsqcup \in \{ \mathbf { Q } , \mathbf { K } , \mathbf { V } , \mathbf { O } , \mathbf { U } , \mathbf { W } \} } \end{array}$ defines the chunkwise matrices that are formed from stacking the ${ \bf q } _ { t } , { \bf k } _ { t } , { \bf v } _ { t } , o _ { t } , { \bf u } _ { t } , { \bf w } _ { t }$ vectors.

Practical considerations. In the above, Eq. 7 is fully recurrent and thus cannot use tensor cores written as is. To solve this, we further leverage the UT transform [44, 23] (see §B.2 for derivations):

$$
\mathbf {T} _ {[ t ]} = \left(\mathbf {I} + \operatorname{tril} (\operatorname{diag} (\beta_ {[ t ]}) \mathbf {K} _ {[ t ]} \mathbf {K} _ {[ t ]} ^ {\top}, - 1)\right) ^ {- 1} \operatorname{diag} \left(\beta_ {[ t ]}\right)\tag{10}
$$

$$
\mathbf {W} _ {[ t ]} = \mathbf {T} _ {[ t ]} \mathbf {K} _ {[ t ]}, \qquad \mathbf {U} _ {[ t ]} = \mathbf {T} _ {[ t ]} \mathbf {V} _ {[ t ]}\tag{11}
$$

to rewrite most operations in matmuls. The inverse of lower triangular matrices could be solved efficiently using forward substitution. Once computed, the hidden state updates (Eq. 8) and the output computations (Eq. 9) are largely the same as in vanilla linear attention. We adapt FLASHLINEARATTENTION [124] to implement $\operatorname { E q . }$ 8 and 9 with hidden states recomputed during the backward pass for saving GPU memory. The $\mathrm { P y } .$ Torch pseudocode for the forward pass is shown in Listing 8.

![](images/56bdcf0102cd47998f1c5c129536efffea3721392e549d412bf044dd1f0586e3.jpg)

Speed comparison. We implement both the pure recurrent form<sup>4</sup> and the chunkwise parallel form in Triton [112] and show the speed-ups for various sequence lengths (L) and head dimensions $( d _ { \mathrm { h e a d } } )$ in the right figure, where the model dimension d is

Figure 1: Speed-up of the chunkwise parallel form vs. the recurrent form.

2048 and we vary batch size and sequence length so that they multily to $1 6 3 8 4 . ^ { 5 }$ Our chunkwise algorithm achieves greater speed-ups as sequence length L and head dimension $d _ { \mathrm { h e a d } }$ increase, where the use of sequence-level parallelism (for high GPU occupancy) and tensor core (for fast matmuls) become more important [124, §3].

Fully Parallel Form for DeltaNet. For completeness, we also discuss the fully parallel form of DeltaNet. While we use the concept of a “pseudo” value, it is possible to avoid modifying values. From Eq. 4, it is straightforward to compute the attention matrix A: $\mathbf { A } _ { i j } = \boldsymbol { k } _ { j } ^ { \intercal } \mathbf { P } _ { j + 1 } ^ { i } \pmb { q } _ { i } \mathrm { i f } \ j \leq i$ and 0 otherwise. Notably, A has the matrix form $\mathbf { A } = \left( \mathbf { Q } \mathbf { K } ^ { T } \odot \mathbf { M } \right)$ T, obtained by combining Eq. 3 and   11. However, computing T requires a matrix inverse (Eq. 10), which scales cubically with sequence length without further algorithmic changes. Due to the above we avoid using the fully parallel form for training DeltaNet; however the “attention” matrix derived from this form could be of interest to the interpretability study for RNNs [3, 134].

## 3.3 DeltaNet Transformer

We describe how the DeltaNet layer primitive is used to build up a transformer-like model using standard modules. We largely follow the LLaMA-architecture [Transformer++, 114] and simply replace the self-attention layer with the DeltaNet layer. We also apply normalization before output projection for stable training [86, 69]. As the additional parameters for computing scalar $\beta _ { t }$ terms are negligible, parameter allocation is roughly the same as in Transformer++, $\mathrm { i . e . , ~ } 4 d ^ { 2 }$ for the DeltaNet layer and $8 d ^ { 2 }$ for the SwiGLU FFN layer [103].

![](images/ba357c35310e488dffacf5cbfcb8863582b89d4854342851e7bac91eaa0787a9.jpg)  
Figure 2: An illustration of DeltaNet neural architecture.

Feature map and normalization. Our key/query vectors are given by $\begin{array} { r } { \pmb { k } _ { t } = \frac { \mathrm { S i L U } ( \mathbf { W } _ { K } \pmb { x } _ { t } ) } { \| \mathrm { S i L U } ( \mathbf { W } _ { K } \pmb { x } _ { t } ) \| _ { 2 } } , \pmb { q } _ { t } = } \end{array}$ $\frac { \mathrm { S i L U } ( \mathbf { W } _ { Q } \pmb { x } _ { t } ) } { \| \mathrm { S i L U } ( \mathbf { W } _ { Q } \pmb { x } _ { t } ) \| _ { 2 } }$ . Schlag et al. [101] originally follow Katharopoulos et al. [48] and apply a $^ { 6 6 } \mathrm { E L U } + 1 ^ { 5 }$ [17] to nonlineary transform the key/query vectors. We instead use the SiLU activation [24], which was found to perform better [88, 19]. For stability, it is crucial to ensure that the norm of each eigenvalue of the transition matrices does not exceed one. The eigenvalues of $\mathbf { I } - \beta _ { t } \pmb { k } _ { t } \pmb { k } _ { t } ^ { \top }$ are 1 with multiplicity $d - 1$ and $1 - \beta _ { t } \| \pmb { k } _ { t } \| _ { 2 }$ with multiplicity 1. Schlag et al. [101] used the $L _ { 1 }$ norm to normalize query/key vectors, ensuring that $0 \leq 1 - \bar { \beta } _ { t } \| \boldsymbol k _ { t } \| _ { 2 } \leq 1$ . We instead apply $L _ { 2 }$ normalization, which we found to perform better and offers a more intuitive interpretation: when $\beta _ { t } = 1 , \mathbf { I } - k _ { t } \pmb { k } _ { t } ^ { \top }$ becomes a projection matrix, erasing information in one subspace while preserving the other $d - 1$ subspaces. This is beneficial for retaining information while enabling more targeted forgetting.

## 3.4 Hybrid Models

Following recent work on combining subquadratic token-mixing layers with existing neural network primitives [6, 21, 55], we also experiment with hybridizing DeltaNet models.

Convolutional layers. Recent linear recurrent models typically incorporate a lightweight depthwise-separable convolution layer after the query/key/value projections [31, 9, 19]. This “short convolution” layer [83] generalizes the shift SSM [26], and is efficient in both number of parameters and computational cost. We also add a short convolution layer after the query/key/value projections.

Local sliding window and global attention. Linear attention largely uses a content-based addressing mechanism [29] and lacks positional information [129]. Arora et al. [6] also argue that linear attention lacks the ability to perform precise local token shifts and comparisons, thus facing difficulties on retrieval-intensive tasks. Motivated by this, we experiment with two different hybrid architectures that incorporate softmax attention. We first explore sliding window attention (SWA) which has been shown to significantly improve linear attention [86, 6, 57, 75]; we follow Griffin [21] and Samba [95] to interleave DeltaNet layers and SWA layers. We also experiment with global attention, which has been found to be helpful [51, 35] even if only few of the recurrent layers are replaced with global attention [55]. We follow Fu et al. [26] to replace only two layers with global attention: the second layer and the $\begin{array} { r } { ( \frac { N } { 2 } + 1 ) } \end{array}$ -th layer, where N is total number of layers.

## 4 Empirical Study

We compare the DeltaNet against strong baselines in both synthetic and real-world language modeling settings. Our main baselines include: LLaMA-architecture Transformer++ [114]; RetNet [108], a linear attention Transformer with non-data-dependent exponential decay and large head dimension;

GLA [124], a linear attention Transformer with data-dependent decay; and Mamba [31], a selective state-space model with data-dependent decay.

## 4.1 Synthetic Benchmarks

We evaluate on three synthetic benchmarks: Multi-query associative recall [MQAR; 4], Mechanistic Architecture Design [MAD; 84], and in-context language learning [RegBench; 2].

<table><tr><td>Model</td><td>Compress</td><td>Fuzzy Recall</td><td>In-Context Recall</td><td>Memorize</td><td>Noisy Recall</td><td>Selective Copy</td><td>Average</td></tr><tr><td>Transformer</td><td>51.6</td><td>29.8</td><td>94.1</td><td>85.2</td><td>86.8</td><td>99.6</td><td>74.5</td></tr><tr><td>Hyena</td><td>45.2</td><td>7.9</td><td>81.7</td><td>89.5</td><td>78.8</td><td>93.1</td><td>66.0</td></tr><tr><td>Multihead Hyena</td><td>44.8</td><td>14.4</td><td>99.0</td><td>89.4</td><td>98.6</td><td>93.0</td><td>73.2</td></tr><tr><td>Mamba</td><td>52.7</td><td>6.7</td><td>90.4</td><td>89.5</td><td>90.1</td><td>86.3</td><td>69.3</td></tr><tr><td>GLA</td><td>38.8</td><td>6.9</td><td>80.8</td><td>63.3</td><td>81.6</td><td>88.6</td><td>60.0</td></tr><tr><td>DeltaNet</td><td>42.2</td><td>35.7</td><td>100</td><td>52.8</td><td>100</td><td>100</td><td>71.8</td></tr></table>

Figure 3: Results on the synthetic MAD benchmark. Results other than DeltaNet are directly borrowed from Poli et al. [84]. (Multi-head) Hyena, DeltaNet and Mamba make use of convolutions, whereas GLA does not.

![](images/316054db7ca964ab652f2b7a19df93fc5cad2b976573075db1bdea6da4186788.jpg)  
Figure 4: Accuracy (%) on MQAR.

MQAR evaluates language models’ ability to (in-context) recall information within a context when faced with multiple recall queries. We use Arora et al. [4]’s training setting and for DeltaNet we use 2 heads. We do not use convolutions for these experiments. Figure 4 shows that DeltaNet performs perfectly (even without convolution) in the hardest setting and outperforms Mamba (which uses convolutions) in the low-dimension setting. Next, we consider the MAD benchmark [84], a suite of synthetic token manipulation tasks designed to probe capabilities of model architectures. The results are shown in Table 3. Compared with other architectures, including MHA, DeltaNet is better at recalling tasks, especially on Fuzzy Recall as expected, although it somehow struggles on the “Memorize” task. We defer the RegBench results to the §A.2 due to space constraints.

## 4.2 Language Modeling

Experimental setup. Following prior work [31, 124], we evaluate on Wikitext perplexity and zero-shot common sense reasoning tasks, including LAMBADA [LMB.; 77], PiQA [12], HellaSwag [Hella.; 127], WinoGrande [Wino.; 99], ARC-easy (ARC-e) and ARC-challenge (Arc-c) [16]. Following Arora et al. [6], we also evaluate the models real-world recall-intensive tasks, including FDA [5], SWDE [60], and SQUAD [93]. Both SWDE and FDA focus on extracting structured information: SWDE from raw HTML to identify semi-structured relationships, and FDA from PDFs to retrieve key-value pairs. SQUAD evaluates language models on reading comprehension by providing a text passage and a related question. See §A.1 for hyperparameter settings.

Results. Our main language modeling results are shown in Table 1. Since Mamba uses convolutions by default while GLA does not, we retrain the GLA with convolution, and also train DeltaNet without convolution. For the 1.3B setting we only train the DeltaNet with convolution due to limited compute resources. In general we find that DeltaNet outperforms the strong Mamba/GLA baselines in terms of both perplexity and downstream task performance. For recall-intensive tasks (i.e., SWDE, SQuAD, FDA), we find that under the same state size at the 340M scale, DeltaNet outperforms GLA, confirming the effectiveness of the delta rule. However, at the 1.3B scale, DeltaNet underperforms GLA due to its poorer state size scability (see §5.3), since state size plays an important role in recallintensive tasks. Finally, we confirm the benefits of hybrid architectures [21, 55]: both the sliding window and global attention hybrids work well, outperforming the strong Transformer++ baselines. We also scale DeltaNet to the 3B parameter scale trained with 1T tokens using the same settings as Shen et al. [104]. The results are shown in Table 5, where 3B DeltaNet slightly underperforms a Transformer architecture trained with the same setting (PowerLM-3B), but outperforms other RNN baselines in the 2B–3B range (though these are trained for a different number of tokens so are not exactly comparable).

Ablations. In Table 1 (bottom) we ablate the choice of feature map and normalization. We find that simply replacing the L -norm with the L -norm greatly increases performance. For the feature map, we experiment with {ReLU, 1 + ELU, SiLU} and find that SiLU performs the best, consistent with prior work [88].

Training throughput. Figure 6 compares the training throughputs of different 1.3B models in different training lengths and batch size settings. The training speed of DeltaNet is close to GLA and significantly faster than Mamba. All linear-time models outperform Transformers for longersequence training.

<table><tr><td>Model</td><td>Wiki. ppl ↓</td><td>LMB. ppl ↓</td><td>LMB. acc ↑</td><td>PIQA acc ↑</td><td>Hella. acc_n ↑</td><td>Wino. acc ↑</td><td>ARC-e acc ↑</td><td>ARC-c acc_n ↑</td><td>Avg.</td><td>SWDE acc ↑</td><td>SQuAD acc ↑</td><td>FDA acc ↑</td><td>State exp.</td></tr><tr><td colspan="14">340M params / 15B tokens</td></tr><tr><td>Transformer++</td><td>28.39</td><td>42.69</td><td>31.0</td><td>63.3</td><td>34.0</td><td>50.4</td><td>44.5</td><td>24.2</td><td>41.2</td><td>42.2</td><td>22.1</td><td>21.4</td><td>N/A</td></tr><tr><td>RetNet (w/o. conv)</td><td>32.33</td><td>49.19</td><td>28.6</td><td>63.5</td><td>33.5</td><td>52.5</td><td>44.5</td><td>23.4</td><td>41.0</td><td>13.3</td><td>27.6</td><td>2.9</td><td>512x</td></tr><tr><td>Mamba (w. conv)</td><td>28.39</td><td>39.66</td><td>30.6</td><td>65.0</td><td>35.4</td><td>50.1</td><td>46.3</td><td>23.6</td><td>41.8</td><td>12.4</td><td>23.0</td><td>2.1</td><td>64x</td></tr><tr><td>GLA (w/o. conv)</td><td>28.65</td><td>43.35</td><td>30.3</td><td>64.8</td><td>34.5</td><td>51.4</td><td>45.1</td><td>22.7</td><td>41.5</td><td>18.6</td><td>27.2</td><td>8.1</td><td>128x</td></tr><tr><td>(w. conv)</td><td>29.47</td><td>45.53</td><td>31.3</td><td>65.1</td><td>33.8</td><td>51.6</td><td>44.4</td><td>24.6</td><td>41.8</td><td>24.0</td><td>24.7</td><td>7.3</td><td>128x</td></tr><tr><td>DeltaNet (w/o. conv)</td><td>29.08</td><td>50.87</td><td>30.0</td><td>63.6</td><td>33.6</td><td>51.7</td><td>46.0</td><td>23.0</td><td>41.3</td><td>24.6</td><td>26.9</td><td>4.5</td><td>128x</td></tr><tr><td>DeltaNet (w. conv)</td><td>28.24</td><td>37.37</td><td>32.1</td><td>64.8</td><td>34.3</td><td>52.2</td><td>45.8</td><td>23.5</td><td>42.1</td><td>26.4</td><td>28.9</td><td>12.8</td><td>128x</td></tr><tr><td>+ Sliding Attn</td><td>27.06</td><td>38.17</td><td>33.4</td><td>64.0</td><td>35.3</td><td>50.9</td><td>45.9</td><td>23.2</td><td>42.1</td><td>39.3</td><td>32.5</td><td>18.8</td><td>N/A</td></tr><tr><td>+ Global Attn (2 layers)</td><td>27.51</td><td>35.04</td><td>33.5</td><td>64.0</td><td>34.5</td><td>51.7</td><td>46.0</td><td>23.3</td><td>42.1</td><td>42.9</td><td>32.1</td><td>23.1</td><td>N/A</td></tr><tr><td colspan="14">1.3B params / 100B tokens</td></tr><tr><td>Transformer++</td><td>16.85</td><td>13.44</td><td>48.9</td><td>70.8</td><td>49.6</td><td>53.6</td><td>56.0</td><td>26.5</td><td>50.9</td><td>66.6</td><td>31.5</td><td>27.4</td><td>N/A</td></tr><tr><td>RetNet (w/o. conv)</td><td>18.64</td><td>17.27</td><td>43.3</td><td>70.0</td><td>47.3</td><td>52.5</td><td>54.8</td><td>25.6</td><td>48.9</td><td>42.8</td><td>34.7</td><td>14.3</td><td>512x</td></tr><tr><td>Mamba (w. conv)</td><td>17.06</td><td>13.89</td><td>46.2</td><td>72.2</td><td>40.1</td><td>54.1</td><td>59.0</td><td>28.2</td><td>50.0</td><td>41.4</td><td>35.2</td><td>6.2</td><td>64x</td></tr><tr><td>GLA (w/o. conv)</td><td>17.22</td><td>14.47</td><td>46.9</td><td>71.8</td><td>49.8</td><td>53.9</td><td>57.2</td><td>26.6</td><td>51.0</td><td>50.6</td><td>42.6</td><td>19.9</td><td>256x</td></tr><tr><td>(w. conv)</td><td>17.25</td><td>14.92</td><td>46.2</td><td>70.6</td><td>49.9</td><td>53.0</td><td>55.3</td><td>27.0</td><td>50.4</td><td>52.4</td><td>37.4</td><td>22.3</td><td>256x</td></tr><tr><td>DeltaNet (w. conv)</td><td>16.87</td><td>12.21</td><td>48.9</td><td>71.2</td><td>50.2</td><td>53.6</td><td>57.2</td><td>28.3</td><td>51.6</td><td>49.5</td><td>37.4</td><td>17.2</td><td>128x</td></tr><tr><td>+ Sliding Attn</td><td>16.56</td><td>11.74</td><td>49.2</td><td>71.8</td><td>51.1</td><td>52.8</td><td>58.9</td><td>28.8</td><td>52.1</td><td>53.3</td><td>43.3</td><td>22.3</td><td>N/A</td></tr><tr><td>+ Global Attn (2 layers)</td><td>16.55</td><td>12.40</td><td>48.8</td><td>70.8</td><td>50.7</td><td>54.2</td><td>58.4</td><td>28.1</td><td>51.8</td><td>71.0</td><td>43.0</td><td>29.8</td><td>N/A</td></tr><tr><td colspan="14">DeltaNet Ablations (340M)</td></tr><tr><td>w.  $L_1$ -norm &amp; 1+ELU</td><td>31.12</td><td>55.96</td><td>26.3</td><td>63.9</td><td>33.0</td><td>50.9</td><td>44.3</td><td>21.8</td><td>40.1</td><td>14.5</td><td>23.9</td><td>6.2</td><td>128x</td></tr><tr><td>w.  $L_2$ -norm &amp; 1+ELU</td><td>28.03</td><td>37.62</td><td>32.2</td><td>65.7</td><td>34.7</td><td>51.8</td><td>45.4</td><td>22.5</td><td>42.1</td><td>23.8</td><td>28.6</td><td>13.1</td><td>128x</td></tr><tr><td>w.  $L_2$ -norm &amp; ReLU</td><td>28.75</td><td>43.53</td><td>30.2</td><td>64.0</td><td>33.9</td><td>48.9</td><td>45.6</td><td>22.8</td><td>40.9</td><td>27.2</td><td>26.7</td><td>9.0</td><td>128x</td></tr></table>

Table 1: Main language modeling results against Transformer++, RetNet [108], Mamba [31], and GLA [124]. All models are trained on the same subset of the SlimPajama dataset with the Mistral tokenizer. The Transformer++, RetNet, Mamba, GLA (w/o. conv) results are taking from Yang et al. [124]. For hybrid models, “Sliding Attn” interleaves a sliding window attention every other layer, and “Global Attn” uses full global attention on two layers. The 340M/1.3B models are trained for 15B/100B tokens respectively. All results are obtained through lm-evaluation-harness [27]. The last column denotes the expansion ratio of the recurrent state size relative to the product of the number of layers and model dimension (see Zhang et al. [131, App. C]).

<table><tr><td>Model</td><td>ARC</td><td>HellaSwag</td><td>OBQA</td><td>PIQA</td><td>WinoGrande</td><td>MMLU</td><td>Average</td></tr><tr><td>Llama-3.2-3B [111]</td><td>59.1</td><td>73.6</td><td>43.4</td><td>77.5</td><td>69.2</td><td>54.1</td><td>62.8</td></tr><tr><td>PowerLM-3B [104]</td><td>60.5</td><td>74.6</td><td>43.6</td><td>79.9</td><td>70.0</td><td>45.0</td><td>62.3</td></tr><tr><td>DeltaNet-3B</td><td>60.4</td><td>72.8</td><td>41.0</td><td>78.5</td><td>65.7</td><td>40.7</td><td>59.8</td></tr><tr><td>RecurrentGemma-2B [30]</td><td>57.0</td><td>71.1</td><td>42.0</td><td>78.2</td><td>67.6</td><td>31.8</td><td>57.9</td></tr><tr><td>RWKV-6-3B [79]</td><td>49.5</td><td>68.6</td><td>40.6</td><td>76.8</td><td>65.4</td><td>28.4</td><td>54.9</td></tr><tr><td>Mamba-2.7B [31]</td><td>50.3</td><td>65.3</td><td>39.4</td><td>75.8</td><td>63.1</td><td>26.1</td><td>53.3</td></tr></table>

Figure 5: Zero-shot model performance across selected benchmarks for 3B models. Llama-3.2-3B and PowerLM-3B are Transformer models, while the others are recurrent models. ARC results are averaged over normalized accuracy across ARC-Easy and ARC-Challenge.

![](images/80f283283617ebafeb0ef06685e5dd95afc1f187ef56bc7339b113d333b3fdbe.jpg)  
Figure 6: Training throughput of 1.3B models on a single H100.

## 5 Discussion and Related Work

## 5.1 DeltaNet vs. State Space Models / Linear RNNs

To discuss DeltaNet against existing linear RNNs (including state-space models) we first introduce a general class of associative RNNs with matrix-valued hidden states. Given a matrix-valued hidden state $\mathbf { S } _ { t } \in \mathbb { R } ^ { d \times n }$ and current input $\pmb { x } _ { t } \in \mathbb { R } ^ { d }$ , these models have the following form:

$$
\begin{array}{l l} \mathbf {S} _ {t} = \mathbf {S} _ {t - 1} \bullet \mathbf {M} _ {t} + \boldsymbol {v} _ {t} \boldsymbol {k} _ {t} ^ {\top}, & \text {(recurrence)} \\ \boldsymbol {o} _ {t} = \mathbf {S} _ {t} \boldsymbol {q} _ {t}, & \text {(memory read - out)} \end{array}
$$

where • is an associative operator (e.g., Hadamard product, matrix multiplication, etc.). The matrix M<sub>t</sub> and vectors ${ \pmb v } _ { t } , { \pmb k } _ { t } , { \pmb q } _ { t }$ are (potentially non-linear) functions of the current input $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$

As is the case in vector-valued linear RNNs [64, 106], the use of an associative operator enables the use of parallel scan [13] to calculate $\mathbf { S } _ { 1 } , \ldots , \mathbf { S } _ { L }$ in O(log $L )$ steps and $O ( L )$ work (ignoring the terms associated with the associative operation) if the inputs $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { L }$ are given (though see our discussion in footnote 1). Hence, as long as the associative operator is not too expensive, training can be efficient. However, parallel scan by itself is not sufficient for training language models at practical scale due to some associative operator’s being too expensive. Recent models such as such as Mamba [31] and gated linear attention Transformers [108, 124, 92, 79, 9] thus make use of cheap element-wise recurrence updates, in particular the Hadamard product, $\mathrm { i . e . , } \bullet = \odot$ . See Table 2 for how recent models can be cast into this form.

<table><tr><td>Model</td><td>Recurrence</td><td>Memory read-out</td></tr><tr><td>Linear Attention [48, 47]</td><td> $\mathbf{S}_{t} = \mathbf{S}_{t-1} + \boldsymbol{v}_{t} \boldsymbol{k}_{t}^{\top}$ </td><td> $\boldsymbol{o}_{t} = \mathbf{S}_{t} \boldsymbol{q}_{t}$ </td></tr><tr><td>+ Kernel</td><td> $\mathbf{S}_{t} = \mathbf{S}_{t-1} + \boldsymbol{v}_{t} \phi(\boldsymbol{k}_{t})^{\top}$ </td><td> $\boldsymbol{o}_{t} = \mathbf{S}_{t} \phi(\boldsymbol{q}_{t})$ </td></tr><tr><td>+ Normalization</td><td> $\mathbf{S}_{t} = \mathbf{S}_{t-1} + \boldsymbol{v}_{t} \phi(\boldsymbol{k}_{t})^{\top}, \boldsymbol{z}_{t} = \boldsymbol{z}_{t-1} + \phi(\boldsymbol{k}_{t})$ </td><td> $\boldsymbol{o}_{t} = \mathbf{S}_{t} \phi(\boldsymbol{q}_{t}) / (\boldsymbol{z}_{t}^{\top} \phi(\boldsymbol{q}_{t}))$ </td></tr><tr><td>DeltaNet [101]</td><td> $\mathbf{S}_{t} = \mathbf{S}_{t-1} (\mathbf{I} - \beta_{t} \boldsymbol{k}_{t} \boldsymbol{k}_{t}^{\top}) + \beta_{t} \boldsymbol{v}_{t} \boldsymbol{k}_{t}^{\top}$ </td><td> $\boldsymbol{o}_{t} = \mathbf{S}_{t} \boldsymbol{q}_{t}$ </td></tr><tr><td>Gated RFA [81]</td><td> $\mathbf{S}_{t} = g_{t} \mathbf{S}_{t-1} + (1 - g_{t}) \boldsymbol{v}_{t} \boldsymbol{k}_{t}^{\top}, \boldsymbol{z}_{t} = g_{t} \boldsymbol{z}_{t-1} + (1 - g_{t}) \boldsymbol{k}_{t}$ </td><td> $\boldsymbol{o}_{t} = \mathbf{S}_{t} \boldsymbol{q}_{t} / (\boldsymbol{z}_{t}^{\top} \boldsymbol{q}_{t})$ </td></tr><tr><td>S4 [32, 106]</td><td> $\mathbf{S}_{t} = \mathbf{S}_{t-1} \odot \exp(-(\boldsymbol{\alpha} \mathbf{1}^{\top}) \odot \exp(\boldsymbol{A})) + \boldsymbol{B} \odot (\boldsymbol{v}_{t} \mathbf{1}^{\top})$ </td><td> $\boldsymbol{o}_{t} = (\mathbf{S}_{t} \odot \boldsymbol{C}) \mathbf{1} + \boldsymbol{d} \odot \boldsymbol{v}_{t}$ </td></tr><tr><td>ABC [82]</td><td> $\mathbf{S}_{t}^{k} = \mathbf{S}_{t-1}^{k} + \boldsymbol{k}_{t} \phi_{t}^{\top}, \mathbf{S}_{t}^{v} = \mathbf{S}_{t-1}^{v} + \boldsymbol{v}_{t} \phi_{t}^{\top}$ </td><td> $\boldsymbol{o}_{t} = \mathbf{S}_{t}^{v} \text{softmax} (\mathbf{S}_{t}^{k} \boldsymbol{q}_{t})$ </td></tr><tr><td>DFW [63]</td><td> $\mathbf{S}_{t} = \mathbf{S}_{t-1} \odot (\boldsymbol{\beta}_{t} \boldsymbol{\alpha}_{t}^{\top}) + \boldsymbol{v}_{t} \boldsymbol{k}_{t}^{\top}$ </td><td> $\boldsymbol{o}_{t} = \mathbf{S}_{t} \boldsymbol{q}_{t}$ </td></tr><tr><td>RetNet [108]</td><td> $\mathbf{S}_{t} = \gamma \mathbf{S}_{t-1} + \boldsymbol{v}_{t} \boldsymbol{k}_{t}^{\top}$ </td><td> $\boldsymbol{o}_{t} = \mathbf{S}_{t} \boldsymbol{q}_{t}$ </td></tr><tr><td>Mamba [31]</td><td> $\mathbf{S}_{t} = \mathbf{S}_{t-1} \odot \exp(-(\boldsymbol{\alpha}_{t} \mathbf{1}^{\top}) \odot \exp(\boldsymbol{A})) + (\boldsymbol{\alpha}_{t} \odot \boldsymbol{v}_{t}) \boldsymbol{k}_{t}^{\top}$ </td><td> $\boldsymbol{o}_{t} = \mathbf{S}_{t} \boldsymbol{q}_{t} + \boldsymbol{d} \odot \boldsymbol{v}_{t}$ </td></tr><tr><td>GLA [124]</td><td> $\mathbf{S}_{t} = \mathbf{S}_{t-1} \odot (\mathbf{1} \boldsymbol{\alpha}_{t}^{\top}) + \boldsymbol{v}_{t} \boldsymbol{k}_{t}^{\top} = \mathbf{S}_{t-1} \text{Diag}(\boldsymbol{\alpha}_{t}) + \boldsymbol{v}_{t} \boldsymbol{k}_{t}^{\top}$ </td><td> $\boldsymbol{o}_{t} = \mathbf{S}_{t} \boldsymbol{q}_{t}$ </td></tr><tr><td>RWKV-6 [79]</td><td> $\mathbf{S}_{t} = \mathbf{S}_{t-1} \text{Diag}(\boldsymbol{\alpha}_{t}) + \boldsymbol{v}_{t} \boldsymbol{k}_{t}^{\top}$ </td><td> $\boldsymbol{o}_{t} = (\mathbf{S}_{t-1} + (\boldsymbol{d} \odot \boldsymbol{v}_{t}) \boldsymbol{k}_{t}^{\top}) \boldsymbol{q}_{t}$ </td></tr><tr><td>HGRN-2 [92]</td><td> $\mathbf{S}_{t} = \mathbf{S}_{t-1} \text{Diag}(\boldsymbol{\alpha}_{t}) + \boldsymbol{v}_{t} (\mathbf{1} - \boldsymbol{\alpha}_{t})^{\top}$ </td><td> $\boldsymbol{o}_{t} = \mathbf{S}_{t} \boldsymbol{q}_{t}$ </td></tr><tr><td>mLSTM [9]</td><td> $\mathbf{S}_{t} = f_{t} \mathbf{S}_{t-1} + i_{t} \boldsymbol{v}_{t} \boldsymbol{k}_{t}^{\top}, \boldsymbol{z}_{t} = f_{t} z_{t-1} + i_{t} \boldsymbol{k}_{t}$ </td><td> $\boldsymbol{o}_{t} = \mathbf{S}_{t} \boldsymbol{q}_{t} / \max\{1, |\boldsymbol{z}_{t}^{\top} \boldsymbol{q}_{t}| \}$ </td></tr><tr><td>Mamba-2 [19]</td><td> $\mathbf{S}_{t} = \gamma_{t} \mathbf{S}_{t-1} + \boldsymbol{v}_{t} \boldsymbol{k}_{t}^{\top}$ </td><td> $\boldsymbol{o}_{t} = \mathbf{S}_{t} \boldsymbol{q}_{t}$ </td></tr><tr><td>GSA [131]</td><td> $\mathbf{S}_{t}^{k} = \mathbf{S}_{t-1}^{k} \text{Diag}(\boldsymbol{\alpha}_{t}) + \boldsymbol{k}_{t} \phi_{t}^{\top}, \mathbf{S}_{t}^{v} = \mathbf{S}_{t-1}^{v} \text{Diag}(\boldsymbol{\alpha}_{t}) + \boldsymbol{v}_{t} \phi_{t}^{\top}$ </td><td> $\boldsymbol{o}_{t} = \mathbf{S}_{t}^{v} \text{softmax} (\mathbf{S}_{t}^{k} \boldsymbol{q}_{t})$ </td></tr><tr><td>Gated DeltaNet [125]</td><td> $\mathbf{S}_{t} = \mathbf{S}_{t-1} \left( \alpha_{t} (\mathbf{I} - \beta_{t} \boldsymbol{k}_{t} \boldsymbol{k}_{t}^{\top}) \right) + \beta_{t} \boldsymbol{v}_{t} \boldsymbol{k}_{t}^{\top}$ </td><td> $\boldsymbol{o}_{t} = \mathbf{S}_{t} \boldsymbol{q}_{t}$ </td></tr></table>

Table 2: Overview of recent linear recurrent models that have been proposed and applied to autoregressive language modeling (ordered in rough chronological order). These works make use of a matrix-valued hidden state $\breve { \mathbf { S } _ { t } } \in \mathbb { R } ^ { d \times n }$ (or two matrix-valued hidden states $\mathbf { S } _ { t } ^ { k } , \mathbf { S } _ { t } ^ { v } , \mathbf { e . g . } , [ 8 2 , 1 3 1 ] )$ updated through an associative recurrence followed by an outer-product-based addition. Here ⊙ is the Hadamard product. Some models make use of an additional linear RNN with hidden state vector $\scriptstyle { z _ { t } }$ , which used to normalized the query vector $\mathbf { \nabla } _ { \mathbf { \eta } } \mathbf { q } _ { t }$ Variables with the subscript $t \ ( \mathbf { e } . \mathbf { g } . , \ v _ { t } , \alpha _ { t } , f _ { t } , \gamma _ { t } )$ are (potentially non-linear) functions of the current input $\mathbf { \Delta } \mathbf { x } _ { t } .$ . Non-time-varying parameters $( \mathrm { e . g . } , A , d , \gamma )$ are denoted without subscripts; these parameters are either learned or set to fixed values. Matrices are denoted with bold upper case letters, vectors with bold lower case, and scalars with italic letters. Many models make use of a kernel φ (e.g., [101, 81]) but we subsume them into the key/value vectors to reduce notational clutter.

Standard matrix multiplications $( \mathrm { i . e . , } \ \mathbf { S } _ { t - 1 } \bullet \mathbf { M } _ { t } = \mathbf { S } _ { t - 1 } \mathbf { M } _ { t } )$ on the other hand can model richer interactions that go beyond elementwise recurrence. Without any structural assumptions on ${ { \bf { M } } _ { t } }$ however, these operations would take $O ( d n ^ { 2 } )$ for each update (as opposed to $O ( d n )$ for elementwise products), which would be prohibitively expensive. Hence, DeltaNet’s use of $\mathbf { M } _ { t } = \mathbf { I } - \beta _ { t } \pmb { k } _ { t } \pmb { k } _ { t } ^ { \top }$ can be seen as exploiting structured matrices to efficiently model interactions beyond elementwise recurrences. Our chunkwise algorithm could generalize to a broader class of matrices in the Diagonal-Plus-Low-Rank (DPLR) form $\mathbf { M } _ { t } = \mathbf { D } - \mathbf { a } _ { t } \mathbf { b } _ { t } ^ { \mathsf { T } }$ , which has been explored in S4 [32], although their DPLR transition matrices are data-independent. We adopt DeltaNet’s parameterization in this work $( \mathrm { i . e . , D } = \mathbf { I } , \pmb { a } _ { t } = \beta _ { t } \pmb { k } _ { t } , \pmb { b } _ { t } = \pmb { k } _ { t } )$ as we are primarily interested in improving recall (through DeltaNet’s key-value update rule) while maintaining parameter efficiency. We leave the exploration of more generalized parameterizations for future work.

## 5.2 Towards a Unifying Framework for Efficient Autoregressive Sequence Transformations

While the above class of models makes it possible to unify recent models, we do not claim that it is the $\mathbf { \ddot { r i g h t } } ^ { \mathbf { \vec { r } } }$ level at which view (autoregressive) sequence transformations of the form $\{ \pmb { x } _ { t } \} _ { t = 1 } ^ { L } \ \mapsto \ \{ \pmb { o } _ { t } \} _ { t = 1 } ^ { L }$ , where $\mathbf { } _ { o _ { t } }$ cannot depend on any $x _ { j } ~ \mathrm { i f } ~ j ~ > ~ t .$ For example, this framing makes it difficult to (neatly) capture other subquadratic models that have been shown to be effective [126, 49, 98, 83]. An alternative unifying framework might be to view the above sequence transformations as a discretization of a continuous state space model [32, 106, 31], or as a matrix multiplication with a masked structured matrix [76, 87, 46, 19]. What does seem important, however, is that a framework should ideally expose efficient algorithms for training, and the algorithm should be hardware-efficient, which, in the case of modern GPUs, means that it should be rich in matrix multiplications. From this perspective, the state-space duality (SSD) framework recently proposed by Dao and Gu [19], which provides a connection between SSM-based sequence transformations and structured matrix multiplications with a semiseparable matrix, seems a promising candidate. However, this framework may not capture an important class of models, $\mathrm { e . g . }$ , models where the associative recurrence involves matrix multiplication with an unstructured matrix, or models that make use of more exotic associative operators (e.g., in Peng et al. [80]).

Finally, we observe that there have been many recent linear-time models that have been proposed which purportedly match or outperform classic transformers. As can be seen in Table 2, the “sequence mixing” component of these works are closely related to one another. However, the way in which the token-mixing primitive is used to build up a transformer-like model varies widely. For example, while most recent works make use of depthwise-separable convolution layers (not shown in Table 2) [31, 9, 19, 15, 125], earlier works generally do not [48, 101, 81]. There are also differences in the parameterizations of the feedforward layers used for the “channel mixing” component. Such variations should be taken into account before declaring a particular model layer superior to another.

## 5.3 Limitations and Future Work

Our work has several limitations. First, in terms of computation, although we propose a new hardware-efficient algorithm, the training speed still lags behind that of GLA. This is due to the overhead caused by modeling state-to-state dependencies as described above, which requires “marginalizing” over the head dimension inside the kernel, similar to the case of softmax attention. However, for GLA since there are no intra-state dependencies (everything is elementwise), and thus it is easy to use tiling to support arbitrary size of head dimension, as implemented in Yang and Zhang [123]. This limitation would potentially limit DeltaNet’s memory size, consequently lowering the recall intensive task performance as we observed in §4.2. However, it may be feasible to adopt block di agonal generalized Householder transition matrices with block sizes fitting GPU SRAM (e.g., 128) while maintaining a overall large head dimension (and thus a large recurrent state size).

We also found that the length generalization of DeltaNet was limited, while GLA and RetNet (and Mamba to an extent) have been found to be able to extrapolate beyond the training length [124]. We speculate that this is because DeltaNet lacks explicit decay factors. This could be improved through incorporating a gating term in the recurrence, as demonstrated in a recent work by Yang et al. [125].

## 6 Related Work

We briefly discuss related work here and give an extended discussion in Appendix C.

Linear transformers can be seen as a type of iterated Hopfield networks [72], and this connection can provide perspectives on the limitations and improvements of linear attention transformers. For example, vanilla linear transformers use a Hebbian-like update rule, which has been shown to have limited memory capacity [68]. Later works in Hopfield networks use higher-order polynomials [22] and exponential kernels [94, 50] to enhance the memory capacity, which is also related to linear attention with polynomial kernels [45, 6, 1]. On the other hand, the delta rule has been shown to have better memory capacity [28, 85, 56, 101]. In this sense, given the fixed size recurrent state, using the delta rule is able to achieve a better frontier of the recall-memory tradeoff curve [6], and has recently been applied to enhance real-world retrieval tasks [74, 96]. Moreover, it outperforms the additive rule used in vanilla linear transformers across multiple domains [101, 38, 40, 36, 42].

Despite these advantages, Irie et al. [42] revealed theoretical limitations of the delta update rule in terms of expressiveness. Recurrent enhancements of DeltaNet, such as Recurrent DeltaNet [37] and the Modern Self-Referential Weight Matrix [41], and the mesa-layer [117] were proposed and found superior. However, these models extend beyond linear RNNs and cannot be parallelized across sequence length. This suggests a fundamental trade-off between parallelism and expressiveness [70]. How to further enhance DeltaNet without sacrificing parallelism remains an open question, and the hybrid cross-chunk nonlinear and intra-chunk linear strategy used in TTT [110] might provide a suitable middle ground. Finally, we remark that delta rule is closely related to meta or online learning via gradient descent [73, 39], which has been revisited in recent works like Longhorn [58] and TTT [110]. Recently, Titans [10] improves TTT by introducing a momentum and weight decay term.

## 7 Conclusion

We describe an algorithm that parallelizes DeltaNet training across the sequence length dimension, achieving significant speed-ups against existing implementations on modern hardware. This makes it possible to scale up DeltaNet to moderate-scale language modeling settings, where we find that it performs well compared to recent linear-recurrent baselines.

## Acknowledgements

This study was supported by funds from MIT-IBM Watson AI Lab and the MIT-Google Program for Computing Innovation. We are grateful to Mayank Mishra for assistance with training and evaluating the 3B models, to Kazuki Irie for valuable feedback on the draft, and to Simran Arora, Liliang Ren and Eric Alcaide for their insightful discussions. We also thank Michael Poli and Armin Thomas for sharing the raw results from the MAD benchmark experiment, as well as to Fan Zhou for identifying an error in Eq. 10.

## References

[1] Y. Aksenov, N. Balagansky, S. M. L. C. Vaina, B. Shaposhnikov, A. Gorbatovski, and D. Gavrilov. Linear Transformers with Learnable Kernel Functions are Better In-Context Models, 2024. URL https://arxiv.org/abs/2402.10644.

[2] E. Akyürek, B. Wang, Y. Kim, and J. Andreas. In-context language learning: Arhitectures and algorithms. ArXiv preprint, abs/2401.12973, 2024. URL https://arxiv.org/abs/ 2401.12973.

[3] A. Ali, I. Zimerman, and L. Wolf. The hidden attention of mamba models, 2024.

[4] S. Arora, S. Eyuboglu, A. Timalsina, I. Johnson, M. Poli, J. Zou, A. Rudra, and C. Ré. Zoology: Measuring and improving recall in efficient language models. ArXiv preprint, abs/2312.04927, 2023. URL https://arxiv.org/abs/2312.04927.

[5] S. Arora, B. Yang, S. Eyuboglu, A. Narayan, A. Hojel, I. Trummer, and C. Ré. Language Mod els Enable Simple Systems for Generating Structured Views of Heterogeneous Data Lakes, 2023. URL https://arxiv.org/abs/2304.09433.

[6] S. Arora, S. Eyuboglu, M. Zhang, A. Timalsina, S. Alberti, D. Zinsley, J. Zou, A. Rudra, and C. Ré. Simple linear attention language models balance the recall-throughput tradeoff. ArXiv preprint, abs/2402.18668, 2024. URL https://arxiv.org/abs/2402.18668.

[7] S. Arora, A. Timalsina, A. Singhal, B. Spector, S. Eyuboglu, X. Zhao, A. Rao, A. Rudra, and C. Ré. Just read twice: closing the recall gap for recurrent language models, 2024. URL https://arxiv.org/abs/2407.05483.

[8] D. Bahdanau, K. Cho, and Y. Bengio. Neural machine translation by jointly learning to align and translate. In Y. Bengio and Y. LeCun, editors, 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, 2015. URL http://arxiv.org/abs/1409.0473.

[9] M. Beck, K. Pöppel, M. Spanring, A. Auer, O. Prudnikova, M. Kopp, G. Klambauer, J. Brandstetter, and S. Hochreiter. xlstm: Extended long short-term memory. ArXiv preprint, abs/2405.04517, 2024. URL https://arxiv.org/abs/2405.04517.

[10] A. Behrouz, P. Zhong, and V. Mirrokni. Titans: Learning to memorize at test time, 2024. URL https://arxiv.org/abs/2501.00663.

[11] C. H. Bischof and C. V. Loan. The WY representation for products of householder matrices. In SIAM Conference on Parallel Processingfor Scientific Computing, 1985. URL https:// api.semanticscholar.org/CorpusID:36094006.

[12] Y. Bisk, R. Zellers, R. LeBras, J. Gao, and Y. Choi. PIQA: reasoning about physical commonsense in natural language. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications ofArtificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020, pages 7432–7439. AAAI Press, 2020. URL https://aaai.org/ojs/index.php/AAAI/article/view/6239.

[13] G. E. Blelloch. Prefix sums and their applications. 1990.

[14] W. Brandon, A. Nrusimha, K. Qian, Z. Ankner, T. Jin, Z. Song, and J. Ragan-Kelley. Striped Attention: Faster Ring Attention for Causal Transformers. ArXiv preprint, abs/2311.09431, 2023. URL https://arxiv.org/abs/2311.09431.

[15] Y. Chou, M. Yao, K. Wang, Y. Pan, R.-J. Zhu, J. Wu, Y. Zhong, Y. Qiao, B. XU, and G. Li. MetaLA: Unified optimal linear approximation to softmax attention map. In The Thirtyeighth Annual Conference on Neural Information Processing Systems, 2024. URL https:// openreview.net/forum?id=Y8YVCOMEpz.

[16] P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. ArXiv preprint, abs/1803.05457, 2018. URL https://arxiv.org/abs/1803.05457.

[17] D. Clevert, T. Unterthiner, and S. Hochreiter. Fast and accurate deep network learning by exponential linear units (elus). In Y. Bengio and Y. LeCun, editors, 4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings, 2016. URL http://arxiv.org/abs/1511.07289.

[18] T. Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. ArXiv preprint, abs/2307.08691, 2023. URL https://arxiv.org/abs/2307.08691.

[19] T. Dao and A. Gu. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality. arXiv preprint arXiv: 2405.21060, 2024.

[20] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper\_files/paper/2022/hash/ 67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html.

[21] S. De, S. L. Smith, A. Fernando, A. Botev, G. Cristian-Muraru, A. Gu, R. Haroun, L. Berrada, Y. Chen, S. Srinivasan, G. Desjardins, A. Doucet, D. Budden, Y. W. Teh, R. Pascanu, N. De Freitas, and C. Gulcehre. Griffin: Mixing Gated Linear Recurrences with Local Atten tion for Efficient Language Models, 2024. URL https://arxiv.org/abs/2402.19427.

[22] M. Demircigil, J. Heusel, M. Löwe, S. Upgang, and F. Vermet. On a model of associative memory with huge storage capacity. ArXiv preprint, abs/1702.01929, 2017. URL https:// arxiv.org/abs/1702.01929.

[23] A. E. T. Dominguez and E. S. Q. Orti. Fast blocking of householder reflectors on graphics processors. 2018 26th Euromicro International Conference on Parallel, Distributed and Networkbased Processing (PDP), pages 385–393, 2018. URL https://api.semanticscholar. org/CorpusID:46960439.

[24] S. Elfwing, E. Uchibe, and K. Doya. Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning, 2017. URL https://arxiv.org/ abs/1702.03118.

[25] M. Fathi, J. Pilault, P.-L. Bacon, C. Pal, O. Firat, and R. Goroshin. Block-state transformer. ArXiv preprint, abs/2306.09539, 2023. URL https://arxiv.org/abs/2306.09539.

[26] D. Y. Fu, T. Dao, K. K. Saab, A. W. Thomas, A. Rudra, and C. Ré. Hungry hungry hippos: Towards language modeling with state space models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/pdf?id=COZDy0WYGg.

[27] L. Gao, J. Tow, S. Biderman, S. Black, A. DiPofi, C. Foster, L. Golding, J. Hsu, K. McDonell, N. Muennighoff, J. Phang, L. Reynolds, E. Tang, A. Thite, B. Wang, K. Wang, and A. Zou. A framework for few-shot language model evaluation, 2021.

[28] E. Gardner. The space of interactions in neural network models. Journal of Physics A, 21: 257–270, 1988.

[29] A. Graves, G. Wayne, and I. Danihelka. Neural Turing Machines, 2014. URL http:// arxiv.org/abs/1410.5401. arXiv:1410.5401 [cs].

[30] R. Griffin and G. Teams. Recurrentgemma: Moving past transformers for efficient open language models. ArXiv preprint, abs/2404.07839, 2024. URL https://arxiv.org/abs/ 2404.07839.

[31] A. Gu and T. Dao. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. In Proceedings of COLM, 2023.

[32] A. Gu, K. Goel, and C. Ré. Efficiently modeling long sequences with structured state spaces. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview.net/forum? id=uYLFoz1vlAC.

[33] K. Helfrich, D. Willmott, and Q. Ye. Orthogonal recurrent neural networks with scaled cayley transform. In J. G. Dy and A. Krause, editors, Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018, volume 80 of Proceedings of Machine Learning Research, pages 1974–1983. PMLR, 2018. URL http://proceedings.mlr.press/v80/helfrich18a.html.

[34] W. Hua, Z. Dai, H. Liu, and Q. V. Le. Transformer quality in linear time. In K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvári, G. Niu, and S. Sabato, editors, International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pages 9099–9117. PMLR, 2022. URL https://proceedings.mlr.press/v162/hua22a.html.

[35] F. Huang, K. Lu, Y. Cai, Z. Qin, Y. Fang, G. Tian, and G. Li. Encoding recurrence into transformers. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview. net/pdf?id=7YfHla7IxBJ.

[36] K. Irie and J. Schmidhuber. Images as weight matrices: Sequential image generation through synaptic learning rules. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https:// openreview.net/pdf?id=ddad0PNUvV.

[37] K. Irie, I. Schlag, R. Csordás, and J. Schmidhuber. Going beyond linear transformers with recurrent fast weight programmers. In M. Ranzato, A. Beygelzimer, Y. N. Dauphin, P. Liang, and J. W. Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 7703–7717, 2021. URL https://proceedings.neurips.cc/paper/ 2021/hash/3f9e3767ef3b10a0de4c256d7ef9805d-Abstract.html.

[38] K. Irie, I. Schlag, R. Csordás, and J. Schmidhuber. Going beyond linear transformers with recurrent fast weight programmers. In M. Ranzato, A. Beygelzimer, Y. N. Dauphin, P. Liang, and J. W. Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 7703–7717, 2021. URL https://proceedings.neurips.cc/paper/ 2021/hash/3f9e3767ef3b10a0de4c256d7ef9805d-Abstract.html.

[39] K. Irie, R. Csordás, and J. Schmidhuber. The dual form of neural networks revisited: Connecting test time predictions to training patterns via spotlights of attention. In K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvári, G. Niu, and S. Sabato, editors, International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pages 9639–9659. PMLR, 2022. URL https://proceedings.mlr.press/v162/irie22a.html.

[40] K. Irie, F. Faccio, and J. Schmidhuber. Neural differential equations for learning to program neural nets through continuous learning rules. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Process-

ing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper\_files/paper/2022/hash/ fc09b26b85ab3abb2832bd555a2e4215-Abstract-Conference.html.

[41] K. Irie, I. Schlag, R. Csordás, and J. Schmidhuber. A modern self-referential weight matrix that learns to modify itself. In K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvári, G. Niu, and S. Sabato, editors, International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings ofMachine Learning Research, pages 9660–9677. PMLR, 2022. URL https://proceedings.mlr.press/v162/ irie22b.html.

[42] K. Irie, R. Csordás, and J. Schmidhuber. Practical computational power of linear transform ers and their recurrent and self-referential extensions. In H. Bouamor, J. Pino, and K. Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 9455–9465, Singapore, 2023. Association for Computational Linguis tics. doi: 10.18653/v1/2023.emnlp-main.588. URL https://aclanthology.org/2023. emnlp-main.588.

[43] L. Jing, C. Gulcehre, J. Peurifoy, Y. Shen, M. Tegmark, M. Soljacic, and Y. Bengio. Gated Orthogonal Recurrent Units: On Learning to Forget. Neural Computation, 31(4):765–783, 2019. ISSN 0899-7667, 1530-888X. doi: 10.1162/neco\_a\_01174. URL https://direct. mit.edu/neco/article/31/4/765-783/8458.

[44] T. Joffrain, T. M. Low, E. S. Quintana-Ortí, R. A. van de Geijn, and F. G. V. Zee. Accumu lating householder transformations, revisited. ACM Trans. Math. Softw., 32:169–179, 2006. URL https://api.semanticscholar.org/CorpusID:15723171.

[45] P. Kacham, V. Mirrokni, and P. Zhong. Polysketchformer: Fast transformers via sketches for polynomial kernels. ArXiv preprint, abs/2310.01655, 2023. URL https://arxiv.org/ abs/2310.01655.

[46] Y. Kang, G. Tran, and H. De Sterck. Fast multipole attention: A divide-and-conquer attention mechanism for long sequences. ArXiv preprint, abs/2310.11960, 2023. URL https:// arxiv.org/abs/2310.11960.

[47] J. Kasai, H. Peng, Y. Zhang, D. Yogatama, G. Ilharco, N. Pappas, Y. Mao, W. Chen, and N. A. Smith. Finetuning pretrained transformers into RNNs. In M.-F. Moens, X. Huang, L. Specia, and S. W.-t. Yih, editors, Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 10630–10643, Online and Punta Cana, Dominican Republic, 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.830. URL https://aclanthology.org/2021.emnlp-main.830.

[48] A. Katharopoulos, A. Vyas, N. Pappas, and F. Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings ofMachine Learning Research, pages 5156–5165. PMLR, 2020. URL http:// proceedings.mlr.press/v119/katharopoulos20a.html.

[49] N. Kitaev, L. Kaiser, and A. Levskaya. Reformer: The efficient transformer. In 8th In ternational Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020. URL https://openreview.net/forum? id=rkgNKkHtvB.

[50] D. Krotov and J. J. Hopfield. Large associative memory problem in neurobiology and machine learning. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview.net/ forum?id=X4y\_10OX-hX.

[51] T. Lei. When attention meets fast recurrence: Training language models with reduced com pute. In M.-F. Moens, X. Huang, L. Specia, and S. W.-t. Yih, editors, Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 7633–7648, Online and Punta Cana, Dominican Republic, 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.602. URL https://aclanthology.org/2021. emnlp-main.602.

[52] T. Lei, R. Tian, J. Bastings, and A. P. Parikh. Simple recurrence improves masked language models. ArXiv preprint, abs/2205.11588, 2022. URL https://arxiv.org/abs/ 2205.11588.

[53] P. S. H. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, S. Riedel, and D. Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6- 12, 2020, virtual, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 6b493230205f780e1bc26945df7481e5-Abstract.html.

[54] B. Liao, X. Wang, L. Zhu, Q. Zhang, and C. Huang. Vig: Linear-complexity visual sequence learning with gated linear attention. ArXiv, abs/2405.18425, 2024. URL https://api. semanticscholar.org/CorpusID:270068351.

[55] O. Lieber, B. Lenz, H. Bata, G. Cohen, J. Osin, I. Dalmedigos, E. Safahi, S. Meirom, Y. Belinkov, S. Shalev-Shwartz, et al. Jamba: A hybrid transformer-mamba language model. ArXiv preprint, abs/2403.19887, 2024. URL https://arxiv.org/abs/2403.19887.

[56] K. C. Lingashetty. Delta learning rule for the active sites model. arXiv preprint arXiv:1007.0417, 2010.

[57] L. D. Lingle. Transformer-vq: Linear-time transformers via vector quantization. ArXiv preprint, abs/2309.16354, 2023. URL https://arxiv.org/abs/2309.16354.

[58] B. Liu, R. Wang, L. Wu, Y. Feng, P. Stone, and Q. Liu. Longhorn: State space models are amortized online learners. ArXiv preprint, abs/2407.14207, 2024. URL https://arxiv. org/abs/2407.14207.

[59] H. Liu, M. Zaharia, and P. Abbeel. Ring Attention with Blockwise Transformers for Near-Infinite Context. ArXiv preprint, abs/2310.01889, 2023. URL https://arxiv.org/abs/ 2310.01889.

[60] C. Lockard, P. Shiralkar, and X. L. Dong. OpenCeres: When open information extraction meets the semi-structured web. In J. Burstein, C. Doran, and T. Solorio, editors, Proceedings ofthe 2019 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 3047– 3056, Minneapolis, Minnesota, 2019. Association for Computational Linguistics. doi: 10. 18653/v1/N19-1309. URL https://aclanthology.org/N19-1309.

[61] X. Ma, C. Zhou, X. Kong, J. He, L. Gui, G. Neubig, J. May, and L. Zettlemoyer. Mega: Moving average equipped gated attention. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/pdf?id=qNLe3iq2El.

[62] X. Ma, X. Yang, W. Xiong, B. Chen, L. Yu, H. Zhang, J. May, L. Zettlemoyer, O. Levy, and C. Zhou. Megalodon: Efficient llm pretraining and inference with unlimited context length. ArXiv preprint, abs/2404.08801, 2024. URL https://arxiv.org/abs/2404.08801.

[63] H. H. Mao. Fine-tuning pre-trained transformers into decaying fast weights. In Y. Goldberg, Z. Kozareva, and Y. Zhang, editors, Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 10236–10242, Abu Dhabi, United Arab Emirates, 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.697. URL https://aclanthology.org/2022.emnlp-main.697.

[64] E. Martin and C. Cundy. Parallelizing linear recurrent neural nets over sequence length. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net, 2018. URL https://openreview.net/forum?id=HyUNwulC-.

[65] E. Martin and C. Cundy. Parallelizing linear recurrent neural nets over sequence length. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net, 2018. URL https://openreview.net/forum?id=HyUNwulC-.

[66] A. Mathiasen, F. Hvilshoj, J. R. Jørgensen, A. Nasery, and D. Mottin. What if neural networks had svds? ArXiv, abs/2009.13977, 2020. URL https://api.semanticscholar.org/ CorpusID:221995651.

[67] A. Mathiasen, F. Hvilshøj, J. R. Jørgensen, A. Nasery, and D. Mottin. Faster orthogonal parameterization with householder matrices. In ICML, Workshop Proceedings, 2020.

[68] R. J. McEliece, E. C. Posner, E. R. Rodemich, and S. S. Venkatesh. The capacity of the hopfield associative memory. IEEE Trans. Inf. Theory, 33:461–482, 1987.

[69] J. Mercat, I. Vasiljevic, S. Keh, K. Arora, A. Dave, A. Gaidon, and T. Kollar. Linearizing large language models. ArXiv preprint, abs/2405.06640, 2024. URL https://arxiv.org/ abs/2405.06640.

[70] W. Merrill, J. Petty, and A. Sabharwal. The Illusion of State in State-Space Models, 2024. URL https://arxiv.org/abs/2404.08819.

[71] Z. Mhammedi, A. D. Hellicar, A. Rahman, and J. Bailey. Efficient orthogonal parametrisation of recurrent neural networks using householder reflections. In D. Precup and Y. W. Teh, editors, Proceedings ofthe 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017, volume 70 of Proceedings ofMachine Learning Research, pages 2401–2409. PMLR, 2017. URL http://proceedings.mlr.press/v70/ mhammedi17a.html.

[72] B. Millidge. Linear Attention as Iterated Hopfield Networks. URL http://www.beren.io/ 2024-03-03-Linear-Attention-as-Iterated-Hopfield-Networks/.

[73] T. Munkhdalai, A. Sordoni, T. Wang, and A. Trischler. Metalearned neural memory. In H. M. Wallach, H. Larochelle, A. Beygelzimer, F. d’Alché-Buc, E. B. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pages 13310–13321, 2019. URL https://proceedings.neurips.cc/paper/ 2019/hash/182bd81ea25270b7d1c2fe8353d17fe6-Abstract.html.

[74] T. Munkhdalai, M. Faruqui, and S. Gopal. Leave no context behind: Efficient infinite context transformers with infini-attention. ArXiv preprint, abs/2404.07143, 2024. URL https:// arxiv.org/abs/2404.07143.

[75] Y. Nahshan, J. Kampeas, and E. Haleva. Linear Log-Normal Attention with Unbiased Concentration, 2023. URL https://arxiv.org/abs/2311.13541.

[76] T. M. Nguyen, V. Suliafu, S. J. Osher, L. Chen, and B. Wang. Fmmformer: Efficient and flexible transformer via decomposed near-field and far-field attention. In M. Ranzato, A. Beygelzimer, Y. N. Dauphin, P. Liang, and J. W. Vaughan, editors, Ad vances in Neural Information Processing Systems 34: Annual Conference on Neural In formation Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 29449–29463, 2021. URL https://proceedings.neurips.cc/paper/2021/hash/ f621585df244e9596dc70a39b579efb1-Abstract.html.

[77] D. Paperno, G. Kruszewski, A. Lazaridou, N. Q. Pham, R. Bernardi, S. Pezzelle, M. Baroni, G. Boleda, and R. Fernández. The LAMBADA dataset: Word prediction requiring a broad discourse context. In K. Erk and N. A. Smith, editors, Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1525–1534, Berlin, Germany, 2016. Association for Computational Linguistics. doi: 10.18653/v1/P16-1144. URL https://aclanthology.org/P16-1144.

[78] J. Park, J. Park, Z. Xiong, N. Lee, J. Cho, S. Oymak, K. Lee, and D. Papailiopoulos. Can mamba learn how to learn? a comparative study on in-context learning tasks. ArXiv preprint, abs/2402.04248, 2024. URL https://arxiv.org/abs/2402.04248.

[79] B. Peng, D. Goldstein, Q. Anthony, A. Albalak, E. Alcaide, S. Biderman, E. Cheah, X. Du, T. Ferdinan, H. Hou, P. Kazienko, K. K. GV, J. Kocon, B. Koptyra, S. Krishna, R. McClel-´ land Jr., N. Muennighoff, F. Obeid, A. Saito, G. Song, H. Tu, S. Wo´zniak, R. Zhang, B. Zhao, Q. Zhao, P. Zhou, J. Zhu, and R.-J. Zhu. Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence, 2024. URL https://arxiv.org/abs/2404.05892.

[80] H. Peng, R. Schwartz, S. Thomson, and N. A. Smith. Rational recurrences. In E. Riloff, D. Chiang, J. Hockenmaier, and J. Tsujii, editors, Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, pages 1203–1214, Brussels, Belgium, 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1152. URL https:// aclanthology.org/D18-1152.

[81] H. Peng, N. Pappas, D. Yogatama, R. Schwartz, N. A. Smith, and L. Kong. Random feature attention. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview.net/ forum?id=QtTKTdVrFBB.

[82] H. Peng, J. Kasai, N. Pappas, D. Yogatama, Z. Wu, L. Kong, R. Schwartz, and N. A. Smith. ABC: Attention with bounded-memory control. In S. Muresan, P. Nakov, and A. Villavicencio, editors, Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7469–7483, Dublin, Ireland, 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.515. URL https:// aclanthology.org/2022.acl-long.515.

[83] M. Poli, S. Massaroli, E. Nguyen, D. Y. Fu, T. Dao, S. Baccus, Y. Bengio, S. Ermon, and C. Ré. Hyena hierarchy: Towards larger convolutional language models. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, editors, International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings ofMachine Learning Research, pages 28043–28078. PMLR, 2023. URL https://proceedings.mlr.press/v202/poli23a.html.

[84] M. Poli, A. W. Thomas, E. Nguyen, P. Ponnusamy, B. Deiseroth, K. Kersting, T. Suzuki, B. Hie, S. Ermon, C. Ré, C. Zhang, and S. Massaroli. Mechanistic Design and Scaling of Hybrid Architectures, 2024. URL https://arxiv.org/abs/2403.17844.

[85] D. Prados and S. Kak. Neural network capacity using delta rule. Electronics Letters, 3(25): 197–199, 1989.

[86] Z. Qin, X. Han, W. Sun, D. Li, L. Kong, N. Barnes, and Y. Zhong. The devil in linear transformer. In Y. Goldberg, Z. Kozareva, and Y. Zhang, editors, Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 7025–7041, Abu Dhabi, United Arab Emirates, 2022. Association for Computational Linguistics. doi: 10. 18653/v1/2022.emnlp-main.473. URL https://aclanthology.org/2022.emnlp-main. 473.

[87] Z. Qin, X. Han, W. Sun, B. He, D. Li, D. Li, Y. Dai, L. Kong, and Y. Zhong. Toeplitz neural network for sequence modeling. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/pdf?id=IxmWsm4xrua.

[88] Z. Qin, D. Li, W. Sun, W. Sun, X. Shen, X. Han, Y. Wei, B. Lv, F. Yuan, X. Luo, et al. Scaling transnormer to 175 billion parameters. ArXiv preprint, abs/2307.14995, 2023. URL https://arxiv.org/abs/2307.14995.

[89] Z. Qin, W. Sun, K. Lu, H. Deng, D. Li, X. Han, Y. Dai, L. Kong, and Y. Zhong. Linearized Relative Positional Encoding, 2023. URL https://arxiv.org/abs/2307.09270.

[90] Z. Qin, S. Yang, and Y. Zhong. Hierarchically gated recurrent neural network for sequence modeling. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, edi tors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper\_files/paper/2023/hash/ 694be3548697e9cc8999d45e8d16fe1e-Abstract-Conference.html.

[91] Z. Qin, W. Sun, D. Li, X. Shen, W. Sun, and Y. Zhong. Lightning attention-2: A free lunch for handling unlimited sequence lengths in large language models. 2024.

[92] Z. Qin, S. Yang, W. Sun, X. Shen, D. Li, W. Sun, and Y. Zhong. HGRN2: Gated Linear RNNs with State Expansion. 2024. URL https://api.semanticscholar.org/ CorpusID:269043328.

[93] P. Rajpurkar, R. Jia, and P. Liang. Know what you don’t know: Unanswerable questions for SQuAD. In I. Gurevych and Y. Miyao, editors, Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 784–789, Melbourne, Australia, 2018. Association for Computational Linguistics. doi: 10.18653/v1/ P18-2124. URL https://aclanthology.org/P18-2124.

[94] H. Ramsauer, B. Schäfl, J. Lehner, P. Seidl, M. Widrich, L. Gruber, M. Holzleitner, T. Adler, D. P. Kreil, M. K. Kopp, G. Klambauer, J. Brandstetter, and S. Hochreiter. Hopfield networks is all you need. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview. net/forum?id=tL89RnzIiCd.

[95] L. Ren, Y. Liu, Y. Lu, Y. Shen, C. Liang, and W. Chen. Samba: Simple hybrid state space models for efficient unlimited context language modeling. ArXiv preprint, abs/2406.07522, 2024. URL https://arxiv.org/abs/2406.07522.

[96] I. Rodkin, Y. Kuratov, A. Bulatov, and M. Burtsev. Associative recurrent memory transformer. ArXiv preprint, abs/2407.04841, 2024. URL https://arxiv.org/abs/2407.04841.

[97] S. Rogers. Adaptive filter theory. Control Engineering Practice, 4:1629–1630, 1996. URL https://api.semanticscholar.org/CorpusID:109523942.

[98] A. Roy, M. Saffar, A. Vaswani, and D. Grangier. Efficient content-based sparse attention with routing transformers. Transactions of the Association for Computational Linguistics, 9:53– 68, 2021. doi: 10.1162/tacl\_a\_00353. URL https://aclanthology.org/2021.tacl-1. 4.

[99] K. Sakaguchi, R. L. Bras, C. Bhagavatula, and Y. Choi. Winogrande: An adversarial winograd schema challenge at scale. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications ofArtificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020, pages 8732–8740. AAAI Press, 2020. URL https://aaai.org/ojs/index.php/AAAI/article/view/6399.

[100] Y. Schiff, C.-H. Kao, A. Gokaslan, T. Dao, A. Gu, and V. Kuleshov. Caduceus: Bidirectional equivariant long-range dna sequence modeling. ArXiv, abs/2403.03234, 2024. URL https://api.semanticscholar.org/CorpusID:268253280.

[101] I. Schlag, K. Irie, and J. Schmidhuber. Linear transformers are secretly fast weight programmers. In M. Meila and T. Zhang, editors, Proceedings ofthe 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 9355–9366. PMLR, 2021. URL http://proceedings. mlr.press/v139/schlag21a.html.

[102] J. Schmidhuber. Learning to control fast-weight memories: An alternative to dynamic recurrent networks. Neural Computation, 4(1):131–139, 1992.

[103] N. Shazeer. Glu variants improve transformer. ArXiv preprint, abs/2002.05202, 2020. URL https://arxiv.org/abs/2002.05202.

[104] Y. Shen, M. Stallone, M. Mishra, G. Zhang, S. Tan, A. Prasad, A. M. Soria, D. D. Cox, and R. Panda. Power scheduler: A batch size and token number agnostic learning rate scheduler. ArXiv preprint, abs/2408.13359, 2024. URL https://arxiv.org/abs/2408.13359.

[105] J. T. H. Smith, A. Warrington, and S. W. Linderman. Simplified state space layers for sequence modeling. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https:// openreview.net/pdf?id=Ai8Hw3AXqks.

[106] J. T. H. Smith, A. Warrington, and S. W. Linderman. Simplified state space layers for sequence modeling. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https:// openreview.net/pdf?id=Ai8Hw3AXqks.

[107] W. Sun, Z. Qin, D. Li, X. Shen, Y. Qiao, and Y. Zhong. Linear attention sequence parallelism. ArXiv preprint, abs/2404.02882, 2024. URL https://arxiv.org/abs/2404.02882.

[108] Y. Sun, L. Dong, S. Huang, S. Ma, Y. Xia, J. Xue, J. Wang, and F. Wei. Retentive network: A successor to transformer for large language models. ArXiv preprint, abs/2307.08621, 2023. URL https://arxiv.org/abs/2307.08621.

[109] Y. Sun, L. Dong, Y. Zhu, S. Huang, W. Wang, S. Ma, Q. Zhang, J. Wang, and F. Wei. You only cache once: Decoder-decoder architectures for language models. ArXiv preprint, abs/2405.05254, 2024. URL https://arxiv.org/abs/2405.05254.

[110] Y. Sun, X. Li, K. Dalal, J. Xu, A. Vikram, G. Zhang, Y. Dubois, X. Chen, X. Wang, O. Koyejo, T. Hashimoto, and C. Guestrin. Learning to (learn at test time): Rnns with expressive hid den states. ArXiv preprint, abs/2407.04620, 2024. URL https://arxiv.org/abs/2407. 04620.

[111] L. Team. The llama 3 herd of models. ArXiv preprint, abs/2407.21783, 2024. URL https:// arxiv.org/abs/2407.21783.

[112] P. Tillet, H. Kung, and D. D. Cox. Triton: an intermediate language and compiler for tiled neural network computations. In Proceedings ofthe 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, MAPL@PLDI 2019, pages 10–19. ACM, 2019. doi: 10.1145/3315508.3329973.

[113] J. M. Tomczak and M. Welling. Improving Variational Auto-Encoders using Householder Flow, 2016. URL https://arxiv.org/abs/1611.09630.

[114] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, et al. Llama: Open and efficient foundation language models. ArXiv preprint, abs/2302.13971, 2023. URL https://arxiv.org/abs/2302.13971.

[115] R. van den Berg, L. Hasenclever, J. M. Tomczak, and M. Welling. Sylvester normalizing flows for variational inference. In A. Globerson and R. Silva, editors, Proceedings of the Thirty-Fourth Conference on Uncertainty in Artificial Intelligence, UAI 2018, Monterey, California, USA, August 6-10, 2018, pages 393–402. AUAI Press, 2018. URL http://auai. org/uai2018/proceedings/papers/156.pdf.

[116] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention is all you need. In I. Guyon, U. von Luxburg, S. Bengio, H. M. Wallach, R. Fergus, S. V. N. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 5998–6008, 2017. URL https://proceedings.neurips.cc/paper/2017/hash/ 3f5ee243547dee91fbd053c1c4a845aa-Abstract.html.

[117] J. von Oswald, E. Niklasson, M. Schlegel, S. Kobayashi, N. Zucchet, N. Scherrer, N. Miller, M. Sandler, B. A. y Arcas, M. Vladymyrov, R. Pascanu, and J. Sacramento. Uncovering mesaoptimization algorithms in transformers. CoRR, abs/2309.05858, 2023. URL https://doi. org/10.48550/arXiv.2309.05858.

[118] E. Vorontsov, C. Trabelsi, S. Kadoury, and C. Pal. On orthogonality and learning recurrent networks with long term dependencies. In D. Precup and Y. W. Teh, editors, Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia,

6-11 August 2017, volume 70 of Proceedings of Machine Learning Research, pages 3570– 3578. PMLR, 2017. URL http://proceedings.mlr.press/v70/vorontsov17a.html.

[119] C. X. Wang, O. Tsepa, J. Ma, and B. Wang. Graph-mamba: Towards long-range graph sequence modeling with selective state spaces. ArXiv, abs/2402.00789, 2024. URL https:// api.semanticscholar.org/CorpusID:267364853.

[120] J. Wang, J. N. Yan, A. Gu, and A. Rush. Pretraining without attention. In H. Bouamor, J. Pino, and K. Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, pages 58–69, Singapore, 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp.5. URL https://aclanthology.org/2023. findings-emnlp.5.

[121] B. Widrow, M. E. Hoff, et al. Adaptive switching circuits. In IRE WESCON convention record, volume 4, pages 96–104. New York, 1960.

[122] J. N. Yan, J. Gu, and A. M. Rush. Diffusion models without attention. 2023.

[123] S. Yang and Y. Zhang. FLA: A Triton-Based Library for Hardware-Efficient Implementations of Linear Attention Mechanism, 2024. URL https://github.com/sustcsonglin/ flash-linear-attention. original-date: 2023-12-20T06:50:18Z.

[124] S. Yang, B. Wang, Y. Shen, R. Panda, and Y. Kim. Gated linear attention transformers with hardware-efficient training. ArXiv preprint, abs/2312.06635, 2023. URL https://arxiv. org/abs/2312.06635.

[125] S. Yang, J. Kautz, and A. Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule, 2024. URL https://arxiv.org/abs/2412.06464.

[126] M. Zaheer, G. Guruganesh, K. A. Dubey, J. Ainslie, C. Alberti, S. Ontañón, P. Pham, A. Ravula, Q. Wang, L. Yang, and A. Ahmed. Big bird: Transformers for longer sequences. In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ c8512d142a2d849725f31a9a7a361ab9-Abstract.html.

[127] R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi. HellaSwag: Can a machine really finish your sentence? In A. Korhonen, D. Traum, and L. Màrquez, editors, Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4791– 4800, Florence, Italy, 2019. Association for Computational Linguistics. doi: 10.18653/v1/ P19-1472. URL https://aclanthology.org/P19-1472.

[128] J. Zhang, Q. Lei, and I. S. Dhillon. Stabilizing gradients for deep neural networks via efficient SVD parameterization. In J. G. Dy and A. Krause, editors, Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018, volume 80 of Proceedings of Machine Learning Research, pages 5801– 5809. PMLR, 2018. URL http://proceedings.mlr.press/v80/zhang18g.html.

[129] J. Zhang, S. Jiang, J. Feng, L. Zheng, and L. Kong. Linear Attention via Orthogonal Memory, 2023. URL https://arxiv.org/abs/2312.11135.

[130] Q. Zhang, D. Ram, C. Hawkins, S. Zha, and T. Zhao. Efficient long-range transformers: You need to attend more, but not necessarily at every layer. In H. Bouamor, J. Pino, and K. Bali, ed itors, Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pages 2775– 2786, Singapore, 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023. findings-emnlp.183. URL https://aclanthology.org/2023.findings-emnlp.183.

[131] Y. Zhang, S. Yang, R. Zhu, Y. Zhang, L. Cui, Y. Wang, B. Wang, F. Shi, B. Wang, W. Bi, P. Zhou, and G. Fu. Gated slot attention for efficient linear-time sequence modeling. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

[132] L. Zhu, Z. Huang, B. Liao, J. H. Liew, H. Yan, J. Feng, and X. Wang. Dig: Scalable and efficient diffusion models with gated linear attention. ArXiv, abs/2405.18428, 2024. URL https://api.semanticscholar.org/CorpusID:270068366.

[133] L. Zhu, B. Liao, Q. Zhang, X. Wang, W. Liu, and X. Wang. Vision mamba: Efficient visual representation learning with bidirectional state space model. ArXiv, abs/2401.09417, 2024. URL https://api.semanticscholar.org/CorpusID:267028142.

[134] I. Zimerman, A. Ali, and L. Wolf. A unified implicit attention formulation for gated-linear recurrent sequence models. ArXiv preprint, abs/2405.16504, 2024. URL https://arxiv. org/abs/2405.16504.

## A Experiments Continued

## A.1 Hyperparameters

We used 8 H100 GPUs for 340M and 1.3B language modeling experiments. Each model uses AdamW for optimization, with a peak learning rate of $3 \times 1 0 ^ { - 4 }$ . The 340M models are trained using 15 billion tokens and a batch size of 0.5M tokens, while the 1.3B models are trained with 100 billion tokens and a batch size of 2M tokens. We use a cosine learning rate schedule, starting with a warm-up phase of 0.5 billion tokens for the 340M models and 1 billion tokens for the 1.3B models. Both configurations have initial and final learning rates set at $3 \times 1 0 ^ { - 5 }$ . We apply a weight decay of 0.01 and use gradient clipping at a maximum of 1.0. The head dimension of DeltaNet is set to 128, and the kernel size for convolution layers is set at 4.

## A.2 Synthetic tasks

We additional conduct experiments on RegBench [2], a synthetic data set designed to assess the in-context language learning capability of different model architectures. Each input sequence in this benchmark consists of 10 to 20 strings drawn from a distinct language defined by a probabilistic finite automaton (PFA), so that a model needs to infer the underlying language from the context on the fly. During testing, a model is evaluated on predicting the next token of testing sequences generated from held-out PFAs. Here again we find that DeltaNet performs strongly compared to baselines, as shown in Figure 7.

![](images/a1d5d763a644c0a62844d9e618859200beb4c5af883000de5bd1023c155d6d3a.jpg)  
Figure 7: Accuracy (%) on RegBench.

## B Method Continued

## B.1 WY representation derivation

To reduce notational clutter, we discuss only the first chunk here.

We first show $\begin{array} { r } { \mathbf { P } _ { n } = \mathbf { I } - \sum _ { t = 1 } ^ { n } w _ { t } \pmb { k } _ { t } ^ { \top } } \end{array}$ by induction,

$$
\begin{array}{l} \mathbf {P} _ {n} = \prod_ {t = 1} ^ {n} (\mathbf {I} - \beta_ {t} \boldsymbol {k} _ {t} \boldsymbol {k} _ {t} ^ {\intercal}) \\ \quad = \mathbf {P} _ {n - 1} (\mathbf {I} - \beta_ {n} \boldsymbol {k} _ {n} \boldsymbol {k} _ {n} ^ {\intercal}) \\ \quad = (\mathbf {I} - \sum_ {t = 1} ^ {n - 1} \boldsymbol {w} _ {t} \boldsymbol {k} _ {t} ^ {\intercal}) (\mathbf {I} - \beta_ {n} \boldsymbol {k} _ {n} \boldsymbol {k} _ {n} ^ {\intercal}) \\ \quad = \mathbf {I} - \sum_ {t = 1} ^ {n - 1} \boldsymbol {w} _ {t} \boldsymbol {k} _ {t} ^ {\intercal} - \beta_ {n} \boldsymbol {k} _ {n} \boldsymbol {k} _ {n} ^ {\intercal} + (\sum_ {t = 1} ^ {n - 1} \boldsymbol {w} _ {t} \boldsymbol {k} _ {t} ^ {\intercal}) \beta_ {n} \boldsymbol {k} _ {n} \boldsymbol {k} _ {n} ^ {\intercal} \\ \quad = \mathbf {I} - \sum_ {t = 1} ^ {n - 1} \boldsymbol {w} _ {t} \boldsymbol {k} _ {t} ^ {\intercal} - \underbrace {\left(\beta_ {n} \boldsymbol {k} _ {n} - \beta_ {n} \sum_ {t = 1} ^ {n - 1} \left(\boldsymbol {w} _ {t} (\boldsymbol {k} _ {t} ^ {\intercal} \boldsymbol {k} _ {n})\right)\right)} _ {\boldsymbol {w} _ {n}} \boldsymbol {k} _ {n} ^ {\intercal} \\ \quad = \mathbf {I} - \sum_ {t = 1} ^ {n} \boldsymbol {w} _ {t} \boldsymbol {k} _ {t} ^ {\intercal} \end{array}
$$

Similarly, we show $\begin{array} { r } { \mathbf { S } _ { n } = \sum _ { t = 1 } ^ { n } { \mathbf { u } } _ { t } { \mathbf { k } } _ { n } ^ { \top } } \end{array}$ by induction,

$$
\begin{array}{l} \mathbf {S} _ {n} = \mathbf {S} _ {n - 1} (\mathbf {I} - \beta_ {n} \boldsymbol {k} _ {n} \boldsymbol {k} _ {n} ^ {\top}) + \beta_ {n} \boldsymbol {v} _ {n} \boldsymbol {k} _ {n} ^ {\top} \\ \qquad = \left(\sum_ {t = 1} ^ {n - 1} \boldsymbol {u} _ {t} \boldsymbol {k} _ {t} ^ {\top}\right) (\mathbf {I} - \beta_ {n} \boldsymbol {k} _ {n} \boldsymbol {k} _ {n} ^ {\top}) + \beta_ {n} \boldsymbol {v} _ {n} \boldsymbol {k} _ {n} ^ {\top} \\ \qquad = \sum_ {t = 1} ^ {n - 1} \boldsymbol {u} _ {t} \boldsymbol {k} _ {t} ^ {\top} - \left(\sum_ {t = 1} ^ {n - 1} \boldsymbol {u} _ {t} \boldsymbol {k} _ {t} ^ {\top}\right) \beta_ {n} \boldsymbol {k} _ {n} \boldsymbol {k} _ {n} ^ {\top} + \beta_ {n} \boldsymbol {v} _ {n} \boldsymbol {k} _ {n} ^ {\top} \\ \qquad = \sum_ {t = 1} ^ {n - 1} \boldsymbol {u} _ {t} \boldsymbol {k} _ {t} ^ {\top} + \underbrace {\left(\beta_ {n} \boldsymbol {v} _ {n} - \beta_ {n} \sum_ {t = 1} ^ {n - 1} \boldsymbol {u} _ {t} \left(\boldsymbol {k} _ {t} ^ {\top} \boldsymbol {k} _ {n}\right)\right)} _ {\boldsymbol {u} _ {n}} \boldsymbol {k} _ {n} ^ {\top} \\ \qquad = \sum_ {t = 1} ^ {n} \boldsymbol {u} _ {t} \boldsymbol {k} _ {n} ^ {\top} \end{array}
$$

## B.2 UT transform derivation

Here we provide a detailed derivation of the UT transform used in §3.2. Specifically, we show that the matrix formulation using the UT transform (Equations 10 and 11 in the main text) is equivalent to the recursive update equations (Equation 7).

We begin with the recursive formulation for computing $\pmb { w } _ { [ t ] } ^ { r }$ and $\pmb { u } _ { [ t ] } ^ { r }$ as given in Equation 7:

$$
\begin{array}{l} \boldsymbol {w} _ {[ t ]} ^ {r} = \beta_ {[ t ]} ^ {r} \left(\boldsymbol {k} _ {[ t ]} ^ {r} - \sum_ {i = 1} ^ {r - 1} \boldsymbol {w} _ {[ t ]} ^ {i} (\boldsymbol {k} _ {[ t ]} ^ {i ^ {\top}} \boldsymbol {k} _ {[ t ]} ^ {r})\right) \\ \boldsymbol {u} _ {[ t ]} ^ {r} = \beta_ {[ t ]} ^ {r} \left(\boldsymbol {v} _ {[ t ]} ^ {r} - \sum_ {i = 1} ^ {r - 1} \boldsymbol {u} _ {[ t ]} ^ {i} (\boldsymbol {k} _ {[ t ]} ^ {i ^ {\top}} \boldsymbol {k} _ {[ t ]} ^ {r})\right) \end{array}
$$

Our goal is to show that these recursive updates can be rewritten in matrix form using the UT transform:

$$
\begin{array}{r l} & {\mathbf {T} _ {[ t ]} = \left(\mathbf {I} + \mathrm{tril} (\mathrm{diag} (\beta_ {[ t ]}) \mathbf {K} _ {[ t ]} \mathbf {K} _ {[ t ]} ^ {\intercal}, - 1)\right) ^ {- 1} \mathrm{diag} \left(\beta_ {[ t ]}\right)} \\ & {\mathbf {W} _ {[ t ]} = \mathbf {T} _ {[ t ]} \mathbf {K} _ {[ t ]}} \\ & {\mathbf {U} _ {[ t ]} = \mathbf {T} _ {[ t ]} \mathbf {V} _ {[ t ]}} \end{array}
$$

First, let us consider the computation of $\pmb { w } _ { [ t ] } ^ { r }$ . For any row r, we can write:

$$
\mathbf {W} _ {[ t ]} [ r,: ] = \beta_ {[ t ]} ^ {r} \mathbf {K} _ {[ t ]} [ r,: ] - \beta_ {[ t ]} ^ {r} \sum_ {i = 1} ^ {r - 1} \mathbf {W} _ {[ t ]} [ i,: ] (\mathbf {K} _ {[ t ]} [ i,: ] \mathbf {K} _ {[ t ]} [ r,: ] ^ {\top})
$$

This system of equations can be written in matrix form. Let us define:

$$
\begin{array}{c} \mathbf {B} _ {[ t ]} = \mathrm{diag} (\beta_ {[ t ]}) \\ \mathbf {L} _ {[ t ]} = \mathrm{tril} (\mathbf {B} _ {[ t ]} \mathbf {K} _ {[ t ]} \mathbf {K} _ {[ t ]} ^ {\intercal}, - 1) \end{array}
$$

Then the system becomes:

$$
\mathbf {W} _ {[ t ]} + \mathbf {L} _ {[ t ]} \mathbf {W} _ {[ t ]} = \mathbf {B} _ {[ t ]} \mathbf {K} _ {[ t ]}
$$

Thus, we can solve for $\mathbf { W } _ { [ t ] }$ :

$$
\mathbf {W} _ {[ t ]} = (\mathbf {I} + \mathbf {L} _ {[ t ]}) ^ {- 1} \mathbf {B} _ {[ t ]} \mathbf {K} _ {[ t ]} = \mathbf {T} _ {[ t ]} \mathbf {K} _ {[ t ]}
$$

where

$$
\mathbf {T} _ {[ t ]} = (\mathbf {I} + \mathbf {L} _ {[ t ]}) ^ {- 1} \mathbf {B} _ {[ t ]}
$$

The same derivation applies for $\mathbf { U } _ { [ t ] }$ by replacing $\mathbf { K } _ { [ t ] }$ with $\mathbf { V } _ { [ t ] }$ in the final step, yielding:

$$
\mathbf {U} _ {[ t ]} = \mathbf {T} _ {[ t ]} \mathbf {V} _ {[ t ]}\tag{12}
$$

## C Related Work Continued

Chunkwise linear attention. Hua et al. [34] first proposed chunkwise form for linear attention; however, they used a hybrid linear and nonlinear attention model similar to Munkhdalai et al. [74]. It is possible to adapt their algorithm to compute the exact output of the pure linear attention, as shown in Sun et al. [108] and Yang et al. [124]. The chunkwise linear attention algorithm has also been independently discovered in several works [108, 45, 19]. Yang et al. [124] and Qin et al. [91] discuss I/O-aware hardware optimization for chunkwise linear attention and Sun et al. [107] make generalization to multi-node distributed training. Inspired by the chunkwise form, we propose a new algorithm for hardware-efficient DeltaNet training, significantly improving the training efficiency and allowing for large-scale experiments.

Hybrid models. Linear recurrent models - including state-space models [32, 105, 31, 120], gated linear RNNs [65, 90], and linear attention mechanisms [48, 124] - have demonstrated remarkable potential as alternatives to traditional softmax attention across diverse domains [122, 132, 54, 133, 119, 100]. Given their complementary strengths, recent research has increasingly focused on developing hybrid architectures that combine linear recurrent layers with local chunk attention [61, 130, 25, 62, 74] or sliding window attention [130, 6, 21, 95] or global attention [51, 52, 35, 26, 55, 78, 109]. Poli et al. [84] systematically study the scaling law of hybrid models. We similarly show that combining DeltaNet with classic attention is an effective strategy.

Householder matrices. Householder matrices, known for preserving norms, are a type of orthogonal matrix extensively used in machine learning [67, 71, 128, 113, 89, 115]. These matrices allow for efficient computation of inverses and their Jacobian determinant of one, making them particularly suitable for applications in normalizing flows [67, 115]. Notably, Mathiasen et al. [67] and Mathiasen et al. [66] developed a chunkwise fast algorithm for computing the cumulative product of Householder matrices for normalizing flows, leveraging the WY representation. Our approach, while sharing the same high-level concept, tackles a different problem and is arguably more general.

There has also been significant interest in using orthogonal matrices to parameterize the transition matrices of RNNs [71, 43, 118, 33] for mitigating vanishing gradients. Mhammedi et al. [71] use the WY representation to reduce the memory footprint when training nonlinear RNNs with Householder transition matrices.

## D Pseudo code

```python
def chunk_delta_rule_forward(Q, K, V, beta, C):
    ' '
    Q/K/V: query, key, value of shape [L, d]
    beta: beta of shape [L]
    C: chunk size
    ' '
    # L: sequence length, d: head dimension
    L, d = Q.shape
    # chunking
    Q, K, V = map(lambda x: x.reshape(-1, C, d), [Q, K, V])
    beta = beta.reshape(-1, C)
    K_beta = K * beta.unsqueeze(-1)
    V_beta = V * beta.unsqueeze(-1)

    # compute eq. 10 with vectorized forward substitution for fast inverse
    T = -(K_beta @ K.t()).tril(-1)
    for i in range(1, C):
    T[i, :i] = T[i, :i] + (T[i, :, None] * T[:, :i]).sum(-2)
    T += torch.eye(C)
    # compute Eq. 11
    W = T @ K_beta
    U = T @ V_beta
    # chunkwise parallel. Eq. 8-9
    S = torch.zeros(d, d)
    O = torch.empty_like(V)
    for i in range(L//C):
    q_i, k_i, w_i = Q[i], K[i], W[i]
    u_i = U[i] - w_i @ S
    o_inter = q_i @ S
    A_i = (q_i @ k_i.t()).tril()
    o_intra = A_i @ u_i
    S += k_i.t() @ u_i
    O[i] = o_intra + o_inter
    return 0.reshape(L, d)
```

Figure 8: Pytorch-like code snippet of the forward pass of our chunkwise algorithm for training DeltaNet. We omit the dimensions of batch size and number of heads for clarity.