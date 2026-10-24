# Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing

Zihang Dai<sup>∗12</sup>, Guokun Lai<sup>∗1</sup>, Yiming Yang<sup>1</sup>, Quoc V. Le<sup>2</sup> <sup>1</sup>Carnegie Mellon University, <sup>2</sup>Google AI Brain Team {dzihang,guokun,yiming}@cs.cmu.edu, qvl@google.com

## Abstract

With the success of language pretraining, it is highly desirable to develop more efficient architectures of good scalability that can exploit the abundant unlabeled data at a lower cost. To improve the efficiency, we examine the much-overlooked redundancy in maintaining a full-length token-level presentation, especially for tasks that only require a single-vector presentation of the sequence. With this intuition, we propose Funnel-Transformer which gradually compresses the sequence of hidden states to a shorter one and hence reduces the computation cost. More importantly, by re-investing the saved FLOPs from length reduction in constructing a deeper or wider model, we further improve the model capacity. In addition, to perform token-level predictions as required by common pretraining objectives, Funnel-Transformer is able to recover a deep representation for each token from the reduced hidden sequence via a decoder. Empirically, with comparable or fewer FLOPs, Funnel-Transformer outperforms the standard Transformer on a wide variety of sequence-level prediction tasks, including text classification, language understanding, and reading comprehension.

## 1 Introduction

With the recent success of unsupervised language pretraining [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], the power of neural self-attention models (a.k.a. Transformer) [13] has been pushed to a new level, leading to dramatic advancements in machine learning and natural language processing (NLP). More importantly, it has been observed that with more FLOPs invested in longer pretraining and/or larger models, the performance of pretrained Transformer models consistently improve. However, it is extremely expensive to pretrain or even just to finetune the state-of-the-art self-attention models, as they require much more FLOPs and memory resources compared to traditional models in NLP. This largely limits their applications and success in more fields.

Given this challenge, there has been an increasing amount of efforts to reduce the costs of pretraining and finetuning self-attention models. From the perspective of post-pretraining processing, typical approaches include distillation, pruning and quantization of various kinds, which try to derive a lighter model from an well-pretrained model by taking advantage of the richer signals in the larger model or learning to remove less important operations. Another line of research aims at designing an architecture that not only has a lower resource-to-performance ratio (more efficient) but also scales as well as the Transformer, at least in certain domains. Most of such methods build upon the Transformer backbone and focus on redesigning its building blocks. Representative solutions include searching for better micro operation or macro module designs [14, 15], replacing the full pairwise attention with local operations such as convolution [16] and dynamic convolution [17], and optimizing the hidden size combinations for existing blocks [18].

Across the wide variety of ideas mentioned above, a common strategy is to identify redundant operations or representations and replace them with more efficient ones. Inspired by this line of thinking, in this work, we will be focusing on the potential redundancy induced by always maintaining afull-length sequence of hidden representations across all layers in Transformer. Intuitively, for many sequence-level NLP tasks such as text classification and ranking, the most common use case is to extract a single vector from the entire sequence, which does not necessarily preserve all information down to the token-level granularity. Hence, for such tasks, the full-length sequence of hidden states may contain significant redundancy. This is analogous to the case of image recognition, where the convolution neural network gradually reduces the spatial resolution/size of feature maps as the neural network goes deeper. In addition, linguistic prior also encourages gradually merging nearby tokens (words) into larger semantic units (phrases), which naturally leads to a shorter sequence of representations.

Concretely, we propose to gradually reduce the sequential resolution (i.e. length) of the hidden representation in self-attention models. Immediately, the reduction in sequence length can lead to significant savings in both FLOPs and memory. More importantly, the saved computational resource can be directly re-invested in constructing a deeper (or wider) model to boost the model capacity without additional computational burden. In addition, to address the challenge that common pretraining objectives such as masked language modeling (MLM) [2] require separate representations for each token, we design a simple strategy to decode a full-length sequence of deep representations from the hidden state of reduced length. As a result, the proposed model can be directly trained without modifying the pretraining objectives, as well as adopted for downstream tasks that require token-level representations.

Empirically, with comparable or even fewer FLOPs, by trading sequential resolution for depth, our proposed model achieves an improved performance over the standard Transformer on a wide variety of sequence-level prediction tasks, including text classification, language understanding, and reading comprehension.

## 2 Method

## 2.1 Background

Transformer Architecture The Transformer architecture [13] is a highly modularized neural network, where each Transformer layer consists of two sub-modules, namely the multi-head selfattention (S-Attn) and position-wise feed-forward network (P-FFN). Both sub-modules are wrapped by a residual connection and layer normalization. Schematically, given a length T sequence of hidden states $\mathbf { h } = [ h _ { 1 } , \ldots , h _ { T } ]$ , the computation of a single Transformer layer can be expressed as

$$
\mathbf {h} \leftarrow \text { LayerNorm } (\mathbf {h} + \text { S - Attn } (Q = \mathbf {h}, \text { KV } = \mathbf {h})),\tag{1}
$$

$$
h _ {i} \leftarrow \operatorname{LayerNorm} (h _ {i} + \mathrm{P-FFN} (h _ {i})), \quad \forall i = 1, \dots , T.\tag{2}
$$

Pretraining Objectives The most commonly used pretraining objective is the masked language modeling (MLM) proposed by BERT [2]. For a length-T natural language sequence x sample from a large unlabeled set D, the MLM objective first constructs a corrupted sequence xˆ by randomly replacing 15% of the tokens of x with a special token [mask] and then trains a Transformer model [2] to reconstruct the original x based on xˆ, i.e.,

$$
\max _ {\boldsymbol {\theta}} \mathcal {J} _ {\mathrm{MLM}} (\boldsymbol {\theta}) = \mathbb {E} _ {\mathbf {x} \sim \mathcal {D}} \mathbb {E} _ {\mathcal {I}} \sum_ {i \in \mathcal {I}} \log P _ {\boldsymbol {\theta}} (x _ {i} \mid \hat {\mathbf {x}} _ {\mathcal {I}}) = \mathbb {E} _ {\mathbf {x} \sim \mathcal {D}} \mathbb {E} _ {\mathcal {I}} \sum_ {i \in \mathcal {I}} \log \frac {\exp (e (x _ {i}) ^ {\top} h _ {i} (\hat {\mathbf {x}} _ {\mathcal {I}}))}{\sum_ {x ^ {\prime}} \exp (e (x ^ {\prime}) ^ {\top} h _ {i} (\hat {\mathbf {x}} _ {\mathcal {I}}))},
$$

where I is the positions of masked tokens, the subscript in $\hat { \mathbf { x } } _ { \mathcal { T } }$ emphasizes its dependence on $\tau , e ( x )$ denotes the embedding of the token x, and $h _ { i } ( \hat { \mathbf { x } } _ { \mathcal { T } } )$ the last-layer hidden state at position i produced by the Transformer model. After pretraining, the entire model is finetuned in downstream tasks.

To show the generality of our proposed model, we also experiment with another pretraining objective ELECTRA [5]. Different from MLM, ELECTRA relies a pair of jointly trained generator and discriminator. Specifically, the generator usually has a smaller size (1/4 of that of the discriminator) and is directly trained via the MLM objective, i.e., $\operatorname* { m a x } _ { \theta _ { G } } \mathcal { I } _ { \mathrm { M L M } } ( \theta _ { G } )$ Then, for each masked position, a token is sampled from the reconstruction distribution of the generator to replace the [mask] token and form a new sequence x˜, i.e., if $i \in \mathcal { T } , \tilde { x } _ { i } \sim { P _ { \theta _ { G } } ( x _ { i } \mid \hat { \mathbf { x } } _ { \mathcal { T } } ) }$ ) else $\tilde { x } _ { i } = x _ { i }$ . Given the new sequence x˜, the discriminator is then trained to distinguish whether each token in x˜ is real (same as x) or fake (different from x) via binary classification. After pretraining, only the discriminator will be used during finetuning and the generator is simply discarded.

Discussion Note that both pretraining objectives introduced above require the ability to produce a hidden state for each input token, i.e., $h _ { i } ( \hat { \mathbf { x } } _ { \mathcal { T } } )$ and $h _ { i } ( \tilde { \mathbf { x } } )$ . Due to this requirement, it seems natural to keep a full sequence of hidden states. However, in contrast, many sequence-level downstream tasks like classification or ranking only need a single-vector summary of the entire sequence. Fundamentally, this suggests that some kind of compression is usually required to remove the unnecessary redundancy during finetuning. This observation immediately leads to the following two questions:

• Can we design a general model that is equally expressive but more efficient by compressing the full sequence of hidden states into a more compact form?

• With the compressed representations, how can the model retain the ability to produce token-level representations for pretraining?

To answer these two questions, we next present our proposed architecture.

## 2.2 Proposed Architecture

![](images/de926e657d0832983a403240e84eff9cea82043a922f00545e3c561a08af4d40.jpg)  
Figure 1: High-level visualization of the proposed Funnel-Transformer.

To inherit the high capacity and optimization advantages of the Transformer architecture, the proposed model keeps the same overall skeleton of interleaved S-Attn and P-FFN sub-modules wrapped by residual connection and layer normalization. But differently, to achieve representation compression and computation reduction, our model employs an encoder that gradually reduces the sequence length of the hidden states as the layer gets deeper. In addition, for tasks involving per-token predictions like pretraining, a simple decoder is used to reconstruct a full sequence of token-level representations from the compressed encoder output.

Encoder As illustrated in the left part of Fig. 1, the encoder consists of several blocks of consecutive Transformer layers. Within each block, the sequence length of the hidden states always remains the same. But when going from a lower-level block to a higher-level block, the length of the hidden sequence is reduced by performing certain type of pooling along the sequence dimension, i.e.,

$$
\mathbf {h} ^ {\prime} \leftarrow \operatorname{Pooling} (\mathbf {h}),\tag{3}
$$

where h $\in \mathbb { R } ^ { T \times D }$ and $\mathbf { h } ^ { \prime } \in \mathbb { R } ^ { T ^ { \prime } \times D }$ for some $T ^ { \prime } < T$ . Importantly, instead of directly feeding the pooled sequence h into the first S-Attn layer of the new block, we only use pooled sequence $\mathbf { \bar { h } } ^ { \prime }$ to construct the query vector (and the residual signal) of the self-attention, while the unpooled sequence h serves that role of key and value vectors, i.e.

$$
\mathbf {h} \leftarrow \text { LayerNorm } \big (\mathbf {h} ^ {\prime} + \text { S - Attn } \big (Q = \mathbf {h} ^ {\prime}, \text { KV } = \mathbf {h} \big) \big).\tag{4}
$$

Note that the output sequence of this special S-Attn module has the same length as the pooled sequence h<sup>0</sup>. To understand the advantage of this particular design, it is helpful to compare the proposed “pool-query-only” variant with the naive alternative of using $\mathbf { h } ^ { \prime }$ for both the query and key-value vectors, i.e., ${ \displaystyle { \bf S } { \bf - } \dot { \bf A t t n } \big ( } { \bf Q } = { \bf h } ^ { \prime } , { \bf K V } = { \bf h } ^ { \prime } \big ) $

• Under the naive approach, the compression is solely controlled by the pooling operation, which is finished before the attention module. Hence, relatively simple pooling methods such as average/mean pooling won’t be able to achieve good compression.

• Under the pool-query-only variant, the compression depends on not only how the pooling is performed, but also how the self-attention weighted sums the unpooled sequence to form each pooled vector. Effectively, the particular attention here can be seen as a type of linear compression that combines T bases into a smaller number of $T ^ { \prime } ^ { \mathrm { ~  ~ } }$ “compressed bases”. Therefore, with minimum computational overhead, this variant makes compression operation more expressive.

With this particular pool-query-only design in place, we find the simplest strided mean pooling applied to each sliding window of the sequence work very well in practice. For simplicity, we only experiment with stride 2 and window size 2 in this work. Hence, the pooling operation will reduce the sequence by half and each pooled hidden state corresponds to a window of 2 unpooled hidden vectors. Intuitively, this type of pooling roughly follows the linguistic prior that nearby tokens could be gradually merged (or compressed) into a larger semantic component. Once the sequence length is halved after the pooling and pool-query-only attention, the rest of the encoder computation simply follows the standard updates in Eqn. (2) and (1).

Finally, as an extra implementation detail, recall that a particular design in language pretraining is to add a special token [cls] to the beginning of the original input sequence, and use the last-layer hidden state corresponding to [cls] $( \mathrm { i . e . , } h _ { 1 } )$ as the representation of the sequence. To prevent the pooling from destroying this special structure, we first separate the [cls] hidden state and the rest of hidden states and only apply the pooling to the rest of hidden states. For some practical implementation issues and an efficient solution, we refer readers to Appendix A.1.

Decoder In order to recover a full sequence of hidden states from the encoder output of reduced length, a natural idea would be performing some kind of up-sampling. For instance, in image generation or super-resolution, deconvolution (transposed convolution) or parameter-free resizing with bilinear interpolation are often used to increase the spatial resolution of the feature map. Hence, we can simply adapt these ideas from 2D processing to our 1D case and apply proper up-sampling to the encoder output.

However, instead of performing multiple up-samplings with small expansion rate (e.g. increasing the sequence length by $2 \mathbf { x }$ each time) as in image domain, we here choose to employ a single up-sampling with a large expansion rate, as shown on the right part of Fig. 1. Specifically, given the output sequence $\mathbf { \bar { h } } ^ { \mathbf { M } }$ of length $T _ { M } = T / 2 ^ { M - 1 }$ from an M-block encoder, we directly up-sample it to a full-length sequence $\mathbf { \bar { h } } ^ { \mathsf { u p } } = \left\lceil h _ { 1 } ^ { \mathsf { u p } } , \cdot \cdot \cdot , h _ { T } ^ { \mathsf { u p } } \right\rceil$ by repeating each hidden vector $2 ^ { M - 1 }$ times:

$$
\forall i = 1, \dots , T, h _ {i} ^ {\mathrm{up}} = h _ {i / / 2 ^ {M - 1}} ^ {M},\tag{5}
$$

where $\cdot / / \cdot$ denotes floor division. However, note that every $2 ^ { M - 1 }$ consecutive vectors in $\mathbf { h } ^ { \mathrm { u p } }$ are exactly the same and hence do not contain detailed token-level information. Hence, we further extract the last-layer hidden states from the first block of the encoder ${ \bf h } ^ { 1 }$ , which still has the full length T and contains the uncompressed token-level information. Then, the lower-level representation ${ \bf h } ^ { 1 }$ and up-sampled higher-level representation $\mathbf { h } ^ { \mathrm { u p } }$ are added together to form a deep token-level representation $\mathbf { g } = \mathbf { h } ^ { 1 } + \mathbf { h } ^ { \mathsf { u p } }$ . Effectively, this forms a residual/skip connection that enables detailed token information and potentially easier optimization. In addition, we stack a few more Transformer layers upon $\mathbf { g }$ to achieve a better deep fusion of the low-level and high-level features. In this work, we always use 2 Transformer layers in decoder.

It is important to emphasize that the decoder is only used ifthe task requires token-level prediction, such as in standard pretraining or sequence labeling. For tasks that only requires a single vectorial representation of the sequence like classification, the decoder is discarded after pretraining and only the encoder is finetuned. Finally, to emphasize the filtering/compression property of the encoder as well as its shape, we name the proposed model Funnel-Transformer (F-TFM).

## 2.3 Complexity & Capacity Analysis

With the architecture design specified, we now analyze how the sequence compression affects the complexity and capacity of the proposed model, especially compared to the standard Transformer. Firstly, for a Transformer layer with an S-Attn and a P-FFN module of hidden size D, the complexity of processing a length-T sequence is $O ( T ^ { 2 } D + T D ^ { 2 } )$ .<sup>2</sup> Hence, every time the sequence length is reduced by half in the encoder, we enjoy a super-linear (more than half) complexity drop. In practice, as the $O ( T D ^ { 2 } )$ term has a large constant, a near-linear speedup is observed more often. The super-linear effect is more detectable when the sequence length is relatively long like in pretraining.

Therefore, given the same FLOPs, we can at least trade a full-length layer in the 1st block for $2 ^ { m - 1 }$ layers in the m-th block, which provides an economical way to increase the depth of network.

On the other hand, the capacity of a compressed-length layer is clearly upper-bounded by that of a normal full-length layer. In most cases where the compression is lossy, reducing the sequence length will inevitably lead to capacity drop. The good news is that the capacity drop of a single layer could be well compensated by re-investing the saved FLOPs in stacking more cheaper layers of reduced length or increasing the width of the model.

As a concrete example, for a Transformer of $\mathbf { B E R T _ { B a s e } }$ size, i.e., 12 layers of hidden size 768 (L12H768), we may construct a Funnel-Transformer of 3 blocks where each block has 6 layers of hidden size 768 (B6-6-6H768). Despite having 18 layers in total, when finetuned for classification, the FLOPs of the B6-6-6H768 architecture only corresponds to at most $6 + 6 / 2 + 6 / 4 = 1 0 . 5$ full-length layers, clearly fewer than that of L12H768. More importantly, as we will show in the experiments, B6-6-6H768 significantly outperforms L12H768. While intuitive, how to construct an optimal block layout given this depth-length trade-off remains an open challenge. For this work, we only consider relatively regular layout and leave more systematic studies for future work.

Finally, notice that trading sequential resolution for depth or width has a side effect of increasing the total number of parameters. For instance, B6-6-6H768 has 1.5x Transformer parameters compared to L12H768. In practice, more parameters may increase communication cost in distributed training as well as the memory consumption and memory access time. A simple remedy is to perform certain parameter sharing, as used in ALBERT, to recover the same parameter count. Taking B6-6-6H768 as an example, one may tie the parameters for every two layers in the 2nd and 3rd blocks, denoted as B6-3x2-3x2H768, which gives back the same number of parameters to L12H768. However, parameter sharing could result in performance loss. Fundamentally, this brings us another trade-off between the gain (capacity) and cost (memory and communication cost) of using more parameters, which can be highly device dependent.

## 3 Related Work

As far as we know, no previous work achieves performance gain via compressing the sequence length of the hidden states under language pretraining. Meanwhile, our proposed model is quite similar to the bottom-up model proposed by a contemporary work [19] for causal language modeling. The key differences include the pool-query-only design for down-sampling, how the up-sampling is performed, and our relative attention parameterization. Another closely related idea is Power-BERT [20], which learns to soft-eliminate word vectors that are less “significant” during finetuning. Hence, for postfinetuning inference, the sequence length can be reduced to achieve acceleration. More generally, our work is also related to previous work on hierarchical recurrent neural networks [21] and Transformer models [22, 23]. Different from these methods, our model does not rely on any pre-defined hierarchy or boundary of semantic meanings and always captures the full-length dependency input with attention.

In contrast, our work draws many inspirations from the computer vision domain. The contracting encoder and expanding decoder framework with residual connections is conceptually similar to the ResUNet [24] for image segmentation. The strided pooling is also widely used to construct modern image recognition networks [25]. Despite the similarities, apart from the obvious difference in data domain and computation modules, our encoder employs a special pool-query-only design to improve the compression, and our decoder only requires a single up-sampling with a large expansion rate.

In addition, a line of research in graph neural networks has tries to gradually reduce the number of nodes in different ways and obtain a single vectorial representation for supervised classification. [26, 27, 28] While these methods could potentially be plugged into our model as alternative compression operations, it remains an open question whether compression techniques developed for supervised graph classification can be extended the large-scale language pretraining.

## 4 Experiment

In this section, we empirically evaluate the proposed F-TFM by first pretraining it and then finetuning it in downstream tasks. Following previous work, for pretraining, we consider two common settings:

• Base scale: Pretraining models for 1M steps with batch size 256 on Wikipedia + Book Corpus. This is the setting used by original BERT [2]. We will rely on this setting to perform fair comparison between F-TFM and the standard Transformer as well as some ablation studies.

• Large scale: Pretraining models for 500K steps with batch size 8K on the five datasets used by XLNet [3] and ELECTRA [5] (Wikipedia + Book Corpus + ClueWeb + Gigaword + Common Crawl). We will compare F-TFM trained at this scale with previous state-of-the-art methods.

For finetuning, we mainly focus on sequence-level tasks that only requires a single vectorial representation of the input sequence, since F-TFM is designed with such a purpose in mind. Specifically, such tasks include the GLUE benchmark for language understanding [29], 7 widely used text (sentiment / topic) classification tasks (IMDB, AD, DBpedia, Yelp-2, Yelp-5, Amazon-2, Amazon-5) [30], and the RACE reading comprehension dataset [31]. In addition, to see how F-TFM performs when token-level prediction is needed, we consider the SQuAD question answering task which requires the model to select a token span from the context paragraph as the answer. For more details of the experiment setting, we refer readers to Appendix B.

Finally, for all models implemented in this work including Transformer baselines in the base-scale comparison section 4.1, we always use the relative positional attention parameterization proposed by Transformer-XL [32] (see Appendix A.2 for some implementation details of Transformer-XL).

## 4.1 Base-scale Results

Firstly, we evaluate how F-TFM performs compared to the standard Transformer under similar amount of computation (i.e., FLOPs). For this purpose, we consider three commonly used model sizes for the standard Transformer, namely large (L24H1024), base (L12H768) and small (L6H768). Then, for each Transformer baseline, we construct F-TFMs of different block layouts and parameters, while ensuring the F-TFMs always have fewer or similar FLOPs. Based on the MLM pretraining objective, the results on GLUE benchmark and text classification are presented in Table 1, where we also include the relative FLOPs and #Params. Here, we can make a few key observations:

<table><tr><td>Model size</td><td>CoLA</td><td>SST-2</td><td>MRPC</td><td>STS-B</td><td>QQP</td><td>MNLI</td><td>QNLI</td><td>RTE</td><td>GLUE-AVG</td></tr><tr><td>L24H1024</td><td>63.2</td><td>94.8</td><td>91.8/88.5</td><td>91.1</td><td>88.7/91.7</td><td>88.7</td><td>94.0</td><td>80.5</td><td>86.6</td></tr><tr><td>B10-10-10</td><td>64.8</td><td>95.0</td><td>92.5/89.5</td><td>90.7</td><td>88.6/91.5</td><td>88.9</td><td>94.0</td><td>81.5</td><td>87.0</td></tr><tr><td>B8-8-8</td><td>63.5</td><td>94.7</td><td>92.2/89.0</td><td>90.7</td><td>88.9/91.7</td><td>88.8</td><td>93.6</td><td>81.2</td><td>86.7</td></tr><tr><td>L12H768</td><td>60.5</td><td>93.0</td><td>92.2/89.0</td><td>89.4</td><td>88.1/91.2</td><td>86.0</td><td>92.2</td><td>73.6</td><td>84.4</td></tr><tr><td>B6-6-6</td><td>62.5</td><td>94.0</td><td>92.2/89.0</td><td>89.5</td><td>88.4/91.4</td><td>87.0</td><td>92.7</td><td>76.5</td><td>85.3</td></tr><tr><td>B6-3x2-3x2</td><td>60.5</td><td>93.6</td><td>92.4/89.2</td><td>89.4</td><td>88.2/91.3</td><td>86.4</td><td>92.5</td><td>75.0</td><td>84.7</td></tr><tr><td>B4-4-4</td><td>59.1</td><td>92.7</td><td>91.8/88.7</td><td>89.1</td><td>88.2/91.3</td><td>85.5</td><td>92.0</td><td>73.2</td><td>83.9</td></tr><tr><td>L6H768</td><td>55.2</td><td>91.5</td><td>91.1/87.8</td><td>88.1</td><td>87.2/90.6</td><td>82.7</td><td>90.0</td><td>64.6</td><td>81.3</td></tr><tr><td>B3-4-4</td><td>59.0</td><td>92.8</td><td>91.8/88.5</td><td>88.5</td><td>87.8/90.9</td><td>84.8</td><td>91.8</td><td>73.2</td><td>83.7</td></tr><tr><td>Model size</td><td>IMDB</td><td>AG</td><td>DBpedia</td><td>Yelp2</td><td>Yelp5</td><td>Amazon2</td><td>Amazon5</td><td>FLOPs</td><td>#Params</td></tr><tr><td>L24H1024</td><td>4.440</td><td>4.987</td><td>0.646</td><td>1.758</td><td>28.73</td><td>2.409</td><td>32.78</td><td>1.00x</td><td>1.00x</td></tr><tr><td>B10-10-10</td><td>4.404</td><td>5.026</td><td>0.617</td><td>1.734</td><td>28.52</td><td>2.400</td><td>32.65</td><td>0.73x</td><td>1.22x</td></tr><tr><td>B8-8-8</td><td>4.552</td><td>5.079</td><td>0.664</td><td>1.713</td><td>28.84</td><td>2.438</td><td>32.87</td><td>0.58x</td><td>1.00x</td></tr><tr><td>L12H768</td><td>5.328</td><td>5.184</td><td>0.663</td><td>2.013</td><td>29.35</td><td>2.571</td><td>33.14</td><td>1.00x</td><td>1.00x</td></tr><tr><td>B6-6-6</td><td>4.908</td><td>5.079</td><td>0.654</td><td>1.939</td><td>29.03</td><td>2.518</td><td>32.91</td><td>0.88x</td><td>1.39x</td></tr><tr><td>B6-3x2-3x2</td><td>5.144</td><td>5.342</td><td>0.649</td><td>1.892</td><td>29.03</td><td>2.570</td><td>33.01</td><td>0.88x</td><td>1.00x</td></tr><tr><td>B4-4-4</td><td>5.348</td><td>5.250</td><td>0.670</td><td>1.979</td><td>29.37</td><td>2.596</td><td>33.16</td><td>0.58x</td><td>1.00x</td></tr><tr><td>L6H768</td><td>6.252</td><td>5.421</td><td>0.697</td><td>2.203</td><td>30.33</td><td>2.801</td><td>33.69</td><td>1.00x</td><td>1.00x</td></tr><tr><td>B3-4-4</td><td>5.520</td><td>5.342</td><td>0.670</td><td>2.042</td><td>29.51</td><td>2.603</td><td>33.16</td><td>1.00x</td><td>1.53x</td></tr></table>

Table 1: MLM pretraining results at the base scale: GLUE dev performances (the higher the better) in the upper panel and text classification error rates (the lower the better) in the lower panel . The FLOPs and #Params both refer to the finetuning setting with only the encoder. The corresponding numbers with the decoder are included in Appendix C.2. The FLOPs is a rough estimation assuming linear complexity w.r.t. the sequence length. The #Params is exact including the embedding matrix.

• Given similar or fewer FLOPs, by trading sequential resolution for more layers, the F-TFM outperforms the standard Transformer in most tasks except STS-B, especially for smaller models.

• When we only compress the sequence length without increasing the depth (and #Params), F-TFM could suffer from some performance loss in certain settings on the GLUE datasets. However, as the model size increases, such performance gaps become smaller or even disappear.

• In addition, we find partial parameter-sharing often harms the performance. Therefore, the practical trade-off should be made according to the actual task and computation device.

To further test generality of F-TFM, we additionally consider ELECTRA for pretraining. The results are summarized in Table 2. Overall, we see a similar trend, though the gain is slightly smaller on the GLUE benchmark. This could be attributed to reusing two key hyper-parameters (discriminator loss coefficient and generator size multiplier) tuned for Transformer to train F-TFMs without any adjustment at all.

<table><tr><td>Model size</td><td>CoLA</td><td>SST-2</td><td>MRPC</td><td>STS-B</td><td>QQP</td><td>MNLI</td><td>QNLI</td><td>RTE</td><td>GLUE-AVG</td></tr><tr><td>L24H1024</td><td>66.5</td><td>94.3</td><td>92.8/90.0</td><td>91.5</td><td>89.6/92.2</td><td>89.4</td><td>94.1</td><td>84.5</td><td>87.8</td></tr><tr><td>B10-10-10</td><td>68.6</td><td>95.0</td><td>93.0/90.0</td><td>91.0</td><td>88.9/91.7</td><td>89.1</td><td>93.6</td><td>84.5</td><td>87.9</td></tr><tr><td>B8-8-8</td><td>66.6</td><td>94.8</td><td>92.6/89.7</td><td>90.7</td><td>88.8/91.7</td><td>89.0</td><td>93.6</td><td>82.1</td><td>87.3</td></tr><tr><td>L12H768</td><td>64.3</td><td>93.1</td><td>92.1/89.2</td><td>90.8</td><td>88.7/91.7</td><td>86.4</td><td>92.1</td><td>75.4</td><td>85.4</td></tr><tr><td>B6-6-6</td><td>64.3</td><td>94.2</td><td>92.8/89.7</td><td>90.1</td><td>88.7/91.6</td><td>87.4</td><td>92.5</td><td>78.3</td><td>86.0</td></tr><tr><td>B6-3x2-3x2</td><td>63.9</td><td>94.2</td><td>93.0/90.2</td><td>89.5</td><td>88.4/91.4</td><td>87.0</td><td>92.2</td><td>77.6</td><td>85.7</td></tr><tr><td>B4-4-4</td><td>62.8</td><td>93.6</td><td>92.5/89.2</td><td>89.2</td><td>88.4/91.3</td><td>86.0</td><td>91.6</td><td>74.3</td><td>84.8</td></tr><tr><td>L6H768</td><td>62.1</td><td>91.1</td><td>90.8/86.8</td><td>88.9</td><td>88.2/91.3</td><td>83.9</td><td>89.7</td><td>66.7</td><td>82.6</td></tr><tr><td>B3-4-4</td><td>59.0</td><td>93.1</td><td>90.8/87.5</td><td>88.7</td><td>88.1/91.0</td><td>85.8</td><td>91.1</td><td>72.5</td><td>83.6</td></tr><tr><td>Model size</td><td>IMDB</td><td>AG</td><td>DBpedia</td><td>Yelp2</td><td>Yelp5</td><td>Amazon2</td><td>Amazon5</td><td>FLOPs</td><td>#Params</td></tr><tr><td>L24H1024</td><td>4.724</td><td>5.053</td><td>0.653</td><td>1.874</td><td>28.84</td><td>2.425</td><td>32.85</td><td>1.00x</td><td>1.00x</td></tr><tr><td>B10-10-10</td><td>4.324</td><td>5.250</td><td>0.639</td><td>1.789</td><td>28.68</td><td>2.419</td><td>32.72</td><td>0.73x</td><td>1.22x</td></tr><tr><td>B8-8-8</td><td>4.364</td><td>5.408</td><td>0.651</td><td>1.729</td><td>28.76</td><td>2.447</td><td>32.85</td><td>0.58x</td><td>1.00x</td></tr><tr><td>L12H768</td><td>5.248</td><td>5.355</td><td>0.657</td><td>1.953</td><td>29.24</td><td>2.596</td><td>33.04</td><td>1.00x</td><td>1.00x</td></tr><tr><td>B6-6-6</td><td>4.792</td><td>5.237</td><td>0.650</td><td>1.850</td><td>28.73</td><td>2.499</td><td>32.79</td><td>0.88x</td><td>1.39x</td></tr><tr><td>B6-3x2-3x2</td><td>4.924</td><td>5.342</td><td>0.671</td><td>1.913</td><td>29.00</td><td>2.523</td><td>32.85</td><td>0.88x</td><td>1.00x</td></tr><tr><td>B4-4-4</td><td>5.152</td><td>5.382</td><td>0.659</td><td>2.032</td><td>29.33</td><td>2.566</td><td>33.03</td><td>0.58x</td><td>1.00x</td></tr><tr><td>L6H768</td><td>6.220</td><td>5.395</td><td>0.674</td><td>2.287</td><td>30.16</td><td>2.759</td><td>33.57</td><td>1.00x</td><td>1.00x</td></tr><tr><td>B3-4-4</td><td>5.396</td><td>5.342</td><td>0.653</td><td>2.000</td><td>29.60</td><td>2.591</td><td>33.09</td><td>1.00x</td><td>1.53x</td></tr></table>

Table 2: ELECTRA pretraining results at the base scale.

Running Time Comparison While FLOPs count offers a general idea of the model speed, it still differs from the actual running time, especially when other overhead exists. Hence, for completeness, we show the speedup provided by the F-TFM in terms of actual running time in Appendix C.2. We also compare the actual memory footprint of F-TFM and TFM in Appendix C.2.

## 4.2 Large-scale Results

Given the encouraging results of F-TFM at base-scale, we next consider training F-TFM under the large-scale setting and compare it with previous models pretrained in similar settings. Due to the slightly better performance of ELECTRA over MLM, we will use the ELECTRA objective for all large-scale experiments. Given the pretrained F-TFM of different sizes, we first compare the finetuning performance on the GLUE benchmark in Table 3. Similar to the base-scale results, with fewer or comparable FLOPs, F-TFM outperforms the corresponding baselines in the majority of tasks, suggesting the good scalability of F-TFM. We also test the models on the 7 text classification tasks. But due to the page constraint, we refer readers to Appendix C.1.

Next, we consider the RACE dataset, which is quite different from the GLUE benchmark. At the core, RACE is a multiple-choice reading comprehension task requiring complex reasoning, which though, can be formulated as classifying the correct choice. Also, paragraphs in RACE are much longer. To F-TFM, this presents both a challenge, as it requires detailed reasoning, and an opportunity to compress long paragraph. As we can see in Table 4, F-TFM achieves better performances compared to all previous models. In particular, within the base model group, the gain is very significant. It shows that F-TFM can also excel for sequence-level task that involves long text and reasoning.

Finally, although F-TFM is mainly designed for tasks that only require a sequence-level representation, it is possible to apply F-TFM to token-level tasks by additionally finetuning the decoder. To test this ability, we finetune F-TFM on the SQuAD datasets and compare it with previous models in Table 5. While F-TFM outperforms previous models in the base group by a large margin, in the large model group, the F-TFM with about 83% FLOPs (B10-10-10) still falls behind the standard Transformer that always maintains a full-length token-level representations. This suggests sequential compression could harm the performance when detailed token-level information is critical. On the other hand, compared to the results on SQuAD1.1, F-TFMs perform relatively better on SQuAD2.0,

<table><tr><td>Model</td><td>CoLA</td><td>SST-2</td><td>MRPC</td><td>STS-B</td><td>QQP</td><td>MNLI</td><td>QNLI</td><td>RTE</td><td>WNLI</td><td>AVG</td></tr><tr><td colspan="11">Dev set results (single model)</td></tr><tr><td> $ROBERTA_{Large}$  [4]</td><td>68.0</td><td>96.4</td><td>-/90.9</td><td>92.4</td><td>-/92.2</td><td>90.2</td><td>94.7</td><td>86.6</td><td>-</td><td>88.9</td></tr><tr><td>XLNetLarge [3]</td><td>69.0</td><td>97.0</td><td>-/90.8</td><td>92.5</td><td>-/92.3</td><td>90.8</td><td>94.9</td><td>85.9</td><td>-</td><td>89.2</td></tr><tr><td> $ELECTRA_{Large}$  [5]</td><td>69.1</td><td>96.9</td><td>-/90.8</td><td>92.6</td><td>-/92.4</td><td>90.9</td><td>95.0</td><td>88.0</td><td>-</td><td>89.5</td></tr><tr><td>B10-10-10H1024</td><td>72.4</td><td>96.8</td><td>93.5/90.9</td><td>92.1</td><td>89.8/92.4</td><td>91.1/-</td><td>95.1</td><td>89.5</td><td>-</td><td>90.0</td></tr><tr><td>B8-8-8H1024</td><td>71.3</td><td>96.8</td><td>93.1/90.7</td><td>91.7</td><td>89.8/92.4</td><td>90.8/-</td><td>94.7</td><td>89.2</td><td>-</td><td>89.7</td></tr><tr><td> $ROBERTA_{Base}$  [4]</td><td>63.6</td><td>94.8</td><td>-/90.2</td><td>91.2</td><td>-/91.9</td><td>87.6/-</td><td>92.8</td><td>78.7</td><td>-</td><td>86.4</td></tr><tr><td>MPNetBase [12]</td><td>65.0</td><td>95.4</td><td>-/91.5</td><td>90.9</td><td>-/91.9</td><td>88.5/-</td><td>93.3</td><td>85.2</td><td>-</td><td>87.7</td></tr><tr><td>B6-6-6H768</td><td>70.1</td><td>96.3</td><td>93.2/90.4</td><td>91.1</td><td>89.2/92.0</td><td>89.7/-</td><td>93.7</td><td>83.4</td><td>-</td><td>88.3</td></tr><tr><td>B6-3x2-3x2H768</td><td>68.5</td><td>95.6</td><td>92.5/89.5</td><td>91.0</td><td>89.3/92.0</td><td>89.1/-</td><td>93.0</td><td>83.4</td><td>-</td><td>87.8</td></tr><tr><td>B4-4-4H768</td><td>68.2</td><td>95.0</td><td>92.8/90.2</td><td>90.3</td><td>89.0/91.8</td><td>88.6/-</td><td>92.6</td><td>79.1</td><td>-</td><td>87.0</td></tr><tr><td colspan="11">Leaderboard test set results (single task &amp; single model)</td></tr><tr><td> $ELECTRA_{Large}$  [5]</td><td>68.1</td><td>96.7</td><td>89.2/92.0</td><td>92.1/91.7</td><td>74.8/90.4</td><td>90.7/90.2</td><td>95.5</td><td>86.1</td><td>65.1</td><td>85.2</td></tr><tr><td>B10-10-10H1024</td><td>68.9</td><td>97.2</td><td>89.4/92.1</td><td>91.6/91.3</td><td>74.3/90.2</td><td>90.9/90.9</td><td>95.5</td><td>86.5</td><td>65.1</td><td>85.4</td></tr><tr><td>B8-8-8H1024</td><td>68.3</td><td>96.9</td><td>89.2/92.0</td><td>91.5/91.1</td><td>73.8/90.1</td><td>90.7/90.7</td><td>95.1</td><td>85.3</td><td>65.1</td><td>85.0</td></tr><tr><td> $ELECTRA_{Base}$  [5]</td><td>64.6</td><td>96.0</td><td>88.1/91.2</td><td>91.0/90.2</td><td>73.2/89.5</td><td>88.5/88.0</td><td>93.1</td><td>75.2</td><td>65.1</td><td>82.7</td></tr><tr><td>B6-6-6H768</td><td>68.3</td><td>96.5</td><td>89.1/91.9</td><td>90.6/89.9</td><td>73.3/89.9</td><td>89.7/89.4</td><td>94.0</td><td>80.4</td><td>65.1</td><td>84.0</td></tr><tr><td>B6-3x2-3x2H768</td><td>65.9</td><td>96.0</td><td>87.8/91.0</td><td>90.0/89.6</td><td>73.3/89.8</td><td>88.9/88.7</td><td>93.8</td><td>79.9</td><td>65.1</td><td>83.4</td></tr><tr><td colspan="11">Leaderboard test set results (multi-task &amp; ensemble)</td></tr><tr><td> $ROBERTA_{Large}$  [4]</td><td>67.8</td><td>96.7</td><td>89.8/92.3</td><td>92.2/91.9</td><td>74.3/90.2</td><td>90.8/90.2</td><td>95.4</td><td>88.2</td><td>89.0</td><td>88.1</td></tr><tr><td> $ELECTRA_{Large}$  [5]</td><td>71.7</td><td>97.1</td><td>90.7/93.1</td><td>92.9/92.5</td><td>75.6/90.8</td><td>91.3/90.8</td><td>95.8</td><td>89.8</td><td>91.8</td><td>89.4</td></tr><tr><td>B10-10-10H1024</td><td>70.5</td><td>97.5</td><td>91.2/93.4</td><td>92.6/92.3</td><td>75.4/90.7</td><td>91.4/91.1</td><td>95.8</td><td>90.0</td><td>94.5</td><td>89.7</td></tr></table>

Table 3: Comparison with previous methods on the GLUE benchmark under large-scale pretraining.

<table><tr><td rowspan="2">Model</td><td colspan="3">RACE</td></tr><tr><td>Total</td><td>High</td><td>Middle</td></tr><tr><td> $ROBERTA_{Large}$  [4]</td><td>83.2</td><td>81.3</td><td>86.5</td></tr><tr><td>XLNetLarge [3]</td><td>85.4</td><td>84.0</td><td>88.6</td></tr><tr><td>B10-10-10</td><td>85.7</td><td>84.4</td><td>88.8</td></tr><tr><td>B8-8-8</td><td>85.2</td><td>83.9</td><td>88.4</td></tr><tr><td> $ALBERT_{Base}$  [6]</td><td>66.0</td><td>-</td><td>-</td></tr><tr><td>MPNetBase [12]</td><td>72.0</td><td>76.3</td><td>70.3</td></tr><tr><td>B6-6-6</td><td>79.7</td><td>78.2</td><td>83.4</td></tr><tr><td>B6-3x2-3x2</td><td>78.8</td><td>77.5</td><td>82.0</td></tr><tr><td>B4-4-4</td><td>76.2</td><td>74.6</td><td>80.0</td></tr></table>

Table 4: RACE test performance comparison.

<table><tr><td rowspan="2">Model</td><td colspan="2">SQuAD2.0</td><td colspan="2">SQuAD1.1</td></tr><tr><td>EM</td><td>F1</td><td>EM</td><td>F1</td></tr><tr><td> $ROBERTA_{Large}$  [4]</td><td>86.5</td><td>89.4</td><td>88.9</td><td>94.6</td></tr><tr><td> $ELECTRA_{Large}$  [5]</td><td>88.0</td><td>90.6</td><td>89.7</td><td>94.9</td></tr><tr><td>B10-10-10</td><td>87.6</td><td>90.4</td><td>89.0</td><td>94.7</td></tr><tr><td>B8-8-8</td><td>87.1</td><td>89.8</td><td>88.7</td><td>94.4</td></tr><tr><td> $ROBERTA_{Base}$  [4]</td><td>80.5</td><td>83.7</td><td>84.6</td><td>91.5</td></tr><tr><td> $MPNet_{Base}$  [21]</td><td>80.5</td><td>83.3</td><td>86.8</td><td>92.5</td></tr><tr><td>B6-6-6</td><td>85.1</td><td>87.7</td><td>87.4</td><td>93.3</td></tr><tr><td>B6-3x2-3x2</td><td>84.2</td><td>87.0</td><td>87.0</td><td>93.0</td></tr><tr><td>B4-4-4</td><td>82.6</td><td>85.5</td><td>85.9</td><td>92.2</td></tr></table>

Table 5: SQuAD dev performance comparison.  
which additionally requires the model to make a sequence-level prediction on whether the question is answerable. This again shows the general effectiveness of the F-TFM in sequence-level tasks.

## 4.3 Ablation Study

<table><tr><td>ID</td><td>Layout</td><td>(FLOPs / Params)</td><td>Pool-Op</td><td>Pool-query-only</td><td>Sep [cls]</td><td>Rel-Attn</td><td>GLUE-AVG</td></tr><tr><td>(1)</td><td>B6-6-6</td><td>(1.00x / 1.00x)</td><td>Mean</td><td>√</td><td>√</td><td>√</td><td>83.5</td></tr><tr><td>(2)</td><td></td><td></td><td>Mean</td><td>√</td><td></td><td>√</td><td>82.9</td></tr><tr><td>(3)</td><td></td><td></td><td>Mean</td><td></td><td>√</td><td>√</td><td>83.0</td></tr><tr><td>(4)</td><td></td><td></td><td>Mean</td><td>√</td><td>√</td><td></td><td>81.4</td></tr><tr><td>(5)</td><td></td><td></td><td>Max</td><td>√</td><td>√</td><td>√</td><td>83.4</td></tr><tr><td>(6)</td><td></td><td></td><td>Top-Attn</td><td>√</td><td>√</td><td>√</td><td>75.8</td></tr><tr><td>(7)</td><td>B8-8</td><td>(1.14x / 0.91x)</td><td>Mean</td><td>√</td><td>√</td><td>√</td><td>83.4</td></tr><tr><td>(8)</td><td>B5-5-5-5</td><td>(0.89x / 1.08x)</td><td>Mean</td><td>√</td><td>√</td><td>√</td><td>82.9</td></tr></table>

Table 6: Ablation study of F-TFMs with different designs.

Finally, based on the GLUE benchmark, we perform a series of ablation studies on the importance of various designs in F-TFM, including the block layout design, the type of pooling operation, the pool-query-only technique, maintaining a separate [cls] vector and the usage of Transformer-XL parameterization.

• Pooling operation: Including the mean pooling we finally employ in F-TFM, we actually test two types of pooling operations.

(1) The first type is just the strided mean/max pooling as described in section 2.

(2) The second type aims to select a subset of “hub” states, which refer to those hidden vectors that are attended most in the previous S-Attn layer and hence likely to carry most critical information about the sequence. Concretely, given the attention map from the previous S-Attn layer, we reduce sum the scores along the number of head and query length dimensions to a score for each position. Then, we simply choose the top 50% of states to achieve the same compression rate. Note that, this type of pooling operation is essentially the same as the important states selection procedure in Power-BERT [20].

• Pool-query-only design

• Separating [cls] in the pooling operation

• Block layout design: In our experiments, all models actually utilize a 3-block design. Here, we compare the 3-blocks design with the 2-blocks and the 4-blocks design.

• Relative attention parameterization proposed in Transformer-XL [32]. We compare this parameterization with the learned absolute position embedding as used in the BERT [2].

The ablation results are included in Table 6. To save the computation resources, the size of model hid den states in table 6 is set as 512. From the ablation results, we can make the following observations:

• Comparing pooling different operation ((1), (5), and (6)), we found that the performance of the mean and max pooling operation is similar. But they are significantly better than the idea of utilizing attention score (Top-Attn pooling) to select the “hub” states.

• Comparing (1) with (2) and (3) respectively, we see that the two special designs, i.e. “pool-queryonly” and maintaining a separate non-pooled [cls] , can both bring a clear improvement to the proposed model.

• Comparing (1) and (4), we find that the relative positional parameterization is key to the performance of the proposed F-TFM. We suspect that the pooling operation could destroy the positional information carried by the absolute position encoding, which is only injected to the model in the input embedding layer. As a result, the higher blocks may not have enough positional information to learn a good enough attention pattern. In comparison, the positional information is injected to each layer under the relative positional attention scheme. Therefore, to achieve good result with F-TFM based on absolute positional embedding, one may inject the absolute positional embedding into each attention layer. Actually, a contemporary application of Transformer to the detection problem in computer vision shows injecting positional embedding into each layer is important [33].

• Finally, we study the influence of block layout design in our framework. With B6-6-6 as the 3-block benchmark, we consider two other layout design with similar FLOPs and number of parameters. Specifically, we consider B8-8 for the 2-block design and B5-5-5-5 for the 4-block design. Comparing the results in (1), (7), and (8), we find that the performance of the 3-block (B6-6-6) design achieves the best performance, which is significantly better than the 4-block design and slightly better than the 2-block design. However, if we further taking the FLOPs/#Params into consideration, it is more clear that the 3-block design is superior. Therefore, in the main paper, we always use the 3-block design.

## 5 Conclusion & Discussion

In this work, under the pretraining-finetuning paradigm, we investigate a largely overlooked dimension of complexity in language processing. With the proposed Funnel-Transformer, we show how sequential resolution can be compressed in a simple form to save computation and how the saved FLOPs can be re-invested in improving the model capacity and hence the performance. Open challenges for future research include the better ways to improve the compression scheme, to optimize the block layout design and to re-invest the saved FLOPs. In addition, combining Funnel-Transformer with model compression techniques like knowledge distillation and quantization would be an important direction towards the enhancement of practical impact.

## References

[1] Matthew E Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. Deep contextualized word representations. arXiv preprint arXiv:1802.05365, 2018.

[2] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

[3] Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R Salakhutdinov, and Quoc V Le. Xlnet: Generalized autoregressive pretraining for language understanding. In Advances in neural information processing systems, pages 5754–5764, 2019.

[4] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019.

[5] Kevin Clark, Minh-Thang Luong, Quoc V Le, and Christopher D Manning. Electra: Pre-training text encoders as discriminators rather than generators. arXiv preprint arXiv:2003.10555, 2020.

[6] Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. Albert: A lite bert for self-supervised learning of language representations. arXiv preprint arXiv:1909.11942, 2019.

[7] Lingpeng Kong, Cyprien de Masson d’Autume, Wang Ling, Lei Yu, Zihang Dai, and Dani Yogatama. A mutual information maximization perspective of language representation learning. arXiv preprint arXiv:1910.08350, 2019.

[8] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv preprint arXiv:1910.10683, 2019.

[9] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, and Luke Zettlemoyer. Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. arXiv preprint arXiv:1910.13461, 2019.

[10] Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, and Tie-Yan Liu. Mass: Masked sequence to sequence pre-training for language generation. arXiv preprint arXiv:1905.02450, 2019.

[11] Xiaodong Liu, Pengcheng He, Weizhu Chen, and Jianfeng Gao. Multi-task deep neural networks for natural language understanding. arXiv preprint arXiv:1901.11504, 2019.

[12] Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, and Tie-Yan Liu. Mpnet: Masked and permuted pre-training for language understanding. arXiv preprint arXiv:2004.09297, 2020.

[13] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing systems, pages 5998–6008, 2017.

[14] David R So, Chen Liang, and Quoc V Le. The evolved transformer. arXiv preprint arXiv:1901.11117, 2019.

[15] Daoyuan Chen, Yaliang Li, Minghui Qiu, Zhen Wang, Bofang Li, Bolin Ding, Hongbo Deng, Jun Huang, Wei Lin, and Jingren Zhou. Adabert: Task-adaptive bert compression with differentiable neural architecture search. arXiv preprint arXiv:2001.04246, 2020.

[16] Zhanghao Wu, Zhijian Liu, Ji Lin, Yujun Lin, and Song Han. Lite transformer with long-short range attention. arXiv preprint arXiv:2004.11886, 2020.

[17] Felix Wu, Angela Fan, Alexei Baevski, Yann N Dauphin, and Michael Auli. Pay less attention with lightweight and dynamic convolutions. arXiv preprint arXiv:1901.10430, 2019.

[18] Zhiqing Sun, Hongkun Yu, Xiaodan Song, Renjie Liu, Yiming Yang, and Denny Zhou. Mobilebert: a compact task-agnostic bert for resource-limited devices. arXiv preprint arXiv:2004.02984, 2020.

[19] Sandeep Subramanian, Ronan Collobert, Marc’Aurelio Ranzato, and Y-Lan Boureau. Multiscale transformer language models. arXiv preprint arXiv:2005.00581, 2020.

[20] Saurabh Goyal, Anamitra Roy Choudhary, Venkatesan Chakaravarthy, Saurabh ManishRaje, Yogish Sabharwal, and Ashish Verma. Power-bert: Accelerating bert inference for classification tasks. arXiv preprint arXiv:2001.08950, 2020.

[21] Rui Lin, Shujie Liu, Muyun Yang, Mu Li, Ming Zhou, and Sheng Li. Hierarchical recurrent neural network for document modeling. In Proceedings ofthe 2015 Conference on Empirical Methods in Natural Language Processing, pages 899–907, 2015.

[22] Xingxing Zhang, Furu Wei, and Ming Zhou. Hibert: Document level pre-training of hierarchical bidirectional transformers for document summarization. arXiv preprint arXiv:1905.06566, 2019.

[23] Vikas K Garg, Inderjit S Dhillon, and Hsiang-Fu Yu. Multiresolution transformer networks: Recurrence is not essential for modeling hierarchical structure. arXiv preprint arXiv:1908.10408, 2019.

[24] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241. Springer, 2015.

[25] Dominik Scherer, Andreas Müller, and Sven Behnke. Evaluation of pooling operations in convolutional architectures for object recognition. In International conference on artificial neural networks, pages 92–101. Springer, 2010.

[26] Zhitao Ying, Jiaxuan You, Christopher Morris, Xiang Ren, Will Hamilton, and Jure Leskovec. Hierarchical graph representation learning with differentiable pooling. In Advances in neural information processing systems, pages 4800–4810, 2018.

[27] Hongyang Gao and Shuiwang Ji. Graph u-nets. arXiv preprint arXiv:1905.05178, 2019.

[28] Junhyun Lee, Inyeop Lee, and Jaewoo Kang. Self-attention graph pooling. arXiv preprint arXiv:1904.08082, 2019.

[29] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint arXiv:1804.07461, 2018.

[30] Xiang Zhang, Junbo Zhao, and Yann LeCun. Character-level convolutional networks for text classification. In Advances in neural information processing systems, pages 649–657, 2015.

[31] Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. Race: Large-scale reading comprehension dataset from examinations. arXiv preprint arXiv:1704.04683, 2017.

[32] Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V Le, and Ruslan Salakhutdinov. Transformer-xl: Attentive language models beyond a fixed-length context. arXiv preprint arXiv:1901.02860, 2019.

[33] Nicolas Carion, F. Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander M Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. ArXiv, abs/2005.12872, 2020.

[34] Qizhe Xie, Zihang Dai, Eduard Hovy, Minh-Thang Luong, and Quoc V Le. Unsupervised data augmentation for consistency training. 2019.

## A Implementation Optimization

## A.1 Sequence Truncation for Separating [cls] trick

As discussed in Section 2.2, to avoid breaking the [cls] structure commonly used in pretraining, we do not apply the pooling operation to the [cls] and keep the hidden state corresponding to [cls] intact. While conceptually simple, a naive implementation could slow down the computation by 15% due to the “irregular” sequence length caused by such an operation. Specifically, assume that sequence length of an input sample is a power of $\mathrm { t w o } , \mathrm { i } . \mathrm { e } . , 2 ^ { p }$ , which usually is 512 in the pretraining phase. After one pooling operation with the [cls] intact, the length of the pooled sequence becomes $2 ^ { p - 1 } + 1$ , which is not a power of 2 anymore. As a result, it can cause memory misalignment and the waste of paralleled computation power in accelerators, leading to substantial speed loss.

To resolve this issue, we employ a simple strategy to truncate the last token after the pooling. Formally, denoting the pooled hidden state as $\mathbf { h } = \left\{ h _ { [ \mathsf { c 1 s } ] } \ , h _ { 1 } , \cdot \cdot \cdot \ , h _ { 2 ^ { p - 1 } } \right\}$ , the truncation can be expressed as

$$
\hat {\mathbf {h}} = \operatorname{truncate} (\mathbf {h}) = \left[ h _ {[ \mathrm{cls} ]}, h _ {1}, \dots , h _ {2 ^ {p - 1} - 1} \right]\tag{6}
$$

With this simple trick, we can always keep the sequence length a power of 2, hence avoiding the slowdown caused by maintaining an independent [cls] hidden state.

## A.2 Relative Positional Attention Implementation

In this work, we use the relative positional attention parameterization proposed in the Transformer-XL [32]. To facilitate further discussion, we first review the details of this parameterization. Taking the case of single head attention as the example head. Let T, D be the sequence length and hidden dimension respectively. Then, the pre-softmax attention score $A _ { i j }$ between a pair of positions i and j consists of two terms:

$$
A _ {i j} = \underbrace {(W _ {Q} h _ {i} + v) ^ {\top} (W _ {K} h _ {j})} _ {\text {content term}} + \underbrace {(W _ {Q} h _ {i} + u) ^ {\top} (W _ {R} r _ {i - j})} _ {\text {position term}}.\tag{7}
$$

where $v , u \in \mathbb { R } ^ { D }$ are two trainable bias vectors, $W _ { Q } , W _ { K } , W _ { R } \in \mathbb { R } ^ { D \times D }$ are three trainable projection matrices, and $r _ { i - j } \in \mathbb { R } ^ { D }$ is the sinusoidal positional encoding that represents the relative distance $i - j$ between the two positions.

To compute the entire attention score matrix A, the content term can easily be obtained via two head projections and an outer product of complexity $O ( T D ^ { 2 } + T ^ { 2 } D )$

$$
\mathbf {A} ^ {\text { content }} = (\mathbf {H} W _ {Q} + v) (\mathbf {H} W _ {K}) ^ {\top},
$$

where $\mathbf { H } = [ h _ { 1 } , \cdots , h _ { T } ] \in \mathbb { R } ^ { T \times D }$ collects all hidden states into a matrix. However, we cannot compute the position term in the same way as each $A _ { i j } ^ { \mathsf { p o s i t i o n } }$ corresponds to a different $r _ { i - j }$ . Hence, a naive solution will be stacking $T ^ { 2 }$ pairs of position encodings into a tensor $\hat { \mathbf { R } } \in \mathbb { R } ^ { T \times T \times D }$ where $\hat { \bf R } _ { i j } = r _ { i - j }$ , and then perform the following tensor product:

$$
\mathbf {A} ^ {\text { position }} = \operatorname{einsum} \left(" \mathrm{id}, \mathrm{ijd} - > \mathrm{ij}", \mathbf {H} W _ {Q} + u, \hat {\mathbf {R}} W _ {R}\right).
$$

Note that the head projection $\mathbf { R } W _ { K }$ now has a complexity of $O ( T ^ { 2 } D ^ { 2 } )$ and a memory footprint of $O ( T ^ { 2 } D )$ , dominating all other computations.

## A.2.1 Standard Solution: Gather / Shift

To resolve the computation burden above, a common technique is to instead collect a matrix $\mathbf { R } \in$ $\mathbb { R } ^ { 2 T - 1 \times D }$ , where

$$
\mathbf {R} = [ r _ {T - 1}, \ldots , r _ {0}, \dots , r _ {1 - T} ]
$$

which includes all possible position encodings arranged from the maximum possible distance value $T - 1$ to the minimum one $1 - T$ . Note that the full $\bar { \hat { \mathbf { R } } }$ can be formed by gathering specific elements from R with an index matrix I of shape $[ T \times T ]$ , i.e.,

$$
\hat {\mathbf {R}} = \operatorname{gather} (\mathbf {R}, \mathbf {I}), \quad I _ {i j} = T + i - j.
$$

Mathematically, this is equivalent to using a permutation tensor $\mathbf { P } \in \mathbb { R } ^ { T \times T \times 2 T - 1 }$ to multiply R, i.e., $\hat { \mathbf { R } } = \mathbf { P } \mathbf { R }$ , where $\mathbf { P } _ { i j } \in \mathbb { R } ^ { 2 T - 1 }$ is a one-hot vector used to select/gather a single position of R. As the attention score computation only involves linear operations, we can rearrange the computation of the position term as follows

$$
\begin{array}{l} \mathbf {A} ^ {\text {position}} = \operatorname{einsum} \bigl (" \mathrm{id}, \mathrm{ijd-} > \mathrm{ij}", \mathbf {H} W _ {Q} + u, (\mathbf {P R}) W _ {R} \bigr) \\ \qquad = \operatorname{einsum} \Bigl (" \mathrm{ijk}, \mathrm{jk-} > \mathrm{ij}", \mathbf {P}, \bigl [ (\mathbf {H} W _ {Q} + v) (\mathbf {R} W _ {R}) ^ {\top} \bigr ] \Bigr) \\ \qquad = \operatorname{gather} \Bigl ((\mathbf {H} W _ {Q} + v) (\mathbf {R} W _ {R}) ^ {\top}, \mathbf {I} \Bigr) \end{array}
$$

Note that, assuming gathering $T ^ { 2 }$ elements only has a complexity of $O ( T ^ { 2 } )$ , which is true for CPU/GPU, this trick reduces the computation complexity back to $\dot { O } ( 2 T D ^ { 2 } + 2 T ^ { 2 } D )$ ). In practice, the gather operation can be implemented via a smart reshape operation, that is even cheaper.

## A.2.2 Optimization for TPU: factorized relative positional attention

However, on $\mathrm { T P U s } ,$ the assumption that gathering $T ^ { 2 }$ elements only has a complexity of $O ( T ^ { 2 } )$ does not hold. Instead, we found that such a gather operation is dramatically slower on TPU. Hence, we here consider another implementation which is significantly faster on TPU.

Firstly, let’s rewrite the position term as follows

$$
\begin{array}{r l} A _ {i j} ^ {\text {position}} & = (W _ {Q} h _ {i} + u) ^ {\top} (W _ {R} r _ {i - j}) \\ & = \Big [ \underbrace {W _ {R} ^ {\top} (W _ {Q} h _ {i} + u)} _ {q _ {i}} \Big ] ^ {\top} r _ {i - j} \\ & = q _ {i} ^ {\top} r _ {i - j}. \end{array}\tag{8}
$$

For easier derivation, we have introduced a notation of $q _ { i }$ . Then, recall the $r _ { i - j }$ is the sinusoidal encoding that consists of the sine and the cosine components $\begin{array} { r } { r _ { i - j } = { \tt c a t } ( \sin _ { i - j } , \cos _ { i - j } ) } \end{array}$ , where

$$
\begin{array}{l} \sin_ {t} = \Big [ \sin \Big (t / 1 0 0 0 0 ^ {2 / D} \Big), \sin \Big (t / 1 0 0 0 0 ^ {4 / D} \Big), \dots , \sin \Big (t / 1 0 0 0 0 ^ {D / D} \Big) \Big ] \in \mathbb {R} ^ {D / 2}, \\ \cos_ {t} = \Big [ \cos \Big (t / 1 0 0 0 0 ^ {2 / D} \Big), \cos \Big (t / 1 0 0 0 0 ^ {4 / D} \Big), \dots , \cos \Big (t / 1 0 0 0 0 ^ {D / D} \Big) \Big ] \in \mathbb {R} ^ {D / 2}. \end{array}
$$

Hence, we similarly divide $q _ { i }$ defined above into two parts, i.e.,

$$
q _ {i} = \mathsf {c a t} (q _ {i} ^ {\sin}, q _ {i} ^ {\cos}).
$$

Given the definitions, we can further break Eqn. (8) into two terms:

$$
A _ {i j} ^ {\text { position }} = q _ {i} ^ {\top} r _ {i - j} = q _ {i} ^ {\sin^ {\top}} \sin_ {i - j} + q _ {i} ^ {\cos^ {\top}} \cos_ {i - j}.
$$

Now, using the trigonometric identities sin $( a - b ) = \sin ( a ) \cos ( b ) - \cos ( a ) \sin ( b )$ and $\cos ( a - b ) =$ cos $( a ) \cos ( b ) + \sin ( a ) \sin ( b )$ , the two terms can be respectively reformulated into

$$
\begin{array}{r l} & {q _ {i} ^ {\sin^ {\top}} \sin_ {i - j} = q _ {i} ^ {\sin^ {\top}} [ \sin_ {i} \odot \cos_ {j} - \cos_ {i} \odot \sin_ {j} ]} \\ & {\quad = q _ {i} ^ {\sin^ {\top}} (\sin_ {i} \odot \cos_ {j}) - q _ {i} ^ {\sin^ {\top}} (\cos_ {i} \odot \sin_ {j})} \\ & {\quad = \left[ q _ {i} ^ {\sin} \odot \sin_ {i} \right] ^ {\top} \cos_ {j} + \left[ q _ {i} ^ {\sin} \odot (- \cos_ {i}) \right] ^ {\top} \sin_ {j}} \end{array}
$$

and

$$
\begin{array}{r l} & {q _ {i} ^ {\cos^ {\top}} \cos_ {i - j} = q _ {i} ^ {\cos^ {\top}} [ \cos_ {i} \odot \cos_ {j} + \sin_ {i} \odot \sin_ {j} ]} \\ & {\quad = q _ {i} ^ {\cos^ {\top}} (\cos_ {i} \odot \cos_ {j}) + q _ {i} ^ {\cos^ {\top}} (\sin_ {i} \odot \sin_ {j})} \\ & {\quad = [ q _ {i} ^ {\cos} \odot \cos_ {i} ] ^ {\top} \cos_ {j} + [ q _ {i} ^ {\cos} \odot \sin_ {i} ] ^ {\top} \sin_ {j}} \end{array}
$$

Hence, combining these two parts together, it follows that

$$
\begin{array}{r l} & q _ {i} ^ {\top} r _ {i - j} = q _ {i} ^ {\sin^ {\top}} \sin_ {i - j} + q _ {i} ^ {\cos^ {\top}} \cos_ {i - j} \\ & \quad = \left[ q _ {i} ^ {\sin} \odot \sin_ {i} \right] ^ {\top} \cos_ {j} + \left[ q _ {i} ^ {\sin} \odot (- \cos_ {i}) \right] ^ {\top} \sin_ {j} + [ q _ {i} ^ {\cos} \odot \cos_ {i} ] ^ {\top} \cos_ {j} + [ q _ {i} ^ {\cos} \odot \sin_ {i} ] ^ {\top} \sin_ {j} \\ & \quad = \left\{\left[ q _ {i} ^ {\sin} \odot \sin_ {i} \right] ^ {\top} \cos_ {j} + [ q _ {i} ^ {\cos} \odot \cos_ {i} ] ^ {\top} \cos_ {j} \right\} + \left\{\left[ q _ {i} ^ {\sin} \odot (- \cos_ {i}) \right] ^ {\top} \sin_ {j} + [ q _ {i} ^ {\cos} \odot \sin_ {i} ] ^ {\top} \sin_ {j} \right\} \\ & \quad = \left[ \underbrace {\mathsf {c a t} (q _ {i} ^ {\sin} , q _ {i} ^ {\cos})} _ {= q _ {i}} \odot \underbrace {\mathsf {c a t} (\sin_ {i} , \cos_ {i})} _ {:= \phi_ {i}} \right] ^ {\top} \underbrace {\mathsf {c a t} (\cos_ {j} , \cos_ {j})} _ {:= \psi_ {j}} \\ & \quad + \left[ \underbrace {\mathsf {c a t} (q _ {i} ^ {\sin} , q _ {i} ^ {\cos})} _ {= q _ {i}} \odot \underbrace {\mathsf {c a t} (- \cos_ {i} , \sin_ {i})} _ {:= \pi_ {i}} \right] ^ {\top} \underbrace {\mathsf {c a t} (\sin_ {j} , \sin_ {j})} _ {:= \omega_ {j}} \\ & \quad = [ q _ {i} \odot \phi_ {i} ] ^ {\top} \psi_ {j} + [ q _ {i} \odot \pi_ {i} ] ^ {\top} \omega_ {j}, \end{array}
$$

where $\phi _ { i } , \psi _ { j } , \pi _ { i } , \omega _ { j }$ above are simply 4 positional encodings formed by concatenating the cosine and sine vectors of the corresponding i and $j$ in different ways. Note that, each term of the last line has afactorized form that can be computed via an outer product, just like the standard content term. Therefore, by stacking $\phi _ { i } , \psi _ { j } , \pi _ { i } , \omega _ { j }$ of all positions $( \mathrm { i } . \mathrm { e } . \ i = 1 , \dots , T$ and $j = 1 , \dots , T )$ into the corresponding Φ, Ψ, Π, $\pmb { \Omega } \in \mathbb { R } ^ { T \times D }$ respectively, the full position term can be expressed in a simple form

$$
\mathbf {A} ^ {\text { position }} = \left\{\left[ (\mathbf {H} W _ {Q} + u) W _ {R} ^ {\top} \right] \odot \boldsymbol {\Phi} \right\} \boldsymbol {\Psi} ^ {\top} + \left\{\left[ (\mathbf {H} W _ {Q} + u) W _ {R} ^ {\top} \right] \odot \boldsymbol {\Pi} \right\} \boldsymbol {\Omega} ^ {\top}
$$

which leads to the complexity of $O ( 2 T D ^ { 2 } + 4 T ^ { 2 } D )$ , which is comparable to the content term.

## A.3 Potential Model Extensions

In this section, we discuss some potential model extensions of Funnel-Transformer. As described in section 2, Funnel-Transformer can be divided into an encoder with a compression functionality and a decoder that recovers the full-length token-level representations. To further extend the proposed model, first note that the encoder-decoder framework can be formulated into a more general form:

$$
\begin{array}{l} \mathbf {h} _ {\text {enc}} = \text {Encoder} (\mathbf {x} _ {\text {enc}}), \\ \mathbf {h} _ {\text {dec}} = \text {Decoder} (\mathbf {h} _ {\text {enc}}, \mathbf {x} _ {\text {dec}}), \end{array}
$$

where $\mathbf { x } _ { \mathrm { e n c } }$ and $\mathbf { x } _ { \mathrm { d e c } }$ are the encoder input sequence and the optional and problem-specific decoder input, respectively. The goal of encoder is to compressing the input sequence $\mathbf { x } _ { \mathrm { e n c } }$ into the hidden representations $\mathbf { h } _ { \mathrm { e n c } }$ with a reduced length. Then, conditioned on the decoder input $\mathbf { h } _ { \mathrm { e n c } }$ if any, the decoder will extract relevant information/representations from $\mathbf { h } _ { \mathrm { e n c } }$ to solve the specific NLP problem at hand. Next, we will how the general form of Funnel-Transformer can be instantiated into specific forms to solve corresponding NLP problems.

Sequence-level prediction This is essentially the case we consider in most of our experiments where we want to obtain a vectorial representation of the input sequence such as text classification. In this case, we don’t really need the decoder $\mathbf { x } _ { \mathrm { d e c } } \left( \mathrm { i . e . \ x _ { \mathrm { d e c } } = } \emptyset \right)$ and the decoder simply extracts the hidden representation corresponding to the [cls] token from $\mathbf { h } _ { \mathrm { e n c } }$ and feeds it into the task-specific structure (e.g. classifier).

Token-level prediction In the token-level prediction tasks such as the MLM pretraining, SQuAD and sequence labeling, we need a decoder to recover the token-level representations from the compressed sequence $\mathbf { h } _ { \mathrm { e n c } } .$ . In many cases, $\mathbf { x } _ { \mathrm { d e c } }$ could simply be the original sequence or a token-level hidden representation of it to provide fine grained low-level information of each token and hence ease the optimization. In this paper, we utilize the last-layer hidden states of the 1st block (before the first pooling operation) as the additional decoder input.

But for problems that utilize additional input signals, such as the permutation order used for permuted language modeling in XLNet [3]. This additional information can be injected into Funnel-Transformer via the decoder input $\mathbf { x } _ { \mathrm { d e c } }$ to (approximately) recover some more complex control of attention mechanism.

Sequence-to-sequence problems Another important category of NLP task is sequence-to-sequence problems, including machine translation, text summarization, and dialog generation, whose state-ofthe-art solution is the conventional encoder-decoder framework. Hence, Funnel-Transformer naturally fits these tasks, where the decoder input $\mathbf { X } _ { \mathrm { d e c } }$ corresponds to the target text sequence and the encoder input $\mathbf { X } _ { \mathrm { e n c } }$ the source text sequence. This way, the key difference compared to conventional models is the source side compression Funnel-Transformer provides.

Overall, we summarize some potential directions to extend Funnel-Transformer presented in section 2.2 to NLP problems. Finally, although we focus on discussion on the NLP tasks in this paper, Funnel-Transformer could be applied to any tasks dealing with sequential data, such as time series and video stream analysis.

## B Experiment Setting and Hyper-parameters

## B.1 Preprocessing & Tokenization

For all experiments conducted in this work, we simply adapt the “uncased” word piece model originally used by BERT [2], where the vocabulary size is about 30K. Other than lower case and the default preprocessing included in the word piece tokenizer, the only additional preprocessing we perform is to remove some http symbols (e.g. <b>) in the 7 text classification tasks.

## B.2 Pretraining

<table><tr><td>Hparam</td><td>Base Scale</td><td>Large Scale</td></tr><tr><td>Hidden dropout</td><td>0.1</td><td></td></tr><tr><td>GeLU dropout</td><td>0.0</td><td></td></tr><tr><td>Attention dropout</td><td>0.1</td><td></td></tr><tr><td>Max sequence length</td><td>512</td><td></td></tr><tr><td>Batch size</td><td>256</td><td>8192</td></tr><tr><td>Learning rate</td><td>1e-4</td><td>2e-4</td></tr><tr><td>Number of steps</td><td>1M</td><td>500K</td></tr><tr><td>Warmup steps</td><td>10K</td><td>30K</td></tr><tr><td>Optimizer</td><td colspan="2">Adam Weight Decay</td></tr><tr><td>Learning rate decay</td><td colspan="2">Linear</td></tr><tr><td>Adam epsilon</td><td colspan="2">1e-6</td></tr><tr><td>Weight decay</td><td colspan="2">0.01</td></tr></table>

Table 7: Hyper-parameters for pretraining.

The hyper-parameters used for the two different pretraining settings are summarized in Table 7. One exception is the learning rate used for B10-10-10H1024 at the base scale. Specifically, we find the training can be unstable when the depth goes beyond 24 layers (in the case of B10-10-10H1024) at base scale, especially for the MLM objective. Hence, we reduce the learning to 8e-5 for the B10-10-10H1024 F-TFM during base-scale pretraining. This has a side effect of a slower training pace and potentially a slightly worse finetuning performance. However, we does not observe such instability when the batch size is increased such as in the large-scale setting.

For ELECTRA, there are two additional important hyper-parameters, i.e., the discriminator loss coefficient and the relative size multiplier of the generator. In this work, we does not tune these two hyper-parameters at all and simply use the numbers from the original paper, i.e., the discriminator loss coefficient of 50 and size multiplier of 1/4 for all architectures trained with ELECTRA. In addition, in ELECTRA training, whenever F-TFM is used as the discriminator, the generator also uses the F-TFM.

In additional, in the all experiments, we only annotate the size of hidden states the rest of model sizes can be derived from on it:

• The embedding size = hidden size

• The size of inner states of P-FFN is “4 × hidden size”.

• The attention head dimension is always 64.

• The number of attention heads is “hidden size/64”.

Finally, another important element in pretraining is the mask sampling strategy. For MLM training, following previous work, we always complete word span (up to 5 complete words) sampling. However, for ELECTRA training, we notice a weird phenomenon that under the base-scale setting, the performance of both the Transformer and the F-TFM drops significantly if we use word span sampling rather than the single-token sampling. On the other hand, under the large-scale setting, using word span sampling works fine. Hence, we use single-token sampling for base-scale ELECTRA training, and word span sampling for large-scale ELECTRA training.

## B.3 Finetuning

<table><tr><td>Hparam</td><td>RTE</td><td>MRPC</td><td>STS-B</td><td>CoLA</td><td>SST-2</td><td>QNLI</td><td>MNLI</td><td>QQP</td></tr><tr><td>Hidden dropout</td><td></td><td></td><td></td><td>0.1</td><td></td><td></td><td></td><td></td></tr><tr><td>GeLU dropout</td><td></td><td></td><td></td><td>0.0</td><td></td><td></td><td></td><td></td></tr><tr><td>Attention dropout</td><td></td><td></td><td></td><td>0.1</td><td></td><td></td><td></td><td></td></tr><tr><td>Max sequence length</td><td></td><td></td><td></td><td>128</td><td></td><td></td><td></td><td></td></tr><tr><td>Batch size</td><td>16</td><td>16</td><td>16</td><td>16</td><td>32</td><td>32</td><td>64</td><td>64</td></tr><tr><td>Number of epochs</td><td>10</td><td>10</td><td>10</td><td>10</td><td>5</td><td>3</td><td>3</td><td>5</td></tr><tr><td>Learning rate decay</td><td></td><td></td><td></td><td>Linear</td><td></td><td></td><td></td><td></td></tr><tr><td>Weight decay</td><td></td><td></td><td></td><td>0.01</td><td></td><td></td><td></td><td></td></tr><tr><td>Warmup proportion</td><td></td><td></td><td></td><td>0.1</td><td></td><td></td><td></td><td></td></tr><tr><td>Adam epsilon</td><td></td><td></td><td></td><td>1e-6</td><td></td><td></td><td></td><td></td></tr><tr><td>Hparam</td><td>IMDB</td><td>AG</td><td>DBpedia</td><td>Yelp-2</td><td>Yelp-5</td><td colspan="2">Amazon-2</td><td>Amazon-5</td></tr><tr><td>Hidden dropout</td><td></td><td></td><td></td><td>0.1</td><td></td><td></td><td></td><td></td></tr><tr><td>GeLU dropout</td><td></td><td></td><td></td><td>0.0</td><td></td><td></td><td></td><td></td></tr><tr><td>Attention dropout</td><td></td><td></td><td></td><td>0.1</td><td></td><td></td><td></td><td></td></tr><tr><td>Max sequence length</td><td>512</td><td>128</td><td>128</td><td>512</td><td>512</td><td colspan="2">512</td><td>512</td></tr><tr><td>Batch size</td><td>32</td><td>32</td><td>64</td><td>128</td><td>128</td><td colspan="2">128</td><td>128</td></tr><tr><td>Number of epochs</td><td>5</td><td>3</td><td>3</td><td>3</td><td>3</td><td colspan="2">3</td><td>3</td></tr><tr><td>Learning rate decay</td><td></td><td></td><td></td><td>Linear</td><td></td><td></td><td></td><td></td></tr><tr><td>Weight decay</td><td></td><td></td><td></td><td>0.01</td><td></td><td></td><td></td><td></td></tr><tr><td>Warmup proportion</td><td></td><td></td><td></td><td>0.1</td><td></td><td></td><td></td><td></td></tr><tr><td>Adam epsilon</td><td></td><td></td><td></td><td>1e-6</td><td></td><td></td><td></td><td></td></tr></table>

Table 8: Hyper-parameters for finetuning on the GLUE benchmark and 7 text classification datasets.

For all the finetuning experiments, we essentially inherit the hyper-parameters used by XLNet [3]. All the performance numbers reported are obtained on TPUs with TensorFlow 2.2.

## B.3.1 GLUE & Text Classification

For GLUE and text classification datasets, we first fix the values of most hyper-parameters shown in Table 8. Then, we only search the learning rates from the set [1e-5, 2e-5, 3e-5], and choose the best one according to the validation set.

Following previous work [3, 4, 5], all GLUE performances correspond to the median result of 5 runs from different random seeds in the base setting and 15 runs in the large setting, respectively.

For the text classification, the base-scale results are the median performance among 5 runs with different random seeds. However, for the large-scale experiments, to be compatible with previous work [34, 3], the results are the best performance among 5 random runs.

## B.3.2 Reading Comprehension

Again, following XLNet [3], the hyper-parameters used for finetuning on the RACE and SQuAD datasets are summarized in Table 9. “Layer-wise decay” means exponentially decaying the learning rates of individual layers in a top-down manner. For example, suppose the 24-th layer uses a learning rate l, and the Layer-wise decay rate is α, then the learning rate of layer m is $l \alpha ^ { \dot { 2 } 4 - m }$ . In addition, for the two versions of SQuAD, we simply reuse the model trained on SQuAD v2.0 when evaluated on SQuAD v1.1.

<table><tr><td>Hparam</td><td>RACE</td><td>SQuAD</td></tr><tr><td>Dropout</td><td>0.1</td><td></td></tr><tr><td>Attention dropout</td><td>0.1</td><td></td></tr><tr><td>Max sequence length</td><td>512</td><td>512</td></tr><tr><td>Training epochs/steps</td><td>5 epochs</td><td>8000 steps</td></tr><tr><td>Warmup proportion/steps</td><td>0.1</td><td>1000 steps</td></tr><tr><td>Batch size</td><td>[16, 32]</td><td>48</td></tr><tr><td>Learning rate</td><td>[1e-5, 2e-5]</td><td>3e-5</td></tr><tr><td>Learning rate decay</td><td colspan="2">linear</td></tr><tr><td>Weight decay</td><td colspan="2">0.01</td></tr><tr><td>Adam epsilon</td><td colspan="2">1e-6</td></tr><tr><td>Layer-wise lr decay</td><td>1.0</td><td>0.75</td></tr></table>

Table 9: Hyper-parameters for RACE and SQuAD.

## C Additional Experimental Results

## C.1 Text Classification at Large Scale

<table><tr><td>Model</td><td>IMDB</td><td>AG</td><td>DBpedia</td><td>Yelp-2</td><td>Yelp-5</td><td>Amazon-2</td><td>Amazon-5</td></tr><tr><td>BERT-Large</td><td>4.51</td><td>-</td><td>0.64</td><td>1.89</td><td>29.32</td><td>2.63</td><td>34.17</td></tr><tr><td>ROBERTA-Large</td><td>3.50</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>XLNet-Large</td><td>3.20</td><td>4.45</td><td>0.64</td><td>1.37</td><td>27.05</td><td>2.11</td><td>31.67</td></tr><tr><td>B10-10-10H1024</td><td>3.36</td><td>4.66</td><td>0.60</td><td>1.33</td><td>27.14</td><td>2.10</td><td>31.64</td></tr><tr><td>B8-8-8H1024</td><td>3.42</td><td>4.96</td><td>0.63</td><td>1.39</td><td>27.20</td><td>2.14</td><td>31.74</td></tr><tr><td>MPNet</td><td>4.40</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>B6-6-6H768</td><td>3.72</td><td>5.00</td><td>0.64</td><td>1.50</td><td>27.73</td><td>2.27</td><td>32.11</td></tr><tr><td>B6-3x2-3x2H768</td><td>3.82</td><td>5.12</td><td>0.64</td><td>1.58</td><td>27.96</td><td>2.32</td><td>32.23</td></tr><tr><td>B4-4-4H768</td><td>4.12</td><td>5.09</td><td>0.67</td><td>1.70</td><td>28.40</td><td>2.35</td><td>32.46</td></tr></table>

Table 10: Text classification performance comparison under the large-scale pretraining.

Table 10 includes the performance comparison on 7 text classification tasks under the large-scale training setting. Similar to the GLUE benchmark results, compared with the previous result based on Transformer, with fewer FLOPs, the proposed F-TFM achieves comparable results.

## C.2 Training Cost Comparison

In this section, we test the pretraining and finetuning speed of the F-TFM in comparison to the standard Transformer on the TPU and GPU platform. For the pretraining speed evaluation, we test F-TFM on TPU v3-16 (16 cores x 16Gb) with TensorFlow. For the finetuning speed evaluation, we test F-TFM on TPU v2-8 (8 cores x 8Gb) with TensorFlow and on Nvidia-V100 (16Gb) GPU with the PyTorch. The TensorFlow version is 2.2.0, and the PyTorch version is 1.5.0. For the GPU experiments, we use an 8-GPU node on the Google Cloud Platform. All running speeds are reported with the FP16 optimizer. In the PyTorch implementation, we use “O2” options of AMP manager in the apex<sup>3</sup> package to handle the FP16 optimization. For finetuning, we consider three different sequence lengths, namely 128, 256 and 512. For pretraining, we only consider the sequence length 512. In each case, we choose the maximum possible batch size allowed by the memory size of the device(s). We measure the actual model running time by performing 1000 steps gradient descent with random input sequences with the fixed length.

Firstly, we compare the model speed in the finetuning stage. Note that the decoder is not used in this setting. Table 11 and 12 summarize the finetuning running time comparison on GPUs and TPUs, respectively.

• In the base model (L12H768) group, we observe that the speed of B6-6-6H768 is similar or faster than the base Transformer model, despite the fact that B6-6-6 is deeper, has more parameters.

<table><tr><td>Sequence length</td><td colspan="3">128</td><td colspan="3">256</td><td colspan="2">512</td><td></td></tr><tr><td rowspan="2">Metrics</td><td colspan="2">Run time</td><td rowspan="2">Mem</td><td colspan="2">Run time</td><td rowspan="2">Mem</td><td colspan="2">Run time</td><td rowspan="2">GLUE</td></tr><tr><td>1 GPU</td><td>8 GPUs</td><td>1 GPU</td><td>8 GPUs</td><td>8 GPUs</td><td>Mem</td></tr><tr><td>Batch size / GPU</td><td colspan="3">64</td><td colspan="3">32</td><td colspan="2">16</td><td></td></tr><tr><td>L12H768</td><td>1.00x</td><td>1.00x</td><td>9.2G</td><td>1.00x</td><td>1.00x</td><td>11.0G</td><td>1.00x</td><td>14.3G</td><td>84.40</td></tr><tr><td>B6-6-6</td><td>0.97x</td><td>0.99x</td><td>9.1G</td><td>0.95x</td><td>0.97x</td><td>10.3G</td><td>0.94x</td><td>12.5G</td><td>85.37</td></tr><tr><td>B6-3x2-3x2</td><td>0.93x</td><td>0.93x</td><td>8.4G</td><td>0.91x</td><td>0.92x</td><td>9.5G</td><td>0.90x</td><td>11.8G</td><td>84.78</td></tr><tr><td>B4-4-4</td><td>0.67x</td><td>0.67x</td><td>6.6G</td><td>0.65x</td><td>0.66x</td><td>7.5G</td><td>0.64x</td><td>9.0G</td><td>83.99</td></tr><tr><td>Batch size / GPU</td><td colspan="3">32</td><td colspan="3">12</td><td colspan="2">4</td><td></td></tr><tr><td>L24H1024</td><td>1.00x</td><td>1.00x</td><td>14.8G</td><td>1.00x</td><td>1.00x</td><td>14.4G</td><td>1.00x</td><td>13.9G</td><td>86.62</td></tr><tr><td>B10-10-10</td><td>0.87x</td><td>0.92x</td><td>14.0G</td><td>0.90x</td><td>0.93x</td><td>13.0G</td><td>0.96x</td><td>12.7G</td><td>87.03</td></tr><tr><td>B8-8-8</td><td>0.70x</td><td>0.73x</td><td>11.6G</td><td>0.73x</td><td>0.75x</td><td>10.8G</td><td>0.78x</td><td>10.5G</td><td>86.70</td></tr></table>

Table 11: Running time and memory consumption comparison between F-TFMs and the standard Transformer on the GPU. In each model group, the standard Transformer (first model) is used as the benchmark for the rest of F-TFM models. Note that, given the same batch size per GPU, the memory consumption is roughly the same for 1 GPU and 8 GPUs.

<table><tr><td>Sequence length</td><td>128</td><td>256</td><td>512</td><td></td></tr><tr><td>Metrics</td><td colspan="3">Run time on 8 TPU cores (TPUv2-8)</td><td>GLUE</td></tr><tr><td>Batch size / TPU core</td><td>64</td><td>32</td><td>16</td><td></td></tr><tr><td>L12H768</td><td>1.00x</td><td>1.00x</td><td>1.00x</td><td>84.40</td></tr><tr><td>B6-6-6</td><td>0.99x</td><td>0.88x</td><td>0.81x</td><td>85.37</td></tr><tr><td>B6-3x2-3x2</td><td>0.97x</td><td>0.87x</td><td>0.77x</td><td>84.78</td></tr><tr><td>B4-4-4</td><td>0.69x</td><td>0.62x</td><td>0.55x</td><td>83.99</td></tr><tr><td>Batch size / TPU core</td><td>16</td><td>8</td><td>4</td><td></td></tr><tr><td>L24H1024</td><td>1.00x</td><td>1.00x</td><td>1.00x</td><td>86.62</td></tr><tr><td>B10-10-10</td><td>0.89x</td><td>0.81x</td><td>0.73x</td><td>87.03</td></tr><tr><td>B8-8-8</td><td>0.66x</td><td>0.60x</td><td>0.56x</td><td>86.70</td></tr></table>

Table 12: Running time between F-TFMs and the standard Transformer on the TPU v2-8. In each model group, the standard Transformer (first model) is used as the benchmark for the rest of F-TFM models.

Moreover, B6-6-6H768 achieves better results compared with the base Transformer model. The similar conclusion applies to the B6-3x2-3x2 model, which has the same amount of parameters as the base model. The B4-4-4 model, which has the same depth and model parameters as the base model, is able to provide 30%-50% speedup without losing too much performance.

• In the large model (L24H1024) group, the conclusion is similar. The speed of the larger model B10-10-10 is almost the same as the large model, and the speed of B8-8-8 is significantly faster than the large model. In addition, when sequence length equals 512, the acceleration of F-TFM on the TPU is more obvious than the GPU.

• In the both groups, all the tested F-TFM variants have smaller memory footprint compared with the standard TFM models, showing the memory efficiency of F-TFM.

Next, we compare the model speed during pretraining under the MLM objective in table 13, which has an additional cost due to the decoder. The results show that the proposed method can still substantially improve the pretraining speed compared to the standard Transformer, though the speed gain is slightly smaller than the finetuning stage. In summary, this study demonstrates that the proposed method is more efficient in both the finetuning and pretraining stages in modern parallel computing platforms.

<table><tr><td>Sequence Length</td><td colspan="2">512</td></tr><tr><td></td><td>Running Time</td><td>FLOPs</td></tr><tr><td>#TPU cores / Total bsz</td><td colspan="2">16 / 512</td></tr><tr><td>L12H768</td><td>1.00x</td><td>1.00x</td></tr><tr><td>B6-6-6H768D2</td><td>0.99x</td><td>1.04x</td></tr><tr><td>B6-3x2-3x2H768D2</td><td>0.97x</td><td>1.04x</td></tr><tr><td>B4-4-4H768D2</td><td>0.79x</td><td>0.75x</td></tr><tr><td>#TPU cores / Total bsz</td><td colspan="2">16 / 128</td></tr><tr><td>L24H1024</td><td>1.00x</td><td>1.00x</td></tr><tr><td>B10-10-10H1024D2</td><td>0.83x</td><td>0.81x</td></tr><tr><td>B8-8-8H1024D2</td><td>0.71x</td><td>0.66x</td></tr></table>

Table 13: TPU pretraining speed comparison. The suffix “D2” means that the F-TFM model has 2 decoder layers.