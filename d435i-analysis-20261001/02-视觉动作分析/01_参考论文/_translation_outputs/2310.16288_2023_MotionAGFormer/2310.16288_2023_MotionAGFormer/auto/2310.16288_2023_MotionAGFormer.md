# MotionAGFormer: Enhancing 3D Human Pose Estimation with a Transformer-GCNFormer Network

Soroush Mehraban<sup>1,2</sup>, Vida Adeli<sup>1,3</sup>, Babak Taati<sup>1,2,3</sup>

<sup>1</sup>KITE Research Institute, <sup>2</sup>Institute of Biomedical Engineering, University of Toronto, <sup>3</sup>Department of Computer Science, University of Toronto

{soroush.mehraban, vida.adeli}@mail.utoronto.ca, babak.taati@uhn.ca

## Abstract

Recent transformer-based approaches have demonstrated excellent performance in 3D human pose estimation. However, they have a holistic view and by encoding global relationships between all the joints, they do not capture the local dependencies precisely. In this paper, we present a novel Attention-GCNFormer (AGFormer) block that divides the number of channels by using two parallel transformer and GCNFormer streams. Our proposed GCN-Former module exploits the local relationship between adjacent joints, outputting a new representation that is complementary to the transformer output. By fusing these two representation in an adaptive way, AGFormer exhibits the ability to better learn the underlying 3D structure. By stacking multiple AGFormer blocks, we propose Motion-AGFormer in four different variants, which can be chosen based on the speed-accuracy trade-off. We evaluate our model on two popular benchmark datasets: Human3.6M and MPI-INF-3DHP. MotionAGFormer-B achieves stateof-the-art results, with P1 errors of 38.4 mm and 16.2 mm, respectively. Remarkably, it uses a quarter of the parameters and is three times more computationally efficient than the previous leading model on Human3.6M dataset. Code and models are available at https://github.com/ TaatiTeam/MotionAGFormer.

## 1. Introduction

Human pose estimation in 3D space is an active area of research with significant implications for numerous applications, from augmented [21] and virtual reality [26] to autonomous vehicles [2, 9, 43], human-computer interaction [27] and beyond. With this vast range of applications, the demand for more accurate and computationally efficient pose estimation models continues to grow. In most real-world scenarios, pose sequences are captured in 2D, primarily due to the prevalent use of standard RGB cameras. Consequently, one of the pivotal challenges in the field has been to effectively lift these 2D sequences into a 3D space. Accurate 3D human pose estimation enables the extraction of rich spatio-temporal information about human movements, and a deeper understanding of activities and interactions. Recent 3D lifting models leverage the inherent spatial and temporal coherence of human movements to enhance the precision of 3D pose predictions. Nonetheless, despite the considerable advancements, there are several significant challenges that require attention.

![](images/06e85691d69a1f01161b546d1468a440d49e5c3491eae9c95b90844ce87f46e2.jpg)  
Figure 1. Comparisons of recent pose uplifting methods on Human3.6M [14] (lower is better). MACs/frame denotes multiplyaccumulate operations per each output frame. The proposed MotionAGFormer presents different variants and attains superior results, while maintaining computational efficiency.

The Transformer architecture [41], originally designed for NLP tasks, has been adapted to various computer vision problems, including pose estimation. Its ability to capture long-range dependencies and its innate self-attention mechanism make it a promising candidate for this domain. However, a sole reliance on global attention mechanisms, as employed by standard Transformers, may not be optimal for pose estimation tasks. Human motion is inherently structured with local spatial and temporal dependencies.

One primary concern is the modeling of skeleton relations over time. Existing methods predominantly rely either on transformer architectures or graph-based models. While transformers excel at capturing long-term dependencies, graph models excel at local dependencies. So, there is an opportunity for a unified architecture that integrates the global perspective of transformers with the local precision of graph models.

Additionally, the race for achieving SOTA accuracy has often led to the development of increasingly complex models with a large number of parameters. Such models, despite their good accuracy, often become impractical for real-world applications where computational efficiency and swift response times are pivotal. Moreover, a predominant approach in recent models has been the prediction of a single 3D pose only for the central frame from a sequence of frames. This method, while seemingly efficient, leads to computational redundancy as it requires the reprocessing of several overlapping sequences. As a result, we instead employ a streamlined inference strategy that optimally exploits sequential data. This approach minimizes redundancy by predicting the complete 3D sequence of the input at a single forward pass.

In this paper, we introduce the MotionAGFormer, a novel transformer-graph hybrid architecture tailored for 3D human pose estimation. At its core, the MotionAGFormer harnesses the power of transformers to capture global information while simultaneously employing Graph Convolutional Networks (GCNs) to integrate local spatial and temporal relationships. We use an adaptive fusion to aggregate features from the transformer and graph streams. By doing so, we ensure a balanced and comprehensive representation of human motion, leading to enhanced accuracy in the 3D pose estimation (See Figure 1).

In summary, the main contributions of our paper are:

• Novel Design: We propose the MotionAGFormer model, in which we introduce a new GCNFormer module that excels in representing local dependencies inherent in human pose sequences.

• Efficiency and Flexibility: i) Our MotionAGFormer stands out due to its lightweight nature and faster speed with fewer parameters compared to previous SOTA methods, without compromising on accuracy. ii) Recognizing diverse needs, we offer different variants of MotionAGFormer, granting users the flexibility to make a balanced choice between accuracy and speed based on their specific requirements.

• SOTA Performance: MotionAGFormer achieves SOTA on two challenging datasets, Human3.6M and MPI-INF-3DHP.

## 2. Related works

3D human pose estimation. Current approaches to tackle this problem can be understood from two perspectives. From one perspective, models can be categorized based on the input video, which can be either multi-view or monocular. Models that rely on multi-view inputs [7, 15, 34, 35, 48] necessitate the simultaneous use of multiple cameras from different angles, which can be less feasible in real-world scenarios. From another perspective, considering their methodology, these models can be categorized as either direct 3D estimation approaches or 2D-3D lifting approaches. Direct 3D estimation methods [31, 32, 39, 52] infer the joints in 3D coordinate from the video frames without any intermediate step. Inspired by the rapid development and availability of accurate 2D pose estimation models, more recently, 2D-3D lifting methods first use an offthe-shelf 2D pose detectors [4, 28, 38] then lift 2D coordinates to 3D space [40, 47, 50, 51, 53]. In this work, we are using 2D-3D lifting methods by having monocular video as the input.

Transformer-based methods. Transformers [41] have shown promising result in different visual tasks [10, 37, 45, 54]. In the field of 3D human pose estimation, Pose-Former [51] was the first purely transformer-based model. PoseFormerV2 [50] improved its computational efficacy by employing a frequency-domain representation that also made it robust against sudden movements in noisy data. MHFormer [20] addressed the problem of self-occlusion and depth ambiguity by proposing a module that learns multiple plausible pose hypotheses. P-STMO [36] proposed masked pose modeling and reduced the final error by self-supervised pretraining the model. Enfalt et al. [11] decreased the computational complexity by leveraging masked token modeling. StridedFormer [19] replaced fully-connected layers in the feed-forward network of the transformer encoder with strided convolutions to progressively shrink the sequence length and efficiently lift the central frame. Unlike the abovementioned models that estimate the 3D pose for only the center frame of a sequence, MixSTE [47] provided 3D estimates for every frame in the input sequence. STCFormer [40] decreased computational complexity by separating the correlation learning into spatial and temporal components. HSTFormer [33] proposed hierarchical transformer encoders for better capturing spatial-temporal correlations. In addition to joint-joint attention which is commonly used, HDFormer [3] included bone-joint and hyperbone-joint interactions. Some works exploited the output motion representation for various tasks. MotionBERT [53] fine-tuned the model learned for the task of 3D human pose estimation for tasks such as action recognition and 3D human mesh recovery, while UPS [12] trained a unified model for action recognition, 3D pose estimation, and early action prediction at the same time.

Graph Convolutional Network. GCN-based methods have achieved remarkable success within the domain of skeleton-based action recognition [5, 18, 22]. Despite their computational efficiency in 3D human pose estimation [6,8,42,49], they usually cannot show competitive error compared to transformer-based counterparts. This is primarily due to their focus on local joints alone. Recently, GLA-GCN [44] introduced an adaptive GCN approach to leverage global representation. By employing a strided design to reduce its temporal scope, they achieve competitive 3D human pose estimation against various transformerbased models, all while maintaining a lighter memory load. However, the effectiveness of the proposed module in extracting global representation is not on par with that of an attention module.

Hybrid methods. These methods use different modules to capture distinct aspects of the input sequence and are not extensively explored yet. Recently, DC-GCT [16] proposed a Local Constraint Module based on GCN, and Global Constraint Module based onf self-attention to exploit both local and global dependecies of the input sequence. However, as their model is designed to operate with both individual input frames and sequences of frames, it does not distinguish between temporal and spatial dimensions. As a result, it falls short in delivering competitive outcomes when contrasted with transformer-based methods.

## 3. Method

## 3.1. Preliminary

We begin this section by reviewing the concept of MetaFormer [46], which forms the core of our encoders. A MetaFormer can be described as a generalization of the Transformer architecture [41], wherein the attention module is substituted with any mixer capable of transforming information among tokens. Specifically, for an input $\check { X } \in \mathbb { R } ^ { N \times C }$ , with $N$ denoting the token numbers and C representing the embedding dimension, the token mixer can be formally expressed as

$$
Y = { \mathrm { T o k e n M i x e r } } ( { \mathrm { N o r m } } ( X ) ) + X ,\tag{1}
$$

where Norm(·) denotes a normalization method such as batch or layer normalization [1,13], and TokenMixer(·) denotes a module that combines information among tokens. Our approach uses two parallel token mixers: Multi-head Self-Attention (MHSA) and Graph Convolutional Networks (GCNs) [17], each contributing uniquely to the information transformation process.

## 3.2. Overall architecture

Our objective is to lift a 2D (pixel coordinate) skeleton sequence to accurate 3D pose sequences. To this end, we propose the MotionAGFormer architecture, which uses both attention (Transformer) and graph convolutional (GC-NFormer) to lift motion sequences. An overview of this architecture is shown in Figure 2a.

The model takes a 2D input sequence with confidence score $\mathbf { X } \in \mathbb { R } ^ { T \times J \times 3 }$ , where $T$ and J refer to the number of frames and joint numbers, respectively. It then proceeds to map each joint in each time frame to a d-dimensional feature, $\mathbf { F } ^ { ( 0 ) } \in \mathbb { R } ^ { T \times J \times d }$ , using a linear projection layer. Then a spatial position embedding $\mathbf { P _ { p o s } ^ { s } } \in \bar { \mathbb { R } ^ { 1 \times J \times d } }$ is added to the tokens. It is important to highlight here that our model does not disregard the temporal token order as that information is preserved in the GCNFormer stream (further discussion in ablation studies 4.5).

Subsequent to position embedding, we use N blocks of AGFormer (Section 3.3) to compute $\mathbf { F } ^ { ( i ) } ~ \in ~ \mathbb { R } ^ { T \times J }$ ×d $( i \ = \ 1 , . . . , N )$ to effectively capture the underlying 3D structure of the skeleton sequence. Finally, we map $\bar { \mathbf { F } } ^ { ( N ) }$ to a higher dimension by applying a linear layer and tanh activation to compute motion semantic $\mathbf { M } \in \dot { \mathbb { R } } ^ { T \times J \times d ^ { \prime } }$ and use a regression head to estimate 3D pose $\hat { \textbf { P } } \in \mathbb { R } ^ { T \times J \times 3 }$ The lifting loss contains position $\left( L _ { 3 D } \right)$ and velocity $( L _ { \Delta \mathbf { P } } )$ terms defined as

$$
\begin{array} { r l } & { L _ { 3 D } = \Sigma _ { t = 1 } ^ { T } \Sigma _ { j = 1 } ^ { J } \Vert \hat { \mathbf { P } } _ { t , j } - \mathbf { P } _ { t , j } \Vert , } \\ & { L _ { \Delta \mathbf { P } } = \Sigma _ { t = 2 } ^ { T } \Sigma _ { j = 1 } ^ { J } \Vert \Delta \hat { \mathbf { P } } _ { t , j } - \Delta \mathbf { P } _ { t , j } \Vert , } \end{array}\tag{2}
$$

where $\Delta \hat { \mathbf P } _ { t } = \hat { \mathbf P } _ { t } - \hat { \mathbf P } _ { t - 1 } , \Delta \mathbf P _ { t } = \mathbf P _ { t } - \mathbf P _ { t - 1 }$ . The total lifting loss is then defined as

$$
{ \cal L } = { \cal L } _ { 3 D } + \lambda _ { \Delta \mathbf { P } } { \cal L } _ { \Delta \mathbf { P } } ,\tag{3}
$$

where the constant coefficient $\lambda _ { \Delta \mathbf { P } }$ is used to balance position accuracy and motion smoothness.

## 3.3. AGFormer Block

The AGFormer block uses a dual-stream architecture. Each stream consists of two components: a Spatial MetaFormer (Figure 2b) followed by a Temporal MetaFormer (Figure 2c). The Spatial MetaFormer processes individual body joints as distinct tokens, effectively capturing intra-frame relationships within a single frame. The Temporal MetaFormer, on the other hand, treats each frame as a single token, thus capturing inter-frame relationships over time. The key distinction between the two streams lies in their token mixer type. While one stream employs Transformers, the other stream uses GCNFormers.

Transformer stream. This stream employs a Spatial Multi-Head Self-Attention (S-MHSA) to capture spatial relationships, followed by a Temporal Multi-Head Self-Attention (T-MHSA) to capture temporal relationships. The S-MHSA is defined as

$$
\begin{array} { r } { \mathrm { S } \mathrm { - } \mathbf { M H S A } ( \mathbf { Q _ { s } } , \mathbf { K _ { s } } , \mathbf { V _ { s } } ) = \mathrm { C o n c a t } ( \mathrm { h e a d _ { i } } , . . . , \mathrm { h e a d _ { h } } ) { \mathbf { W _ { s } } } ^ { ( O ) } , } \\ { \mathrm { h e a d _ { i } } = \mathrm { s o f t m a x } ( \frac { \mathbf { Q _ { s } } ^ { ( i ) } ( \mathbf { K _ { s } } ^ { ( i ) } ) ^ { T } } { \sqrt { d _ { k } } } ) \mathbf { V _ { s } } ^ { ( i ) } , } \end{array}\tag{4}
$$

![](images/9040ca4ff143fc96416114afe51105046bf3dfb82106fc1057dab2fdfd25fe37.jpg)  
Figure 2. (a) MotionAGFormer is a novel architecture featuring N dual-stream spatio-temporal blocks, wherein one stream employs Transformers and the other leverages GCNFormers. (b) Spatial MetaFormer. Each input token represents an individual joint of the human body. (c) Temporal MetaFormer. Input tokens are frames of pose sequence.

where ${ \bf W _ { s } } ^ { ( O ) }$ is a projection parameter matrix, h is the number of parallel attention heads, and $d _ { k }$ is the feature dimension of $\mathbf { K _ { s } }$ . For computing the query matrix $\mathbf { Q _ { s } }$ , the key matrix $\mathbf { K _ { s } }$ , and the value matrix $\mathbf { V _ { s } } .$ , we have

$$
\mathbf { Q _ { s } } ^ { i } = \mathbf { F _ { s } } \mathbf { W _ { s } } ^ { ( Q , i ) } , \mathbf { K _ { s } } ^ { i } = \mathbf { F _ { s } } \mathbf { W _ { s } } ^ { ( K , i ) } , \mathbf { V _ { s } } ^ { i } = \mathbf { F _ { s } } \mathbf { W _ { s } } ^ { ( V , i ) } ,\tag{5}
$$

where $\mathbf { F _ { s } } ~ \in ~ \mathbb { R } ^ { B T \times J \times d }$ is spatial feature and $\mathbf { W _ { s } } ^ { ( Q , i ) }$ $\mathbf { W _ { s } } ^ { ( K , i ) } , \ \mathbf { W _ { s } } ^ { ( V , i ) }$ are projection matrices and $B$ is the batch size. The S-MHSA result is subsequently fed into a multilayer perceptron (MLP), followed by a residual connection and LayerNorm. This completes the first MetaFormer, i.e. the Spatial Transformer.

Next, we reshape $\mathbf { F _ { s } }$ into $\mathbf { F _ { T } } \in \mathbb { R } ^ { B J \times T \times d }$ to prepare per-joint temporal feature as the input of T-MHSA. Here we have

$$
\begin{array} { r } { \mathrm { { T } - M H S A ( { \mathbf { Q } } _ { T } , { \mathbf { K } } _ { T } , { \mathbf { V } } _ { T } ) } = \mathrm { { C o n c a t } } ( \mathrm { { h e a d } } _ { \mathrm { { i } } } , . . . , \mathrm { { h e a d } _ { h } } ) { \mathbf { W } } _ { \mathbf { T } } { \mathbf { \Lambda } } ^ { ( O ) } , } \\ { \mathrm { { h e a d } } _ { \mathrm { { i } } } = \mathrm { { s o f t m a x } } ( \frac { { \mathbf { Q } } _ { \mathbf { T } } { \mathbf { \Lambda } } ^ { ( i ) } ( { \mathbf { K } } _ { \mathbf { T } } { \mathbf { \Lambda } } ^ { ( i ) } ) ^ { T } } { \sqrt { d _ { k } } } ) { \mathbf { V } } _ { \mathbf { T } } { \mathbf { \Lambda } } ^ { ( i ) } , } \end{array}\tag{6}
$$

where ${ \bf Q _ { T } } , { \bf K _ { T } }$ , and $\mathbf { V _ { T } }$ are calculated similar to Eqn. (5).

GCNFormer stream. Unlike the Transformer stream, which aggregates global information, the GCNFormer stream focuses on local spatial and temporal relationships present within the skeleton sequence. While the local information is also available to the Transformers, the inclusion of this parallel stream allows the model to more effectively balance the integration of local and global information (see ablation analysis 4.5). The customized GCN module [24] used in our GCNFormer is defined as:

$$
G C N ( \mathbf { F } ^ { ( \mathbf { i } ) } ) = \sigma ( V ^ { l } + \mathrm { N o r m } ( \tilde { D } ^ { - \frac { 1 } { 2 } } \tilde { A } \tilde { D } ^ { - \frac { 1 } { 2 } } \mathbf { F } ^ { ( \mathbf { i } ) } W _ { 1 } + \mathbf { F } ^ { ( \mathbf { i } ) } W _ { 2 } ) .\tag{7}
$$

Where $\tilde { A } = A + I _ { N }$ represents the adjacency matrix with self-connections added, $I _ { N }$ stands for the identity matrix, $\tilde { D } i i = \Sigma j \tilde { A } j j$ is defined as the sum of elements along the diagonal of A<sup>˜</sup>, and $W _ { 1 } , W _ { 2 }$ denote trainable weight matrices specific to each layer. The activation function σ(·), such as ReLU, is applied, along with Batch Normalization [13]. The GCN’s output is then passed through an MLP, followed by residual connection and LayerNorm.

The difference between the Spatial GCNFormer and the Temporal GCNFormer lies in their adjacency matrices and input features. The input features resemble that of the Transformer stream. In the Spatial GCNFormer, the adjacency matrix represents the human topology (Figure 3a). For Temporal GCNFormer, on the other hand, we calculate the similarity between a single joint at different time frames using $\begin{array} { r } { { S i m } ( { { \bf \ddot { F } _ { T } } ^ { t _ { i } } } , { \bf F _ { T } } ^ { t _ { j } } ) = ( { \bf \ddot { F } _ { T } } ^ { t _ { i } } ) ^ { T } { \bf F _ { T } } ^ { t _ { j } } } \end{array}$ and choose the K nearest neighbors as the connected nodes in the graph (Figure 3b). Hence, the graph topology in Temporal GCN-Former is determined by the learned node features.

Adaptive Fusion. Similar to MotionBERT [53], we use adaptive fusion to aggregate extracted features of the Transformer and GCNFormer streams. This is defined as:

$$
\mathbf { F } ^ { ( i ) } = \alpha _ { T F } { } ^ { ( i ) } \circ \mathbf { F } _ { \mathbf { T F } } { } ^ { ( i - 1 ) } + \alpha _ { G F } { } ^ { ( i ) } \circ \mathbf { F } _ { \mathbf { G F } } { } ^ { ( i - 1 ) } ,\tag{8}
$$

where $\mathbf { F } ^ { ( i ) }$ represents the feature embedding extracted at depth i, the element-wise multiplication denoted by $^ { \mathrm { ~ O ~ } , }$ and $\mathbf { F _ { T F } } ^ { \prime } ( i \mathrm { - } 1 ) , ~ \mathbf { F _ { G F } } ^ { ( i \mathrm { - } 1 ) }$ refer to the extracted Transformer stream and GCNFormer stream features at depth $i - 1$ , respectively. The adaptive fusion weights $\alpha _ { T F }$ and $\alpha _ { G F }$ are

![](images/6a33e70d9192fd5beca3835f68e907f609f67a5248b8064b2241a9e044c1e474.jpg)  
Figure 3. GCNFormer module topology. (a) The Spatial GCN-Former employs the human skeleton as its underlying topology. (b) The Temporal GCNFormer uses K-nearest neighbor (K-NN) to determine connected edges by considering the highest similarity of each joint (e.g., left wrist and right ankle in the figure) across the entire time frame. After K-NN, each row is connected to K columns.

defined as

$$
\alpha _ { T F } { } ^ { ( i ) } , \alpha _ { G F } { } ^ { ( i ) } = \mathrm { s o f t m a x } ( W \cdot \mathrm { C o n c a t } ( { \bf F } _ { { \bf T F } } { } ^ { ( i - 1 ) } , { \bf F } _ { { \bf G F } } { } ^ { ( i - 1 ) } ) ) ,\tag{9}
$$

where W is a learnable linear transformation.

## 4. Experiments

We evaluate the performance of our proposed Motion-AGFormer on two large-scale 3D human pose estimation datasets, i.e., Human3.6M [14] and MPI-INF-3DHP [25].

## 4.1. Datasets and Evaluation Metrics

Human3.6M is a widely used indoor dataset for 3D human pose estimation. It contains 3.6 million video frames of 11 subjects performing 15 different daily activities. To ensure fair evaluation, we follow the standard approach and train the model using data from subjects 1, 5, 6, 7, and 8, and then test it on data from subjects 9 and 11. Following previous works [40, 50], we use two protocols for evaluation. The first protocol (referred to as P1) uses Mean Per Joint Position Error (MPJPE) in millimeters between the estimated pose and the actual pose, after aligning their root joints (sacrum). The second protocol (referred to as P2) measures Procrustes-MPJPE, where the actual pose and the estimated pose are aligned through a rigid transformation.

MPI-INF-3DHP is another large-scale dataset gathered in three different settings: green screen, non-green screen, and outdoor environments. Following previous works [40, 50], MPJPE, Percentage of Correct Keypoint (PC) within 150 mm range, and Area Under the Curve (AUC) are reported as evaluation metrics.

## 4.2. Implementation Details

Model Variants. We build four different configurations of our model, as summarized in Table 1. Our base model, known as MotionAGFormer-B, strikes a balance between accurate estimation and computational cost. The remaining variants are named according to their parameter size and computational demands and selection of each variant can be based on an application’s requirements, such as choosing between real-time processing or more precise estimations. The motion semantic dimension is $d ^ { \prime } = 5 1 2$ , the expansion layer of each MLP is $\alpha = 4 ,$ the number of attention heads is $h = 8 ,$ , and the number of temporal neighbours in GCNformer stream is $k = 2 .$ , for all experiments.

Table 1. Details of MotionAGFormer model variants. N: Number of layers. d: Hidden size. T: Number of input frames.
<table><tr><td>Method</td><td>N</td><td>d</td><td>T</td><td>Params</td><td>MACs</td></tr><tr><td>MotionAGFormer-XS</td><td>12</td><td>64</td><td>27</td><td>2.2 M</td><td>1.0 G</td></tr><tr><td>MotionAGFormer-S</td><td>26</td><td>64</td><td>81</td><td>4.8M</td><td>6.6 G</td></tr><tr><td>MotionAGFormer-B</td><td>16</td><td>128</td><td>243</td><td>11.7 M</td><td>48.3 G</td></tr><tr><td>MotionAGFormer-L</td><td>26</td><td>128</td><td>243</td><td>19.0 M</td><td>78.3 G</td></tr></table>

Experimental settings. Our model is implemented using PyTorch [30] and executed on a setup with two NVIDIA A40 GPUs. We apply horizontal flipping augmentation for both training and testing following [50, 53]. For model training, we set each mini-batch as 16 sequences. The network parameters are optimized using AdamW [23] optimizer over 90 epochs with a weight decay of 0.01. The initial learning rate is set to 5e-4 with an exponential learning rate decay schedule and the decay factor is 0.99. We use the Stacked Hourglass [29] 2D pose detection results and 2D ground truths on Human3.6M, following [53]. For MPI-INF-3DHP, ground truth 2D detection is used following a similar approach used in comparison baselines [40, 50].

## 4.3. Performance comparison on Human3.6M

We compare our MotionAGFormer with other models on Human3.6M. For a fair comparison, only results of models without extra pre-training on additional data is included. The results, outlined in Table 2, demonstrate that MotionAGFormer-L attains a P1 error of 38.4 mm for estimated 2D pose and 17.3 mm for ground truth 2D pose. Remarkably, this is achieved with approximately half the computational requirements and parameters compared to the previous SOTA (MotionBERT [53]), while being 0.8 mm and 0.5 mm more accurate, respectively. When comparing the Base and Large variants, MotionAGFormer-L shares the same P1 error as MotionAGFormer-B in the presence of noisy data, while exhibiting a 2.1 mm reduction in error when using ground truth 2D data. Furthermore,

(a)  
(b)  
(c)  
![](images/9edeb83edb3f78d75fe02bc377889d44f61122c898cf37390107b84bb54e1824.jpg)  
Figure 4. Qualitative comparisons of MotionAGFormer with STCformer [40], PoseFormerV2 [50], and MotionBERT [53]. The transparent gray skeleton is the ground-truth 3D pose. Blue represents the torso and left part, and Red indicates the right part of the estimated body.

Table 2. Quantitative comparisons on Human3.6M. T: Number of input frames. CE: Estimating center frame only. P1: MPJPE error (mm). P2: P-MPJPE error (mm). P1<sup>†</sup>: P1 error on 2D ground truth. (\*) denotes using HRNet [38] for 2D pose estimation. The best and second-best scores are in bold and underlined, respectively. For per action result, refer to the supplementary material.
<table><tr><td>Method</td><td>T</td><td>CE</td><td>Param</td><td>MACs</td><td>MACs/frame</td><td>P1↓/P2↓</td><td>P1†↓</td></tr><tr><td>*MHFormer [20] CVPR&#x27;22</td><td>351</td><td>√</td><td>30.9 M</td><td>7.0 G</td><td>7,096M</td><td>43.0/34.4</td><td>30.5</td></tr><tr><td>MixSTE [47] CVPR’22</td><td>243</td><td>X</td><td>33.6 M</td><td>139.0 G</td><td>572M</td><td>40.9/32.6</td><td>21.6</td></tr><tr><td>P-STMO [36] ECCV’22</td><td>243</td><td>√</td><td>6.2 M</td><td>0.7 G</td><td>740 M</td><td>42.8/34.4</td><td>29.3</td></tr><tr><td>Stridedformer [19] TMM&#x27;22</td><td>351</td><td>√</td><td>4.0 M</td><td>0.8 G</td><td>801 M</td><td>43.7/35.2</td><td>28.5</td></tr><tr><td>Einfalt et al. [11] WACV&#x27;23</td><td>351</td><td>√</td><td>10.4 M</td><td>0.5 G</td><td>498M</td><td>44.2/35.7</td><td></td></tr><tr><td>STCFormer [40] CVPR’23</td><td>243</td><td>X</td><td>4.7 M</td><td>19.6 G</td><td>80 M</td><td>41.0/32.0</td><td>21.3</td></tr><tr><td>STCFormer-L [40] CVPR’23</td><td>243</td><td>X</td><td>18.9 M</td><td>78.2 G</td><td>321 M</td><td>40.5/31.8</td><td></td></tr><tr><td>PoseFormerV2 [50] CVPR’23</td><td>243</td><td>√</td><td>14.3 M</td><td>0.5 G</td><td>528M</td><td>45.2/35.6</td><td></td></tr><tr><td>UPS [12] CVPR’23</td><td>243</td><td>√</td><td></td><td></td><td></td><td>40.8/32.5</td><td></td></tr><tr><td>GLA-GCN [44] ICCV’23</td><td>243</td><td>√</td><td>1.3 M</td><td>1.5 G</td><td>1,556 M</td><td>44.4/34.8</td><td>21.0</td></tr><tr><td>MotionBERT [53] ICCV&#x27;23</td><td>243</td><td>X</td><td>42.5 M</td><td>174.7 G</td><td>719 M</td><td>39.2/32.9</td><td>17.8</td></tr><tr><td>HDFormer [3] IJCAI&#x27;23</td><td>96</td><td>X</td><td>3.7 M</td><td>0.6 G</td><td>6M</td><td>42.6/33.1</td><td>21.6</td></tr><tr><td>HSTFormer [33] arXiv&#x27;23</td><td>81</td><td>X</td><td>22.7 M</td><td>1.0 G</td><td>13M</td><td>42.7/33.7</td><td>27.8</td></tr><tr><td>DC-GCT [16] arXiv&#x27;23</td><td>81</td><td>√</td><td>3.1 M</td><td>41 M</td><td>41 M</td><td>44.71-</td><td></td></tr><tr><td>MotionAGFormer-XS</td><td>27</td><td>X</td><td>2.2 M</td><td>1.0 G</td><td>37M</td><td>45.1/36.9</td><td>28.1</td></tr><tr><td>MotionAGFormer-S</td><td>81</td><td>X</td><td>4.8M</td><td>6.6 G</td><td>81 M</td><td>42.5/35.3</td><td>26.5</td></tr><tr><td>MotionAGFormer-B</td><td>243</td><td>X</td><td>11.7 M</td><td>48.3 G</td><td>198 M</td><td>38.4/32.6</td><td>19.4</td></tr><tr><td>MotionAGFormer-L</td><td>243</td><td>X</td><td>19.0 M</td><td>78.3 G</td><td>322 M</td><td>38.4/32.5</td><td>17.3</td></tr></table>

MotionAGFormer-S takes in only a third of the frames compared to the baselines, yet it manages to attain a superior P1 error compared to a majority of them. Similarly, MotionAGFormer-XS delivers comparable performance to PoseTransformerV2, even though it receives only a ninth of the input information (27 vs. 243 frames) and operates with approximately seven times fewer parameters.

## 4.4. Performance comparison on MPI-INF-3DHP

In evaluating our method on the MPI-INF-3DHP dataset, we modified our base and large variants to use 81 frames due to shorter video sequences. Across all variants, our method consistently outperforms others in terms MPJPE. Notably, our large variant achieves remarkable results with an 85.3% AUC and a 16.2 mm P1 error. This outperforms the best models by a significant margin of 1.4% in AUC and 6.9 mm in P1 error. However, it achieves 98.2% PCK, which is 0.5% lower than the PCK performance of the compared models (Table 3).

Table 3. Quantitative comparisons on MPI-INF-3DHP. T: Number of input frames. The best and second-best scores are in bold and underlined, respectively.
<table><tr><td>Method</td><td>T</td><td>PCK↑ AUC↑</td><td>MPJPE↓</td></tr><tr><td>MHFormer [20]</td><td>9</td><td>93.8 94.4</td><td>63.3 58.0</td></tr><tr><td>MixSTE [47] P-STMO [36]</td><td>27 81</td><td>66.5 75.8</td><td>54.9 32.2</td></tr><tr><td>Einfalt et al. [11]</td><td>81</td><td>97.9 95.4 67.6</td><td>46.9</td></tr><tr><td>STCFormer [40]</td><td>81</td><td>98.7 83.9</td><td>23.1</td></tr><tr><td>PoseFormerV2 [50]</td><td></td><td></td><td></td></tr><tr><td></td><td>81</td><td>97.9 78.8</td><td>27.8</td></tr><tr><td>GLA-GCN [44]</td><td>81</td><td>98.5 79.1</td><td>27.7</td></tr><tr><td>HSTFormer [33]</td><td>81</td><td>97.3 71.5</td><td>41.4</td></tr><tr><td>HDFormer [3]</td><td>96</td><td>98.7 72.9</td><td>37.2</td></tr><tr><td>MotionAGFormer-XS</td><td>27</td><td>98.2 83.5</td><td>19.2</td></tr><tr><td>MotionAGFormer-S</td><td>81</td><td>98.3 84.5</td><td>17.1</td></tr><tr><td>MotionAGFormer-B</td><td>81</td><td>98.3 84.2</td><td>18.2</td></tr><tr><td>MotionAGFormer-L</td><td>81</td><td>98.2 85.3</td><td>16.2</td></tr></table>

## 4.5. Ablation Studies

A series of ablation studies were conducted on the Human3.6M dataset to explore different design choices for the AGFormer block.

The initial ablation study investigates the influence of favoring the number of AGFormer blocks and the width of our model on the P1 error. As shown in Table 4, the trend generally leans towards favoring a model that is deeper but narrower. Interestingly, a model with 16 AGFormer blocks and a width of 128 exhibits similar performance to a model featuring 12 AGFormer blocks with a width of 256, while the first configuration uses approximately three times less memory and computational resources.

The second part of the ablation study explores alternative modules for the graph stream within the AGFormer block. The results are presented in Table 5. Shifting from GCNFormer to GCN blocks degrades the P1 error by 0.7 mm. Similarly, substituting the Temporal GCNFormer with TCN degrades the P1 error by 0.2 mm when using GCN and 0.9 mm when using GCNFormer. Lastly, replacing the GCN with a CTR-GCN [5] increases the P1 error by 2.5 mm. We hypothesize that is because of the continuous refinement of topology for each channel in CTR-GCN during training. As a result, every joint not only obtains information from its immediate neighbors but also from all other joints. This prevents the generation of complementary information for the transformer stream.

To explore the impact of positional embedding on the the final performance, we conducted a series of experiments outlined in Table 6. Surprisingly, including temporal positional embedding in addition to spatial positional embedding leads to a 0.6 mm increase in P1 error. As mentioned in Section 3.2, this result stems from the non-permutation equivariant nature of the GCNFormer stream. Unlike transformers, our network inherently maintains the temporal sequence of frames. However, as shown in Figure 5, adding temporal positional embedding leads to a better acceleration error (0.88 mm) compared to using spatial positional embedding (0.94 mm).

Finally, to verify the efficiency of the proposed AG-Former block, Table 7 shows alternative blocks. When using GCNFormer, a P1 error of 57.5 mm is observed, indicating its limited capability to accurately capture the underlying 3D sequence structure. Nevertheless, a hybrid approach involving both GCNFormer and Transformer yields a noteworthy improvement, reducing the P1 error by 5.2 mm compared to using Transformer alone. Furthermore, the sequential fusion of these two modules is not as effective as their parallel integration.

Table 4. The P1 error comparison by varying number of AG-Former blocks and number of channels. d: Number of channels in each AGFormer block. d<sup>′</sup>: Number of channels before regression head. T is kept 243 in all experiments.
<table><tr><td>Layers</td><td> $d$ </td><td> $\overline { { d ^ { \prime } } }$ </td><td>Param</td><td>MACs</td><td>P1</td></tr><tr><td>5</td><td>512</td><td>512</td><td>58M</td><td>252.8 G</td><td>39.2</td></tr><tr><td>4</td><td>256</td><td>512</td><td>11.7 M</td><td>54.G G</td><td>40.2</td></tr><tr><td>6</td><td>256</td><td>512</td><td>17.5 M</td><td>81.64 G</td><td>39.6</td></tr><tr><td>8</td><td>256</td><td>512</td><td>23.3 M</td><td>108.6 G</td><td>39.5</td></tr><tr><td>10</td><td>256</td><td>512</td><td>29.1 M</td><td>135.7 G</td><td>38.6</td></tr><tr><td>12</td><td>256</td><td>512</td><td>34.9 M</td><td>162.7 G</td><td>38.4</td></tr><tr><td>14</td><td>256</td><td>512</td><td>40.7 M</td><td>189.7 G</td><td>38.6</td></tr><tr><td>16</td><td>128</td><td>512</td><td>11.7 M</td><td>64.7 G</td><td>38.4</td></tr><tr><td>16</td><td>128</td><td>256</td><td>11.7 M</td><td>64.5 G</td><td>38.7</td></tr><tr><td>16</td><td>128</td><td>1024</td><td>11.7 M</td><td>64.9 G</td><td>39.2</td></tr><tr><td>26</td><td>64</td><td>512</td><td>4.8M</td><td>39.5 G</td><td>39.5</td></tr></table>

Table 5. The P1 error comparison when using alternative modules in graph stream of AGFormer block.
<table><tr><td>GCN</td><td>TCN</td><td>GCNFormer</td><td>CTR-GCNFormer</td><td>P1</td></tr><tr><td>√</td><td>-</td><td>1</td><td>1</td><td>39.1</td></tr><tr><td>√</td><td>√</td><td>-</td><td>1</td><td>39.3</td></tr><tr><td>1</td><td>√</td><td>√</td><td>-</td><td>39.7</td></tr><tr><td>-</td><td>1</td><td>-</td><td>√</td><td>40.9</td></tr><tr><td>-</td><td>一</td><td>√</td><td>-</td><td>38.4</td></tr></table>

## 4.6. Qualitative Analysis

Validation of MotionAGFormer-B is conducted using the adjacency matrix of temporal GCNFormer and visualization of 3D human pose estimation. The instances for validation are randomly chosen from the evaluation set of Human3.6M.

![](images/420be680ed62a311c53a6fb02f6447b988fdea8c55d9b19cd20eafb5a4837c4d.jpg)  
Figure 5. Acceleration error when using different type of positional embedding. Using temporal positional embedding leads to 0.88 mm error and a faster convergence compared to spatial positional embedding, without embedding, and both embeddings that reach 0.94 mm, 0.90 mm, and 0.97 mm respectively.

Qualitative Comparisons. Figure 4 compares MotionAGFormer-B with recent approaches including STCFormer [40], PoseFormerV2 [50], and Motion-BERT [53]. By design, confidence scores of the 2D detector are only included into MotionBERT and MotionAGFormer-B. Overall, MotionAGFormer-B shows better reconstruction results across all three samples than PoseFormerV2 and STCFormer, while maintaining competitive performance with MotionBERT. Specifically, in Figure 4a, MotionAG-Former exhibits improved alignment with the ground truth in comparison to alternative approaches. In Figure 4b, it displays a slightly superior alignment, whereas in Figure 4c, its alignment is marginally less optimal than that of Motion-BERT.

Temporal adjacency visualization. The adjacency matrix of temporal GCNFormer is visualized in Figure 6. In the initial layers, individual joints exhibit distinct adjacency matrices, which should vary across different sequences due to the changing joint positions over time. Nevertheless, as we progress through the model’s depth, it appears to learn a representation where each joint is most akin to its own state in neighboring frames. Consequently, this leads to connections being formed with adjacent frames.

## 5. Conclusion

We introduced MotionAGFormer, a novel approach that leverages GCNFormer to capture intricate local joint relationships, and combines it with Transformer that effectively captures global joint interdependencies. This fusion enhances the model’s ability to comprehend the inherent 3D structure within input 2D sequences. Additionally, Motion-AGFormer offers various adaptable variants, allowing the selection of an optimal balance between speed and accuracy. Empirical evaluations show that our method surpasses alternative methods on Human3.6M and MPI-INF-3DHP.

![](images/24cd5bacd6136fe50c16c84bf76d35db6d70d7341d9bc742046482955abf04c9.jpg)  
Figure 6. Temporal adjacency matrix of a random sequence on three different joints, from the Human3.6M dataset, at the first layer (left), middle layer (center), and last layer (right). K in K-NN is set to 2.

Table 6. The P1 error comparison when using different positional embedding.
<table><tr><td>Temporal Embedding</td><td>Spatial Embedding</td><td>P1</td></tr><tr><td>1</td><td>-</td><td>39.3</td></tr><tr><td>-</td><td>√</td><td>38.4</td></tr><tr><td>√</td><td>-</td><td>38.9</td></tr><tr><td>√</td><td>√</td><td>40.5</td></tr></table>

Table 7. Comparison of different MetaFormer integration. All the models are trained on Human3.6M with our MotionAGFormer-B settings.
<table><tr><td>Method</td><td>P1</td></tr><tr><td>GCNFormer only</td><td>57.5</td></tr><tr><td>Transformer only</td><td>43.6</td></tr><tr><td>GCNFormer → Transformer (Sequential)</td><td>39.1</td></tr><tr><td>Transformer → GCNFormer (Sequential)</td><td>38.9</td></tr><tr><td>Transformer - GCNFormer (Parallel)</td><td>38.4</td></tr></table>

## References

[1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016. 3

[2] Peter Bauer, Arij Bouazizi, Ulrich Kressel, and Fabian B Flohr. Weakly supervised multi-modal 3d human body pose estimation for autonomous driving. In 2023 IEEE Intelligent Vehicles Symposium (IV), pages 1–7. IEEE, 2023. 1

[3] Hanyuan Chen, Jun-Yan He, Wangmeng Xiang, Wei Liu, Zhi-Qi Cheng, Hanbing Liu, Bin Luo, Yifeng Geng, and Xuansong Xie. HDFormer: High-order directed transformer for 3d human pose estimation. arXiv preprint arXiv:2302.01825, 2023. 2, 6, 7, 12

[4] Yilun Chen, Zhicheng Wang, Yuxiang Peng, Zhiqiang Zhang, Gang Yu, and Jian Sun. Cascaded pyramid network for multi-person pose estimation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7103–7112, 2018. 2

[5] Yuxin Chen, Ziqi Zhang, Chunfeng Yuan, Bing Li, Ying Deng, and Weiming Hu. Channel-wise topology refinement graph convolution for skeleton-based action recognition. In Proceedings of the IEEE/CVF international conference on computer vision, pages 13359–13368, 2021. 3, 7

[6] Hongsuk Choi, Gyeongsik Moon, and Kyoung Mu Lee. Pose2Mesh: Graph convolutional network for 3d human pose and mesh recovery from a 2d human pose. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part VII 16, pages 769–787. Springer, 2020. 3

[7] Sungho Chun, Sungbum Park, and Ju Yong Chang. Learnable human mesh triangulation for 3d human pose and shape estimation. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision, pages 2850–2859, 2023. 2

[8] Hai Ci, Chunyu Wang, Xiaoxuan Ma, and Yizhou Wang. Optimizing network structure for 3d human pose estimation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2262–2271, 2019. 3

[9] Phillip Czech, Markus Braun, Ulrich Kreßel, and Bin Yang. On-board pedestrian trajectory prediction using behavioral features. In 2022 21st IEEE International Conference on Machine Learning and Applications (ICMLA), pages 437– 443. IEEE, 2022. 1

[10] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 2

[11] Moritz Einfalt, Katja Ludwig, and Rainer Lienhart. Uplift and upsample: Efficient 3d human pose estimation with uplifting transformers. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), January 2023. 2, 6, 7, 12

[12] Lin Geng Foo, Tianjiao Li, Hossein Rahmani, Qiuhong Ke, and Jun Liu. Unified pose sequence modeling. In Proceed-

ings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13019–13030, 2023. 2, 6, 12

[13] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International conference on machine learning, pages 448–456. pmlr, 2015. 3, 4

[14] Catalin Ionescu, Dragos Papava, Vlad Olaru, and Cristian Sminchisescu. Human3. 6m: Large scale datasets and predictive methods for 3d human sensing in natural environments. IEEE transactions on pattern analysis and machine intelligence, 36(7):1325–1339, 2013. 1, 5

[15] Karim Iskakov, Egor Burkov, Victor Lempitsky, and Yury Malkov. Learnable triangulation of human pose. In Proceedings of the IEEE/CVF international conference on computer vision, pages 7718–7727, 2019. 2

[16] Hongbo Kang, Yong Wang, Mengyuan Liu, Doudou Wu, Peng Liu, and Wenming Yang. Double-chain constraints for 3d human pose estimation in images and videos. arXiv preprint arXiv:2308.05298, 2023. 3, 6

[17] Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907, 2016. 3

[18] Jungho Lee, Minhyeok Lee, Dogyoon Lee, and Sangyoun Lee. Hierarchically decomposed graph convolutional networks for skeleton-based action recognition. arXiv preprint arXiv:2208.10741, 2022. 3

[19] Wenhao Li, Hong Liu, Runwei Ding, Mengyuan Liu, Pichao Wang, and Wenming Yang. Exploiting temporal contexts with strided transformer for 3d human pose estimation. IEEE Transactions on Multimedia, 25:1282–1293, 2022. 2, 6, 12

[20] Wenhao Li, Hong Liu, Hao Tang, Pichao Wang, and Luc Van Gool. MHFormer: Multi-hypothesis transformer for 3d human pose estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13147–13156, 2022. 2, 6, 7, 12

[21] Huei-Yung Lin and Ting-Wen Chen. Augmented reality with human body interaction based on monocular 3d pose estimation. In International Conference on Advanced Conceptsfor Intelligent Vision Systems, pages 321–331. Springer, 2010. 1

[22] Ziyu Liu, Hongwen Zhang, Zhenghao Chen, Zhiyong Wang, and Wanli Ouyang. Disentangling and unifying graph convolutions for skeleton-based action recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 143–152, 2020. 3

[23] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 5

[24] Cheng Luo, Siyang Song, Weicheng Xie, Linlin Shen, and Hatice Gunes. Learning multi-dimensional edge featurebased au relation graph for facial action unit recognition. arXiv preprint arXiv:2205.01782, 2022. 4

[25] Dushyant Mehta, Helge Rhodin, Dan Casas, Pascal Fua, Oleksandr Sotnychenko, Weipeng Xu, and Christian Theobalt. Monocular 3d human pose estimation in the wild using improved cnn supervision. In 2017 international conference on 3D vision (3DV), pages 506–516. IEEE, 2017. 5

[26] Dushyant Mehta, Srinath Sridhar, Oleksandr Sotnychenko, Helge Rhodin, Mohammad Shafiei, Hans-Peter Seidel,

Weipeng Xu, Dan Casas, and Christian Theobalt. VNect: Real-time 3d human pose estimation with a single rgb camera. Acm transactions on graphics (tog), 36(4):1–14, 2017. 1

[27] Tewodros Legesse Munea, Yalew Zelalem Jembre, Halefom Tekle Weldegebriel, Longbiao Chen, Chenxi Huang, and Chenhui Yang. The progress of human pose estimation: A survey and taxonomy of models applied in 2d human pose estimation. IEEE Access, 8:133330–133348, 2020. 1

[28] Alejandro Newell, Kaiyu Yang, and Jia Deng. Stacked hourglass networks for human pose estimation. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part VIII 14, pages 483–499. Springer, 2016. 2

[29] Alejandro Newell, Kaiyu Yang, and Jia Deng. Stacked hourglass networks for human pose estimation. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part VIII 14, pages 483–499. Springer, 2016. 5

[30] Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in pytorch. 2017. 5

[31] Georgios Pavlakos, Xiaowei Zhou, and Kostas Daniilidis. Ordinal depth supervision for 3d human pose estimation. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 7307–7316, 2018. 2

[32] Georgios Pavlakos, Xiaowei Zhou, Konstantinos G Derpanis, and Kostas Daniilidis. Coarse-to-fine volumetric prediction for single-image 3d human pose. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7025–7034, 2017. 2

[33] Xiaoye Qian, Youbao Tang, Ning Zhang, Mei Han, Jing Xiao, Ming-Chun Huang, and Ruei-Sung Lin. HSTFormer: Hierarchical spatial-temporal transformers for 3d human pose estimation. arXiv preprint arXiv:2301.07322, 2023. 2, 6, 7, 12

[34] N Dinesh Reddy, Laurent Guigues, Leonid Pishchulin, Jayan Eledath, and Srinivasa G Narasimhan. TesseTrack: End-toend learnable multi-person articulated 3d pose tracking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15190–15200, 2021. 2

[35] Edoardo Remelli, Shangchen Han, Sina Honari, Pascal Fua, and Robert Wang. Lightweight multi-view 3d pose estimation through camera-disentangled representation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6040–6049, 2020. 2

[36] Wenkang Shan, Zhenhua Liu, Xinfeng Zhang, Shanshe Wang, Siwei Ma, and Wen Gao. P-STMO: Pre-trained spatial temporal many-to-one model for 3d human pose estimation. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part V, pages 461–478. Springer, 2022. 2, 6, 7, 12

[37] Weijie Su, Xizhou Zhu, Chenxin Tao, Lewei Lu, Bin Li, Gao Huang, Yu Qiao, Xiaogang Wang, Jie Zhou, and Jifeng Dai. Towards all-in-one pre-training via maximizing multi-modal mutual information. In Proceedings of the IEEE/CVF Con-

ference on Computer Vision and Pattern Recognition, pages 15888–15899, 2023. 2

[38] Ke Sun, Bin Xiao, Dong Liu, and Jingdong Wang. Deep high-resolution representation learning for human pose estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5693–5703, 2019. 2, 6, 12

[39] Xiao Sun, Bin Xiao, Fangyin Wei, Shuang Liang, and Yichen Wei. Integral human pose regression. In Proceedings of the European conference on computer vision (ECCV), pages 529–545, 2018. 2

[40] Zhenhua Tang, Zhaofan Qiu, Yanbin Hao, Richang Hong, and Ting Yao. 3D human pose estimation with spatiotemporal criss-cross attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4790–4799, 2023. 2, 5, 6, 7, 8, 12, 13

[41] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 1, 2, 3

[42] Jingbo Wang, Sijie Yan, Yuanjun Xiong, and Dahua Lin. Motion guided 3d pose estimation from videos. In European Conference on Computer Vision, pages 764–780. Springer, 2020. 3

[43] Julian Wiederer, Arij Bouazizi, Ulrich Kressel, and Vasileios Belagiannis. Traffic control gesture recognition for autonomous vehicles. In 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 10676–10683. IEEE, 2020. 1

[44] Bruce X. B. Yu, Zhi Zhang, Yongxu Liu, Sheng hua Zhong, Yan Liu, and Chang Wen Chen. GLA-GCN: Global-local adaptive graph convolutional network for 3d human pose estimation from monocular video, 2023. 3, 6, 7, 12

[45] Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. CoCa: Contrastive captioners are image-text foundation models. arXiv preprint arXiv:2205.01917, 2022. 2

[46] Weihao Yu, Mi Luo, Pan Zhou, Chenyang Si, Yichen Zhou, Xinchao Wang, Jiashi Feng, and Shuicheng Yan. MetaFormer is actually what you need for vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10819–10829, 2022. 3

[47] Jinlu Zhang, Zhigang Tu, Jianyu Yang, Yujin Chen, and Junsong Yuan. MixSTE: Seq2seq mixed spatio-temporal encoder for 3d human pose estimation in video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13232–13242, June 2022. 2, 6, 7, 12

[48] Zhe Zhang, Chunyu Wang, Weichao Qiu, Wenhu Qin, and Wenjun Zeng. Adafuse: Adaptive multiview fusion for accurate human pose estimation in the wild. International Journal ofComputer Vision, 129:703–718, 2021. 2

[49] Long Zhao, Xi Peng, Yu Tian, Mubbasir Kapadia, and Dimitris N Metaxas. Semantic graph convolutional networks for 3d human pose regression. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3425–3435, 2019. 3

![](images/96a6acffdd1502a2b8907a8d2f85cabf6914ff0c6a07caab8e18da79254d06a7.jpg)  
Figure 7. The per-joint error comparisons in terms of MPJPE (mm) with other models on Human3.6M dataset. ’L’ and ’R’ denote left and right, respectively.

[50] Qitao Zhao, Ce Zheng, Mengyuan Liu, Pichao Wang, and Chen Chen. PoseFormerV2: Exploring frequency domain for efficient and robust 3d human pose estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8877–8886, June 2023. 2, 5, 6, 7, 8, 13

[51] Ce Zheng, Sijie Zhu, Matias Mendieta, Taojiannan Yang, Chen Chen, and Zhengming Ding. 3d human pose estimation with spatial and temporal transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11656–11665, 2021. 2

[52] Kun Zhou, Xiaoguang Han, Nianjuan Jiang, Kui Jia, and Jiangbo Lu. HEMlets pose: Learning part-centric heatmap triplets for accurate 3d human pose estimation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2344–2353, 2019. 2

[53] Wentao Zhu, Xiaoxuan Ma, Zhaoyang Liu, Libin Liu, Wayne Wu, and Yizhou Wang. MotionBERT: A unified perspective on learning human motion representations. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, 2023. 2, 4, 5, 6, 8, 12, 13

[54] Zhuofan Zong, Guanglu Song, and Yu Liu. DETRs with collaborative hybrid assignments training. arXiv preprint arXiv:2211.12860, 2022. 2

## Appendix

## A. Additional Results on Human3.6M

## A.1. Per-action Result

To assess the effectiveness of our proposed methods for each action, Table 8 provides a comparison of P1 and P2 errors between our method and alternative methods on the Human3.6M dataset. Our proposed methods demonstrate superior performance in terms of P1 error across various actions such as Direction, Discuss, Eating, Greet, Purchase, Sitting, Sitting Down, Smoke, Wait, and Walk Dog, when compared to other existing methods. For the remaining actions, our methods deliver the second-best results.

![](images/84dceb1b06aec64a89443daa9c8586762ea1e33b69acc7560ad844432cdc75f9.jpg)  
Figure 8. Comparison of acceleration error among different alternative modules designed for capturing local temporal dependencies.

## A.2. Per-joint Error Comparison

Figure 7 shows per-joint comparisons of MotionAGFormer-B, MotionBERT, STCFormer, and StridedFormer on Human3.6M benchmark dataset. STC-Former demonstrates a slight performance advantage on hips, whereas for shoulders, MotionAGFormer exhibits a slight superiority over MotionBERT and significantly outperforms other models. MotionBERT generally outperforms MotionAGFormer on joints with a degree of freedom (DoF) of 2, including knees and elbows, with a notable gap between them and other models. However, for joints with a DoF of 3, such as ankles and wrists, MotionAGFormer consistently outperforms all other models, which is particularly important due to their greater contribution to overall error and their significance in applications like hand/object interaction and gait analysis.

## B. Ablation on Temporal GCNFormer

In the main manuscript, we examined various alternatives for the temporal GCNFormer and reported their corresponding P1 errors. Nevertheless, we have yet to investigate the acceleration error associated with each of these alternatives. The acceleration error is defined as follows:

$$
\begin{array} { r l r } & { } & { A c c ( { \bf P } , \hat { \bf P } ) = \Sigma _ { t = 1 } ^ { T - 1 } L ( { \bf P } , t ) - L ( \hat { \bf P } , t ) , } \\ & { } & { L ( { \bf X } , t ) = { \bf X } _ { t - 1 } - 2 { \bf X } _ { t } + { \bf X } _ { t + 1 } , \quad } \end{array}\tag{10}
$$

where P and $\hat { \mathbf { P } }$ denote the ground truth 3D pose and estimated 3D pose, respectively.

Figure 8 illustrates the acceleration error of different models when used to capture temporal local dependencies. Interestingly, despite CTR-GCN and TCN resulting in higher P1 error (as mentioned in the main manuscript), their acceleration error convergence during training is faster. In addition, it leads to an eventual acceleration error improvement of approximately 0.1 mm.

Table 8. Quantitative comparisons of 3D human pose estimation per action on Human3.6M. (Top) MPJPE (mm) using detected 2D pose sequence. (Bottom) P-MPJPE (mm) using detected 2D pose sequence. (\*) denotes using HRNet [38] for 2D pose estimation. (†) denotes manually evaluated using their provided evaluation code.
<table><tr><td>MPJPE</td><td>T</td><td>Dire.</td><td>Disc.</td><td>Eat</td><td>Greet</td><td>Phone</td><td>Photo</td><td>Pose</td><td>Purch.</td><td>Sit</td><td>SitD</td><td>Smoke</td><td>Wait</td><td>WalkD</td><td>Walk</td><td>WalkT</td><td>Avg</td></tr><tr><td>*MHFormer [20]</td><td>351</td><td>39.2</td><td>43.1</td><td>40.1</td><td>40.9</td><td>44.9</td><td>51.2</td><td>40.6</td><td>41.3</td><td>53.5</td><td>60.3</td><td>43.7</td><td>41.1</td><td>43.8</td><td>29.8</td><td>30.6</td><td>43.0</td></tr><tr><td>MixSTE [47]</td><td>243</td><td>37.6</td><td>40.9</td><td>37.3</td><td>39.7</td><td>42.3</td><td>49.9</td><td>40.1</td><td>39.8</td><td>51.7</td><td>55.0</td><td>42.1</td><td>39.8</td><td>41.0</td><td>27.9</td><td>27.9</td><td>40.9</td></tr><tr><td>P-STMO [36]</td><td>243</td><td>38.9</td><td>42.7</td><td>40.4</td><td>41.1</td><td>45.6</td><td>49.7</td><td>40.9</td><td>39.9</td><td>55.5</td><td>59.4</td><td>44.9</td><td>42.2</td><td>42.7</td><td>29.4</td><td>29.4</td><td>42.8</td></tr><tr><td>StridedFormer [19]</td><td>351</td><td>40.3</td><td>43.3</td><td>40.2</td><td>42.3</td><td>45.6</td><td>52.3</td><td>41.8</td><td>40.5</td><td>55.9</td><td>60.6</td><td>44.2</td><td>43.0</td><td>44.2</td><td>30.0</td><td>30.2</td><td>43.7</td></tr><tr><td>Einfalt et al. [11]</td><td>351</td><td>39.6</td><td>43.8</td><td>40.2</td><td>42.4</td><td>46.5</td><td>53.9</td><td>42.3</td><td>42.5</td><td>55.7</td><td>62.3</td><td>45.1</td><td>43.0</td><td>44.7</td><td>30.1</td><td>30.8</td><td>44.2</td></tr><tr><td>STCFormer [40]</td><td>243</td><td>39.6</td><td>41.6</td><td>37.4</td><td>38.8</td><td>43.1</td><td>51.1</td><td>39.1</td><td>39.7</td><td>51.4</td><td>57.4</td><td>41.8</td><td>38.5</td><td>40.7</td><td>27.1</td><td>28.6</td><td>41.0</td></tr><tr><td>STCFormer-L [40]</td><td>243</td><td>38.4</td><td>41.2</td><td>36.8</td><td>38.0</td><td>42.7</td><td>50.5</td><td>38.7</td><td>38.2</td><td>52.5</td><td>56.8</td><td>41.8</td><td>38.4</td><td>40.2</td><td>26.2</td><td>27.7</td><td>40.5</td></tr><tr><td>UPS [12]</td><td>243</td><td>37.5</td><td>39.2</td><td>36.9</td><td>40.6</td><td>39.3</td><td>46.8</td><td>39.0</td><td>41.7</td><td>50.6</td><td>63.5</td><td>40.4</td><td>37.8</td><td>44.2</td><td>26.7</td><td>29.1</td><td>40.8</td></tr><tr><td>GLA-GCN [44]</td><td>243</td><td>41.3</td><td>44.3</td><td>40.8</td><td>41.8</td><td>45.9</td><td>54.1</td><td>42.1</td><td>41.5</td><td>57.8</td><td>62.9</td><td>45.0</td><td>42.8</td><td>45.9</td><td>29.4</td><td>29.9</td><td>44.4</td></tr><tr><td>† MotionBERT [53]</td><td>243</td><td>36.6</td><td>39.3</td><td>37.8</td><td>33.5</td><td>41.4</td><td>49.9</td><td>37.0</td><td>35.5</td><td>50.4</td><td>56.5</td><td>41.4</td><td>38.2</td><td>37.3</td><td>26.2</td><td>26.9</td><td>39.2</td></tr><tr><td>HDFormer [3]</td><td>96</td><td>38.1</td><td>43.1</td><td>39.3</td><td>39.4</td><td>44.3</td><td>49.1</td><td>41.3</td><td>40.8</td><td>53.1</td><td>62.1</td><td>43.3</td><td>41.8</td><td>43.1</td><td>31.0</td><td>29.7</td><td>42.6</td></tr><tr><td>HSTFormer [33]</td><td>81</td><td>39.5</td><td>42.0</td><td>39.9</td><td>40.8</td><td>44.4</td><td>50.9</td><td>40.9</td><td>41.3</td><td>54.7</td><td>58.8</td><td>43.6</td><td>40.7</td><td>43.4</td><td>30.1</td><td>30.4</td><td>42.7</td></tr><tr><td>MotionAGFormer-XS</td><td>27</td><td>42.7</td><td>46.7</td><td>41.1</td><td>39.4</td><td>47.0</td><td>56.3</td><td>44.4</td><td>41.0</td><td>52.4</td><td>66.3</td><td>46.7</td><td>43.2</td><td>43.4</td><td>32.4</td><td>34.0</td><td>45.1</td></tr><tr><td>MotionAGFormer-S</td><td>81</td><td>41.9</td><td>42.7</td><td>40.4</td><td>37.6</td><td>45.6</td><td>51.3</td><td>41.0</td><td>38.0</td><td>54.1</td><td>58.8</td><td>45.5</td><td>40.4</td><td>39.8</td><td>29.4</td><td>31.0</td><td>42.5</td></tr><tr><td>MotionAGFormer-B</td><td>243</td><td>36.4</td><td>38.4</td><td>36.8</td><td>32.9</td><td>40.9</td><td>48.5</td><td>36.6</td><td>34.6</td><td>51.7</td><td>52.8</td><td>41.0</td><td>36.4</td><td>36.5</td><td>26.7</td><td>27.0</td><td>38.4</td></tr><tr><td>MotionAGFormer-L</td><td>243</td><td>36.8</td><td>38.5</td><td>35.9</td><td>33.0</td><td>41.1</td><td>48.6</td><td>38.0</td><td>34.8</td><td>49.0</td><td>51.4</td><td>40.3</td><td>37.4</td><td>36.3</td><td>27.2</td><td>27.2</td><td>38.4</td></tr><tr><td>P-MPJPE</td><td>T</td><td>Dire.</td><td>Disc.</td><td>Eat</td><td>Greet</td><td>Phone</td><td>Photo</td><td>Pose</td><td>Purch.</td><td>Sit</td><td>SitD</td><td>Smoke</td><td>Wait</td><td>WalkD</td><td>Walk</td><td>WalkT</td><td>Avg</td></tr><tr><td>*MHFormer [20]</td><td>351</td><td>31.5</td><td>34.9</td><td>32.8</td><td>33.6</td><td>35.3</td><td>39.6</td><td>32.0</td><td>32.2</td><td>43.5</td><td>48.7</td><td>36.4</td><td>32.6</td><td>34.3</td><td>23.9</td><td>25.1</td><td>34.4</td></tr><tr><td>MixSTE [47]</td><td>243</td><td>30.8</td><td>33.1</td><td>30.3</td><td>31.8</td><td>33.1</td><td>39.1</td><td>31.1</td><td>30.5</td><td>42.5</td><td>44.5</td><td>34.0</td><td>30.8</td><td>32.7</td><td>22.1</td><td>22.9</td><td>32.6</td></tr><tr><td>P-STMO [36]</td><td>243</td><td>31.3</td><td>35.2</td><td>32.9</td><td>33.9</td><td>35.4</td><td>39.3</td><td>32.5</td><td>31.5</td><td>44.6</td><td>48.2</td><td>36.3</td><td>32.9</td><td>34.4</td><td>23.8</td><td>23.9</td><td>34.4</td></tr><tr><td>StridedFormer [19]</td><td>351</td><td>32.7</td><td>35.5</td><td>32.5</td><td>35.4</td><td>35.9</td><td>41.6</td><td>33.0</td><td>31.9</td><td>45.1</td><td>50.1</td><td>36.3</td><td>33.5</td><td>35.1</td><td>23.9</td><td>25.0</td><td>35.2</td></tr><tr><td>Einfalt et al. [11]</td><td>351</td><td>32.7</td><td>36.1</td><td>33.4</td><td>36.0</td><td>36.1</td><td>42.0</td><td>33.3</td><td>33.1</td><td>45.4</td><td>50.7</td><td>37.0</td><td>34.1</td><td>35.9</td><td>24.4</td><td>25.4</td><td>35.7</td></tr><tr><td>STCFormer [40]</td><td>243</td><td>29.5</td><td>33.2</td><td>30.6</td><td>31.0</td><td>33.0</td><td>38.0</td><td>30.4</td><td>29.4</td><td>41.8</td><td>45.2</td><td>33.6</td><td>29.5</td><td>31.6</td><td>21.3</td><td>22.6</td><td>32.0 31.8</td></tr><tr><td>STCFormer-L [40] UPS [12]</td><td>243 243</td><td>29.3 30.3</td><td>33.0 32.2</td><td>30.7 30.8</td><td>30.6 33.1</td><td>32.7 31.1</td><td>38.2 35.2</td><td>29.7 30.3</td><td>28.8 32.1</td><td>42.2 39.4</td><td>45.0 49.6</td><td>33.3 32.9</td><td>29.4 29.2</td><td>31.5 33.9</td><td>20.9 21.6</td><td>22.3 24.5</td></table>

## C. Qualitative comparison for a sequence

Apart from making qualitative comparisons using a random frame from the Human3.6M dataset, we also analyzed consecutive frames within a single sequence. This allows us to observe the behavior of the compared models over a sequence, rather than qualitatively evaluating them solely based on a single frame. Figure 9 compares MotionAGFormer-B with STCFormer, PoseFormerV2, and MotionBERT. Overall, MotionAGFormer and Motion-BERT demonstrate superior alignment with the ground truth compared to the other two models. When comparing MotionAGFormer and MotionBERT, MotionAGFormer exhibits slightly better performance on joints with a DoF, such as wrists and ankles, as opposed to MotionBERT. There are also challenging sequences (see Figure 10) that all models fail to accurately estimate the joints.

## D. In-the-wild Evaluation

To verify the generalization of the proposed MotionAG-Former, we tested our base variant on some unseen in-thewild videos. Our estimations show that even in scenarios where the inferred 2D poses exhibit some noise (as observed in the final frame of the second example shown in Figure 11), the corresponding 3D pose estimations remain reasonable and resilient against sudden noisy movements in the 2D sequence. Moreover, when the subject is positioned at a distance from the camera, the 3D pose estimates exhibit a high degree of accuracy (as demonstrated in the last example of Figure 11).

![](images/83f0974fc3244db8329c6f16a5a2b560994de9439be2172a80bdf6b8d8b57f4e.jpg)

Time

MotionAGFormer

Figure 9. Qualitative comparisons of MotionAGFormer with STCformer [40], PoseFormerV2 [50], and MotionBERT [53] on a random sequence from Human3.6M dataset.  
![](images/34207cb83284243909e9ea726c50d5fe9f5f177a5ab66f0df97c480fdffb990e.jpg)

![](images/67c4cfd92ef4e6ecda93683c98aa4afd12cd3a738b82017029d82563f4df4e67.jpg)

Time

![](images/733e8be5afb0def96804cb7252e73dd69fe90c8d0aa6dce479f569ca12f9838e.jpg)

![](images/1c53222141fa42783b69e75eaeee5fae8ed766f795aa3cc1f5aaac34cb9fe8b7.jpg)

![](images/4f76a42cd6620371844905b04ca56d5b85c4ee5e0c9063b26e5842d440cb0991.jpg)

STCFormer

![](images/7188d9780303debc205803f7f029a2ebaf65bf2743a9d5b6fd9f05aa446758ba.jpg)

![](images/578a8d6bdd788d35ba9583813b190480d0925901b41cfbd983abab349f4bb4f6.jpg)

![](images/496a12d146a0dc75f27f0346d502908076317699cd2183e28cf1edee7407533e.jpg)

![](images/9201da1d3190d2541836d87f3b9afc1401be80fa965dc3ff0f4e51f213cb54bd.jpg)

![](images/60209afdd7ed4d51e6adb9ed859fc6ddddd977a76485bb3b321268b458529b39.jpg)

PoseFormerV2

![](images/fe36c2fd8788e8f25c1fcbf72d665340592b9fca9cf6d42100947c75102ef21b.jpg)

![](images/6c60374bd68c2a08378814abc05a38d3ddb488b8ef126f8b75708120a62eb9c4.jpg)

![](images/b8bf9d2a300a3353d69b8187ddb8c81030f35d0db8528031c0189986c4d02918.jpg)

![](images/5e14566f45efb9fd4bf12cebf04a106d61ff0f0d7e9c5e2f7261ea742b7e66a9.jpg)

![](images/93e50d575d32932210de653943a923c510e664becf76c6ff3a968208703dce9f.jpg)

![](images/436e37fa9b844db1c88d9364bcd65dbaa3402b1172fe6be215bcba26970a7ab7.jpg)

MotionBERT

![](images/e4491a93da569732848482c4e54637c1e52d954678c220b5522caf9fdf14235f.jpg)

![](images/410489a2a26876287e3c503bed3d2914d0e32b46867c1df43e4cab8e3900072c.jpg)

![](images/2e04026dd08848d08ca6eb67752159e0e5e150f3bcd92645c781179484de422f.jpg)

![](images/4fa37ff4d7a960a839f4f2e4c20c68b03d49e760a1d647d5efae4e1ac5c31ed7.jpg)  
Figure 10. Failure case of the models on a random sequence from Human3.6M dataset.

![](images/a5b8956f4315c16fafe0ed6c5ce86543103a29342d2ae079f3b13caa92fd911c.jpg)  
Figure 11. In-the-wild video evaluation of MotionAGFormer. Estimations are based on our base variant.