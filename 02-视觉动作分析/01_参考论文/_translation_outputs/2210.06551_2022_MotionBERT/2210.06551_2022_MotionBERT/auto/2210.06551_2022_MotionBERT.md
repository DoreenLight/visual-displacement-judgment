# MotionBERT: A Unified Perspective on Learning Human Motion Representations

Wentao Zhu<sup>1</sup> Xiaoxuan Ma<sup>1</sup> Zhaoyang Liu<sup>2</sup> Libin Liu<sup>1</sup> Wayne Wu<sup>2</sup> Yizhou Wang<sup>1†</sup>

<sup>1</sup> Peking University <sup>2</sup> Shanghai AI Laboratory

{wtzhu,maxiaoxuan,libin.liu,yizhou.wang}@pku.edu.cn

{zyliumy,wuwenyan0503}@gmail.com

## Abstract

We present a unified perspective on tackling various human-centric video tasks by learning human motion representations from large-scale and heterogeneous data resources. Specifically, we propose a pretraining stage in which a motion encoder is trained to recover the underlying 3D motion from noisy partial 2D observations. The motion representations acquired in this way incorporate geometric, kinematic, and physical knowledge about human motion, which can be easily transferred to multiple downstream tasks. We implement the motion encoder with a Dual-stream Spatio-temporal Transformer (DSTformer) neural network. It could capture long-range spatio-temporal relationships among the skeletal joints comprehensively and adaptively, exemplified by the lowest 3D pose estimation error so far when trainedfrom scratch. Furthermore, our proposedframework achieves state-of-the-art performance on all three downstream tasks by simplyfinetuning the pretrained motion encoder with a simple regression head (1-2 layers), which demonstrates the versatility of the learned motion representations. Code and models are available at https://motionbert.github.io/

## 1. Introduction

Perceiving and understanding human activities have long been a core pursuit of machine intelligence. To this end, researchers define various tasks to estimate human-centric semantic labels from videos, e.g. skeleton keypoints [14, 35], action classes [64, 123], and surface meshes [46, 71]. While significant progress has been made in each of these tasks, they tend to be modeled in isolation, rather than as interconnected problems. For example, Spatial Temporal Graph Convolutional Networks (ST-GCN) have been applied to modeling spatio-temporal relationship of human joints in both 3D pose estimation [13, 116] and action recognition [96, 123], but their connections have not been fully explored. Intuitively, these models should all have learned to identify typical human motion patterns, despite being designed for different problems. Nonetheless, current methods fail to mine and utilize such commonalities across the tasks. Ideally, we could develop a unified human-centric video representation that can be shared across all relevant tasks.

![](images/2667e0cd5fae1cacf4ff392a8452bf486a1df647537aa2697f1697dd8b78c350.jpg)  
Figure 1. Framework overview. We utilize a motion encoder to learn human motion representations via recovering 3D human motion from corrupted 2D skeleton sequences. To adapt to different downstream tasks, we finetune the pretrained motion representations with a linear layer or a simple MLP.

One significant challenge to developing such a representation is the heterogeneity of available data resources. Motion capture (Mocap) systems [38, 76] provide high-fidelity 3D motion data obtained with markers and sensors, but the appearances of captured videos are usually constrained to simple indoor scenes. Action recognition datasets provide annotations of the action semantics, but they either contain no human pose labels [16, 95] or feature limited motion of daily activities [63, 64, 93]. In contrast, in-the-wild human videos offer a vast and diverse range of appearance and motion. However, obtaining precise 2D pose annotations requires considerable effort [3], and acquiring ground-truth (GT) 3D joint locations is almost impossible. Consequently, most existing studies focus on a specific task using a single type of human motion data, and they are not able to enjoy the advantages of other data resources.

In this work, we provide a new perspective on learning human motion representations. The key idea is that we can learn a versatile human motion representation from heterogeneous data resources in a unified manner, and utilize the representation to handle different downstream tasks in a unified way. We present a two-stage framework, consisting of pretraining and finetuning, as depicted in Figure 1. In the pretraining stage, we extract 2D skeleton sequences from diverse motion data sources and corrupt them with ran dom masks and noises. Subsequently, we train the motion encoder to recover the 3D motion from the corrupted 2D skeletons. This challenging pretext task intrinsically requires the motion encoder to i) infer the underlying 3D human structures from its temporal movements; ii) recover the erroneous and missing observations. In this way, the motion encoder implicitly captures human motion commonsense such as joint linkages, anatomical constraints, and temporal dynamics. In practice, we propose Dual-stream Spatio-temporal Transformer (DSTformer) as the motion encoder to capture the long-range relationship among skeleton keypoints. We suppose that the motion representations learned from largescale and diversified data resources could be shared across different downstream tasks and benefit their performance. Therefore, for each downstream task, we adapt the pretrained motion representations using task-specific training data and supervisory signals with a simple regression head.

In summary, the contributions of this work are three-fold: 1) We provide a new perspective on solving various humancentric video tasks through a shared framework of learning human motion representations. 2) We propose a pretraining method to leverage the large-scale yet heterogeneous human motion resources and learn generalizable human motion representations. Our approach could take advantage of the precision of 3D mocap data and the diversity of in-the-wild RGB videos at the same time. 3) We design a dual-stream Transformer network with cascaded spatio-temporal selfattention blocks that could serve as a general backbone for human motion modeling. The experiments demonstrate that the above designs enable a versatile human motion representation that can be transferred to multiple downstream tasks, outperforming the task-specific state-of-the-art methods.

## 2. Related Work

Learning Human Motion Representations. Early works formulate human motion with Hidden Markov Models [53, 108] and graphical models [51, 99]. Kanazawa et al. [42] design a temporal encoder and a hallucinator to learn representations of 3D human dynamics. Zhang et al. [132] predict future 3D dynamics in a self-supervised manner. Sun et al. [102] further incorporate action labels with an action memory bank. From the action recognition perspective, a variety of pretext tasks are designed to learn motion representations in a self-supervised manner, including future prediction [100], jigsaw puzzle [60], skeletoncontrastive [107], speed change [101], cross-view consistency [62], and contrast-reconstruction [117]. Similar techniques are also explored in tasks like motion assessment [33,85] and motion retargeting [126,139]. These methods leverage homogeneous motion data, design corresponding pretext tasks, and apply them to a specific downstream task. In this work, we propose a unified pretrain-finetune framework to incorporate heterogeneous data resources and demonstrate its versatility in various downstream tasks.

3D Human Pose Estimation. Recovering 3D human poses from monocular RGB videos is a classical problem, and the methods can be categorized into two categories. The first is to estimate 3D poses with CNN directly from images [82, 104, 136]. However, one limitation of these approaches is that there is a trade-off between 3D pose precision and appearance diversity due to current data collection techniques. The second category is to extract the 2D pose first, then lift the estimated 2D pose to 3D with a separate neural network. The lifting can be achieved via Fully Connected Network [29, 78], Temporal Convolutional Network (TCN) [22, 89], GCN [13, 28, 116], and Transformer [56, 94, 134, 135]. Our framework is built upon the second category as we use the proposed DSTformer to accomplish 2D-to-3D lifting.

Skeleton-based Action Recognition. The pioneering works [74, 115, 127] point out the inherent connection between action recognition and human pose estimation. Towards modeling the spatio-temporal relationship among human joints, previous studies mainly employ LSTM [98, 138] and GCN [21, 55, 68, 96, 123]. Most recently, PoseConv3D [32] proposes to apply 3D-CNN on the stacked 2D joint heatmaps and achieves improved results. In addition to the fully-supervised action recognition task, NTU-RGB+D-120 [64] brings attention to the challenging one-shot action recognition problem. To this end, SL-DML [81] applies deep metric learning to multi-modal signals. Sabater et al. [92] explores one-shot recognition in therapy scenarios with TCN. We demonstrate that the pretrained motion representations could generalize well to action recognition tasks, and the pretrain-finetune framework is a suitable solution for the one-shot challenges.

Human Mesh Recovery. Based on the parametric human models such as SMPL [71], many research works [41, 75, 83, 122, 133] focus on regressing the human mesh from a single image. SPIN [48] additionally incorporates fitting the body

![](images/060922f440e32944b5b4bd9338d8b222b83916ea3ffc1a6455f2e4ad46247f92.jpg)  
Figure 2. Model architecture. We propose the Dual-stream Spatio-temporal Transformer (DSTformer) as a general backbone for human motion modeling. DSTformer consists of N dual-stream-fusion modules. Each module contains two branches of spatial or temporal MHSA and MLP. The Spatial MHSA models the connection among different joints within a timestep, while the Temporal MHSA models the movement of one joint.

## 3.2. Network Architecture

model to 2D joints in the training loop. Despite their promising per-frame results, these methods yield jittery and unstable results [46,130] when applied to videos. To improve their temporal coherence, PoseBERT [8] and SmoothNet [130] propose to employ a denoising and smoothing module to the single-frame predictions. Several works [24,42,46,106] take video clips as input to exploit the temporal cues. Another common problem is that paired images and GT meshes are mostly captured in constrained scenarios, which limits the generalization ability of the above methods. To that end, Pose2Mesh [25] proposes to first extract 2D skeletons using an off-the-shelf pose estimator, then lift them to 3D mesh vertices. Our approach is complementary to state-of-the-art human mesh recovery methods and could further improve their temporal coherence with the pretrained motion representations.

## 3. Method

## 3.1. Overview

As discussed in Section 1, our approach consists of two stages, namely unified pretraining and task-specific finetuning. In the first stage, we train a motion encoder to accomplish the 2D-to-3D lifting task, where we use the proposed DSTformer as the backbone. In the second stage, we finetune the pretrained motion encoder and a few new layers on the downstream tasks. We use 2D skeleton sequences as input for both pretraining and finetuning because they could be reliably extracted from all kinds of motion sources [3, 10, 76, 86, 103], and is more robust to variations [19, 32]. Existing studies have shown the effectiveness of using 2D skeleton sequences for different downstream tasks [25, 32, 89, 109]. We will first introduce the architec ture of DSTformer, and then describe the training scheme in detail.

Figure 2 shows the network architecture for 2D-to-3D lifting. Given an input 2D skeleton sequence $\mathbf { x } \in \mathbb { R } ^ { T \times J \times C _ { \mathrm { i n } } }$ we first project it to a high-dimensional feature ${ \bf F } ^ { 0 } \in  { }$ $\mathbb { R } ^ { T \times J \times C _ { 1 } }$ <sup>f</sup>, then add learnable spatial positional encoding $\mathbf { P } _ { \mathrm { p o s } } ^ { \mathrm { S } } \in \mathbb { R } ^ { 1 \times J \times C _ { \mathrm { f } } }$ and temporal positional encoding $\mathbf { P } _ { \mathrm { p o s } } ^ { \mathrm { T } } \bar { \in }$ $\mathbb { R } ^ { \sum _ { T \times 1 \times C _ { \mathrm { f } } } }$ to it. We then use the sequence-to-sequence model DSTformer to calculate $\mathbf { F } ^ { i } \in \bar { \mathbb { R } } ^ { T \times J \times C _ { \mathrm { f } } } ( i \mathbf { \theta } = 1 , \dots , N )$ where N is the network depth. We apply a linear layer with tanh activation [30] to $\mathbf { F } ^ { N }$ to compute the motion representation $\mathbf { E } \in \mathbb { R } ^ { \bar { T } \times J \times C _ { \mathrm { e } } }$ . Finally, we apply a linear transformation to E to estimate 3D motion $\hat { \hat { \mathbf { X } } } \in \mathbb { R } ^ { T \times J \times C _ { \mathrm { o u t } } }$ <sup>t</sup> . Here, $T$ denotes the sequence length, and J denotes the number of body joints. $C _ { \mathrm { i n } } , C _ { \mathrm { f } } , C _ { \mathrm { e } } ,$ , and $C _ { \mathrm { { o u t } } }$ denote the channel numbers of input, feature, embedding, and output respectively. We first introduce the basic building blocks of DSTformer, i.e. Spatial and Temporal Blocks with Multi-Head Self-Attention (MHSA), and then explain the DSTformer architecture design.

Spatial Block. Spatial MHSA (S-MHSA) aims at modeling the relationship among the joints within the same time step. It is defined as

$$
\begin{array} { r } { \mathrm { S } \mathrm { - } \mathbf { M } \mathrm { H S A } ( \mathbf { Q } _ { \mathrm { S } } , \mathbf { K } _ { \mathrm { S } } , \mathbf { V } _ { \mathrm { S } } ) = [ \mathrm { h e a d } _ { 1 } ; \dots ; \mathrm { h e a d } _ { h } ] \mathbf { W } _ { \mathrm { S } } ^ { P } , } \\ { \mathrm { h e a d } _ { i } = \mathrm { s o f t m a x } ( \frac { \mathbf { Q } _ { \mathrm { S } } ^ { i } ( \mathbf { K } _ { \mathrm { S } } ^ { i } ) ^ { \prime } } { \sqrt { d _ { K } } } ) \mathbf { V } _ { \mathrm { S } } ^ { i } , } \end{array}\tag{1}
$$

where $\mathbf { W } _ { \mathrm { S } } ^ { P }$ is a projection parameter matrix, h is the number of the heads, $i \in { 1 , \dots , h } ,$ , and <sup>′</sup> denotes matrix transpose. We utilize self-attention to get the query $\mathbf { Q } _ { \mathrm { S } }$ , key $\mathbf { K } _ { \mathrm { S } } .$ , and value ${ \bf V } _ { \mathrm { S } }$ from input per-frame spatial feature $\mathbf { F } _ { \mathrm { S } } ^ { \mathrm { } } \in \mathbb { R } ^ { J \times C _ { \mathrm { e } } }$ for each head ,

$$
\mathbf { Q } _ { \mathrm { s } } ^ { i } = \mathbf { F } _ { \mathrm { s } } \mathbf { W } _ { \mathrm { s } } ^ { ( Q , i ) } , \mathbf { K } _ { \mathrm { s } } ^ { i } = \mathbf { F } _ { \mathrm { s } } \mathbf { W } _ { \mathrm { s } } ^ { ( K , i ) } , \mathbf { V } _ { \mathrm { s } } ^ { i } = \mathbf { F } _ { \mathrm { s } } \mathbf { W } _ { \mathrm { s } } ^ { ( V , i ) } ,\tag{2}
$$

where $\mathbf { W } _ { \mathrm { S } } ^ { ( Q , i ) } , \mathbf { W } _ { \mathrm { S } } ^ { ( K , i ) } , \mathbf { W } _ { \mathrm { S } } ^ { ( V , i ) }$ are projection matrices, and $d _ { K }$ is the feature dimension of $\mathbf { K } _ { \mathrm { S } }$ . We apply S-MHSA to features of different time steps in parallel. Residual connection and layer normalization (LayerNorm) are used to the S-MHSA result, which is further fed into a multilayer perceptron (MLP), and followed by a residual connection and LayerNorm following [112]. We denote the entire spatial block with MHSA, LayerNorm, MLP, and residual connections by .

Temporal Block. Temporal MHSA (T-MHSA) aims at modeling the relationship across the time steps for a body joint. Its computation process is similar with S-MHSA except that the MHSA is applied to the per-joint temporal feature $\mathbf { F } _ { \mathrm { T } } \in \mathbb { R } ^ { T \times C _ { \mathrm { e } } }$ and parallelized over the spatial dimension.

$$
\begin{array} { r } { \mathrm { T } \mathbf { \cdot } \mathbf { M } \mathrm { H S A } ( \mathbf { Q } _ { \mathrm { T } } , \mathbf { K } _ { \mathrm { T } } , \mathbf { V } _ { \mathrm { T } } ) = [ \mathrm { h e a d } _ { 1 } ; \dots ; \mathrm { h e a d } _ { h } ] \mathbf { W } _ { \mathrm { T } } ^ { P } , } \\ { \mathrm { h e a d } _ { i } = \operatorname { s o f t m a x } ( \frac { \mathbf { Q } _ { \mathrm { T } } ^ { i } ( \mathbf { K } _ { \mathrm { T } } ^ { i } ) ^ { \prime } } { \sqrt { d _ { K } } } ) \mathbf { V } _ { \mathrm { T } } ^ { i } , } \end{array}\tag{3}
$$

where $i \in { 1 , \ldots , h , \mathbf { Q } _ { \mathrm { { T } } } , \mathbf { K } _ { \mathrm { { T } } } , \mathbf { V } _ { \mathrm { { T } } } }$ are computed similar with Formula 2. We denote the entire temporal block by $\tau$

Dual-stream Spatio-temporal Transformer. Given spatial and temporal MHSA that captures the intra-frame and inter-frame body joint interactions respectively, we assemble the basic building blocks to fuse the spatial and temporal information in the flow. We design a dual-stream architec ture with the following assumptions: 1) Both streams should be capable of modeling the comprehensive spatio-temporal context. 2) Each stream should be specialized in different spatio-temporal aspects. 3) The two streams should be fused together, with the fusion weights dynamically balanced de pending on the input spatio-temporal characteristics.

Hence, we stack the spatial and temporal MHSA blocks in different orders, forming two parallel computation branches. The output features of the two branches are fused using adaptive weights predicted by an attention regressor. The dual-stream-fusion module is then repeated for N times:

$$
\mathbf { F } ^ { i } = \alpha _ { \mathrm { S T } } ^ { i } \circ \mathcal { T } _ { 1 } ^ { i } ( S _ { 1 } ^ { i } ( \mathbf { F } ^ { i - 1 } ) ) + \alpha _ { \mathrm { T S } } ^ { i } \circ S _ { 2 } ^ { i } ( \mathcal { T } _ { 2 } ^ { i } ( \mathbf { F } ^ { i - 1 } ) ) , \quad i \in 1 , \dots , N ,\tag{4}
$$

where $\mathbf { F } ^ { i }$ denotes the feature embedding at depth $i ,$ denotes element-wise production. Orders of and $\tau$ blocks are shown in Figure $^ { 2 , }$ and different blocks do not share weights. Adaptive fusion weights α<sub>ST</sub>, $\alpha _ { \mathrm { T S } } \in \mathbb { R } ^ { N \times T \times J }$ are given by

$$
\alpha _ { \mathrm { S T } } ^ { i } , \alpha _ { \mathrm { T S } } ^ { i } = \mathrm { s o f t m a x } ( \mathcal { W } ( [ \mathcal { T } _ { 1 } ^ { i } ( S _ { 1 } ^ { i } ( \mathbf { F } ^ { i - 1 } ) ) , S _ { 2 } ^ { i } ( \mathcal { T } _ { 2 } ^ { i } ( \mathbf { F } ^ { i - 1 } ) ) ] ) ) ,\tag{5}
$$

where  is a learnable linear transformation. [, ] denotes concatenation.

## 3.3. Unified Pretraining

We address two key challenges when designing the unified pretraining framework: 1) How to learn a powerful motion representation with a universal pretext task. 2) How to utilize large-scale but heterogeneous human motion data in all kinds of formats.

For the first challenge, we follow the successful practices in language [12, 30, 90] and vision [7, 36] modeling to construct the supervision signals, i.e. mask part of the input and use the encoded representations to reconstruct the whole input. Note that such “cloze” task naturally exists in human motion analysis, that is to recover the lost depth information from the 2D visual observations, i.e. 3D human pose estimation. Inspired by this, we leverage the large-scale 3D mocap data [76] and design a 2D-to-3D lifting pretext task. We first extract the 2D skeleton sequences x by projecting the 3D motion orthographically. Then, we corrupt x by randomly masking and adding noise to produce the corrupted 2D skeleton sequences, which also resemble the 2D detection results as it contains occlusions, detection failures, and errors. Both joint-level and frame-level masks are applied with certain probabilities. We use the aforementioned motion encoder to get motion representation E and reconstruct 3D motion Xˆ . We then compute the joint loss $\mathcal { L } _ { 3 \mathrm { D } }$ between $\hat { \mathbf X }$ and GT 3D motion X. We also add the velocity loss $\mathcal { L } _ { 0 }$ following previous works [89, 134]. The 3D reconstruction losses are thus given by

$$
\mathcal { L } _ { \mathrm { 3 D } } = \sum _ { t = 1 } ^ { T } \sum _ { j = 1 } ^ { J } \parallel \hat { \mathbf { X } } _ { t , j } - \mathbf { X } _ { t , j } \parallel _ { 2 } , \quad \mathcal { L } _ { 0 } = \sum _ { t = 2 } ^ { T } \sum _ { j = 1 } ^ { J } \parallel \hat { \mathbf { O } } _ { t , j } - \mathbf { O } _ { t , j } \parallel _ { 2 } ,
$$

where $\begin{array} { r } { \hat { \mathbf { O } } _ { t } = \hat { \mathbf { X } } _ { t } - \hat { \mathbf { X } } _ { t - 1 } , \mathbf { O } _ { t } = \mathbf { X } _ { t } - \mathbf { X } _ { t - 1 } . } \end{array}$

(6)

For the second challenge, we notice that 2D skeletons could serve as a universal medium as they can be extracted from all sorts of motion data sources. We further incorporate in-the-wild RGB videos into the 2D-to-3D lifting framework for unified pretraining. For RGB videos, the 2D skeletons x could be given by manual annotation [3] or 2D pose estimators [14, 103], and the depth channel of the extracted 2D skeletons is intrinsically “masked”. Similarly, we add extra masks and noises to degrade x (if x already contains detection noise, only masking is applied). As 3D motion GT X is not available for these data, we apply a weighted 2D re-projection loss which is calculated by

$$
\mathcal { L } _ { \mathrm { 2 D } } = \sum _ { t = 1 } ^ { T } \sum _ { j = 1 } ^ { J } \delta _ { t , j } \| \hat { \mathbf { x } } _ { t , j } - \mathbf { x } _ { t , j } \| _ { 2 } ,\tag{7}
$$

where $\hat { \bf x }$ is the 2D orthographical projection of the estimated 3D motion $\hat { \bf X }$ , and $\pmb { \delta } \in \mathbf { \bar { \mathbb { R } } } ^ { \bar { T } \times J }$ is given by visibility annotation or 2D detection confidence.

The total pretraining loss is computed by

$$
\small \mathcal { L } = \underbrace { \mathcal { L } _ { \mathrm { 3 D } } + \lambda _ { 0 } \mathcal { L } _ { 0 } } _ { \mathrm { f o r 3 D \ : d a t a } } + \underbrace { \mathcal { L } _ { \mathrm { 2 D } } } _ { \mathrm { f o r 2 D \ : d a t a } } ,\tag{8}
$$

where $\lambda _ { 0 }$ is a constant coefficient to balance the losses.

## 3.4. Task-specific Finetuning

The learned feature embedding E serves as a 3D-aware and temporal-aware human motion representation. For downstream tasks, we adopt the minimalist design principle, i.e. implementing a shallow downstream network and training without bells and whistles. In practice, we use an extra linear layer or an MLP with one hidden layer. We then finetune the whole network end-to-end.

3D Pose Estimation. As we utilize 2D-to-3D lifting as the pretext task, we simply reuse the whole pretrained network. During finetuning, the input 2D skeletons are estimated from videos without extra masks or noises.

Skeleton-based Action Recognition. We directly apply a global average pooling over different persons and timesteps. The result is then fed into an MLP with one hidden layer. The network is trained with cross-entropy classification loss. For one-shot learning, we apply a linear layer after the pooled features to extract clip-level action representation. We introduce the detailed setup of one-shot learning in Section 4.4.

Human Mesh Recovery. We use SMPL [71] model to represent the human mesh and regress its parameters. The SMPL model consists of pose parameters $\theta \ \in \ \mathbb { R } ^ { 7 2 }$ and shape parameters $\beta \in \mathbb { R } ^ { 1 0 }$ , and calculates the 3D mesh as $\mathcal { M } ( \boldsymbol { \theta } , \bar { \boldsymbol { \beta } } ) \in \mathbb { R } ^ { 6 8 9 0 \times 3 }$ . To regress the pose parameters for each frame, we feed the motion embeddings E to an MLP with one hidden layer and get $\pmb { \hat { \theta } } \in \mathbb { R } ^ { T \times \breve { 7 } 2 }$ . To estimate shape parameters, considering that the human shape over a video sequence is supposed to be consistent, we first perform an average pooling of E over the temporal dimension and then feed it into another MLP to regress a single $\hat { \beta }$ and then expand it to the entire sequence as $\hat { \beta } \in \mathbb { R } ^ { \bar { T } \times 1 0 }$ . The shape MLP has the same architecture as the pose regression one, and they are initialized with the mean shape and pose, respectively, as in [46]. The overall loss is computed as

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { 3 \mathrm { { D } } } ^ { \mathrm { { m } } } \mathcal { L } _ { 3 \mathrm { { D } } } ^ { \mathrm { { m } } } + \lambda _ { \theta } \mathcal { L } _ { \theta } + \lambda _ { \beta } \mathcal { L } _ { \beta } + \lambda _ { \mathrm { { n } } } \mathcal { L } _ { \mathrm { { n o r m } } } + \lambda _ { 0 } ^ { \mathrm { { m } } } \mathcal { L } _ { 0 } ^ { \mathrm { { m } } } , } \end{array}\tag{9}
$$

where each term is calculated as

$$
\begin{array} { r l } & { \mathcal { L } _ { 3 \mathrm { D } } ^ { \mathrm { m } } = \| \hat { \mathbf { X } } ^ { \mathrm { m } } - \mathbf { X } ^ { \mathrm { m } } \| _ { 1 } , \quad \mathcal { L } _ { \boldsymbol { \theta } } = \| \hat { \pmb { \theta } } - \pmb { \theta } \| _ { 1 } , \quad \mathcal { L } _ { \boldsymbol { \beta } } = \| \hat { \pmb { \beta } } - \pmb { \beta } \| _ { 1 } , } \\ & { \mathcal { L } _ { \mathrm { n o r m } } = \| \hat { \pmb { \theta } } \| _ { 2 } + \| \hat { \pmb { \beta } } \| _ { 2 } , \quad \mathcal { L } _ { 0 } ^ { \mathrm { m } } = \| \hat { \mathbf { O } } ^ { \mathrm { m } } - \mathbf { O } ^ { \mathrm { m } } \| _ { 2 } . } \end{array}\tag{10}
$$

Note that each 3D pose in motion $\mathbf { X } ^ { \mathrm { m } }$ at frame t is regressed from mesh vertices by $\mathbf { X } _ { t } ^ { \mathrm { m } } = \mathbf { J } \mathcal { M } ( \theta _ { t } , \beta _ { t } )$ , where J R<sup>J×6890</sup> is a pre-defined matrix [10]. ${ \bf O } ^ { \mathrm { m } } = { \bf X } _ { t + 1 } ^ { \mathrm { m } } - { \bf X } _ { t } ^ { \mathrm { m } }$ $\hat { \mathbf { O } } ^ { \mathrm { m } } = \hat { \mathbf { X } } _ { t + 1 } ^ { \mathrm { m } } - \hat { \mathbf { X } } _ { t } ^ { \mathrm { m } } . \lambda _ { 3 \mathrm { D } } ^ { \mathrm { m } } , \lambda _ { \theta } , \lambda _ { \beta } , \lambda _ { \mathrm { n } }$ and $\lambda _ { 0 } ^ { \mathrm { m } }$ are constant coefficients to balance the training loss.

## 4. Experiments

## 4.1. Implementation

We implement the proposed motion encoder DSTformer with depth $N = 5 .$ , number of heads $h = 8 ,$ , feature size $C _ { \mathrm { f } } = 5 1 2$ , embedding size $C _ { \mathrm { e } } = 5 1 2$ . For pretraining, we use sequence length $T = 2 4 3$ . The pretrained model could handle different input lengths thanks to the Transformerbased backbone. During finetuning, we set the backbone learning rate to be 0.1 of the new layer learning rate. We introduce the experiment datasets in the following sections respectively. Please refer to the appendix for more experimental details.

## 4.2. Pretraining

We collect diverse and realistic 3D human motion from two datasets, Human3.6M [38] and AMASS [76]. Human3.6M [38] is a commonly used indoor dataset for 3D human pose estimation which contains 3.6 million video frames of professional actors performing daily actions. Following previous works [78, 89], we use subjects 1, 5, 6, 7, 8 for training, and subjects 9, 11 for testing. AMASS [76] integrates most existing marker-based Mocap datasets [1,2,5,11,15,18,34,37,52,69,72,77,84,97,110,111] and parameterizes them with a common representation. We do not use the images or 2D detection results of the two datasets during pretraining as Mocap datasets usually do not provide raw videos. Instead, we use orthographic projection to get the uncorrupted 2D skeletons. We further incorporate two in-the-wild RGB video datasets PoseTrack [3] (annotated) and InstaVariety [42] (unannotated) for higher motion diversity. We align the body keypoint definitions with Human3.6M and calibrate the camera coordinates to pixel coordinates following [27]. We randomly zero out 15% joints, and sample noises from a mixture of Gaussian and uniform distributions [17]. We first train on 3D data only for 30 epochs, then train on both 3D data and 2D data for 60 epochs, following the curriculum learning practices [9, 118].

## 4.3. 3D Pose Estimation

We evaluate the 3D pose estimation performance on Human3.6M [38] and report the mean per joint position error (MPJPE) in millimeters, which measures the average distance between the predicted joint positions and the GT after aligning the root joint. We also compute the mean per-joint velocity error (MPJVE) to evalute the temporal smoothness following previous works [134, 135]. We use the Stacked Hourglass (SH) networks [86] to extract the 2D skeletons from videos, and finetune the entire network on Human3.6M [38] training set. In addition, we train a separate model of the same architecture, but with random initialization rather than pretrained weights. As shown in Table 1 (top), the model trained from scratch outperforms previous methods including other Transformer-based designs with spatio-temporal modeling. It shows the effectiveness of the proposed DSTformer in terms of learning 3D geometric structures and temporal dynamics. To further evaluate the upper bound of the models’ capability, we compare the performance when using 2D GT pose sequences as input, which gets rid of the influence of different 2D detectors. As shown in Table 1 (middle), our models significantly outperform all the previous approaches. Table 1 (bottom) shows that both of our models also surpass previous works in terms of MPJVE, implying better temporal coherence. We attribute the performance advantage of our scratch model to the proposed DSTformer design. We include more comparisons and analysis to demonstrate the advantage of DSTformer with regard to other spatio-temporal architectures in Section 4.6 and supplementary materials. Additionally, our method achieves lower errors with the proposed pretraining stage.

<table><tr><td>Method</td><td>T</td><td>Dire.</td><td>Disc.</td><td>Eat</td><td>Greet</td><td>Phone</td><td>Photo</td><td>Pose</td><td>Purch.</td><td>Sit</td><td>SitD</td><td>Smoke</td><td>Wait</td><td>WalkD</td><td>Walk</td><td>WalkT</td><td>Avg</td></tr><tr><td>Martinez et al. [78] ICCV&#x27;17</td><td>1</td><td>51.8</td><td>56.2</td><td>58.1</td><td>59.0</td><td>69.5</td><td>78.4</td><td>55.2</td><td>58.1</td><td>74.0</td><td>94.6</td><td>62.3</td><td>59.1</td><td>65.1</td><td>49.5</td><td>52.4</td><td>62.9</td></tr><tr><td>Pavlakos et al. [88] CVPR’18</td><td>1</td><td>48.5</td><td>54.4</td><td>54.4</td><td>52.0</td><td>59.4</td><td>65.3</td><td>49.9</td><td>52.9</td><td>65.8</td><td>71.1</td><td>56.6</td><td>52.9</td><td>60.9</td><td>44.7</td><td>47.8</td><td>56.2</td></tr><tr><td>LCN [28] ICCV&#x27;19</td><td>1</td><td>46.8</td><td>52.3</td><td>44.7</td><td>50.4</td><td>52.9</td><td>68.9</td><td>49.6</td><td>46.4</td><td>60.2</td><td>78.9</td><td>51.2</td><td>50.0</td><td>54.8</td><td>40.4</td><td>43.3</td><td>52.7</td></tr><tr><td>Xu et al. [121] CVPR’21</td><td>1</td><td>45.2</td><td>49.9</td><td>47.5</td><td>50.9</td><td>54.9</td><td>66.1</td><td>48.5</td><td>46.3</td><td>59.7</td><td>71.5</td><td>51.4</td><td>48.6</td><td>53.9</td><td>39.9</td><td>44.1</td><td>51.9</td></tr><tr><td>VideoPose3D [89] CVPR&#x27;19</td><td>243</td><td>45.2</td><td>46.7</td><td>43.3</td><td>45.6</td><td>48.1</td><td>55.1</td><td>44.6</td><td>44.3</td><td>57.3</td><td>65.8</td><td>47.1</td><td>44.0</td><td>49.0</td><td>32.8</td><td>33.9</td><td>46.8</td></tr><tr><td>Cai et al. [13] ICCV’19</td><td>7</td><td>44.6</td><td>47.4</td><td>45.6</td><td>48.8</td><td>50.8</td><td>59.0</td><td>47.2</td><td>43.9</td><td>57.9</td><td>61.9</td><td>49.7</td><td>46.6</td><td>51.3</td><td>37.1</td><td>39.4</td><td>48.8</td></tr><tr><td>Yeh et al. [129] NeurIPS’19</td><td>243</td><td>44.8</td><td>46.1</td><td>43.3</td><td>46.4</td><td>49.0</td><td>55.2</td><td>44.6</td><td>44.0</td><td>58.3</td><td>62.7</td><td>47.1</td><td>43.9</td><td>48.6</td><td>32.7</td><td>33.3</td><td>46.7</td></tr><tr><td>Liu et al. [67] CVPR’20</td><td>243</td><td>41.8</td><td>44.8</td><td>41.1</td><td>44.9</td><td>47.4</td><td>54.1</td><td>43.4</td><td>42.2</td><td>56.2</td><td>63.6</td><td>45.3</td><td>43.5</td><td>45.3</td><td>31.3</td><td>32.2</td><td>45.1</td></tr><tr><td>* Cheng et al. [22] AAAI&#x27;20</td><td>128</td><td>36.2</td><td>38.1</td><td>42.7</td><td>35.9</td><td>38.2</td><td>45.7</td><td>36.8</td><td>42.0</td><td>45.9</td><td>51.3</td><td>41.8</td><td>41.5</td><td>43.8</td><td>33.1</td><td>28.6</td><td>40.1</td></tr><tr><td>* UGCN [116] ECCV’20</td><td>96</td><td>38.2</td><td>41.0</td><td>45.9</td><td>39.7</td><td>41.4</td><td>51.4</td><td>41.6</td><td>41.4</td><td>52.0</td><td>57.4</td><td>41.8</td><td>44.4</td><td>41.6</td><td>33.1</td><td>30.0</td><td>42.6</td></tr><tr><td>† PoseFormer [135] ICCV’21</td><td>81</td><td>41.5</td><td>44.8</td><td>39.8</td><td>42.5</td><td>46.5</td><td>51.6</td><td>42.1</td><td>42.0</td><td>53.3</td><td>60.7</td><td>45.5</td><td>43.3</td><td>46.1</td><td>31.8</td><td>32.2</td><td>44.3</td></tr><tr><td>* Wehrbein et al. [119] ICCV&#x27;21</td><td>200</td><td>38.5</td><td>42.5</td><td>39.9</td><td>41.7</td><td>46.5</td><td>51.6</td><td>39.9</td><td>40.8</td><td>49.5</td><td>56.8</td><td>45.3</td><td>46.4</td><td>46.8</td><td>37.8</td><td>40.4</td><td>44.3</td></tr><tr><td>† MHFormer [56] CVPR’22</td><td>351</td><td>39.2</td><td>43.1</td><td>40.1</td><td>40.9</td><td>44.9</td><td>51.2</td><td>40.6</td><td>41.3</td><td>53.5</td><td>60.3</td><td>43.7</td><td>41.1</td><td>43.8</td><td>29.8</td><td>30.6</td><td>43.0</td></tr><tr><td>*† MixSTE [134] CVPR&#x27;22</td><td>243</td><td>36.7</td><td>39.0</td><td>36.5</td><td>39.4</td><td>40.2</td><td>44.9</td><td>39.8</td><td>36.9</td><td>47.9</td><td>54.8</td><td>39.6</td><td>37.8</td><td>39.3</td><td>29.7</td><td>30.6</td><td>39.8</td></tr><tr><td>† P-STMO [94] ECCV’22</td><td>243</td><td>38.4</td><td>42.1</td><td>39.8</td><td>40.2</td><td>45.2</td><td>48.9</td><td>40.4</td><td>38.3</td><td>53.8</td><td>57.3</td><td>43.9</td><td>41.6</td><td>42.2</td><td>29.3</td><td>29.3</td><td>42.1</td></tr><tr><td>† Ours (scratch)</td><td>243</td><td>36.3</td><td>38.7</td><td>38.6</td><td>33.6</td><td>42.1</td><td>50.1</td><td>36.2</td><td>35.7</td><td>50.1</td><td>56.6</td><td>41.3</td><td>37.4</td><td>37.7</td><td>25.6</td><td>26.5</td><td>39.2</td></tr><tr><td>† Ours (finetune)</td><td>243</td><td>36.1</td><td>37.5</td><td>35.8</td><td>32.1</td><td>40.3</td><td>46.3</td><td>36.1</td><td>35.3</td><td>46.9</td><td>53.9</td><td>39.5</td><td>36.3</td><td>35.8</td><td>25.1</td><td>25.3</td><td>37.5</td></tr><tr><td>Method</td><td>T</td><td>Dire.</td><td>Disc.</td><td>Eat</td><td>Greet</td><td>Phone</td><td>Photo</td><td>Pose</td><td>Purch.</td><td>Sit</td><td>SitD</td><td>Smoke</td><td>Wait</td><td>WalkD</td><td>Walk</td><td>WalkT</td><td>Avg</td></tr><tr><td>Martinez et al. [78] ICCV&#x27;17</td><td>1</td><td>37.7</td><td>44.4</td><td>40.3</td><td>42.1</td><td>48.2</td><td>54.9</td><td>44.4</td><td>42.1</td><td>54.6</td><td>58.0</td><td>45.1</td><td>46.4</td><td>47.6</td><td>36.4</td><td>40.4</td><td>45.5</td></tr><tr><td>LCN [28] ICCV’19</td><td>1</td><td>36.3</td><td>38.8</td><td>29.7</td><td>37.8</td><td>34.6</td><td>42.5</td><td>39.8</td><td>32.5</td><td>36.2</td><td>39.5</td><td>34.4</td><td>38.4</td><td>38.2</td><td>31.3</td><td>34.2</td><td>36.3</td></tr><tr><td>Xu et al. [121] CVPR’21</td><td>1</td><td>35.8</td><td>38.1</td><td>31.0</td><td>35.3</td><td>35.8</td><td>43.2</td><td>37.3</td><td>31.7</td><td>38.4</td><td>45.5</td><td>35.4</td><td>36.7</td><td>36.8</td><td>27.9</td><td>30.7</td><td>35.8</td></tr><tr><td>UGCN [116] ECCV&#x27;20</td><td>96</td><td>23.0</td><td>25.7</td><td>22.8</td><td>22.6</td><td>24.1</td><td>30.6</td><td>24.9</td><td>24.5</td><td>31.1</td><td>35.0</td><td>25.6</td><td>24.3</td><td>25.1</td><td>19.8</td><td>18.4</td><td>25.6</td></tr><tr><td>† PoseFormer [135] ICCV’21</td><td>81</td><td>30.0</td><td>33.6</td><td>29.9</td><td>31.0</td><td>30.2</td><td>33.3</td><td>34.8</td><td>31.4</td><td>37.8</td><td>38.6</td><td>31.7</td><td>31.5</td><td>29.0</td><td>23.3</td><td>23.1</td><td>31.3 30.5</td></tr><tr><td>† MHFormer [56] CVPR’22 † MixSTE [134] CVPR’22</td><td>351</td><td>27.7 21.6</td><td>32.1 22.0</td><td>29.1 20.4</td><td>28.9 21.0</td><td>30.0 20.8</td><td>33.9 24.3</td><td>33.0 24.7</td><td>31.2 21.9</td><td>37.0 26.9</td><td>39.3 24.9</td><td>30.0 21.2</td><td>31.0 21.5</td><td>29.4 20.8</td><td>22.2 14.7</td><td>23.0 15.6</td></table>

Table 1. Quantitative comparison of 3D human pose estimation on Human3.6M. (Top) MPJPE (mm) using detected 2D pose sequences. (Middle) MPJPE (mm) using GT 2D pose sequences. (Bottom) MPJVE (mm) using detected 2D pose sequences. T denotes the clip length used by the method. We select the best results reported by each work. <sup>∗</sup> denotes using HRNet [103] for 2D detection. <sup>†</sup> denotes implemented with a spatio-temporal Transformer design. The best and second-best results are highlighted in bold and underlined formats.

## 4.4. Skeleton-based Action Recognition

We further explore the possibility to learn action semantics with the pretrained human motion representations. We use the human action dataset NTU-RGB+D [93] which contains 57K videos of 60 action classes, and we follow the data splits Cross-subject (X-Sub) and Cross-view (X-View). The dataset has an extended version, NTU-RGB+D-120 [64], which contains 114K videos of 120 action classes. We follow the suggested One-shot action recognition protocol on NTU-RGB+D-120. For both datasets, we use HRNet [103] to extract 2D skeletons following [32]. Similarly, we train a scratch model with random initialization for comparison. As Table 2 (left) shows, our methods are comparable or superior to the state-of-the-art approaches. Notably, the pretraining stage accounts for a large performance gain.

Additionally, we delve into the one-shot setting which holds significant practical importance. Real-world applications often require fine-grained action recognition in specific domains such as education, sports, and healthcare. Unfortunately, the action classes in these scenarios are not typically defined in public datasets. As a result, only limited annotations for these novel action classes are available, making accurate recognition a challenging task. As proposed in [64], we report the results on the evaluation set of 20 novel classes using only 1 labeled video for each class. The auxiliary set contains the other 100 classes, and all samples of these classes can be used. We train the model on the auxiliary set using the supervised contrastive learning technique [44]. For a batch of auxiliary data, samples of the same class are pulled together, while samples of different classes are pushed away in the action embedding space. During the evaluation, we calculate the cosine distance between the test examples and the exemplars, and use 1-nearest neighbor to determine the class. Table 2 (right) illustrates that the proposed models outperform state-of-the-art by a considerable margin. Moreover, it is noteworthy that our pretrained model achieves optimal performance with only 1-2 epochs of fine-tuning. Our results indicate that the pretraining stage is effective in learning a robust motion representation that generalizes well to novel downstream tasks, even with limited data annotations.

<table><tr><td rowspan=1 colspan=4>Method</td><td rowspan=1 colspan=1>X-Sub X-View</td></tr><tr><td rowspan=4 colspan=4>ST-GCN [123] AAAI&#x27;182s-AGCN [96] CVPR’19MS-G3D [68] CVPR’20Shift-GCN [21] CVPR’20</td><td rowspan=1 colspan=1>81.5  88.3</td></tr><tr><td rowspan=1 colspan=1>88.5  95.1</td></tr><tr><td rowspan=1 colspan=1>3D [68] CVPR'20</td><td rowspan=1 colspan=1>91.5  96.2</td></tr><tr><td rowspan=1 colspan=3>hift-GCN [21] CVPR'20</td><td rowspan=1 colspan=1>90.7  96.5</td></tr><tr><td rowspan=1 colspan=1>Cro</td><td></td><td rowspan=1 colspan=2>sSCLR [62] CVPR’21</td><td rowspan=1 colspan=1>86.2  92.5</td></tr><tr><td rowspan=1 colspan=4>MCC (finetune) [101] ICCV&#x27;21</td><td rowspan=1 colspan=1>89.7  96.3</td></tr><tr><td rowspan=1 colspan=4>SCC (finetune) [125] ICCV’21</td><td rowspan=1 colspan=1>88.0  94.9</td></tr><tr><td rowspan=3 colspan=4>UNIK (finetune) [124] BMVC&#x27;21CTR-GCN [20] ICCV’21PoseConv3D [32] CVPR’22</td><td rowspan=1 colspan=1>86.8  94.4</td></tr><tr><td rowspan=1 colspan=1>92.4  96.8</td></tr><tr><td rowspan=1 colspan=1>93.1  95.7</td></tr><tr><td rowspan=2 colspan=4>Ours (scratch)Ours (finetune)</td><td rowspan=1 colspan=1>87.7  94.1</td></tr><tr><td rowspan=1 colspan=1>93.0  97.2</td></tr></table>

<table><tr><td>Method</td><td>Accuracy</td></tr><tr><td>ST-LSTM + AvgPool [65]</td><td>42.9</td></tr><tr><td>ST-LSTM + FC [66]</td><td>42.1</td></tr><tr><td>ST-LSTM + Attention [66]</td><td>41.0</td></tr><tr><td>APSR [64]</td><td>45.3</td></tr><tr><td>TCN OneShot [92]</td><td>46.5</td></tr><tr><td>SL-DML [81]</td><td>50.9</td></tr><tr><td>Skeleton-DML [80]</td><td>54.2</td></tr><tr><td>Ours (scratch)</td><td>61.0</td></tr><tr><td>Ours (finetune)</td><td>67.4</td></tr></table>

Table 2. Quantitative comparison of skeleton-based action recognition accuracy. (Left) Cross-subject and cross-view recognition accuracy on NTU-RGB+D. All the methods are evaluated using only the “joint” modality with 1-clip sampling for the fairness of comparison. (Right) One-shot recognition accuracy on NTU-RGB+D-120. All results are top-1 accuracy (%).
<table><tr><td rowspan="2">Method</td><td rowspan="2">Input</td><td rowspan="2"> $T$ </td><td colspan="3">Human3.6M</td><td colspan="3">3DPW</td></tr><tr><td>MPVE↓</td><td>MPJPE↓</td><td>PA-MPJPE↓</td><td>MPVE↓</td><td>MPJPE↓</td><td>PA-MPJPE↓</td></tr><tr><td>HMR [41] CVPR&#x27;18</td><td>image</td><td>1</td><td></td><td>88.0</td><td>56.8</td><td></td><td>130.0</td><td>81.3</td></tr><tr><td>† SPIN [48] ICCV&#x27;19</td><td>image</td><td>1</td><td>82.3</td><td>59.4</td><td>39.3</td><td>129.1</td><td>100.9</td><td>59.1</td></tr><tr><td>Pose2Mesh [25] ECCV&#x27;20</td><td>2D pose</td><td>1</td><td>85.3</td><td>64.9</td><td>48.7</td><td>109.3</td><td>91.4</td><td>60.1</td></tr><tr><td>I2L-MeshNet [83] ECCV&#x27;20</td><td>image</td><td>1</td><td></td><td>55.7</td><td>41.7</td><td>110.1</td><td>93.2</td><td>58.6</td></tr><tr><td>† HybrIK [54] CVPR&#x27;21</td><td>image</td><td>1</td><td>58.1</td><td>47.4</td><td>30.1</td><td>82.4</td><td>71.3</td><td>41.9</td></tr><tr><td>METRO [58] CVPR’21</td><td>image</td><td>1</td><td></td><td>54.0</td><td>36.7</td><td>88.2</td><td>77.1</td><td>47.9</td></tr><tr><td>Mesh Graphormer [59] ICCV&#x27;21</td><td>image</td><td>1</td><td></td><td>51.2</td><td>34.5</td><td>87.7</td><td>74.7</td><td>45.6</td></tr><tr><td>PARE [47] ICCV’21</td><td>image</td><td>1</td><td></td><td></td><td></td><td>88.6</td><td>74.5</td><td>46.5</td></tr><tr><td>ROMP [105] ICCV’21</td><td>image</td><td>1</td><td></td><td></td><td></td><td>108.3</td><td>91.3</td><td>54.9</td></tr><tr><td>PyMAF [133] ICCV&#x27;21</td><td>image</td><td>1</td><td></td><td>57.7</td><td>40.5</td><td>110.1</td><td>92.8</td><td>58.9</td></tr><tr><td>ProHMR [50] ICCV&#x27;21</td><td>image</td><td>1</td><td></td><td></td><td>41.2</td><td></td><td></td><td>59.8</td></tr><tr><td>OCHMR [43] CVPR&#x27;22</td><td>image</td><td>1</td><td></td><td></td><td></td><td>107.1</td><td>89.7</td><td>58.3</td></tr><tr><td>3DCrowdNet [26] CVPR&#x27;22</td><td>image</td><td>1</td><td></td><td></td><td></td><td>98.3</td><td>81.7</td><td>51.5</td></tr><tr><td>CLIFF [57] ECCV’22</td><td>image</td><td>1</td><td></td><td>47.1</td><td>32.7</td><td>81.2</td><td>69.0</td><td>43.0</td></tr><tr><td>FastMETRO [23] ECCV’22</td><td>image</td><td>1</td><td></td><td>52.2</td><td>33.7</td><td>84.1</td><td>73.5</td><td>44.6</td></tr><tr><td>VisDB [128] ECCV&#x27;22</td><td>image</td><td>1</td><td></td><td>51.0</td><td>34.5</td><td>85.5</td><td>73.5</td><td>44.9</td></tr><tr><td>TemporalContext [6] CVPR&#x27;19</td><td>video</td><td>32</td><td></td><td>77.8</td><td>54.3</td><td>-</td><td></td><td>72.2</td></tr><tr><td>HMMR [42] CVPR&#x27;19</td><td>video</td><td>20</td><td></td><td></td><td>56.9</td><td>139.3</td><td>116.5</td><td>72.6</td></tr><tr><td>DSD-SATN [106] ICCV&#x27;19</td><td>video</td><td>9</td><td></td><td>59.1</td><td>42.4</td><td></td><td></td><td>69.5</td></tr><tr><td>VIBE [46] CVPR’20</td><td>video</td><td>16</td><td></td><td>65.6</td><td>41.4</td><td>99.1</td><td>82.9</td><td>51.9</td></tr><tr><td>TCMR [24] CVPR’21</td><td>video</td><td>16</td><td></td><td>62.3</td><td>41.1</td><td>102.9</td><td>86.5</td><td>52.7</td></tr><tr><td>† MAED [114] ICCV’21</td><td>video</td><td>16</td><td>84.1</td><td>60.4</td><td>38.3</td><td>93.3</td><td>79.0</td><td>45.7</td></tr><tr><td>MPS-Net [120] CVPR&#x27;22</td><td>video</td><td>16</td><td></td><td>69.4</td><td>47.4</td><td>99.7</td><td>84.3</td><td>52.1</td></tr><tr><td>* PoseBERT [8] TPAMI&#x27;22 (+SPIN [48])</td><td>video</td><td>16</td><td></td><td></td><td></td><td>-</td><td></td><td>57.3↓2.3</td></tr><tr><td>* SmoothNet [130] ECCV&#x27;22 (+SPIN [48])</td><td>video</td><td>32</td><td></td><td>67.5↓1.0</td><td>46.3↓0.2</td><td>-</td><td>86.7↓0.9</td><td>52.7↓0.6</td></tr><tr><td>Ours (scratch)</td><td>2D motion</td><td>16</td><td>75.7</td><td>62.8</td><td>41.0</td><td>99.1</td><td>85.5</td><td>50.2</td></tr><tr><td>Ours (finetune)</td><td>2D motion</td><td>16</td><td>65.5</td><td>53.8</td><td>34.9</td><td>88.1</td><td>76.9</td><td>47.2</td></tr><tr><td>Ours (finetune) + SPIN [48]</td><td>video</td><td>16</td><td>63.7↓18.6</td><td> $5 2 . 2 \downarrow 7 . 2$ </td><td>35.7↓3.6</td><td>92.8 ↓36.3</td><td>79.6↓21.3</td><td>48.2↓10.9</td></tr><tr><td>Ours (finetune) + MAED [114] Ours (finetune) + HybrIK [54]</td><td>video video</td><td>16 16</td><td> $6 6 . 8 \downarrow 1 7 . 3$   $5 2 . 6 \downarrow 5 . 5$ </td><td> $5 4 . 8 \downarrow 5 . 6$   $4 3 . 1 \downarrow 4 . 3$ </td><td>36.4↓1.9  $2 7 . 8 \downarrow 2 . 3$ </td><td>84.4↓8.9  $7 9 . 4 \downarrow 3 . 0$ </td><td>72.3 ↓ 6.7 68.8↓2.5</td><td> $4 2 . 3 \downarrow 3 . 4$   ${ \bf 4 0 . 6 . } | 1 . 3$ </td></tr></table>

Table 3. Quantitative comparison of human mesh recovery on Human3.6M and 3DPW datasets. T denotes the clip length used by the method. <sup>†</sup> denotes the results obtained with official model weights. The rest are all officially reported results. The gains in <sup>∗</sup> correspond to different re-implemented SPIN [48] results.

## 4.5. Human Mesh Recovery

We conduct experiments on Human3.6M [38] and 3DPW [113] datasets and additionally add COCO [61] dataset during training following [46, 58, 114]. We keep the same training and test split for both datasets as in [78] (Section 4.2)

![](images/4d86d3f94a3a07c4a343700db80f9d6b038f4cc84b9e309ddd754b4adebf0200.jpg)

![](images/a4677945c5211b4e33f869a09a0b272a6758b908e8c56c5ccb81b680869a07fa.jpg)

![](images/860450794e7623f5bf6e3d2abb99f077575793fa7971030a4267aa5d7da16b3d.jpg)  
Figure 3. Learning curves of finetuning and training from scratch.

<table><tr><td>Backbone (frozen)</td><td>MPJPE↓ (3D pose)</td><td>MPVE↓ (mesh)</td><td>Accuracy ↑ (action x-view)</td><td>Accuracy↑ (action 1-shot)</td></tr><tr><td>Random Pretrained</td><td>404.4mm 40.3mm</td><td>114.4mm 72.1mm</td><td>47.6% 87.3%</td><td>46.8% 60.7%</td></tr></table>

Table 4. Comparison of partial finetuning.

and [46, 58, 114], respectively. Following the common practice [41,46,49,114], we report MPJPE (mm) and PA-MPJPE (mm) of 14 joints obtained by J (θ, β). PA-MPJPE calculates MPJPE after aligning with GT in translation, rotation, and scale. We further report the mean per vertex error (MPVE) (mm) of the mesh (θ, β), which measures the average distance between the estimated and GT vertices after aligning the root joint. Note that most previous works [24, 41, 46, 48, 54, 58, 73, 114] use more datasets other than COCO [61] during training, such as LSP [39], MPI-INF-3DHP [79], etc., while we do not. Table 3 demonstrates that our finetuned model delivers competitive results on both Human3.6M and 3DPW datasets, surpassing all the state-of-theart video-based methods, including MAED [114], especially on the MPVE error. Nonetheless, we note that estimating full-body mesh from sparse 2D keypoints alone [10,25] is an ill-posed problem because it lacks human shape information. In light of this, we propose a hybrid approach that leverages the strengths of both our framework (coherent motion) and RGB-based methods (accurate shape). We introduce a refiner module that can be easily integrated with existing image/video-based methods, similar to [8,130]. Specifically, our refiner module is an MLP that takes the combination of our pretrained motion representations and an initial prediction, regressing a residual in joint rotations. Our approach effectively improves the state-of-the-art methods [48,54,114] and achieves the lowest error to date.

## 4.6. Ablation Studies

Finetune vs. Scratch. We compare the training progress of finetuning the pretrained model and training from scratch. As Figure 3 shows, models initialized with pretrained weights demonstrate superior performance and faster convergence on all three tasks. This observation suggests that the pretrained model learns transferable knowledge about human motion, facilitating the learning of multiple downstream tasks.

<table><tr><td>Pretrain</td><td>Noise</td><td>Mask 2D</td><td>MPJPE↓ (3D pose)</td><td>MPVE↓ (mesh)</td><td>Accuracy↑ (action x-sub)</td></tr><tr><td>-</td><td></td><td></td><td>39.2mm</td><td>75.7mm</td><td>87.7%</td></tr><tr><td>√</td><td>-</td><td></td><td>38.8mm</td><td>70.6mm</td><td>89.4%</td></tr><tr><td>√</td><td>√</td><td>一 1</td><td>38.1mm</td><td>68.4mm</td><td>90.7%</td></tr><tr><td>√</td><td>√</td><td>√ –</td><td>37.4mm</td><td>67.8mm</td><td>91.9%</td></tr><tr><td>√</td><td>√</td><td>√ √</td><td>37.5mm</td><td>65.5mm</td><td>93.0%</td></tr></table>

Table 5. Comparison of pretraining strategies.

Partial Finetuning. In addition to end-to-end finetuning, we freeze the motion encoder backbone and only train the regression head for each downstream task. To verify the effectiveness of the pretrained motion representations, we compared the pretrained motion encoder with a randomly initialized motion encoder. We report results of 3D pose and mesh on Human3.6M, action on NTU-RGB+D and NTU RGB+D-120 (same for the tables below). It can be seen in Table 4 that based on the frozen pretrained motion representations, our method still achieves competitive performance on multiple downstream tasks and shows a large improvement compared to the baseline. Pretraining and partial finetuning make it possible for all the downstream tasks to share the same backbone, significantly reducing computation overhead for applications requiring multi-task inference.

Pretraining Strategies. We evaluate how different pretraining strategies influence the performance of downstream tasks. Starting from the scratch baseline, we apply the proposed strategies one by one. As shown in Table 5, a vanilla 2D-to-3D pretraining stage brings benefits to all the downstream tasks. Introducing corruptions additionally improves the learned motion embeddings. Unified pretraining with in-the-wild videos (w. 2D) enjoys higher motion diversity, which further helps several downstream tasks.

Pretraining with Different Backbones. We further study the universality of the proposed pretraining approach. We replace the motion encoder backbone with two variants: TCN [89] and PoseFormer [135]. The models are slightly modified to a seq2seq version, while all the configurations for pretraining and finetuning are simply followed. Table 6 shows that the proposed approach consistently benefits different backbone models on different tasks.

<table><tr><td>Setting</td><td>MPJPE↓ (3D pose)</td><td>MPVE↓ (mesh)</td><td>Accuracy ↑ (action x-view)</td><td>Accuracy↑ (action 1-shot)</td></tr><tr><td>TCN (scratch)</td><td>50.1mm</td><td>92.6mm</td><td>91.5%</td><td>52.4%</td></tr><tr><td>TCN (finetune)</td><td>47.9mm</td><td>86.3mm</td><td>92.8%</td><td>59.9%</td></tr><tr><td>PoseFormer (scratch) PoseFormer (finetune)</td><td>44.8mm 41.5mm</td><td>85.9mm 80.5mm</td><td>94.2% 95.9%</td><td>57.4% 60.7%</td></tr></table>

Table 6. Comparison of different backbones.
<table><tr><td>Arch. Design</td><td>(a) S-T</td><td>(b)</td><td>(c)</td><td>(d)</td><td>(e) S-T + T-S S-T + T-S</td><td>(f)</td></tr><tr><td>MPJPE↓</td><td></td><td>T-S</td><td>S+T</td><td>ST-MHSA</td><td></td><td>(Average) (Adaptive)</td></tr><tr><td></td><td> $4 0 . 5 8 _ { \pm 0 . 3 1 }$ </td><td></td><td></td><td>41.05±0.24 41.76±0.22 41.54±0.35</td><td> $3 9 . 8 7 _ { \pm 0 . 3 2 }$ </td><td> $3 9 . 2 5 _ { \pm 0 . 2 7 }$ </td></tr></table>

Table 7. Comparison of model architecture variants. All the methods are trained on Human3.6M from scratch over 5 runs and measured by MPJPE (mm) with mean and standard deviation.

Model Architecture. Finally, we study the design choices of DSTformer. From (a) to (f) in Table 8, we compare different structure designs of the basic Transformer module. (a) and (b) are single-stream versions with different orders. (a) is conceptually similar to PoseFormer [135], MH-Former [56], and MixSTE [134]. (c) limits each stream to either temporal or spatial modeling before fusion and is similar to MAED [114]. (d) directly connects S-MHSA and T-MHSA without the MLP in between and is similar to the MSA-T variant in MAED [114]. (e) replaces the adaptive fusion with average pooling on two streams. (f) is the proposed DSTformer design. The result statistically confirms our design principles that both streams should be capable and meanwhile complementary, as introduced in Section 3.2. In addition, we find out that pairing each self-attention block with an MLP is crucial, as it could project the learned feature interactions and bring nonlinearity. In general, we design the model architecture for the 3D pose estimation task and apply it to all other tasks without additional adjustment.

## 5. Conclusion

In this work, we provide a unified perspective to tackling various human-centric video tasks. We develop a pretraining approach to learn human motion representations from large-scale and heterogeneous data sources. We also propose DSTformer as a universal human motion encoder. Experimental results on multiple benchmarks demonstrate the versatility of the learned motion representations. Future work could explore fusing the learned motion representations with generic video architectures as a human-centric semantic feature and applying it to more tasks (e.g., action assessment, segmentation).

Acknowledgement This work was partially supported by MOST-2022ZD0114900. We would like to thank Hai Ci and Jiefeng Li for their exceptional support.

## References

[1] Advanced Computing Center for the Arts and Design. ACCAD MoCap Dataset. 5

[2] Ijaz Akhter and Michael J. Black. Pose-conditioned joint angle limits for 3D human pose reconstruction. In CVPR, pages 1446–1455, June 2015. 5

[3] Mykhaylo Andriluka, Umar Iqbal, Eldar Insafutdinov, Leonid Pishchulin, Anton Milan, Juergen Gall, and Bernt Schiele. Posetrack: A benchmark for human pose estimation and tracking. In CVPR, pages 5167– 5176, 2018. 1, 3, 4, 5, 16

[4] Mykhaylo Andriluka, Leonid Pishchulin, Peter Gehler, and Bernt Schiele. Human pose estimation: New benchmark and state of the art analysis. In CVPR, 2014. 16

[5] Andreas Aristidou, Ariel Shamir, and Yiorgos Chrysanthou. Digital dance ethnography: Organizing large dance collections. JOCCH, 12(4):1–27, 2019. 5

[6] Anurag Arnab, Carl Doersch, and Andrew Zisserman. Exploiting temporal context for 3d human pose estimation in the wild. In CVPR, pages 3395–3404, 2019. 7

[7] Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. BEit: BERT pre-training of image transformers. In ICLR, 2022. 4

[8] Fabien Baradel, Romain Brégier, Thibault Groueix, Philippe Weinzaepfel, Yannis Kalantidis, and Grégory Rogez. Posebert: A generic transformer module for temporal 3d human modeling. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022. 3, 7, 8

[9] Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In ICML, page 41–48, 2009. 5

[10] Federica Bogo, Angjoo Kanazawa, Christoph Lassner, Peter Gehler, Javier Romero, and Michael J Black. Keep it smpl: Automatic estimation of 3d human pose and shape from a single image. In ECCV, pages 561–578, 2016. 3, 5, 8, 16

[11] Federica Bogo, Javier Romero, Gerard Pons-Moll, and Michael J Black. Dynamic faust: Registering human bodies in motion. In CVPR, pages 6233–6242, 2017. 5

[12] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. NeurIPS, 33:1877–1901, 2020. 4

[13] Yujun Cai, Liuhao Ge, Jun Liu, Jianfei Cai, Tat-Jen Cham, Junsong Yuan, and Nadia Magnenat Thalmann.

Exploiting spatial-temporal relationships for 3d pose estimation via graph convolutional networks. In ICCV, pages 2272–2281, 2019. 1, 2, 6

[14] Zhe Cao, Tomas Simon, Shih-En Wei, and Yaser Sheikh. Realtime multi-person 2d pose estimation using part affinity fields. In CVPR, pages 7291–7299, 2017. 1, 4, 16

[15] Carnegie Mellon University. CMU MoCap Dataset. 5

[16] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In CVPR, pages 6299–6308, 2017. 1

[17] Ju Yong Chang, Gyeongsik Moon, and Kyoung Mu Lee. Poselifter: Absolute 3d human pose lifting network from a single noisy 2d human pose. arXiv preprint arXiv:1910.12029, 2019. 5, 16

[18] Anargyros Chatzitofis, Leonidas Saroglou, Prodromos Boutis, Petros Drakoulis, Nikolaos Zioulis, Shishir Subramanyam, Bart Kevelham, Caecilia Charbonnier, Pablo Cesar, Dimitrios Zarpalas, et al. Human4d: A human-centric multimodal dataset for motions and immersive media. IEEE Access, 8:176241– 176262, 2020. 5

[19] Xipeng Chen, Kwan-Yee Lin, Wentao Liu, Chen Qian, and Liang Lin. Weakly-supervised discovery of geometry-aware representation for 3d human pose estimation. In CVPR, 2019. 3

[20] Yuxin Chen, Ziqi Zhang, Chunfeng Yuan, Bing Li, Ying Deng, and Weiming Hu. Channel-wise topology refinement graph convolution for skeleton-based action recognition. In ICCV, 2021. 7

[21] Ke Cheng, Yifan Zhang, Xiangyu He, Weihan Chen, Jian Cheng, and Hanqing Lu. Skeleton-based action recognition with shift graph convolutional network. In CVPR, 2020. 2, 7

[22] Yu Cheng, Bo Yang, Bo Wang, and Robby T Tan. 3d human pose estimation using spatio-temporal networks with explicit occlusion training. In AAAI, 2020. 2, 6

[23] Junhyeong Cho, Kim Youwang, and Tae-Hyun Oh. Cross-attention of disentangled modalities for 3d human mesh recovery with transformers. In European Conference on Computer Vision (ECCV), 2022. 7

[24] Hongsuk Choi, Gyeongsik Moon, Ju Yong Chang, and Kyoung Mu Lee. Beyond static features for temporally consistent 3d human pose and shape from a video. In CVPR, pages 1964–1973, 2021. 3, 7, 8

[25] Hongsuk Choi, Gyeongsik Moon, and Kyoung Mu Lee. Pose2mesh: Graph convolutional network for 3d human pose and mesh recovery from a 2d human pose. In ECCV, pages 769–787, 2020. 3, 7, 8, 16

[26] Hongsuk Choi, Gyeongsik Moon, JoonKyu Park, and Kyoung Mu Lee. Learning to estimate robust 3d human mesh from in-the-wild crowded scenes. In CVPR, pages 1475–1484, June 2022. 7

[27] Hai Ci, Xiaoxuan Ma, Chunyu Wang, and Yizhou Wang. Locally connected network for monocular 3d human pose estimation. IEEE TPAMI, pages 1–1, 2020. 5, 16

[28] Hai Ci, Chunyu Wang, Xiaoxuan Ma, and Yizhou Wang. Optimizing network structure for 3d human pose estimation. In ICCV, pages 2262–2271, 2019. 2, 6

[29] Hai Ci, Mingdong Wu, Wentao Zhu, Xiaoxuan Ma, Hao Dong, Fangwei Zhong, and Yizhou Wang. Gfpose: Learning 3d human pose prior with gradient fields. CVPR, 2023. 2

[30] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In NAACL, 2019. 3, 4, 16

[31] Haodong Duan, Jiaqi Wang, Kai Chen, and Dahua Lin. Pyskl: Towards good practices for skeleton action recognition, 2022. 16

[32] Haodong Duan, Yue Zhao, Kai Chen, Dian Shao, Dahua Lin, and Bo Dai. Revisiting skeleton-based action recognition. In CVPR, 2022. 2, 3, 6, 7, 16

[33] Mark Endo, Kathleen L. Poston, Edith V. Sullivan, Li Fei-Fei, Kilian M. Pohl, and Ehsan Adeli. Gaitforemer: Self-supervised pre-training of transformers via human motion forecasting for few-shot gait impairment severity estimation. In MICCAI, pages 130–139, 2022. 2

[34] Saeed Ghorbani, Kimia Mahdaviani, Anne Thaler, Konrad Kording, Douglas James Cook, Gunnar Blohm, and Nikolaus F. Troje. MoVi: A large multipurpose motion and video dataset, 2020. 5

[35] Rıza Alp Güler, Natalia Neverova, and Iasonas Kokkinos. Densepose: Dense human pose estimation in the wild. In CVPR, pages 7297–7306, 2018. 1

[36] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross B. Girshick. Masked autoencoders are scalable vision learners. In CVPR, 2022. 4, 18

[37] Ludovic Hoyet, Kenneth Ryall, Rachel McDonnell, and Carol O’Sullivan. Sleight of hand: Perception of finger motion from reduced marker sets. In Proceedings ofthe ACM SIGGRAPH Symposium on Interactive 3D Graphics and Games, 2012. 5

[38] Catalin Ionescu, Dragos Papava, Vlad Olaru, and Cristian Sminchisescu. Human3.6m: Large scale datasets

and predictive methods for 3d human sensing in natural environments. IEEE TPAMI, 2014. 1, 5, 7, 16

[39] Sam Johnson and Mark Everingham. Clustered pose and nonlinear appearance models for human pose estimation. In BMVC, volume 2, page 5, 2010. 8

[40] Hanbyul Joo, Natalia Neverova, and Andrea Vedaldi. Exemplar fine-tuning for 3d human model fitting towards in-the-wild 3d human pose estimation. In 3DV, pages 42–52. IEEE, 2021. 16

[41] Angjoo Kanazawa, Michael J Black, David W Jacobs, and Jitendra Malik. End-to-end recovery of human shape and pose. In CVPR, pages 7122–7131, 2018. 2, 7, 8, 16

[42] Angjoo Kanazawa, Jason Y Zhang, Panna Felsen, and Jitendra Malik. Learning 3d human dynamics from video. In CVPR, pages 5614–5623, 2019. 2, 3, 5, 7, 16

[43] Rawal Khirodkar, Shashank Tripathi, and Kris Kitani. Occluded human mesh recovery. In CVPR, pages 1715–1725, June 2022. 7

[44] Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. Supervised contrastive learning. In NeurIPS, 2020. 7, 16

[45] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint, arXiv:1412.6980, 2014. 16

[46] Muhammed Kocabas, Nikos Athanasiou, and Michael J Black. Vibe: Video inference for human body pose and shape estimation. In CVPR, pages 5253–5263, 2020. 1, 3, 5, 7, 8, 16

[47] Muhammed Kocabas, Chun-Hao P. Huang, Otmar Hilliges, and Michael J. Black. Pare: Part attention regressor for 3d human body estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 11127–11137, October 2021. 7

[48] Nikos Kolotouros, Georgios Pavlakos, Michael J Black, and Kostas Daniilidis. Learning to reconstruct 3d human pose and shape via model-fitting in the loop. In ICCV, pages 2252–2261, 2019. 2, 7, 8, 16

[49] Nikos Kolotouros, Georgios Pavlakos, and Kostas Daniilidis. Convolutional mesh regression for singleimage human shape reconstruction. In CVPR, pages 4501–4510, 2019. 8

[50] Nikos Kolotouros, Georgios Pavlakos, Dinesh Jayaraman, and Kostas Daniilidis. Probabilistic modeling for human mesh recovery. In ICCV, pages 11605– 11614, October 2021. 7

[51] Björn Krüger, Anna Vögele, Tobias Willig, Angela Yao, Reinhard Klein, and Andreas Weber. Efficient unsupervised temporal segmentation of motion data. IEEE Transactions on Multimedia, 2017. 2

[52] Bio Motion Lab. BMLhandball Motion Capture Database. 5

[53] Andreas Lehrmann, Peter V. Gehler, and Sebastian Nowozin. Efficient non-linear markov models for human motion. In CVPR, 2014. 2

[54] Jiefeng Li, Chao Xu, Zhicun Chen, Siyuan Bian, Lixin Yang, and Cewu Lu. Hybrik: A hybrid analytical-neural inverse kinematics solution for 3d human pose and shape estimation. In CVPR, pages 3383–3393, 2021. 7, 8

[55] Maosen Li, Siheng Chen, Xu Chen, Ya Zhang, Yan feng Wang, and Qi Tian. Actional-structural graph convolutional networks for skeleton-based action recognition. In CVPR, 2019. 2

[56] Wenhao Li, Hong Liu, Hao Tang, Pichao Wang, and Luc Van Gool. Mhformer: Multi-hypothesis transformer for 3d human pose estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13147–13156, 2022. 2, 6, 9, 16

[57] Zhihao Li, Jianzhuang Liu, Zhensong Zhang, Songcen Xu, and Youliang Yan. Cliff: Carrying location information in full frames into human pose and shape estimation. In ECCV, 2022. 7

[58] Kevin Lin, Lijuan Wang, and Zicheng Liu. End-toend human pose and mesh reconstruction with transformers. In CVPR, pages 1954–1963, 2021. 7, 8

[59] Kevin Lin, Lijuan Wang, and Zicheng Liu. Mesh graphormer. In ICCV, pages 12939–12948, 2021. 7

[60] Lilang Lin, Sijie Song, Wenhan Yang, and Jiaying Liu. Ms2l: Multi-task self-supervised learning for skeleton based action recognition. In ACM Interna tional Conference on Multimedia, pages 2490–2498, 2020. 2, 17

[61] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, pages 740–755, 2014. 7, 8, 16

[62] Li Linguo, Wang Minsi, Ni Bingbing, Wang Hang, Yang Jiancheng, and Zhang Wenjun. 3d human action representation learning via cross-view consistency pursuit. In CVPR, 2021. 2, 7, 17

[63] Chunhui Liu, Yueyu Hu, Yanghao Li, Sijie Song, and Jiaying Liu. Pku-mmd: A large scale benchmark for continuous multi-modal human action understanding. arXiv preprint arXiv:1703.07475, 2017. 1

[64] Jun Liu, Amir Shahroudy, Mauricio Perez, Gang Wang, Ling-Yu Duan, and Alex C Kot. Ntu rgb+ d 120: A large-scale benchmark for 3d human activity understanding. IEEE TPAMI, 42(10):2684–2701, 2019. 1, 2, 6, 7

[65] Jun Liu, Amir Shahroudy, Dong Xu, Alex C Kot, and Gang Wang. Skeleton-based action recognition using spatio-temporal lstm network with trust gates. IEEE TPAMI, 40(12):3007–3021, 2017. 7

[66] Jun Liu, Gang Wang, Ping Hu, Ling-Yu Duan, and Alex C Kot. Global context-aware attention lstm networks for 3d action recognition. In CVPR, pages 1647–1656, 2017. 7

[67] Ruixu Liu, Ju Shen, He Wang, Chen Chen, Sen-ching Cheung, and Vijayan Asari. Attention mechanism exploits temporal contexts: Real-time 3d human pose reconstruction. In CVPR, 2020. 6

[68] Ziyu Liu, Hongwen Zhang, Zhenghao Chen, Zhiyong Wang, and Wanli Ouyang. Disentangling and unifying graph convolutions for skeleton-based action recognition. In CVPR, pages 143–152, 2020. 2, 7

[69] Matthew Loper, Naureen Mahmood, and Michael J. Black. MoSh: Motion and Shape Capture from Sparse Markers. ACM Transactions on Graphics, 33(6), Nov. 2014. 5

[70] Matthew Loper, Naureen Mahmood, and Michael J Black. Mosh: Motion and shape capture from sparse markers. TOG, 33(6):1–13, 2014. 16

[71] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J Black. Smpl: A skinned multi-person linear model. ACM Transactions on Graphics, 34(6):1–16, 2015. 1, 2, 5

[72] Eyes JAPAN Co. Ltd. Eyes Japan MoCap Dataset. 5

[73] Zhengyi Luo, S Alireza Golestaneh, and Kris M Kitani. 3d human motion estimation via motion compression and refinement. In ACCV, 2020. 8

[74] Diogo C. Luvizon, David Picard, and Hedi Tabia. 2d/3d pose estimation and action recognition using multitask deep learning. In CVPR, pages 5137–5146, 2018. 2

[75] Xiaoxuan Ma, Jiajun Su, Chunyu Wang, Wentao Zhu, and Yizhou Wang. 3d human mesh estimation from virtual markers. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 534–543, June 2023. 2

[76] Naureen Mahmood, Nima Ghorbani, Nikolaus F. Troje, Gerard Pons-Moll, and Michael J. Black. AMASS: Archive of motion capture as surface shapes. In ICCV, pages 5441–5450, Oct. 2019. 1, 3, 4, 5, 16

[77] Christian Mandery, Ömer Terlemez, Martin Do, Nikolaus Vahrenkamp, and Tamim Asfour. The kit wholebody human motion database. In ICAR, pages 329– 336. IEEE, 2015. 5

[78] Julieta Martinez, Rayat Hossain, Javier Romero, and James J Little. A simple yet effective baseline for 3d human pose estimation. In ICCV, pages 2640–2649, 2017. 2, 5, 6, 7

[79] Dushyant Mehta, Helge Rhodin, Dan Casas, Pascal Fua, Oleksandr Sotnychenko, Weipeng Xu, and Christian Theobalt. Monocular 3d human pose estimation in the wild using improved cnn supervision. In 3DV, pages 506–516. IEEE, 2017. 8

[80] Raphael Memmesheimer, Simon Häring, Nick Theisen, and Dietrich Paulus. Skeleton-dml: Deep metric learning for skeleton-based one-shot action recognition. In WACV, pages 3702–3710, 2022. 7

[81] Raphael Memmesheimer, Nick Theisen, and Dietrich Paulus. Sl-dml: Signal level deep metric learning for multimodal one-shot action recognition. In ICPR, 2021. 2, 7

[82] Gyeongsik Moon, Juyong Chang, and Kyoung Mu Lee. Camera distance-aware top-down approach for 3d multi-person pose estimation from a single rgb image. In ICCV, 2019. 2

[83] Gyeongsik Moon and Kyoung Mu Lee. I2l-meshnet: Image-to-lixel prediction network for accurate 3d human pose and mesh estimation from a single rgb im age. In ECCV, pages 752–768, 2020. 2, 7

[84] Meinard Müller, Tido Röder, Michael Clausen, Bernhard Eberhardt, Björn Krüger, and Andreas Weber. Documentation mocap database HDM05. Technical Report CG-2007-2, Universität Bonn, June 2007. 5

[85] Mahdiar Nekoui and Li Cheng. Enhancing human motion assessment by self-supervised representation learning. In BMVC, 2021. 2

[86] Alejandro Newell, Kaiyu Yang, and Jia Deng. Stacked hourglass networks for human pose estimation. In ECCV, 2016. 3, 5

[87] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In NeurIPS. 2019. 16

[88] Georgios Pavlakos, Xiaowei Zhou, and Kostas Daniilidis. Ordinal depth supervision for 3d human pose estimation. In CVPR, pages 7307–7316, 2018. 6

[89] Dario Pavllo, Christoph Feichtenhofer, David Grangier, and Michael Auli. 3d human pose estimation in video with temporal convolutions and semisupervised training. In CVPR, pages 7753–7762, 2019. 2, 3, 4, 5, 6, 9, 16

[90] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofMachine Learning Research, 2020. 4

[91] Javier Romero, Dimitrios Tzionas, and Michael J. Black. Embodied hands: Modeling and capturing hands and bodies together. ACM Transactions on Graphics, (Proc. SIGGRAPH Asia), 2017. 16

[92] Alberto Sabater, Laura Santos, Jose Santos-Victor, Alexandre Bernardino, Luis Montesano, and Ana C. Murillo. One-shot action recognition in challenging therapy scenarios. In CVPR Workshop, 2021. 2, 7

[93] Amir Shahroudy, Jun Liu, Tian-Tsong Ng, and Gang Wang. Ntu rgb+ d: A large scale dataset for 3d human activity analysis. In CVPR, pages 1010–1019, 2016. 1, 6

[94] Wenkang Shan, Zhenhua Liu, Xinfeng Zhang, Shanshe Wang, Siwei Ma, and Wen Gao. P-stmo: Pretrained spatial temporal many-to-one model for 3d human pose estimation. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part V, pages 461–478. Springer, 2022. 2, 6

[95] Dian Shao, Yue Zhao, Bo Dai, and Dahua Lin. Finegym: A hierarchical video dataset for fine-grained action understanding. In CVPR, pages 2616–2625, 2020. 1

[96] Lei Shi, Yifan Zhang, Jian Cheng, and Hanqing Lu. Two-stream adaptive graph convolutional networks for skeleton-based action recognition. In CVPR, 2019. 1, 2, 7

[97] Leonid Sigal, Alexandru O Balan, and Michael J Black. Humaneva: Synchronized video and motion capture dataset and baseline algorithm for evaluation of articulated human motion. IJCV, 87(1):4–27, 2010. 5

[98] Sijie Song, Cuiling Lan, Junliang Xing, Wenjun Zeng, and Jiaying Liu. An end-to-end spatio-temporal attention model for human action recognition from skeleton data. In AAAI, page 4263–4270, 2017. 2

[99] Yang Song, Luis Goncalves, and Pietro Perona. Unsupervised learning of human motion models. In T. Dietterich, S. Becker, and Z. Ghahramani, editors, NeurIPS, volume 14. MIT Press, 2001. 2

[100] Kun Su, Xiulong Liu, and Eli Shlizerman. Predict & cluster: Unsupervised skeleton based action recognition. In CVPR, pages 9631–9640, 2020. 2

[101] Yukun Su, Guosheng Lin, and Qingyao Wu. Selfsupervised 3d skeleton action representation learning with motion consistency and continuity. In ICCV, 2021. 2, 7, 17

[102] Jiangxin Sun, Zihang Lin, Xintong Han, Jian-Fang Hu, Jia Xu, and Wei-Shi Zheng. Action-guided 3d human motion prediction. In NeurIPS, volume 34, pages 30169–30180. Curran Associates, Inc., 2021. 2

[103] Ke Sun, Bin Xiao, Dong Liu, and Jingdong Wang. Deep high-resolution representation learning for human pose estimation. In CVPR, 2019. 3, 4, 6, 16

[104] Xiao Sun, Bin Xiao, Fangyin Wei, Shuang Liang, and Yichen Wei. Integral human pose regression. In ECCV, 2018. 2

[105] Yu Sun, Qian Bao, Wu Liu, Yili Fu, Michael J Black, and Tao Mei. Monocular, one-stage, regression of multiple 3d people. In ICCV, pages 11179–11188, 2021. 7

[106] Yu Sun, Yun Ye, Wu Liu, Wenpeng Gao, Yili Fu, and Tao Mei. Human mesh recovery from monocular images via a skeleton-disentangled representation. In ICCV, pages 5349–5358, 2019. 3, 7

[107] Fida Mohammad Thoker, Hazel Doughty, and Cees GM Snoek. Skeleton-contrastive 3d action representation learning. In ACM International Conference on Multimedia, pages 1655–1663, 2021. 2, 17

[108] Dorra Trabelsi, Samer Mohammed, Faicel Cham roukhi, Latifa Oukhellou, and Yacine Amirat. An unsupervised approach for automatic activity recognition based on hidden markov model regression. IEEE Transactions on Automation Science and Engineering, 2013. 2

[109] Shashank Tripathi, Siddhant Ranade, Ambrish Tyagi, and Amit K. Agrawal. Posenet3d: Learning temporally consistent 3d human pose via knowledge distillation. In 8th International Conference on 3D Vision, 2020. 3

[110] Nikolaus F. Troje. Decomposing biological motion: A framework for analysis and synthesis of human gait patterns. Journal of Vision, 2(5):2–2, Sept. 2002. 5

[111] Matthew Trumble, Andrew Gilbert, Charles Malleson, Adrian Hilton, and John P. Collomosse. Total capture: 3d human pose estimation fusing video and inertial sensors. In BMVC, 2017. 5

[112] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. 4

[113] Timo von Marcard, Roberto Henschel, Michael J Black, Bodo Rosenhahn, and Gerard Pons-Moll. Recovering accurate 3d human pose in the wild using imus and a moving camera. In ECCV, pages 601–617, 2018. 7, 16

[114] Ziniu Wan, Zhengjia Li, Maoqing Tian, Jianbo Liu, Shuai Yi, and Hongsheng Li. Encoder-decoder with multi-level attention for 3d human shape and pose estimation. In ICCV, pages 13033–13042, 2021. 7, 8, 9, 16

[115] Chunyu Wang, Yizhou Wang, and Alan L. Yuille. An approach to pose-based action recognition. In CVPR, pages 915–922, 2013. 2

[116] Jingbo Wang, Sijie Yan, Yuanjun Xiong, and Dahua Lin. Motion guided 3d pose estimation from videos. In European Conference on Computer Vision, pages 764–780. Springer, 2020. 1, 2, 6

[117] Peng Wang, Jun Wen, Chenyang Si, Yuntao Qian, and Liang Wang. Contrast-reconstruction representation learning for self-supervised skeleton-based action recognition. arXiv preprint arXiv:2111.11051, 2021. 2

[118] Xin Wang, Yudong Chen, and Wenwu Zhu. A survey on curriculum learning. IEEE TPAMI, 2022. 5

[119] Tom Wehrbein, Marco Rudolph, Bodo Rosenhahn, and Bastian Wandt. Probabilistic monocular 3d human pose estimation with normalizing flows. In ICCV, 2021. 6

[120] Wen-Li Wei, Jen-Chun Lin, Tyng-Luh Liu, and Hong-Yuan Mark Liao. Capturing humans in motion: Temporal-attentive 3d human pose and shape estimation from monocular video. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2022. 7

[121] Tianhan Xu and Wataru Takano. Graph stacked hourglass networks for 3d human pose estimation. In CVPR, 2021. 6

[122] Yuanlu Xu, Song-Chun Zhu, and Tony Tung. Denserac: Joint 3d pose and shape estimation by dense render-and-compare. In ICCV, pages 7760– 7770, 2019. 2

[123] Sijie Yan, Yuanjun Xiong, and Dahua Lin. Spatial temporal graph convolutional networks for skeletonbased action recognition. In AAAI, 2018. 1, 2, 7, 16

[124] Di Yang, Yaohui Wang, Antitza Dantcheva, Lorenzo Garattoni, Gianpiero Francesca, and Francois Bremond. Unik: A unified framework for real-world skeleton-based action recognition. In BMVC, 2021. 7

[125] Siyuan Yang, Jun Liu, Shijian Lu, Meng Hwa Er, and Alex C. Kot. Skeleton cloud colorization for

unsupervised 3d action representation learning. In ICCV, 2021. 7, 17

[126] Zhuoqian Yang, Wentao Zhu, Wayne Wu, Chen Qian, Qiang Zhou, Bolei Zhou, and Chen Change Loy. Transmomo: Invariance-driven unsupervised video motion retargeting. In CVPR, 2020. 2

[127] Angela Yao, Juergen Gall, and Luc Gool. Coupled action recognition and pose estimation from multiple views. Int. J. Comput. Vision, page 16–37, 2012. 2

[128] Chun-Han Yao, Jimei Yang, Duygu Ceylan, Yi Zhou, Yang Zhou, and Ming-Hsuan Yang. Learning visibility for robust dense human body estimation. In European conference on computer vision (ECCV), 2022. 7

[129] Raymond Yeh, Yuan-Ting Hu, and Alexander Schwing. Chirality nets for human pose regression. In NeurIPS, 2019. 6

[130] Ailing Zeng, Lei Yang, Xuan Ju, Jiefeng Li, Jianyi Wang, and Qiang Xu. Smoothnet: A plug-and-play network for refining human poses in videos. In European Conference on Computer Vision. Springer, 2022. 3, 7, 8

[131] Feng Zhang, Xiatian Zhu, Hanbin Dai, Mao Ye, and Ce Zhu. Distribution-aware coordinate representation for human pose estimation. In CVPR, June 2020. 16

[132] Feng Zhang, Xiatian Zhu, and Mao Ye. Fast human pose estimation. In CVPR, 2019. 2

[133] Hongwen Zhang, Yating Tian, Xinchi Zhou, Wanli Ouyang, Yebin Liu, Limin Wang, and Zhenan Sun. Pymaf: 3d human pose and shape regression with pyramidal mesh alignment feedback loop. In ICCV, pages 11446–11456, 2021. 2, 7

[134] Jinlu Zhang, Zhigang Tu, Jianyu Yang, Yujin Chen, and Junsong Yuan. Mixste: Seq2seq mixed spatiotemporal encoder for 3d human pose estimation in video. In CVPR, 2022. 2, 4, 5, 6, 9, 16

[135] Ce Zheng, Sijie Zhu, Matias Mendieta, Taojiannan Yang, Chen Chen, and Zhengming Ding. 3d human pose estimation with spatial and temporal transformers. ICCV, 2021. 2, 5, 6, 9, 16

[136] Kun Zhou, Xiaoguang Han, Nianjuan Jiang, Kui Jia, and Jiangbo Lu. Hemlets pose: Learning part-centric heatmap triplets for accurate 3d human pose estimation. In ICCV, 2019. 2

[137] Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. On the continuity of rotation representations in neural networks. In CVPR, 2019. 16

[138] Wentao Zhu, Cuiling Lan, Junliang Xing, Wenjun Zeng, Yanghao Li, Li Shen, and Xiaohui Xie. Cooccurrence feature learning for skeleton based action

recognition using regularized deep lstm networks. In AAAI, 2016. 2

[139] Wentao Zhu, Zhuoqian Yang, Ziang Di, Wayne Wu, Yizhou Wang, and Chen Change Loy. Mocanet: Motion retargeting in-the-wild via canonicalization networks. In AAAI, 2022. 2

## A. Appendix

## A.1. Experimental Details

Setup. We implement the proposed model with $\mathrm { P y } .$ Torch [87]. All the experiments are conducted on a Linux machine with 8 NVIDIA V100 GPUs, which is intended for accelerating pretraining. A single GPU is usually sufficient for finetuning and inference.

Pretraining. For AMASS [76], we first render the parameterized human model SMPL+H [91], then extract 3D keypoints with a pre-defined regression matrix [10]. We extract 3D keypoints from Human3.6M [38] by camera projection [27]. We sample motion clips with length $T = 2 4 3$ for 3D mocap data. For 2D data, we utilize the provided annotations of PoseTrack [3]. We further include 2D motion from an unannotated video dataset InstaVariety [42] extracted by OpenPose [14]. Since the valid sequence lengths for in-thewild videos are much shorter, we use $T = 3 0$ (PoseTrack) and $T = 8 1$ (InstaVariety). We convert keypoints of 2D datasets (COCO and OpenPose format) to Human3.6M using permutation and interpolation following previous works. We set the input channels $C _ { \mathrm { i n } } = 3 \ : ( x$ , y coordinates and confidence) following [32, 123]. Random horizontal flipping is applied as data augmentation. The whole network is trained for 90 epochs with learning rate 0.0005 and batch size 64 using an Adam [45] optimizer. The weights of the loss terms are $\lambda _ { 0 } = 2 0$ . We set the 2D skeleton masking rat $\mathrm { i o } = 1 5 \%$ , same as BERT [30]. More specifically, we use 10% frame-level masks and 5% joint-level masks. We vary the proportion of the two types of masks and only observe marginal differences. We follow [17] to fit a mixture of distributions and sample per-joint noises from it. To keep the noise smooth and avoid severe jittering, we first sample the noise $\boldsymbol { z } \in \mathbb { R } ^ { T _ { K } \times J }$ for $T _ { K } = 2 7$ keyframes, then upsample it to $z ^ { \prime } \in \mathbb { R } ^ { T \times J }$ , and add a small gaussian noise $\mathcal { N } ( 0 , 0 . 0 0 2 ^ { 2 } )$ .

3D Pose Estimation. The 2D skeletons are provided by 2D pose estimator trained on MPII [4] and Human3.6M [38] following the common practice [27, 89]. For training from scratch, we train for 60 epochs with learning rate 0.0002 and batch size 32. For finetuning, we load the pretrained weights and train for 30 epochs.

Skeleton-based Action Recognition. We use the 2D skeleton sequences extracted with HRNet [103] provided by PYSKL [31]. We then upsample the skeleton sequences to a uniform length $T = 2 4 3$ . For $\mathrm { N T U - R G B + D } _ { \mathrm { ~ \tiny ~  ~ } }$ , the DSTformer output after global average pooling is fed into an MLP including dropout $p = 0 . 5$ , a hidden layer of 2048 channels, BatchNorm, and ReLU. We train for 200 epochs with learning rate 0.001 and batch size 32 for training from scratch. We set learning rate 0.0001 for the backbone, and learning rate 0.001 for the downstream MLP for finetuning. For one-shot recognition on NTU-RGB+D-120, we apply dropout $p = 0 . 1$ , and use a linear layer to get action rep resentation of size 2048. We train with batch size 16, and each batch includes 8 action pairs. Samples from the same action class are set as positives against the negatives from the remainder of the batch. We use the supervised contrastive loss [44] with temperature 0.1.

Human Mesh Recovery. For the Human3.6M dataset, we use the same 2D skeleton sequences as the 3D pose estimation task. The SMPL ground-truth (GT) parameters are obtained by MoSh [70] which are fitted to the physical mocap markers. For the 3DPW dataset, we obtain its detected 2D skeleton sequences from DarkPose [131] provided by [25]. The SMPL GT parameters of 3DPW are obtained using IMUs [113]. Following previous work [41, 46, 114], we add COCO [61] datasets for training, following [25]. The pseudo-GT SMPL parameters of COCO are provided by EFT [40]. Following [46, 48], we use the 6D continuous rotation representations [137] instead of original axis angles when estimating pose parameters. The shape and pose MLPs are the same, including one hidden layer of 512 channels, BatchNorm, and ReLU. We sample motion clips with length $T = 1 6$ and stride 8 for both datasets. The weights of the loss terms are $\lambda _ { \mathrm { 3 D } } ^ { \mathrm { m } } = 0 . 5 , \lambda _ { \theta } = 1 0 0 0 , \lambda _ { \beta } = 1 , \lambda _ { \mathrm { n } } = 2 0$ and $\lambda _ { 0 } ^ { \mathrm { m } } = 1 0$ . We train the whole network end-to-end for 60 epochs with batch size 128, with learning rates 0.00005 for the backbone, and 0.0005 for the downstream network, respectively.

## A.2. Additional Experiments

We first provide a deeper analysis of the proposed model architecture, then present more results on skeleton-based action recognition. We also include an ablation study on mask ratio for pertaining as well as the qualitative results.

## A.2.1 Analysis of DSTformer

We investigate the proposed DSTformer model to interpret its performance improvement and re-evaluate the design assumptions in Section 3.2. The major difference between DSTformer and existing works [56,134,135] lies in the dualstream design and the adaptive fusion mechanism. Therefore, we probe the fusion weights $\alpha _ {  { \mathrm { S } } \mathrm { T } }$ and $\alpha _ { \mathrm { T S } }$ produced by the attention regressor to understand how DSTformer balances the two streams and adapts to different inputs.

We first examine how different input actions influence the dual-stream fusion weights, and the results are shown in Figure 4 (left). Note that $\alpha _ { \mathrm { S T } } + \alpha _ { \mathrm { T S } } = 1$ , so we only analyze $\pmb { \alpha } _ {  { \mathrm { S T } } }$ for simplicity. We find that actions with a higher degree of movement $( e . g .$ . walking) lead to a smaller fusion weight for the S-T stream, which means the model is inclined to the $T . S$ stream. Oppositely, when the input is from a more static action (e.g. sitting and eating), the model tends to increase the weights of the S-T stream. These results align with our assumption that the two streams could focus on different spatial-temporal aspects of features and that the model can reweigh the two streams based on the input characteristics. Intuitively, the S-T branch is more specialized for spatial modeling, while the T-S branch is more focused on temporal modeling.

![](images/2299231489af578781352e29d151384299721cbff9af7bfdf8ccc6372a468167.jpg)

![](images/53588678bc20dfca06f9595cb1c7b46b3eb4d59010da38e06f39bc3063ae7316.jpg)  
Figure 4. Statistic of fusion weights grouped by actions (left) and joints (right). We plot the average fusion weights for the S-T stream on the Human3.6M test set. The third layer of a 3D pose estimation model trained from scratch on Human3.6M is used for probing.

<table><tr><td colspan="2">Depth(N) Heads(h)</td><td>Channels(Ce)</td><td>MPJPE</td></tr><tr><td>4</td><td>8</td><td>512</td><td>39.7</td></tr><tr><td>6</td><td>8</td><td>512</td><td>40.4</td></tr><tr><td>5</td><td>6</td><td>512</td><td>39.6</td></tr><tr><td>5</td><td>10</td><td>512</td><td>39.5</td></tr><tr><td>5</td><td>8</td><td>256</td><td>40.4</td></tr><tr><td>5</td><td>8</td><td>512</td><td>39.2</td></tr></table>

Table 8. Comparison of model architecture design. All the methods are trained on Human3.6M from scratch and measured by MPJPE (mm).

We further study the distribution of fusion weights across the body joints. As illustrated in Figure 4 (right), the fu sion weights vary dramatically across different body joints. Specifically, the model leans towards the T-S branch for keypoints with large temporal variations (limb joints), while it gives more importance to the S-T branch for keypoints that are more stable (torso joints). The result also corresponds to our “division and cooperation” design of the two streams. In addition, we analyze the correlation of fusion weights for different body joints. Figure 5 indicates that fusion weights of joints from the same functional part (e.g. arms, legs, head, torso) are naturally correlated, which implies that the model learns the joint connections implicitly and adjusts their weights in an interrelated manner with regard to input changes.

![](images/cb635fb597f1c162f8cb709c840d092e48939ac822db68dcf145834aefca6ce9.jpg)  
Figure 5. Correlation matrix of the joint-wise fusion weights. The results are computed on the Human3.6M test set. We use the third layer of a 3D pose estimation model trained from scratch on Human3.6M for probing.

## A.2.2 Architecture Hyperparameter Analysis

We explore how different architecture hyperparameters influence the performance of DSTformer. In Table 8, we vary the number of stacking modules (depth), self-attention heads (heads), and embedding size (channels) and measure their performance on the 3D human pose estimation task. We then select the optimal combination and apply the design to all the tasks without additional adjustment.

<table><tr><td rowspan="2">Method</td><td colspan="3">X-Sub</td><td colspan="3">X-View</td></tr><tr><td>scratch</td><td>finetune</td><td>diff</td><td>scratch</td><td>finetune</td><td>diff</td></tr><tr><td>MS2L [60]</td><td>78.4</td><td>78.6</td><td>+0.2</td><td>-</td><td>-</td><td>-</td></tr><tr><td>CrosSCLR [62]</td><td>85.2</td><td>86.2</td><td>+1.0</td><td>91.4</td><td>92.5</td><td>+1.1</td></tr><tr><td>Skel-Con [107]</td><td>72.9</td><td>79.3</td><td>+6.4</td><td>87.8</td><td>90.4</td><td>+2.6</td></tr><tr><td>MCC [101]</td><td>88.5</td><td>89.7</td><td>+1.2</td><td>95.1</td><td>96.3</td><td>+1.2</td></tr><tr><td>SCC [125]</td><td>-</td><td>88.0</td><td>-</td><td>-</td><td>94.9</td><td>-</td></tr><tr><td>Ours</td><td>87.7</td><td>93.0</td><td>+5.3</td><td>94.1</td><td>97.2</td><td>+3.1</td></tr></table>

Table 9. Quantitative comparison with self-supervised action recognition approaches. We measure the top-1 accuracy on NTU-RGB+D. diff measures the performance gap between scratch and finetune.

## A.2.3 Comparison with Self-supervised Action Recognition Approaches

Our framework also shares some commonalities with the research direction of self-supervised skeleton-based action recognition [60, 62, 101, 107, 125]. These methods design various pretext tasks on 3D skeleton sequences and learn 3D motion representations without action labels (i.e. selfsupervised learning). To our best knowledge, our method is the first to use 2D-to-3D lifting as the pretext task for skeleton-based action recognition. In this way, it learns 3D-aware motion representations on 2D skeleton sequences.

![](images/4fe66109a057b27547399eab45d8f6dc2dfd0e8dd2b82e446da0f28127d52c51.jpg)  
Figure 6. Visualization of action representations for one-shot recognition. We show the action representations of the 20 novel classes after training on the other 100 auxiliary classes with contrastive learning. We compute the pairwise cosine distance and apply t-SNE. Diamonds indicate the labeled exemplars for one-shot recognition.

While previous methods learn 3D motion representations that require 3D skeleton sequences as input (usually obtained with RGB-D devices such as Azure Kinect), the 2D motion representations can be applied to in-the-wild videos (from mobile devices or the Internet) more easily. We conduct a quantitative comparison and show the results in Table 9. The accuracy gain of finetuning compared to training from scratch could be considered as the contribution of learned motion representations. Although our network is not specifically designed for the action recognition task (scratch), it surpasses the previous methods on fine-tuned performances (finetune) and shows considerable performance gains (diff).

## A.2.4 One-shot Action Recognition

Figure 6 shows the learned action representation for the novel classes with our finetuned model. We can see that meaningful clusters emerge although the model has never seen these actions before. We perform one-shot action recognition using 1-nearest-neighbor with the exemplars and achieve 67.4% top-1 accuracy.

## A.2.5 Mask Ratio Analysis

We study the effect of different masking ratios on both pretraining and downstream finetuning. We measure the performance by MPJPE on the Human3.6M test set. Frame-level mask and joint-level mask are fixed to 2 : 1. As Figure 7 shows, when the masking ratio is less than 45%, the pretrained models could learn 2D-to-3D lifting well, and pretraining improves the finetuning performance (better than the scratch baseline). As we continue to increase the masking ratio, the pretrained models perform much worse at 2D-to-3D lifting, and pretraining is no longer beneficial for the finetuning performance. Our observation is slightly different from MAE [36] where a high masking ratio (75%) works well. One possible explanation is that skeletons are highly abstract representations of human motion. A high masking ratio leaves too few cues to complete the whole sequence, therefore the model tends to overfit. It is also worth noting that in our pretraining task (2D-to-3D lifting), the depth channel $( \frac 1 3$ of original data) is intrinsically masked.

![](images/e182c8738d33d61f0a308d017a78c8b39298f479f55164c32ed1dccf21f4985b.jpg)  
Figure 7. Ablation study of pretraining mask ratios. For each mask ratio, We test the Human3.6M MPJPE of the pretrained models with/without finetuning respectively.

## A.3. Limitations

The primary limitation of this work lies in its focus on single-person skeleton sequences, which results in the motion representation being insensitive to factors such as appearance, surroundings, and interactions. To overcome these limitations, future work could explore combining the learned motion representation with generic video features or scene representations.