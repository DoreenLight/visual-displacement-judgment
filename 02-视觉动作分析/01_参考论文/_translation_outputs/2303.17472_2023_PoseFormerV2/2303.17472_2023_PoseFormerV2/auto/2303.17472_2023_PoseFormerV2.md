# PoseFormerV2: Exploring Frequency Domain for Efficient and Robust 3D Human Pose Estimation

Qitao Zhao<sup>1\*</sup> Ce Zheng<sup>2</sup> Mengyuan Liu<sup>3</sup> Pichao Wang<sup>4</sup> Chen Chen<sup>2</sup>

<sup>1</sup>Shandong University <sup>2</sup>Center for Research in Computer Vision, University of Central Florida <sup>4</sup>Amazon Prime Video

<sup>3</sup>Key Laboratory of Machine Perception, Peking University, Shenzhen Graduate School

qitaozhao@mail.sdu.edu.cn cezheng@knights.ucf.edu nkliuyifang@gmail.com

pichaowang@gmail.com

chen.chen@crcv.ucf.edu

## Abstract

Recently, transformer-based methods have gained sig- <sup>0.9</sup> nificant success in sequential 2D-to-3D lifting human pose estimation. As a pioneering work, PoseFormer captures spatial relations of human joints in each video frame and human dynamics across frames with cascaded transformer layers and has achieved impressive performance. However, in real scenarios, the performance of PoseFormer and its follow-ups is limited by two factors: (a) The length of the input joint sequence; (b) The quality of 2D joint detection. Existing methods typically apply self-attention to allframes ofthe input sequence, causing a huge computational burden when theframe number is increased to obtain advanced estimation accuracy, and they are not robust to noise naturally brought by the limited capability of2Djoint detectors. In this paper, we propose PoseFormerV2, which exploits a compact representation oflengthy skeleton sequences in the frequency domain to efficiently scale up the receptive field and boost robustness to noisy 2D joint detection. With minimum modifications to PoseFormer, the proposed method effectively fuses features both in the time domain and frequency domain, enjoying a better speed-accuracy trade-off than its precursor. Extensive experiments on two benchmark datasets (i.e., Human3.6M and MPI-INF-3DHP) demonstrate that the proposed approach significantly outperforms the original PoseFormer and other transformer-based variants. Code is released at https://github.com/ QitaoZhao/PoseFormerV2.

## 1. Introduction

3D human pose estimation (HPE) aims at localizing human joints in 3-dimensional space based on monocular videos (without intermediate 2D representations) [25,28] or 2D human joint sequences (referred to as 2D-to-3D lifting approaches) [5,18,37,43]. With the large availability of 2D human pose detectors [6, 26] plus the lightweight nature of 2D skeleton representation of humans, lifting-based methods are now dominant in 3D human pose estimation. Compared to raw monocular videos, 2D coordinates of human joints in each video frame are much more memory-friendly, making it possible for lifting-based methods to utilize a long joint sequence to boost pose estimation accuracy.

![](images/ba656e9713e79b8b8e967b4abf6b9ef69031adf5a923af0bcfd8559bd31a32e6.jpg)  
Figure 1. Comparisons of PoseFormerV2 and PoseFormerV1 [46] on Human3.6M [13]. RF denotes Receptive Field and k×RF indicates that the ratio between the full sequence length and the number of frames as input into the spatial encoder of PoseFormerV2 is k, i.e., the RF of the spatial encoder is expanded by k× with a few low-frequency coefficients of the full sequence. The proposed method outperforms PoseFormerV1 by a large margin in terms of speed-accuracy trade-off, and the larger k brings more significant improvements, e.g., 4.6× speedup with the k of 27.

Transformers [36] first gain huge success in the field of natural language processing (NLP) [3, 8] and then extend their capacity to the computer vision community, becoming the de facto approach for several vision tasks, e.g., image classification [9, 19, 35], object detection [4, 47] and video recognition [1,2,42]. The discreteness of human joint representation and the requirement for long-range temporal dependency modeling in a skeleton sequence make transformers an excellent fit for lifting-based human pose estimation. Previous works [15, 16, 32, 44, 46] have adopted transformers as the backbone for 3D human pose estimation and shown promising results.

![](images/0458882d3d801664b33f3f32f00757ecf61973d1bf24277e3d51f0084f118123.jpg)  
Figure 2. Overview of PoseFormerV1. PoseFormerV1 mainly consists of two modules: the spatial transformer encoder and the temporal transformer encoder. The temporal encoder of Pose-FormerV1 applies self-attention to all frames given a 2D joint sequence for human motion modeling.

Table 1. The computational cost and performance drop brought by replacing ground-truth 2D detection with CPN [6] 2D pose detection for the SOTA transformer-based methods. The performance drop is reported on Human3.6M dataset (Protocol 1) [13]. RF: Receptive Field, sharing the same meaning as that in Fig. 1.
<table><tr><td colspan="2">Method</td><td>Seq. Length</td><td>GFLOPs</td><td>Perform. Drop (mm)</td></tr><tr><td colspan="2">PoseFormerV1 [46]</td><td>81</td><td>1.36</td><td>13.0</td></tr><tr><td rowspan="3">StridedTransformer [15] MixSTE [44]</td><td>ICCV&#x27;21 TMM&#x27;22</td><td>243</td><td>1.37</td><td>15.2</td></tr><tr><td>CVPR&#x27;22</td><td>81</td><td>92.46</td><td>16.5</td></tr><tr><td>CVPR&#x27;22 ECCV&#x27;22</td><td>81</td><td>3.12</td><td>11.8</td></tr><tr><td colspan="2">P-STMO [32] PoseFormerV2 (9×RF) PoseFormerV2 (27× RF)</td><td>243 81</td><td>1.74 0.35</td><td>13.5 8.2</td></tr></table>

As the pioneering work among transformer-based methods, PoseFormer [46] factorizes joint sequence feature extraction into two stages (see Fig. 2) and outperforms traditional convolution-based approaches. First, all joints within each frame are linearly projected into high-dimensional vectors (i.e., joint tokens) as input into the spatial transformer encoder. The spatial encoder builds up inter-joint dependencies in single frames with the self-attention mechanism. In the second stage, joint tokens of each frame are combined as one frame token, serving as input to the temporal encoder for human motion modeling across all frames in sequence. More details are included in Sec. 3.1.

Despite its capacity, the performance of PoseFormer (and other transformer-based methods) is limited by two crucial factors. (a) The length (number of frames) of the input 2D skeleton sequence. State-of-the-art transformerbased methods typically use extremely long sequences to obtain advanced performance, e.g., 81 frames for Pose-Former [46], 243 frames for P-STMO [15] and 351 frames for MHFormer [16]. However, densely applying selfattention to such long sequences is highly computationally expensive, e.g., the single-epoch wall-time training cost of 3-frame PoseFormer is ∼5 minutes while for 81- frame PoseFormer the cost surges to ∼1.5 hour on an RTX 3090 GPU. (b) The quality of 2D joint detection. 2D joint detectors inevitably introduce noise due to bias in their training dataset and the temporal inconsistency brought by the single-frame estimation paradigm. For example, Pose-Former achieves 31.3mm MPJPE (Mean Per Joint Position Error) using the ground-truth 2D detection on the Human3.6M dataset [13]. This result drops significantly to 44.3mm when the clean input is replaced by the CPN [6] 2D pose detection. In practice, the long-sequence inference may be unaffordable for hardware deployment on resourcelimited devices such as AR/VR headsets and high-quality 2D detection is hard to obtain. More quantitative results about the efficiency to process long sequences and the robustness to noisy 2D joint detection of existing transformerbased methods are available in Table 1.

Driven by these practical concerns, we raise two important research questions:

• Q1: How to efficiently utilize long joint sequences for better estimation precision?

• Q2: How to improve the robustness of the model against unreliable 2D pose detection?

Few works have tried to answer either of these two questions by incorporating hand-crafted modules, e.g., the downsampling-and-uplifting module [10] that only processes a proportion of video frames for improved efficiency, the multi-hypothesis module [16] to model the depth ambiguity of body parts and the uncertainty of 2D detectors. However, none of them manages to find a single solution to these two questions simultaneously, and even worse, a paradox seemingly exists between solutions to the questions above, e.g., multiple hypotheses [16] improve robustness but bring additional computation cost (see also Table 1).

In this paper, we present our initial attempt to “kill” two birds with one stone. With restrained modifications to the prior art PoseFormer, we show that the appropriate form of representation for input sequences might be the key to answering these questions simultaneously. Specifically, we shed light on the barely explored frequency domain in 3D HPE literature and propose to encode the input skeleton sequences into low-frequency coefficients. The insight behind this representation is surprisingly simple: On the one hand, low-frequency components are enough to represent the entire visual identity [38, 41] (e.g., 2D images in image compression and joint trajectories in this case), thus removing the need for expensive all-frame self-attention; On the other, the low-frequency representation of the skeleton sequence itself filters out high-frequency noise (jitters and outliers) [21, 22] contained in detected joint trajectories.

We inherit the spatial-temporal architecture from Pose-Former but force the spatial transformer encoder to only “see” a few central frames in a long sequence. Then we complement “short-sighted” frame-level features (the output of the spatial encoder) with global features from lowfrequency components of the complete sequence. Without resorting to the expensive frame-to-frame self-attention for all time steps, the temporal transformer encoder is reformulated as a Time-Frequency Feature Fusion module.

Extensive experiments on two 3D human pose estimation benchmarks (i.e., Human3.6M [13] and MPI-INF-3DHP [23]) demonstrate that the proposed approach, dubbed as PoseFormerV2, significantly outperforms its precursor (see Fig. 1) and other transformer-based variants in terms of speed-accuracy trade-off and robustness to noise in 2D joint detection. Our contributions are three-fold:

• To the best of our knowledge, we are the first to utilize a frequency-domain representation of input joint sequences for 2D-to-3D lifting HPE. We find this representation an ideal fit to concurrently solve two important issues in the field (i.e., the efficiency to process long sequences and the robustness to unreliable joint detection), and experimental evidence shows that this approach can easily generalize to other models.

• We design an effective Time-Frequency Feature Fusion module to narrow the gap between features in the time domain and frequency domain, enabling us to strike a flexible balance between speed and accuracy.

• Our PoseFormerV2 outperforms other transformer-based methods in terms of the speed-accuracy trade-off and robustness on Human3.6M and achieves the state-of-the-art on MPI-INF-3DHP.

## 2. Related Work

Our method is built on conceptually simple PoseFormer [46], and we aim at improving its efficiency to operate long sequences and its robustness to noisy joint detection from a frequency-domain perspective. Therefore, here we mainly focus on this line of works (transformer-based methods) in 2D-to-3D lifting HPE and introduce applications of frequency domain representations in computer vision literature, especially in skeleton-based tasks that are most related to lifting-based 3D HPE.

## 2.1. Transformer-based 3D Human Pose Estimation

PoseFormer [46] is the first work to adopt the vision transformer as the backbone network in lifting-based 3D human pose estimation, and it outperforms previous CNNbased methods by a large margin. Though being competitive, Zhang et al. [44] point out that the spatial-thentemporal paradigm of PoseFormer may neglect distinct temporal patterns for each joint, and propose to adopt alternate spatial-temporal transformer layers for fine-grained jointspecific feature extraction. MHFormer [16] further incorporates task-related prior knowledge into transformers for 3D HPE. Specifically, 2D-to-3D lifting is an inverse problem where more than one reasonable solutions exist, therefore they generate multiple hypotheses to model ambiguous body parts and uncertainty in joint detectors, achieving advanced performance. Inspired by the progress of Masked Image Modeling (MIM) in image classification [12, 39, 40], P-STMO [32] applies Masked Joint Modeling to 3D HPE with self-supervised learning.

Another line of works [10, 15] improves the efficiency of transformer-based methods. Taking advantage of the temporal redundancy in 2D joint sequences, StridedTransformer [15] replaces the parameter-heavy fully-connected layers with strided convolutions. Einfalt et al. [10] claim that the per-frame 2D joint detection is even more computationally expensive than lifting models themselves and propose to downsample input video frames with a fixed interval and adopt the 2D joint detector and lifting model only on these sampled frames. While being more efficient than previous works, aforementioned methods [10, 15] reduce participants in self-attention along the temporal dimension utilizing only the consistency in adjacent video frames rather than from a global view, and therefore they may suffer from a considerable performance drop.

## 2.2. Frequency Representation in Vision

Since the human visual system is more sensitive to lowfrequency components of images, traditional image compression algorithms, e.g., JPEG [30] and JPEG 2000 [33], reduce memory cost to store 2D images by allocating more storage budget to low-frequency Discrete Cosine Transform (DCT) coefficients of the image. Following the same logic, [41] proposes to adaptively remove uninformative channels of DCT components for 2D images to boost image classification efficiency. More recently, some works [11, 31] propose to replace the costly self-attention mechanism with frequency transforms that can be accelerated by their fast algorithms. GFN [31] proposes to efficiently mix visual tokens with learnable frequency filters, and AFNO [11] further improves the performance of token mixer in the frequency domain with operator learning. Moreover, Wang et al. [38] utilize low-frequency Fast Fourier Transform (FFT) components to compress vision transformers.

Skeleton-based tasks are more relevant to our work that takes 2D skeleton sequences as input. In the human motion prediction literature, previous works [21, 22] transform the skeleton sequence from the time domain into DCT coefficients to encode human dynamics as compared to static joint coordinates. They observe that discarding a few highfrequency coefficients does not necessarily bring a performance drop but even improves the smoothness of predicted future motions. However, frequency-domain representations of 2D joint sequences have not yet been explored in lifting-based 3D human pose estimation.

Our approach is inspired by these former attempts of applying frequency transforms to vision tasks but from a different view. We include more details about our motivations to choose the DCT coefficient representation in Sec. 3.2.1.

## 3. Method

PoseFormer [46] facilitates 3D human pose estimation by factoring sequence feature extraction into two stages, $i . e .$ , the spatial encoder and temporal encoder, which has proved to be effective. However, it suffers from a huge computational burden when the length of input sequences is increased, and is sensitive to noisy joint detection. In this section, we introduce the details of PoseFormerV2 which utilizes a frequency representation of the input sequence to overcome two aforementioned problems.

## 3.1. Preliminaries of PoseFormerV1 [46]

We start by giving a brief overview of PoseFormerV1 (see Fig. 2), laying the basis for the discussions on its improvements. PoseFormerV1 consists of two main modules, the spatial encoder for single-frame joint correlation modeling and the temporal encoder for cross-frame human motion modeling. Given an input 2D skeleton sequence $\mathbf { x } \in \mathbb { R } ^ { F \times J \times 2 }$ , where $F$ denotes the sequence length and $J$ denotes the joint number of the human representation. First, coordinates of all joints of a person in each frame are linearly projected to a c-dimensional vector $( i . e .$ , joint embedding) denoted as ${ \bf z } _ { 0 } \in \mathbb { R } ^ { F \times J \times c }$ . A learnable spatial positional embedding $\mathbf { E } _ { S P o s } \in \mathbb { R } ^ { 1 \times J \times c }$ [9] is added to $\mathbf { z } _ { 0 }$ to encode joint-dependent information.

Spatial transformer encoder builds up spatial dependencies for joint embeddings of each frame $\mathbf { z } _ { 0 } ^ { i } \in \mathbb { R } ^ { 1 \times J \times c }$ individually with the self-attention mechanism. In this stage, the number of tokens fed into each transformer block is $J .$ The output of the spatial transformer encoder of $L$ layers for the i-th frame is denoted by $\mathbf { z } _ { L } ^ { i } \in \mathbb { R } ^ { 1 \times J \times c }$ . Then per-frame representations are flattened and concatenated as input $\mathbf { Z } _ { 0 } \in \bar { \mathbb { R } } ^ { F \times ( J \cdot c ) }$ into the temporal transformer encoder.

Temporal transformer encoder. Similarly, the input $\mathbf { Z } _ { 0 }$ is added with a learnable temporal positional embedding $\mathbf { E } _ { T P o s } \in \mathbb { R } ^ { F \times ( J \cdot c ) }$ to encode index-dependent information for each frame. The temporal encoder with M transformer layers densely models frame-to-frame dependencies across the whole sequence, and its output is denoted by $\mathbf { Z } _ { M } \in \mathbb { R } ^ { F \times ( J \cdot c ) }$ . In this stage, the token number for each transformer layer is $F$ , which is the input sequence length.

Regression head. To estimate the 3D pose of the central frame in sequence, a simple 1D convolution is used to gather temporal information and a linear projection outputs the final pose representation $\mathbf { y } \in \mathbb { R } ^ { 1 \times ( J \cdot 3 ) }$

![](images/2cc303397f35a8fd019593356f654f5071f490d5c252aa5702fcdd332f505b5f.jpg)

![](images/2cfc5f6dfb263d825376ef7741f68f751a99d0abe821cbb579a797ede72784d6.jpg)  
Figure 3. A randomly selected example of the CPN-detected [6] joint trajectory in Human3.6M [13] and its reconstructions with first 3, 9, and 27 DCT coefficients (81 in total). Note that even with only the first 3 coefficients, the reconstructed (orange) curve captures the overall characteristics of the raw input, and is smoother.

Limitations of PoseFormerV1. Modeling joint dependencies within each frame and human motions across frames with transformer layers is straightforward. While such dense modeling brings advanced estimation accuracy, it is computationally unfriendly due to the quadratic computation growth of self-attention with respect to the token number $( i . e .$ , the joint number in the spatial encoder and the sequence length in the temporal encoder) especially when the input sequence length is increased. Although the token number for spatial transformer layers $( i . e .$ , the joint number) is independent of the frame number, it is worth noting that the sequence length implicitly affects the computational budgets of the spatial encoder in real scenarios because of the limited parallelization ability of GPUs. In addition to the efficiency issue, PoseFormerV1 is sensitive to the qual ity of input 2D joint detection (experimental evidence is available in Table 1 and Sec. 4.3).

In the following, we present an alternative solution to overcome the limitations of PoseFormerV1 with the frequency-domain representation of the input sequence.

## 3.2. PoseFormerV2

## 3.2.1 Frequency Representation of Skeleton Sequence

Motivation. We propose to transform the input skeleton sequence into the frequency domain with Discrete Cosine Transform (DCT) and utilize only a portion of lowfrequency coefficients. DCT coefficients encode multiple levels of temporal information for the input time series. Specifically, low-frequency coefficients encode its rough contour while high-frequency ones encode its details, $e . g .$ jitters or sharp changes. To better illustrate our motivation to choose this representation, we provide an 81-frame example of the CPN-detected [6] joint trajectory of action “Directions” in the Human3.6M [13] dataset and its reconstructions with first 3, 9, and 27 DCT coefficients respectively (see Fig. 3). As the number of kept DCT coefficients increases, the reconstructed trajectory becomes closer to the raw input but less smooth. Note that with only 3 DCT coefficients (denoted by the orange curve), the overall trend of the original trajectory is captured, and with 9 and 27 coefficients (pink and green curves), the characteristics of the raw sequence are better preserved while high-frequency noise (zig-zags) is removed. These observations motivate us to exploit a few highly informative low-frequency DCT components of the input joint sequence as the compact and denoised sequence representation in our work. With such representation, we significantly reduce the effective length of the sequence as input and promote the robustness of our model against the noise contained in 2D joint detection. We include a formal introduction to DCT in supplementary.

![](images/dc34f1ee98c262a220c697a9c9c5caa8475811c7103b307a503ee5df60707577.jpg)  
Figure 4. (a) Overview of PoseFormerV2. (b) Spatial Transformer and Time-Frequency Feature Fusion Transformer. (c) FreqMLP (Frequency Multi-Layer Perceptron). To exemplify, in (a) we use 3 central frames (index -1, 0, and 1) for fine-grained frame-level feature extraction and the first 3 DCT coefficients of the full 9-frame sequence for global frequency-domain feature extraction. Therefore, the effective number of frames as input to the spatial encoder and temporal encoder is reduced compared to PoseFormerV1 (3,6 vs. 9,9).

## 3.2.2 Architecture

In this part, we introduce the architecture of the proposed approach, PoseFormerV2 (see Fig. 4 for an overview).

Spatial transformer encoder. Given a 2D skeleton sequence $\mathbf { x } ~ \in ~ \mathbb { R } ^ { F \times J \times 2 }$ (preferably a long sequence, $e . g .$ $F$ is 81), we first sample $F ^ { \prime }$ (typically $F ^ { \prime } \ll F )$ frames around the sequence center (the frame of index 0 in Fig. 4 (a)), denoted by $\mathbf { x } ^ { \prime } \in \mathbb { R } ^ { F ^ { \prime } \times J \times 2 }$ , as input to the spatial encoder. The output of the spatial encoder is denoted by $\mathbf { z } ^ { T i m e } \in \mathbb { R } ^ { F ^ { \prime } \times ( \bar { J ` } c ) }$ (frame-level features in the time domain). The design of the spatial encoder directly follows PoseFormerV1.

Low-frequency DCT coefficients. $\mathbf { z } ^ { T i m e }$ is referred to as “short-sighted” because its receptive field $( F ^ { \prime } )$ is restricted in comparison to the entire sequence length (F). To efficiently exploit the long-range human dynamics of the original sequence, we resort to its frequency-domain representation. We first convert the full sequence $\textbf { x } \in$ $\mathbb { R } ^ { \sum _ { F \times J \times 2 } ^ { \star } }$ into DCT coefficients, denoted by $\dot { \mathbf { C } } \in \mathbb { R } ^ { F \times J \times 2 }$ Then we keep only the first $N \ ( \ll \ F )$ coefficients ${ \bf C } ^ { \prime } \in  { \bf \Sigma }$ $\mathbb { R } ^ { N \times J \times 2 }$ using a low-pass filter for every joint trajectory, where temporal information of the original sequence is largely maintained and high-frequency noise is removed.

Low-frequency coefficients $\mathbf { C ^ { \prime } }$ are flattened and linearly projected to $\mathbf { \bar { z } } ^ { F r e q } ~ \in ~ \mathbb { R } ^ { N \times ( J \cdot c ) }$ (the embedding of frequency coefficients). ${ \mathbf z } ^ { F r e q }$ is summed with a learnable frequency positional embedding $\mathbf { E } _ { F P o s }$ (like $\mathbf { E } _ { T P o s }$ in Pose-FormerV1). Features from both the time domain and frequency are concatenated together, formulated as

$$
{ \bf z } = [ { \bf z } ^ { T i m e } ; { \bf z } ^ { F r e q } ] ,\tag{1}
$$

fed to the Time-Frequency Feature Fusion module.

Time-Frequency Feature Fusion. We adopt transformer layers for cross-frame temporal dependency modeling as in PoseFormerV1. Compared to PoseFormerV1 which entirely extracts features in the time domain, the proposed method fuses features from both the time domain and frequency domain. To narrow the gap between the two domains, we introduce simple modifications to vanilla transformer layers. (1) Time-domain and frequency-domain features share self-attention but use separate feed-forward networks; (2) We apply FreqMLP (Frequency Multi-Layer Perceptron) in the feed-forward networks for time-domain features $\mathbf { z } ^ { T i m e }$ (see Fig. 4 (b)(c)). In our FreqMLP, we utilize DCT and IDCT before and after the vanilla MLP. The intuition behind this approach is: High-frequency noise is filtered out of frequency domain features with a low-pass filter, but detailed human motion features (e.g., fast local motions) may also be lost as noise. To address this issue, FreqMLP acts as a trainable frequency-domain filter, allowing us to adaptively adjust the weight of each frequency component in the embedding of 2D joint coordinates $( i . e . ,$ time-domain features), being a complement to frequency features. These modules are formulated as:

$$
\mathbf { z } _ { k } ^ { ' } = \mathrm { M S A } ( \mathbf { z } _ { k } ) ,\tag{2}
$$

$$
\begin{array} { r } { \mathbf { z } _ { k } ^ { T i m e } , \mathbf { z } _ { k } ^ { F r e q } = \mathbf { z } _ { k } ^ { ' } [ : F ^ { \prime } ] , \mathbf { z } _ { k } ^ { ' } [ F ^ { \prime } : ] , } \end{array}\tag{3}
$$

$$
\mathbf { z } _ { k + 1 } = \mathrm { C o n c a t } ( \mathrm { F r e q M L P } ( \mathbf { z } _ { k } ^ { T i m e } ) , \mathrm { M L P } ( \mathbf { z } _ { k } ^ { F r e q } ) ) ,\tag{4}
$$

where MSA denotes Multi-head Self-Attention and $F ^ { \prime }$ is the number of sampled central frames. The effectiveness of the aforementioned modifications is verified in Sec. 4.4. It’s important to recognize that the concatenation operation results in a higher number of tokens for the transformer. However, by restricting the spatial encoder to only observe a limited number of central frames and incorporating a small percentage of low-frequency DCT coefficients to expand its receptive field, we can decrease the overall computation in a flexible manner. This approach not only reduces computational costs but also enhances the model’s resistance to noise compared to PoseFormerV1.

Regression head and loss function. Following Pose-FormerV1, we use the 1D convolution layer to gather temporal information and a linear projection to obtain the final 3D pose $\mathbf { y } \in \mathbb { R } ^ { 1 \times ( J \cdot 3 ) }$ for the central frame of the sequence. We use the standard MPJPE (Mean Per Joint Position Error) loss as PoseFormerV1 to train our model.

## 4. Experiments

## 4.1. Datasets and Evaluation Metrics

We conduct experiments on two 3D human pose estimation datasets, i.e., Human3.6M [13] and MPI-INF-3DHP [23] to demonstrate the effectiveness of our method. More detailed descriptions of both datasets and their respective evaluation metrics are in the supplementary.

## 4.2. Implementation Details and Analysis

The proposed method includes three important hyperparameters that are specific to experimental settings. These include the number of frames (f) used as input in the spatial encoder, the length of the entire input sequence (F) representing the enlarged receptive field, and the number of kept DCT coefficients (n) utilized to incorporate long-range temporal information. If not specified, we simply set $n = f$ for convenience. In practice, they can be further tuned for a flexible speed-accuracy trade-off. When f equals 1, n is set to 3 because a single DCT coefficient may be insufficient to encode temporal information from lengthy input sequences. As $f$ and n are fixed, the computational complexity of the model is predetermined (i.e., the token number for the spatial encoder and that for the feature-fusion module are fixed). We may vary $F$ to effectively expand the model’s receptive field from a limited f to an arbitrary value, bringing no additional computational overhead. This enables us to efficiently use long sequences to improve accuracy. We provide details of the hyper-parameters for model architecture and training in the supplementary.

## 4.3. Comparisons with State-of-the-art Methods

Human3.6M. We compare our method with Pose-FormerV1 and other transformer-based methods on Human3.6M (Table 2). We demonstrate the flexibility of our model by varying the value of $f$ and the sequence length. Our method is particularly efficient when the expanding ratio $( i . e . ,$ , the ratio of full sequence length to f) is large. For example, with an expanding ratio of 81, it achieves 47.6mm MPJPE with only 77.2 MFLOPs as compared to the 47.8mm MPJPE of MHFormer [16] with 342.9 MFLOPs (4.4× slower). Moreover, with a similar computational budget (around 350 MFLOPs) and the same full sequence length (81), our method achieves 46.0mm MPJPE whereas StridedTransformer [15] obtains 47.5mm MPJPE (3.2%↑). Fig. 5 presents a clearer comparison, showing that the proposed method outperforms other transformer-based methods in terms of speed-accuracy trade-off. Note that the methods with an additional pre-training stage and computationally heavy MixSTE [44] (3420 MFLOPs for only 3- frame input) are not included. The improvements of Pose-FormerV2 over PoseFormerV1 are provided in Fig. 1.

Table 2. Quantitative comparisons with previous transformerbased methods on Human3.6M (in mm). $f \colon$ number of frames as input to the model, Seq. Len.: length of the entire input sequence $( i . e .$ , the effective Receptive Field). The best scores are marked in bold. (\*) indicates using an additional pre-training stage and (†) indicates our re-implementation.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>f</td><td rowspan=1 colspan=1>Seq.Len.</td><td rowspan=1 colspan=1>MFLOPs</td><td rowspan=1 colspan=1>MPJPE↓/P-MPJPE↓</td></tr><tr><td rowspan=6 colspan=1>PoseFormerV1 [46] ICCV&#x27;21StridedTrans. [15] TMM&#x27;22MixSTE [44](†) CVPR’22MHFormer [16] CVPR&#x27;22MHFormer [16] CVPR’22P-STMO [32](*) ECCV’22P-STMO [32](*) ECCV’22Einfalt et al. [10] WACV&#x27;23</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>542.1</td><td rowspan=1 colspan=1>47.0/-</td></tr><tr><td rowspan=1 colspan=1>81</td><td rowspan=1 colspan=1>81</td><td rowspan=1 colspan=1>342.5</td><td rowspan=5 colspan=1>47.5/-49.6/38.947.8/-45.9/-46.8/-45.6/-47.9/-</td></tr><tr><td rowspan=4 colspan=1>392727819</td><td rowspan=3 colspan=1>3927</td><td rowspan=1 colspan=1>3420</td></tr><tr><td rowspan=1 colspan=1>342.9</td></tr><tr><td rowspan=1 colspan=1>1031.8</td></tr><tr><td rowspan=1 colspan=1>818181</td><td rowspan=1 colspan=1>163493543</td></tr><tr><td rowspan=1 colspan=1>PoseFormerV2PoseFormerV2PoseFormerV2</td><td rowspan=1 colspan=1>111</td><td rowspan=1 colspan=1>92781</td><td rowspan=1 colspan=1>77.277.277.2</td><td rowspan=1 colspan=1>49.9/38.748.7/37.847.6/37.3</td></tr><tr><td rowspan=1 colspan=1>PoseFormerV2PoseFormerV2PoseFormerV2</td><td rowspan=1 colspan=1>333</td><td rowspan=1 colspan=1>92781</td><td rowspan=1 colspan=1>117.3117.3117.3</td><td rowspan=1 colspan=1>49.5/38.547.9/37.447.1/37.3</td></tr><tr><td rowspan=1 colspan=1>PoseFormerV2PoseFormerV2</td><td rowspan=1 colspan=1>99</td><td rowspan=1 colspan=1>2781</td><td rowspan=1 colspan=1>351.7351.7</td><td rowspan=1 colspan=1>47.6/37.146.0/36.1</td></tr><tr><td rowspan=1 colspan=1>PoseFormerV2</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>243</td><td rowspan=1 colspan=1>1054.8</td><td rowspan=1 colspan=1>45.2/35.6</td></tr></table>

![](images/e6f83f756239388cfcde477e74fcac1b5aa23d88eac452c18078b0b126f5a212.jpg)  
Figure 5. Comparisons of PoseFormerV2 and other state-of-theart transformer-based methods on Human3.6M (in mm). RF: Receptive Field and $k { \times } \mathrm { R F }$ indicates that the RF of PoseFormerV2 is expanded by $k \times$ with a few low-frequency DCT coefficients of the full sequence. The proposed approach outperforms others in terms of speed-accuracy trade-off, and the larger $k ,$ the larger improvements over other methods. (Best viewed in color)

![](images/d960e8f3a09ecce37349bb023ede13022d7fe8d6d5e7a39c19ae0f1c0eaa72bf.jpg)  
Figure 6. Comparisons of PoseFormerV2 and other transformerbased methods [16, 44, 46] in terms of robustness to noise on Human3.6M. Zero-mean Gaussian noise of standard deviation sigma is added to ground truth 2D detection, and we show their performance drop (△MPJPE, in mm) as sigma increases. The size of markers indicates the computational cost of models.

Table 3. Quantitative comparisons with previous methods on MPI-INF-3DHP. T: the entire sequence length, 1 by default. The best scores are marked in bold. (\*) indicates using an additional pretraining stage and (†) indicates our re-implementation.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>PCK↑  AUC ↑  MPJPE↓</td></tr><tr><td rowspan=11 colspan=1>Mehta et al. [23]Mehta et al. [24]Pavllo et al. [29] (T=81)Pavllo et al. [29] (T=243)Lin et al. [17] (T=25)Li et al. [14]Chen et al. [5] (T=81)PoseFormerV1 [46] (T=9)(†)MHFormer [16] (T=9)MixSTE [44] (T=27)P-STMO [32] (T=81)(*)</td><td rowspan=3 colspan=1>3DV&#x27;17ACM ToG&#x27;17CVPR&#x27;19</td><td rowspan=1 colspan=1>75.7    39.3     117.6</td></tr><tr><td rowspan=1 colspan=1>76.6    40.4     124.7</td></tr><tr><td rowspan=1 colspan=1>86.0    51.9     84.0</td></tr><tr><td rowspan=6 colspan=1>CVPR&#x27;19BMVC&#x27;19CVPR’20TCSVT&#x27;21ICCV’21CVPR&#x27;22</td><td rowspan=1 colspan=1>85.5    51.5     84.8</td></tr><tr><td rowspan=1 colspan=1>83.6    51.4     79.8</td></tr><tr><td rowspan=1 colspan=1>81.2    46.1     99.7</td></tr><tr><td rowspan=1 colspan=1>87.9    54.0     78.8</td></tr><tr><td rowspan=1 colspan=1>95.4    63.2     57.7</td></tr><tr><td rowspan=1 colspan=1>93.8    63.3     58.0</td></tr><tr><td rowspan=2 colspan=1>CVPR&#x27;22ECCV&#x27;22</td><td rowspan=1 colspan=1>94.4    66.5     54.9</td></tr><tr><td rowspan=1 colspan=1>97.9    75.8     32.2</td></tr><tr><td rowspan=1 colspan=1>PoseFormerV2 (T=81)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>97.9    78.8     27.8</td></tr></table>

In order to demonstrate that the inclusion of lowfrequency DCT coefficients helps improve the robustness of the proposed method, we make the lifting-based pose estimation task more challenging by adding zero-mean Gaussian noise to the ground-truth 2D detection on the Human3.6M dataset [13] (Fig. 6). To ensure a fair comparison, we keep the input sequence length the same for all methods (in this case, 27 frames). For our method, $f = n = 3 .$ The experimental evidence reveals that PoseFormerV2 suffers from less performance drop as the standard deviation of Gaussian noise (sigma) increases while being more efficient. We observe that the performance of PoseFormerV1 drops drastically as sigma increases from 8 to 10. In contrast, the proposed method presents a more stable trend. Moreover, our method even outperforms MHFormer [16] that incorporates the uncertainty of 2D detectors into the model design. Intriguingly, we find that minor noise may improve the accuracy of 3D pose estimation (sigma = 3).

Table 4. Ablation study on several modifications to Pose-FormerV1. We show how a 9-frame PoseFormerV1 is converted to PoseFormerV2 (with 9 DCT coefficients from an 81-frame sequence) step by step. The evaluation is performed on Human3.6M (Protocol 1, in mm). RF indicates Receptive Field.
<table><tr><td>Step</td><td>Description</td><td>RF</td><td>MPJPE↓</td></tr><tr><td>(0)</td><td>Original 9-frame PoseFormerV1.</td><td>9</td><td>49.9</td></tr><tr><td>(1)</td><td>Frames are sampled from a longer sequence.</td><td>9</td><td>49.9</td></tr><tr><td>(2)</td><td>Append the embedding of DCT coefficients.</td><td>81</td><td>47.1 (2.8↓)</td></tr><tr><td>(3)</td><td>Replace the vanilla MLP with FreqMLP.</td><td>81</td><td>46.0 (3.9↓)</td></tr></table>

Table 5. Ablation study on the number of frames and the number of DCT coefficients that are used as input to PoseFormerV2. The evaluation is performed on Human3.6M (Protocol 1, in mm).
<table><tr><td>Frame Number (f)</td><td>Coefficient Number (n)</td><td>Full Length</td><td>MFLOPs</td><td>MPJPE</td></tr><tr><td>1</td><td>1</td><td>27</td><td>39.2</td><td>51.1</td></tr><tr><td>1</td><td>3</td><td>27</td><td>77.2</td><td>48.7 (2.4↓)</td></tr><tr><td>3</td><td>1</td><td>27</td><td>79.4</td><td>50.1 (1.0↓)</td></tr><tr><td>3</td><td>3</td><td>27</td><td>117.3</td><td>47.9 (3.2↓)</td></tr><tr><td>9</td><td>9</td><td>27</td><td>351.7</td><td>47.6 (3.5↓)</td></tr></table>

MPI-INF-3DHP. We also compare our method with others on MPI-INF-3DHP [23] (Table 3). We use 9 central frames and the first 9 DCT coefficients from the input 81- frame sequence. The proposed method outperforms other approaches including P-STMO [32] with masked joint pretraining. This result verifies the effectiveness of our method. Our implementation follows [32].

Qualitative comparisons. We provide qualitative comparisons of our method with competitive MHFormer [16] and PoseFormerV1 [46] in Fig. 7. All methods use 81- frame 2D joint sequences as input. To further illustrate the robustness of our approach, we make the pose estimation task more difficult by adding Gaussian noise to the sequential 2D detection of a randomly selected joint (e.g., “left wrist”, “right foot”). The proposed method obtains reliable 3D human pose even under highly-deviated 2D detection (indicated by the light-yellow arrows). Note that our model is ∼9× more efficient than MHFormer (3.12 GFLOPs vs. 0.35 GFLOPs) and ∼4× more efficient compared to Pose-FormerV1 (1.36 GFLOPs vs. 0.35 GFLOPs).

## 4.4. Ablation Study

In this section, we show how a few modifications to PoseFormerV1 bring significant improvements in a step-bystep way. Moreover, to investigate more insights into the frequency-domain representation of input sequences, we reveal the impact of the number of input frames and that of kept DCT coefficients on our method.

Convert PoseFormerV1 into PoseFormerV2. We inherit the overall spatial-temporal architecture from Pose-FormerV1 and introduce restrained modifications to its temporal transformer for better multi-domain feature fusion. To exemplify, we illustrate how a 9-frame PoseFormerV1 is converted to PoseFormerV2 step by step: (1) the input (i.e., 9 frames) is sampled from a longer sequence (e.g., 81 frames) at the sequence center. This step brings no performance improvement or increase in the receptive field since the input to the model is in fact unchanged. (2) The output of the spatial encoder of PoseFormerV1, z<sup>T</sup> <sup>ime</sup>, is appended to the embedding of the first n DCT coefficients (denoted by z<sup>F</sup> <sup>req</sup>) of the complete sequence (81 frames in this case) as input into the temporal encoder. For convenience, we set n to 9. (3) We replace the vanilla MLP for z<sup>T</sup> <sup>ime</sup> (z<sup>T</sup> <sup>ime</sup> and z<sup>F</sup> <sup>req</sup> already use separate vanilla MLPs before replacement) in the temporal encoder with FreqMLP (details in Sec. 3.2.2). PoseFormerV1 is converted to Pose-FormerV2 after these steps, with an enlarged receptive field (from 9 to 81). We present the improvement brought by each step in Table 4. It is worth noting that by introducing 9 DCT coefficients from a longer sequence (i.e., 81 frames), the MPJPE of 9-frame PoseFormerV1 is reduced by 7.8% (49.9mm vs. 46.0mm), which verifies the effectiveness of the proposed DCT representation of input joint sequences.

![](images/bf3b736fe0eda437905f7d17d7536a4a0d18f9c702ab0e2e2d5e35e94923eb58.jpg)  
Figure 7. Qualitative comparisons of PoseFormerV2 with MHFormer [16] and PoseFormerV1 [46]. We randomly add Gaussian noise to the 2D detection of a specific joint. We highlight the deviated 2D detection with light-yellow arrows and corresponding 3D pose estimations with orange arrows. PoseFormerV2 shows better robustness to highly noisy input than existing methods.

Number of input frames and DCT coefficients. In Table 5, we investigate the impact of the number of frames (f) as input to the spatial encoder and the number of retained DCT coefficients (n). Here we keep the length of the entire joint sequence fixed, i.e., 27. The baseline model uses only one central frame and one DCT coefficient $( f = n =$

1). Increasing both parameters brings consistent improvements, and the increase in n translates to more error reduction (2.4↓ for n = 3 vs. 1.0↓ for f = 3) since only a few DCT coefficients help capture the global characteristics of the entire sequence. We empirically find that the matched f and n with an expanding ratio of 9 (i.e., f = n = 3) achieve a satisfactory speed-accuracy trade-off.

## 4.5. Generalization Ability

The proposed frequency-domain approach can generalize to other methods, e.g., MixSTE [44] and MHFormer [16], as they also use transformers for temporal modeling. We improve both methods by incorporating low-frequency DCT coefficients. Details are in supplementary.

## 5. Conclusion

We present a solution to reconcile two seemingly unrelated or even contracted issues in lifting-based 3D human pose estimation – the efficiency of processing longsequence input and the robustness to noisy joint detection – simultaneously from a barely explored frequency-domain perspective. The proposed method, PoseFormerV2, exploits a compact frequency representation of long 2D joint sequences to efficiently enlarge the receptive field of the model while improving its robustness. Experimental results show that our method outperforms previous transformerbased methods on Human3.6M and MPI-INF-3DHP.

## Supplementary Material

## A. Overview

The supplementary material includes sections as follows:

• Section B: A formal introduction to Discrete Cosine Transform.

• Section C: Datasets and evaluation metrics.

• Section D: More implementation details.

• Section E: Comparisons of PoseFormerV2 and a simple baseline model purely in the frequency domain.

• Section F: Generalization of our approach to more models.

• Section G: Visualizations and analysis.

• Section H: Broader impacts and limitations.

## B. Discrete Cosine Transform

We now give a formal introduction to DCT. Given a 2D joint sequence denoted by $\mathbf { x } \in \mathbb { R } ^ { F \times J \times 2 }$ , where F is the sequence length and J is the joint number in each frame, the trajectory of the x (or y) coordinate of the j-th joint denoted as $\mathbf { x } _ { j , 0 } \in \mathbb { R } ^ { F } \left( \operatorname { o r } \mathbf { x } _ { j , 1 } \in \mathbb { R } ^ { F } \right.$ , both denoted by $\hat { \mathbf { x } } _ { j }$ for convenience) is a 1D time series and we apply DCT to each trajectory $( J * 2$ trajectories in total) individually.

For trajectory ${ \hat { \mathbf { x } } } _ { j } ,$ the i-th DCT coefficient is calculated as

$$
\begin{array} { r } { C _ { j , i } = \sqrt { \frac { 2 } { F } } \sum _ { f = 1 } ^ { F } x _ { j , f } \frac { 1 } { \sqrt { 1 + \delta _ { i 1 } } } \cos \left( \frac { \pi } { 2 F } ( 2 f - 1 ) ( i - 1 ) \right) } \end{array}\tag{5}
$$

where $\delta _ { i 1 } = 1$ when $i \ = \ 1$ , otherwise $\delta _ { i 1 } = 0$ . Each time step in trajectory yields one DCT coefficient, $i . e . , i \in$ $\{ 1 , 2 , \cdots , F \}$ . DCT coefficients encode multiple levels of temporal information in the input time series. Specifically, low-frequency coefficients $( i . e .$ , when i is small) encode the rough contour of the input sequence while high-frequency coefficients $( i . e .$ , for the large i) encode details, e.g., jitters or sharp changes in the input sequence. The original input sequence in the time domain can be restored using Inverse Discrete Cosine Transform (IDCT), which is given by

$$
\begin{array} { r } { x _ { j , f } = \sqrt { \frac { 2 } { F } } \sum _ { i = 1 } ^ { F } C _ { j , i } \frac { 1 } { \sqrt { 1 + \delta _ { i 1 } } } \cos \left( \frac { \pi } { 2 F } ( 2 f - 1 ) ( i - 1 ) \right) } \end{array}\tag{6}
$$

and $f \in \{ 1 , 2 , \cdots , F \}$ . DCT is lossless if we keep all its coefficients intact. In practice, we can slightly lossily recover the input sequence using only a few low-frequency coefficients and set other coefficients to zero. It is worth noting that the recovered curve would be smoother compared to the original one since we discard some of the highfrequency coefficients. This property of DCT is desirable – only a small proportion of DCT coefficients are enough to represent the whole input sequence, even in a cleaner manner. This motivates us to use such representation to efficiently operate long sequences while improving the robustness of the model to low-quality 2D detection where highfrequency noise often occurs.

## C. Datasets and Evaluation Metrics

Human3.6M is the most widely used benchmark for 3D human pose estimation. Over 3.6 million video frames are captured indoors from 4 cameras at different places. This dataset contains 11 subjects performing 15 different actions, $e . g .$ “Walking” and “Phoning”. We train our model on 5 subjects (S1, S5, S6, S7, S8) and use other 2 subjects (S9, S11) for testing, following [5, 18, 29, 46].

MPI-INF-3DHP is collected in both controlled indoor environments and challenging outdoor environments. It also provides different subjects and actions from multiple camera views similar to Human3.6M.

Evaluation Metrics. We report two common metrics, MPJPE and P-MPJPE [45] on Human3.6M. MPJPE (Mean Per Joint Position Error, referred to as Protocol 1) measures the mean Euclidean distance between the estimated 3D pose and the ground truth 3D pose. P-MPJPE (Protocol 2) applies a rigid transformation to the estimated 3D pose and the distance is computed between the aligned estimated 3D pose and the ground truth 3D pose.

For the MPI-INF-3DHP dataset, we report MPJPE, Percentage of Correct Keypoint (PCK) within the 150mm range, and Area Under Curve (AUC) as in [5, 17, 37].

## D. More Implementation Details

Our method is built upon PoseFormerV1 [46]. Aiming at better demonstrating the effectiveness of our DCT coefficient representation of input sequences and providing fair comparisons to PoseFormerV1, we directly adopt optimal hyper-parameters for model architecture from Pose-FormerV1, although further investigation may bring additional improvements.

Model architecture hyper-parameters. The embedded feature dimension c in the spatial transformer is 32 and the layer number of the spatial transformer and feature-fusion transformer is 4, following [46]. Plus, the design of Spatial-Temporal Positional Embedding is also adopted from [46].

Experimental settings. Our experiments are conducted with Pytorch [27] on a single NVIDIA RTX 3090. For both training and testing, we apply horizontal flipping augmentation following [5, 18, 29, 46]. We train our model using the AdamW [20] optimizer for 80 epochs with a weight decay of 0.1. The initial learning rate is set to 8e-4 with an exponential learning rate decay schedule and the decay factor is 0.99. We adopt the CPN [7] 2D pose detection on Human3.6M, following [5, 18, 29]. As for the MPI-INF-3DHP dataset, we use ground truth 2D detection, following [17, 24].

## E. Simple Baseline

In our approach, the temporal encoder of PoseFormerV1 [46] is reformulated as a Time-Frequency Feature Fusion module and we show that the low-frequency coefficients of the input sequence help improve the efficiency of the model to process long sequences and its robustness against noisy joint detection. Given the effectiveness of this representation, readers may raise a question: Why not entirely extract features from DCT coefficients of the input sequence but additionally combine them with features in the time domain? Here we design a baseline model where we simply replace the input to PoseFormerV1 [46] (joint coordinates in the time domain) with low-frequency DCT coefficients of the input sequence. The full sequence length and the number of the retained DCT coefficients (denoted as n) are kept the same for the baseline model and our approach. For convenience, the number of frames (f) as input into the spatial encoder of PoseFormerV2 is set to n. We provide quantitative results to demonstrate that this straightforward approach does not work well, especially when the ratio between the full sequence length and n is increased (see Table 6). The features of only a few central frames in the sequence significantly boost accuracy, $e . g .$ , with 3 central frames of the full input sequence of length 81, the MPJPE is reduced from 49.7mm to 47.1mm (5.2%↓, the 3rd row in Table 6).

Table 6. Comparisons of PoseFormerV2 and a simple baseline. The evaluation is performed on Human3.6M (Protocol 1, MPJPE) [13] and the Frame Number (f) is only applicable to PoseFormerV2.
<table><tr><td>Frame Number (f)</td><td>Coefficient Number (n)</td><td>Full Length</td><td>Baseline</td><td>PoseFormerV2</td></tr><tr><td>3</td><td>3</td><td>9</td><td>50.2</td><td>49.5 (0.7↓)</td></tr><tr><td>3</td><td>3</td><td>27</td><td>48.7</td><td>47.9 (0.8↓)</td></tr><tr><td>3</td><td>3</td><td>81</td><td>49.7</td><td>47.1 (2.6↓)</td></tr><tr><td>9</td><td>9</td><td>27</td><td>48.8</td><td>47.6 (1.2↓)</td></tr><tr><td>9</td><td>9</td><td>81</td><td>47.8</td><td>46.0 (1.8↓)</td></tr></table>

Intuitively, the spatial encoder of PoseFormerV2 that encodes joint coordinates of a few central frames in the time domain helps capture the fine-grained human motions, benefiting 3D pose estimation for the frame at the sequence center. In contrast, low-frequency coefficients of the input sequence filter out high-frequency noise and human motion details $( e . g .$ , fast motions) that may be informative to human pose estimation $( i . e .$ , the over-smoothing problem). Therefore, features from the time domain and frequency domain, $i . e .$ , the joint coordinate of central frames and lowfrequency coefficients of the sequence, carry complementary semantics. These considerations necessitate our proposed Time-Frequency Feature Fusion design.

## F. Generalization to More Models

In the main text, we focus on improving PoseFormerV1 [46] from a barely explored frequency-domain perspective. In this part, we show that the proposed frequency-domain approach also generalizes well to other existing state-of-theart methods, $e . g .$ , MixSTE [44] and MHFormer [16]. Since these approaches [16, 44] also apply self-attention along the time dimension to all frames as PoseFormerV1 [46], the proposed method can be easily incorporated into their model without complex redesigns for model architecture. For fair comparisons, we directly adopt optimal hyperparameters $( e . g .$ , the layer number, channel dimension) for these original methods. Further tuning of hyper-parameters may bring additional improvements.

![](images/83d2f5dd387d4a4117380c10d2d1507256539fa10397d5e70a0781864d115066.jpg)  
Figure 8. Comparisons of MixSTE [44] and its improved version with frequency representations of the sequence on Human3.6M [13]. RF: Receptive Field and k×RF indicate that the RF of MixSTE is expanded by k× with a few low-frequency DCT coefficients of the full sequence. The proposed approach helps MixSTE gain a better speed-accuracy trade-off. (Best viewed in color)

![](images/1bb4c0cffec9d4ad83397ca30b1d4b6313841fece6c81d10de2381c399f95d81.jpg)  
Figure 9. Comparisons of MixSTE [44] and its improved version using low-frequency DCT coefficients of the sequence in terms of robustness to noise on Human3.6M [13]. Zero-mean Gaussian noise of standard deviation sigma is added to ground truth 2D detection, and we show their performance drop (△MPJPE) as sigma increases. The size of markers indicates the computational cost of models.

MixSTE [44] adopts the spatial-temporal architecture as PoseFormerV1 [46]. Compared to the spatial-thentemporal paradigm of PoseFormerV1, MixSTE alternately uses spatial and temporal transformer encoders. Similarly, we centrally sample a few video frames from a longer sequence as input into the spatial encoders of MixSTE. For temporal encoders, we append the time-domain features (the output of the spatial encoders) with the embedding of low-frequency coefficients of the complete input sequence. The comparisons of MixSTE and its improved version are presented in Fig. 8, 9. Original MixSTE is highly computationally expensive and our approach improves its efficiency and accuracy simultaneously, e.g., MixSTE achieves 46.2mm MPJPE taking 30.8 GFLOPs, while its improved version achieves 45.3mm MPJPE with 15.4 GFLOPs (2× faster and 1.9%↑ error reduction, see the bright red curve in Fig. 8). We also show that our method improves the robustness of MixSTE against noisy 2D joint detection (Fig. 9). Specifically, we add zero-mean Gaussian noise to the ground-truth 2D joint sequence of 27 frames on Human3.6M [13]. The improved MixSTE (denoted as MixSTE+) uses 3 central frames as input to its spatial encoders and the first 3 DCT coefficients as a cleaner global representation of the full sequence. MixSTE+ suffers from less performance drop while being 6× more efficient (30.8 GFLOPs vs. 5.1 GFLOPs, indicated by the marker size).

![](images/f734e1d043bbb1c8e2a74a59a37952250f04eaa7948dcad212432740081b1040.jpg)  
Figure 10. Comparisons of MHFormer [16] and its improved version using low-frequency DCT coefficients of the sequence in terms of robustness to noise on Human3.6M [13]. Experimental settings follow Fig. 9. The size of markers indicates the computational cost of models.

MHFormer [16] introduces multiple hypotheses into its architecture to model depth ambiguity of body parts and uncertainty of joint detectors and is thus relatively robust (experimental results are available in the main paper). Besides, MHFormer also includes spatial-temporal transformer modules as in PoseFormerV1 [46]. To further verify the universality of our approach, we similarly improve MHFormer following MixSTE+. Experimental evidence shows that the proposed method promotes the robustness of MHFormer while reducing its computational cost (see Fig. 10), even though it already equips itself with prior knowledge of noisy joint detection. Therefore, this result demonstrates that, in terms of improvements in the robustness of models, our method is compatible with other approaches.

We have so far generalized our approach to other two transformer-based methods, i.e., MixSTE [44] and MH-Former [16]. We may explore the generalization of the proposed method to a wider range of model architectures in the future, such as CNN-based and GNN-based methods in 3D human pose estimation. Moreover, we believe our method can also be utilized in other tasks, especially skeleton-based ones where the computational cost of long-sequence processing and the quality of human skeleton representations can become problems.

## G. Visualizations and Analysis

In this section, we provide a series of qualitative results on challenging in-the-wild images to showcase the robustness of PoseFormerV2 in real scenarios.

Fig. 11 presents several representative hard cases with HRNet [34] 2D joint detection: (a) Occlusions where joints overlap with each other; (b)(c) Missed joints; (d) Switched joints. Specifically, the right arm of the person in the 4th image of (b) and the left arm of the person in the 3rd image of (c) are missed. Moreover, in the 2nd image of (d), two legs of the person are switched (highlighted with lightyellow circles). Despite the imperfect 2D joint input, Pose-FormerV2 still infers correct positions for these joints in 3D space (marked with orange circles).

Analysis. The robustness of PoseFormerV2 is attributed to the usage of an appropriate representation – lowfrequency DCT coefficients – of the input joint sequence, instead of hand-crafted modules that may bring additional computational cost such as the multi-hypothesis generation module in [16]. Low-frequency DCT coefficients provide a global vision of the input sequence and therefore the noise contained in individual video frames is dwarfed. This utilization of DCT coefficients also brings an extra advantage to PoseFormerV2, the temporal consistency of the estimated 3D pose between adjacent frames. We provide a video demo to illustrate that the proposed method keeps an excellent consistency (i.e., temporal stability) under extremely corrupted 2D joint detection.

## H. Broader Impacts and Limitations

Broader impacts. In this paper, we attempt to reconcile two critical issues in real-scenario applications of 3D HPE, i.e., the efficiency of models to process long sequences for improved precision and their robustness against noisy 2D detection as high-quality joint sequences are hard to obtain. To encourage more real-world applications, we may shift our research focus from marginal improvements on carefully controlled datasets to overcoming the drawbacks of existing approaches in practical use. We expect more research to follow this line.

![](images/182193826ce1c603545cc4320658a9369dadadde3e0ee64364dbc6bb19686a97.jpg)

![](images/e15afc600afae65f3fd975e0cc2bf11caded96344cd15b71a8f2ae0c39276707.jpg)

![](images/16489bac2c530db24cf2389623257d73077f258dbfc9d15692ba505ba31d9853.jpg)

![](images/234b8b05a13cfddd407d4bbe467c31636a62e66c5b194b2dd814ba9723e01244.jpg)  
(a)

![](images/4697f656eeb6286f4bc624233739d6ed71d65fb7098449284c4c7b5baa245231.jpg)  
(b)

![](images/e49ec90610852bba009ef2d6fb504793637ec3a8d68191885f6bb7261bac3e71.jpg)  
(c)  
(d)  
Figure 11. Qualitative results of PoseFormerV2 under challenging in-the-wild images: (a) Occlusions; (b)(c) Missed 2D joint detection; (d) Switched 2D joints. We highlight the unreliable 2D detection with light-yellow circles and corresponding 3D pose estimations with orange circles. PoseFormerV2 shows great robustness to imperfect 2D joint detection.

On the other hand, this work is done based on a scarcely investigated frequency method, i.e., Discrete Cosine Transform (DCT) which plays an important role in conventional image compression algorithms. We hope this research will inspire more research to revisit traditional signal processing techniques as various data we treat in the deep learning era is actually signals of different forms. An appropriate combination of these choreographed techniques and recently developed deep learning approaches may bring surprising advantages.

Limitations. Our method includes two important hyperparameters – the number of sampled central frames and that of the kept DCT coefficients of the complete input sequence. Currently, they are chosen on the basis of experimental results or human experience for the trade-off between speed and accuracy. In the future, we may reshape them as learnable parameters that can be automatically learned from input data, or we may further theoretically formulate the optimal choices for them, thus removing the need for parameter-searching.

## References

[1] Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Luciˇ c, and Cordelia Schmid. Vivit: A video vi-´ sion transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 6836– 6846, October 2021. 1

[2] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? In Proceedings of the International Conference on Machine Learning (ICML), July 2021. 1

[3] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. 1

[4] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-toend object detection with transformers. In European conference on computer vision, pages 213–229. Springer, 2020. 1

[5] Tianlang Chen, Chen Fang, Xiaohui Shen, Yiheng Zhu, Zhili Chen, and Jiebo Luo. Anatomy-aware 3d human pose estimation with bone-based pose decomposition. IEEE Transactions on Circuits and Systems for Video Technology, 2021. 1, 7, 9

[6] Yilun Chen, Zhicheng Wang, Yuxiang Peng, Zhiqiang Zhang, Gang Yu, and Jian Sun. Cascaded pyramid network for multi-person pose estimation. In CVPR, 2018. 1, 2, 4

[7] Yilun Chen, Zhicheng Wang, Yuxiang Peng, Zhiqiang Zhang, Gang Yu, and Jian Sun. Cascaded pyramid network for multi-person pose estimation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7103–7112, 2018. 9

[8] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018. 1

[9] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR, 2021. 1, 4

[10] Moritz Einfalt, Katja Ludwig, and Rainer Lienhart. Uplift and upsample: Efficient 3d human pose estimation with uplifting transformers. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), January 2023. 2, 3, 6

[11] John Guibas, Morteza Mardani, Zongyi Li, Andrew Tao, Anima Anandkumar, and Bryan Catanzaro. Adaptive fourier neural operators: Efficient token mixers for transformers. arXiv preprint arXiv:2111.13587, 2021. 3

[12] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross Girshick. Masked autoencoders are scalable´ vision learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16000– 16009, 2022. 3

[13] C. Ionescu, D. Papava, V. Olaru, and C. Sminchisescu. Human3.6m: Large scale datasets and predictive methods for 3d human sensing in natural environments. IEEE TPAMI, 2014. 1, 2, 3, 4, 6, 7, 10, 11

[14] Shichao Li, Lei Ke, Kevin Pratama, Yu-Wing Tai, Chi-Keung Tang, and Kwang-Ting Cheng. Cascaded deep monocular 3d human pose estimation with evolutionary training data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020. 7

[15] Wenhao Li, Hong Liu, Runwei Ding, Mengyuan Liu, Pichao Wang, and Wenming Yang. Exploiting temporal contexts with strided transformer for 3d human pose estimation. IEEE Transactions on Multimedia, 2022. 2, 3, 6

[16] Wenhao Li, Hong Liu, Hao Tang, Pichao Wang, and Luc Van Gool. Mhformer: Multi-hypothesis transformer for 3d human pose estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13147–13156, June 2022. 2, 3, 6, 7, 8, 10, 11

[17] Jiahao Lin and Gim Hee Lee. Trajectory space factorization for deep video-based 3d human pose estimation. In BMVC, 2019. 7, 9

[18] Ruixu Liu, Ju Shen, He Wang, Chen Chen, Sen-ching Cheung, and Vijayan Asari. Attention mechanism exploits temporal contexts: Real-time 3d human pose reconstruction. In CVPR, 2020. 1, 9

[19] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In

Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10012–10022, 2021. 1

[20] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 9

[21] Wei Mao, Miaomiao Liu, and Mathieu Salzmann. History repeats itself: Human motion prediction via motion attention. In European Conference on Computer Vision, pages 474–489. Springer, 2020. 2, 3

[22] Wei Mao, Miaomiao Liu, Mathieu Salzmann, and Hongdong Li. Learning trajectory dependencies for human motion prediction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9489–9497, 2019. 2, 3

[23] Dushyant Mehta, Helge Rhodin, Dan Casas, Pascal Fua, Oleksandr Sotnychenko, Weipeng Xu, and Christian Theobalt. Monocular 3d human pose estimation in the wild using improved cnn supervision. In 2017 international conference on 3D vision (3DV), pages 506–516. IEEE, 2017. 3, 6, 7

[24] Dushyant Mehta, Srinath Sridhar, Oleksandr Sotnychenko, Helge Rhodin, Mohammad Shafiei, Hans-Peter Seidel, Weipeng Xu, Dan Casas, and Christian Theobalt. Vnect: Real-time 3d human pose estimation with a single rgb camera. ACM Transactions on Graphics (TOG), 36(4):1–14, 2017. 7, 9

[25] Gyeongsik Moon and Kyoung Mu Lee. I2l-meshnet: Imageto-lixel prediction network for accurate 3d human pose and mesh estimation from a single rgb image. In ECCV, 2020. 1

[26] Alejandro Newell, Kaiyu Yang, and Jia Deng. Stacked hourglass networks for human pose estimation. In European conference on computer vision, pages 483–499. Springer, 2016. 1

[27] Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in pytorch. 2017. 9

[28] Georgios Pavlakos, Xiaowei Zhou, and Kostas Daniilidis. Ordinal depth supervision for 3D human pose estimation. In CVPR, 2018. 1

[29] Dario Pavllo, Christoph Feichtenhofer, David Grangier, and Michael Auli. 3d human pose estimation in video with temporal convolutions and semi-supervised training. In CVPR, 2019. 7, 9

[30] William B Pennebaker and Joan L Mitchell. JPEG: Still image data compression standard. Springer Science & Business Media, 1992. 3

[31] Yongming Rao, Wenliang Zhao, Zheng Zhu, Jiwen Lu, and Jie Zhou. Global filter networks for image classification. Advances in Neural Information Processing Systems, 34:980– 993, 2021. 3

[32] Wenkang Shan, Zhenhua Liu, Xinfeng Zhang, Shanshe Wang, Siwei Ma, and Wen Gao. P-stmo: Pre-trained spatial temporal many-to-one model for 3d human pose estimation. arXiv preprint arXiv:2203.07628, 2022. 2, 3, 6, 7

[33] Athanassios Skodras, Charilaos Christopoulos, and Touradj Ebrahimi. The jpeg 2000 still image compression standard. IEEE Signal processing magazine, 18(5):36–58, 2001. 3

[34] Ke Sun, Bin Xiao, Dong Liu, and Jingdong Wang. Deep high-resolution representation learning for human pose estimation. In CVPR, 2019. 11

[35] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Herve J´ egou. Training´ data-efficient image transformers & distillation through attention. arXiv preprint arXiv:2012.12877, 2020. 1

[36] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. 2017. 1

[37] Jingbo Wang, Sijie Yan, Yuanjun Xiong, and Dahua Lin. Motion guided 3d pose estimation from videos. In European Conference on Computer Vision, pages 764–780. Springer, 2020. 1, 9

[38] Zhenyu Wang, Hao Luo, Pichao WANG, Feng Ding, Fan Wang, and Hao Li. VTC-LFC: Vision transformer compression with low-frequency components. In Thirty-Sixth Conference on Neural Information Processing Systems, 2022. 2, 3

[39] Chen Wei, Haoqi Fan, Saining Xie, Chao-Yuan Wu, Alan Yuille, and Christoph Feichtenhofer. Masked feature prediction for self-supervised visual pre-training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14668–14678, 2022. 3

[40] Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, and Han Hu. Simmim: A simple framework for masked image modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9653–9663, 2022. 3

[41] Kai Xu, Minghai Qin, Fei Sun, Yuhao Wang, Yen-Kuang Chen, and Fengbo Ren. Learning in the frequency domain. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020. 2, 3

[42] Shen Yan, Xuehan Xiong, Anurag Arnab, Zhichao Lu, Mi Zhang, Chen Sun, and Cordelia Schmid. Multiview transformers for video recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3333–3343, 2022. 1

[43] Ailing Zeng, Xiao Sun, Fuyang Huang, Minhao Liu, Qiang Xu, and Stephen Lin. Srnet: Improving generalization in 3d human pose estimation with a split-and-recombine approach. In ECCV, 2020. 1

[44] Jinlu Zhang, Zhigang Tu, Jianyu Yang, Yujin Chen, and Junsong Yuan. Mixste: Seq2seq mixed spatio-temporal encoder for 3d human pose estimation in video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13232–13242, 2022. 2, 3, 6, 7, 8, 10, 11

[45] Ce Zheng, Wenhan Wu, Taojiannan Yang, Sijie Zhu, Chen Chen, Ruixu Liu, Ju Shen, Nasser Kehtarnavaz, and Mubarak Shah. Deep learning-based human pose estimation: A survey, 2020. 9

[46] Ce Zheng, Sijie Zhu, Matias Mendieta, Taojiannan Yang, Chen Chen, and Zhengming Ding. 3d human pose estimation with spatial and temporal transformers. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 11656–11665, October 2021. 1, 2, 3, 4, 6, 7, 8, 9, 10, 11

[47] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable detr: Deformable transformers for end-to-end object detection. arXiv preprint arXiv:2010.04159, 2020. 1