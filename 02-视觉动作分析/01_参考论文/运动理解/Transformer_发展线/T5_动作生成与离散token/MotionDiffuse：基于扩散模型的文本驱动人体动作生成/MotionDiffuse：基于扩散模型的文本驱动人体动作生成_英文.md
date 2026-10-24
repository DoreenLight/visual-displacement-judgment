# MotionDifuse: Text-Driven Human Motion Generation with Difusion Model

Mingyuan Zhang\* · Zhongang Cai\* · Liang Pan · Fangzhou Hong · Xinying Guo · Lei Yang · Ziwei Liu

Received: date / Accepted: date

Abstract Human motion modeling is important for many modern graphics applications, which typically require professional skills. In order to remove the skill barriers for laymen, recent motion generation methods can directly generate human motions conditioned on natural languages. However, it remains challenging to achieve diverse and fine-grained motion generation with various text inputs. To address this problem, we propose MotionDifuse, the first difusion model-based text-driven motion generation framework, which demonstrates several desired properties over existing methods. 1) Probabilistic Mapping. Instead of a deterministic language-motion mapping, MotionDifuse generates motions through a series of denoising steps in which variations are injected. 2) Realistic Synthesis. MotionDifuse excels at modeling complicated data distribution and generating vivid motion sequences. 3) Multi-Level Manipulation. MotionDifuse responds to fine-grained instructions on body parts, and arbitrary-length motion synthesis with time-varied text prompts. Our experiments show MotionDifuse outperforms existing SoTA methods by convincing margins on text-driven motion generation and action-conditioned motion generation. A qualitative analysis further demonstrates MotionDiffuse’s controllability for comprehensive motion generation. Homepage: https://mingyuan-zhang.github. io/projects/MotionDiffuse.html

Keywords Motion Synthesis · Conditional Motion Generation · Difusion Model · Text-driven Generation

## 1 Introduction

Human motion modeling is a critical component of animating virtual characters to imitate vivid and rich human movements, which has been a vital topic for many applications, such as film-making, game development, and virtual YouTuber animation. To mimic human motions, virtual characters should be capable of moving around naturally, reacting to environmental stimuli, and meanwhile expressing sophisticated emotions. Despite decades of exciting technological breakthroughs, it requires sophisticated equipment (e.g., expensive motion capture systems) and domain experts to produce lively and authentic body movements. In order to remove skill prerequisites for layman users and potentially scale to the mass audience, it is vital to create a versatile human motion generation model that could produce diverse, easily manipulable motion sequences.

![](images/02d9db4a2fb689b356e16d9df43d0d967d362dcc2f3892e87bcc5ff8b03df932.jpg)  
Fig. 1 MotionDifuse is a difusion model-based text-driven motion generation method that features 1) Probabilistic Mapping 2) Realistic Synthesis that results in highly diverse motions with high-fidelity shown in a)-d), and 3) Multi-Level Manipulation that empowers comprehensive motion generation such as that in c) where multiple body parts are involved and d) where time-varied prompts are given.

Various condition signals, including pre-defined motion categories (Guo et al., 2020; Petrovich et al., 2021; Cervantes et al., 2022), music pieces (Huang et al., 2020; Li et al., 2020, 2021; Zhuang et al., 2020; Siyao et al., 2022), and natural language (Lin et al., 2018; Ahuja and Morency, 2019; Ghosh et al., 2021; Petrovich et al., 2022), have been leveraged in previous human motion generation methods. Among them, natural language is arguably the most user-friendly and convenient input format for motion sequence synthesis, and hence we focus on text-driven motion generation in this work. Recently, TEMOS (Petrovich et al., 2022) utilizes KIT Motion-Language MoCap dataset (Plappert et al., 2016) to demonstrate fine-grained trajectory synthesis. However, it does not support stylizing the generated motions and, therefore, could not achieve high diversity. MotionCLIP (Tevet et al., 2022) could generate stylized motions, but it is still limited to short text inputs and fails to handle complicated motion descriptions. In addition, they (Petrovich et al., 2022; Tevet et al., 2022) typically only accept a single text prompt, which greatly limits users’ creativity.

To tackle the aforementioned challenges, we propose MotionDifuse, a versatile and controllable motion generation framework that could generate diverse motions with comprehensive texts. Inspired by the recent progress of the text-conditioned image generation (Dhariwal and Nichol, 2021; Nichol and Dhariwal, 2021; Nichol et al., 2021; Ramesh et al., 2022), we propose to incorporate Denoising Difusion Probabilistic Models (DDPM) (Ho et al., 2020) into motion generation. Unlike classical DDPM which is only capable of fixed-size generation, we propose a Cross-Modality Linear Transformer to achieve motion synthesis with an arbitrary length depending on the motion duration. Instead of learning a direct mapping between the text space and the motion space (Tevet et al., 2022), we propose to guide the generation pipeline with input texts softly, which could significantly increase the diversity of the generation results. To maintain the uncertainties in the denoising process, we process the noise terms conditioned on the input texts by several transformer decoder layers for each denoising step. In this way, the text conditions would not dominate the motion generation in a deterministic way, which facilitates generating diverse motion sequences from the driving texts.

Furthermore, MotionDifuse can achieve body partindependent control with fine-grained texts. Specifically, to accommodate the human body structures, MotionDifuse divides the whole-body motion into several near-independent parts(e.g. upper body and lower body). Based on the fine-grained body parts definition, we propose ‘noise interpolation’ to separately control diferent body parts while taking their correlations into consideration. Moreover, to synthesize arbitrary-length motion sequences, we propose a new sampling method to denoise several overlapped sequences simultaneously. Specifically, MotionDifuse first gets results from each sequence independently and then mixes them with correction terms. Diferent from auto-regressive inference schemes that often require many long motion sequences for training, MotionDifuse is capable of modeling the correlations between continuous actions without introducing additional training costs.

We perform both extensive qualitative experiments on popular benchmarks, and quantitative evaluation on comprehensive motion generation. Firstly, we demonstrate significant improvements on text-driven motion generation over the current art on HumanML3D and KIT-ML. Secondly, to illustrate the superiority of MotionDifuse, we directly apply it to action-conditioned motion generation tasks, and it outperforms all existing works on the HumanAct12 dataset and the UESTC datasets. Furthermore, we also demonstrate more possibilities of MotionDifuse by conditioning the model on mixed control signals that allow body-part level manipulation and long motion generation.

In summary, our proposed MotionDifuse has several desired properties over prior arts:

– Probabilistic Mapping. Benefiting from our new formulation of motion generation, where the DDPM is incorporated in, MotionDifuse can be conditioned on text descriptions to generate motions in a probabilistic style, naturally leading to high diversity.

Realistic Synthesis. The careful design of the architecture allows MotionDifuse to synthesize highfidelity motion sequences and achieves state-of-theart on two conditional motion generation tasks. – Multi-Level Manipulation. With the extended design, MotionDifuse handles fine-grained text descriptions that mobilize the entire body (e.g. ‘a person is drinking water while walking’) and timevaried signals (e.g. ‘a person is walking and then running’).

## 2 Related Work

## 2.1 Motion Generative Model

Motion Generation has been studied for decades. Some early works focus on unconditional motion generation (Rose et al., 1998; Ikemoto et al., 2009; Mukai and Kuriyama, 2005). While some other works try to predict future motions given an initial pose or a starter motion sequence (Futrelle and Speckert, 1978; Gavrila, 1999; O’rourke and Badler, 1980). Statistical models such as PCA (Ormoneit et al., 2005), Motion Graph (Min and Chai, 2012) are applied for these purposes.

With the rapid development of Deep Learning (DL) techniques, more generative architectures occur and flourish. Previous works can be broadly divided into four groups: 1) Variational Auto Encoder (VAE); 2) Generative Adversarial Networks (GAN); 3) Normalization Flow Network; 4) Implicit Neural Representations.

VAE (Kingma and Welling, 2013) is one of the most commonly used generative models in motion synthesis. Yan et al. (2018) and Aliakbarian et al. (2020) regard the motion generation task as predicting a small future motion sequence with the given small current motion sequence. They use VAE to encode the pair of current sequence and future sequence and then reconstruct the future one. ACTOR (Petrovich et al., 2021) proposes a transformer-based encoder and decoder architecture. Transformer Encoder Layers and Transformer Decoder Layers (Vaswani et al., 2017) are the basic blocks to build up a motion encoder and a motion decoder. This architecture is also employed in later works (Tevet et al., 2022; Hong et al., 2022; Petrovich et al., 2022).

GAN (Goodfellow et al., 2014) introduces an auxiliary module, discriminator network, to justify the qual ity and validity of generated samples. Some works focus on proposing appropriate discriminator networks for motion generation to improve the synthesis quality (Barsoum et al., 2018; Harvey et al., 2020; Wang et al., 2020). HP-GAN (Barsoum et al., 2018) attempts to supervise the motion prediction results without the specific ground truth. Therefore, a data-driven discriminator is involved in learning a motion prior, which is used to justify the prediction quality. Harvey et al. (2020) target solving the blurriness of the predicted motion in the Motion In-between task and propose two discriminators for both short-term critic and long-term critic. Wang et al. (2020) build up a cyclic pipeline. With the help of a discriminator, the proposed pipeline can generate both class-specific and mixed-class motion sequences.

Normalization Flow Network (Dinh et al., 2014) has a long history and has been studied extensively for image synthesis (Dinh et al., 2016; Kingma and Dhariwal, 2018). This kind of architecture builds up a reversible neural network and will map the input data into a multi-dimensional Gaussian distribution. Hence, we can generate an initially random vector from this distribution and feed them into the reversed network to generate motion samples. Inspired by the success of GLOW (Kingma and Dhariwal, 2018), MoGlow (Henter et al., 2020) proposes an auto-regressive normalization network to model motion sequences. History features from an LSTM model (Hochreiter and Schmidhuber, 1997) serve as the condition of the flow network, which predicts the next pose.

Recently, another generative model has attracted much attention with the considerable success achieved by NeRF (Mildenhall et al., 2020; Jain et al., 2021) in rendering realistic images. Implicit Neural Representations (INR) are a series of neural networks that optimize their parameters to fit one sample instead of the whole distribution. One principal advantage is that this technique has superb generalization ability on spatial or temporal dimensions. For example, Cervantes et al. (2022) propose an implicit scheme, which simultaneously models action category and timestamp. Similar to the original NeRF, the timestamp is represented by sinusoidal values. After supervised training, the proposed method can generate a variable-length motion sequence for each action category.

This paper proposes a new motion generation pipeline based on the Denoising Difusion Probabilistic Model (DDPM) (Ho et al., 2020). One of the principal advantages of DDPM is that the formation of the original motion sequence can be retained. It means that we can easily apply more constraints during the denoising process. In the later sections, we will explore more potential of DDPM in diferent types of conditions. Besides, benefiting from this nature, DDPM can generate more diverse samples.

## 2.2 Conditional Motion Generation

The increasing maturity of various generative models stimulates researchers’ enthusiasm to study conditional motion generation. For example, some works (Guo et al., 2020; Petrovich et al., 2021; Cervantes et al., 2022) aim at synthesizing motion sequences of several specific categories. Action2Motion (Guo et al., 2020) builds up a recurrent conditional VAE for motion generation. Given history memory, this model predicts the next pose under the constraints of the action category. ACTOR (Petrovich et al., 2021) also uses VAE for random sampling. Unlike Action2Motion, ACTOR embeds the whole motion sequence into the latent space. This design avoids the accumulative error in the recurrent scheme. Besides, ACTOR proposes a Transformerbased motion encoder and decoder architecture. This structure significantly outperforms recurrent methods. Cervantes et al. (2022) attempt to model motion sequence with implicit functions, which can generate motion sequences with varied lengths.

Another significant conditional motion generation task is music to dance. This task requires that the generated motion has beat-wise connectivity, is a specific kind of dance, or can express similar content with the music. Many works attempt to embed the music feature and motion feature into a joint space (Lee et al., 2019; Sun et al., 2020; Li et al., 2020, 2021). Unlike direct feature embedding, Bailando (Siyao et al., 2022) proposes a two-stage dance generator. It first learns a quantized codebook of meaningful dance pieces and then attempts to generate the whole sequence with a series of elements from a codebook.

Similar to music-to-dance, text-driven motion generation can be regarded as learning a joint embedding of text feature space and motion feature space. There are two major diferences. The first one is that language commands correlate more with the human body. Therefore, we expect to control each body part accurately. The second diference is that text-driven motion generation contains a vast range of motions. Some descriptions are direct commands to a specific body part, such as “touch head”. Some describes arbitrary concepts like “playing the violin”. Such huge complexity of motions brings many dificulties to the architecture design. Recently, many works have proposed text-driven motion generation pipelines. Most of them are deterministic generation (Ahuja and Morency, 2019; Ghosh et al., 2021; Tevet et al., 2022), which means they can only generate a single result from the given text. TEMOS (Petrovich et al., 2022) introduces the VAE architec ture into this task. It can generate diferent motion sequences given one text description. However, these methods attempt to acquire a joint embedding space of motion and natural language. This design significantly compresses the information from text. Therefore, these works can hardly generate correct motion sequences from a detailed description. Guo et al. (2022) proposes an auto-regressive pipeline. It first encodes language de scriptions into features and then auto-regressively generates motion frames conditioned on the text features. However, this method is hard to capture the globa relation due to the auto-regressive scheme. Moreover, the generation quality is inferior. Instead, our proposed MotionDifuse softly fuses text features into generation and can yield the whole sequence simultaneously. The experiment results prove the superiority of our design.

## 2.3 Motion Datasets

Human motion modeling has been a long-standing problem in computer vision and computer graphics. With the advent of deep learning, data has become increasingly important for training neural networks that perceive, understand, and generate human motions.

A common form of datasets containing videos of human subjects are recorded with annotations such as 2D keypoints (Jhuang et al., 2013; Andriluka et al., 2018), 3D keypoints (Ionescu et al., 2013; Joo et al., 2015; Mehta et al., 2017; Trumble et al., 2017; Li et al., 2021) and statistical model parameters (Yu et al., 2020; Patel et al., 2021; Cao et al., 2020; Cai et al., 2021, 2022). Action labels are also a popular attribute of datasets for human action understanding that contains humancentric actions (Kuehne et al., 2011; Soomro et al., 2012; Karpathy et al., 2014; Gu et al., 2018; Shao et al., 2020; Chung et al., 2021), interaction (Carreira et al., 2019; Monfort et al., 2019; Zhao et al., 2019), finegrained action understanding (Gu et al., 2018; Shao et al., 2020; Chung et al., 2021) and 3D data (Liu et al., 2019). For the action-conditioned motion generation task, HumanAct12 (Guo et al., 2020), UESTC (Ji et al., 2018), and NTU RGB+D (Liu et al., 2019) are three commonly used benchmarks. However, the abovementioned datasets do not provide paired sophisticated semantic labels to the motion sequences.

KIT (Plappert et al., 2016) contains motion capture data annotated with detailed descriptions. Zhang et al. (2020) recruit actors and actresses to record body movements expressing emotions. Recently, BABEL (Punnakkal et al., 2021) and HumanML3D (Guo et al., 2022) re-annotates AMASS (Mahmood et al., 2019), a large scale motion capture dataset, with English language labels.

In this paper, we use the HumanML3D dataset and KIT dataset to evaluate the proposed methods for the text-driven motion generation task. HumanAct12 and UESTC are used to demonstrate the wide applicability of the proposed pipeline. Furthermore, we use the BABEL dataset for additional applications.

## 3 Methodology

We present a difusion model-based framework, MotionDifuse, for high-fidelity and controllable textdriven motion generation. We first give the problem definition, settings of the original text-driven motion generation in Section 3.1. After that, we provide an overall illustration of the proposed MotionDifuse in Section 3.2, followed by introducing the difusion model in Section 3.3 and the transformer-based architecture in Section 3.4. Finally, the inference strategy is illustrated for the fine-grained generation scenarios in Section 3.5.

## 3.1 Preliminaries

The motion sequence Θ is an array of $( \theta _ { i } ) , \ i \in$ $\{ 1 , 2 , \ldots , F \}$ , where $\theta _ { i } \in \mathbb { R } ^ { D }$ represents the pose state in the i-th frame, and $F$ is the number of frames. The representation of each pose state $\theta _ { i }$ is distinct in diferent datasets. It generally contains joint rotation, joint position, joint velocity, and foot contact conditions. Our proposed MotionDifuse is robust to the various motion representations. Therefore, we do not specify the components of $\theta _ { i }$ in this section, and leave the details in Section 4.

For standard Text-driven Motion Generation, the training datasets consist of $( \theta _ { i } , \mathrm { t e x t } _ { i } )$ pairs, where text is the language description of motion sequence $\theta _ { i } .$ During inference, given a set of descriptions $\{ \mathrm { t e x t } _ { i } \}$ , we are requested to generate motion sequences conditioned on the given descriptions. This task can also be regarded as a text-to-motion translation (T2M) task. We will use this abbreviation below.

An related task is Action-conditioned Motion Generation. Given a pre-defined action category set, models are supposed to fit the data distribution and synthesize motion sequences of each category. Annotated data in this task can be represented as $( y _ { i } , \theta _ { i } )$ , where $y _ { i }$ is the category index of i-th data, $\theta _ { i }$ is the motion sequence of i-th data. In this paper, we replace $y _ { i }$ by its semantic description $\mathrm { t e x t } _ { i }$ . Then we can use the same pipeline as in the T2M task.

## 3.2 Pipeline Overview

Following the literature on the difusion model in the image synthesis field (Ho et al., 2020), we first build up a text-conditioned motion generation pipeline using a denoising difusion probabilistic model (DDPM). This model is the basis of our proposed MotionDifuse. For the denoising process, we propose a Cross-Modality Linear Transformer to process input sequences conditioned on the given text prompts. Beyond the direct application of text-driven motion generation, we take one step further to explore methods that are conditioned on motion representation during denoising. Specifically, we experiment with two types of additional signals: part-aware text controlling and time-varied controlling. The former assigns diferent text conditions to diferent body parts so that we can accurately control each part of the body and generate more complicated motion sequences. The latter divides the whole sequence into several parts and assigns independent text conditions for each interval. Therefore, we can synthesize arbitrary-length motion sequences that incorporate several actions. These two kinds of conditions significantly expand the capability of MotionDifuse. The overall pipeline is shown in Figure 2. We introduce each part of this architecture in the following subsections.

## 3.3 Difusion Model for Motion Generation

Generative Adversarial Networks (GANs) involve a discriminator to improve the generation quality in an adversarial manner. GANs are typically challenging to train, especially for conditional motion generation tasks. Implicit Functions use Multi-Layer Perceptron (MLP) to fit motion sequences. This neat architecture is easily trained on a small number of data but tends to be less generalizable when it is subjected to complicated conditions. Auto-Encoder (AE) and Variational Auto-Encoder (VAE) are the most widely used approaches in text-driven motion generation (Ghosh et al., 2021; Petrovich et al., 2022). Previous works learn a joint embedding of motion sequences and languages that explicitly apply the text condition in the deterministic language-motion mapping. However, high-level text features typically contain insuficient fine-grained details to guide the generation of subtly diferent motion. Hence, directly linking text embedding to motion embedding results in the limited diversity of the generated motions.

![](images/84c7a9ad6c9b7502ddd7c0205998754542a83b81acb1e4887b3f9bcd688cb2a3.jpg)  
Fig. 2 Overall Pipeline of the proposed MotionDifuse. The colors of the arrows indicate diferent stages: blue for training, red for inference, and black for both training and inference.

To tackle the problem, we build our text-driven motion generation pipeline based on difusion models. Diffusion Models (Ho et al., 2020; Dhariwal and Nichol, 2021; Nichol and Dhariwal, 2021; Nichol et al., 2021) are a new class of generative models. A probabilistic model is learned to gradually denoises a Gaussian noise to generate a target output, such as a 2D image or 3D point cloud. Formally, difusion models are formulated as $\begin{array} { r } { p _ { \theta } ( \mathbf { x } _ { 0 } ) : = \int p _ { \theta } ( \mathbf { x } _ { 0 : T } ) d \mathbf { x } _ { 1 : T } } \end{array}$ , where $\mathbf { x } _ { 0 } \sim q ( \mathbf { x } _ { 0 } )$ is the real data, and $\mathbf { x } _ { 1 } , \cdots , \mathbf { x } _ { T }$ are the latent data. They generally have a difusion process and a reverse process. To approximate posterior $q \big ( \mathbf { x } _ { 1 : T } | \mathbf { x } _ { 0 } \big )$ , the diffusion process follows a Markov chain to gradually add Gaussian noise to the data until its distribution is close to the latent distribution $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , according to variance schedules given by $\beta _ { t }$ :

$$
\begin{array}{l} q (\mathbf {x} _ {1: T} | \mathbf {x} _ {0}) := \prod_ {t = 1} ^ {T} q (\mathbf {x} _ {t} | \mathbf {x} _ {t - 1}), \\ q (\mathbf {x} _ {t} | \mathbf {x} _ {t - 1}) := \mathcal {N} (\mathbf {x} _ {t}; \sqrt {1 - \beta_ {t}} \mathbf {x} _ {t - 1}, \beta_ {t} \mathbf {I}). \end{array}\tag{1}
$$

The reverse process $p _ { \theta } ( \mathbf { x } _ { 0 : T } )$ is also a Markov chain that predicts and eliminates the noise with learned Gaussian transitions starting at $p ( \mathbf { x } _ { T } ) = \mathcal { N } ( \mathbf { x } _ { T } ; \mathbf { 0 } , \mathbf { I } )$

$$
\begin{array}{r l} & p _ {\theta} (\mathbf {x} _ {0: T}) := p (\mathbf {x} _ {T}) \prod_ {t = 1} ^ {T} p _ {\theta} (\mathbf {x} _ {t - 1} | \mathbf {x} _ {t}), \\ & p _ {\theta} (\mathbf {x} _ {t - 1} | \mathbf {x} _ {t}) := \mathcal {N} (\mathbf {x} _ {t - 1}; \mu_ {\theta} (\mathbf {x} _ {t}, t), \Sigma_ {\theta} (\mathbf {x} _ {t}, t)). \end{array}\tag{2}
$$

To accomplish the reverse process of the difusion model, we need to construct and optimize a neural network. During training, first we uniformly sample steps t for each ground truth motion $\mathbf { x } _ { \mathrm { 0 } }$ and then generate a sample from $q ( \mathbf { x } _ { t } | \mathbf { x } _ { 0 } )$ . Instead of repeatedly adding noises on $\mathbf { x } _ { \mathrm { 0 } }$ , Ho et al. (2020) formulate the difusion process as

$$
q (\mathbf {x} _ {t} | \mathbf {x} _ {0}) = \sqrt {\bar {\alpha_ {t}}} \mathbf {x} _ {0} + \epsilon \sqrt {1 - \bar {\alpha_ {t}}}, \epsilon \sim \mathcal {N} (\mathbf {0}, \mathbf {I}),\tag{3}
$$

![](images/47195fd95167963d94767aa4d7b43b37d11e3b96d96caca3caade404ce60ba73.jpg)  
Fig. 3 Architecture of our Linear Self-attention.

where $\begin{array} { r } { \alpha _ { t } = 1 - \beta _ { t } , \bar { \alpha _ { t } } = \prod _ { s = 0 } ^ { t } \alpha _ { s } } \end{array}$ . Hence, we can simply sample a noise  and then directly generate $\mathbf { x } _ { t }$ by this formulation. Instead of predicting $\mathbf { x } _ { t - 1 }$ , here we follow GLIDE (Nichol et al., 2021) and predict the noise term . It means that we need to construct a network to fit $\epsilon _ { \theta } ( \mathbf { x } _ { t } , t , \mathrm { t e x t } )$ . We optimize the model parameters to decrease a mean squared error as

$$
\mathcal {L} = \mathrm{E} _ {t \in [ 1, T ], \mathbf {x} _ {0} \sim q (\mathbf {x} _ {0}), \epsilon \sim \mathcal {N} (\mathbf {0}, \mathbf {I})} [ \| \epsilon - \epsilon_ {\theta} (\mathbf {x} _ {t}, t, \text { text }) \| ].\tag{4}
$$

This is the only loss we used in model training. To generate samples from the given text description, we denoise the sequence from $p ( \mathbf { x } _ { T } ) = \mathcal { N } ( \mathbf { x } _ { T } ; \mathbf { 0 } , \mathbf { I } )$ . Equation 2 shows that we need to estimate $\mu _ { \theta } ( \mathbf { x } _ { t } , t ,$ text) and $\textstyle \sum _ { \theta } ( \mathbf { x } _ { t } , t , \mathrm { t e x t } )$ . To simplify the problem, we set $\varSigma _ { \theta } ( \mathbf { x } _ { t } , t , \mathrm { t e x t } )$ as a constant number $\beta _ { t } . \ \mu _ { \theta } ( \mathbf { x } _ { t } , t , \mathrm { t e x t } $ ) can be estimated as

$$
\mu_ {\theta} (\mathbf {x} _ {t}, t, \text {text}) = \frac {1}{\sqrt {\mathbf {x} _ {t}}} (\mathbf {x} _ {t} - \frac {1 - \alpha_ {t}}{\sqrt {1 - \bar {\alpha} _ {t}}} \epsilon_ {\theta} (\mathbf {x} _ {t}, t, \text {text})).\tag{5}
$$

Therefore we can denoise the motion sequence step by step and finally get a clean motion sequence, which is conditioned on the given text.

## 3.4 Cross-Modality Linear Transformer

In Section 3.3, we illustrate difusion models as motion generators and a neural network $\epsilon _ { \theta } ( \mathbf { x } _ { t } , t , \mathrm { t e x t } )$ is essential for denoising steps. In this section, we will introduce the design of $\epsilon _ { \theta } ( \mathbf { x } _ { t } , t , \mathrm { t e x t } )$ in our proposed MotionDiffuse.

Previous works (Ho et al., 2020; Dhariwal and Nichol, 2021; Nichol and Dhariwal, 2021; Nichol et al., 2021) mainly utilize UNet-like structure as the denoising model. However, the target motion sequences are variable-length in the motion generation task, making convolution-based networks unsuitable. Therefore, we propose a Cross-Modality Linear Transformer, as shown in Figure 2. Similar to the machine translation task, our proposed model includes a text encoder and a motion decoder. To meet the requirement of the diffusion models, we further customize each layer of the motion decoder.

Text Encoder Here we directly use classical transformer Vaswani et al. (2017) to extract text features. Specifically, the input data first passes through an embedding layer to get the embedding feature from raw text and then is further processed by a series of transformer blocks. Each block contains two components: a multi-head attention module (MHA) and a feedforward network (FFN). Suppose the input feature is $\mathbf { X } \in \mathbb { R } ^ { n \times d }$ (n denotes the number of elements and d denotes the element feature dimension), MHA extracts query feature vectors $\mathbf { Q } \in \mathbb { R } ^ { n \times d }$ , key feature vectors K $\in \mathbb { R } ^ { n \times d }$ , and value feature vectors $\mathbf { V } \in \mathbb { R } ^ { n \times d }$ as:

$$
\mathbf {Q} = W _ {q} \mathbf {X}, \quad \mathbf {K} = W _ {k} \mathbf {X}, \quad \mathbf {V} = W _ {v} \mathbf {X},\tag{6}
$$

where $W _ { q } , \ W _ { k }$ and $W _ { v }$ are the corresponding linear projections to generate Q, K and V, respectively. The value features are then aggregated with attention weights $\mathbf { A } \in \mathbb { R } ^ { n \times n }$ :

$$
\mathbf {A} = \mathrm{softmax} (\frac {\mathbf {Q} \otimes \mathbf {K} ^ {\top}}{\sqrt {d}}), \quad \mathbf {Y} = \mathbf {A} \otimes \mathbf {V},\tag{7}
$$

where $\mathbf { Y } \in \mathbb { R } ^ { n \times d }$ is the output of MHA modules, d is the dimension of each element in X, and ⊗ denotes the matrix multiplication. The multi-head mechanism divides the input vector into several parts, which pass through the process in Equation 6 and 7 independently. The outputs are concatenated so that the dimension remains unchanged. A residual connection is applied between input and output of the MHA modules. This feature is further processed by FFN, which contains three linear transformations and two GELU (Hendrycks and Gimpel, 2016) layers between them.

To enhance the generalization ability, we use parameter weights in CLIP (Radford et al., 2021) to initialize the first several layers. This part of the parameters is frozen and will not be optimized in the later training.

Linear Self-attention. This module aims at enhancing motion features by modeling correlations between diferent frames. The principal advantage of selfattention is to get an overview of the input sequence and is thus beneficial to estimating the injected noise . However, the time complexity of calculating attention weight $\mathbf { A } \in \mathbb { R } ^ { n \times n } { \mathrm { ~ i s ~ } } { \mathcal { O } } ( n ^ { 2 } d )$ . When the target motion sequence length increases, the time cost increases quadratically. In the T2M task, the length can be several hundred, which leads to low speed. Hence, we adopt Eficient Attention (Shen et al., 2021) to speed up the self-attention module. Instead of calculating pair-wise attention weights, eficient attention generates global feature map $\mathbf { F } _ { g } \in \mathbb { R } ^ { d _ { k } \times d _ { k } }$ , where $d _ { k }$ is the dimension of feature after multi-head split:

$$
\mathbf {F} _ {g} = \operatorname{softmax} (\mathbf {K} ^ {\top}) \otimes \mathbf {V}, \quad \mathbf {Y} = \operatorname{softmax} (\mathbf {Q}) \otimes \mathbf {F} _ {g}.\tag{8}
$$

The time complexity of these two steps is $\mathcal { O } ( d _ { k } ^ { 2 } n k ) =$ $\mathcal { O } ( d d _ { k } n )$ , where n is the number of the elements in the sequences, k is the number of heads of self-attention.

Another advantage of eficient attention for the diffusion model is that feature map $\mathbf { F } _ { g }$ explicitly aggregates global information while classical self-attention focus more on pair-wise relation. Global information gives more clues about the semantic meaning of the motion sequence than pair-wise one. The experiment results also prove this conclusion.

Linear Cross-attention. Cross-attention replaces X in K and V calculation by the text feature. Other formulations are the same as Linear Self-attention. Text features are injected into motion sequences in this process to generate motion conditioned on the given text.

Stylization Block. In each denoising step, the output is conditioned on the given text and the timestamp t. Linear Cross-attention fuses the text features into motion sequences. We need another Stylization Block component to bring timestamp t to the generation process. This block is applied after each Linear Self-attention block, Linear Cross-attention block, and FFN block.

Similar to GLIDE (Nichol et al., 2021), we first get a text embedding $e _ { \mathrm { t e x t } }$ by a linear transformation on the text features and a timestamp embedding $\mathbf { e _ { t } }$ by positional embedding (Vaswani et al., 2017). These two terms are summed together into one single vector e. Given the original output Y from other blocks, the Styl ization block will process the feature as:

$$
\mathbf {B} = \psi_ {b} (\phi (\mathbf {e})), \quad \mathbf {W} = \psi_ {w} (\phi (\mathbf {e})), \quad \mathbf {Y} ^ {\prime} = \mathbf {Y} \cdot \mathbf {W} + \mathbf {B},\tag{9}
$$

where (·) denotes Hadamard product, $\mathbf { Y } ^ { \prime }$ is the output of stylization blocks. $\psi _ { b } , \psi _ { w } , \phi$ denote three diferent linear projections. In classical transformers, the output from each block is added to the original input as a residual connection, as shown in Figure 2. In Motion-Difuse, these outputs pass through stylization blocks and are added to the information. This modification enables our proposed method to know the timestamp t.

## 3.5 Fine-grained Controlling

To enrich the capability of MotionDifuse, we explore the properties of both the motion representation and the denoising process of DDPM. Unlike VAE, the generated motion sequence is in its explicit form instead of being compressed in the latent space. This characteristic of DDPM-based motion generation allows more operations to be applied to this motion sequence to increase the manipulability.

Body Part-independent Controlling. Due to the lack of diversity in text descriptions, we cannot achieve accurate motion control for each body part from text descriptions only. For example, the prompt ‘a person is running and waving left hand’ is highly challenging to the model because the expected motion sequence is significantly far from the training distribution. Even if we manually split the original description into two independent ones: ‘a person is running’ for lower limbs, and $ { \mathbf { \hat { \rho } } } _ { \mathrm { a } }$ person is waving left hand’ for upper limbs, it is still dificult for the model to generate correct motions. An intuitive solution for this situation is to separately generate two motion sequences and combine the upperlimb motion of the first sequence and the lower-limb motion of the second sequence. This simple solution mitigates the problem to some extent. However, it ignores the correlation between these two parts. Specifi cally for ‘running and waving left hand’, the frequencies of the two motions should match. Otherwise, the motion generated by this naive method appears unnatural. To better solve this problem, we propose a body partindependent controlling scheme.

Recall that, during the denoising process, our diffusion model predicts the noise term $\epsilon _ { \theta } ( \mathbf { x } _ { t } , t , \mathrm { t e x t } ) \ \in$ $\mathbb { R } ^ { F \times D }$ , where $F$ represents the number of frames, D denotes the dimension of each pose state, which includes translation and rotations of body joints. This noise term determines the denoising direction of the whole body.

Inspired by the application of the latent code interpolation, here we propose ‘noise interpolation’ to separately control the diferent parts of the human body. Suppose we have n text descriptions $\left\{ \mathrm { t e x t } _ { i } \right\}$ for diferent body parts $\{ s _ { i } \}$ . We want to calculate the noise term $\epsilon = \{ \epsilon _ { i } ^ { \mathrm { j o i n t } } \} , i \in [ 1 , m ]$ , where $\epsilon _ { i } ^ { \mathrm { j o i n t } }$ represents the noise term for the i-th body part, m denotes the number of partition. We first estimate the noise $\epsilon _ { i } ^ { \mathrm { p a r t } } =$ $\epsilon _ { \theta } \big ( \mathbf { x } _ { t } , t , \mathrm { t e x t } _ { i } \big ) , \epsilon _ { i } ^ { \mathrm { p a r t } } \ \in \ \mathbb { R } ^ { F \times D }$ . An intuitive method for combining these terms is $\begin{array} { r } { \epsilon ^ { \mathrm { p a r t } } = \sum _ { i = 1 } ^ { m } \epsilon _ { i } ^ { \mathrm { p a r t } } \cdot M _ { i } . } \end{array}$ , where $M _ { i } \in \{ 0 , 1 \} ^ { D }$ is a binary vector to show which body part should we focus. (·) denotes the Hadamard product, and here we ignore the broadcast in computation for simplicity. Although this method succeeds to some extent, the direct ignoring of some parts in $\epsilon _ { i } ^ { \mathrm { p a r t } }$ will increase the combination dificulty and lead to low-quality generation results. Therefore, we add a correction item for smoothing interpolation:

$$
\bar {\epsilon} ^ {\text { part }} = \sum_ {i = 1} ^ {m} \epsilon_ {i} ^ {\text { part }} \cdot M _ {i} + \lambda_ {1} \cdot \nabla (\sum_ {1 \leq i, j \leq m} \| \epsilon_ {i} ^ {\text { part }} - \epsilon_ {j} ^ {\text { part }} \|),\tag{10}
$$

where ∇ denotes gradient calculation, $\lambda _ { 1 }$ is a hyperparameter to balance these two items. This correction item enforces a smoother denoising direction so that the motion of diferent body parts will be more natural.

Time-varied Controlling Long-term motion generation plays a vital role in real-world applications. Previous works mainly focus on motion generation with a single type of motion. Auto-regressive methods (Henter et al., 2020; Guo et al., 2022) have solved this problem with satisfactory performance. However, none of them are capable of synthesizing diferent actions in a continuous manner. Benefiting from the nature of DDPM, here we propose another sampling method to meet this requirement.

Recall that we are given an array $\{ \mathrm { t e x t } _ { i , j } , [ l _ { i , j } , r _ { i , j } ] \}$ $i \in [ 1 , m ]$ , where m is the number of intervals. Similar to the method we proposed in the previous paragraph, we first estimate the noise term $\epsilon _ { i } ^ { \mathrm { t i m e } }$ for i-th interval independently. Suppose the overall length of the target sequence is $F ^ { \prime } .$ By padding zeros, we extend each noise term into the same dimension $F ^ { \prime } \times D$ . Then we interpolate these noises with a correcting term:

$$
\bar {\epsilon} ^ {\text {time}} = \sum_ {i = 1} ^ {m} \bar {\epsilon} _ {i} ^ {\text {time}} + \lambda_ {2} \cdot \nabla (\sum_ {1 \leq i, j \leq m} \| \bar {\epsilon} _ {i} ^ {\text {time}} - \bar {\epsilon} _ {j} ^ {\text {time}} \|),\tag{11}
$$

where $\overline { { \epsilon } } _ { j } ^ { \mathrm { t i m e } }$ is the padded term from $\epsilon _ { i } ^ { \mathrm { t i m e } } , \lambda _ { 2 }$ is a hyperparameter.

## 4 Experiments

We evaluate MotionDifuse with three categories of experiments: text-driven motion generation (Section 4.1), action-conditioned motion generation (Section 4.2), and motion manipulation (Section 4.3). In all the evaluated benchmarks, MotionDifuse could significantly outperform previous SoTA methods.

## 4.1 Text-driven Motion Generation

Datasets. KIT Motion Language datset (Plappert et al., 2016) provides 3911 motion sequences and 6353 sequence-level natural language descriptions. HumanML3D (Guo et al., 2022) re-annotates the AMASS dataset (Mahmood et al., 2019) and the HumanAct12 dataset (Guo et al., 2020). It provides 44970 annotations on 14616 motion sequences. KIT and HumanML3D are two important benchmarks for textdriven motion generation tasks. Following Guo et al. (2022), we utilize the pretrained text-motion contrastive model.

Evaluation Metrics. We evaluate all methods with five diferent metrics. 1) Frechet Inception Distance (FID). Features are extracted from both the generated results and ground truth motion sequences by the pretrained motion encoder. FID is calculated between these two distributions to measure the similarity. 2) R Precision. For each pair of generated sequence and description text, 31 other prompts are randomly selected from the test set. The pretrained contrastive model cal culates the average top-k accuracy. This section reports the top-1, top-2, and top-3 accuracies. 3) Diversity. The generated sequences from all test texts are randomly split into pairs. Then the average joint diferences are calculated in each pair, which serves as the diversity metric. 4) Multimodality. As for a single text description, we randomly generate 32 motion sequences. Multimodality measures the diferences in joint positions between these homogeneous motion sequences. 5) Multimodal Distance. Assisted by the pretrained contrastive model, we can calculate the diference between the text feature from the given description and the motion feature from the generated results, called multimodal distance.

In this section, R Precision and FID are the principal metrics from which we make the most conclusions. Besides, for a fair comparison, we run each evaluation 20 times and report the statistic interval with 95% confidence.

Implementation Details. For both HumanML3D and KIT-ML datasets, we build up an 8-layer transformer as the motion decoder. As for the text encoder, we first directly use the text encoder in the CLIP ViT-B/32 (Radford et al., 2021), and then add four more transformer encoder layers. The latent dimension of the text encoder and the motion decoder are 256 and 512, respectively. As for the difusion model, the number of difusion steps is 1000, and the variances $\beta _ { t }$ are linearly from 0.0001 to 0.02. We opt for Adam as the optimizer to train the model with a 0.0002 learning rate. We use 8 Tesla V100 for the training, and there are 128 samples on each GPU, so the total batch size is 1024. The total number of iterations is 40K for KIT-ML and 100K for HumanML3D.

Following Guo et al. (2022), pose states in this series of experiments mainly contain seven diferent parts: $( r ^ { v a } , r ^ { v x } , r ^ { v z } , r ^ { h } , \mathbf { j } ^ { p } , \mathbf { j } ^ { v } , \mathbf { j } ^ { r } )$ . Here Y-axis is perpendicular to the ground. $r ^ { v a } , r ^ { v x } , r ^ { v z } \in \mathbb { R }$ denotes the root joint’s angular velocity along Y-axis, linear velocity along Xaxis and Z-axis, respectively. $r ^ { h } \in \mathbb { R }$ is the height of the root joint. $\mathbf { j } ^ { p } , \mathbf { j } ^ { v } \in \mathbb { R } ^ { J \times 3 }$ are the position and linear velocity of each joint, where J is the number of joints. $\mathbf { j } ^ { r } \in \mathbb { R } ^ { J \times 6 }$ is the 6D rotation (Zhou et al., 2019) of each joint. Specifically, J is 22 in HumanML3D and 21 in KIT-ML.

Table 1 Quantitative results on the HumanML3D test set. All methods use the real motion length from the ground truth. ‘→’ means results are better if the metric is closer to the real motions. We run all the evaluation 20 times and ± indicates the 95% confidence interval. The best results are in bold.

<table><tr><td rowspan="2">Methods</td><td colspan="3">R Precision↑</td><td rowspan="2">FID↓</td><td rowspan="2">MultiModal Dist↓</td><td rowspan="2">Diversity→</td><td rowspan="2">MultiModality</td></tr><tr><td>Top 1</td><td>Top 2</td><td>Top 3</td></tr><tr><td>Real motions</td><td> $0.511^{\pm .003}$ </td><td> $0.703^{\pm .003}$ </td><td> $0.797^{\pm .002}$ </td><td> $0.002^{\pm .000}$ </td><td> $2.974^{\pm .008}$ </td><td> $9.503^{\pm .065}$ </td><td>-</td></tr><tr><td>Language2Pose</td><td> $0.246^{\pm .002}$ </td><td> $0.387^{\pm .002}$ </td><td> $0.486^{\pm .002}$ </td><td> $11.02^{\pm .046}$ </td><td> $5.296^{\pm .008}$ </td><td> $7.676^{\pm .058}$ </td><td>-</td></tr><tr><td>Text2Gesture</td><td> $0.165^{\pm .001}$ </td><td> $0.267^{\pm .002}$ </td><td> $0.345^{\pm .002}$ </td><td> $7.664^{\pm .030}$ </td><td> $6.030^{\pm .008}$ </td><td> $6.409^{\pm .071}$ </td><td>-</td></tr><tr><td>MoCoGAN</td><td> $0.037^{\pm .000}$ </td><td> $0.072^{\pm .001}$ </td><td> $0.106^{\pm .001}$ </td><td> $94.41^{\pm .021}$ </td><td> $9.643^{\pm .006}$ </td><td> $0.462^{\pm .008}$ </td><td> $0.019^{\pm .000}$ </td></tr><tr><td>Dance2Music</td><td> $0.033^{\pm .000}$ </td><td> $0.065^{\pm .001}$ </td><td> $0.097^{\pm .001}$ </td><td> $66.98^{\pm .016}$ </td><td> $8.116^{\pm .006}$ </td><td> $0.725^{\pm .011}$ </td><td> $0.043^{\pm .001}$ </td></tr><tr><td>Guo et al.</td><td> $0.457^{\pm .002}$ </td><td> $0.639^{\pm .003}$ </td><td> $0.740^{\pm .003}$ </td><td> $1.067^{\pm .002}$ </td><td> $3.340^{\pm .008}$ </td><td> $9.188^{\pm .002}$ </td><td> $2.090^{\pm .083}$ </td></tr><tr><td>Ours</td><td> $0.491^{\pm .001}$ </td><td> $0.681^{\pm .001}$ </td><td> $0.782^{\pm .001}$ </td><td> $0.630^{\pm .001}$ </td><td> $3.113^{\pm .001}$ </td><td> $9.410^{\pm .049}$ </td><td> $1.553^{\pm .042}$ </td></tr></table>

Table 2 Quantitative results on the KIT-ML test set. All methods use the real motion length from the ground truth.

<table><tr><td rowspan="2">Methods</td><td colspan="3">R Precision↑</td><td rowspan="2">FID↓</td><td rowspan="2">MultiModal Dist↓</td><td rowspan="2">Diversity→</td><td rowspan="2">MultiModality</td></tr><tr><td>Top 1</td><td>Top 2</td><td>Top 3</td></tr><tr><td>Real motions</td><td> $0.424^{\pm .005}$ </td><td> $0.649^{\pm .006}$ </td><td> $0.779^{\pm .006}$ </td><td> $0.031^{\pm .004}$ </td><td> $2.788^{\pm .012}$ </td><td> $11.08^{\pm .097}$ </td><td>-</td></tr><tr><td>Language2Pose</td><td> $0.221^{\pm .005}$ </td><td> $0.373^{\pm .004}$ </td><td> $0.483^{\pm .005}$ </td><td> $6.545^{\pm .072}$ </td><td> $5.147^{\pm .030}$ </td><td> $9.073^{\pm .100}$ </td><td>-</td></tr><tr><td>Text2Gesture</td><td> $0.156^{\pm .004}$ </td><td> $0.255^{\pm .004}$ </td><td> $0.338^{\pm .005}$ </td><td> $12.12^{\pm .183}$ </td><td> $6.964^{\pm .029}$ </td><td> $9.334^{\pm .079}$ </td><td>-</td></tr><tr><td>MoCoGAN</td><td> $0.022^{\pm .002}$ </td><td> $0.042^{\pm .003}$ </td><td> $0.063^{\pm .003}$ </td><td> $82.69^{\pm .242}$ </td><td> $10.47^{\pm .012}$ </td><td> $3.091^{\pm .043}$ </td><td> $0.250^{\pm .009}$ </td></tr><tr><td>Dance2Music</td><td> $0.031^{\pm .002}$ </td><td> $0.058^{\pm .002}$ </td><td> $0.086^{\pm .003}$ </td><td> $115.4^{\pm .240}$ </td><td> $10.40^{\pm .016}$ </td><td> $0.241^{\pm .004}$ </td><td> $0.062^{\pm .002}$ </td></tr><tr><td>Guo et al.</td><td> $0.370^{\pm .005}$ </td><td> $0.569^{\pm .007}$ </td><td> $0.693^{\pm .007}$ </td><td> $2.770^{\pm .109}$ </td><td> $3.401^{\pm .008}$ </td><td> $10.91^{\pm .119}$ </td><td> $1.482^{\pm .065}$ </td></tr><tr><td>Ours</td><td> $0.417^{\pm .004}$ </td><td> $0.621^{\pm .004}$ </td><td> $0.739^{\pm .004}$ </td><td> $1.954^{\pm .062}$ </td><td> $2.958^{\pm .005}$ </td><td> $11.10^{\pm .143}$ </td><td> $0.730^{\pm .013}$ </td></tr></table>

Table 3 Ablation of the pretrained CLIP and the eficient attention technique. All results are reported on the KIT-ML test set.

<table><tr><td rowspan="2">CLIP</td><td rowspan="2">EFF</td><td colspan="3">R Precision↑</td></tr><tr><td>Top 1</td><td>Top 2</td><td>Top 3</td></tr><tr><td>N</td><td>N</td><td> $0.288^{\pm .004}$ </td><td> $0.440^{\pm .004}$ </td><td> $0.539^{\pm .004}$ </td></tr><tr><td>N</td><td>Y</td><td> $0.136^{\pm .003}$ </td><td> $0.233^{\pm .003}$ </td><td> $0.309^{\pm .003}$ </td></tr><tr><td>Y</td><td>N</td><td> $0.357^{\pm .004}$ </td><td> $0.555^{\pm .004}$ </td><td> $0.679^{\pm .005}$ </td></tr><tr><td>Y</td><td>Y</td><td> $0.417^{\pm .004}$ </td><td> $0.621^{\pm .004}$ </td><td> $0.739^{\pm .004}$ </td></tr></table>

Quantitative Results. We compare our proposed MotionDifuse with five baseline models: Language2Pose (Ahuja and Morency, 2019), Text2Gesture (Bhattacharya et al., 2021), MoCo-GAN (Tulyakov et al., 2018), Dance2Music (Lee et al., 2019), and Guo et al. (2022). All baselines’ performances are quoted from Guo et al. (2022). Table 1 and Table 2 show the quantitative comparison on the HumanML3D dataset and the KIT-ML dataset. Our proposed MotionDifuse outperforms all existing works with a remarkable margin in aspects of precision, FID, MultiDodal Distance, and Diversity. The precision of MotionDifuse is even close to that of real motions, which suggests that our generated motion sequences are satisfyingly high-fidelity and realistic.

To further understand the function of CLIP initialization and eficient attention, we report ablation results in Table 3. The models without pretrained CLIP sufer from severe performance drops, which indicates the necessity of a pretrained language model for the T2M task. As for eficient attention, it is significantly beneficial when we use CLIP simultaneously. However, this module also limits the model’s performance when without CLIP. A possible explanation for this phenomenon is that the global relation in eficient attention is misleading when the semantic information from given text is insuficient.

Guo et al. (2022) states that the results on the MultiModality metric should be larger whenever possible. However, the literature in action-conditioned motion generation task (Guo et al., 2020; Petrovich et al., 2021; Cervantes et al., 2022) argue that this metric should be close to the real motion. In the T2M task, it is dificult to calculate this metric of real motions. Therefore, we only report these results without comparison.

We explore how the size of architecture influences the performance. Table 4 suggests that the latent dimension plays a more important role. The models with 512 latent dimension significantly outperform the models with 256 latent dimension. On the contrary, the increase of the number of layers improves the performance when the latent dimension is either 128 or 256, but has little efect when the dimension is 512.

Qualitative Results Figure 4 shows a comparison between our method and Guo et al. (2022) as a baseline. We highlight that MotionDifuse achieves a balance between diversity and realness. For example, for prompt ‘A person swings a golf club and hits the ball’, our generated motions portraits the described motion more faithfully. In contrast, the baseline method has high multi-modality at the expenses of accuracy. In addition, given a complicated prompt such as ”A person is pushed hard to the left and then recovers into a standing position”, MotionDifuse is able to generate high-quality motions that reflects the detailed description whereas the baseline method fails to produce any meaningful movement.

![](images/f2e2999ac9e2a4da18e3259c6ca531a79adf6a5fd53e01225d2b8eb5532d502a.jpg)  
Fig. 4 Qualitative results on the HumanML3D dataset. We compare our method with Guo et al. (2022) and visualize two examples for each given prompt. MotionDifuse is able to achieve both accuracy and diversity.

Table 4 Ablation of the latent dimension and the number of transformer layers. All results are reported on the KIT-ML test set.

<table><tr><td rowspan="2">#layers</td><td rowspan="2">Dim</td><td colspan="3">R Precision↑</td></tr><tr><td>Top 1</td><td>Top 2</td><td>Top 3</td></tr><tr><td>4</td><td>128</td><td>0.033±.002</td><td>0.066±.003</td><td>0.097±.003</td></tr><tr><td>4</td><td>256</td><td>0.095±.002</td><td>0.166±.003</td><td>0.227±.003</td></tr><tr><td>4</td><td>512</td><td>0.405±.005</td><td>0.620±.005</td><td>0.743±.004</td></tr><tr><td>8</td><td>128</td><td>0.025±.002</td><td>0.053±.002</td><td>0.086±.002</td></tr><tr><td>8</td><td>256</td><td>0.198±.003</td><td>0.335±.004</td><td>0.441±.004</td></tr><tr><td>8</td><td>512</td><td>0.417±.004</td><td>0.621±.004</td><td>0.739±.004</td></tr><tr><td>12</td><td>128</td><td>0.031±.002</td><td>0.063±.003</td><td>0.091±.002</td></tr><tr><td>12</td><td>256</td><td>0.209±.003</td><td>0.348±.004</td><td>0.452±.003</td></tr><tr><td>12</td><td>512</td><td>0.412±.006</td><td>0.616±.004</td><td>0.741±.004</td></tr></table>

## 4.2 Action-conditioned Motion Generation

Datasets. HumanAct12 dataset (Guo et al., 2020) provides 12 kinds of motion sequences. This dataset is adapted from PHSPD dataset (Zou et al., 2020), which contains 1191 videos. HumanAct12 further arranges these videos into trimmed motion clips. UESTC dataset (Ji et al., 2018) is also a significant benchmark for action-conditioned motion generation tasks, which includes 25K motion sequences across 40 diferent action categories. Petrovich et al. (2021) further uses pre-trained VIBE (Kocabas et al., 2020) to extract SMPL (Loper et al., 2015) sequences from the UESTC dataset and provides pretrained action recognition model for evaluation.

Evaluation Metrics. Four evaluation metrics are applied for this task: FID, Accuracy, Diversity, and Multimodality. The pretrained action recognition module can directly calculate the average accuracy for all action categories without arranging mini-batches. This metric has a similar function to R Precision. The other three metrics have been introduced in Section 4.1. HumanAct12 has no oficial split, and we report the FID on the whole dataset. UESTC has a test split, so we report the FID on it, which is more representative than the train split. In this section, FID and Accuracy are two principal metrics. Our conclusion are mainly based on them.

Table 5 Quantitative results for Action-conditioned Motion Generation. As for UESTC dataset, we report FID on the test split. MM: MultiModality.

<table><tr><td rowspan="2">Methods</td><td colspan="4">HumanAct12</td><td colspan="4">UESTC</td></tr><tr><td>FID↓</td><td>Accuracy↑</td><td>Diversity→</td><td>MM→</td><td>FID↓</td><td>Accuracy↑</td><td>Diversity→</td><td>MM→</td></tr><tr><td>Real motions</td><td> $0.020^{\pm .010}$ </td><td> $0.997^{\pm .001}$ </td><td> $6.850^{\pm .050}$ </td><td> $2.450^{\pm .040}$ </td><td> $2.79^{\pm .29}$ </td><td> $0.988^{\pm .001}$ </td><td> $33.34^{\pm .320}$ </td><td> $14.16^{\pm .06}$ </td></tr><tr><td>Action2Motion</td><td> $0.338^{\pm .015}$ </td><td> $0.917^{\pm .003}$ </td><td> $6.879^{\pm .066}$ </td><td> $2.511^{\pm .023}$ </td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ACTOR</td><td> $0.12^{\pm .00}$ </td><td> $0.955^{\pm .008}$ </td><td> $6.84^{\pm .03}$ </td><td> $2.53^{\pm .02}$ </td><td> $23.43^{\pm 2.20}$ </td><td> $0.911^{\pm .003}$ </td><td> $31.96^{\pm .33}$ </td><td> $14.52^{\pm .09}$ </td></tr><tr><td>INR</td><td> $0.088^{\pm .004}$ </td><td> $0.973^{\pm .001}$ </td><td> $6.881^{\pm .048}$ </td><td> $2.569^{\pm .040}$ </td><td> $15.00^{\pm .09}$ </td><td> $0.941^{\pm .001}$ </td><td> $31.59^{\pm .19}$ </td><td> $14.68^{\pm .07}$ </td></tr><tr><td>Ours</td><td> $0.07^{\pm .00}$ </td><td> $0.992^{\pm .13}$ </td><td> $6.85^{\pm .02}$ </td><td> $2.46^{\pm .02}$ </td><td> $9.10^{\pm .437}$ </td><td> $0.950^{\pm .000}$ </td><td> $32.42^{\pm .214}$ </td><td> $14.74^{\pm .07}$ </td></tr></table>

Implementation Details. All the setting are the same to those for text-driven motion generation tasks except for the learning rate, the number of iterations and the motion representation. In this series of expreiments, we train 100K iterations for the HumanAct12 dataset and 500K for the UESTC dataset, both with a 0.0001 learning rate.

Motion represention in this task is slightly diferent from the T2M task. As for the HumanAct12 dataset, each pose state can be representated as $( \mathbf { j } ^ { x } , \mathbf { j } ^ { y } , \mathbf { j } ^ { z } )$ , where $\mathbf { j } ^ { x } , \mathbf { j } ^ { y } , \mathbf { j } ^ { z } \in \mathbb { R } ^ { 2 4 \times 3 }$ are the coordinates of 24 joints. We use $( r ^ { x } , r ^ { y } , r ^ { z } , \mathbf { j } ^ { r } )$ as the pose representation for the UESTC dataset, where $r ^ { x } , r ^ { y } , r ^ { z } \in \mathbb { R }$ are the coordinates of the root joint, and $\mathbf { j } ^ { r } \in \mathbb { R } ^ { 2 4 \times 6 }$ is the rotation angle of each joint in 6D representation.

Quantitative Results. Following Cervantes et al. (2022), three baseline models are selected: Action2Motion (Guo et al., 2020), ACTOR (Petrovich et al., 2021), INR (Cervantes et al., 2022). Table 5 shows the quantitative results on the HumanAct12 dataset and the UESTC datasets. Our proposed MotionDifuse achieves the best performance in aspects of FID and Accuracy when compared to other existing works. We want to highlight that our results of the HumanAct12 dataset are notably close to real motions on all four metrics.

Table 6 Ablation of the latent dimension and the number of transformer layers. All results are reported on the HumanAct12 dataset.

<table><tr><td>#layers</td><td>Dim</td><td>FID↓</td><td>Accuracy↑</td></tr><tr><td>4</td><td>128</td><td> $0.29^{\pm 0.00}$ </td><td> $0.892^{\pm 1.97}$ </td></tr><tr><td>4</td><td>256</td><td> $0.14^{\pm 0.00}$ </td><td> $0.958^{\pm 0.51}$ </td></tr><tr><td>4</td><td>512</td><td> $0.09^{\pm 0.00}$ </td><td> $0.984^{\pm 0.21}$ </td></tr><tr><td>8</td><td>128</td><td> $0.22^{\pm 0.00}$ </td><td> $0.929^{\pm 1.04}$ </td></tr><tr><td>8</td><td>256</td><td> $0.09^{\pm 0.00}$ </td><td> $0.983^{\pm 0.23}$ </td></tr><tr><td>8</td><td>512</td><td> $\mathbf{0.07^{\pm 0.00}}$ </td><td> $0.992^{\pm 0.13}$ </td></tr><tr><td>12</td><td>128</td><td> $0.11^{\pm 0.00}$ </td><td> $0.954^{\pm 0.67}$ </td></tr><tr><td>12</td><td>256</td><td> $0.10^{\pm 0.00}$ </td><td> $0.988^{\pm 0.21}$ </td></tr><tr><td>12</td><td>512</td><td> $0.08^{\pm 0.00}$ </td><td> $\mathbf{0.996^{\pm 0.08}}$ </td></tr></table>

Here we also try diferent combination of latent dimension and the number of layers, as shown in Table 6. Similar to the conclusions we found in Section 4.1, latent dimension is more important than the number of layers.

## 4.3 Motion Manipulation

To better evaluate the capability of text-driven motion generation models, we design two task variants. First, Spatially-diverse T2M task (T2M-S). T2M-S requires the generated motion sequence to contain multiple actions on diferent body parts $( e . g . \ ^ { \circ } \mathrm { \dot { a } }$ person is running and drinking water simultaneously’). Specifically, i-th test sample in T2M-S task can be represented by a set of text-mask pairs $\{ ( \mathrm { t e x t } _ { i , j } , \mathrm { M } _ { i , j } ) \}$ }, where $\mathrm { M } _ { i , j } \in \{ 0 , 1 \} ^ { D }$ is a D-dimension binary vector. It indicates which body part we should focus on when given the text description $\mathrm { t e x t } _ { i , j } .$ . Second, Temporallydiverse T2M task (T2M-T). T2M-T expects models to generate a long motion sequence, which includes multiple actions in a specific order spanning over diferent time intervals (e.g. ‘a person is walking and then running’). The i-th test sample is an array of text-duration pairs $\{ \mathrm { t e x t } _ { i , j } , [ l _ { i , j } , r _ { i , j } ] \} , l _ { i , j } < r _ { i , j }$ . It means that the motion clip from $l _ { i , j } \mathrm { - t h }$ frame to $r _ { i , j }$ frame is supposed to contain the action tex $^ { ; } i , j ^ { \cdot }$

Implementation Details. We train our proposed MotionDifuse on the BABEL dataset (Punnakkal et al., 2021) with 50K iterations. Each pose state is represented by $( r ^ { x } , r ^ { y } , r ^ { z } , \mathbf { j } ^ { r } )$ , which is same to the setting for the UESTC dataset. Other settings remain unchanged. $\lambda _ { 1 } = \lambda _ { 2 } = 0 . 0 1$ are used for the visualization.

Qualitative Results. As shown in Fig 5, MotionDiffuse has the capability to handle highly comprehensive prompts that assign motions to multiple body parts (such as “Kicking and punching” and “Jumping and raising arms” that require coordination of the upper and lower body). Moreover, MotionDifuse is able to generate long sequences according to a complex instruction that includes multiple actions (such as “Tying the shoe, standing up and then walking forward” that includes a series of vastly diferent motions).

![](images/b6ded070c760e45eb3c4bdafb851187d5413daab63acd9057a82784ff218bff5.jpg)

![](images/e5757fbb65523a17eab251b023b95f52ee2f7ef0c47f1f0d921c8672cd02ba72.jpg)  
d) Tying the shoe, standing up and then walking forward

Fig. 5 Qualitative results on the BABEL dataset. MotionDifuse is able to generate dynamic sequences according to complicated prompt that involves multiple body parts or actions.

## 5 Conclusion, Limitations and Future Work

We propose MotionDifuse, the first difusion modelbased method for text-driven motion generation. MotionDifuse demonstrates three major strengths: Probabilistic Mapping that enhances diversity, Realistic Synthesis that ensures plausibility of motion sequences, and Multi-Level Manipulation that allows for per-part manipulation and long sequence generation. Both quantitative and qualitative evaluations show that MotionDiffuse outperforms existing arts on various tasks such as text-driven motion generation and action-conditioned motion generation, and demonstrates remarkable motion manipulation capabilities.

Although MotionDifuse has pushed forward the performance boundary of motion generation tasks, there still exist some problems. First, difusion models require a large amount of difusion steps during inference and it is challenging to generate motion sequences in real-time. Second, current pipeline only accepts a single form of motion representation. A more generalized pipeline that adapts concurrently to all datasets would be more versatile for various scenarios.

Acknowledgements This work is supported by NTU NAP, MOE AcRF Tier 2 (T2EP20221-0033), and under the RIE2020 Industry Alignment Fund - Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and inkind contribution from the industry partner(s). Corresponding author: Ziwei Liu (ziwei.liu@ntu.edu.sg).

## References

Ahuja C, Morency LP (2019) Language2pose: Natural language grounded pose forecasting. In: 2019 International Conference on 3D Vision (3DV), IEEE, pp 719–728

Aliakbarian S, Saleh FS, Salzmann M, Petersson L, Gould S (2020) A stochastic conditioning scheme for diverse human motion prediction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 5223–5232

Andriluka M, Iqbal U, Insafutdinov E, Pishchulin L, Milan A, Gall J, Schiele B (2018) Posetrack: A benchmark for human pose estimation and tracking. In: Proceedings of the IEEE conference on computer vision and pattern recognition, pp 5167–5176

Barsoum E, Kender J, Liu Z (2018) Hp-gan: Probabilistic 3d human motion prediction via gan. In: Proceedings of the IEEE conference on computer vision and pattern recognition workshops, pp 1418–1427

Bhattacharya U, Rewkowski N, Banerjee A, Guhan P, Bera A, Manocha D (2021) Text2gestures: A transformer-based network for generating emotive body gestures for virtual agents. In: 2021 IEEE Virtual Reality and 3D User Interfaces (VR), IEEE, pp 1–10

Cai Z, Zhang M, Ren J, Wei C, Ren D, Li J, Lin Z, Zhao H, Yi S, Yang L, et al. (2021) Playing for 3d human recovery. arXiv preprint arXiv:211007588

Cai Z, Ren D, Zeng A, Lin Z, Yu T, Wang W, Fan X, Gao Y, Yu Y, Pan L, et al. (2022) Humman: Multimodal 4d human dataset for versatile sensing and modeling. arXiv preprint arXiv:220413686

Cao Z, Gao H, Mangalam K, Cai QZ, Vo M, Malik J (2020) Long-term human motion prediction with scene context. In: European Conference on Computer Vision, Springer, pp 387–404

Carreira J, Noland E, Hillier C, Zisserman A (2019) A short note on the kinetics-700 human action dataset. arXiv preprint arXiv:190706987

Cervantes P, Sekikawa Y, Sato I, Shinoda K (2022) Implicit neural representations for variable length human motion generation. arXiv preprint arXiv:220313694

Chung J, Wuu Ch, Yang Hr, Tai YW, Tang CK (2021) Haa500: Human-centric atomic action dataset with curated videos. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 13465–13474

Dhariwal P, Nichol A (2021) Difusion models beat gans on image synthesis. Advances in Neural Information Processing Systems 34

Dinh L, Krueger D, Bengio Y (2014) Nice: Non-linear independent components estimation. arXiv preprint arXiv:14108516

Dinh L, Sohl-Dickstein J, Bengio S (2016) Density estimation using real nvp. arXiv preprint arXiv:160508803

Futrelle R, Speckert G (1978) Extraction of motion data by interactive processing. In: Proc. 1978 IEEE Conf. Pattern Recog. and Image Processing, Chicago, IL, pp 405–408

Gavrila DM (1999) The visual analysis of human movement: A survey. Computer vision and image understanding 73(1):82–98

Ghosh A, Cheema N, Oguz C, Theobalt C, Slusallek P (2021) Synthesis of compositional animations from textual descriptions. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 1396–1406

Goodfellow I, Pouget-Abadie J, Mirza M, Xu B, Warde-Farley D, Ozair S, Courville A, Bengio Y (2014) Generative adversarial nets. Advances in neural information processing systems 27

Gu C, Sun C, Ross DA, Vondrick C, Pantofaru C, Li Y, Vijayanarasimhan S, Toderici G, Ricco S, Sukthankar R, et al. (2018) Ava: A video dataset of spatio-temporally localized atomic visual actions. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp 6047–6056

Guo C, Zuo X, Wang S, Zou S, Sun Q, Deng A, Gong M, Cheng L (2020) Action2motion: Conditioned gen-

eration of 3d human motions. In: Proceedings of the 28th ACM International Conference on Multimedia, pp 2021–2029

Guo C, Zou S, Zuo X, Wang S, Ji W, Li X, Cheng L (2022) Generating diverse and natural 3d human motions from text. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 5152–5161

Harvey FG, Yurick M, Nowrouzezahrai D, Pal C (2020) Robust motion in-betweening. ACM Transactions on Graphics (TOG) 39(4):60–1

Hendrycks D, Gimpel K (2016) Gaussian error linear units (gelus). arXiv preprint arXiv:160608415

Henter GE, Alexanderson S, Beskow J (2020) Moglow: Probabilistic and controllable motion synthesis using normalising flows. ACM Transactions on Graphics (TOG) 39(6):1–14

Ho J, Jain A, Abbeel P (2020) Denoising difusion probabilistic models. Advances in Neural Information Processing Systems 33:6840–6851

Hochreiter S, Schmidhuber J (1997) Long short-term memory. Neural computation 9(8):1735–1780

Hong F, Zhang M, Pan L, Cai Z, Yang L, Liu Z (2022) Avatarclip: Zero-shot text-driven generation and animation of 3d avatars. arXiv preprint arXiv:220508535

Huang R, Hu H, Wu W, Sawada K, Zhang M, Jiang D (2020) Dance revolution: Long-term dance generation with music via curriculum learning. arXiv preprint arXiv:200606119

Ikemoto L, Arikan O, Forsyth D (2009) Generalizing motion edits with gaussian processes. ACM Transactions on Graphics (TOG) 28(1):1–12

Ionescu C, Papava D, Olaru V, Sminchisescu C (2013) Human3.6m: Large scale datasets and predictive methods for 3d human sensing in natural environments. IEEE transactions on pattern analysis and machine intelligence 36(7):1325–1339

Jain A, Tancik M, Abbeel P (2021) Putting nerf on a diet: Semantically consistent few-shot view synthesis. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 5885–5894

Jhuang H, Gall J, Zufi S, Schmid C, Black MJ (2013) Towards understanding action recognition. In: Proceedings of the IEEE international conference on computer vision, pp 3192–3199

Ji Y, Xu F, Yang Y, Shen F, Shen HT, Zheng WS (2018) A large-scale rgb-d database for arbitraryview human action recognition. In: Proceedings of the 26th ACM international Conference on Multimedia, pp 1510–1518

Joo H, Liu H, Tan L, Gui L, Nabbe B, Matthews I, Kanade T, Nobuhara S, Sheikh Y (2015) Panoptic studio: A massively multiview system for social mo-

tion capture. In: Proceedings of the IEEE International Conference on Computer Vision, pp 3334–3342

Karpathy A, Toderici G, Shetty S, Leung T, Sukthankar R, Fei-Fei L (2014) Large-scale video classification with convolutional neural networks. In: Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pp 1725–1732

Kingma DP, Dhariwal P (2018) Glow: Generative flow with invertible 1x1 convolutions. Advances in neural information processing systems 31

Kingma DP, Welling M (2013) Auto-encoding variational bayes. arXiv preprint arXiv:13126114

Kocabas M, Athanasiou N, Black MJ (2020) Vibe: Video inference for human body pose and shape estimation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp 5253–5263

Kuehne H, Jhuang H, Garrote E, Poggio T, Serre T (2011) Hmdb: a large video database for human motion recognition. In: 2011 International conference on computer vision, IEEE, pp 2556–2563

Lee HY, Yang X, Liu MY, Wang TC, Lu YD, Yang MH, Kautz J (2019) Dancing to music. Advances in Neural Information Processing Systems 32

Li J, Yin Y, Chu H, Zhou Y, Wang T, Fidler S, Li H (2020) Learning to generate diverse dance motions with transformer. arXiv preprint arXiv:200808171

Li R, Yang S, Ross DA, Kanazawa A (2021) Ai choreographer: Music conditioned 3d dance generation with aist++. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 13401– 13412

Lin AS, Wu L, Corona R, Tai K, Huang Q, Mooney RJ (2018) Generating animated videos of human activities from natural language descriptions. In: NeurIPS Workshop

Liu J, Shahroudy A, Perez M, Wang G, Duan LY, Kot AC (2019) Ntu rgb+ d 120: A large-scale benchmark for 3d human activity understanding. IEEE transactions on pattern analysis and machine intelligence 42(10):2684–2701

Loper M, Mahmood N, Romero J, Pons-Moll G, Black MJ (2015) Smpl: A skinned multi-person linear model. ACM transactions on graphics (TOG) 34(6):1–16

Mahmood N, Ghorbani N, Troje NF, Pons-Moll G, Black MJ (2019) Amass: Archive of motion capture as surface shapes. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 5442–5451

Mehta D, Rhodin H, Casas D, Fua P, Sotnychenko O, Xu W, Theobalt C (2017) Monocular 3d human pose estimation in the wild using improved cnn supervi-

sion. In: 2017 international conference on 3D vision (3DV), IEEE, pp 506–516

Mildenhall B, Srinivasan PP, Tancik M, Barron JT, Ramamoorthi R, Ng R (2020) Nerf: Representing scenes as neural radiance fields for view synthesis. In: European conference on computer vision, Springer, pp 405–421

Min J, Chai J (2012) Motion graphs++ a compact generative model for semantic motion analysis and synthesis. ACM Transactions on Graphics (TOG) 31(6):1–12

Monfort M, Andonian A, Zhou B, Ramakrishnan K, Bargal SA, Yan T, Brown L, Fan Q, Gutfreund D, Vondrick C, et al. (2019) Moments in time dataset: one million videos for event understanding. IEEE transactions on pattern analysis and machine intelligence 42(2):502–508

Mukai T, Kuriyama S (2005) Geostatistical motion interpolation. In: ACM SIGGRAPH 2005 Papers, pp 1062–1070

Nichol A, Dhariwal P, Ramesh A, Shyam P, Mishkin P, McGrew B, Sutskever I, Chen M (2021) Glide: Towards photorealistic image generation and editing with text-guided difusion models. arXiv preprint arXiv:211210741

Nichol AQ, Dhariwal P (2021) Improved denoising diffusion probabilistic models. In: International Conference on Machine Learning, PMLR, pp 8162–8171

Ormoneit D, Black MJ, Hastie T, Kjellstr¨om H (2005) Representing cyclic human motion using functional analysis. Image and Vision Computing 23(14):1264– 1276

O’rourke J, Badler NI (1980) Model-based image analysis of human motion using constraint propagation. IEEE Transactions on Pattern Analysis and Machine Intelligence (6):522–536

Patel P, Huang CHP, Tesch J, Hofmann DT, Tripathi S, Black MJ (2021) AGORA: Avatars in geography optimized for regression analysis. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 13468–13478

Petrovich M, Black MJ, Varol G (2021) Actionconditioned 3d human motion synthesis with transformer vae. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 10985– 10995

Petrovich M, Black MJ, Varol G (2022) Temos: Generating diverse human motions from textual descriptions. arXiv preprint arXiv:220414109

Plappert M, Mandery C, Asfour T (2016) The kit motion-language dataset. Big data 4(4):236–252

Punnakkal AR, Chandrasekaran A, Athanasiou N, Quiros-Ramirez A, Black MJ (2021) Babel: Bodies,

action and behavior with english labels. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 722–731

Radford A, Kim JW, Hallacy C, Ramesh A, Goh G, Agarwal S, Sastry G, Askell A, Mishkin P, Clark J, et al. (2021) Learning transferable visual models from natural language supervision. arXiv preprint arXiv:210300020

Ramesh A, Dhariwal P, Nichol A, Chu C, Chen M (2022) Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:220406125

Rose C, Cohen MF, Bodenheimer B (1998) Verbs and adverbs: Multidimensional motion interpolation. IEEE Computer Graphics and Applications 18(5):32–40

Shao D, Zhao Y, Dai B, Lin D (2020) Finegym: A hierarchical video dataset for fine-grained action understanding. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 2616–2625

Shen Z, Zhang M, Zhao H, Yi S, Li H (2021) Eficient attention: Attention with linear complexities. In: Proceedings of the IEEE/CVF winter conference on applications of computer vision, pp 3531–3539

Siyao L, Yu W, Gu T, Lin C, Wang Q, Qian C, Loy CC, Liu Z (2022) Bailando: 3d dance generation by actorcritic gpt with choreographic memory. arXiv preprint arXiv:220313055

Soomro K, Zamir AR, Shah M (2012) Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:12120402

Sun G, Wong Y, Cheng Z, Kankanhalli MS, Geng W, Li X (2020) Deepdance: music-to-dance motion choreography with adversarial learning. IEEE Transactions on Multimedia 23:497–509

Tevet G, Gordon B, Hertz A, Bermano AH, Cohen-Or D (2022) Motionclip: Exposing human motion generation to clip space. arXiv preprint arXiv:220308063

Trumble M, Gilbert A, Malleson C, Hilton A, Collomosse JP (2017) Total capture: 3d human pose estimation fusing video and inertial sensors. In: BMVC, vol 2, pp 1–13

Tulyakov S, Liu MY, Yang X, Kautz J (2018) Mocogan: Decomposing motion and content for video generation. In: Proceedings of the IEEE conference on computer vision and pattern recognition, pp 1526– 1535

Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, Kaiser L, Polosukhin I (2017) Attention is all you need. Advances in neural information processing systems 30

Wang Z, Yu P, Zhao Y, Zhang R, Zhou Y, Yuan J, Chen C (2020) Learning diverse stochastic human-action generators by learning smooth latent transitions. In: Proceedings of the AAAI conference on artificial intelligence, vol 34, pp 12281–12288

Yan X, Rastogi A, Villegas R, Sunkavalli K, Shechtman E, Hadap S, Yumer E, Lee H (2018) Mt-vae: Learning motion transformations to generate multimodal human dynamics. In: Proceedings of the European conference on computer vision (ECCV), pp 265–281

Yu Z, Yoon JS, Lee I, Venkatesh P, Park J, Yu J, Park H (2020) Humbi: A large multiview dataset of human body expressions. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) pp 2987–2997

Zhang M, Yu L, Zhang K, Du B, Zhan B, Chen S, Jiang X, Guo S, Zhao J, Wang Y, et al. (2020) Kinematic dataset of actors expressing emotions. Scientific data 7(1):1–8

Zhao H, Torralba A, Torresani L, Yan Z (2019) Hacs: Human action clips and segments dataset for recognition and temporal localization. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 8668–8678

Zhou Y, Barnes C, Lu J, Yang J, Li H (2019) On the continuity of rotation representations in neural networks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 5745–5753

Zhuang W, Wang C, Xia S, Chai J, Wang Y (2020) Music2dance: Dancenet for music-driven dance generation. arXiv preprint arXiv:200203761

Zou S, Zuo X, Qian Y, Wang S, Xu C, Gong M, Cheng L (2020) 3d human shape reconstruction from a polarization image. In: European Conference on Computer Vision, Springer, pp 351–368