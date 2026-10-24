# Fine-grained Motion Retrieval via Joint-Angle Motion Images and Token-Patch Late Interaction

Yao Zhang<sup>1</sup>, Zhuchenyang Liu<sup>1</sup>, Yanlan He<sup>2</sup>, Thomas Ploetz<sup>3</sup>, and Yu Xiao<sup>1</sup>

<sup>1</sup> Aalto University, Espoo, Finland

{yao.1.zhang,zhuchenyang.liu,yu.xiao}@aalto.fi

<sup>2</sup> Fudan University, Shanghai, China

23307130480@m.fudan.edu.cn <sup>3</sup> Georgia Institute of Technology, Atlanta GA, USA thomas.ploetz@gatech.edu

Abstract. Text-motion retrieval aims to learn a semantically aligned latent space between natural language descriptions and 3D human motion skeleton sequences, enabling bidirectional search across the two modalities. Most existing methods use a dual-encoder framework that compresses motion and text into global embeddings, discarding finegrained local correspondences, and thus reducing accuracy. Additionally, these global-embedding methods ofer limited interpretability of the retrieval results. To overcome these limitations, we propose an interpretable, joint-angle-based motion representation that maps jointlevel local features into a structured pseudo-image, compatible with pretrained Vision Transformers. For text-to-motion retrieval, we employ MaxSim, a token-wise late interaction mechanism, and enhance it with Masked Language Modeling regularization to foster robust, interpretable text-motion alignment. Extensive experiments on HumanML3D and KIT-ML show that our method outperforms state-of-the-art text-motion retrieval approaches while ofering interpretable fine-grained correspondences between text and motion.

Keywords: Text-Motion Retrieval · Motion Understanding · Fine-grained Cross-modal Alignment · Late Interaction · Motion Representation

## 1 Introduction

Text-motion retrieval is a pivotal task in human motion understanding, whose core objective is to establish a semantic-aligned latent space between 3D human motion skeleton sequences and natural language descriptions [9, 22, 23]. A wellaligned latent space facilitates bidirectional retrieval, where the accuracy and granularity of alignment significantly impacts the performance of downstream tasks such as text-driven motion generation [28], motion captioning [10,13], and language-guided motion editing [38].

Recent eforts aim to achieve accurate and fine-grained alignment from various perspectives. On the motion representation front, some studies [34, 37] have converted motion skeleton sequences into pseudo-images to leverage pre-trained vision models. However, these representations, derived directly from raw 3D joint positions, conflate global translational movement with individual joint movements, hindering the distinction of subtle kinematic diferences (see Sec. 3.1). For text modality, previous works have employed large language models for generating augmented or part-level descriptions [37, 39], or developed complex cross-modal attention modules for finer alignment [27,31]. Nonetheless, these solutions introduce significant external dependencies and computational overhead. Furthermore, most existing methods, such as TMR [22], adhere to a globalembedding alignment paradigm by encoding the input sequences into single global vectors for alignment. Although computationally eficient, this globalembedding paradigm compresses the rich content of a motion sequence and its textual description into a single vector, inevitably discarding fine-grained local information. As a result, it restricts retrieval performance and complicates the alignment of specific text tokens with precise body movements, essential for distinguishing similar motions.

This paper tackles these challenges from two key perspectives. First, we address the issue of conflating global and local joint movements by introducing a joint-angle-based motion representation. Anatomically defined joint angles [29,30] are translation-invariant and encode each joint independently, explicitly decoupling the body’s global trajectory from local joint movement (detailed in Sec. 2.1). By projecting these decoupled features, rather than raw 3D joint positions, into a structured 224×224 pseudo-image, we efectively leverage visual priors from pre-trained Vision Transformers (ViT). This approach ensures that distinct spatial regions correspond to specific joints, providing a natural basis for part-level alignment.

Second, inspired by late interaction mechanisms efective in text retrieval [14] and visual document retrieval [6], we replace the global-embedding alignment with a Maximum Similarity (MaxSim) operator for explicit token-to-patch matching. This operator retains the maximum similarity score for each text token across all motion patches, producing interpretable, fine-grained correspondence maps. However, since MaxSim computes similarity at the individual token level, a key challenge arises: each token and patch embedding must carry suficient contextual information to support reliable matching. On the motion side, sequences are often dominated by static or transitional poses, which may lead MaxSim to assign high similarity to uninformative patches that happen to be feature-similar to a query token. On the text side, semantically vacuous tokens such as “a” or “person” can act as noise anchors, matching to arbitrary motion patches and diluting the alignment signal; even content words like “hand” may match unrelated patches if they lack sentence-level context. To address these, we introduce Masked Language Modeling (MLM) [4] as regularization, training the text encoder to reconstruct masked tokens from their surrounding context. This innovative approach produces contextually enriched embeddings that stabilize fine-grained matching, all without the need for data augmentation or external models. To the best of our knowledge as discussed in Sec. 2.2 and 2.3, this is the first work to replace the global-embedding paradigm in text-motion retrieval with a structurally grounded, fine-grained late interaction mechanism.

Extensive experiments on HumanML3D and KIT-ML demonstrate that our approach, which combines joint-angle-based motion representation with tokenpatch late interaction, consistently surpasses state-of-the-art text–motion retrieval baselines. Additionally, it produces interpretable correspondence maps that transparently align textual semantics with specific body joints and temporal phases. Details of our methods and evaluation results are presented in Secs. 3 and 4, respectively, followed by the conclusion in Sec. 5.

## 2 Background and Related Work

This section reviews existing human motion representations (Sec. 2.1), textmotion retrieval approaches (Sec. 2.2), and late interaction mechanisms used in text or document retrieval (Sec. 2.3).

## 2.1 Human Motion Representations

3D human motion sequences are typically represented as frame-wise skeletal features. The most widely adopted format is the 263-dimensional feature vector proposed by Guo et al. [9], which concatenates, per frame, the root angular velocity, root linear velocity, joint positions, 6D joint rotations, joint velocities, and binary foot-contact labels. Recent motion-generation studies revisit the choice of motion representation: MARDM [21] analyzes the redundancy of such multiview pose encodings, while ACMDM [20] shows that directly using absolute joint coordinates can ease motion generation. For retrieval tasks, a compact and discriminative representation is also preferrable. MoPatch [34] took a step forward by converting motion sequences into image-like patches and leveraging pre-trained ViT encoders, demonstrating that visual priors from ImageNet can alleviate data scarcity in motion understanding. However, its motion images are constructed directly from raw 3D joint positions grouped by body region, obscuring fine-grained kinematic diferences between joints.

In biomechanics, the Joint Coordinate Systems (JCS) standardized by the International Society of Biomechanics (ISB) decompose each joint’s motion into clinically meaningful components, such as flexion/extension, abduction/adduction and internal/external rotation, along well-defined anatomical planes [29, 30]. These angles ofer anatomical interpretability, with each degree of freedom (DoF) corresponding to a specific type of joint articulation [26]. They are also inherently translation-invariant, describing how a joint moves relative to its parent segment (e.g., the thigh is the parent segment of the knee joint, and the pelvis is the parent of the hip). Despite these benefits, joint angles remain underexplored for cross-modal motion retrieval. We address this by constructing a structured Motion Image from per-joint angular features, with each spatial region encoding a distinct joint to enable fine-grained token-to-patch alignment.

## 2.2 Text-Motion Retrieval

Cross-modal retrieval between natural language and 3D human skeleton motion has attracted growing attention [9, 19, 23]. TMR [22], inspired by the architecture of image-text [24] and video-text retrieval models [32], introduced a dual-encoder framework for text-motion retrieval in which a motion encoder and a text encoder independently map their inputs into a shared latent space, then the Text-to-Motion (T2M) and Motion-to-Text (M2T) retrieval are performed via cosine similarity between the resulting global embeddings. Using a similar architecture as TMR [22], several works seek finer-grained alignment by either leveraging LLM to enrich the training data (e.g., CAR [7], SGAR [37], and KinMo [39]) or introducing coarser notions of locality (e.g., Lyu et al. [31], ReMoGPT [35], SECL [27]); more recent eforts pursue part-specific or hierarchical fine-grained matching (e.g., WaMo [25], Chen et al. [2]). Recent works also target multi-modal retrieval that aligns motion with additional modalities such as audio or images to learn a broader joint embedding space (e.g., Motion-Bind [15], MonSTeR [3], and Multi-Modal Motion Retrieval [33, 36]). However, none of these establishes direct, structurally grounded correspondence between individual words and specific body regions or temporal segments, precisely the capability our token-to-patch alignment provides.

## 2.3 Late Interaction in Retrieval

Late interaction was introduced by ColBERT [14] for text retrieval. Unlike singlevector models that compress documents into one embedding, ColBERT retains per-token representations and computes relevance via a Maximum Similarity (MaxSim) operator: each text token in the query finds its best-matching document token, and the final score aggregates these maxima. This preserves finegrained detail while retaining the computational eficiency of independent encoding. Specifically, unlike cross-attention mechanisms [4, 27] that require joint processing of the query and every gallery item through the model at runtime, late interaction allows all gallery items to be pre-encoded ofline. During inference, only the query needs a forward pass, and retrieval is achieved via lightweight similarity computations. The paradigm has since been extended to multi-modal settings: ColPali [6] applies it to visual document retrieval, matching query text tokens against image patch embeddings without requiring OCR. These developments show that late interaction naturally suits cross-modal tasks where finegrained correspondences exist between modality elements, however, it has not been explored in the motion domain. To the best of our knowledge, our work is the first to introduce this mechanism, enhanced with MLM-based regularization, for text-motion retrieval.

## 3 Methodology

To enable interpretable fine-grained text-motion retrieval, we propose a framework designed to overcome the information bottleneck of traditional global embeddings. As illustrated in Fig. 1, given a 3D skeletal motion sequence (represented as per-frame joint positions from SMPL [9] or similar skeletons) and a natural language description, our pipeline operates in three steps.

![](images/8616a821e6d2f4ec9fdcbef57b85305ead78a0ac4eee66cc6bc68861d97aaac3.jpg)  
Fig. 1: Overview of the three-stage training pipeline.

First, to explicitly decouple local joint movements from global body trajectories, a raw motion sequence is converted into a structured 224×224 pseudo-image, termed theMotion Image. This is achieved by extracting joint angle features via inverse kinematics and projecting each joint’s Degrees of Freedom (DoF) into a uniform 16-pixel horizontal band, so that each spatial region encodes a distinct joint (Sec. 3.1).

Second, the Motion Image is fed into a Vision Transformer, while the text description is processed by a Transformer-based language model. They output patch-level and token-level embeddings, respectively, preserving spatiotemporal nuances for downstream matching instead of collapsing everything into a single global vector (Sec. 3.2). These embeddings are combined through a MaxSim late interaction mechanism [14] to yield a fine-grained MaxSim Score for text-motion alignment (Sec. 3.3). This score serves as the unified retrieval metric for both training and inference: during training, it is used within in-batch contrastive learning $\left( \mathcal { L } _ { t 2 m } \right)$ ; during inference, it directly ranks candidate motions (or texts) for retrieval.

Third, because token-level matching is sensitive to semantic noise, it is paired with a MLM auxiliary objective that enriches token embeddings with sentencelevel context to stabilize the alignment (Sec. 3.4). MLM has been widely used alongside contrastive objectives in vision-language pre-training [1,16] to improve text representation quality; in our framework, it serves the targeted purpose of ensuring that each token embedding encodes not just lexical identity but also its role within the broader sentence, providing a stable foundation for fine-grained cross-modal matching.

The third step is used only during training; the others are performed in both training and inference. Regarding inference, for T2M retrieval, given a text query $\tau$ and a gallery of G candidate motions $\{ \mathcal { M } _ { g } \} _ { g = 1 } ^ { G }$ , the text encoder and motion encoder independently produce token-level and patch-level embeddings, respectively. The MaxSim Score Sim $( \mathcal { T } , \mathcal { M } _ { g } )$ is computed for each candidate, and motions are ranked accordingly. For M2T retrieval, the roles are reversed: a motion query is scored against all candidate texts using the transposed similarity matrix. Since the two encoders are independent, candidate embeddings need only be computed once and can be reused across multiple queries.

## 3.1 Joint-angle-based Motion Representation

Joint angles describe how a joint bends or rotates relative to its parent limb segment, regardless of where the body is located in world coordinates. Unlike previous methods [22, 31, 34] that rely on raw joint positions, we adopt a jointangle-based motion representation to explicitly decouple local joint movement from global movement to construct fine-grained Motion Image. This provides three benefits for retrieval: (1) each spatial band in the Motion Image encodes a distinct joint’s behavior independently, enabling part-level alignment; (2) the representation is naturally robust to global translation and heading variation; and (3) discriminative kinematic patterns $( e . g .$ , the periodic hip flexion during walking) are preserved without being masked by smooth global drift. This anglebased representation is also invertible; we demonstrate the high-fidelity recovery of 3D joint positions via forward kinematics in Supplementary Material 1.4.

As illustrated in Fig. 2(a), we decompose a motion into $K { = } 1 4$ distinct kinematic joints (e.g., hips, knees, shoulders). Following the anatomical DoF definitions for each joint $( e . g . , 3$ dimensions for ball-and-socket joints such as hips, 1 dimension for hinge joints such as knees) and standard medical conventions for joint angle definitions [29, 30], we compute per-part angular features as well as global position from the raw joint positions via an inverse kinematics pipeline. This yields an input vector $p _ { t } \in \mathbb { R } ^ { 2 9 }$ at each time step t. Table 1 summarizes the computed joint angle features across all 14 joints.

Joint angle extraction. We construct a body-centric coordinate system $R _ { \mathrm { b o d y } }$ anchored to the pelvis at each frame (Supplementary Material 1.1), and extract joint angles via inverse kinematics depending on joint type. For ball-and-socket joints (hips, shoulders; 3-DoF), we transform the child position into the parent’s local frame to obtain $\tilde { v } \in \mathbb { R } ^ { 3 }$ and decompose it into flexion/extension and adduction/abduction:

$$
\theta_ {\mathrm{flex}} = \mathrm{atan2} (\tilde {v} _ {x}, - \tilde {v} _ {y}), \quad \theta_ {\mathrm{add}} = \mathrm{sign} (\tilde {v} _ {z}) \cdot \arccos \left(\frac {\tilde {v} \cdot \tilde {v} _ {x y}}{\| \tilde {v} \| \| \tilde {v} _ {x y} \|}\right)\tag{1}
$$

where $\tilde { v } _ { x y }$ is the sagittal-plane projection of v˜. The axial rotation $\theta _ { \mathrm { { r o t } } }$ is obtained by propagating the parent frame and measuring the grandchild twist (Supplementary Material 1.2). Hinge joints (1-DoF) use the angle between adjacent limb segments, and spinal joints (2–3 DoF) follow the same decomposition but reference the upward y-axis. All frames follow hierarchical recursive propagation, ensuring translation-invariant features (full derivation in Supplementary Material 1).

Table 1: Summary of joint angle features. We compute 29 kinematic dimensions corresponding to 14 joints.

<table><tr><td>Joint</td><td>Features</td><td>DoF</td><td>Joint Type</td></tr><tr><td>Pelvis (orientation)</td><td>tilt, list, rotation</td><td>3</td><td>Facing direction</td></tr><tr><td>Pelvis (translation)</td><td>tx, ty, tz</td><td>3</td><td>Global position</td></tr><tr><td>Hips (×2)</td><td>flexion, adduction, rotation</td><td>6</td><td>Ball-and-socket</td></tr><tr><td>Knees (×2)</td><td>bending</td><td>2</td><td>Hinge</td></tr><tr><td>Ankles (×2)</td><td>bending</td><td>2</td><td>Hinge</td></tr><tr><td>Lumbar</td><td>extension, bending, rotation</td><td>3</td><td>Spinal (3-DoF)</td></tr><tr><td>Shoulders (×2)</td><td>flexion, adduction, rotation</td><td>6</td><td>Ball-and-socket</td></tr><tr><td>Elbows (×2)</td><td>flexion</td><td>2</td><td>Hinge</td></tr><tr><td>Neck</td><td>flexion, adduction</td><td>2</td><td>Spinal (2-DoF)</td></tr><tr><td>Total</td><td></td><td>29</td><td></td></tr></table>

Motion Image construction. Each motion is split into joint-specific features $\{ p _ { t , k } \} _ { k = 1 } ^ { K }$ . Since joints have varying DoF (1–3 dimensions), we use learnable linear projections $\{ \phi _ { k } \} _ { k = 1 } ^ { K }$ to map them into a unified space:

$$
h _ {t, k} = \phi_ {k} (p _ {t, k}) \in \mathbb {R} ^ {d _ {p a r t}}\tag{2}
$$

where $d _ { p a r t } { = } 1 6$ matches the ViT [5] patch size so that each joint occupies exactly one 16-pixel horizontal band, yielding a one-to-one joint-to-patch mapping (adjustable to other ViT configurations, $e . g . , d _ { p a r t } { = } 1 4 )$ . At each frame t, the projected features of all K joints are concatenated as one column $H _ { t } \in \mathbb { R } ^ { D _ { e m } }$ b $( D _ { e m b } = K \times d _ { p a r t } = 2 2 4 )$ . Stacking all frames along the temporal axis and padding to T=224 produces a pseudo-image $I _ { m o t i o n } \in \bar { \mathbb { R } } ^ { 2 2 4 \times 2 2 4 }$ termed the “Motion Image”. As shown in Fig. 1, each horizontal band corresponds to a specific joint, naturally supporting part-level alignment with pre-trained ViTs.

Fig. 2 illustrates the translation-invariant features using the motion “a person walks slowly forward” as an example. The right hip’s joint angles (Fig. 2(b)) exhibit clear periodic gait patterns, while the corresponding joint positions (Fig. 2(c)) are dominated by global trajectory drift. At the whole-body level, the joint angle Motion Image (Fig. 2(d)) shows temporally localized, joint-specific activations, whereas the position-based image (Fig. 2(e)) shows uniform drift across all bands, obscuring kinematic diferences.

![](images/438c1d75583459ca3075a81bbfc63beff65d3f677f03b2c25d1e29b94671a837.jpg)

![](images/f342570fef59c0ad048ef5df297eadf7a4725f0f0f429d833a2b747770d98ed2.jpg)

![](images/7e6f82ebe47c6656dfa37b27148c8659369e24c504afcbd3df44eff1a632a344.jpg)

![](images/f499d775c3a86b9cfb35e2c67af587469150f7712a3b0bf2981eb023d345903e.jpg)  
Fig. 2: Joint-angle vs. joint-position-based representations for “a person walks slowly forward”. (a) Skeletal structure and body-centric axes. (b) Right hip angles. (c) Right knee positions. (d) Our 29-dimension joint angle Motion Image: each band encodes a distinct joint. (e) MoPatch [34] position image.

## 3.2 Dual-Stream Architecture

The framework consists of a dual-stream architecture comprising a Motion Encoder and a Text Encoder, designed to extract dense feature representations for the subsequent late interaction.

Motion Encoder. The motion encoder ${ \mathcal { E } } _ { m }$ uses a ViT backbone. The singlechannel Motion Image $I _ { m o t i o n }$ is repeated across RGB channels to match the ImageNet-pre-trained input format, yielding patch embeddings:

$$
V = \mathcal {E} _ {m} (I _ {m o t i o n}) = \{v _ {1}, v _ {2}, \dots , v _ {N} \} \in \mathbb {R} ^ {N \times d}\tag{3}
$$

where N is the number of patches $( e . g .$ ., 196 for a 14 × 14 grid) and d is the feature dimension. Unlike CLIP-style models, we retain the full patch sequence V rather than pooling into a global [CLS] token.

Text Encoder. The text encoder $\mathcal { E } _ { t }$ (a Transformer-based language model, e.g., DistilBERT) maps a description $\tau$ to token-level hidden states:

$$
L = \mathcal {E} _ {t} (\mathcal {T}) = \{l _ {1}, l _ {2}, \dots , l _ {M} \} \in \mathbb {R} ^ {M \times d}\tag{4}
$$

where M is the sequence length. We use the content-token embeddings {l<sub>i</sub>} (excluding [CLS] and [SEP]) for fine-grained interaction.

## 3.3 Fine-Grained Late Interaction (MaxSim)

While standard global pooling aggregates information prematurely, MaxSim explicitly models the alignment between distinct text tokens and motion patches. We define the token-patch interaction matrix $S \in \mathbb { R } ^ { M \times N }$ as the dot product between normalized text and motion features:

$$
S _ {i j} = \frac {l _ {i} \cdot v _ {j} ^ {T}}{\| l _ {i} \| \| v _ {j} \|}\tag{5}
$$

We compute MaxSim in the T2M direction: for each text token $l _ { i } \ \left( e . g . , \right.$ representing “hand”), we identify the motion patch $v _ { j }$ that yields the maximum activation response. The similarity score for a text-motion pair is then obtained by averaging these per-token maxima:

$$
\operatorname{Sim} (\mathcal {T}, \mathcal {M}) = \frac {1}{M} \sum_ {i = 1} ^ {M} \max _ {j = 1} ^ {N} (S _ {i j})\tag{6}
$$

This mechanism allows the model to dynamically ground each word to its most relevant kinematic feature, filtering out irrelevant motion background. We compute MaxSim only in the T2M direction because motion sequences contain far more patches $\left( N { = } 1 9 6 \right)$ than text tokens $( M \ll N )$ , and many patches encode static or repetitive postures with no textual counterpart; a reverse max-over-text would introduce substantial noise.

## 3.4 Context-Aware Regularization via MLM

To ensure that token embeddings carry suficient contextual information for reliable MaxSim matching, we introduce MLM [4,16] as a context-aware regularization task. During training, we randomly mask a proportion of the input tokens $( e . g . , 1 5 \% )$ to generate a corrupted text $\mathcal { T } _ { m a s k e d }$ . The Text Encoder is then required to reconstruct the original tokens based on the contextual dependency of the visible tokens. This auxiliary task forces the encoder to deeply encode the syntactic and semantic relationships within the sentence:

$$
\mathcal {L} _ {m l m} = - \sum_ {i \in \mathrm{mask}} \log P (w _ {i} | \mathcal {T} _ {m a s k e d}; \theta_ {t e x t})\tag{7}
$$

By optimizing $\mathcal { L } _ { m l m }$ , we ensure that the output embeddings used for MaxSim are not merely isolated word representations but are enriched with global context. Note that we apply MLM exclusively to the text encoder rather than to both modalities. This design is motivated by the same asymmetry underlying our unidirectional MaxSim: since it is the text tokens that query the motion patches, the contextual quality of each token embedding is the primary bottleneck for alignment accuracy.

## 3.5 Training Strategy and Loss Function

For a batch of B text-motion pairs, we compute the pairwise similarity matrix $\mathbf { S } _ { \mathrm { b a t c h } } \in \mathbb { R } ^ { B \times B }$ using the T2M MaxSim Score $\left( \mathrm { E q . ~ 6 } \right)$ . The T2M retrieval loss applies in-batch cross-entropy with the diagonal as the target:

$$
\mathcal {L} _ {t 2 m} = - \frac {1}{B} \sum_ {i = 1} ^ {B} \log \frac {\exp (\mathbf {S} _ {\mathrm{batch}} [ i , i ] / \tau)}{\sum_ {j = 1} ^ {B} \exp (\mathbf {S} _ {\mathrm{batch}} [ i , j ] / \tau)}\tag{8}
$$

where $\tau$ is a learnable temperature. The M2T loss $\mathcal { L } _ { m 2 t }$ applies the same formulation on the transpose $\mathbf { S } _ { \mathrm { b a t c h } } ^ { \top }$ . The total objective combines the symmetric retrieval loss with the MLM regularization:

$$
\mathcal {L} _ {r e t} = \frac {1}{2} (\mathcal {L} _ {t 2 m} + \mathcal {L} _ {m 2 t}), \qquad \mathcal {L} _ {t o t a l} = \mathcal {L} _ {r e t} + \lambda_ {m l m} \mathcal {L} _ {m l m}\tag{9}
$$

where $\lambda _ { m l m }$ balances retrieval and regularization. During training, clean text is used for $\mathcal { L } _ { \mathit { r e t } }$ and masked text for $\mathcal { L } _ { m l m } ;$ both objectives share the text encoder parameters.

## 4 Experiments

## 4.1 Datasets and Implementation Details

We evaluate our proposed method on two datasets. HumanML3D [9] aggregates motion skeleton data from AMASS [19] and HumanAct12 [11], comprising 14,616 motion clips and 44,970 text descriptions covering diverse daily activities. We follow the standard split [9, 22]: 23,384/1,460/4,380 for training/validation/testing at 20 FPS [7, 22, 34, 35]. KIT-ML [23] contains 3,911 motions with 6,278 descriptions, split into 4,888/300/830 for training/ validation/ testing at 12.5 FPS [7, 22, 34, 35].

The system performs T2M by ranking gallery motions given a text query, and M2T by ranking texts given a motion. We report Recall@K (R@K) and Median Rank (MedR) for both T2M and M2T retrieval; higher R@K and lower MedR are better. R@1/R@2 sufer high cross-seed variance due to near-duplicate/similarmotion interference (see Supplementary Material 3); we therefore report the more stable R@5/R@10/MedR in Table 2 with the same seed 42 to ensure fair comparison with previous works. Our base model, which we name MotionColBert, uses ViT-Base and DistilBERT following prior work. To assess scalability, we also report a larger variant, MotionColBert-L, with ViT-Large [5] and RoBERTa-Large [17]. For fairness, we compare against MoPatch-L and SGAR-L, which scale MoPatch [34] and SGAR [37] to the same ViT-Large and RoBERTa-Large backbones.

Both encoders project their outputs into a shared latent space with a dimension of d = 256. We train for 60 epochs on a single NVIDIA H200 GPU with batch size 128, using AdamW [18] (weight decay 0.1) with learning rates of $1 \times 1 0 ^ { - 4 }$ and $1 \times 1 0 ^ { - 5 }$ for HumanML3D and KIT-ML, respectively. We mask 15% of tokens for the MLM loss with $\lambda _ { \mathrm { m l m } } { = } 0 . 2$

Table 2: Comparison with state-of-the-art methods on HumanML3D and KIT-ML datasets. MotionColBert-L uses ViT-Large and RoBERTa-Large backbones. †: methods used extra LLM augmentation as input.

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Methods</td><td rowspan="2">Venue</td><td colspan="4">T2M Retrieval</td><td colspan="4">M2T Retrieval</td></tr><tr><td>R@3↑</td><td>R@5↑</td><td>R@10↑</td><td>MedR↓</td><td>R@3↑</td><td>R@5↑</td><td>R@10↑</td><td>MedR↓</td></tr><tr><td rowspan="12">HML3D</td><td>TMR [22]</td><td>ICCV&#x27;23</td><td>16.33</td><td>22.06</td><td>33.37</td><td>25.00</td><td>16.90</td><td>22.92</td><td>32.21</td><td>26.00</td></tr><tr><td>CAR [7]</td><td>ECCV&#x27;24</td><td>18.96</td><td>26.42</td><td>37.28</td><td>19.50</td><td>20.19</td><td>26.65</td><td>36.37</td><td>20.50</td></tr><tr><td> $KinMo^†$  [39]</td><td>ICCV&#x27;25</td><td>20.47</td><td>28.62</td><td>41.60</td><td>16.00</td><td>21.42</td><td>29.50</td><td>41.43</td><td>16.00</td></tr><tr><td>PL-TMR $^†$  [35]</td><td>AAAI&#x27;25</td><td>22.18</td><td>29.48</td><td>43.43</td><td>14.00</td><td>21.45</td><td>28.34</td><td>39.11</td><td>19.00</td></tr><tr><td>Lyu et al. [31]</td><td>ICLR&#x27;25</td><td>23.25</td><td>30.81</td><td>43.36</td><td>14.00</td><td>22.17</td><td>29.25</td><td>40.34</td><td>17.00</td></tr><tr><td>MoPatch [34]</td><td>CVPR&#x27;24</td><td>21.94</td><td>29.15</td><td>40.78</td><td>17.00</td><td>21.94</td><td>29.08</td><td>38.73</td><td>18.00</td></tr><tr><td>MoPatch-L</td><td>-</td><td>22.88</td><td>30.36</td><td>42.36</td><td>15.00</td><td>22.51</td><td>29.36</td><td>40.10</td><td>17.00</td></tr><tr><td> $SGAR^†$  [37]</td><td>NeurIPS&#x27;25</td><td>23.52</td><td>30.75</td><td>43.00</td><td>15.00</td><td>23.38</td><td>30.09</td><td>41.83</td><td>16.00</td></tr><tr><td> $SGAR-L^†$ </td><td>-</td><td>21.59</td><td>28.50</td><td>41.37</td><td>16.00</td><td>20.84</td><td>27.38</td><td>39.37</td><td>17.00</td></tr><tr><td>Multi-Modal  $MR^†$  [36]</td><td>TMM&#x27;26</td><td>19.19</td><td>27.10</td><td>40.69</td><td>16.00</td><td>20.78</td><td>27.64</td><td>41.61</td><td>15.00</td></tr><tr><td>MotionColBert</td><td>Proposed</td><td>23.53</td><td>31.71</td><td>43.80</td><td>14.00</td><td>22.82</td><td>30.45</td><td>41.45</td><td>16.00</td></tr><tr><td>MotionColBert-L</td><td>Proposed</td><td>26.22</td><td>34.89</td><td>48.08</td><td>11.00</td><td>25.47</td><td>33.22</td><td>44.74</td><td>13.00</td></tr><tr><td rowspan="9">KIT-ML</td><td>TMR [22]</td><td>ICCV&#x27;23</td><td>20.74</td><td>30.03</td><td>44.66</td><td>14.00</td><td>22.14</td><td>29.39</td><td>38.55</td><td>16.00</td></tr><tr><td>CAR [7]</td><td>ECCV&#x27;24</td><td>25.78</td><td>37.22</td><td>53.01</td><td>11.00</td><td>24.93</td><td>32.53</td><td>46.74</td><td>14.00</td></tr><tr><td>Lyu et al. [31]</td><td>ICLR&#x27;25</td><td>31.61</td><td>36.81</td><td>54.12</td><td>8.00</td><td>30.06</td><td>35.63</td><td>47.53</td><td>10.50</td></tr><tr><td>MoPatch [34]</td><td>CVPR&#x27;24</td><td>28.60</td><td>34.33</td><td>50.78</td><td>10.00</td><td>26.14</td><td>34.70</td><td>44.23</td><td>13.00</td></tr><tr><td>MoPatch-L</td><td>-</td><td>28.08</td><td>33.89</td><td>50.31</td><td>10.00</td><td>25.92</td><td>34.07</td><td>44.60</td><td>13.00</td></tr><tr><td> $SGAR^†$  [37]</td><td>NeurIPS&#x27;25</td><td>30.48</td><td>40.00</td><td>53.61</td><td>9.00</td><td>25.54</td><td>35.54</td><td>48.31</td><td>11.00</td></tr><tr><td> $SGAR-L^†$ </td><td>-</td><td>30.36</td><td>39.88</td><td>54.70</td><td>8.00</td><td>25.90</td><td>36.02</td><td>49.76</td><td>11.00</td></tr><tr><td>MotionColBert</td><td>Proposed</td><td>32.29</td><td>43.37</td><td>59.28</td><td>7.00</td><td>25.78</td><td>35.06</td><td>49.64</td><td>11.00</td></tr><tr><td>MotionColBert-L</td><td>Proposed</td><td>30.60</td><td>42.65</td><td>56.39</td><td>8.00</td><td>28.80</td><td>38.80</td><td>51.33</td><td>9.00</td></tr></table>

## 4.2 Performance

Comparison with State-of-the-Art. Table 2 presents a comprehensive comparison with recent methods on HumanML3D and KIT-ML. On both benchmarks, our base model matches or exceeds all methods that do not rely on LLM-based text augmentation, and is competitive with those that do. Notably, on KIT-ML, our base model achieves the best R@10 (59.28%) and MedR (7.00) in T2M retrieval, outperforming the strongest competing method SGAR-L [37] by 4.58 points in R@10, suggesting that fine-grained matching is particularly advantageous when subtle inter-candidate diferences must be resolved.

Scaling to Larger Backbones. A key advantage of our framework is its effective scalability. On HumanML3D, MotionColBert-L improves over the base model by +4.28% in T2M R@10 and reduces MedR from 14 to 11, whereas MoPatch-L and SGAR-L show only marginal or negative gains over their base variants. A similar trend holds on KIT-ML (detailed scaling analysis in Supplementary Material 2.1). These results suggest that global-matching methods sufer from diminishing returns when scaling, as representations are compressed into a single vector. Our structurally decoupled motion representation provides semantically coherent patch features, and the token-level late interaction directly leverages them, allowing additional model capacity to translate into more discriminative matching.

![](images/fb13f84169e8cd8b1acc13670b70282d373145da17662b1fad06b3f642438f24.jpg)  
Fig. 3: Joint-angle vs. position-based representations and their token-to-patch maps. Top: our joint-angle Motion Image (each band is one joint) and its MaxSim interaction score. Bottom: the position-based image (grouped by body region) and its attention map. The joint-angle interaction is sharply localized to the relevant joints and temporal phases, whereas the position-based attention is difuse.

## 4.3 Joint-Angle vs. Position Score Maps

To make the benefit of our representation concrete, Figure 3 contrasts the jointangle Motion Image (ours) with the position-based image (MoPatch [34]) for the same query, together with the token-to-patch interaction maps. In the joint-angle image every horizontal band encodes a single joint independently, so distinct kinematic events remain separable; in the position image all body-region bands drift together with the global trajectory (the left-to-right colour shift), blurring per-joint detail. By aggregating the per-token, per-patch MaxSim scores into an interaction score map, our score map grounds each phase of motion on the correct joints: it attends to the lumbar and leg bands (L/R knee and hip) during the stepping phase, then shifts to the right shoulder and the global-position row when the hand reaches the rail (reflecting the translational displacement). The position-based attention map is difuse and concentrated on the arm region, with no clear temporal or part-level correspondence. The joint-angle representation paired with MaxSim thus achieving fine-grained alignment and provides an interpretability that position-based methods lack. Additional score-map and retrieval comparisons are given in Supplementary Material 5.

Table 3: Ablation study of feature representations, retrieval strategies, and Masked Language Modeling (MLM) on HumanML3D.

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Repr.</td><td rowspan="2">Strategy</td><td rowspan="2">MLM</td><td colspan="4">T2M Retrieval</td><td colspan="4">M2T Retrieval</td></tr><tr><td>R@3↑</td><td>R@5↑</td><td>R@10↑</td><td>MedR↓</td><td>R@3↑</td><td>R@5↑</td><td>R@10↑</td><td>MedR↓</td></tr><tr><td rowspan="8">HML3D</td><td>Pos</td><td>Global</td><td>✘</td><td>22.88</td><td>30.36</td><td>42.36</td><td>15.00</td><td>22.51</td><td>29.36</td><td>40.10</td><td>17.00</td></tr><tr><td>Pos</td><td>Global</td><td>✓</td><td>20.72</td><td>28.89</td><td>40.02</td><td>17.00</td><td>20.45</td><td>28.20</td><td>36.29</td><td>19.50</td></tr><tr><td>Pos</td><td>MaxSim</td><td>✘</td><td>23.51</td><td>30.54</td><td>43.17</td><td>15.00</td><td>22.19</td><td>29.49</td><td>40.64</td><td>17.00</td></tr><tr><td>Pos</td><td>MaxSim</td><td>✓</td><td>23.88</td><td>31.00</td><td>44.18</td><td>14.00</td><td>23.20</td><td>30.13</td><td>41.76</td><td>16.00</td></tr><tr><td>Angle</td><td>Global</td><td>✘</td><td>23.63</td><td>31.11</td><td>44.72</td><td>14.00</td><td>23.67</td><td>30.21</td><td>40.92</td><td>17.00</td></tr><tr><td>Angle</td><td>Global</td><td>✓</td><td>22.57</td><td>29.60</td><td>41.08</td><td>16.00</td><td>20.90</td><td>28.08</td><td>37.39</td><td>19.00</td></tr><tr><td>Angle</td><td>MaxSim</td><td>✘</td><td>25.03</td><td>32.39</td><td>45.64</td><td>12.00</td><td>23.64</td><td>31.67</td><td>42.30</td><td>15.00</td></tr><tr><td>Angle</td><td>MaxSim</td><td>✓</td><td>26.22</td><td>34.89</td><td>48.08</td><td>11.00</td><td>25.47</td><td>33.22</td><td>44.74</td><td>13.00</td></tr></table>

## 4.4 Ablation Study

Table 3 reports a systematic ablation starting from joint positions (Pos) with global matching (Global), incrementally adding: (1) joint angle representation (Angle), (2) MaxSim late interaction, and (3) MLM regularization. We also provide (4) an eficiency analysis. All experiments use the ViT-Large and RoBERTa-Large backbones (our MotionColBert-L configuration). Additional ablations regarding encoder scaling and hyperparameter sensitivity are in Supplementary Material 2.

Efect of joint angle representation. Switching from joint positions to joint angles consistently improves higher-recall metrics under both matching strategies. The improvement is more pronounced under MaxSim (e.g., on HumanML3D T2M R@5 improves by +1.85% under MaxSim vs. +0.75% under global matching), confirming that decoupling local joint movement from global trajectory produces a cleaner signal for retrieval. This benefit is amplified when combined with fine-grained matching, since MaxSim aligns individual tokens to motion patches, the structured, per-joint nature of the angle representation provides more semantically coherent patch features.

Efect of MaxSim late interaction. Replacing global matching with token-level MaxSim improves retrieval under both representations (e.g., T2M R@10 rises from 42.36 to 43.17 for joint positions and from 44.72 to 45.64 for joint angles). This synergy is strongest once MaxSim is further paired with MLM (Angle+MaxSim+MLM reaches 48.08 T2M R@10), validating our design of colocating fine-grained matching with a structurally decoupled representation.

Efect of MLM regularization. MLM helps only when combined with token-level MaxSim, and is detrimental under global pooling. With MaxSim, adding MLM improves every metric: on HumanML3D it raises Angle+MaxSim T2M R@10 from 45.64 to 48.08 (+2.44) and M2T R@10 from 42.30 to 44.74 (+2.44), with similar gains for joint positions. In contrast, under global matching the same objective hurts performance (Angle+Global T2M R@10 drops from 44.72 to 41.08, and Pos+Global from 42.36 to 40.02). This asymmetry reinforces the coupled design of our framework: MLM enriches each token embedding with sentencelevel context, but this context can only be exploited by a token-level matching operator. A single pooled embedding discards the per-token structure that MLM regularizes, so the auxiliary objective merely competes with the contrastive signal rather than aiding it. Further sensitivity analysis on the MLM loss weight $\left( \lambda _ { m l m } \right)$ is provided in Supplementary Material 2.2.

Table 4: Eficiency and compression trade-of on HumanML3D. PQ: Product Quantization; Binary: asymmetric binary hashing. Query latency measured on a single NVIDIA H200 GPU.

<table><tr><td>Method</td><td>Storage</td><td>Comp.</td><td>T2M R@10↑</td><td>M2T R@10↑</td><td>Latency (ms)</td></tr><tr><td>MoPatch (global)</td><td>4.28 MB</td><td>-</td><td>40.78</td><td>38.73</td><td>3.14</td></tr><tr><td>MotionColBert (Float32)</td><td>837.21 MB</td><td>1×</td><td>43.80</td><td>41.45</td><td>4.10</td></tr><tr><td>MotionColBert + PQ (m=64, 8bit)</td><td>52.32 MB</td><td>16×</td><td>43.39 (-0.41)</td><td>41.06 (-0.39)</td><td>-</td></tr><tr><td>MotionColBert + Binary</td><td>26.14 MB</td><td>32×</td><td>43.10 (-0.70)</td><td>39.87 (-1.58)</td><td>-</td></tr></table>

Eficiency analysis. Table 4 reports the eficiency and compression trade-ofs on HumanML3D. Following standard retrieval practice [6, 14], all motion embeddings are pre-computed ofline. Since MaxSim retains N=196 patch embeddings per motion, gallery storage increases to ∼837 MB vs. ∼4 MB for MoPatch’s global vectors. However, query latency increases only marginally (4.10 ms vs. 3.14 ms; on KIT-ML: 3.19 ms vs. 2.79 ms) due to GPU-parallel similarity computation. Post-hoc compression efectively mitigates storage: Product Quantization (PQ) [12] with m=64 achieves 16× compression with ≤0.41% R@10 loss, and binary hashing [8] at 32× retains 43.10% T2M R@10.

## 5 Conclusion

We presented a framework for fine-grained text-motion retrieval that overcomes the information bottleneck of global-embedding methods by combining an anatom ically defined joint-angle representation with a MaxSim late-interaction mechanism and MLM regularization. Extensive experiments demonstrate state-ofthe-art retrieval performance and interpretable token-to-patch correspondence maps. In practice, this transparency empowers animators to search vast motion databases while verifying specific part-level matches, and provides a robust foundation for downstream tasks like language-driven motion generation and localized editing. While our approach unlocks superior accuracy and interpretability, retaining dense patch embeddings introduces a trade-of in increased storage and computational overhead for ofline galleries. Future work will explore eficient indexing strategies, such as approximate nearest-neighbor search and vector quantization, to scale this interpretable retrieval framework to massive, industry-level motion repositories.

## References

1. Bao, H., Wang, W., Dong, L., Liu, Q., Mohammed, O.K., Aggarwal, K., Som, S., Piao, S., Wei, F.: Vlmo: Unified vision-language pre-training with mixture-ofmodality-experts. Advances in neural information processing systems 35, 32897– 32912 (2022) 5

2. Chen, H., Lyu, G., Xu, C., Yan, J., Yang, X., Deng, C.: Beyond global alignment: Fine-grained motion-language retrieval via pyramidal shapley-taylor learning. arXiv preprint arXiv:2601.21904 (2026) 4

3. Collorone, L., Gioia, M., Pappa, M., Leoni, P., Ficarra, G., Litany, O., Spinelli, I., Galasso, F.: Monster: a unified model for motion, scene, text retrieval. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 10940– 10949 (2025) 4

4. Devlin, J., Chang, M.W., Lee, K., Toutanova, K.: Bert: Pre-training of deep bidirectional transformers for language understanding. In: Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers). pp. 4171–4186 (2019) 2, 4, 9

5. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020) 7, 10

6. Faysse, M., Sibille, H., Wu, T., Omrani, B., Viaud, G., Hudelot, C., Colombo, P.: Colpali: Eficient document retrieval with vision language models. In: International Conference on Learning Representations. vol. 2025, pp. 61424–61449 (2025) 2, 4, 14

7. Fujiwara, K., Tanaka, M., Yu, Q.: Chronologically accurate retrieval for temporal grounding of motion-language models. In: European Conference on Computer Vision. pp. 323–339. Springer (2024) 4, 10, 11

8. Gong, Y., Lazebnik, S., Gordo, A., Perronnin, F.: Iterative quantization: A procrustean approach to learning binary codes for large-scale image retrieval. IEEE transactions on pattern analysis and machine intelligence 35(12), 2916–2929 (2012) 14

9. Guo, C., Zou, S., Zuo, X., Wang, S., Ji, W., Li, X., Cheng, L.: Generating diverse and natural 3d human motions from text. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 5152–5161 (2022) 1, 3, 4, 5, 10

10. Guo, C., Zuo, X., Wang, S., Cheng, L.: Tm2t: Stochastic and tokenized modeling for the reciprocal generation of 3d human motions and texts. In: European Conference on Computer Vision. pp. 580–597. Springer (2022) 1

11. Guo, C., Zuo, X., Wang, S., Zou, S., Sun, Q., Deng, A., Gong, M., Cheng, L.: Action2motion: Conditioned generation of 3d human motions. In: Proceedings of the 28th ACM international conference on multimedia. pp. 2021–2029 (2020) 10

12. Jegou, H., Douze, M., Schmid, C.: Product quantization for nearest neighbor search. IEEE transactions on pattern analysis and machine intelligence 33(1), 117– 128 (2010) 14

13. Jiang, B., Chen, X., Liu, W., Yu, J., Yu, G., Chen, T.: Motiongpt: Human motion as a foreign language. Advances in Neural Information Processing Systems 36, 20067–20079 (2023) 1

14. Khattab, O., Zaharia, M.: Colbert: Eficient and efective passage search via contextualized late interaction over bert. In: Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval. pp. 39–48 (2020) 2, 4, 5, 14

15. Kinfu, K., Vidal, R.: Motionbind: Multi-modal human motion alignment for retrieval, recognition, and generation. Advances in Neural Information Processing Systems 38, 122518–122548 (2026) 4

16. Li, J., Selvaraju, R., Gotmare, A., Joty, S., Xiong, C., Hoi, S.C.H.: Align before fuse: Vision and language representation learning with momentum distillation. Advances in neural information processing systems 34, 9694–9705 (2021) 5, 9

17. Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., Stoyanov, V.: Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692 (2019) 10

18. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017) 10

19. Mahmood, N., Ghorbani, N., Troje, N.F., Pons-Moll, G., Black, M.J.: Amass: Archive of motion capture as surface shapes. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 5442–5451 (2019) 4, 10

20. Meng, Z., Han, Z., Peng, X., Xie, Y., Jiang, H.: Absolute coordinates make motion generation easy. arXiv preprint arXiv:2505.19377 (2025) 3

21. Meng, Z., Xie, Y., Peng, X., Han, Z., Jiang, H.: Rethinking difusion for text-driven human motion generation: Redundant representations, evaluation, and masked autoregression. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 27859–27871 (2025) 3

22. Petrovich, M., Black, M.J., Varol, G.: Tmr: Text-to-motion retrieval using contrastive 3d human motion synthesis. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 9488–9497 (2023) 1, 2, 4, 6, 10, 11

23. Plappert, M., Mandery, C., Asfour, T.: The kit motion-language dataset. Big data 4(4), 236–252 (2016) 1, 4, 10

24. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PmLR (2021) 4

25. Ren, J., Zhang, G., Fu, H., Wu, P., Wang, H.: Wamo: Wavelet-enhanced multifrequency trajectory analysis for fine-grained text-motion retrieval. arXiv preprint arXiv:2508.03343 (2025) 4

26. Schlegel, K., Jiang, L., Ni, H.: Using joint angles based on the international biomechanical standards for human action recognition and related tasks. arXiv preprint arXiv:2406.17443 (2024) 3

27. Shi, H., Zhang, H.: Sequence-event semantic consistent learning for text-to-motion retrieval. In: Proceedings of the 33rd ACM International Conference on Multimedia. pp. 8740–8749 (2025) 2, 4

28. Tevet, G., Raab, S., Gordon, B., Shafir, Y., Cohen-Or, D., Bermano, A.H.: Human motion difusion model. arXiv preprint arXiv:2209.14916 (2022) 1

29. Wu, G., Van der Helm, F.C., Veeger, H.D., Makhsous, M., Van Roy, P., Anglin, C., Nagels, J., Karduna, A.R., McQuade, K., Wang, X., et al.: Isb recommendation on definitions of joint coordinate systems of various joints for the reporting of human joint motion—part ii: shoulder, elbow, wrist and hand. Journal of biomechanics 38(5), 981–992 (2005) 2, 3, 6

30. Wu, G., Siegler, S., Allard, P., Kirtley, C., Leardini, A., Rosenbaum, D., Whittle, M., D D’Lima, D., Cristofolini, L., Witte, H., et al.: Isb recommendation on definitions of joint coordinate system of various joints for the reporting of human joint motion—part i: ankle, hip, and spine. Journal of biomechanics 35(4), 543–548 (2002) 2, 3, 6

31. Xu, C., Yan, J., Yang, M., Deng, C., et al.: Towards unified human motion-language understanding via sparse interpretable characterization. In: International Conference on Learning Representations. vol. 2025, pp. 15529–15553 (2025) 2, 4, 6, 11

32. Xu, H., Ghosh, G., Huang, P.Y., Okhonko, D., Aghajanyan, A., Metze, F., Zettlemoyer, L., Feichtenhofer, C.: Videoclip: Contrastive pre-training for zero-shot video-text understanding. In: Proceedings of the 2021 conference on empirical methods in natural language processing. pp. 6787–6800 (2021) 4

33. Yin, K., Zou, S., Ge, Y., Tian, Z.: Tri-modal motion retrieval by learning a joint embedding space. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 1596–1605 (2024) 4

34. Yu, Q., Tanaka, M., Fujiwara, K.: Exploring vision transformers for 3d human motion-language models with motion patches. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 937–946 (2024) 1, 3, 6, 8, 10, 11, 12

35. Yu, Q., Tanaka, M., Fujiwara, K.: Remogpt: Part-level retrieval-augmented motionlanguage models. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 39, pp. 9635–9643 (2025) 4, 10, 11

36. Yu, S., Wang, Z.A., Yin, K., Tian, Z., Zhang, M., Si, W., Zou, S.: Multi-modal motion retrieval by learning a fine-grained joint embedding space. IEEE Transactions on Multimedia (2026) 4, 11

37. Zhang, J., Lin, L., Yang, S., Liu, J.: Sgar: Structural generative augmentation for 3d human motion retrieval. Advances in Neural Information Processing Systems 38, 106931–106959 (2026) 1, 2, 4, 10, 11

38. Zhang, M., Cai, Z., Pan, L., Hong, F., Guo, X., Yang, L., Liu, Z.: Motiondifuse: Text-driven human motion generation with difusion model. IEEE transactions on pattern analysis and machine intelligence 46(6), 4115–4128 (2024) 1

39. Zhang, P., Liu, P., Garrido, P., Kim, H., Chaudhuri, B.: Kinmo: Kinematic-aware human motion understanding and generation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 11187–11197 (2025) 2, 4, 11