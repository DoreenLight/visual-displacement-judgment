<!-- source: https://doi.org/10.1109/tmm.2023.3321438 -->
<!-- title: Global and Local Spatio-Temporal Encoder for 3D Human Pose Estimation -->
<!-- fetched: 2026-10-01T03:49:39Z -->
<!-- via: local_paper_fulltext(pdftotext -layout) -->
<!-- note: 用户于 2026-10-01 从学校数据库下载的原文 PDF：papers/Global_and_Local_Spatio-Temporal_Encoder_for_3D_Human_Pose_Estimation.pdf -->

Abstract—Transformers have been used for 3D human                              [2] and person re-identification [3], [4], [5], this is a longstand-
pose estimation with excellent performance; however, most                         ing and challenging task in computer vision. Its applications
transformers focus on encoding the global spatio-temporal                         include human computer interaction [6], action recognition [7],
correlation of all joints in the human body and there are few
studies on the local Spatio-temporal correlation of each joint in                 [8], [9], augmented reality (AR), virtual reality (VR) [10], med-
the human body. In this article, we propose a Global and Local                    ical assistance, attracting a lot of attention in recent years. Ex-
Spatio-Temporal Encoder (GLSTE) to model the Spatio-temporal                      isting works are generally divided into 1) directly inferring a
correlation. Specifically, a Global Spatial Encoder (GSE) and a                   3D human pose from a 2D image or video frame, 2) inferring
Global Temporal Encoder (GTE) are constructed to capture the                      2D pose from monocular images or videos by any off-the-shelf
global spatial information of all joints in a single frame and the
global temporal information of all frames, respectively. A Local                  2D pose estimators, and then inferring 3D human pose from
Spatio-Temporal Encoder (LSTE) is constructed to capture the                      estimated 2D pose through 2D to 3D lifting pipeline. The supe-
spatial and temporal information of each joint in the local N                     rior performance of the state-of-the-art 2D pose detector makes
frames. Furthermore, we propose a parallel attention module                       the 2D-to-3D lifting [11], [12], [13] even better. However, since
with weight sharing to better incorporate spatial and temporal                    multiple 3D poses may correspond to almost identical 2D projec-
information into each node simultaneously. Extensive experiments
show that GLSTE outperforms state-of-the-art methods with fewer                   tions, even the slightest change in joint position and appearance
parameters and less computational overhead on two challenging                     can affect the prediction results in this case.
datasets: Human3.6 M and MPI-INF-3DHP. Especially in the                             In order to alleviate this problem, video can provide temporal
evaluation of Human3.6 M dataset, the results of our method with                  information to improve 3 d human pose estimation (3DHPE)
27 frames as input are better than the vast majority of recent SOTA               accuracy and robustness, instead of estimating 3D human poses
methods with 81 and 243 frames as input, which indicates that the
model can learn more useful information with smaller inputs.                      from monocular images. Some recent methods [12], [14], [15],
                                                                                  [16] have been proposed to improve the 3D pose prediction for
  Index Terms—3D Human pose estimation, transformer, spatio-                      the current (middle) frame by aggregating 2D pose sequence
temporal encoder, parallel attention.
                                                                                  information from multiple frames, which can obtain a more ac-
                                                                                  curate 3D representation and reduce the uncertainty from 2D
                            I. INTRODUCTION                                       to 3D. Our work falls under this category of methods. Trans-
                                                                                  former [17] is developed for exploiting long-range dependencies
        ONOCULAR 3D human pose estimation is to estimate
M       the 3D joint position of the human body through images
or videos. Similar to tasks such as 2D human pose estimation [1],
                                                                                  and achieves tremendous success in natural language process-
                                                                                  ing [18], [19] and computer vision [20], [21], [22], [23]. The
                                                                                  global correlation of long input sequences can be clearly cap-
                                                                                  tured due to the self-attention mechanism of the transformer. It
   Manuscript received 8 October 2022; revised 23 January 2023, 27 June 2023,     is a particularly suitable architecture for sequential data prob-
and 20 September 2023; accepted 20 September 2023. Date of publication 2          lems, and a natural extension to the fundamental vision task of
October 2023; date of current version 23 February 2024. This work was sup-        3DHPE.
ported in part by the National Natural Science Foundation of China under Grant
62171251, and in part by the Special Foundations for the Development of Strate-      In this article, the recent works [14], [15], [16] of Trans-
gic Emerging Industries of Shenzhen under Grants JCYJ20200109143035495,           former in 3DHPE has been investigated. PoseFormer, proposed
CJGJZD20210408092804011, and JSGG20211108092812020. The Associate                 by Zheng et al. [14], uses Transformer’s self-attention mech-
Editor coordinating the review of this manuscript and approving it for publi-
cation was Prof. Jianguo Zhang. (Yong Wang and Hongbo Kang contributed            anism to describe spatial dependencies between joints in each
equally to this work.) (Corresponding author: Hongbo Kang.)                       frame and temporal dependencies between frames in a sequence.
   Yong Wang, Hongbo Kang, and Doudou Wu are with the School of Arti-             However, due to the similarity of the human pose in the adja-
ficial Intelligence, Chongqing University of Technology, Chongqing 400054,
China (e-mail: ywang@cqut.edu.cn; khb1698@163.com; doudouwu@stu.cqut.             cent frames in the video, there is a lot of redundant information
edu.cn).                                                                          in the whole time series and it requires large computational re-
   Wenming Yang is with the Shenzhen International Graduate School, Depart-       sources. Furthermore, poseformer ignores local contexts, but the
ment of Electronic Engineering, Tsinghua University, Shenzhen 518055, China
(e-mail: yang.wenming@sz.tsinghua.edu.cn).                                        use of local information is critical in computer vision tasks. To
   Longbin Zhang is with the School of Mechanical and Aerospace Engineering,      solve the above problems, Li et al. [15] proposed to model the
Rehabilitation Research Institute of Singapore, Nanyang Technological Univer-     long-term dependence relationship (global temporal correlation)
sity, Singapore 639798 (e-mail: longbin.zhang@ntu.edu.sg).

                         1520-9210 © 2023 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.
                                       See https://www.ieee.org/publications/rights/index.html for more information.

     Authorized licensed use limited to: South-Central Minzu University. Downloaded on October 01,2026 at 03:36:18 UTC from IEEE Xplore. Restrictions apply.

                                                                                       The proposed GLSTE is a global-to-local architecture that
                                                                                    captures global and local information separately through differ-
                                                                                    ent modules. The global information in this article refers to the
                                                                                    spatio-temporal correlation of all joints of the human body in
                                                                                    all frames. According to [14], [16], we also sequentially model
                                                                                    from spatial to temporal with modules of GSE and GTE. For
                                                                                    GSE, we use a simple but effective fully connected residual
                                                                                    network to integrate all joints information in each frame to sim-
                                                                                    ulate global spatial correlation. For GTE, we use the vanilla
                                                                                    Transformer encoder [17], which takes spatial information of
                                                                                    each frame as a patch to simulate global temporal correlation.
                                                                                    According to our observation, the motion of each joint of the
                                                                                    human body is different, and the relationship of the same joint
                                                                                    in local several frames is more obvious. Therefore, the local in-
                                                                                    formation in this article refers to the spatio-temporal correlation
Fig. 1. Illustration of local spatio-temporal correlation. In the spatio-temporal   of the same joint in local N frames. In order to better capture
graph of a skeleton sequence, gray nodes are body joints fused with local spatio-
temporal information. Temporal correlation refers to the temporal information       local information, in LSTE, we propose a multi-head spatial and
of the same joint between local frames (e.g. 3 frames), and spatial correlation     temporal self-attention (MSA-ST) with parallel attention mod-
refers to the spatial information between each joint in each frame.                 ule to simulate local spatio-temporal correlation, and we design
                                                                                    a spatial and temporal attention fusion block (STAF) to fuse the
                                                                                    two attention. Since the task is to predict the 3D human pose
model the local temporal correlation through STE to gradually                       of the middle frame, we design a temporal aggregator (TA) to
reduce the sequence length. But they acquire spatial informa-                       continuously aggregate local information through multi-layer
tion by a single f c layer, which has insufficient representation                   information learning to obtain the final 3D pose.
capability. With this in mind, Shan et al. [16] proposed P-STMO                        Our contributions are summarized as follows:
to capture the global spatial information of all joints in a single                     r Our proposed novel global-to-local 3D human pose esti-
frame through an MLP block. At the same time, it adopts the                               mation architecture, named GLSTE, efficiently infers 3D
temporal downsampling strategy and pre-training technology to                             poses from 2D joint positions by modeling global and local
reduce the training difficulty of the model. Since the joints of the                      spatio-temporal correlations.
human body are connected to each other, and the motion of the                           r In our LSTE, we innovatively propose a MSA-ST module
torso affects other joints, it is necessary to combine the global                         with parallel attention with weight sharing to model local
motion information of the human body and the local motion in-                             spatio-temporal correlation, where spatial and temporal at-
formation of each joint to model the human pose. However, the                             tention are fused through the STAF module.
above methods aggregate all joints together and capture the tem-                        r Our method achieves state-of-the-art performance on two
poral context of the whole body without considering the motion                            benchmarks with fewer parameters and smaller compu-
differences of each joint, which leads to insufficient learning of                        tational budgets than alternative methods, and model can
local spatio-temporal correlation (see Fig. 1). To improve this,                          learn more useful information with smaller inputs.
we believe that modeling the global and local spatio-temporal
correlations are more suitable for the 2D to 3D lifting tasks in                                            II. RELATED WORK
video.
                                                                                    A. 3D Human Pose Estimation
   Motivated by the above, in this work, we propose a new
3D pose estimation method, called Global and Local Spatio-                             3D human pose estimation can be divided into direct esti-
Temporal Encoder (GLSTE) as shown in Fig. 2. This model pre-                        mation approaches and 2D to 3D lifting approaches. The direct
dicts the 3D joint position of the middle frame from the estimated                  estimation methods infer the 3D human pose from 2D images
2D pose sequence in the monocular video. We adopt the same                          without intermediately estimating 2D human pose representa-
two-stage training method as [16]: 1) In the pre-training stage, is                 tion, e.g. [24], [25], [26], [27]. The 2D-3D lifting approaches
to train the global part, Global Spatial Encoder (GSE) and Global                   firstly estimate the 2D human pose, and then infer 3D human
Temporal Encoder (GTE), and learn the 2D spatio-temporal cor-                       pose from the intermediately estimated 2D human pose. Due
relation and inherent global features by reconstructing the 2D                      to the rapid development of 2d human pose estimation algo-
poses of randomly mask some frames and some 2D joints, which                        rithms [1], [2], the second approach also brings better perfor-
gave the model a favorable initialization. 2) In the fine-tuning                    mance. Martinez et al. [11] proposed a fully connected residual
stage, we load the pre-trained weights into GSE and GTE, then                       network to accomplish the 2D-to-3D HPE task. In some subse-
train the whole model, and output 3D joint positions at the full                    quent works [28], [29], [30] also try to generate accurate 3D pose
sequence scale and single target frame scale. In addition, in or-                   estimates from underlying 2D keypoints.Despite this, generating
der to alleviate the problem of data redundancy and enlarging                       accurate 3D poses from a single image is still an ill-posed prob-
the temporal receptive field, the 2D pose as input in both stages                   lem. Recent works [31], [32], [33] utilized temporal information
adopts the temporal downsampling strategy (TDS) [16].                               from video to improve the accuracy and robustness of 3D HPE.

   Authorized licensed use limited to: South-Central Minzu University. Downloaded on October 01,2026 at 03:36:18 UTC from IEEE Xplore. Restrictions apply.

WANG et al.: GLOBAL AND LOCAL SPATIO-TEMPORAL ENCODER                                                                                                          4041

Fig. 2. Overview of our proposed global and local spatio-temporal encoder (GLSTE), the input is a 2D pose sequence after temporal downsampling strategy
(TDS) and the outputs are 3D joint positions at both the full sequence scale and single target frame scale. The model needs to be loaded with pre-trained weights
before fine-tuning. In our model, the global feature obtains the spatial information of all joints through global spatial encoder (GSE), and then obtains temporal
information of the entire frames through global temporal encoder (GTE). The local feature obtains the spato-temporal information of each joint in the local frames
through local spatio-temporal encoder (LSTE), and the local features are aggregated and new local features are generated by temporal aggregator (TA).

Hossain et al. [34] proposed a seq-to-seq network consisting of                   proposed to solve visual tasks using self-attention mechanism.
Long Short-Term Memory (LSTM) unit with layer-normalized.                         Vision Transformer (ViT) [21], the pioneering work has directly
Pavllo et al. [12] proposed a network based on dilated tempo-                     applied the Transformer [17] architecture to image classifica-
ral convolutions for regressing 3D poses from continuous 2D                       tion. DeiT [42] further extended ViT by employing a novel dis-
sequences. Chen et al. [35] proposed a network to predict bone                    tillation method. DETR [22] is proposed for object detection
direction and bone length to ensure the temporal consistency                      and panoramic segmentation. There are also many works based
of human anatomy in videos. Liu et al. [36] proposed an atten-                    on transformer for 3D human pose estimation [43], [44], [45].
tional mechanism to adaptively identify the important frames                      Poseforme [14] uses pure ViT model to capture the relation-
and tensor outputs of each deep neural network layer to ob-                       ship between human joints and the global dependencies across
tain better estimates. Wang et al. [33] proposed a U-shaped                       frames in the entire temporal sequence, but it ignores local infor-
graph convolution network (GCN) to capture both short-term                        mation of temporal. Strided Transformer [15] incorporates local
and long-term motion information. Hu et al. [37] further pro-                     temporal context into standard Transformer. On the basis of the
posed a spatial-temporal conditional directed graph convolution                   former, P-STMO [16] strengthens the modeling of global spatial
to leverage varying non-local dependence of different poses by                    information, and proposes a temporal downsampling strategy
representing the human skeleton as a directed graph.                              to alleviate data redundancy and a two-stage training method
   Most of these works are based on traditional neural network                    to reduce the difficulty of model training, which also achieves
structures such as LSTM, CNN and GCN. However, these meth-                        better results. However, the above methods aggregate all joints
ods still have some limitations in capturing long-term dependen-                  together and capture the temporal context of the whole body
cies. In this article, we propose to use transformer with better                  without considering the motion differences of each joint. This
global correlation to model global temporal correlation. In ad-                   results in shortcomings in learning local spatio-temporal infor-
dition, we also improve the self-attention mechanism of trans-                    mation. By combining different self-attention mechanisms we
former to better capture local spatio-temporal correlation.                       designed, our approach can simultaneously consider both the
                                                                                  global spatio-temporal relationships within the human body and
                                                                                  the local spatio-temporal relationships of each joint across mul-
B. Self-Attention and Transformer                                                 tiple local frames. Since we account for the motion of each joint,
  Lately, transformer [17] has been widely concerned in vari-                     we can more comprehensively capture information and achieve
ous fields. Thus, many methods [38], [39], [40], [41] have been                   better performance and generalization capabilities.

     Authorized licensed use limited to: South-Central Minzu University. Downloaded on October 01,2026 at 03:36:18 UTC from IEEE Xplore. Restrictions apply.

Fig. 3. Instantiation of GLSTE for 3D pose estimation. We sample 27 frames by TDS as input. The tensor dimensions are enclosed in parentheses. In the global
modules are modeled for all joints, that is, (27,34) denotes 27 frames and input dimensions of 34 (2D coordinates of 17 joints). In the local modules are modeled
for each joint separately, that is, (17,27,2) denotes 17 joints, 27 frames, and the input dimension of each joint is 2. (a) Global spatial encoder. (b) Global temporal
encoder. (c) Local spatio-temporal encoder.

                        III. PROPOSED METHOD                                         form of attention scores. The input is a query matrix Q, key
                                                                                     matrix K and value matrix V . Q, K, V ∈ Rn×dm , where n is
   The overview of the proposed GLSTE is depicted in Fig. 2.
This is a global-to-local architecture with dual-channel input and                   the number of input patches, and dm is the dimension of each
output, and we use the two stages of pre-training and fine-tuning                    patch. The scaled dot-product attention is defined as follows:
                                                                                                                                        
to optimize the model, suggested by [16]. In the first stage, the                                                                 QK T
global part (GSE and GTE) of GLSTE model is pre-trained by                                   Attention(Q, K, V ) = sof tmax √              V      (1)
                                                                                                                                     dm
solving the masked pose modeling task in a self-supervised man-
ner, and the main purpose is to recover the 2D pose sequence                           The MSA splits the query, key and value into multiple heads,
from the damaged 2D pose to learn 2D spatio-temporal cor-                            and then applies scaled dot-product attention to each head. The
relation and inherent global features. In the second stage, we                       output of MSA is concatenated and projected to obtain the final
fine-tune the entire model by loading the pre-trained weights                        output.
of the global part. In this stage, the global part is first used to                    MLP: The MLP consists of two linear transformations with a
model the global spatio-temporal information and supervised by                       ReLU activation in between, and is defined as follows:
the full sequence scale to enforce temporal smoothing. The lo-                                     M LP (x) = ReLU (xW1 + b1 )W2 + b2                              (2)
cal embedding module then guides the projection from global to
local. Finally, the output of the middle frame is obtained through                   where W1 ∈ Rdm ×df , W2 ∈ Rdf ×dm , b1 ∈ Rdf and b2 ∈ Rdm
local part (LSTE and TA), which is supervised by a single object                     are parameter matrices. df is the dimension of the intermediate
frame scale to produce more accurate estimates. In both stages,                      layer.
2D human pose coordinates after temporal downsampling by
TDS are used as input sequence P̂ ∈ RT ×N ×2 , where T is the                        B. Global Spatial Encoder
number of frames, N is the number of joints of each 2D pose, and
2 indicates joint coordinates (x,y) in 2D space. The detailed ar-                       The purpose of GSE is to fusion the spatial information of
chitecture of GLSTE is illustrated in Fig. 3. The input sequence                     all joints of the human body into a single frame, which needs
will go through GSE, GTE, LSTE and TA to obtain the final out-                       to learn enough spatial information from 2D poses and map to
put. The details of these modules are described in the following                     higher dimensions. Martinez et al. [11] proposed the use of fully
subsections.                                                                         connected residual network to model the spatial information of
                                                                                     a single pose, which is also applicable to the fusion of global
                                                                                     spatial information in this article. The global spatial information
A. Preliminary                                                                       of each frame can be obtained by independently input each 2D
   In this section, we introduce some background on Trans-                           pose in the sequence, as shown in Fig. 3(a). Meanwhile, GSE
formers [17], including a multi-head self-attention (MSA) and                        also played the role of Global embedding. The 2-dimensional
a multi-layer perceptron (MLP).                                                      coordinates of all joints in each frame of the 2D pose sequence
   MSA: The scaled dot-product attention is a single attention                       P are merged as input X̂ ∈ RT ×(N.2) , and through the linear
head of the MSA. It implicitly associates input patches in the                       layer of GSE, the spatial information of each frame is embedded

   Authorized licensed use limited to: South-Central Minzu University. Downloaded on October 01,2026 at 03:36:18 UTC from IEEE Xplore. Restrictions apply.

WANG et al.: GLOBAL AND LOCAL SPATIO-TEMPORAL ENCODER                                                                                                          4043

                                                                                   Global-to-Local Block (GLB) to deal with global feature ZgM ∈
                                                                                   RN ×dg . Given that global feature is the feature for merging all
                                                                                   nodes and local feature is feature for a single node, our GLB
                                                                                   implements global to local operation by projecting and splitting,
                                                                                   which is defined as follows:
                                                                                                           P roj&Split
                                                                                                     ZgM       −→        Ẑ1 ∈ RN ×T ×dl                       (7)
                                                                                      The spatio-temporal channels of 2D pose sequence P are
                                                                                   exchanged to obtain the input X ∈ RN ×T ×2 of the local em-
                                                                                   bedding. Then, the local embedding module maps the two-
                                                                                   dimensional features of each node to a higher dimension dl by
                                                                                   linear projection. This process also guides the projecting of GLB
                                                                                   from global to local by combining with Ẑ1 , which is defined as
                                                                                   follows:
                                                                                                       Z1 = [. . . ; xi,j E; . . . ] + Ẑ1                     (8)
                                                                                   where xi,j denotes the i-th node of the j-th frame of X and
                                                                                   i = 1, 2, . . . , N, j = 1, 2, . . . , T , E ∈ R2×dl is the parameter
Fig. 4. (a) The multi-head self-attention (MSA) in global temporal encoder         matrix of the local embedding.
(GTE). (b) The multi-head spatial and temporal self-attention (MSA-ST) in local
spatio-temporal encoder (LSTE), where the left is Attns and the right is Attnt .
The dashed box is the attention map matrix.                                        E. Local Spatio-Temporal Encoder
                                                                                      According to our observation, the motions of each joint of
                                                                                   the human body are different, the same joint relationship in the
into the high-dimensional feature Z0 ∈ RT ×dg , where dg is the                    local frames is more obvious. Therefore, we design LSTE (see
intermediate dimension of the global modules, so as to obtain                      Fig. 3(c)) to model the spatio-temporal correlation of each joint
a set of potential features as the input of GTE. The features of                   in the local frames. In LSTE, we innovatively propose a MSA-ST
each frame are treated as an individual token.                                     with parallel attention module to simulate local spatio-temporal
                                                                                   correlation, and we design a spatial and temporal attention fusion
C. Global Temporal Encoder                                                         block (STAF) to fuse the two attention. In this section, dl is the
                                                                                   intermediate dimension of the local modules (LSTE and TA).
   We have simulated the global spatial information of each
                                                                                      Multi-head Spatio-Temporal Self-Attention (MSA-ST): In
frame by GSE, but the long-term temporal information of the
                                                                                   MSA-ST, the main is the weight-sharing cascade attention mod-
entire sequence is still missing. Thus, we design a global tempo-
                                                                                   ule, as shown in Fig. 4(b), and it contains spatial attention
ral encoder, as shown in Fig. 3(b). The global temporal encoder
                                                                                   (Attns ) and temporal attention (Attnt ). The three mapped learn-
is a transformer encoder [17] and composed of M layers. The
                                                                                   able matrices Q, K, V ∈ RN ×T ×dl are shared by the two at-
trainable temporal position embedding Et0 ∈ RT ×dg is used be-
                                                                                   tentions, but need to be changed accordingly, Qs , Ks , Vs are
fore the first layer of the encoder, which can be formulate as
                                                                                   obtained by swapping spatial and temporal dimensions, and
follows:
                                                                                   Qt , Kt , Vt are obtained by dividing the complete sequence T
                              Zg0 = Z0 + Et0                                (3)    into S sequences with local frames number T̄ , as follows:
                                                                                                           Reshape
 The global temporal encoder is applied to the input Zg0 ∈                                  {Q, K, V }       −→      {Qs , Ks , Vs } ∈ RT ×N ×dl               (9)
RT ×dg to obtain the output ZgM ∈ RT ×dg as follows:                                                       Split
                                                                                            {Q, K, V } −→ {Qt , Kt , Vt } ∈ RN ×S×T̄ ×dl                   (10)
                Ẑgm = M SA(LN (Zgm−1 )) + Zgm−1                            (4)
                                                                                      The spatial attention Attns is used to simulate the spatial re-
                      Zgm = M LP (LN (Ẑgm )) + Ẑgm                        (5)    lationship between each joint in a single frame, and the temporal
                                                                                   attention Attnt is used to simulate the temporal relationship be-
where m = 1, 2, . . ., M, LN (·) is the layer normalization [46].                  tween the same joints in local T̄ frames. The Attns and Attnt
M SA in GTE as shown in Fig. 4(a), and the matrix size of its                      are calculated as follows:
attention map is T × T . The above formulas can be expressed
by using a function GT E(·) as follows:                                              Attns (Q, K, V ) = Reshape(Attention(Qs , Ks , Vs )) (11)
                          Zgm = GT E(Zgm−1 )                                (6)      Attnt (Q, K, V ) = Concat(Attention(Qt , Kt , Vt ))                   (12)
                                                                                   where Attention(·) is the scaled dot-product attention in (1).
D. Global-to-Local
                                                                                   The matrix size of the attention map of Attns and Attnt are
   Since the output of global part (GSE and GTE) cannot be                         T × N × N and N × S × T̄ × T̄ , respectively. Suppose that
directly utilized in local part (LSTE and TA), we propose a                        the MSA-ST has h heads and the MSA-ST operation is simply

     Authorized licensed use limited to: South-Central Minzu University. Downloaded on October 01,2026 at 03:36:18 UTC from IEEE Xplore. Restrictions apply.

                                                                                 process is applied to the input Zl0 ∈ RN ×T ×dl to obtain the out-
                                                                                 put ZlN ∈ RN ×1×dl as follows:
                                                                                                  Ẑln = Zln−1 + Etn                                         (19)

                                                                                                  Ẑln = M SA−ST (LN (Ẑln )) + Ẑln                         (20)

                                                                                                  Ẑln = M LP (LN (Ẑln )) + Ẑln                            (21)

                                                                                                  Zln = T A(Ẑln , T̄n )                                     (22)
Fig. 5. Proposed spatial and temporal attention fusion(STAF) module, which
is used to fuse the spatial and temporal attention.                              where n = 1, 2, . . ., N , T A(·) is the temporal aggregator, T̄n is
                                                                                 the number of local frames in n layer, Etn denotes the temporal
                                                                                 position embedding before the n-th layer. The above formulas
achieved by concatenating all heads as follows:                                  can be expressed as follows:

 M SA−ST = F(Concat(Attnis (·), Concat(Attnit (·))) · W                                          Zln = T A(LST E(Zln−1 + Etn ), T̄n )                        (23)
                                                     (13)
                                                                                 F. Temporal Aggregator
where i = 1, 2, . . ., h, W ∈ Rdl ×dl is a learnable matrix, F(·)                   This work is a task of regressing a single frame (middle
is a fusion function in (17), and Attnis (·), Attnit (·) are the i-th            frame) from the whole sequence. If the temporal sequence is
head of Attns (·), Attnt (·) respectively.                                       only learned without aggregation, the model will get a lot of re-
   Spatial and Temporal Attention Fusion (STAF): With the aim                    dundant information and increase the computational overhead.
of better integrating the two branch information of Attns and                    Therefore, we propose a temporal aggregator to better regress
Attnt , instead of simple summation, we design the STAF to dy-                   the 3D pose of the intermediate frame and aggregate the local
namically adjust spatial and temporal attention information, as                  information. Specifically, we aggregate the temporal informa-
shown in Fig. 5. We first concatenate Attns and Attnt in the fea-                tion of each node separately through a 1D convolution, and the
ture channel, and the resulting dimension is RN ×T ×(2·dl ) . The                temporal sequence of the input changes from T to S, where T
Average pooling and Max pooling operations are used to com-                      and S denote the number of complete sequences of the current
press the temporal dimension of the input features. This process                 layer and the number of sequences after the current layer has
generates two features of dimension RN ×1×(2·dl ) , and combines                 been divided, respectively. The above method can be expressed
the average and maximum features by adding them. Finally, the                    as a function T A(·, T̄ ), T̄ is the number of local frames of the
fully connected network and sigmoid operation are performed                      divided sequences in current layer, and we need to use it in con-
to produce the weights αs , αt ∈ RN ×1×dl . The fusion of Attns                  junction with LSTE as shown in (23).
and Attnt is defined as follows:
                                                                                 G. Regression Head
  fcat = Concat(Attns , Attnt )                                         (14)
                                                                                    Since we use a sequence of frames to predict the 3D pose
  favg+max = F C(AvgP ool(fcat ) + M axP ool(fcat )) (15)
                                                                                 of the center frame, supervising only a single pose will lose
  αs , αt = Split(Sigmoid(favg+max ))                                   (16)     the smoothness of time. Thus we impose an additional tem-
                                                                                 poral smoothness constraint by adding a regression head from
  F(Attns , Attnt ) = αs · Attns + αt · Attnt                           (17)
                                                                                 the output of GTE during training and obtain the full sequence
   Network Structure of LSTE: The input Z1 of LSTE is obtained                   prediction output Ŷ ∈ RT ×N ×3 of 3D pose sequences. Lastly,
by global to local projection, in (7) and (8). In order to locate the            a regression head is added to the output of LSTE to obtain
spatial position of a patch, we add positional encodings to inject               the prediction output Y ∈ RN ×3 of middle frame’s 3D joint
sequence information into the input, namely spatial positional                   locations.
encoding Es ∈ RN ×1×dl . It is a trainable sequence matrix that
is added to the input and can be expressed as follows:                           H. Loss Function
                                                                                    Since the output of GTE is the feature information of all
                            Zl0 = Z1 + Es                               (18)     frames, we need to supervise the full sequence scale to achieve
                                                                                 temporal consistency on the sequence. Therefore we adopt the
   The network structure of LSTE as shown in Fig. 3(c). The                      sequence loss [15] to minimize the errors between predictions
local spatio-temporal encoder and temporal aggregator (in Sec-                   and ground truths:
tion III-F) combine to model local features and consist of N
                                                                                                              1 
                                                                                                                    N T
layers. The temporal dimension changes for each layer, so the
                                                                                               Lglobal =                   Pi,j − Ŷi,j 2                  (24)
model is prefixed with trainable temporal position embeddings                                                N · T i=1 j=1
before each layer, and the first temporal position embedding is
Et1 ∈ R1×T ×dl , after which the temporal embedding of each                      where Ŷi,j and Pi,j represent the sequence of estimated and
layer changes with the temporal dimension to change T. This                      ground truth 3D positions of joint i at frame j, respectively.

   Authorized licensed use limited to: South-Central Minzu University. Downloaded on October 01,2026 at 03:36:18 UTC from IEEE Xplore. Restrictions apply.

WANG et al.: GLOBAL AND LOCAL SPATIO-TEMPORAL ENCODER                                                                                                         4045

   Given that LSTE models local frames and TA aggregates tem-
poral information to finally obtain the characteristics of inter-
mediate frames, we must supervise a single pose. We apply the
standard Mean Per Joint Position Error (MPJPE) loss to mini-
mize the error between the predicted and ground truth pose as:

where Yi and Pi are the middle frame’s estimated and ground
truth 3D positions of i-th joint in the current pose respectively.
                                                                                 Fig. 6. MPJPE metric comparison under different input frames on the Hu-
   The network is trained in an end-to-end manner, the final loss                man3.6 M test.
function L is defined as:
                     L = λg Lglobal + λl Llocal                         (26)
where λg and λl are weighting factors.
                                                                                    In our experiments, the proposed GLSTE, dimensions dg =
                                                                                 256 and dl = 32, and multi-attention heads h = 8. The temporal
                           IV. EXPERIMENT
                                                                                 downsampling rate [16] of TDS is 9 at T = 9, 27 and 3 at T =
A. Datasets and Evaluation Protocols                                             81, 243. For GTE, the layer M = 3 at T = 9, 27, 81, and M = 4
   We evaluate the proposed model on two 3D human pose esti-                     at T = 243. For LSTE, when T = 9, 27, 81, 243 the number of
mation datasets: Human3.6M [47] and MPI-INF-3DHP [48].                           local frames T̄ of each layer is set to (3, 3), (3, 3, 3), (3, 3, 3,
   Human3.6 M is a widely used dataset for 3D human pose                         3) and (9, 9, 3) respectively. The weight factors λg and λl of the
estimation. It contains 3.6 million video frames and the corre-                  loss function are set to 1.
sponding 3D human pose annotations. The dataset consists of
15 different activities of 11 actors captured by four cameras
with different views. Following the same strategy of the pre-                    C. Comparison With State-of-the-Art Methods
vious methods [12], [14], [16], [44], the model is trained on
                                                                                    Results on Human3.6 M: We compare our GLSTE with previ-
five subjects (S1, S5, S6, S7, S8) and evaluated on two subjects
                                                                                 ous state-of-the-art methods on the Human3.6 m dataset. We use
(S9, S11) on a 17-joint skeleton. We use the most commonly
                                                                                 CPN [1] as a 2D keypoint detector and then train our network
used evaluation metrics (MPJPE and PMPJPE). MPJPE (i.e.
                                                                                 on the detected 2D pose sequences. As shown in Table I, the
Protocol 1) is to measure the mean Euclidean distance between
                                                                                 performance of our model at 243 frames is shown. Our method
predicted and ground-truth joint positions without any transfor-
                                                                                 performed well in protocol 1 (42.2 mm), protocol 2 (33.8 mm).
mation. P-MPJPE (i.e. Protocol 2) is the MPJPE after aligning
                                                                                 We also train our model using the same refining module as [15],
the predicted 3D pose with the ground-truth using a rigid trans-
                                                                                 [16], [31], [33]. It achieves 41.9 mm under protocol 1, outper-
formation.
                                                                                 forming all other methods.
   MPI-INF-3DHP is a challenging dataset for both indoor and
                                                                                    As shown in Table II, our method achieves the best results
outdoor scenarios. It recorded 8 activities performed by 8 actors
                                                                                 (27.9 mm in MPJPE) when using ground true 2D poses as input,
from 14 camera views, containing three scenes: green screen,
                                                                                 which outperforms all other methods. And the detection results
non-green screen, and outdoor. Following the Setting in [14],
                                                                                 of our method are better than previous work except Directions
[33], we report metrics of MPJPE, Percentage of Correct Key-
                                                                                 action, which also proves the robustness of our model.
point (PCK) within 150 mm range, and Area Under Curve (AUC)
                                                                                    In Table III, we report a comparison with state-of-the-art
for a range of PCK thresholds as evaluation metrics in our ex-
                                                                                 methods for different input frames (T) in terms of the total num-
periments.
                                                                                 ber of parameters and the estimated floating-point operations
                                                                                 (FLOPs) per frame, and the performance of the model. It can be
B. Implementation Details
                                                                                 seen that our method has the least number of parameters but the
   We implement our proposed approach with Pytorch [52].                         best performance. Compared with the method based on the fully
The training and testing were performed on a single GeForce                      convolutional architecture [12], [35], the method based on the
GTX 3090 GPU. We use two stages of pre-training [16] and                         Transformer architecture [16] is more effective, and our method
fine-tuning to train the model. In both stages, we use Adam op-                  outperforms methods that also use pre-training and downsam-
timizer [53] to train the model for 80 epochs, with initial learning             pling strategies [16]. It is worth noting that the performance of
rates of 1e-4 and 7e-4, respectively, and the decay factor of each               our model in 27 frames, with 43.7 mm MPJPE, exceeds the re-
epoch was 0.97. Set the batch size to 256. We use horizontal flip                sults of Shan et al. [16] in 81 frames and Chen et al. [35] in
as data augmentation in training and testing. Following [12], the                243 frames, and our param and FLOPs are far less than them.
cascaded pyramid network (CPN) [1] is used for Human3.6 M                        Meanwhile, Fig. 6 also shows the comparison of MPJPE metrics
and the 2D ground-truth poses are used for MPI-INF-3DHP.                         under different input frames in the table.

<!-- [clean: truncated to max_content_length] -->