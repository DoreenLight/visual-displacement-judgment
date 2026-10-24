<!-- 本文件是 MinerU 自动抽取的**英文结构化正文**，不是中文翻译。 -->
<!-- 中文翻译若需要，须另走 06_附录/pdf2zh调用脚本/translate_all.py，不得把本文件当译文引用。 -->
<!-- 抽取来源：01_参考论文/康复辅助/宋嵘_辅助与训练强度/Zheng2025-AdvIntellSys：基于高斯混合模型的上肢线缆驱动康复机器人肌电控制/Zheng2025-AdvIntellSys：基于高斯混合模型的上肢线缆驱动康复机器人肌电控制.pdf -->
<!-- DOI：10.1002/aisy.202400505 -->
<!-- 图片：15 张，位于同级 images/，已按正文引用相对路径改写。 -->

# An Electromyographic-Based Control Using Gaussian Mixture Model on an Upper-Limb Cable-Driven Rehabilitation Robot

Jianlin Zheng, Zhijun Li, Ting Xu, and Rong Song\*

Electromyographic (EMG)-based admittance control by arm force can provide continuous motion control in robot-assisted rehabilitation. Natural and complex physical human–robot interactions utilizing intelligent EMG-based interfaces require a computational estimation model for 3D voluntary forces. Existing computational models infrequently directly encode the interconnections among the spatial-dimensional components of 3D arm forces and experience performance degradation caused by confounding variables like muscle fatigue in EMGbased control. Motivated by this challenge, a computational model is proposed using the Gaussian mixture model (GMM), which only requires a user-friendly calibration by an incremental GMM approach to reduce the effect of muscle fatigue. GMM aims to encode the multivariable connection between EMG and voluntary force by joint probability density distribution. The EMG-based control system retrieves the estimated voluntary forces given EMG from GMM by utilizing the Gaussian mixture regression. Herein, the performance ofthe proposed EMG-based admittance control is tested using GMM by trajectory-tracking experiments before and after the fatigue-inducing experiment. The experiments include eight healthy participants. The experimental outcomes prove that the EMG-based control using calibrated GMM shows an increase of 23.66% and 8.17% in tracking precision and motion compliance, showing potential application across diverse domains involving physical human–robot interactions.

## 1. Introduction

Stroke can lead to movement impairments and impact the quality of life of poststroke survivors. Rehabilitation training plays a vital role in restoring damaged motor pathways<sup>[1]</sup> and improving patients’ daily functional movement ability.<sup>[2]</sup> However, physical therapists are in short supply, and the quality of their treatment can be affected due to exhaustion. As rehabilitation robots can tirelessly provide consistent therapy and facilitate the recovery of motor functions, robot-assisted rehabilitation is becoming an attractive research subject.<sup>[3]</sup>

Participant engagement influences robot-assisted rehabilitation, while the control strategy directly affects human engagement levels. One ofthe pivotal components of rehabilitation robots lies in the control strategy. Passive control schemes assist patients with severely impaired motor function to complete rehabilitation training under a robot-dominant mode, such as the passive mode in MIT-MANUS,<sup>[4]</sup> MIME,<sup>[5]</sup> and ARMin.<sup>[6]</sup> Encouraging active participation is essential to improve the efficacy of rehabilitation,<sup>[7]</sup> especially for those with fewer impairments. Active impedance control strategies can facilitate muscle recovery by promoting patients’ voluntary participation.<sup>[8,9]</sup> Electromyographic (EMG) directly reflects the intention of motions and occurs before actions happen.<sup>[10]</sup> Employing EMG as a control source can make robots sense and decode motion intentions and facilitate human–robot movement synchronization, improving the level of active participation.<sup>[11,12]</sup>

Existing EMG-based control strategies typically conducted discrete motion control by pattern recognition, such as linear discriminant analysis<sup>[13]</sup> and support vector machines.<sup>[14]</sup> However, physical interactions between human arms and intelligent rehabilitation robots are essential to robot-assisted rehabilitation.<sup>[15,16]</sup> Controlling an end-effector through arm force involves continuous changes offorces in direction and magnitude, implying that extracting voluntary force intention demands a continuous estimation approach.<sup>[17]</sup> For example, Louis et al. achieved force estimation in a single dimension and performed reaching tasks by a Hill-type model.<sup>[18]</sup> Xiong et al. developed an artificial neural network with a small input dimension<sup>[19]</sup> and a trans-scale feature decoding network<sup>[20]</sup> to extract continuous joint torques from EMG. Xie et al. applied a state-space model to decode 3D voluntary forces for reaching tasks.<sup>[21]</sup> Recently, rehabilitation robotics performed biomimetic strategies to improve the intelligence of EMG-based control, facilitating bioinspired attributes in physical human-robot interaction (pHRI), such as adaptability.<sup>[22]</sup> For example, Zeng et al. employed Gaussian process to learn and predict EMG-based continuous motion intention.<sup>[17]</sup>

A computational model that can decode 3D voluntary forces helps perform natural and complex pHRI under intelligent EMG-based interfaces. The Hill-type model can achieve force estimation in a single dimension<sup>[18]</sup> but was limited to lowdimensional force or joint torque estimation.<sup>[23]</sup> Zhang et al. employed three parallel long short-term memory (LSTM) to estimate the direction and strength of three 1D forces separately, preventing the complexity of the network model required for directly establishing the relationship between 3D voluntary arm force and EMG signals.<sup>[24]</sup> Su et al. divided the force according to three spatial dimensions, applied a deep convolutional neural network to perform estimations, and calibrated it independently.<sup>[15]</sup> Some methods encoded the interconnections among the spatial-dimensional components ofthe 3D arm force. Zhang et al. modeled the relationship between EMG and 3D arm force by a hierarchical neural network with a low-layer shared feature extractor and three parallel high-layer dimension-specific fine-tuners.<sup>[16]</sup> The computational model that directly encodes the 3D components of voluntary forces may reduce errors attributable to modeling assumptions. Wu et al. employed a generalized regression neural network (GRNN) to estimate 3D push–pull forces but necessitated the accessibility of the original data.<sup>[25]</sup> Huang et al. utilized a state-space model for decoding 3D forces from EMG signals.<sup>[26]</sup> However, this approach was limited to specific tasks and suitable only for a simple combination of straight-line trajectory tasks. Xie et al. proposed a deep LSTM utilizing a regression layer with 3D output and demanded a time-consuming static–dynamic calibration before the experiment began.<sup>[27]</sup>

Although the field of EMG-based control made progress, existing EMG-to-force computational models insufficiently match human users and typically consider the performance without the influence of intrinsic and extrinsic factors.<sup>[28,29]</sup> Furthermore, as the upper-limb movements of humans involve three dimensions inherently,<sup>[30]</sup> a natural and complex pHRI utilizing intelligent EMG-based interfaces requires a computational model that can encode 3D voluntary forces. In this study, we proposed to use GMM to learn the nonlinear relationship between EMG and 3D force for the EMG-based control on an upper-limb cable-driven rehabilitation robot (ULCDRR). GMM can achieve a fast 3D voluntary force output in real-time through Gaussian mixture regression (GMR) without the need for original training data.<sup>[31]</sup> GMM can be regarded as a bioinspired approach as it emulates the processes of probabilistic modeling and reasoning performed by the human brain,<sup>[32,33]</sup> in which such a humanmatched model may facilitate human–robot synchronization during pHRI movement. The existing development based on GMM focused on pattern recognition<sup>[34–36]</sup> and imitation learning,<sup>[37,38]</sup> lacking research in 3D continuous motion control with intelligent EMG-based robotic systems. The need for adequateintensity treatment often leads to muscle fatigue in stroke patients,<sup>[39]</sup> which can affect the performance of EMG-based control schemes, including the accuracy of the force estimation<sup>[40,41]</sup> and control compliance.<sup>[42]</sup> Thus, we calibrated the trained GMM based on the incremental GMM approach to reduce the effect of muscle fatigue on the consistency and precision of robot-assisted rehabilitation employing intelligent EMG-based interfaces.<sup>[24,43]</sup> To perform compliance control, we employed the force estimated as the input of an EMG-based admittance controller.

The main contributions of this article are the following: 1) We proposed an EMG-based admittance control using GMM and achieved real-time 3D motion control. 2) We developed an incremental GMM technique to calibrate the GMM and embedded it into hardware to perform calibration conveniently and quickly. 3) We designed a fatigue-inducing experiment and verified the performance of the proposed approach under muscle fatigue through trajectory-tracking experiments.

## 2. System Description

We developed an upper limb cable-driven rehabilitation robot<sup>[21,44]</sup> (Figure 1). The end-effector of the robot is connected to four motors (DM1B-045G, Yokogawa, Japan) using four cables, with each cable’s tension measured by a force sensor (HSTL-BLSM, Beijing Huakong Xingye Technology Development Co., Ltd., China). The motion capture system (OptiTrack 2, NaturalPoint, USA) is to acquire the position information of the end-effector. The six-axis force sensor (SRI-V-210105-G) integrated with the end-effector measures the pHRI force. The employment of the dSPACE MicroLabBox 1202 platform (dSPACE, Paderborn, Germany) combined with MATLAB R2019b software (The Mathworks Inc., MA, USA) is to record real-time signals and generate control commands. Additionally, we design the human–robot interface using the dSPACE ControlDesk 7.2 software, and subjects can receive visual feedback of their movement performance through the computer screen in front of them.

![](images/bcece6eb368566ced2d592cf5ce892412911ae9eaae4394b75c7b132bfd2d5d6.jpg)  
Figure 1. Structure diagram of ULCDRR. The coordinate direction in the <sup>fi</sup>gure is the Cartesian coordinate system established by the motion capture system.

## 3. Experimental Section

## 3.1. Control Framework of ULCDRR

The control framework of ULCDRR comprises two essential phases (Figure 2): a learning phase and a task phase. In the learning phase, we established the joint probability density by GMM between processed EMG ${ \pmb \xi } _ { s }$ and force $\pmb { \xi } _ { f }$ signals. The incremental GMM approach was to calibrate the trained GMM $\{ \pi _ { j } , \pmb { \mu } _ { j } , \pmb { \Sigma } _ { j } \} _ { j = 1 } ^ { K }$ In the task phase, GMR utilized the new incoming ${ \pmb \xi } _ { s }$ to estimate the 3D force $\hat { \pmb { \xi } } _ { f }$ based on $\{ \pi _ { j } , \pmb { \mu } _ { j } , \pmb { \Sigma } _ { j } \} _ { j = 1 } ^ { K }$ . The desired estimated voluntary force $\hat { F }$ was the denormalization result of $\hat { \pmb { \xi } } _ { f } .$ Finally, the low-level controller converted $\hat { F }$ into the four cable forces $\mathbf { F } _ { 1 }$ ofULCDRR, thereby driving the end-effector to the target position.

![](images/6883ac3b1f09f295832ff2a6161797acbc6412f70802166ac13d3bcebd88e3d6.jpg)  
Figure 2. The overall control framework of ULCDRR. During the learning phase, GMM encoded the relationship between the processed EMG signals $\pmb { \xi } _ { s }$ and 3D forces $\pmb { \xi } _ { f } ,$ , producing the initial parameter set $\{ \pi _ { j } , \mu _ { j } , { \pmb \Sigma } _ { j } \} _ { j = 1 } ^ { K } .$ During the task phase, GMR was initialized with the trained parameter set $\{ \pi _ { j } , \mu _ { j } , \pmb { \Sigma } _ { j } \} _ { j = 1 } ^ { K }$ and utilized the new incoming ${ \pmb \xi } _ { s }$ to estimate $\hat { \pmb { \xi } } _ { f }$ . The incremental GMM approach calibrated the trained parameter set $\{ \pi _ { j } , \mu _ { j } , \pmb { \Sigma } _ { j } \} _ { j = 1 } ^ { K } .$ The estimated forces $\hat { \pmb { \xi } } _ { f }$ were denormalized to $\hat { \pmb F } ,$ and the low-level controller converted $\hat { \pmb F }$ into four cable forces $[ F _ { { \sf I } _ { 1 } } , F _ { { \sf I } _ { 2 } } , F _ { { \sf I } _ { 3 } } , F _ { { \sf I } _ { 4 } } ]$ . Speci<sup>fi</sup>cally, the EMG-based admittance controller converted $\hat { \pmb F }$ into the desired position $\pmb { x } _ { \mathrm { d } }$ of the end-effector based on the current position. Then, the cable planner determined the desired cable length $\pmb { p } _ { \mathrm { d , l } } = [ p _ { \mathrm { d , l _ { 1 } } } , p _ { \mathrm { d , l _ { 2 } } } , p _ { \mathrm { d , l _ { 3 } } } ]$ , and PID controller calculated the required cable force $\pmb { F } _ { 1 } = [ F _ { 1 _ { 1 } } , F _ { 1 _ { 2 } } , F _ { 1 _ { 3 } } ]$ to minimize the error between actua $\pmb { p } _ { \mathtt { C } , \rVert }$ and desired cable lengths $\pmb { p } _ { \mathrm { d } , | }$ . The cable tension controller generated $F _ { { \vert } _ { 4 } }$ to establish a positive distribution in the three upper cables. Finally, the tension of the four cables displaces the end-effector to achieve pHRI, matching the motion of a human user.

## 3.1.1. GMM

In this study, we established the joint probability density distribution between ${ \pmb \xi } _ { s }$ and $\pmb { \xi } _ { f }$ by GMM. GMM is a probabilistic model consisting of K Gaussian components and is used to encode features for classification or regression.<sup>[45]</sup> The joint probability density function $\mathcal { P }$ can be written as

$$
\mathcal { P } ( \xi _ { s } , \xi _ { f } ) = \sum _ { j = 1 } ^ { K } \pi _ { j } \mathcal { N } \big ( \xi _ { s } , \xi _ { f } \big | \mu _ { j } , \Sigma _ { j } \big )\tag{1}
$$

with

$$
\pmb { \mu } _ { j } = \left[ \pmb { \mu _ { s , j } } \right] , \quad \pmb { \Sigma } _ { j } = \left[ \pmb { \Sigma } _ { s s , j } \quad \pmb { \Sigma } _ { s f , j } \right]\tag{2}
$$

where $\pmb { \xi } _ { s } \in \mathbb { R } ^ { D _ { s } \times N _ { 0 } }$ is the $D _ { \mathrm { s } } .$ -dimensional feature vector after principle component analysis (PCA) processing and ${ \pmb { \xi } } _ { f } \in$ $\mathbb { R } ^ { D _ { f } \times N _ { 0 } }$ is the $D _ { \mathrm { f } }$ dimensional force vector normalized by the min–max approach. $N _ { 0 }$ denotes the amount of the training data. PCA here can mitigate the issue ofthe singular or ill-conditioned covariance matrices that can arise in training GMM due to linear correlations among input features.

We employed the shared covariance, a parameter-sharing technique, to mitigate the risk ofoverfitting and reduce the influence of the initialization method. The shared covariance is defined as

$$
\pmb { \Sigma } _ { 1 } = \pmb { \Sigma } _ { 2 } = \dots = \pmb { \Sigma } _ { K } = \pmb { \Sigma } ^ { ' } = \frac { \sum _ { j = 1 } ^ { K } \pi _ { j } \pmb { \Sigma } _ { j } } { \sum _ { j = 1 } ^ { K } \pi _ { j } }\tag{3}
$$

The parameter $\boldsymbol \theta _ { j } = \left\{ \pi _ { j } , \mu _ { j } , \pmb { \Sigma } _ { j } \right\}$ of each Gaussian component was starting from a rough estimation by k-meansþþ segmentation. Then, the initial GMM $\pmb \theta = \{ \theta _ { j } \} _ { j = 1 } ^ { K }$ (Initial-GMM) was trained offline by the standard expectation–maximization (EM) algorithm. The number of Gaussian components K was determined by a fivefold cross-validation approach.

## 3.1.2. Incremental GMM

In this study, we calibrated the Initial-GMM by an incremental GMM approach. This technique assumes that the posterior probabilities $\mathscr { P } ( \pmb { \theta } | \pmb { \xi } _ { m } )$ remain the same while utilizing the new data $\{ \pmb { \xi } _ { m } \} _ { m = 1 } ^ { N _ { n } }$ to calibrate the model.<sup>[46]</sup> $N _ { n }$ represents the number ofthe recorded samples associated with the nth EM iteration. In the gth EM step of the nth EM iteration, the expectation step (E-step) is

$$
\begin{array} { r } { \left\{ { \mathcal { P } } \Big ( \theta _ { j } ^ { ( g ) } | \pmb { \xi } _ { m } \Big ) = \frac { \pi _ { j } ^ { ( g ) } \mathcal { N } \Big ( \pmb { \xi } _ { m } | \pmb { \mu } _ { j } ^ { ( g ) } , \pmb { \Sigma } _ { j } ^ { ( g ) } \Big ) } { \sum _ { i = 1 } ^ { K } \pi _ { i } ^ { ( g ) } \mathcal { N } \Big ( \pmb { \xi } _ { m } | \pmb { \mu } _ { i } ^ { ( g ) } , \pmb { \Sigma } _ { i } ^ { ( g ) } \Big ) } \right. } \\ { \left. E _ { j } ^ { ( g + 1 ) } = \sum _ { m = 1 } ^ { N _ { n } } \mathcal { P } \Big ( \pmb { \theta } _ { j } ^ { ( g ) } | \pmb { \xi } _ { m } \Big ) \right. } \end{array}\tag{4}
$$

Then, the maximization step (M-step) is

$$
\begin{array} { r l } & { ( \pi _ { j } ^ { ( \varepsilon + 1 ) } = \frac { E _ { j } ^ { ( 0 ) } + E _ { j } ^ { ( \varepsilon + 1 ) } } { N _ { 0 } + N _ { n } }  } \\ & {   \mu _ { j } ^ { ( \varepsilon + 1 ) } = \frac { E _ { j } ^ { ( 0 ) } \mu _ { j } ^ { ( 0 ) } + \sum _ { n = 1 } ^ { N _ { \varepsilon } } P ( \theta _ { j } ^ { ( \varepsilon ) } | \xi _ { n } ) \xi _ { n } } { E _ { j } ^ { ( 0 ) } + E _ { j } ^ { ( \varepsilon + 1 ) } }  } \\ & {  \sum _ { j } ^ { [ z _ { j } ^ { ( \varepsilon + 1 ) } =  } \frac { E _ { j } ^ { ( 0 ) } ( \sum _ { j } ^ { ( 0 ) } + ( \mu _ { j } ^ { ( 0 ) } - \mu _ { j } ^ { ( \varepsilon + 1 ) } ) ( \mu _ { j } ^ { ( 0 ) } - \mu _ { j } ^ { ( \varepsilon + 1 ) } ) ^ { T } ) } { E _ { j } ^ { ( 0 ) } + E _ { j } ^ { ( \varepsilon + 1 ) } }  } \\ & {  + \frac { \sum _ { n = 1 } ^ { N _ { \varepsilon } } P ( \theta _ { j } ^ { ( \varepsilon ) } | \xi _ { n } ) ( \xi _ { n } - \mu _ { j } ^ { ( \varepsilon + 1 ) } ) ( \xi _ { n } - \mu _ { j } ^ { ( \varepsilon + 1 ) } ) ( \xi _ { n } - \mu _ { j } ^ { ( \varepsilon + 1 ) } ) ^ { T } } { E _ { j } ^ { ( 0 ) } + E _ { j } ^ { ( \varepsilon + 1 ) } }  } \end{array}\tag{5}
$$

In this study, we defined the complete parameter set $\mathbf { \Theta } _ { \mathbf { \Theta } } \Theta$ as $\{ ( \pmb \theta _ { j } , E _ { j } ) \} _ { j = 1 } ^ { K } ,$ where $\begin{array} { r } { E _ { j } = \sum _ { m = 1 } ^ { N _ { n } } \mathcal { P } \big ( \pmb { \theta } _ { j } | \pmb { \xi } _ { m } \big ) } \end{array}$ . At the beginning of the first EM iteration, $\mathbf { \delta } _ { \mathbf { \Theta } } \mathbf { \Theta } \Theta _ { 0 }$ (Initial-GMM) is set as the initial, i.e., $\left\{ \pmb { \theta } _ { j } ^ { ( 0 ) } , E _ { j } ^ { ( 0 ) } \right\} _ { i = 1 } ^ { K } = \pmb { \Theta } _ { 0 } ,$ . In the incremental GMM proposed by Calinon et al.<sup>[46]</sup> they utilized the average log-likelihood to perform the convergence assessment after the M-step. The average log-likelihood is

$$
\mathcal { L } = \frac { 1 } { N _ { n } } \sum _ { m = 1 } ^ { N _ { n } } \log \left( \sum _ { j = 1 } ^ { K } \pi _ { j } \mathcal { P } \big ( \pmb { \xi } _ { m } | \pmb { \theta } _ { j } \big ) \right)\tag{6}
$$

The criterion is

$$
\frac { \mathcal { L } ^ { ( g + 1 ) } } { \mathcal { L } ^ { ( g ) } } < 1 + \varepsilon\tag{7}
$$

where $\mathcal { L } ^ { \left( g \right) }$ and $\mathcal { L } ^ { ( g + 1 ) }$ are based on $\mathbf { \Theta } \Theta ^ { ( g ) }$ and $\mathbf { \Theta } \Theta ^ { ( g + 1 ) }$ , respectively. If (7) returned true, the first EM iteration would stop and $\mathbf { \Theta _ { 6 } } =$ $\mathbf { \Theta } \Theta ^ { ( g + 1 ) }$ would be the calibrated result. When the nth EM iteration was completed, $\left\{ \pmb { \theta } _ { j } ^ { ( 0 ) } , E _ { j } ^ { ( 0 ) } \right\} _ { j = 1 } ^ { K }$ would be replaced by $\mathbf { \Theta } _ { \mathbf { e } }$ , and $N _ { 0 }$ would be incremented by $N _ { n }$ to prepare for the (n þ 1)th EM iteration. In this study, the tolerance $\varepsilon = 1 \times 1 0 ^ { - 6 }$ was set. T bl 1 displays the definitions of the parameter notations.

We implemented the incremental GMM on the dSPACE controller. In the calibration experiment, the dSPACE can automatically calibrate the model without human intervention using the collected new incoming data. The data collection and model calibration program run as two separate threads, wherein the sample rate for capturing is 1 kHz, and the clock rate for task scheduling in calibration program is 100 Hz. The data collector would record the mean features of each window data (including EMG and force), where we arranged 128 windows, each with an 8 ms width, of data collection at equal intervals within a 30 s single data collection cycle.

Table 1. Parameters notations for estimation model learning.
<table><tr><td>Symbols</td><td>Definition and description</td></tr><tr><td> ${ \pmb \xi } _ { s } , { \pmb \xi } _ { f }$ </td><td>The processed EMG and force signals.</td></tr><tr><td> $\pmb { \xi } _ { f } , \widehat { \pmb { \Sigma } } _ { f } , \pmb { F }$ </td><td>The estimated conditional expectation, and extracted variability, and the estimated 3D voluntary force.</td></tr><tr><td> $D _ { \mathsf { c h s } }$ </td><td>The channel number of the raw EMG signals.</td></tr><tr><td> $D _ { \mathsf { s } } , D _ { \mathsf { f } }$ </td><td>The dimensionality of EMG and force features.</td></tr><tr><td> $N _ { 0 } , \ N _ { n }$ </td><td>The count of data collected for the offline pretraining and the</td></tr><tr><td> $c , \nu$ </td><td>nth EM iteration. The principal component coefficients and the estimated</td></tr><tr><td> $n _ { \mathsf { p } }$ </td><td>centered mean. The number of components requested.</td></tr><tr><td> $K , j$ </td><td>The number of Gaussian components and the jth Gaussian</td></tr><tr><td> $\{ \pi _ { j } , \mu _ { j } , \pmb { \Sigma } _ { j } \} , \ E _ { j }$ </td><td>component. Parameters and cumulative posterior probability of the jth</td></tr><tr><td>βj</td><td>Gaussian component. The probability that the component j has of being responsible  $\mathsf { f o r } \ \pmb { \xi } _ { s } .$ </td></tr><tr><td> $\theta _ { j } , \Theta$ </td><td>θj denotes  $\{ \pi _ { j } , \mu _ { j } , \Sigma _ { j } \}$  and Θ denotes the complete parameter set  $\{ ( \pmb \theta _ { j } , E _ { j } ) \} _ { j = 1 } ^ { K } .$ </td></tr><tr><td>g, n</td><td>The gth EM step and the nth EM iteration.</td></tr><tr><td> $\mathbf { \Theta } \Theta _ { 0 } , \mathbf { \Theta } \Theta _ { n } , \mathbf { \Theta } \Theta ^ { ( g ) }$ </td><td>The parameter sets involved Initial-GMM, CA-GMM accumulated from n EM iterations, and the results of the</td></tr></table>

We adjusted the computation order based on the execution logic of the general incremental GMM program Calinon et $\mathrm { { \bar { a l } . ^ { [ 4 6 ] } } }$ proposed, avoiding redundant computation parts and maintaining the equivalence with Calinon’s approach. The repeated parts occur in the computation of sample likelihood, i.e., $\mathcal { P } \Big ( \pmb { \xi } _ { m } | \pmb { \theta } _ { j } ^ { ( g + 1 ) } \Big )$ . The calculation of this redundant sample likelihood appears in the calculation of the average log-likelihood value $\pmb { \mathscr { L } } ^ { ( g + 1 ) }$ when making convergence judgment in the gth EM step, and in the calculation of the responsibilities $\mathcal { P } \big ( \pmb \theta _ { j } ^ { ( g + 1 ) } | \pmb \xi _ { m } \big )$ of the subsequent (gþ1)th EM step. Algorithm 1 shows the pseudocode of the incremental GMM performed on dSPACE. In this study, after the generation of $\mathcal { P } \big ( \pmb { \xi } _ { m } | \pmb { \theta } _ { j } ^ { ( g ) } \big )$ in the E-step ofthe gth EM step (Lines 6–11), we immediately calculated $\mathcal { L } ^ { \left( g \right) }$ (Line 12) and then performed the convergence assessment (Line 14) combined with $\pmb { \mathscr { L } } ^ { ( g - 1 ) }$ stored in the (g-1)th EM step. If (7) returned false, the M-step would perform (Lines 15–18); otherwise, the program would stop and set $\mathbf { \Theta } \Theta _ { n } = \mathbf { \Theta } \Theta ^ { ( g ) }$ (Line 24). To make the program compatible with the computing capability of the dSPACE controller, we assigned the summation of the denominator described in (4) to K computation steps (Lines 6–9) and updated the parameter subset of each Gaussian component over K computation steps (Line 15–17).

Algorithm 1. Incremental GMM algorithm embedded in dSPACE controller.

Input: K, $\left\{ \pmb { \theta } _ { j } ^ { ( 0 ) } , E _ { j } ^ { ( 0 ) } \right\} _ { j = 1 } ^ { K } , \mathcal { L } _ { n - 1 } , \left\{ \pmb { \xi } _ { m } \right\} _ { m = 1 } ^ { N _ { n } }$   
Output: Θ<sub>n</sub>   
1: MaxIter ← 20   
2: $\varepsilon \gets 1 \times 1 0 ^ { - 6 }$   
$3 \colon { \mathcal { P } } _ { \mathsf { m e m o r y } } \gets 0$   
4: $\mathcal { L } _ { \mathrm { m e m o r y } }  \mathcal { L } _ { n - }$   
5: for $g = 1 , 2 , \ldots ,$ MaxIter do   
6: for $i = 1 , 2 , \dots , K$ do   
7: $\mathcal { P } ( \pmb { \theta } _ { i } ^ { ( g ) } | \pmb { \xi } _ { m } )  \pi _ { i } ^ { ( g ) } \mathcal { N } ( \pmb { \xi } _ { m } | \pmb { \mu } _ { i } ^ { ( g ) } , \pmb { \Sigma } _ { i } ^ { ( g ) } )$   
8: $\mathcal { P } _ { \sf m e m o r y }  \mathcal { P } _ { \sf m e m o r y } + \mathcal { P } ( \pmb { \theta } _ { i } ^ { ( g ) } | \pmb { \xi } _ { m } )$   
9: end for   
10: normalize $\left\{ \mathcal { P } ( \pmb \theta _ { j } ^ { ( g ) } | \pmb \xi _ { m } ) \right\} _ { j = 1 } ^ { K } \ b \pmb \ y \mathcal { P } _ { m e m o r y }$   
11: $\left\{ E _ { j } ^ { ( g + 1 ) } \right\} _ { j = 1 } ^ { K } \gets \mathsf { E q u a t i o n ~ } ( 4 )$   
12: L<sup>ðgÞ</sup> ← Equation (6)   
13: $\mathcal { L } ^ { ( g - 1 ) }  \mathcal { L } _ { \mathrm { m e m o r y } }$   
14: $\mathfrak { i f } \frac { \mathcal { L } ^ { ( g ) } } { \mathcal { L } ^ { ( g - 1 ) } } > \mathbb { 1 } + \varepsilon$ then   
15: $\mathbf { f o r } j = 1 , 2 , \ldots , K$ do   
16: $\pi _ { j } ^ { ( g + 1 ) } , \mu _ { j } ^ { ( g + 1 ) } , \Sigma _ { j } ^ { ( g + 1 ) } \gets \mathsf { E q u a t i o n } ( 5 )$   
17: end for   
18: $\{ \pmb { \Sigma } _ { j } ^ { ( g + 1 ) } \} _ { j = 1 } ^ { K }  \mathsf { E q u a t i o n } \ ( 3 )$   
19: else   
20: break   
21: end if   
22: $\mathcal { L } _ { \mathrm { m e m o r y } }  \mathcal { L } ^ { ( g ) }$   
23: end for   
24: $\mathbf { \Theta } \Theta _ { n } \gets \mathbf { \Theta } \Theta ^ { ( g ) }$

## 3.1.3. GMR

We employed GMR technique to obtain the conditional expectation of $\pmb { \xi } _ { f }$ given ξ<sub>s</sub> based on the joint distribution described in (1). GMR provides a fast regression approach to compute the multivariate output online.<sup>[31]</sup> Thus, the prediction task can be defined as

$$
\mathcal { P } \big ( \pmb { \xi } _ { f } | \pmb { \xi } _ { s } \big ) = \sum _ { j = 1 } ^ { K } \beta _ { j } \mathcal { N } \Big ( \pmb { \xi } _ { f } | \widehat { \pmb { \mu } } _ { f | s , j } , \widehat { \pmb { \Sigma } } _ { f | s , j } \Big )\tag{8}
$$

with

$$
\beta _ { j } = \frac { \pi _ { j } \mathcal { N } \big ( \pmb { \xi } _ { s } | \pmb { \mu } _ { s , j } , \pmb { \Sigma } _ { s s , j } \big ) } { \sum _ { i = 1 } ^ { K } \pi _ { i } \mathcal { N } \big ( \pmb { \xi } _ { s } | \pmb { \mu } _ { s , i } , \pmb { \Sigma } _ { s s , i } \big ) }\tag{9}
$$

and

$$
\left\{ \begin{array} { l l } { \pmb { \hat { \mu } } _ { f \vert s , j } = \pmb { \mu } _ { f , j } + \pmb { \Sigma } _ { f s , j } \big ( \pmb { \Sigma } _ { s s , j } \big ) ^ { - 1 } \big ( \pmb { \xi } _ { s } - \pmb { \mu } _ { s , j } \big ) } \\ { \hat { \pmb { \Sigma } } _ { f \vert s , j } = \pmb { \Sigma } _ { f f , j } - \pmb { \Sigma } _ { f s , j } \big ( \pmb { \Sigma } _ { s s , j } \big ) ^ { - 1 } \pmb { \Sigma } _ { s f , j } } \end{array} \right.\tag{10}
$$

where $\beta _ { j }$ was the responsibility of each conditional probability density distribution to the new incoming predictor ξ . The estimation $E \big [ \mathcal { P } \big ( \pmb { \xi } _ { f } | \pmb { \xi } _ { s } \big ) \big ]$ of the conditional expectation was

$$
\widehat { \pmb { \xi } } _ { f } = \sum _ { j = 1 } ^ { K } \beta _ { j } \widehat { \pmb { \mu } } _ { f | s , j } , \quad \widehat { \pmb { \Sigma } } _ { f } = \sum _ { j = 1 } ^ { K } \beta _ { j } ^ { 2 } \widehat { \pmb { \Sigma } } _ { f | s , j }\tag{11}
$$

The denormalization method of the min–max approach was

$$
\hat { \pmb { F } } = ( b - a ) \hat { \pmb { \xi } } _ { f } + a\tag{12}
$$

where the minimum a and maximum b factors were empirically set to $[ - 3 5 , - 5 5 , - 4 5 ]$ and [35, 55, 45] respectively. Finally, the estimated 3D voluntary force $\hat { F }$ was the denormalization result of $\widehat { \pmb { \xi } } _ { f } . \widehat { \pmb { \Sigma } } _ { f }$ was the variability extracted from demonstrations.<sup>[47]</sup>

## 3.1.4. Low-Level Controller

In our previous work,<sup>[21]</sup> we proposed a low-level controller responsible for converting the received estimated force $\hat { F }$ into the four cable forces $F _ { 1 }$ of ULCDRR, facilitating the movement of the end-effector to reach the target position.

The desired position $\pmb { x } _ { \mathrm { d } }$ of the end-effector was generated via the EMG-based admittance controller

$$
- M _ { \mathrm { h } } \ddot { \pmb x } _ { \mathrm { c } } - \pmb { C } _ { \mathrm { h } } \dot { \pmb x } _ { \mathrm { c } } + \pmb { K } _ { \mathrm { h } } ( \pmb { x } _ { \mathrm { d } } - \pmb { x } _ { \mathrm { c } } ) = \pmb { F }\tag{13}
$$

where $M _ { \mathrm { h } } , \thinspace C _ { \mathrm { h } } ,$ and $K _ { \mathrm { h } }$ were the mass, damper, and stiffness matrices, respectively. In this article, we used the simplified admittance mode $\mathrm { l } ^ { [ 4 8 ] }$ rewritten as

$$
- C _ { \mathrm { h } } \dot { \pmb { x } } _ { \mathrm { c } } + \pmb { K } _ { \mathrm { h } } ( \pmb { x } _ { \mathrm { d } } - \pmb { x } _ { \mathrm { c } } ) = \pmb { F }\tag{14}
$$

where $\pmb { x } _ { \mathrm { c } }$ was the current position of the end-effector. Thus, we had the following desired position

$$
{ \pmb x } _ { \mathrm { d } } = K _ { \mathrm { h } } ^ { - 1 } ( { \pmb F } + { \pmb C } _ { \mathrm { h } } { \dot { \pmb x } } _ { \mathrm { c } } ) + { \pmb x } _ { \mathrm { c } }\tag{15}
$$

Subsequently, the desired position $\pmb { x } _ { \mathrm { d } }$ of the end-effector was transformed to the desired length $\pmb { p } _ { \mathrm { d } , 1 }$ of the three upper cables using a cable planner

$$
p _ { \mathrm { d } , \mathrm { l } _ { i } } = \sqrt { \| x _ { \mathrm { d } } - c _ { \mathrm { l } _ { i } } \| ^ { 2 } } \quad , \quad ( i = 1 , 2 , 3 )\tag{16}
$$

where $p _ { \mathrm { d } , \mathrm { l } _ { i } }$ was the desired length ofthe $l _ { i }$ th cable and $c _ { \mathrm { l _ { i } } }$ was the position of the $l _ { i }$ th pulley in Cartesian coordinates. Additionally, the proportional–integral–derivative controller was applied to drive the cables of ULCDRR so that the end-effector can track the desired position

$$
\begin{array} { r } { { F } _ { l } = { K } _ { P } \big ( { p } _ { d , l } - { p } _ { c , l } \big ) + { K } _ { I } \int _ { 1 } ^ { t } \big ( { p } _ { d , l } - { p } _ { c , l } \big ) d t + { K } _ { D } \big ( \dot { p } _ { d , l } - \dot { p } _ { c , l } \big ) } \end{array}\tag{17}
$$

where $\boldsymbol { F } _ { \mathrm { l } } = \left[ F _ { \mathrm { l } _ { 1 } } , F _ { \mathrm { l } _ { 2 } } , F _ { \mathrm { l } _ { 3 } } \right] ^ { \mathrm { T } }$ was the cable force vector (Figure 1), and $\pmb { p } _ { \mathrm { d } , \mathrm { l } }$ and $\pmb { p } _ { \mathrm { c , l } }$ were the desired and the actual position vector in joint space, respectively. $K _ { \mathrm { P } } , K _ { \mathrm { I } }$ , and $K _ { \mathrm { D } }$ were diagonal matrices. The driving force $F _ { \mathrm { l _ { 4 } } }$ ofthe lower $\mathsf { c a b l e } f _ { \mathrm { l } _ { 4 } }$ was controlled by a cable tension controller, aiming to set a positive tension distribution in the three upper cables

$$
F _ { l _ { 4 } } = \left\{ { { 0 \atop { 2 7 \left( { { f _ { l _ { 4 } } } - f _ { \operatorname* { m a x } } } \right) } } \ , \ f _ { l _ { 4 } } < f _ { \operatorname* { m i n } } } \atop { { f _ { \operatorname* { m i n } } } \leq f _ { l _ { 4 } } \leq f _ { \operatorname* { m a x } } }  \right.\tag{18}
$$

where $f _ { \mathrm { l _ { 4 } } }$ was cable-force measurement, and $f _ { \mathrm { m i n } }$ and $f _ { \mathrm { m a x } }$ were set to 7 and 10 N, respectively.

## 3.1.5. Signal Processing

In this study, we recorded the raw EMG signals at a sampling frequency of 1000 Hz. The acquired EMG underwent filtering by a notch filter at 50 Hz and a band-pass filter at 20–450 Hz. The processed EMG underwent full-wave rectification and was further low-pass filtered by a fourth-order Butterworth filter with a 4 Hz cutoff frequency, thereby extracting amplitude envelopes. The amplitude signals were normalized based on the maximum voluntary isometric contraction $\mathrm { ( M V C ) , } ^ { [ 4 9 ] }$ and the MVC normalization factors were incorporated into the program. The root mean square (RMS) features $\pmb { e } ( t ) \in \mathbb { R } ^ { D _ { \mathrm { c h s } } \times \cdot }$ were extracted from the normalized signals using a 300 ms moving window,<sup>[50]</sup> where $D _ { \mathrm { c h s } }$ denotes the channel number of EMG signals and · denotes the sampling data count.

The pca function in MATLAB 2019b, utilizing the singular value decomposition algorithm, produced the principal component coefficients $\boldsymbol { C } \in \mathbb { R } ^ { n _ { \mathrm { p } } \times D _ { \mathrm { c h s } } }$ s and the estimated centered mean $\pmb { \nu } \in \mathbb { R } ^ { D _ { \mathrm { c h s } } \times 1 }$ . The determination of $n _ { \mathrm { p } } ,$ the number ofcomponents requested, was sequentially accumulating the variance contributions of the PCA components in descending order until the cumulative sum attained 99%. The coefficients of C and ν were embedded into the dSPACE controller, and ${ \pmb \xi } _ { s } = { \bf C } \times ( { \pmb e } ( t ) - { \pmb \nu } )$ was performed to obtain the predictor ξ in the real-time control.

The 3D voluntary force measured by the six-axis force sensor was processed by median filtering with a 5 ms sliding window. The force was then normalized to the range of 0–1 by min–max approach and obtained $\pmb { \xi } _ { f }$

## 3.2. Metrics

The accuracy oftracking tasks and smoothness oftrajectories are universally applied to assess the performance ofEMG-based control schemes used for rehabilitation robots. The root mean square error (RMSE) was employed to measure tracking accuracy (RMSE ) between the actual $\pmb { x } _ { \mathrm { c } }$ and the targeted position $\pmb { x } _ { \mathrm { r } } .$ RMSE was also implemented to evaluate the estimation accuracy (RMSE ) between the prediction F<sup>ˆ</sup> and measured value F of 3D forces. The root mean square jerk (RMSJ) was utilized to analyze the compliance of trajectories. In addition, according to the study,<sup>[12]</sup> we adopted the percentage muscular activation (EMG level) and interaction force (IF) to evaluate the effort in pHRI. The mentioned metrics are as follows:

$$
\mathrm { R M S E } _ { x } = \sqrt { \frac { 1 } { N } \sum _ { m = 1 } ^ { N } { \| \pmb { x } _ { r , m } - \pmb { x } _ { c , m } \| ^ { 2 } } }\tag{19}
$$

$$
\mathrm { R M S E } _ { f } = \sqrt { \frac { 1 } { N } { \sum _ { m = 1 } ^ { N } { \| \hat { \boldsymbol { F } } _ { m } - \boldsymbol { F } _ { m } \| ^ { 2 } } } }\tag{20}
$$

$$
\mathrm { R M S J } = \sqrt { \frac { 1 } { N } \sum _ { m = 1 } ^ { N } { \| \\\overleftrightarrow { \mathbf { x } } _ { c , m } \| ^ { 2 } } }\tag{21}
$$

$$
\mathrm { E M G l e v e l } _ { \mathrm { h } } = \sqrt { \frac { 1 } { N } \sum _ { m = 1 } ^ { N } \| \pmb { e } _ { \mathrm { h } } ( m ) \| ^ { 2 } }\tag{22}
$$

$$
\mathrm { I F } = \sqrt { \frac { 1 } { N } \sum _ { m = 1 } ^ { N } | | \boldsymbol { F } _ { m } | | ^ { 2 } }\tag{23}
$$

where N is the number of sampling points of the trajectory, m is the mth point in the trajectory, and h is the hth channel of EMG signals.

## 3.3. Statistical Analysis

All statistical tests were at a significance level of $\alpha = 0 . 0 5$ . We performed a normality test (Lilliefors test) on the difference between the paired samples of interest. The paired-sample Wilcoxon signed-rank test and the paired-sample t-test were used for nonparametric and parametric analyses, respectively.

## 4. Experimental Protocol

In this study, the experimental protocol mainly consisted ofthree parts: 1) the data collection experiment, 2) the calibration and trajectory-tracking experiment, and 3) the fatigue-inducing experiment. We recorded EMG in real time by arranging eight sets of AgCl electrodes on the participant’s left arm: biceps brachii (BIC), long head of triceps (TRI-LO), medial head of triceps (TRI-ME), lateral head of triceps (TRI-LA), anterior of deltoid (DA), medial of deltoid (DM), posterior of deltoid (DP), and brachioradialis (BR). The 3D voluntary force data were collected by a six-axis force sensor integrated into the end-effector. In the data collection experiment, we obtained the training data (including EMG and force) for training an initial model (Initial-GMM). The one-trial experiment followed the following procedure: First, we calibrated the Initial-GMM by the incremental GMM approach to obtain a calibrated model (CA-GMM). Then, we employed the Initial-GMM and CA-GMM to perform trajectory-tracking experiments, respectively. Subsequently, we conducted the fatigueinducing experiment. Finally, we calibrated the Initial-GMM to obtain a new CA-GMM and repeated the same trajectorytracking experiments muscle fatigue. We recruited eight healthy subjects (three females and five males, $2 3 . 9 \pm 2 . 8$ years old) for the experiments. One person is left-handed, and all other participants are right-handed. All participants performed two trials following the experimental procedure above. This study was approved by School of Medicine, Sun Yat-sen University Institutional Review Board, and the ethical approval document number is Medical Ethics 2021 No. 14. All participants signed the written informed consent forms.

## 4.1. Data Collection Experiment

In the data collection experiment, we obtained training data for Initial-GMM, including EMG and force. We designed a virtual slider-tracking task on the visual feedback screen to guide this data collection experiment. The setting of the virtual task included a reference and a controlled slider (Figure 3a). The one-complete movement task of the reference slider was to complete a designed 30 s motion in the Cartesian coordinate system, corresponding to a one-cycle data collection. The magnitudes of the terminal positions along the X, Y, and Z axes were determined empirically as [35, 55, 45], respectively. Specifically, the motion was starting from the origin, advancing to the end of $+ X ,$ returning to the origin, proceeding to the end $\operatorname { o f } - X ,$ and returning to the origin, and then performing in the Z direction and finally in the Y direction. The moving direction and position ofthe reference slider guided the direction and magnitude ofthe force applied by participants. The position ofthe controlled slider reflected the direction and magnitude of the force exerted by participants, where the force-to-slider mapping was achieved by a six-axis force sensor integrated into the end-effector.

In this study, we collected training data through static isometric contraction experiments, where the end-effector was constrained at the origin by the four cables. Experiments included muscle relaxation demonstrations and force application demonstrations. During demonstrations with muscle relaxation, the participant raised the left arm without exerting force on the end-effector. In demonstrations with force application, the participant controlled the controlled slider to track the motion of the reference slider by applying force on the end-effector. We collected 20-cycle training data, including 2-cycle data from muscle relaxation demonstrations and 18-cycle data from force application demonstrations. After completing the data collection, we determined the number of Gaussian components of Initial-GMM for each subject by a fivefold cross-validation approach.<sup>[51]</sup>

(a)  
![](images/c91d3a893853cd97c848af0d5872c26b5519705480e4f39e8893244b81f38417.jpg)

(b)  
![](images/ae34ab7a71849c5f0f7780beb8f4ad4db71c74808d4eb238850db5c4e74f7a24.jpg)  
(d)

## 4.2. Calibration and Trajectory-Tracking Experiment

In the calibration experiment, we designed a demonstration task involving six EM iterations to calibrate the Initial-GMM. The demonstration in this experiment was identical to the demonstration with force application implemented in the data-collection experiment.

In the trajectory-tracking experiment, we designed a virtual 3D task space on the visual feedback screen, including a red reference slider and a rounded rectangle trajectory (Figure 3b). The subject tracked the motion of a red reference slider, drawing the required trajectory using EMG-based control. We designed two blocks for trajectory-tracking tasks. Each block included two EMG-based control schemes: one based on Initial-GMM $( \mathbf { \Theta } _ { \mathbf { 0 } } )$ and the other based on CA-GMM $( \Theta _ { n } )$ . Each control scheme performed one trajectory-tracking cycle in each block. The selection of control schemes was random, and subjects were unaware of the current scheme.

We conducted the calibration and trajectory-tracking experiment before and after the fatigue-inducing experiment, where the CA-GMM obtained were both iterative from the Initial-GMM. Additionally, the calibration and trajectory-tracking experiments that followed the fatigue-inducing experiment were performed immediately without any rest time. Before the formal experiment, participants engaged in four to five tracking practices using the scheme based on the Initial-GMM.

![](images/2cb930007dc0f6869217cbda088a70552d980b081d2e4352c3af23517dcde53a.jpg)

(c)  
![](images/06c35b68d20b9a9749332088b3d1c4d81ede31866bd13ba2bedec6437f2562e7.jpg)

![](images/3a7bed413ecc419102999ade04cdda2d1b72dd030a852f5b7d9085d0e0953195.jpg)  
Figure 3. Schematic diagram of the visual feedback on the visual screen in a) data collection experiment, b) trajectory-tracking experiment, and c) fatigue-inducing experiment. Participants can obtain their performance on the current control task by receiving information on a visual feedback screen. d) The real-world experimental scenarios. The Cartesian coordinate system in (a), (b), and (c) is constructed following the right-hand rule and aligns with the real-world scenarios (d).

## 4.3. Fatigue-Inducing Experiment

In the fatigue-inducing experiment, we designed an isometric contraction task to induce muscle fatigue. We utilized a targetapproaching task in a virtual space on the visual feedback screen, including a fixed red target slider and a green controlled slider (Figure 3c). The end-effector was constrained at the origin by the four cables and converted the force applied by the subject to the position of the controlled slider. The controlled slider was the force measurement that reflected the magnitude and direction of the force. This setup followed the protocol by Tkach et al.<sup>[52]</sup> where participants performed a fatigue-inducing experiment by static isometric contractions under the guidance of the force measurement. Participants controlled the controlled slider to move toward the target slider and did their best to maintain the effort level defined by the target slider. During the fatigue-inducing experiment, we encouraged participants to persevere. If subjects reported being too tired to continue, the experiment would stop. The calibration and trajectory-tracking experiments that followed the fatigue-inducing experiment were performed immediately. There was no rest time to prevent recovery from muscle fatigue. The assumption was that the same level of muscle fatigue was sustained in the calibration and trajectorytracking experiments after the fatigue-inducing experiment.<sup>[53]</sup>

## 5. Results

As shown in Figure 4, the average RMSE of 3D voluntary force estimation based on Initial-GMM and CA-GMM before the fatigue-inducing experiment was lower than that of after. As the number of EM iterations in the calibration experiment increases, the average RMSE based on CA-GMM decreases monotonically. Before the fatigue-inducing experiment, although the average RMSE of CA-GMM was higher than that of Initial-GMM, the discrepancy of average RMSE between the CA-GMM and Initial-GMM gradually narrowed with the increase of EM iterations. After the fatigue-inducing experiment, the average RMSE results of CA-GMM were higher than that of Initial-GMM during the initial four EM iterations. However, the RMSE outcome of CA-GMM was lower than that of Initial-GMM from the fifth EM iteration onward, indicating better performance in the accuracy of 3D force estimation.

Figure 5 displays tracking trajectories drawn by a subject without and with the influence of muscle fatigue. Figure 5a shows that the EMG-based control separately utilizing the Initial-GMM and CA-GMM drew 3D rounded rectangles with smoothness in trajectory-tracking experiments. Figure 5b shows the impact of muscle fatigue on the Initial-GMM-based control, manifesting as a decrease in compliance and increased tracking deviation. In contrast, the control scheme based on the CA-GMM improves tracking accuracy and compliance control.

Figure 6 illustrates the EMG level of eight-channel EMG signals in trajectory-tracking experiments, in which the high EMG level means the more muscular effort of participants during EMG-based control. Detailed values are available in Table 2. Under muscle fatigue conditions, CA-GMM reduced the EMG level more than Initial-GMM, especially at the BIC, DM, DP, TRI-LA, and TRI-LO channels. The improvements in these channels indicate that the proposed approach can reduce the muscle efforts of human users in the current task related to the inward pulling and outward extension of the forearm. The EMG level at the BIC, DM, DP, TRI-LA, and TRI-LO channels dropped by 4.02%, 4.81%, 8.85%, 4.68%, and 13.91%, respectively.

![](images/579967a7b6c1b60781cae812348a2e52a610dfa442dae84d0661b96631322c0c.jpg)  
Figure 4. The change of average RMSE for 3D voluntary force estimation with increasing EM iterations. Lower RMSE corresponds to better estimation performance. An EM iteration represents a single calibration using one-cycle data, comprising multiple EM steps to achieve convergence. Numbers 1–6 indicate the count of EM iterations. The testing data were an extra recorded one-cycle data. Gray bars in each sub<sup>fi</sup>gure on the left indicate the absolute change achieved by CA-GMM over Initial-GMM, where percentage values show the relative improvement and numerica values indicate the change.

![](images/2d43b8e11664bbb1b554a908cb37138b7a5397a48ecd3be59ab665d57de6d761.jpg)

![](images/66e86d0b6b388d8421bdce1cdc89f827ba452f0c88b049ef2fe6a59aca20fa0e.jpg)

![](images/68c179d099c45d2f964df4000b7e2944cf31639d7d9a47f7ec1156e911374ac0.jpg)

![](images/145440a9b2834b5a9afb8b0d19080f212823c337930151b10065546484136115.jpg)  
Figure 5. Trajectory-tracking results of a subject are obtained through EMG-based control utilizing Initial-GMM and CA-GMM, respectively, on the ULCDRR platform. The tracking outcomes are present in two scenarios: a) without muscle fatigue and b) with muscle fatigue.

![](images/98426a4a67ff06ce6814c89f6f332b6abeb56ef2e6f52a37eb4021b61fa58d51.jpg)

![](images/a1a6e0167a69532579fa47baa03a630c7210685b5ff19386d9b17f41707f492f.jpg)  
Figure 6. Average EMG level of eight-channel EMG signals. Each bar represents the average performance of all subjects under identical experimental conditions. Bars on a white background indicate the results before a fatigue-inducing experiment, and bars on a gray background indicate the results after.

Figure 7 presents the average performance for tracking accuracy (RMSE ), trajectory smoothness (RMSJ), and IF in EMGbased control. Detailed values are available in Table 2. Before the fatigue-inducing experiment, the RMSE results obtained using the control model based on Initial-GMM were lower than those based on CA-GMM. The normality test result $( p = 0 . 1 > 0 . 0 5 )$ shows that the data should be statistically analyzed using a paired t-test. The result indicates no significant difference in the RMSE values from the Initial-GMM and CA-GMM $( p = 0 . 0 6 5 5 > 0 . 0 5 )$ . Under the influence of muscle fatigue, the EMG-based control strategy based on Initial-GMM suffered performance degradation, with a 36.46% increase in the RMSE , a 10.19% increase in the RMSJ, and a 7.29% increase in the IF. During periods when subjects experienced muscle fatigue, utilizing CA-GMM compared to Initial-GMM shows a decrease of 23.66%, 8.17%, and 5.04% in RMSE , RMSJ, and IF. The normality test result of RMSE $( p = 0 . 0 2 5 1 < 0 . 0 5 )$

![](images/154e7eecb7491087d5b3c2890f00313dd71ae39489f67924812274d99ae6e6bc.jpg)  
Figure 7. Average RMSE , RMSJ, and IF in trajectory-tracking tasks. Each bar represents the average performance of all subjects under identical experimental conditions. Bars on a white background indicate the results before a fatigue-inducing experiment, and bars on a gray background indicate the results after. ns denotes no statistical difference, whereas \* denotes a statistical difference at $p < 0 . 0 0 1$

Table 2. Experimental results of trajectory-tracking experiments before and after the fatigue-inducing experiment.
<table><tr><td colspan="3">Before the fatigue-inducing experiment</td><td colspan="2">After the fatigue-inducing experiment</td></tr><tr><td>Model</td><td>Initial-GMM</td><td>CA-GMM</td><td>Initial-GMM</td><td>CA-GMM</td></tr><tr><td>RMSEx</td><td> $0 . 0 1 9 2 \pm 0 . 0 0 6 2$ </td><td> $0 . 0 2 7 5 \pm 0 . 0 0 7 0$ </td><td> $0 . 0 2 6 2 \pm 0 . 0 0 9 6$ </td><td>0.0200± 0.0070</td></tr><tr><td>RMSJ</td><td> $4 2 2 \pm 1 9 0 . 4 8$ </td><td> $4 3 6 \pm 1 8 2 . 4 5$ </td><td> $4 6 5 \pm 2 1 2 . 4 0$ </td><td> $4 2 7 \pm 2 0 7 . 7 4$ </td></tr><tr><td>IF</td><td> $2 0 . 1 6 \pm 3 . 8 6$ </td><td> $2 0 . 2 2 \pm 3 . 7 7$ </td><td> $2 1 . 6 3 \pm 4 . 2 7$ </td><td> $2 0 . 5 4 \pm 3 . 8 6$ </td></tr><tr><td>EMG level, BIC</td><td> $1 4 . 8 0 \pm 8 . 0 0$ </td><td> $1 5 . 3 7 \pm 9 . 2 6$ </td><td> $1 4 . 6 6 \pm 6 . 4 5$ </td><td> $1 4 . 0 7 \pm 6 . 7 9$ </td></tr><tr><td>EMG level, BR</td><td> $2 7 . 7 3 \pm 1 0 . 3 7$ </td><td> $2 7 . 3 5 \pm 1 0 . 9 6$ </td><td> $2 8 . 7 5 \pm 9 . 9 7$ </td><td> $2 8 . 7 4 \pm 9 . 7 1$ </td></tr><tr><td>EMG level, DA</td><td> $1 7 . 0 5 \pm 7 . 0 6$ </td><td> $1 7 . 2 4 \pm 6 . 7 7$ </td><td> $1 7 . 6 9 \pm 6 . 9 9$ </td><td> $1 7 . 6 5 \pm 7 . 0 8$ </td></tr><tr><td>EMG level, DM</td><td> $1 1 . 9 0 \pm 8 . 8 7$ </td><td> $1 1 . 5 4 \pm 8 . 3 2$ </td><td> $1 2 . 4 7 \pm 9 . 3 1$ </td><td> $1 1 . 8 7 \pm 8 . 9 8$ </td></tr><tr><td>EMG level, DP</td><td> $1 1 . 0 8 \pm 7 . 0 8$ </td><td> $1 0 . 5 6 \pm 6 . 6 5$ </td><td> $1 1 . 8 6 \pm 7 . 1 4$ </td><td> $1 0 . 8 1 \pm 6 . 2 6$ </td></tr><tr><td>EMG level, TRI-LA</td><td> $2 4 . 5 5 \pm 7 5 . 4 4$ </td><td> $2 4 . 5 7 \pm 1 5 . 3 0$ </td><td> $2 5 . 4 4 \pm 1 5 . 7 1$ </td><td> $2 4 . 2 5 \pm 1 5 . 0 0$ </td></tr><tr><td>EMG level, TRI-LO</td><td> $1 8 . 3 6 \pm 1 2 . 4 7$ </td><td> $1 7 . 5 3 \pm 1 1 . 4 9$ </td><td> $1 9 . 7 0 \pm 1 1 . 8 6$ </td><td> $1 6 . 9 6 \pm 1 0 . 7 9$ </td></tr><tr><td>EMG level, TRI-ME</td><td> $1 7 . 7 6 \pm 9 . 6 5$ </td><td> $1 7 . 3 6 \pm 9 . 4 8$ </td><td> $1 7 . 6 8 \pm 9 . 5 4$ </td><td> $1 7 . 4 2 \pm 9 . 2 0$ </td></tr></table>

indicates a statistical analysis using the paired Wilcoxon signedrank test. The analysis indicates a significant difference in the paired RMSE values obtained from the Initial-GMM and CA-GMM (p < 0.001).

## 6. Discussion

This study proposed an EMG-based control model employing GMM to estimate 3D continuous forces, wherein continuous intentions can facilitate participant-independent control tasks.<sup>[11]</sup> Chao et al. emphasized the importance of considering the coupling of dimensional components in their study on 3D finger force.<sup>[54]</sup> Compared with Hill-type model<sup>[18]</sup> and the method of parallel 1D force estimation,<sup>[15,24]</sup> GMM directly encoded the connection involving uncertainty between EMG and 3D forces by establishing the joint probability density distribution. Specifically, the cross-correlation coefficients of the GMM’s covariance matrices encoded the interconnections among the 3D components and the relationship between forces and EMG signals. Zhang et al. proposed a modeling assumption that the relationship between EMG and 3D force involved shared features and dimension-specific features.<sup>[16]</sup> Hogan et al. proposed that the Gaussian distribution can model the relationship between muscle contraction and myoelectric activity.<sup>[55]</sup> Each Gaussian component of the GMM can embody the picking of such a single event, and their weighted combination to form GMM can parallel the constitution of macroscopic muscle activity. GMM can flexibly deal with model learning of EMG–force relationships with varying complexity by adjusting the number of Gaussian components. Compared with Zhang et al.<sup>[16]</sup> GMM with multiple Gaussian components may be suitable for the estimation of 3D arm force using multichannel EMG signals.

The embedded computational model in intelligent EMGbased interfaces should consider computational cost and generalization. In contrast to kernel-based multioutput approaches such as GRNN<sup>[25]</sup> and multi-output support vector regression,<sup>[56]</sup> GMM can achieve a fast 3D voluntary force output by GMR to perform real-time EMG-based control without accessing source data during prediction.<sup>[31]</sup> Compared with models requiring specialized architectural design,<sup>[15,16,24,26,27]</sup> GMM only necessitates determining the number of Gaussian components. When it comes to the time-varying nature of EMG<sup>[42]</sup> and the increased force variability caused by muscle fatigue,<sup>[57]</sup> the variability in the EMG–force relationship encoded by GMM’s covariance can facilitate the probabilistic inference under the influence of environmental changes involving uncertainty. In this study, we also shared the covariance parameters of each component in GMM (i.e., shared covariance) to improve the generalization performance under the inductive bias.<sup>[58]</sup>

GMM combined with GMR can be regarded as a bioinspired technique because it emulates the processes of probabilistic modeling and reasoning performed by the human brain.<sup>[32,33]</sup> Complex human–robot interactions necessitate human involvement<sup>[15,59]</sup> and depend on the regulation based on the human central nervous system.<sup>[60]</sup> Bioinspired models can facilitate the control of the human central nervous system, improving the performance of EMG-based physical human– robot interaction.<sup>[22,61]</sup>

Owing to the fatigue-induced shift in the EMG–force relationship, the performance of EMG-based admittance control might have a degradation. We integrated the incremental GMM-based calibration algorithm into a dSPACE controller in response to the effect of muscle fatigue. As shown in Figure 4, the accuracy of arm force estimation using CA-GMM could surpass that of Initial-GMM after five EM iterations under muscle fatigue. The result suggests that EMG-based computational models may not achieve performance compensation if only a single calibration at a time. Incremental learning-based calibrations can facilitate model–subject matching by continuously accumulating experience, potentially improving robot-assisted rehabilitation with EMG-based interfaces. As shown in Figure 7, EMG-based control using CA-GMM improved the tracking accuracy under the influence of muscle fatigue, facilitating control compliance and reducing IF. The results indicated that the intelligent EMG-based human–robot interface was less affected when utilizing a suitable EMG decoder model. The improvement could be attributed to two main factors: first, inaccurate mean vectors directly reflect distribution shifts, and the calibration ofthe mean vectors allows the GMM and EMG–force relationship to be rematched. Second, the calibration of the covariance matrix accommodated the increased uncertainty of the EMG–force relationship caused by muscle fatigue. The EMG-based control using CA-GMM improved tracking accuracy in the presence of muscle fatigue, which did not require additional compliance loss and increased IFs. The reduced cost suggests that the proposed approach could improve the comfort and safety of EMG-based control systems, potentially facilitating sustained engagement in training among participants while maintaining consistent training intensity. Before the fatigue-inducing experiment, the EMG-based control using CA-GMM did not achieve higher tracking accuracy, potentially due to the inadequate estimation performance ofCA-GMM compared with that ofInitial-GMM in terms of the six EM iterations (shown in Figure 4). Nevertheless, the statistical result demonstrated similar tracking performance, which implicated human motion adaptation may cope with some differences in EMG-based control models.<sup>[62]</sup> However, such motion adaptability may prove insufficient in response to muscle fatigue during EMG-based pHRI, necessitating the proposed adaptive calibration via machine intelligence employing incremental GMM.

There are two limitations in this study. First, this study only involved healthy subjects. Our follow-up work will recruit suitable patient participants. Second, the proposed technique may have performance differences on different platforms, so further investigation is needed.

## 7. Conclusions

This article proposed an improved GMM to estimate the 3D voluntary force, performing the EMG-based admittance control on the ULCDRR platform. Experimental results demonstrate that the proposed EMG-based admittance control using GMM can improve control precision and compliance during physical human–robot interaction. The utilization of calibrated GMM reduced the effect of muscle fatigue during the EMG-based control. Therefore, the proposed approach holds potential applications in EMG-based robot-assisted rehabilitation and other fields that involve physical human–robot interaction. Future work can evaluate the performance of the proposed approach on different groups of participants.

## Appendix

## A1. A Formula Derivation in GMM

The maximum likelihood estimation for solving GMM encounters difficulties with partial derivatives; thus, the EM algorithm is typically employed.

$$
\pmb { \theta } ^ { ( g + 1 ) } = \arg \operatorname* { m a x } \underbrace { \mathbb { E } _ { z | \pmb { \xi } , \pmb { \theta } ^ { ( g ) } } [ \log \mathcal { P } ( \pmb { \xi } , z | \pmb { \theta } ) ] } _ { Q \left( \pmb { \theta } , \pmb { \theta } ^ { ( g ) } \right) }\tag{A1}
$$

where $Q \big ( \pmb \theta , \pmb \theta ^ { ( g ) } \big )$ is the objective function of the EM algorithm and $z \in \{ z _ { m } = j | j = 1 , \ \cdot \ \cdot \ , K \}$ is a latent variable.

$$
\begin{array} { r l } & { \displaystyle { \mathcal { Q } \big ( \pmb { \theta } , \pmb { \theta } ^ { ( g ) } \big ) = \int _ { Z } \log \mathcal { P } ( \pmb { \xi } , Z | \pmb { \theta } ) \times \mathcal { P } \big ( Z | \pmb { \xi } , \pmb { \theta } ^ { ( g ) } \big ) d Z } } \\ & { \displaystyle { = \sum _ { z _ { 1 } , \dots , z _ { n } } \Bigg \{ \Bigg [ \sum _ { m = 1 } ^ { N } \log \mathcal { P } ( \pmb { \xi } _ { m } , z _ { m } | \pmb { \theta } ) \Bigg ] \prod _ { m = 1 } ^ { N } \mathcal { P } \big ( z _ { m } | \pmb { \xi } _ { m } , \pmb { \theta } ^ { ( g ) } \big ) \Bigg \} } } \\ & { \displaystyle { = \sum _ { m = 1 } ^ { N } \sum _ { z _ { m } = 1 } ^ { K } \log \mathcal { P } ( \pmb { \xi } _ { m } , z _ { m } | \pmb { \theta } ) \times \mathcal { P } \big ( z _ { m } | \pmb { \xi } _ { m } , \pmb { \theta } ^ { ( g ) } \big ) } } \end{array}\tag{A2}
$$

According to the Bayesian rule, we have

$$
\mathcal { P } ( \pmb { \xi } , z | \pmb { \theta } ) = \mathcal { P } ( z | \pmb { \theta } ) \mathcal { P } ( \pmb { \xi } | z , \pmb { \theta } )\tag{A3}
$$

$$
\mathcal { P } ( z | \xi , \pmb { \theta } ^ { ( g ) } ) = \frac { \mathcal { P } ( \xi , z | \pmb { \theta } ) } { \mathcal { P } \left( \xi | \pmb { \theta } ^ { ( g ) } \right) }\tag{A4}
$$

Substituting (A3) and (A4) into (A2), we can derive

$$
\begin{array} { r l } & { \displaystyle { \mathcal Q } \big ( \pmb { \theta } , \pmb { \theta } ^ { ( g ) } \big ) = \sum _ { j = 1 } ^ { K } \sum _ { m = 1 } ^ { N } \log \mathcal P \big ( \pmb { \xi } _ { m } , \pmb { \theta } _ { j } \big ) \times \mathcal P \Big ( \pmb { \theta } _ { j } ^ { ( g ) } | \pmb { \xi } _ { m } \Big ) } \\ & { \displaystyle = \sum _ { j = 1 } ^ { K } \sum _ { m = 1 } ^ { N } \log \big ( \pi _ { j } \mathcal N \big ( \pmb { \xi } _ { m } | \pmb { \mu } _ { j } , \pmb { \Sigma } _ { j } \big ) \big ) \times \frac { \pi _ { j } ^ { ( g ) } \mathcal N \Big ( \pmb { \xi } _ { m } | \pmb { \mu } _ { j } ^ { ( g ) } , \pmb { \Sigma } _ { j } ^ { ( g ) } \Big ) } { \sum _ { i = 1 } ^ { K } \pi _ { i } ^ { ( g ) } \mathcal N \Big ( \pmb { \xi } _ { m } | \pmb { \mu } _ { i } ^ { ( g ) } , \pmb { \Sigma } _ { i } ^ { ( g ) } \Big ) } } \end{array}\tag{A5}
$$

To solve the partial derivative of $\pi _ { j } { \mathrm { : } }$ , we can rewrite (A5) as

$$
G ( \pi , \lambda ) = \sum _ { j = 1 } ^ { K } \sum _ { m = 1 } ^ { N } \log \pi _ { j } \times \mathcal { P } \Big ( \pmb { \theta } _ { j } ^ { ( g ) } | \pmb { \xi } _ { m } \Big ) + \lambda \Bigg ( \sum _ { j = 1 } ^ { K } \pi _ { j } - 1 \Bigg )\tag{A6}
$$

where λ is the Lagrange multiplier. To solve the partial derivative of $\pmb { \mu } _ { j }$ and $\Sigma _ { j } ,$ we can simplify (A5) to

$$
G ( \pmb { \mu } , \pmb { \Sigma } ) = \sum _ { j = 1 } ^ { K } \sum _ { m = 1 } ^ { N } \log \mathcal { N } \big ( \pmb { \xi } _ { m } | \pmb { \mu } _ { j } , \pmb { \Sigma } _ { j } \big ) \times \mathcal { P } \Big ( \pmb { \theta } _ { j } ^ { ( g ) } | \pmb { \xi } _ { m } \Big )\tag{A7}
$$

By calculating the partial derivatives in formulas (A6) and (A7), we can obtain the iteration formula for the M-step in the EM algorithm.

$$
\left\{ \begin{array} { l } { \displaystyle \pi _ { j } ^ { ( g + 1 ) } = \frac { 1 } { N } \sum _ { m = 1 } ^ { N } \mathcal { P } \Big ( \theta _ { j } ^ { ( g ) } | \xi _ { m } \Big ) } \\ { \displaystyle \mu _ { j } ^ { ( g + 1 ) } = \frac { \sum _ { m = 1 } ^ { N } \mathcal { P } \Big ( \theta _ { j } ^ { ( g ) } | \xi _ { m } \Big ) \xi _ { m } } { \sum _ { m = 1 } ^ { N } \mathcal { P } \Big ( \theta _ { j } ^ { ( g ) } | \xi _ { m } \Big ) } } \\ { \displaystyle \sum _ { j } ^ { ( g + 1 ) } = \frac { \sum _ { m = 1 } ^ { N } \mathcal { P } \Big ( \theta _ { j } ^ { ( g ) } | \xi _ { m } \Big ) \Big ( \xi _ { m } - \mu _ { j } ^ { ( g + 1 ) } \Big ) \Big ( \xi _ { m } - \mu _ { j } ^ { ( g + 1 ) } \Big ) ^ { T } } { \sum _ { m = 1 } ^ { N } \mathcal { P } \Big ( \theta _ { j } ^ { ( g ) } | \xi _ { m } \Big ) } } \end{array} \right.\tag{A8}
$$

Thus, the E-step of GMM only needs to calculate

$$
\mathcal { P } \Big ( \pmb { \theta } _ { j } ^ { ( g ) } | \pmb { \xi } _ { m } \Big ) = \frac { \pi _ { j } ^ { ( g ) } \mathcal { N } \Big ( \pmb { \xi } _ { m } | \pmb { \mu } _ { j } ^ { ( g ) } , \pmb { \Sigma } _ { j } ^ { ( g ) } \Big ) } { \sum _ { i = 1 } ^ { K } \pi _ { i } ^ { ( g ) } \mathcal { N } \Big ( \pmb { \xi } _ { m } | \pmb { \mu } _ { i } ^ { ( g ) } , \pmb { \Sigma } _ { i } ^ { ( g ) } \Big ) } = \gamma _ { m , j } ^ { ( g + 1 ) }\tag{A9}
$$

and $\begin{array} { r } { E _ { j } ^ { ( g + 1 ) } = \sum _ { m = 1 } ^ { N } \gamma _ { m , j } ^ { ( g + 1 ) } } \end{array}$ instead of the entire Q function.

The outcome of the incremental GMM algorithm was provided in the study.<sup>[46]</sup> The derivation trick ofdirect method based on (A8) is to separate the part that related to the initial dataset $\left\{ \pmb { \xi } _ { m } \right\} _ { m = 1 } ^ { N _ { 0 } }$ from the part that corresponding to the current sampled dataset $\{ \pmb { \xi } _ { m } \} _ { m = 1 } ^ { N _ { n } }$ . Hence, we have

$$
\pi _ { j } ^ { ( g + 1 ) } = \frac { \sum _ { m = 1 } ^ { N _ { 0 } } \gamma _ { m , j } ^ { ( 0 ) } + \sum _ { m = N _ { 0 } + 1 } ^ { N _ { 0 } + N _ { n } } \gamma _ { m , j } ^ { ( g + 1 ) } } { N _ { 0 } + N _ { n } } = \frac { E _ { j } ^ { ( 0 ) } + E _ { j } ^ { ( g + 1 ) } } { N _ { 0 } + N _ { n } }\tag{A10}
$$

where $\gamma _ { m , j } ^ { ( 0 ) } = \mathcal { P } \Big ( \pmb { \theta } _ { j } ^ { ( 0 ) } | \pmb { \xi } _ { m } \Big )$ and $\begin{array} { r } { E _ { j } ^ { ( 0 ) } = \sum _ { m = 1 } ^ { N _ { 0 } } \gamma _ { m , j } ^ { ( 0 ) } } \end{array}$ . The derivation of $\pmb { \mu } _ { j } ^ { ( g + 1 ) }$ and $\pmb { \Sigma } _ { j } ^ { ( g + 1 ) }$ is

$$
\begin{array} { c } { { \pmb { \mu } _ { j } ^ { ( g + 1 ) } = \frac { \sum _ { m = 1 } ^ { N _ { 0 } } \gamma _ { m , j } ^ { ( 0 ) } \pmb { \xi } _ { m } + \sum _ { m = N _ { 0 } + 1 } ^ { N _ { 0 } + N _ { n } } \gamma _ { m , j } ^ { ( g + 1 ) } \pmb { \xi } _ { m } } { \sum _ { m = 1 } ^ { N _ { 0 } } \gamma _ { m , j } ^ { ( 0 ) } + \sum _ { m = N _ { 0 } + 1 } ^ { N _ { 0 } + N _ { n } } \gamma _ { m , j } ^ { ( g + 1 ) } } } } \\ { { = \frac { E _ { j } ^ { ( 0 ) } \pmb { \mu } _ { j } ^ { ( 0 ) } + \sum _ { m = 1 } ^ { N _ { n } } \gamma _ { m , j } ^ { ( g + 1 ) } \pmb { \xi } _ { m } } { E _ { j } ^ { ( 0 ) } + E _ { j } ^ { ( g + 1 ) } } } } \end{array}\tag{A11}
$$

and

$$
\begin{array} { r l } & { \Sigma _ { j } ^ { ( g + 1 ) } = \frac { \sum _ { m = 1 } ^ { N _ { 0 } } \gamma _ { m , j } ^ { ( 0 ) } \left( \xi _ { m } - \mu _ { j } ^ { ( g + 1 ) } \right) \left( \xi _ { m } - \mu _ { j } ^ { ( g + 1 ) } \right) ^ { T } } { \sum _ { m = 1 } ^ { N _ { 0 } } \gamma _ { m , j } ^ { ( 0 ) } + \sum _ { m = N _ { 0 } + 1 } ^ { N _ { 0 } + N _ { n } } \gamma _ { m , j } ^ { ( g + 1 ) } } } \\ & { \qquad + \frac { \sum _ { m = N _ { 0 } + 1 } ^ { N _ { 0 } + N _ { n } } \gamma _ { m , j } ^ { ( g + 1 ) } \left( \xi _ { m } - \mu _ { j } ^ { ( g + 1 ) } \right) \left( \xi _ { m } - \mu _ { j } ^ { ( g + 1 ) } \right) ^ { T } } { \sum _ { m = 1 } ^ { N _ { 0 } } \gamma _ { m , j } ^ { ( 0 ) } + \sum _ { m = N _ { 0 } + 1 } ^ { N _ { 0 } + N _ { n } } \gamma _ { m , j } ^ { ( g + 1 ) } } } \end{array}\tag{A12}
$$

The numerator of the first fractional term in (A12) can be further derived as

© 2024 The Author(s). Advanced Intelligent Systems published by Wiley-VCH GmbH

$$
\begin{array} { r l } & { \frac { \partial } { \partial x } \frac { \partial } { \partial y } ( x ^ { 2 } - y ^ { 2 } ) } \\ & { = - \frac { \partial } { \partial x } \frac { \partial } { \partial x } ( x ^ { 2 } - y ^ { 2 } ) \{ ( k _ { x } - \frac { p _ { x } ^ { 2 } + 1 } { k _ { y } ^ { 2 } } ) \{ ( k _ { x } - \frac { p _ { x } ^ { 2 } + 1 } { k _ { y } ^ { 2 } } ) ^ { \frac { 1 } { 2 } } \} \Gamma } \\ & { - \frac { \partial } { \partial x } \frac { \partial } { \partial x } ( x ^ { 2 } - y ^ { 2 } ) \{ ( k _ { x } - \frac { p _ { x } ^ { 2 } + 1 } { k _ { y } ^ { 2 } } ) ^ { \frac { 1 } { 2 } } \} \Gamma } \\ & { - \frac { \partial } { \partial x } \frac { \partial } { \partial x } ( x ^ { 2 } - y ^ { 2 } ) \{ ( k _ { x } ^ { 2 } - k _ { y } ^ { 2 } ) ^ { \frac { 1 } { 2 } } \} \Gamma } \\ & { + \frac { \partial } { \partial x } \frac { \partial } { \partial x } \frac { \partial } { \partial x } \{ ( k _ { x } ^ { 2 } + 1 ) ^ { \frac { 1 } { 2 } } \} \Gamma } \\ & { - \frac { \partial } { \partial y } ( x ^ { 2 } - y ^ { 2 } ) \frac { \partial } { \partial x } ( x ^ { 2 } - y ^ { 2 } ) \frac { \partial } { \partial x } \frac { \partial } { \partial x } \rho _ { x ^ { 2 } } ( x ^ { 2 } - y ^ { 2 } ) \{ ( k _ { x } ^ { 2 } - k _ { y } ^ { 2 } ) ^ { \frac { 1 } { 2 } } \} \Gamma } \\ & { - \frac { \partial } { \partial x } \frac { \partial } { \partial x } ( x ^ { 2 } - y ^ { 2 } ) \{ ( k _ { x } - \frac { p _ { x } ^ { 2 } + 1 } { k _ { y } ^ { 2 } } ) ^ { \frac { 1 } { 2 } } \} \Gamma } \\  \end{array}\tag{A13}
$$

Then, we can substitute (A13) into (A12) and obtain

$$
\begin{array} { r l } & { \Sigma _ { j } ^ { ( g + 1 ) } = \frac { E _ { j } ^ { ( 0 ) } \left( \Sigma _ { j } ^ { ( 0 ) } + \left( \mu _ { j } ^ { ( 0 ) } - \mu _ { j } ^ { ( g + 1 ) } \right) \left( \mu _ { j } ^ { ( 0 ) } - \mu _ { j } ^ { ( g + 1 ) } \right) ^ { T } \right) } { E _ { j } ^ { ( 0 ) } + E _ { j } ^ { ( g + 1 ) } } } \\ & { \quad + \frac { \sum _ { m = 1 } ^ { N _ { n } } \mathcal { P } \left( \theta _ { j } ^ { ( g ) } | \xi _ { m } \right) \left( \xi _ { m } - \mu _ { j } ^ { ( g + 1 ) } \right) \left( \xi _ { m } - \mu _ { j } ^ { ( g + 1 ) } \right) ^ { T } } { E _ { j } ^ { ( 0 ) } + E _ { j } ^ { ( g + 1 ) } } } \end{array}\tag{A14}
$$

## Acknowledgements

This research was supported in part by the Shenzhen Science and Technology Research Program (No. SGDX20210823103405040 and No. JCYJ20220818101602005), in part by the ANSO Collaborative Research Project (No. ANSO-CR-PP-2020-03), and in part by the Shenzhen Medical Research Fund (No. C2301002).

## Con<sup>fl</sup>ict of Interest

The authors declare no con<sup>fl</sup>ict of interest.

## Author Contributions

Jianlin Zheng: Investigation (lead); Methodology (lead); Visualization (lead); Writing—original draft (lead). Zhijun Li: Conceptualization (supporting); Writing—review and editing (supporting). Ting Xu: Investigation (supporting); Methodology (supporting); Writing—review and editing (supporting). Rong Song: Conceptualization (lead); Funding acquisition (lead); Resources (lead); Supervision (lead); Writing—review and editing (lead).

## Data Availability Statement

Research data are not shared.

## Keywords

cable-driven robot, electromyographic-based control, Gaussian mixture model, robot-assisted rehabilitation

Received: June 19, 2024

Revised: July 30, 2024

Published online: September 19, 2024

[1] H. Yang, J. Wan, Y. Jin, X. Yu, Y. Fang, IEEE Sens.J. 2022, 22, 23649.

[2] T. B. Wyller, J. Holmen, P. Laake, K. Laake, Stroke 1998, 29, 363.

[3] P. Maciejasz, J. Eschweiler, K. Gerlach-Hahn, A. Jansen-Troy, S. Leonhardt, J. NeuroEng. Rehabil. 2014, 11, 3.

[4] H. I. Krebs, M. Ferraro, S. P. Buerger, M. J. Newbery, A. Makiyama, M. Sandmann, D. Lynch, B. T. Volpe, N. Hogan,J. NeuroEng. Rehabil. 2004, 1, 5.

[5] P. S. Lum, C. G. Burgar, P. C. Shor, M. Majmundar, M. Van der Loos, Arch. Phys. Med. Rehabil. 2002, 83, 952.

[6] R. Riener, M. Guidali, U. Keller, A. Duschau-Wicke, V. Klamroth, T. Nef, Top. Spinal Cord Inj. Rehabil. 2011, 17, 54.

[7] Z. Li, K. Zhao, L. Zhang, X. Wu, T. Zhang, Q. Li, X. Li, C.-Y. Su, IEEE/ ASME Trans. Mechatron. 2021, 26, 2700.

[8] N. Stergiou, R. Harbourne, J. Cavanaugh,J. Neurol. Phys. Ther. 2006, 30, 120.

[9] N. Hogan, J. Dyn. Syst. Meas. Control. 1985, 107, 1.

[10] D. Xiong, D. Zhang, X. Zhao, Y. Zhao, IEEE/CAAJ. Autom. Sin. 2021, 8, 512.

[11] J. Xue, K. W. C. Lai, Adv. Intell. Syst. 2024, 6, 2300318.

[12] Y. Zhuang, S. Yao, C. Ma, R. Song, IEEE Trans. Ind. Inform. 2019, 15, 1211.

[13] X. Chen, D. Zhang, X. Zhu, J. NeuroEng. Rehabil. 2013, 10, 44.

[14] G. R. Naik, D. K. Kumar, IEEE T. Inf. Technol. Biomed 2010, 14, 301.

[15] H. Su, W. Qi, Z. Li, Z. Chen, G. Ferrigno, E. De Momi, IEEE Trans. Artif. Intell. 2021, 2, 404.

[16] T. Zhang, H. Chu, Y. Zou, H. Sun, Expert Syst. Appl. 2024, 238, 122003.

[17] Y. Zeng, J. Yang, C. Peng, Y. Yin, IEEE Trans. Biomed. Eng. 2019, 66, 2556.

[18] N. Louis, P. Gorce, Med. Biol. Eng. Comput. 2009, 47, 1173.

[19] B. Xiong, N. Zeng, H. Li, Y. Yang, Y. Li, M. Huang, W. Shi, M. Du, Y. Zhang, IEEE Access 2019, 7, 29973.

[20] B. Xiong, Y. Chang, N. Zeng, Y. Niu, H. Zeng, H. Li, G. Mao, Y. Xu, IEEE Trans. Instrum. Meas. 2024, 73, 2516011.

[21] C. Xie, Q. Yang, Y. Huang, S. W. Su, T. Xu, R. Song, IEEE Trans. Biomed. Circuits Syst. 2021, 15, 1332.

[22] Y. Liu, Z. Li, H. Liu, Z. Kan, B. Xu, IEEE Trans. Ind. Inform. 2020, 16, 4308.

[23] P. K. Artemiadis, K. J. Kyriakopoulos, IEEE Trans. Rob. 2010, 26, 393.

[24] T. Zhang, H. Sun, Y. Zou, H. Chu, J. Manuf. Syst. 2022, 64, 330.

[25] C. Wu, H. Zeng, A. Song, Front. Neurosci. 2017, 11, 251824.

[26] Y. Huang, R. Song, A. Argha, A. V. Savkin, B. G. Celler, S. W. Su, IEEE Trans. Neural Syst. Rehabil. Eng. 2020, 28, 277.

[27] C. Xie, Z. Li, R. Song, IEEE Trans. Ind. Electron. 2024, 1, https://doi. org/10.1109/TIE.2024.3398675.

[28] D. Farina, N. Jiang, H. Rehbaum, A. Holobar, B. Graimann, H. Dietl, O. C. Aszmann, IEEE Trans. Neural Syst. Rehabil. Eng. 2014, 22, 797.

[29] R. Roy, Y. Zheng, D. G. Kamper, X. Hu, IEEE Trans. Biomed. Eng. 2023, 70, 1911.

[30] G. Rau, C. Disselhorst-Klug, R. Schmidt, J. Biomech. 2000, 33, 1207.

[31] S. Calinon, IEEE Rob. Autom. Mag. 2020, 27, 33.

[32] T. L. Grif<sup>fi</sup>ths, N. Chater, C. Kemp, A. Perfors, J. B. Tenenbaum, Trends Cognit. Sci. 2010, 14, 357.

[33] A. Pouget, J. M. Beck, W. J. Ma, P. E. Latham, Nat. Neurosci. 2013, 16, 1170.

[34] Y. Huang, K. Englehart, B. Hudgins, A. Chan, IEEE Trans. Biomed. Eng. 2005, 52, 1801.

[35] Q. Ding, J. Han, X. Zhao, Y. Chen, IEEE Trans. Ind. Electron. 2015, 62, 4994.

[36] J. Zheng, Z. Li, L. Huang, Y. Gao, B. Wang, M. Peng, Y. Wang, IEEE Sens. J. 2022, 22, 19556.

[37] M. Maaref, A. Rezazadeh, K. Shamaei, M. Tavakoli, IEEEJ. Sel. Top Signal Process. 2016, 10, 904.

[38] R. J. Escarabajal, J. L. Pulloquinga, P. Zamora-Ortiz, Á Valera, V. Mata, M. Vallés, IEEE Rob. Autom. Lett. 2023, 8, 4283.

[39] E. L. Glader, B. Stegmayr, K. Asplund, Stroke 2002, 33, 1327.

[40] K. Gui, H. Liu, D. Zhang, IEEE/ASME Trans. Mechatron. 2019, 24, 483.

[41] I. Kyranou, S. Vijayakumar, M. S. Erden, Front. Neurorob. 2018, 12, 58.

[42] L. Xu, X. Chen, S. Cao, X. Zhang, X. Chen, IEEE Trans. Neural Syst. Rehabil. Eng. 2018, 26, 2153.

[43] Z. Q. Tang, H. L. Heung, X. Q. Shi, K. Y. Tong, Z. Li, IEEE Trans. Biomed. Eng. 2022, 69, 1016.

[44] X. Li, Q. Yang, R. Song, IEEE Trans. Biomed. Eng. 2021, 68, 1351.

[45] A. K. Tanwani, S. Calinon, IEEE Rob. Autom. Lett. 2016, 1, 235.

[46] S. Calinon, A. Billard, 2007 2nd ACM/IEEE International Conference on Human-Robot Interaction (HRI). Arlington, VA 2007, pp. 255–262.

[47] M. Deng, Z. Li, Y. Kang, C. L. P. Chen, X. Chu, IEEE Trans. Cybern. 2020, 50, 112.

[48] Y. Li, S. S. Ge, IEEE/ASME Trans. Mechatron. 2014, 19, 1007.

[49] C. Xie, Y. Lyu, G. Li, R. K.-Y. Tong, H. Xia, R. Song, Z. Li, IEEE Trans. Rob. 2024, 40, 3199.

[50] Z. Yang, Y. Chen, Z. Tang, J. Wang, Neurocomputing 2016, 207, 568.

[51] G. Hajian, A. Etemad, E. Morin, Biomed. Signal Process. Control. 2021, 70, 103012.

[52] D. Tkach, H. Huang, T. A. Kuiken,J. NeuroEng. Rehabil. 2010, 7, 21.

[53] Y. Na, J. Kim, IEEE Trans. Neural Syst. Rehabil. Eng. 2017, 25, 1431.

[54] E. Chao, J. Opgrande, F. Axmear, J. Biomech. 1976, 9, 387.

[55] N. Hogan, R. W. Mann, IEEE Trans. Biomed. Eng. 1980, BME-27, 382.

[56] S. Xu, X. An, X. Qiao, L. Zhu, L. Li, Pattern Recognit. Lett. 2013, 34, 1078.

[57] O. Missenard, D. Mottet, S. Perrey, Muscle Nerve. 2009, 40, 1019.

[58] A. Goyal, Y. Bengio, Proc. R. Soc. A 2022, 478, 20210068.

[59] H. Oliff, Y. Liu, M. Kumar, M. Williams, IEEE Trans. Autom. Sci. Eng. 2020, 17, 1597.

[60] D. P. Losey, C. G. McDonald, E. Battaglia, M. K. O’Malley, ASME Appl. Mech. Rev. 2018, 70, 010804.

[61] M. S. Madhav, N. J. Cowan, Annu. Rev. Control Rob. Autonom. Syst. 2020, 3, 243.

[62] J. He, D. Zhang, N. Jiang, X. Sheng, D. Farina, X. Zhu, J. Neural Eng. 2015, 12, 046005.