# Continuous Motion Intention Prediction Using sEMG for Upper-Limb Rehabilitation: A Systematic Review of Model-Based and Model-Free Approaches

Zijun Wei, Zhi-Qiang Zhang, Member, IEEE, Sheng Quan Xie, Fellow, IEEE

Abstract—Upper limb functional impairments persisting after stroke significantly affect patients' quality of life. Precise adjustment of robotic assistance levels based on patients' motion intention using sEMG signals is crucial for active rehabilitation. This paper systematically reviews studies over the past decade on continuous prediction of upper limb single joint and multi-joint combinations motion intention using Model-Based (MB) and Model-Free (MF) approaches, based on 186 relevant studies screened from six major electronic databases. The findings indicate ongoing challenges in terms of subject composition, algorithm robustness and generalization, and feasibility for practical applications. Moreover, it suggests integrating the strengths of both MB and MF approaches to improve existing algorithms. Therefore, future research should further explore personalized MB-MF combination methods incorporating deep learning, attention mechanisms, muscle synergy features, motor unit features, and closed-loop feedback to achieve precise, real-time, and long-duration prediction of multi-joint complex movements, while further refining the transfer learning strategy for rapid algorithm deployment across days and subjects. In summary, this review summarizes the current state and challenges of research, aiming to provide inspiration for future research on predicting upper limb motion intention based on sEMG.

Index Terms—Surface electromyography (sEMG), upperlimb rehabilitation, musculoskeletal model, deep learning, muscle synergy, motor unit, continuous joint kinematics and dynamics estimation methods, systematic review.

Manuscript received 14 December 2023;

Zijun Wei, Zhi-Qiang Zhang, and Sheng Quan Xie are with the Institute of Robotics, Autonomous System and Sensing, School of Electrical and Electronic Engineering, University of Leeds, LS2 9JT Leeds, U.K. (e-mail: elzw@leeds.ac.uk; z.zhang3@leeds.ac.uk; s.q.xie@leeds.ac.uk).

## I. INTRODUCTION

Stroke ranks as the second leading cause of death and disability worldwide, with increasing incidence and mortality rates each year. As reported in [1], approximately 60% of patients worldwide still exhibit upper limb functional impairments after stroke rehabilitation. This severely impedes their ability to perform basic Activities of Daily Living (ADL), significantly affecting their quality of life. Hence, expedited upper limb rehabilitation is of paramount importance. Studies on rehabilitation robotics-assisted therapy [2] have shown that passive movements assisted by rehabilitation robots alone do not improve patients’ motor function. Instead, the training intensity and the patient's active participation are crucial factors in improving rehabilitation outcomes, rather than the mere use of the robot itself. This aligns with theories of neural plasticity, where active patient participation is crucial for inducing neural plasticity and improving rehabilitation efficiency. Furthermore, research [3] has also indicated that the motor function improvements provided by current rehabilitation robots are limited, highlighting the need for more effective assist-asneeded (AAN) control strategies beyond traditional impedance and admittance control, aiming to maximize training intensity while ensuring active patient participation. Thus, precisely adjusting the robotic assistance level according to the patients motion intention is pivotal for achieving active rehabilitation.

Studies [4], [5] and [6] categorized patients’ motion intention into biological and non-biological signals. Biological signals include Electroencephalography (EEG), Electromyography (EMG), Force Myography (FMG), and Mechanomyography (MMG), while non-biological signals consist of video, Inertial Measurement Units (IMU), and force sensors. However, nonbiological signals have inherent time delays and cannot predict motion intention when the patient's limb is static, which is not friendly for stroke and amputee patients. In contrast, EMG, besides having the Electromechanical Delay (EMD) of 50- 100ms that reduces prediction latency, is more stable and less susceptible to interference in practical applications than EEG. Moreover, EMG sensors are more portable and easier to wear than EEG. Therefore, predicting patients' motion intentions using surface EMG (sEMG) is a highly promising approach.

Current sEMG-based motion intention prediction research can be categorized into discrete classification and continuous regression. However, as discussed in the review [6], only 11.6% of studies from 1996-2017 focused on continuous regression, and the first review on sEMG-based continuous motion intention estimation was not published until 2019 [4]. Moreover, according to the prediction methods used in previous studies, sEMG-based continuous regression can be divided into modelbased (MB) and model-free (MF) approaches. Hence, given the current research landscape and the fact that rehabilitation robots are continuously controlled, this review will focus on studies using MB and MF methods for continuous upper limb motion intention prediction (i.e., joint kinematics and dynamics).

As illustrated in Figure 1, MF approaches mainly encompass both Machine Learning (ML) and Deep Learning (DL) methods. The difference between ML and DL is that ML requires manual feature extraction and selection from preprocessed sEMG signals. In contrast, DL can automatically extract advanced features from sEMG and utilize the neural network's potent fitting capacity to approximate the highly nonlinear relationship between features and motion intention, thereby avoiding the reliance on optimal feature sets based on empiricism similar to ML [5]. Although end-to-end MF methods are convenient to train and quick to deploy, their inherently 'black box' nature may overlook the potential causal relationships between input and output data, consequently struggling to generalize beyond the training data and risking overfitting [6], [7], [8]. In contrast, MB methods with inherent physiological causality can convert sEMG signals to muscle-tendon forces according to the neuralphysiological mechanisms of muscle activation and contraction dynamics, before predicting joint kinematics and dynamics from joint torques using musculoskeletal (MSK) geometry and Newtonian motion equations [6]. Additionally, most MB studies employed Hill-type MSK models with series elastic (SE), contractile (CE), parallel elastic (PE), and viscoelastic (VE) components. However, due to physiological differences among patients, the generic proportional Hill model derived from extensive cadaveric specimens can lead to significant prediction errors. Therefore, parameter optimization for subject-specific MSK models in the final stage of MB methods is necessary to achieve precise predictions [9], [10].

In summary, the main contributions of this review towards achieving more effective active rehabilitation are threefold:

1) Comprehensively collect and screen all MB and MF studies in the past decade from six major databases and illustrate the current research landscape.

2) Review and identify the research results and corresponding limitations of MB and MF studies based on different single joint and multi-joint combinations.

3) Conclude and analyze the key findings, challenges, and opportunities from MB and MF studies to determine future research directions.

The remainder of this paper is structured as follows: Section II introduces the literature collection methodology and selection criteria, summarizing the screening results. Section III provides a comprehensive review of MB and MF studies based on different single joint and multi-joint combinations catalogs. Section IV discusses the key findings, challenges, and future research directions identified from MB and MF studies over the years. Finally, Section V concludes this review.

![](images/b91852fb93b87784b81495a400799ed7065bd26c9b9d63925253212080116717.jpg)  
Fig. 1. Continuous motion intention prediction process for Model-Free and Model-Based approaches.

## II. METHODS

## A. Search strategy

To conduct a comprehensive systematic review of the studies based on MB and MF methods, this review initially employed search keywords to query six major electronic databases, namely PubMed, Web of Science, Scopus, IEEE Xplore, ScienceDirect, and SpringerLink. It was observed that the volume of relevant studies began to increase from 2012. Therefore, the review's search span was set to cover the past decade, collecting all pertinent articles published from January 2013 to June 2023 within these databases, following the PRISMA guidelines. Additionally, pertinent literature from Google Scholar was also selectively included to ensure a systematic and exhaustive search outcome. The following keywords were utilized for the literature search:

((EMG OR sEMG) AND (Continuous) AND (Prediction OR Estimation) AND (Shoulder OR Elbow OR Wrist OR Hand OR Finger OR Upper-Limb))

## B. Inclusion and exclusion criteria

Initially, since this review exclusively focused on studies that involve continuous regression instead of discrete classification, research on pattern recognition, classification, and piecewise discretization were excluded from consideration. Additionally, only articles published in English and accessible to the author were considered for inclusion.

Subsequently, this review concentrated solely on MB and MF studies driven by sEMG or EMG signals, thereby excluding studies that exclusively utilize other motor intention signals for prediction without sEMG or EMG signals. Excluded signal sources encompass non-biological signals, such as IMU, Kinect cameras, Electrical Impedance Tomography (EIT), ultrasonic sensors, and Force-Sensitive Resistors (FSR), as well as other biological signals like EEG, functional Magnetic Resonance Imaging (fMRI), and Near-Infrared Spectroscopy (NIRS).

Finally, this review was dedicated to upper limb joints, thereby excluding all other joints, such as lower limb joints. Additionally, the MSK models discussed in reviewed studies were restricted to macroscopic physiological models of muscles, tendons, and bones, excluding, for instance, the finite element model based on the stress-strain relationships amongst muscle tissues. Furthermore, the review of MSK models was limited to human models without addressing other biological species.

## C. Study Selection results

As depicted in Figure 2(a), 674 relevant publications were identified using the specified search keywords in the selected databases, which included 33 supplemental publications from Google Scholar. Following the application of inclusion and exclusion criteria, 186 publications were ultimately included in this review, with 36 and 150 studies based on MB and MF approaches, respectively.

Figures 2(b) and 2(c) present the synthesis and categorization of the 186 selected publications. Figure 2(b) indicates a gradual increase in the number of publications related to continuous prediction of upper limb motion intention from 2013 to 2020, followed by an exponential growth beginning in 2021. This trend suggests that, unlike discrete classification in pattern recognition, the continuous prediction of upper limb motion intentions is an emerging research field that has garnered extensive attention in the past three years. Figure 2(c) compares the current research status of each upper limb single joint and multi-joint combinations, revealing that most studies focus on the elbow, wrist, and hand joints, as well as the hand-wrist and elbow-shoulder combinations. In contrast, there are only a few studies based on the MF approaches concerning the remaining shoulder joint, wrist-elbow joint combinations, and the entire upper limb, with an absence of studies based on MB methods.

![](images/b5361e3694f7a0fd28a43854c11d107edeeb8a9e90689833ce9541b6dcf71e38.jpg)  
Fig. 2. (a) Flowchart of the study selection process based on Prisma strategy. (b) Specific number of publications per year for both MB and MF approaches over a decade. (c) Detailed number of MB and MF studies for single joint and multi-joint combinations.

## III. RESULTS

Before conducting a comprehensive review of MB and MF studies based on each single joint and multi-joint combinations, Figure 3 below illustrates the distribution of methodologies applied in both MB and MF research, as well as the percentages of offline/online prediction, subjects/databases, and different subject attributes in these studies. It is observed that approaches in MB studies can be classified into six categories, in addition to the Hill model, the integrative approaches combining MB with MF are also of particular interest. Regarding the MF research encompassing eleven approaches, besides the highly regarded traditional neural network and DL model, approaches based on attention mechanisms, muscle synergy (MS) features, and motor unit (MU) neural features have also shown promising research potential, meriting further in-depth investigation.

A further subdivision within the predominant categories of traditional neural network and DL model is warranted to provide a comprehensive review of studies employing the MF approach. The collected 37 studies based on traditional neural networks can be further categorized into three categories: Feedforward Neural Networks (FNNs: 25 studies), Traditional Recurrent Neural Networks (T-RNNs: 11 studies), and Spiking Neural Network (SNN: 1 study). The FNNs include Artificial Neural Network (ANN), Multilayer Perceptron (MLP), Backpropagation Neural Network (BPNN), Extreme Learning Machine (ELM), Radial Basis Function Neural Network (RBFNN), Wavelet Neural Network (WNN), and Generalized

Regression Neural Network (GRNN). The T-RNNs include Elman network (ElmanNN), Nonlinear AutoRegressive with eXogenous inputs (NARX) model, Time Delay Neural Network (TDNN), and Recurrent Fuzzy Neural Network (RFNN). Regarding the 44 studies focused on DL, these can be further delineated into four distinct categories: Convolutional Neural Networks (CNNs: 12 studies), Advanced Recurrent Neural Networks (A-RNNs: 20 studies), convolutionalrecurrent hybrid networks (10 studies), and Autoencoders (AEs: 2 studies), with the A-RNNs category comprising Long Short-Term Memory (LSTM), Bidirectional LSTM (Bi-LSTM), and Gated Recurrent Unit (GRU) networks.

Regarding the predictive content of the collected research, the following movements were predicted for each upper limb joint under various angular and force ranges, velocities, loads, and durations: internal/external rotation of the shoulder joint, as well as the adduction/abduction and flexion/extension in both vertical and horizontal planes; flexion/extension of the elbow joint in the vertical and horizontal planes; flexion/extension, ulnar/radial deviation, and pronation/supination of the wrist joint; the independent and simultaneous flexion/extension of the Metacarpophalangeal (MCP), Proximal Interphalangeal (PIP), and Distal Interphalangeal (DIP) joints, encompassing both single and multiple fingers movements, along with grasp tasks based on different grip strengths and different object sizes and shapes. Furthermore, the study also conducted experiments based on static isometric contraction under different intensities, compound synergistic movements, and mirrored movements, as well as simultaneous predictions of joint kinematics and dynamics. Additionally, over one-third of these studies utilized the public NinaPro dataset, along with other public datasets (e.g., putEMG-Force [11], Biopatrec [12], and KIN-MUS UJI [13]) for the development of prediction algorithms.

![](images/b86599359b0ff6fd8737573ecc61d9f1413571c085a76de0cdf0b05d222b3124.jpg)

![](images/d503968cacd1327ab8826b5cb15e9228beb967f1f11b6f34ce7330da8de078d7.jpg)  
Fig. 3. Percentage distribution of different method types, offline/online predictions, subjects/databases, and subject attributes in both MB and MF studies.

## A. MB Approaches

## 1) Shoulder Joints

Compared to other upper limb joints, the shoulder joint is less studied due to its anatomical complexity, diverse multi-degree of freedom (DoF) movement patterns, and the challenges posed by sEMG signal acquisition. Study [14] integrated a muscle activation model with EMD and the ELM to achieve real-time prediction with 32ms low latency in random movement speeds.

## 2) Elbow Joint

Both studies [15] and [16] utilized the simplified Hill model containing only CE. Specifically, [16] combined the Hill model with a state-switching model for real-time prediction. However, the state-switching model exhibits time delays, and robustness ofthe simplified model is poor for small changes in joint angles.

Apart from the study [17], which used a complete Hill model encompassing CE, PE, and SE, studies [18]–[23] employed the rigid-tendon Hill model containing only the CE and PE. However, the rigid-tendon Hill model's neglect of muscle stiffness variation results in substantial torque prediction errors. As a refinement, [20] employed a Hill model optimized via Genetic Algorithm (GA) and the Short-Range Stiffness (SRS) model based on torque balance equations to concurrently predict joint angles and time-varying stiffness. [21] employed GA to optimize the Hill model enhanced with more additional physiological parameters, achieving robustness across various movement loads. Regarding other optimization algorithms, [23] optimized the Hill model using nonlinear least squares.

Studies [15], [17], [18], [22] integrated the MB and MF approaches. Specifically, [15] employed polynomials to approximate the relationship between muscle-tendon force and elbow angle, replacing the MSK geometric analysis and motion equations in the MB approach. Moreover, [18] developed a State Space (SS) model optimized with Extended Kalman Filter (EKF) by merging the Hill model with time-domain (TD) features for real-time closed-loop estimation. Additionally, [17] compared the capabilities of the Hill model and BPNN in predicting elbow isometric contraction force, concluding that

BPNN provides superior predictive performance due to the linear relationship between joint forces and sEMG, but it lacks model interpretability. Hence, combining MB and MF methods offers complementary advantages. For instance, [22] trained muscle activation optimization factors using the RBFNN, reducing the conversion bias between sEMG and muscle activation, thereby adapting to individual variances. The results indicated that this hybrid MB-MF approach outperformed standalone MB or MF methods, further enhancing the accuracy of joint torque predictions.

## 3) Wrist Joints

Studies [24]–[27] employed the rigid-tendon Hill model. Specifically, [24] and [25] balanced the Hill model's predictive accuracy and computational complexity through sensitivity analysis and GA optimization, addressing the oversimplified model that tends to overestimate parameters and thus neglect subject specificity. Moreover, based on the mirrored bilateral motion experiment, [25] proved that there is no statistical difference between the performance of this method on the ipsilateral side and the contralateral side. Additionally, since the supinator is the deep-seated muscle challenging to measure via sEMG directly, studies [26] and [27] employed Non-negative Matrix Factorization (NMF)-based virtual MS co-activation to replace the muscle activation of the pronator and supinator, subsequently inputting these into the Hill model for prediction. Results indicated that this approach outperformed the linear regression (LR) and ANN based on TD and MS features, exhibiting robustness across various upper limb postures.

Studies [28]–[30] integrated the MB and MF approaches. Specifically, [28] utilized the BPNN to identify distinct motion phases based on muscle activation, subsequently employing the MSK optimized by Bayesian LR for low-latency real-time joint force prediction. The results highlighted the superiority of the Bayesian LR over the GA in simplicity and efficiency. Additionally, [29] introduced physical MSK knowledge as a soft constraint added to the CNN model's loss function. This method's predictive performance was not only superior to Support Vector Regression (SVR), ELM, ML-ELM, and CNN but also had a simpler architecture than traditional CNN, requiring less training data and converging faster. Moreover, [30] furthered [29]'s work by sharing the pre-trained CNN parameters and updating only the fully connected layer for transfer learning. Results showcased that this method of sharing CNN's advanced sEMG feature extraction knowledge not only had an excellent convergence rate and generalization but also required minimal individual data for rapid transfer learning.

## 4) Hand Joints

Similar to research [17], study [31] employed the filtered sEMG's Root Mean Square (RMS) signal as input and predicted using the same complete Hill model. While studies [32]–[34] compared the performance of muscle activation models and TD features combined with FNN, single-output Gaussian Process Regression (GPR), and multi-output GPR, respectively. The results indicated superior performance using muscle activation models with EMD over the TD feature. Moreover, due to its ability to effectively model the inherent correlation between joints, multi-output GPR outperformed FNN and single-output GPR, underscoring the significance of considering MS.

Unlike most studies employing the Hill model, study [35] was the only study that leveraged the Huxley model for realtime predictions. It reduced the computational complexity of the high-dimensional Huxley model using the spectral method, Galerkin method, and balanced truncation method, further employing Particle Swarm Optimization (PSO) for parameter optimization. Finally, this approach balanced the prediction performance and calculation time of the Huxley model, along with excellent generalizability across days.

## 5) Hand-Wrist Joints

Studies [36]–[40] employed the rigid tendon Hill model optimized via the GlobalSearch optimization (GO). Results indicated that subject-generic models with average Hill model parameters based on all subjects performed better than subjectspecific models, LR, and ANN.

Regarding the MSK model built on MS features and MU neural features, study [41] suggested that the Synergistic Linear Regression Model (SLRM) based on Hierarchical Alternating Least Squares (HALS) and LR slightly surpassed traditional MSK approaches. Studies [42], [43] integrated MS features with MSK models, where [42] modeled MSs extracted through NMF-HP with L2 regularization constraint (NMF-HP-L2) into the GO-optimized Hill model, achieving superior predictive performance and stability over both Hill and NMF-Hill models. [43] constructed the MSK model using MSs extracted from independent components obtained via Adaptive Mixture Independent Component Analysis (AMICA) with NMF, resulting in better predictive performance than traditional MSK models, LR, and SVR. Additionally, echoing findings from [44]–[49], study [50] input MU discharge frequency extracted from HD-sEMG and FastICA into the GO-optimized Hill model. The outcomes highlighted that the MU-Hill model had significantly improved accuracy and robustness over Hill models based on TD features.

Regarding reinforcement learning (RL) based MSK models, studies [51]–[53] utilized the DDPG algorithm under the Actor-Critic framework and Proximal Policy Optimization (PPO), enabling multi-agents to compute joint angles using MSK's forward dynamics model based on predicted joint torques. The resulting RL-MSK model showed comparable accuracy to the Hill model but outperformed MLP and NARX, with strong robustness against movement speed variations. However, the training time of this method is about 8 hours and lacks generalization to the untrained new data.

## 6) Elbow-Shoulder Joints

Studies [54] and [55] have constructed closed-loop SS models grounded on the Hill model and forward dynamics. Specifically, [54] integrated the complete Hill model with NARX state equations and BPNN measurement equations, then employing the Unscented Kalman Filter (UKF) for real-time prediction and closed-loop estimation, resulting in superior outcomes than the open-loop estimation using solely NARX and BPNN. [55] combined the rigid-tendon Hill model with fused TD features of sEMG and MMG to build the Unscented Particle Filter (UPF)-optimized SS model, outperforming the BPNN, SVR, and GRNN and significantly reducing the demand for training data volume.

## B. MF Approaches

## 1) Shoulder Joints

a) Traditional Neural Networks: Study [56] fed extracted TD and frequency-domain (FD) features into the TDNN with shortterm memory capabilities for prediction.

b) Other Methods: Study [57] utilized the capability of Fast Orthogonal Search (FOS) for rapid fitting and system identification of non-linear models. Additionally, this study considered the coupling effect between the output force and joint angles during shoulder activity to enhance prediction accuracy. This coupling effect mirrors the force-velocity and force-length relationships between muscle fibers and muscle force in the Hill model, suggesting integration of MF methods with the Hill model to further boost predictive performance.

## 2) Elbow Joint

a) Linear Models: Research [58] integrated adaptive weighted peak sEMG signals with linear least squares to predict mirrored elbow movements. Study [59] demonstrated the superior predictive performance of the Autoregressive with Exogenous Input (ARX) model over Autoregressive Moving-Average with Exogenous Input (ARMAX), Autoregressive Integrated Moving-Average with Exogenous Input (ARIMAX), and SS models under various movement loads.

b) Machine Learning: Study [60] utilized tree-based Hierarchical Projection Regression (HPR) algorithms and incremental learning for real-time prediction of elbow joint angles under different loads. Study [61] employed Random Forests (RF) for the prediction and automatic selection of significant time-delayed features. Similarly, [62] incorporated Gray Feature Weighted Support Vector Machines (GFWSVM) to assign weights to sEMG features based on their significance.

c) Traditional Neural Networks: Studies [63]–[66] leveraged extracted TD features to predict joint angles using ANN, WNN, and GRNN, respectively, demonstrating WNN outperformed both Support Vector Machines (SVM) and RBFNN. Moreover, [63] highlighted the superior performance of RBFNN in predicting both joint angles and angular velocities compared to MLP. Numerous studies, including [67]–[71], employed BPNN

WEI et al.: Continuous Motion Intention Prediction Using sEMG Signals for Upper-Limb Rehabilitation: A Systematic Review of Model-Based and Model-Free Approaches.

for prediction. Specifically, [67] showed that inter-subject variability significantly impacts the predictive performance by comparing generic and personalized models. Similarly, [71] evidenced greater inter-subject variability than intra-subject by comparing TD features and subject-invariant features extracted using Maximum Independent Domain Adaptation (MIDA). Consequently, [68] incorporated the GA feature selection to eliminate the inter-subject redundant and low-correlation features, thereby enhancing the inter-subject generalizability. Lastly, [72] introduced the SNN, which emulate biological neuronal spiking mechanisms and membrane potential variations, achieving predictive accuracy comparable to LSTM.

As for the time-series neural networks, studies [73] and [74] employed the TDNN for prediction based on TD features. On the other hand, studies [75] and [76] integrated the NARX model with the MLP, ElmanNN, and the Adaptive Neuro-Fuzzy Inference System (ANFIS) from [77] for predictions based on TD features, ultimately proving the superior predictive performance of the ANFIS-NARX model.

d) Deep Learning: Studies [78] and [79] employed LSTM based on TD features for precise joint angle predictions. Study [80] showed the superiority of CNN-LSTM over individual CNN and LSTM models, emphasizing the importance of establishing long-term contextual dependencies among extracted advanced features.

e) Muscle Synergy Features: Study [81] fed MS features extracted through Multivariate Curve Resolution-Alternating Least Squares (MCR-ALS) into Bi-LSTM for prediction. The results indicated that MS features demonstrate greater robustness to sEMG variations compared to traditional TD features. Furthermore, MS features extracted by MCR-ALS maintained a higher correlation across a five-day experimental period for both intra-subject and inter-subject than those extracted using Principal Component Analysis (PCA) and NMF.

f) Other Methods: Studies [82] and [83] applied Kalman Filters (KF) to achieve real-time predictions of joint angles and torques under varying loads and motion speeds. Studies [84] and [85] employed nonlinear system identification methods. Specifically, [84] validated the superiority of the Parallel Cascade Identification (PCI) model over the FOS algorithm in [57], while [85] presented the robustness of the Hammerstein-Wiener model coupled with WNN for predictions across days, although it still requires individual calibration and its predictive performance can be influenced by motion loads.

## 3) Wrist Joints

a) Linear Models: Study [86] utilized ridge regression based on TD features for predictions and demonstrated that the Least Absolute Shrinkage and Selection Operator (LASSO) is an effective method for HD-sEMG channel selection. Additionally, study [87] indicated that although ANN outperformed LR and NMF in offline experiments, their online performances had no significant differences.

b) Machine Learning: Studies [88] and [89] input timedelayed TD features into Least Squares Support Vector Machine (LSSVM) and RF for predictions, respectively, achieving superior performance than that of SVM and BPNN. This again highlights the importance of considering EMD.

c) Traditional Neural Networks: In research [90], TD and FD features were fed into a GA-optimized ELM for prediction, achieving better results than GRNN. However, ELM still exhibits some instability. Studies [91] and [92] utilized ANN and BPNN based on TD features, respectively. Notably, [91] found that using sEMG signals based on large-area muscle activity superimposition outperformed non-directional iEMG signals targeting local muscle information, suggesting using HD-sEMG sensors to collect more comprehensive muscle activity data. Additionally, the study [93] also employed BPNN but incorporated both FD and time-frequency domain (TFD) features as inputs, further applying KF for post-processing.

d) Deep Learning: Compared to PCA, studies [94] and [95] proved that Deep Neural Networks based on Stacked AEs (SAE-DNN) can extract more representative sEMG features, hence achieving better predictions than LR and SVR.

Both studies [96] and [97] employed CNN. Specifically, [96] achieved superior predictive performance using raw TD sEMG images and Fast Fourier Transform (FFT) based FD images compared to six manually feature-engineered machine learning models, namely LR, SVR, RF, GPR, and MLP. Meanwhile, [97] indicated that CNN based on TD images outperformed ANN founded on Histogram of Oriented Gradients (HOG) features. Moreover, PCA was employed in this study to show the higher correlation of CNN-extracted spatial features with actual joint torques over pixel variationfocused HOG features and empirically-based manual features.

Studies [98] and [99] used LSTM models for predictions. Among them, [98] showcased LSTM's exceptional predictive performance under various loads, while [99] employed CNN-LSTM to extract spatiotemporal data from FFT-based sEMG spectrograms, outperforming CNN, LSTM, and ML models.

However, the generalization across days and subjects of CNN-LSTM still needs to be improved. This was further validated by [100], which utilized LSTM-AE to evaluate and quantify the domain shift in CNN-LSTM for the task across days based on the reconstruction error of CNN features, enabling model performance degradation monitoring and timely model recalibration. To further enhance generalization across subjects, [101] proposed a transfer learning method employing a dual-stream CNN to extract domain-invariant features from both source and target domain data, subsequently adjusting CNN weights via regression loss, Maximum Mean Discrepancy (MMD) loss, and regression contrastive loss.

e) Motor Unit Neural Features: Study [102] applied the convolutional blind source separation to input HD-sEMG's Decomposed Spike Count (DSC) features and residual sEMG TD features into LR for prediction. Results indicated that MU features offer greater improvements for amputees compared to TD features. Nonetheless, the DSC feature overlooks spatial information and interactions among MUs. Therefore, [103] first convolved the Motor Unit Spike Train (MUST) and Motor Unit Action Potential (MUAP) obtained from convolution kernel compensation (CKC) decomposition to reconstruct MU images, which were then fed into CNN for predictions, significantly outperforming the DSC feature-based LR, SVR, and ANN. Research [104] and [105] also utilized the CKC for MU decomposition and LR for predictions, with [104] further refining the MU pool to identify dominant MUs. Findings revealed that the MU twitch force model proposed in [105] outperformed both the DSC model and the MU discharge frequency model based on Cumulative Spike Train (CST),

This article has been accepted for publication in IEEE Transactions on Neural Systems and Rehabilitation Engineering. This is the author's version which has not been fully edited and content may change prior to final publication. Citation information: DOI 10.1109/TNSRE.2024.3383857

WEI et al.: Continuous Motion Intention Prediction Using sEMG Signals for Upper-Limb Rehabilitation: A Systematic Review of Model-Based and Model-Free Approaches.

MUST, and PCA.

f) Other Methods: Study [106] applied the Gaussian Mixture Regression (GMR) statistical model to the symmetric positivedefinite matrix manifolds, achieving superior prediction performance than GMR in Euclidean space. Study [107] input time-delayed TD features into the Kernel Recursive Least Squares Tracker (KRLS-T) as an online non-linear adaptive filter for predictions, significantly outperforming ANN and Kernel Ridge Regression (KRR) due to its amalgamation of nonlinear kernel regression, online adaptive estimation benefits, and lower computational cost.

## 4) Hand Joints

a) State Space Model: Studies [108] and [109] applied the SS model based on the N4SID parameter identification method to predict finger joint angles under different static wrist postures during the mirrored bilateral movement. Additionally, study [110] utilized Recursive Least Squares (RLS) for SS model parameter estimation and the KF for post-processing, ultimately outperforming MLP, NARX, and LDA models.

b) Linear and Non-linear Models: Study [111] implemented LR based on spatial filtering features for prediction, and study [112] merged ridge regression, an extension of LR, with incremental learning to achieve real-time predictions for grasping tasks across days. However, study [113] found that non-linear KRR outperformed the LR across tasks, emphasizing the importance of considering the non-linear relationship between sEMG and motion intention. Studies [114] and [115] further compared linear and non-linear approaches, specifically the linear Vector Autoregressive Moving Average model with Exogenous inputs (VARMAX) against the non-linear Gaussian Process (GP). While their performances were comparable, GP's non-linearity could model more complex motion intentions. However, the computational cost for GP increases with growing training data. Therefore, improvements can be inspired by the Sparse Pseudoinput Gaussian Process (SPGP) regression model in study [116] or by extracting MS features via Gaussian Process Latent Variable Model (GPLVM) as in study [117].

c) Traditional Neural Networks: Networks utilized in studies include: ANN with Mel spectrum features robust to grip force variations [118], BPNN [119], the GRNN outperforming KRLS and LR [120], [121], as well as FNN, Cascading FNN (CFNN), and RBFNN in [122] outperforming GRNN. Moreover, study [123] combined the deep RF model based on eight TD features with ANN, achieving better prediction outcomes than GP.

d) Deep Learning: The DL networks used in research can be classified into three categories: CNNs, RNNs, and CNN-RNN hybrids. The CNNs include one-dimensional CNN (1D-CNN) [124], 2D-CNN [125], 3D-CNN [126], [127], parallel CNN [128]–[130], AlexNet and ResNet [131], [132], as well as Temporal Convolutional Networks (TCN) [133], and Large-Scale TCN (LS-TCN) [134]. The RNNs encompass NARX [135], LSTM [13], [131], [132], [136]–[139], and GRU [131], [132]. The hybrid category includes CNN-LSTM [140], Long Exposure Convolutional Memory Network (LE-ConvMN) [141], and Attention-ConvGRU [142].

CNNs: According to [124], 1D-CNN exhibited superior realtime prediction performance for finger force compared to 2D-CNN and LR due to its ability to learn deeper advanced features while reducing data dimensions and avoiding redundant spatial information. [125] suggested that using TD and FD feature images as inputs to 2D-CNN can further reduce noise and improve predictive accuracy compared to raw sEMG images. [126], [127] demonstrated that 3D-CNN can learn deeper muscle anatomy, MS, and motion velocity features from multiple electrode perspectives, enabling the prediction of untrained random new movements. Moreover, [131] and [132] noted that AlexNet exhibits better performance than ResNet, LSTM, and GRU. Additionally, [133] proved that TCN outperformed LSTM, and TCN model size can be reduced to 70.9Kb with 4.76ms latency using int8 quantization. Lastly, [134] indicated that sEMG signals contain atomic segments highly correlated with movement during specific time frames, suggesting the CNN kernel size should approximate or slightly exceed atomic segment length to fully extract sEMG features while minimizing network parameters. Therefore, the LS-TCN, whose kernel sizes are similar to the atomic segment length, addressed the feature extraction limitations of TCN and attained better performance than TCN, RNN, and SPGP.

RNNs: [135] integrated post-PCA TD features and EKF post-processing with NARX, achieving prediction outcomes surpassing MLP. [13], [137]–[139] demonstrated that LSTM outperforms FNN, RBFNN, and SPGP in predicting all hand joint angles, with generalizability to untrained data. In addition, [136] employed the Gaussian Mixture Model (GMM) with LSTM to construct a probabilistic mapping model from sEMG to hand postures, which formed confidence intervals and led to LSTM-GMM outperforming LSTM. Finally, [131], [132] noted that although GRU exhibits faster training speeds, LSTM consistently delivers more stable prediction outcomes.

CNN-RNN Hybrids: [140]'s CNN-LSTM combines the advantages of CNN and RNN, and similar to [30], it can employ transfer learning by merely retraining the fully connected layer. To address LSTM's drawback of flattening multi-dimensional inputs into 1D vectors, which leads to spatial information loss when processing spatiotemporal data, [141]'s LE-ConvMN replaces the LSTM's fully connected layer with the 2D-CNN. Moreover, by progressively reducing the dimensions of longexposure sEMG samples, it can extract high-dimensional spatiotemporal features across multiple electrode channels. Consequently, LE-ConvMN outperformed SPGP and LSTM in prediction and generalization across subjects and joints. However, due to LE-ConvMN's lengthy training time and high computational cost, it is not suitable for real-time applications.

e) Muscle Synergy Features: Study [143], utilized the Common Spatial Pattern (CSP) algorithm frequently used in EEG analysis. The results show that compared to NMF, as it tends to learn the differential features between samples, CSP can more effectively differentiate fingertip force signals of the highly correlated index and middle fingers in real time without interference from inter-finger signal crosstalk. To enhance NMF's predictive performance, study [144] introduced the Hadamard product into NMF (NMF-HP), reducing erroneous estimations of non-active finger activations. Results indicated that NMF-HP can provide more accurate real-time estimations for simultaneously activated fingers than LR, CSP, and NMF. Moreover, for the dimensionality reduction and reconstruction of sEMG, studies [145] and [146] utilized Partial Least Squares Regression (PLSR), as the extension of PCA, and NMF, respectively. Notably, study [146] successfully reconstructed the original sEMG data using only three MSs by iteratively optimizing the NMF activation coefficient matrix.

f) Motor Unit Neural Features: Study [111] emphasized that using monopolar electrode arrays and more sEMG channels can reduce prediction errors, again demonstrating the advantages of using HD-sEMG. Therefore, studies [44]–[49] extracted the sum of discharge rates of MUs related to targeted fingers and tasks from HD-sEMG using FastICA and MU pool refinement, and then input them to LR, enabling simultaneous prediction of joint angles and fingertip forces under various wrist postures. Studies [128]–[130] first estimated the overall discharge rates of FastICA-decomposed MUs by using parallel CNN based on FFT spectrograms and RMS TD images of HD-sEMG, and then input them to LR for real-time finger force and joint angle prediction. However, it still relied on the accuracy of FastICA.

g) Attention mechanisms: The potential of attention mechanisms was implied early in [11], where the gradient boosting machine (LightGBM) model could iteratively omit insignificant features, achieving better performance than LR, MLP, SVR, and CNN. Subsequent studies [142], [147]–[149] further introduced attention mechanisms. They incorporated self-attention or multi-head attention modules into MLP, CNN, and ConvGRU (the combination of GRU and 1D-CNN), achieving significantly higher accuracy and generalizability than LSTM, GRU, TCN, and SPGP, with shorter training times than GRU and LSTM. Furthermore, studies [150]–[152] utilized attention-based Transformer models for further improvements, including the BERT, CNN-Transformer, and LSTA-Conv network based on Long-Short Time Aggregation (LSTA) module. Specifically, BERT notably surpassed LSTM, TCN, and LE-ConvMN in prediction performance, training time, and generalization across subjects. CNN-Transformer was proven to outperform CNN-LSTM, Bi-LSTM, and Transformer. Finally, LSTA-Conv, incorporating the selfattention-based Transformer and multi-scale ResNet, outperformed RNN, LSTM, SPGP, and CNN-Attention.

h) Other Methods: Study [153] employed KF combined with TD features for prediction. Although this method exhibited some generalizability across subjects, its predictive capability largely depends on the accurate representation of model parameters for the target system. Study [119] employed Gene Expression Programming (GEP) based on GA and genetic programming, which outperformed BPNN. Study [154] utilized a logarithmic regression model, achieving real-time estimation of grasping forces under transient and steady states while being robust to sEMG drifts and instantaneous variations. Moreover, contrary to most methods based on steady-state sEMG, study [155] employed LR with elastic net regularization for accurate real-time prediction of grip force using a single transient sEMG activation, showing promising results even for amputees. Lastly, study [156] applied the energy conservation and transfer theory, stating that kinetic and potential energy within each finger dynamically interconvert and distribute within a given muscle activation level, but the total energy across all fingers remains constant. It initially extracted MS features with ICA, then deduced each finger energy under the extreme conditions of complete fixation and free movement, finally employing ANN to learn the real-time mapping between MS features and finger energy. Although this reduced computation costs and demands on training samples, prediction errors increased with energy growth, and finger flexion predictions outperformed extension.

## 5) Hand-Wrist Joints

a) Linear Models: Study [157] combined TD features with linear system identification to compare four linear time-series models: ARX, Autoregressive exogenous Regularized (ARXR), Output Error (OE), and ARMAX. The findings indicate that the OE model demonstrated superior predictive capabilities.

b) Machine Learning: Studies [158]–[160] deployed SVR based on TD, FD, and TFD features for joint angle and grip force predictions. Study [161] utilized TD features alongside Gradient Boosted Regression Trees (GBRT) built on cascaded decision trees to predict joint angles with the generalizability to untrained new data.

c) Traditional Neural Networks: Studies [122], [162] indicated that TDNN could also be applied to amputees and found that RBFNN outperformed FNN, CFNN, and GRNN. However, RBFNN's prediction errors for finger grasping movements exceeded those for wrist joints.

d) Deep Learning: Study [163] achieved low-latency real-time prediction with the Channel-wise-CNN model, where each kernel corresponded to an sEMG channel, and enabled transfer learning by updating only the fully connected layers. LSTMbased [164] found that using all sEMG channels outperformed one-to-one mapping. The Deep Kalman Filtering Network (DKFN) in [165] extracted advanced features via CNN and trained KF parameters using LSTM, outperforming CNN and CNN-LSTM. However, its performance was still limited by sEMG sequence lengths. The Temporal Convolution (TC) model in [12] utilized 1D-CNN and PCA for advanced feature extraction, AE for unsupervised learning of MS features, and finally RNN for real-time mapping between MS and motion intention. This method outperformed instantaneous mixture models in MS reconstruction and predictive performance and showed generalizability to untrained new data. However, the correlation across subjects of AE-extracted MS features was low, and the suppression of inactive joint activations was limited, suggesting improvements via NMF-HP-L2 as in [42].

e) Muscle Synergy Features: Mirroring [42], Study [166] also utilized NMF-HP-L2 to better suppress activations for inactive joints, with higher extracted MSs correlations at different joint movement speeds than PCA, Non-linear PCA (NLPCA), NMF, and Sparse Constraint NMF (SCNMF).

f) Other Methods: Study [167] harnessed the statistical model based on GMM and the Hidden Markov Model (GMM-HMM), using the Viterbi algorithm and model pruning to compute state probabilities and establish long-term memory, outperforming LSTM and GRU in both accuracy and computation time.

## 6) Wrist-Elbow Joints

Three studies on the wrist-elbow joint combination were all conducted based on the MF method and involved LSTM. Study [168] demonstrated that LSTM is more suitable than the GAoptimized BPNN for simultaneously predicting multi-joint movement. Study [169] established that LSTM, employing correlation-based feature selection and PSO optimization, outperformed BPNN and required less training time. Lastly, study [170] integrated CNN-LSTM with self-attention and KF (Attention-CNN-LSTM-KF), achieving superior prediction performance over CNN, CNN-LSTM, Attention-CNN, CNN-

WEI et al.: Continuous Motion Intention Prediction Using sEMG Signals for Upper-Limb Rehabilitation: A Systematic Review of Model-Based and Model-Free Approaches.

KF, and CNN-LSTM-KF, further emphasizing the benefits of employing attention mechanisms and KF post-processing.

## 7) Elbow-Shoulder Joints

a) Traditional Neural Networks: Studies [171] and [172] utilized PCA and ICA, respectively, to extract MS features with MLP and ANN for prediction, demonstrating superior source muscle activity separation using ICA compared to PCA. Studies [173] and [174] utilized TD features and BPNN for predictions, with [174] introducing an AE before BPNN input to extract advanced features through unsupervised learning. As for studies [175] and [176] based on RNN, [175] employed TFD features for prediction with GA-optimized ElmanNN, while [176] utilized time-delayed TD features with RFNN to enhance robustness to movement speed variations. However, although GA-ElmanNN outperformed both ElmanNN and GA-BPNN, GA increased computational cost for real-time prediction.

b) Deep Learning: Study [177] applied Squeeze-Excitation Network (SE-Net) prior to using the TCN, where SE-Net can increase the weight of features that dominate muscle movement and TCN can overcome LSTM's lengthy training and gradient explosion issues, outperforming both BPNN and LSTM. Studies [178]–[180] employed LSTM and Bi-LSTM, with [178] demonstrating the superior multi-joint predictive capability of LSTM over MLP, but the temporal variability of sEMG caused LSTM's accuracy to decline over time. Both [179] and [180] verified that Bi-LSTM not only outperformed MLP, CNN, LSTM, and GUR but also effectively addressed the issues of asynchrony and tremors between sEMG and joint angles caused by muscle deformation, and with generalizability to untrained new data. Studies [181] and [182] respectively input TFD features and neural activation into CNN-LSTM for prediction, outperforming the SVR and CNN, yet requiring improvement for generalization across days. Lastly, study [183] replaced the CNN in CNN-LSTM with Short-Connection AE (SCA), achieving better performance and generalization than MLP and CNN. This is attributed to AE's ability to extract redundant information of sEMG across different target movements, which is removed by short connections, allowing SCA to extract motion-specific information akin to MS features.

c) Other Methods: Similar to SCA, study [184] initially employed the correlation-based redundancy segmentation to remove redundant multi-joint sEMG before using the same SS model and KF post-processing for real-time prediction as in [54]. Although outperforming BPNN and NARX, its threshold selection method for redundancy segmentation is yet to be refined. Additionally, study [185] solely utilized the multiparameter combined KF based on least squares estimation for prediction, reaffirming KF's robustness for random complex movements and its excellence as a post-processing method.

## 8) (Hand), Wrist, Elbow, and Shoulder Joints

Most current rehabilitation robots focus on assisting the independent movement of selected upper limb joints, rarely providing coordinated multi-joint training for the entire upper limb. Study [186] indicated that comprehensive multi-joint coordination training for the entire upper limb brings better rehabilitation outcomes for stroke patients compared to singlejoint training. However, only nine studies based on MF methods predicted all upper limb joint combinations.

a) Traditional Neural Networks: Study [187] utilized TD features with GRNN to predict grasping and pushing-pulling forces under specific conditions. Study [188] found that NARX outperformed TDNN for prediction. Study [189] indicated that ElmanNN outperformed BPNN, reiterating the importance of considering contextual relationships.

b) Deep Learning: Studies [190] and [191] noted that the openloop LSTM had minor prediction errors due to uncertainties in the LSTM modeling process (e.g., number of hidden neurons and dataset sizes) and physiological effects like joint damping. Thus, they combined LSTM with Zeroing Neural Network (L-ZNN) and Noise-Tolerant ZNN (L-NTZNN) to construct error functions using ZNN closed-loop feedback to eliminate errors. Ultimately, L-NTZNN outperformed L-ZNN, L-GNN, LSTM, and GPR. Additionally, Studies [192] and [193] employed two Bi-LSTM for prediction and transfer learning, with the first as the shared network and new data only training the second personalized Bi-LSTM and the fully connected layer.

c) Other Methods: Study [186] input muscle activations to the N4SID-optimized SS model for real-time prediction, while study [194] input TD features into three parallel linearnonlinear cascaded regression decoders for low-latency realtime prediction. However, although these decoders operated rapidly, their nonlinear static functions still had limitations, suggesting further improvements through neural networks.

## IV. DISCUSSION

This discussion section firstly summarizes key findings from research conducted over the past decade on upper limb single joint and multi-joint combinations motion intention prediction. Subsequently, it critically examines current research limitations and challenges in this field, thereby proposing clear future research directions for upper limb motion intention prediction algorithms. Additionally, Table I below highlights those studies among the 186 adopted papers that were identified by author as having significant referential value for future research.

## A. Significant Findings

1) Advantages of HD-sEMG Sensors and Multi-sensor Fusion

Studies [91], [111], [126], [127] indicated that standard sEMG sensors, apart from having significantly lower spatial resolution compared to HD-sEMG sensors, are also less efficient in capturing comprehensive and high-quality muscle activation information. HD-sEMG, with its broader electrode coverage, can acquire data from multiple electrode perspectives, thereby reducing prediction errors. It also substantially mitigates the impacts of electrode placement errors, motion artifacts, and electrode displacement. Therefore, HD-sEMG exhibits inherent advantages over sEMG in extracting MU features and predictive performance. Regarding multi-sensor fusion, as discussed in studies [21], [55], [79], [114], [140], [151], [160], integrating sEMG sensors with EEG, IMU, FMG, and MMG sensors can further enhance the performance and robustness, especially in scenarios of isometric contractions and under external force interference.

## 2) Closed-Loop Feedback

Contrasting with open-loop models, studies [18], [54], [55], [190], [191] demonstrated the superiority of closed-loop models. These include constructing error functions using closed-loop feedback based on ZNN and NTZNN or employing KF post-processing based on prior knowledge and probability distribution, such as EKF and UKF, to eliminate cumulative errors inherent in open-loop models while enabling safer and more cautious control strategies. Additionally, MS and MU features based on redundant sEMG information and noise are also effective for closed-loop correction.

## 3) Advanced Feature Extraction, Contextual Relationships, and Attention Mechanisms

Studies [94]–[96], [131]–[133] using SAE, CNN, and various CNN variants, such as 3D-CNN, TCN, AlexNet, and ResNet, indicated that DL's automated advanced feature extraction outperforms manual features in traditional ML by encompassing richer sEMG information. Moreover, RNNbased models like LSTM [98] and Bi-LSTM [179] can improve robustness to load and movement speed variations as well as multi-joint simultaneous prediction accuracy through long-term contextual dependencies. Therefore, combining CNN and RNN strengths to extract spatiotemporal information simultaneously, such as CNN-LSTM [80], [140] and LE-ConvMN [141], can outperform the methods using only CNNs and RNNs. As for the attention mechanisms, studies [11], [62], [177] using GFWSVM, LightGBM, and SE-Net have already proven the significance of adjusting key feature weight distributions. In the author's view, sEMG signals during movement can be considered as the text or video, where each segmented sEMG time window represents a word in text or a frame in video, suggesting employing Natural Language Processing (NLP) algorithms to establish intrinsic relationships. Hence, attention mechanisms, which can adaptively capture the contextual dependencies between local and global features to enhance the weight of key features and have parallel computing capabilities, have been extensively studied recently. As indicated in Table I, besides integrating attention mechanisms with DL networks, attention-based models like Transformer and its variant BERT are recent research trends that can be further expanded.

## 4) Muscle Synergy Features

Considering the individual variability and time-varying nature of sEMG signals, difficulties in deep muscle sEMG signal collection, as well as coupling and crosstalk between active and passive muscles during movement, extracting MS features that are highly correlated with each type of movement is of great significance. Studies [164] and [19] affirm this, showing that even non-participating synergistic muscles can provide valuable contextual information, and separately extracting MS features for different joints can further improve prediction accuracy [41]. Moreover, studies [32]–[34], [146] proved that MS features not only outperform traditional TD features but also improve performance for complex multi-joint movement prediction and are robust against variations across days and electrode displacement. Additionally, studies [131], [132], [183] indicated that since sEMG signals consist of common components, individual differences, and random noise, MS analysis can eliminate redundant individual differences and noise, as well as extract highly correlated inter-subject features to enhance generalizability across subjects. As indicated in

Table I, currently outstanding MS feature extraction methods include MCR-ALS [81] and NMF-HP-L2 [166].

## 5) Motor Unit Neural Features

MU signals provide a more direct reflection of neural information transmitted from the brain to muscles than sEMG signals formed by the superposition of MUAPs. This has been substantiated by studies [44]–[49], [102], [128]–[130], which demonstrated that MU neural features, such as the DSC features, MU discharge rates, and MU images, are not only unaffected by movement speed and contain additional information not captured by TD features, but are also more robust against various forearm postures, sEMG crosstalk, and electrode and muscle fiber displacement. Furthermore, combining MU decomposition methods like FastICA and AMICA with HDsEMG can more effectively separate mixed signals from superficial and deep muscles. Additionally, given the more direct relationship between sEMG and MU activities compared to joint kinematics and dynamics, directly predicting MU activities using DL-based methods surpasses traditional MU decomposition and TD feature-based methods in performance, computational efficiency, generalizability across subjects and fingers, and the robustness of long-term prediction [128].

## 6) Integration of MB-MF Methods

Studies [27] and [24] indicated that since MF methods ignore the physiological relationships between muscle activation, muscle-tendon force, joint torque, and joint motion, as well as lack model interpretability, they may fail to predict new movements not covered in training sets and risk overfitting which reduces robustness. Although biomechanics-based MB methods can explicitly define the exact relationships between sEMG and motions, overly complex MB models with numerous physiological parameters are not conducive to real-time applications, while oversimplified MB models without considering individual differences may also increase prediction errors. Therefore, combining strengths of MB and MF methods for complementarity is proposed. For instance, as demonstrated in Table I, using RBFNN to train the muscle activation optimization factors [22] due to specific muscle activation models suitable for certain actions [19], adding physiological MSK constraints to CNN loss functions [29], or combining the Hill model with MF methods to construct SS models [54], [55].

## 7) Transfer Learning

Considering the inherent physiological and muscle control strategy differences among subjects, inter-subject variability significantly exceeds intra-subject variability [67], [71]. Moreover, based on the fact that extracting highly correlated inter-subject features from multi-subject training can improve model generalizability, and current MB and MF methods lengthy training times preclude the rapid deployment across subjects [68], the capability for effective transfer learning becomes crucial. Current transfer learning strategies include retraining only the fully connected layers in CNN, Bi-LSTM, and CNN-LSTM networks for parameter sharing [30], [140], [192], adjusting CNN weights based on domain-invariant features and loss functions in dual-stream CNN [101], as well as the subject adversarial knowledge (SAK) strategy in [152].

WEI et al.: Continuous Motion Intention Prediction Using sEMG Signals for Upper-Limb Rehabilitation: A Systematic Review of Model-Based and Model-Free Approaches.  
TABLE I  
STUDIES WITH SIGNIFICANT REFERENCE VALUES FROM THE 186 ADOPTED PAPERS
<table><tr><td>Categories</td><td>Approaches</td><td>Paper / Year</td><td>Regression Task</td><td>Methods Description</td><td>Performance Results</td></tr><tr><td rowspan="3">MB Approaches</td><td rowspan="2">MS with MSK</td><td>[42]/2022</td><td>MCP and wrist angle</td><td>NMF-HP-L2 with GO-Hill model</td><td>R = 0.89/NRMSE = 0.16</td></tr><tr><td>[43]/2022</td><td>MCP and wrist angle</td><td>AMICA-NMF with MSK</td><td>CC = 0.92/NRMSE = 0.1</td></tr><tr><td>MU with Hill Model</td><td>[50]/2022</td><td>MCP and wrist angle</td><td>MU frequency with GO-Hill model</td><td>R = 0.87/NRMSE = 0.18</td></tr><tr><td rowspan="10">MF Approaches</td><td rowspan="2">Deep Learning</td><td>[126]/2022</td><td>Finger angle and force</td><td>3D-CNN with data augmentation</td><td>CC = 0.94/MAE = 2.78°</td></tr><tr><td>[165]/2021</td><td>MCP and wrist angle</td><td>CNN with KF based on LSTM</td><td>R² = 0.8/RMSE = 13.6°</td></tr><tr><td rowspan="2"></td><td>[141]/2021</td><td>Finger angle</td><td>LE-ConvMN based on ConvLstm cell</td><td>CC = 0.82/NRMSE = 0.12</td></tr><tr><td>[170]/2023</td><td>Elbow and wrist angle</td><td>CNN-LSTM-Self-attention with KF</td><td>R² = 0.7</td></tr><tr><td rowspan="3">Attention Mechanisms</td><td>[150]/ 2022</td><td>Finger angle</td><td>BERT with GHM and OHEM</td><td>CC = 0.85/NRMSE = 0.08</td></tr><tr><td>[152]/ 2023</td><td>Finger angle</td><td>LSTA-ConvNN with SAK</td><td>CC = 0.83/RMSE = 7.93°</td></tr><tr><td>[103] /2020</td><td>Wrist torque</td><td>Reconstructed MU-image with CNN</td><td>R² = 0.88</td></tr><tr><td rowspan="2">MU Neural Features</td><td>[105] /2023</td><td>Wrist torque</td><td>MU twitch force model with LR</td><td>R² = 0.76/NRMSE = 0.11</td></tr><tr><td>[128]/ 2023</td><td>Finger angle and force</td><td>MU rate predicted by parallel CNN</td><td>R² = 0.78/RMSE = 10°</td></tr><tr><td rowspan="2">MS Features</td><td>[81]/2023</td><td>Elbow angle</td><td>MCR-ALS with Bi-LSTM</td><td>MAE = 3.24°/CC &gt; 0.9</td></tr><tr><td>[166] /2023</td><td>MCP and wrist angle</td><td>NMF-HP-L2</td><td>ASNR = 2.57/ R² = 0.9</td></tr><tr><td rowspan="5">MB-MF Approaches</td><td>Hill model with RBFNN</td><td>[22]/2021</td><td>Elbow torque</td><td>RBFNN for muscle activation</td><td>CC = 0.98/NRMSE = 0.05</td></tr><tr><td>MSK soft constraint</td><td>[29]/2022</td><td>Wrist angle and force</td><td>estimation with the Hill model CNN with MSK loss constraint</td><td>CC = 0.98/RMSE = 4.1°</td></tr><tr><td>State-space model</td><td>[54]/2019</td><td>angle and velocity of</td><td>SS model constructed by Hill model</td><td>RMSE = 1.15°/5.8%/s</td></tr><tr><td></td><td></td><td>elbow and shoulder</td><td>with NARX and BPNN</td><td>(angle/velocity)</td></tr><tr><td>RL with MSK</td><td>[51]/2022</td><td>MCP and wrist angle</td><td>DDPG-PPO-RL with MSK</td><td>R = 0.86/NRMSE = 0.15</td></tr></table>

CC/R: Pearson’s Correlation Coefficient, R<sup>2</sup>: Coefficient of Determination, RMSE: Root Mean Square Error, NRMSE: Normalized Root Mean Square Error, MAE: Mean Absolute Error, ASNR: Average Signal-to-Noise Ratio

## B. Current Challenges

## 1) Experimental Protocols Supplement

As illustrated in Figure 3, over three-quarters of studies relied on self-organized experiments rather than public databases. Besides, approximately 90% of the experiments recruited fewer than 10 subjects, and on average fewer than 15% of the subjects were disabled. Moreover, the proportion of male subjects was much higher than that of females. Additionally, since most studies employed the Ninapro database, which is predominantly composed of simple and highly controlled movements, more complex ADL-based databases, such as KIN-MUS-UJI [13] and Biopatrec [12], should be employed for comprehensive testing of developed predictive algorithms.

Regarding subjects, aside from recruiting more participants, studies [79], [81], [194] highlighted that motor control impairments and aberrant muscle activation patterns caused by neurological injuries can lead to tremors and unsmooth movements in stroke patients, potentially degrading prediction performance. Therefore, future experiments should also involve more stroke patients to assess the practicality of algorithms. Furthermore, due to the gender difference in sEMG signals leading to better overall prediction performance in males [136], future experiments should balance the gender ratio to test the algorithm generalizability across gender. Additionally, study [177] suggested recruiting subjects with diverse ages, heights, weights, and occupations to maximize physiological variability coverage for enhancing the model's generalizability.

As for experimental content, research [15], [16], [18], [22], [26], [54], [59], [178] indicated the need to include additional robustness and generalizability tests in future experiments. Robustness tests include sEMG signal crosstalk, drift, and electrode displacement; isometric contractions and external force disturbances; error effects caused by different upper limb postures and non-target joint movements; variations in movement speed and load; more complex random movements; muscle fatigue and skin sweating; as well as the model's longduration predictive capability. Moreover, for rapid changes in movement speed, studies [62] and [61] suggested improvement through adaptive sliding windows. As for generalizability tests, since multi-joint training can provide greater therapeutic benefits for stroke patients compared to single-joint training [186] and considering current research demonstrated poorer predictive performance in multi-joint combinations compared to the single joint, it is essential to improve the algorithm's multi-joint predictive performance and generalizability across joints, days, and subjects.

Concerning the joint distribution of studies and proportion of MB/MF methods employed, as illustrated in Figure 2(c), there is a significant scarcity of studies predicting motion intentions for the shoulder joint, wrist-elbow joint combination, and the entire upper limb. Additionally, among all adopted 186 studies, only approximately 20% have involved the MB method. Therefore, future research should not only further explore the aforementioned shoulder joint and joint combinations but also intensify efforts toward MB research to enhance the potential for discovering superior upper limb motion intention prediction algorithms, even extending to lower limb prediction algorithms.

Regarding the algorithm feasibility in practical applications, since offline prediction performance is not directly correlated

WEI et al.: Continuous Motion Intention Prediction Using sEMG Signals for Upper-Limb Rehabilitation: A Systematic Review of Model-Based and Model-Free Approaches.

to real-time prediction capability [87] and it is currently challenging to distinguish between the subjects' adaptability and the genuine contribution of the algorithm during real-time prediction [37], greater emphasis should be placed on the realtime predictive performance of future algorithms. However, as Figure 3 illustrates, over three-quarters of studies involved only offline analysis without real-time validation. In addition, since the predictive performance of algorithms can also be influenced by motor noise from exoskeletons worn by subjects [188] and the contact forces between robots and patients in human-robot collaboration [69], necessitating more rigorous robustness tests in practical human-machine interaction environments.

## 2) MB and MF Methods

Regarding MB methods, it is crucial to further balance the real-time predictive performance of MSK models with the complexity of model parameters, while also considering changes in muscle stiffness. Therefore, sensitivity analysis could be conducted to identify the significance of each MSK parameter, such as the tendon length proportion factor and tendon length in the Hill model, which significantly impact predictive performance, in contrast to the pennation angle [24]. MSK parameter optimization methods also need enhancement, such as the GA with lengthy optimization process and tendency to converge to local optima. Therefore, beyond developing new optimization algorithms superior to GA, GO, Simulated Annealing (SA), and PSO, incremental and online learning could also be employed for real-time updating of MSK model parameters. Regarding MF methods, since the Bi-LSTM and LSTM models that cannot be trained in parallel have lengthy training times, and the Transformer model that can be trained in parallel has high computational costs, the Hill model with inherent causality could replace them to establish more robust contextual relationships. In addition, current MB-MF methods can be improved, such as by upgrading the CNN in [29] to CNN-LSTM and adding more Hill model-based physiological constraints to the loss function while considering the weight distribution between CNN loss and physiological loss.

## 3) Muscle Synergy Features

The MS feature extraction methods used in current studies require further improvement. In addition to effectively suppressing erroneous estimations for inactive joint activations, enhancing the correlations across days and subjects of MS features is essential. Additionally, the coordinated movement caused by mechanical coupling between joint-tendon structures and skin should be considered [144], and real-time tests should be conducted in multi-joint complex movements with potential coupling among DoFs [166]. Furthermore, the challenges posed by different MU activation patterns during concentric and eccentric contractions need to be addressed [84].

## 4) Motor Unit Neural Features

Current MU decomposition and MU pool refinement still require improvement. Studies [44]–[49], [105] have indicated that utilizing all MUs may obscure key information in dominant MUs, leading to prediction errors. Therefore, it is crucial to initially increase the number of decomposed MUs, followed by accurately identifying dominant MUs of joint movements. However, the process of MU decomposition is time-consuming.

Although the pre-calculated MU separation matrices can reduce computation time, their performance degrades over time. Hence, exploring incremental and online learning for MU separation matrices is imperative. In addition, the similarity in movement patterns caused by extreme MU discharge rates also reduces predictive performance, necessitating the improvement of MU feature robustness in complex and extreme movements.

## 5) Transfer Learning

Current transfer learning approaches are still limited by computational costs and hardware reset during recalibration [101], and online Domain Adaptation (DA) methods still suffer from significant delays [71]. In addition, only few studies have considered the impact of intra-subject variability on model generalizability. Future research could explore unsupervised transfer learning strategies based on MSK physical constraints, and employ unlabeled data collected during daily practices for online parameter optimization and online transfer learning [30].

## V. CONCLUSION AND FUTURE WORK

This review has comprehensively surveyed the studies conducted over the past decade on the continuous prediction of motion intentions for upper limb single joints and multi-joint combinations, detailing the MB and MF methods used in these scenarios. It is evident that integrating the strengths of MB and MF methods for prediction represents the future research trend. Moreover, to inspire future research, this review discussed seven significant findings from past studies and the five major challenges currently faced in this field. It suggested that beyond refining subject structure, experimental content, and feasibility of algorithms in practical applications, it is also essential to enhance the robustness and generalizability of algorithms based on the physiological nature of motion intention generation and transmission, particularly focusing on superior MS and MU neural feature extraction. Therefore, future research can focus on the following aspects refer to Section IV and Table I:

1) Extracting MS features from the perspective of physiological structure and refined dominant MUs.

2) Establishing the biomechanical contextual relationship among MS features using personalized MSK models.

3) Implementing closed-loop feedback based on KFs, redundant sEMG information, and multi-sensor fusion.

4) Employing real-time parameter update mechanism based on incremental learning and online learning.

Regarding the limitations of this review, it may have overlooked equally valuable publications from earlier periods or other databases. Additionally, it may have potentially overlooked studies published in other languages and those currently under review.

Overall, in the author's view, establishing a robust, precise, low-latency, and long-duration mapping between sEMG features and motion intentions is fundamentally crucial for practical motion intention prediction. Furthermore, in addition to employing one-to-one and one-to-many transfer learning across subjects to reduce training time, generic models developed from multi-subject training, or even model libraries akin to large language models, could be utilized in many-to-one and many-to-many new subject personalization scenarios.

## VI. REFERENCES

[1] B. Sheng, J. Zhao, Y. Zhang, S. Xie, and J. Tao, “Commercial device-based hand rehabilitation systems for stroke patients: State of the art and future prospects,” Heliyon, vol. 9, no. 3, Mar. 2023, doi: 10.1016/J.HELIYON.2023.E13588.

[2] A. A. Blank, J. A. French, A. U. Pehlivan, and M. K. O’Malley, “Current Trends in Robot-Assisted Upper-Limb Stroke Rehabilitation: Promoting Patient Engagement in Therapy,” Curr Phys Med Rehabil Rep, vol. 2, no. 3, p. 184, Sep. 2014, doi: 10.1007/S40141-014-0056-Z.

[3] J. Berning, G. E. Francisco, S. H. Chang, B. J. Fregly, and M. K. O’Malley, “Myoelectric control and neuromusculoskeletal modeling: Complementary technologies for rehabilitation robotics,” Curr Opin Biomed Eng, vol. 19, p. 100313, Sep. 2021, doi: 10.1016/J.COBME.2021.100313.

[4] L. Bi, A. Feleke, and C. Guan, “A review on EMG-based motor intention prediction of continuous human upper limb motion for human-robot collaboration,” Biomed Signal Process Control, vol. 51, pp. 113–127, May 2019, doi: 10.1016/J.BSPC.2019.02.011.

[5] D. Xiong, D. Zhang, X. Zhao, and Y. Zhao, “Deep Learning for EMG-based Human-Machine Interaction: A Review,” IEEE/CAA Journal of Automatica Sinica, vol. 8, no. 3, pp. 512–533, Mar. 2021, doi: 10.1109/JAS.2021.1003865.

[6] S. P. Sitole and F. C. S. IV, “Continuous Prediction of Human Joint Mechanics using EMG Signals: A Review of Model-based and Model-free Approaches,” IEEE Trans Med Robot Bionics, 2023, doi: 10.1109/TMRB.2023.3292451.

[7] M. Sartori, D. G. Llyod, and D. Farina, “Neural data-driven musculoskeletal modeling for personalized neurorehabilitation technologies,” IEEE Trans Biomed Eng, vol. 63, no. 5, pp. 879–893, May 2016, doi: 10.1109/TBME.2016.2538296.

[8] C. Pizzolato et al., “Neuromusculoskeletal modeling-based prostheses for recovery after spinal cord injury,” Front Neurorobot, vol. 13, p. 97, Dec. 2019, doi: 10.3389/FNBOT.2019.00097/BIBTEX.

[9] C. Pizzolato et al., “Bioinspired technologies to connect musculoskeletal mechanobiology to the person for training and rehabilitation,” Front Comput Neurosci, vol. 11, p. 96, Oct. 2017, doi: 10.3389/FNCOM.2017.00096/BIBTEX.

[10] J. V. Dos Santos, M. V. M. Ramis, V. M. Oliveira, and R. Z. Azzolin, “A Survey on Mathematical Modeling of Muscle Using for Rehabilitation Systems,” IECON Proceedings (Industrial Electronics Conference), vol. 2019-October, pp. 249–254, Oct. 2019, doi: 10.1109/IECON.2019.8927318.

[11] Y. Ye, C. Liu, N. Zemiti, and C. Yang, “Optimal Feature Selection for EMG-Based Finger Force Estimation Using LightGBM Model,” 2019 28th IEEE International Conference on Robot and Human Interactive Communication, RO-MAN 2019, Oct. 2019, doi: 10.1109/RO-MAN46459.2019.8956453.

[12] X. Hu, H. Zeng, A. Song, and D. Chen, “Robust Continuous Hand Motion Recognition Using Wearable Array Myoelectric Sensor,” IEEE Sens J, vol. 21, no. 18, pp. 20596–20605, Sep. 2021, doi: 10.1109/JSEN.2021.3098120.

[13] A. Salatiello and M. A. Giese, “Continuous Decoding of Daily-Life Hand Movements from Forearm Muscle Activity for Enhanced Myoelectric Control of Hand Prostheses,” Proceedings of the International Joint Conference on Neural Networks, vol. 2021-July, Jul. 2021, doi: 10.1109/IJCNN52387.2021.9533327.

[14] Y. M. Aung, K. Anam, and A. Al-Jumaily, “Continuous prediction of shoulder joint angle in real-time,” International IEEE/EMBS Conference on Neural Engineering, NER, vol. 2015-July, pp. 755– 758, Jul. 2015, doi: 10.1109/NER.2015.7146733.

[15] M. Pang and S. Guo, “A novel method for elbow joint continuous prediction using EMG and musculoskeletal model,” 2013 IEEE International Conference on Robotics and Biomimetics, ROBIO 2013, pp. 1240–1245, 2013, doi: 10.1109/ROBIO.2013.6739634.

[16] M. Pang, S. Guo, Q. Huang, H. Ishihara, and H. Hirata, “Electromyography-based quantitative representation method for upper-limb elbow joint angle in sagittal plane,” J Med Biol Eng, vol. 35, no. 2, pp. 165–177, Apr. 2015, doi: 10.1007/S40846-015-0033- 8/TABLES/3.

[17] Z. Yang, S. Guo, and Y. Liu, “Comparison of Isometric Force Estimation Methods for Upper Limb Elbow Joints,” 2020 IEEE International Conference on Mechatronics and Automation, ICMA

2020, pp. 1558–1563, Oct. 2020, doi: 10.1109/ICMA49215.2020.9233869.

[18] J. Han, Q. Ding, A. Xiong, and X. Zhao, “A state-space EMG model for the estimation of continuous joint movements,” IEEE Transactions on Industrial Electronics, vol. 62, no. 7, pp. 4267– 4275, Jul. 2015, doi: 10.1109/TIE.2014.2387337.

[19] T. Desplenter and A. L. Trejos, “Evaluating Muscle Activation Models for Elbow Motion Estimation,” Sensors 2018, Vol. 18, Page 1004, vol. 18, no. 4, p. 1004, Mar. 2018, doi: 10.3390/S18041004.

[20] K. Li, J. Zhang, H. Zhang, and M. Zhang, “A Novel Method for Estimating Continuous Motion and Time-Varying Stiffness of Human Elbow Joint,” 2018 IEEE International Conference on Robotics and Biomimetics, ROBIO 2018, pp. 299–304, Jul. 2018, doi: 10.1109/ROBIO.2018.8665045.

[21] K. Li, J. Zhang, X. Liu, and M. Zhang, “Estimation of continuous elbow joint movement based on human physiological structure,” Biomed Eng Online, vol. 18, no. 1, pp. 1–15, Mar. 2019, doi: 10.1186/S12938-019-0653-2/FIGURES/10.

[22] D. Xu, Q. Wu, and Y. Zhu, “Development of a sEMG-Based Joint Torque Estimation Strategy Using Hill-Type Muscle Model and Neural Network,” J Med Biol Eng, vol. 41, no. 1, pp. 34–44, Feb. 2021, doi: 10.1007/S40846-020-00539-2/TABLES/5.

[23] V. I. Ramírez-Vera, M. O. Mendoza-Gutiérrez, and I. Bonilla-Gutiérrez, “Elbow Torque Estimation for Human-Robot Interaction Control,” IFMBE Proc, vol. 86, pp. 760–769, 2023, doi: 10.1007/978-3-031-18256-3\_80/FIGURES/6.

[24] Y. Zhao, Z. Zhang, Z. Li, Z. Yang, A. A. Dehghani-Sanij, and S. Xie, “An EMG-Driven Musculoskeletal Model for Estimating Continuous Wrist Motion,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 28, no. 12, pp. 3113–3120, Dec. 2020, doi: 10.1109/TNSRE.2020.3038051.

[25] Y. Zhao, Z. Li, Z. Zhang, K. Qian, and S. Xie, “An EMG-driven musculoskeletal model for estimation of wrist kinematics using mirrored bilateral movement,” Biomed Signal Process Control, vol. 81, p. 104480, Mar. 2023, doi: 10.1016/J.BSPC.2022.104480.

[26] R. Wang, K. Zhu, Z. Zhang, J. Li, and L. Pan, “Extracting Control Information from Multi-channel Surface EMG Signals to Drive a Musculoskeletal Model,” 2021 27th International Conference on Mechatronics and Machine Vision in Practice, M2VIP 2021, pp. 253–257, 2021, doi: 10.1109/M2VIP49856.2021.9665114.

[27] J. Li, R. Wang, and L. Pan, “An enhanced EMG-driven Musculoskeletal model based on non-negative matrix factorization,” Biomed Signal Process Control, vol. 79, p. 104178, Jan. 2023, doi: 10.1016/J.BSPC.2022.104178.

[28] S. Zhang et al., “Muscle strength assessment system using sEMGbased force prediction method for wrist joint,” J Med Biol Eng, vol. 36, no. 1, pp. 121–131, Feb. 2016, doi: 10.1007/S40846-016-0112- 5/FIGURES/8.

[29] J. Zhang et al., “Physics-Informed Deep Learning for Musculoskeletal Modeling: Predicting Muscle Forces and Joint Kinematics From Surface EMG,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 31, pp. 484–493, 2023, doi: 10.1109/TNSRE.2022.3226860.

[30] J. Zhang et al., “Boosting Personalized Musculoskeletal Modeling With Physics-Informed Knowledge Transfer,” IEEE Trans Instrum Meas, vol. 72, 2023, doi: 10.1109/TIM.2022.3227604.

[31] M. Pang, S. Guo, Z. Song, and S. Zhang, “SEMG signal and hill model based continuous prediction for hand grasping motion,” 2013 ICME International Conference on Complex Medical Engineering, CME 2013, pp. 329–333, 2013, doi: 10.1109/ICCME.2013.6548264.

[32] J. Ngeo et al., “Control of an optimal finger exoskeleton based on continuous joint angle estimation from EMG signals,” Proceedings of the Annual International Conference of the IEEE Engineering in Medicine and Biology Society, EMBS, pp. 338–341, 2013, doi: 10.1109/EMBC.2013.6609506.

[33] J. Ngeo, T. Tamei, and T. Shibata, “Estimation of continuous multi-DOF finger joint kinematics from surface EMG using a multi-output Gaussian Process,” 2014 36th Annual International Conference of the IEEE Engineering in Medicine and Biology Society, EMBC 2014, pp. 3537–3540, Nov. 2014, doi: 10.1109/EMBC.2014.6944386.

[34] J. G. Ngeo, T. Tamei, and T. Shibata, “Continuous and simultaneous estimation of finger kinematics using inputs from an EMG-to-muscle activation model,” J Neuroeng Rehabil, vol. 11, no.

1, pp. 1–14, Aug. 2014, doi: 10.1186/1743-0003-11- 122/FIGURES/8.

[35] X. Xu, H. Deng, Y. Zhang, and J. Chen, “Continuous Grasping Force Estimation With Surface EMG Based on Huxley-Type Musculoskeletal Model,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 31, pp. 346–355, 2023, doi: 10.1109/TNSRE.2022.3214866.

[36] D. L. Crouch and H. Huang, “Musculoskeletal model predicts multijoint wrist and hand movement from limited EMG control signals,” Proceedings of the Annual International Conference of the IEEE Engineering in Medicine and Biology Society, EMBS, vol. 2015- November, pp. 1132–1135, Nov. 2015, doi: 10.1109/EMBC.2015.7318565.

[37] L. Pan, D. L. Crouch, and H. Huang, “Myoelectric Control Based on a Generic Musculoskeletal Model: Toward a Multi-User Neural-Machine Interface,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 26, no. 7, pp. 1435–1442, Jul. 2018, doi: 10.1109/TNSRE.2018.2838448.

[38] L. Pan, A. Harmody, and H. Huang, “A Reliable Multi-User EMG Interface Based on A Generic-Musculoskeletal Model against Loading Weight Changes ∗,” Proceedings of the Annual International Conference of the IEEE Engineering in Medicine and Biology Society, EMBS, vol. 2018-July, pp. 2104–2107, Oct. 2018, doi: 10.1109/EMBC.2018.8512685.

[39] L. Pan, D. L. Crouch, and H. Huang, “Comparing EMG-based human-machine interfaces for estimating continuous, coordinated movements,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 27, no. 10, pp. 2145–2154, Oct. 2019, doi: 10.1109/TNSRE.2019.2937929.

[40] L. Pan, Z. Ding, and J. Li, “Comparing EMG Continuous Movement Decoding With Joints Unconstrained and Constrained,” IEEE Robot Autom Lett, vol. 7, no. 4, pp. 9613–9619, Oct. 2022, doi: 10.1109/LRA.2022.3191533.

[41] Y. Kim, S. Stapornchaisit, H. Kambara, N. Yoshimura, and Y. Koike, “Muscle Synergy and Musculoskeletal Model-Based Continuous Multi-Dimensional Estimation of Wrist and Hand Motions,” J Healthc Eng, vol. 2020, 2020, doi: 10.1155/2020/5451219.

[42] J. Zhao, Y. Yu, X. Sheng, and X. Zhu, “Extracting Stable Control Information from EMG Signals to Drive a Musculoskeletal Model - A Preliminary Study,” Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), vol. 13456 LNAI, pp. 735–746, 2022, doi: 10.1007/978-3-031-13822-5\_66/TABLES/1.

[43] Z. He, Z. Qin, and Y. Koike, “Continuous Estimation of Finger and Wrist Joint Angles Using a Muscle Synergy Based Musculoskeletal Model,” Applied Sciences (Switzerland), vol. 12, no. 8, p. 3772, Apr. 2022, doi: 10.3390/APP12083772/S1.

[44] R. Roy, D. G. Kamper, and X. Hu, “Concurrent Decoding of Finger Kinematic and Kinetic Variables based on Motor Unit Discharges,” Proceedings of the 2022 IEEE International Conference on Human-Machine Systems, ICHMS 2022, 2022, doi: 10.1109/ICHMS56717.2022.9980636.

[45] Y. Zheng and X. Hu, “Dexterous Force Estimation during Finger Flexion and Extension Using Motor Unit Discharge Information,” Proceedings of the Annual International Conference of the IEEE Engineering in Medicine and Biology Society, EMBS, vol. 2020- July, pp. 3130–3133, Jul. 2020, doi: 10.1109/EMBC44109.2020.9175236.

[46] F. Xu, Y. Zheng, and X. Hu, “Estimation of joint kinematics and fingertip forces using motoneuron firing activities: A preliminary report,” International IEEE/EMBS Conference on Neural Engineering, NER, vol. 2021-May, pp. 1035–1038, May 2021, doi: 10.1109/NER49283.2021.9441433.

[47] R. Roy, Y. Zheng, D. G. Kamper, and X. Hu, “Concurrent and Continuous Prediction of Finger Kinetics and Kinematics via Motoneuron Activities,” IEEE Trans Biomed Eng, vol. 70, no. 6, pp. 1911–1920, Jun. 2023, doi: 10.1109/TBME.2022.3232067.

[48] R. Roy, D. G. Kamper, and X. Hu, “Optimized Model Selection for Concurrent Decoding of Finger Kinetics and Kinematics,” IEEE Access, vol. 11, pp. 17348–17358, 2023, doi: 10.1109/ACCESS.2023.3246950.

[49] N. Rubin, Y. Zheng, H. Huang, and X. Hu, “Finger Force Estimation Using Motor Unit Discharges Across Forearm Postures,”

IEEE Trans Biomed Eng, vol. 69, no. 9, pp. 2767–2775, Sep. 2022, doi: 10.1109/TBME.2022.3153448.

[50] S. Yue, J. Li, and L. Pan, “A Neural-Driven Musculoskeletal Model for Continuous Estimation of Hand and Wrist Movements,” 2022 12th International Conference on CYBER Technology in Automation, Control, and Intelligent Systems, CYBER 2022, pp. 856–860, 2022, doi: 10.1109/CYBER55403.2022.9907258.

[51] J. Berman, R. Hinson, I. C. Lee, and H. Huang, “Harnessing Machine Learning and Physiological Knowledge for a Novel EMG-Based Neural-Machine Interface,” IEEE Trans Biomed Eng, vol. 70, no. 4, pp. 1125–1136, Apr. 2023, doi: 10.1109/TBME.2022.3210892.

[52] W. Wu, K. R. Saul, and H. Huang, “Using reinforcement learning to estimate human joint moments from electromyography or joint kinematics: An alternative solution to musculoskeletal-based biomechanics,” J Biomech Eng, vol. 143, no. 4, Apr. 2021, doi: 10.1115/1.4049333/1092397.

[53] J. Berman, R. Hinson, and H. Huang, “Comparing Reinforcement Learning Agents and Supervised Learning Neural Networks for EMG-Based Decoding of Continuous Movements,” Proceedings of the Annual International Conference of the IEEE Engineering in Medicine and Biology Society, EMBS, pp. 6297–6300, 2021, doi: 10.1109/EMBC46164.2021.9630744.

[54] L. Sun, H. An, H. Ma, and J. Gao, “Real-Time Human Intention Recognition of Multi-Joints Based on MYO,” IEEE Access, vol. 8, pp. 4235–4243, 2020, doi: 10.1109/ACCESS.2019.2942108.

[55] X. Xi, C. Yang, S. M. Miran, Y. B. Zhao, S. Lin, and Z. Luo, “SEMG-MMG State-Space Model for the Continuous Estimation of Multijoint Angle,” Complexity, vol. 2020, 2020, doi: 10.1155/2020/4503271.

[56] A. Ehrampoosh, A. Yousefi-Koma, S. S. Mohtasebi, and M. Ayati, “EMG-based estimation of shoulder kinematic using neural network and quadratic discriminant analysis,” 4th RSI International Conference on Robotics and Mechatronics, ICRoM 2016, pp. 471– 476, Mar. 2017, doi: 10.1109/ICROM.2016.7886786.

[57] T. Zhang, H. Sun, and Y. Zou, “An electromyography signals-based human-robot collaboration system for human motion intention recognition and realization,” Robot Comput Integr Manuf, vol. 77, p. 102359, Oct. 2022, doi: 10.1016/J.RCIM.2022.102359.

[58] Z. Song and S. Zhang, “Preliminary Study on Continuous Recognition of Elbow Flexion/Extension Using sEMG Signals for Bilateral Rehabilitation,” Sensors 2016, Vol. 16, Page 1739, vol. 16, no. 10, p. 1739, Oct. 2016, doi: 10.3390/S16101739.

[59] L. F. Sommer, C. Barreira, C. Noriega, F. Camargo-Junior, R. T. Moura, and A. Forner-Cordero, “Elbow Joint Angle Estimation with Surface Electromyography Using Autoregressive Models,” Proceedings of the Annual International Conference of the IEEE Engineering in Medicine and Biology Society, EMBS, vol. 2018- July, pp. 1472–1475, Oct. 2018, doi: 10.1109/EMBC.2018.8512512.

[60] Y. Chen, X. Zhao, and J. Han, “Hierarchical projection regression for online estimation of elbow joint angle using EMG signals,” Neural Comput Appl, vol. 23, no. 3–4, pp. 1129–1138, Sep. 2013, doi: 10.1007/S00521-012-1045-8/FIGURES/12.

[61] F. Xiao, Y. Wang, Y. Gao, Y. Zhu, and J. Zhao, “Continuous estimation of joint angle from electromyography using multiple time-delayed features and random forests,” Biomed Signal Process Control, vol. 39, pp. 303–311, Jan. 2018, doi: 10.1016/J.BSPC.2017.08.015.

[62] F. Xiao, Y. Wang, Y. Gao, Y. Zhu, and J. Zhao, “Continuous Estimation of Elbow Joint Angle by Multiple Features of Surface Electromyographic Using Grey Features Weighted Support Vector Machine,” J Med Imaging Health Inform, vol. 7, no. 3, pp. 574–583, Jun. 2017, doi: 10.1166/JMIHI.2017.2054.

[63] R. Raj, R. Rejith, and K. S. Sivanandan, “Real Time Identification of Human Forearm Kinematics from Surface EMG Signal Using Artificial Neural Network Models,” Procedia Technology, vol. 25, pp. 44–51, Jan. 2016, doi: 10.1016/J.PROTCY.2016.08.079.

[64] Q. Ding, X. Zhao, and J. Han, “A hybrid EMG model for the estimation of multijoint movement in activities of daily living,” Proceedings of 2014 International Conference on Multisensor Fusion and Information Integration for Intelligent Systems, MFI 2014, Dec. 2014, doi: 10.1109/MFI.2014.6997746.

[65] Y. Luo, Y. Gao, Q. Li, J. Zhao, and X. Fei, “Elbow joint angle estimation in wavelet neural network using relieff selected features

of sEMG and post filter,” Proceedings of 2018 IEEE International Conference on Mechatronics and Automation, ICMA 2018, pp. 1287–1292, Oct. 2018, doi: 10.1109/ICMA.2018.8484538.

[66] S. Guo, Z. Yang, and Y. Liu, “EMG-based Continuous Prediction of the Upper Limb Elbow Joint Angle Using GRNN,” Proceedings of 2019 IEEE International Conference on Mechatronics and Automation, ICMA 2019, pp. 2168–2173, Aug. 2019, doi: 10.1109/ICMA.2019.8816401.

[67] H. Li, S. Guo, D. Bu, and H. Wang, “Inter-subject Variability Evaluation of Continuous Elbow Angle from sEMG using BPNN,” 2022 IEEE International Conference on Mechatronics and Automation, ICMA 2022, pp. 388–393, 2022, doi: 10.1109/ICMA54519.2022.9856005.

[68] H. Li, S. Guo, D. Bu, and H. Wang, “A Two-Stage GA-Based sEMG Feature Selection Method for User-Independent Continuous Estimation of Elbow Angles,” IEEE Trans Instrum Meas, vol. 72, 2023, doi: 10.1109/TIM.2023.3276522.

[69] Y. Huang, K. Chen, X. Zhang, K. Wang, and J. Ota, “Motion estimation of elbow joint from sEMG using continuous wavelet transform and back propagation neural networks,” Biomed Signal Process Control, vol. 68, p. 102657, Jul. 2021, doi: 10.1016/J.BSPC.2021.102657.

[70] Z. Lei, “An upper limb movement estimation from electromyography by using BP neural network,” Biomed Signal Process Control, vol. 49, pp. 434–439, Mar. 2019, doi: 10.1016/J.BSPC.2018.12.020.

[71] H. Li, S. Guo, H. Wang, and D. Bu, “Subject-Independent Continuous Estimation of sEMG-Based Joint Angles Using Both Multisource Domain Adaptation and BP Neural Network,” IEEE Trans Instrum Meas, vol. 72, 2023, doi: 10.1109/TIM.2022.3225015.

[72] Y. Du, J. Jin, Q. Wang, and J. Fan, “EMG-Based Continuous Motion Decoding of Upper Limb with Spiking Neural Network,” Conference Record - IEEE Instrumentation and Measurement Technology Conference, 2022, doi: 10.1109/I2MTC48687.2022.9806710.

[73] Triwiyanto, O. Wahyunggoro, H. A. Nugroho, and Herianto, “Time Delay Neural Network to Estimate the Elbow Joint Angle Based on Electromyography,” Proceedings of 2018 2nd International Conference on Biomedical Engineering: Smart Technology for Better Society, IBIOMED 2018, pp. 100–105, Nov. 2018, doi: 10.1109/IBIOMED.2018.8534939.

[74] A. Ehrampoosh, A. Yousefi-Koma, and M. Ayati, “Development of myoelectric interface based on pattern recognition and regression based models,” 2016 Artificial Intelligence and Robotics, IRANOPEN 2016, pp. 145–150, Aug. 2016, doi: 10.1109/RIOS.2016.7529505.

[75] R. Raj, R. Ramakrishna, and K. S. Sivanandan, “A real time surface electromyography signal driven prosthetic hand model using PID controlled DC motor,” Biomed Eng Lett, vol. 6, no. 4, pp. 276–286, Nov. 2016, doi: 10.1007/S13534-016-0240-4/METRICS.

[76] R. Raj and K. S. Sivanandan, “Comparative study on estimation of elbow kinematics based on EMG time domain parameters using neural network and&nbsp;ANFIS NARX model,” Journal of Intelligent & Fuzzy Systems, vol. 32, no. 1, pp. 791–805, Jan. 2017, doi: 10.3233/JIFS-16070.

[77] P. Rajalakshmy, E. Jacob, and T. Joclyn Sharon, “Estimation of elbow joint angle from surface electromyogram signals using ANFIS,” Lecture Notes in Computational Vision and Biomechanics, vol. 31, pp. 247–253, 2019, doi: 10.1007/978-3-030-04061- 1\_25/FIGURES/5.

[78] Y. Chai, K. Liu, Z. Sun, G. Wang, and T. Shi, “A Long Short Term Memory Network Based on Surface Electromyography for Continuous Estimation of Elbow Joint Angle,” Proceedings of 2020 IEEE 9th Data Driven Control and Learning Systems Conference, DDCLS 2020, pp. 522–527, Nov. 2020, doi: 10.1109/DDCLS49620.2020.9275122.

[79] V. del C. Silva-Acosta, I. Román-Godínez, S. Torres-Ramos, and R. A. Salido-Ruiz, “Automatic estimation of continuous elbow flexion–extension movement based on electromyographic and electroencephalographic signals,” Biomed Signal Process Control, vol. 70, p. 102950, Sep. 2021, doi: 10.1016/J.BSPC.2021.102950.

[80] L. Xu, X. Chen, S. Cao, X. Zhang, and X. Chen, “Feasibility Study of Advanced Neural Networks Applied to sEMG-Based Force

Estimation,” Sensors 2018, Vol. 18, Page 3226, vol. 18, no. 10, p. 3226, Sep. 2018, doi: 10.3390/S18103226.

[81] D. Zhao et al., “MCR-ALS-based muscle synergy extraction method combined with LSTM neural network for motion intention detection,” Front Neurorobot, vol. 17, p. 1174710, Jun. 2023, doi: 10.3389/FNBOT.2023.1174710/BIBTEX.

[82] T. Triwiyanto, O. Wahyunggoro, H. A. Nugroho, and H. Herianto, “Evaluating the performance of Kalman filter on elbow joint angle prediction based on electromyography,” International Journal of Precision Engineering and Manufacturing, vol. 18, no. 12, pp. 1739–1748, Dec. 2017, doi: 10.1007/S12541-017-0202- 5/METRICS.

[83] L. Lu, Q. Wu, X. Chen, Z. Shao, B. Chen, and H. Wu, “Development of a sEMG-based torque estimation control strategy for a soft elbow exoskeleton,” Rob Auton Syst, vol. 111, pp. 88–98, Jan. 2019, doi: 10.1016/J.ROBOT.2018.10.017.

[84] J. Hashemi, E. Morin, P. Mousavi, and K. Hashtrudi-Zaad, “Enhanced dynamic EMG-force estimation through calibration and PCI modeling,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 23, no. 1, pp. 41–50, Jan. 2015, doi: 10.1109/TNSRE.2014.2325713.

[85] L. F. Sommer and A. Forner-Cordero, “Elbow Joint Angle Estimation from Surface Electromyography Using Hammerstein-Wiener Models,” Proceedings of the IEEE RAS and EMBS International Conference on Biomedical Robotics and Biomechatronics, vol. 2018-August, pp. 1044–1049, Oct. 2018, doi: 10.1109/BIOROB.2018.8487940.

[86] H. J. Hwang, J. Mathias Hahne, and K. R. Müller, “Channel selection for simultaneous and proportional myoelectric prosthesis control of multiple degrees-of-freedom,” J Neural Eng, vol. 11, no. 5, p. 056008, Aug. 2014, doi: 10.1088/1741-2560/11/5/056008.

[87] N. Jiang, I. Vujaklija, H. Rehbaum, B. Graimann, and D. Farina, “Is accurate mapping of EMG signals on kinematics needed for precise online myoelectric control?,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 22, no. 3, pp. 549–558, 2014, doi: 10.1109/TNSRE.2013.2287383.

[88] Y. Gao, Y. Luo, Q. Li, and J. Zhao, “Continuous Joint Angle Estimation by Least Support Vector Machine from Time-Delayed sEMG Features,” ACM International Conference Proceeding Series, pp. 13–17, May 2018, doi: 10.1145/3232059.3232071.

[89] D. Bu, S. Guo, and W. Gao, “Continuous Estimation of a sEMG-Based Upper Limb Joint,” Proceedings of 2019 IEEE International Conference on Mechatronics and Automation, ICMA 2019, pp. 904– 909, Aug. 2019, doi: 10.1109/ICMA.2019.8816458.

[90] Y. Liu et al., “Wrist angle prediction under different loads based on GA-ELM neural network and surface electromyography,” Concurr Comput, vol. 34, no. 3, p. e6574, Feb. 2022, doi: 10.1002/CPE.6574.

[91] E. N. Kamavuako, E. J. Scheme, and K. B. Englehart, “Wrist torque estimation during simultaneous and continuously changing movements: Surface vs. untargeted intramuscular EMG,” J Neurophysiol, vol. 109, no. 11, pp. 2658–2665, 2013, doi: 10.1152/JN.00086.2013/ASSET/IMAGES/LARGE/Z9K011131949 0007.JPEG.

[92] X. Sun, X. Zhang, Z. Lu, R. Li, H. Li, and T. Zhang, “SEMG Based Continuous Estimation of Wrist Joint Angle using BP Neural Network,” 9th IEEE International Conference on Cyber Technology in Automation, Control and Intelligent Systems, CYBER 2019, pp. 221–225, Jul. 2019, doi: 10.1109/CYBER46603.2019.9066565.

[93] L. Zhang, J. Long, R. G. Zhao, H. Cao, and K. Zhang, “Estimation of the Continuous Pronation–Supination Movement by Using Multichannel EMG Signal Features and Kalman Filter: Application to Control an Exoskeleton,” Front Bioeng Biotechnol, vol. 9, p. 771255, Mar. 2022, doi: 10.3389/FBIOE.2021.771255/BIBTEX.

[94] Y. Yu, C. Chen, X. Sheng, and X. Zhu, “Multi-DoF continuous estimation for wrist torques using stacked autoencoder,” Biomed Signal Process Control, vol. 57, p. 101733, Mar. 2020, doi: 10.1016/J.BSPC.2019.101733.

[95] Y. Yu, C. Chen, X. Sheng, and X. Zhu, “Continuous estimation of wrist torques with stack-autoencoder based deep neural network: A preliminary study,” International IEEE/EMBS Conference on Neural Engineering, NER, vol. 2019-March, pp. 473–476, May 2019, doi: 10.1109/NER.2019.8716941.

[96] T. Bao, A. Zaidi, S. Xie, and Z. Zhang, “Surface-EMG based wrist kinematics estimation using convolutional neural network,” 2019

IEEE 16th International Conference on Wearable and Implantable Body Sensor Networks, BSN 2019 - Proceedings, May 2019, doi: 10.1109/BSN.2019.8771100.

[97] Y. Yu, C. Chen, J. Zhao, X. Sheng, and X. Zhu, “Surface Electromyography Image-Driven Torque Estimation of Multi-DoF Wrist Movements,” IEEE Transactions on Industrial Electronics, vol. 69, no. 1, pp. 795–804, Jan. 2022, doi: 10.1109/TIE.2021.3050367.

[98] Y. Zhang, X. Zhang, Z. Lu, Z. Jiang, and T. Zhang, “A Novel Wrist Joint Torque Prediction Method Based on EMG and LSTM,” 10th IEEE International Conference on Cyber Technology in Automation, Control and Intelligent Systems, CYBER 2020, pp. 242–245, Oct. 2020, doi: 10.1109/CYBER50695.2020.9279119.

[99] T. Bao, S. A. R. Zaidi, S. Xie, P. Yang, and Z. Q. Zhang, “A CNN-LSTM Hybrid Model for Wrist Kinematics Estimation Using Surface Electromyography,” IEEE Trans Instrum Meas, vol. 70, 2021, doi: 10.1109/TIM.2020.3036654.

[100] T. Bao, C. Wang, P. Yang, S. Q. Xie, Z. Q. Zhang, and P. Zhou, “LSTM-AE for Domain Shift Quantification in Cross-Day Upper-Limb Motion Estimation Using Surface Electromyography,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 31, pp. 2570–2580, 2023, doi: 10.1109/TNSRE.2023.3281455.

[101] T. Bao, S. A. R. Zaidi, S. Xie, P. Yang, and Z. Q. Zhang, “Inter-Subject Domain Adaptation for CNN-Based Wrist Kinematics Estimation Using sEMG,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 29, pp. 1068–1078, 2021, doi: 10.1109/TNSRE.2021.3086401.

[102] T. Kapelner et al., “Predicting wrist kinematics from motor unit discharge timings for the control of active prostheses,” J Neuroeng Rehabil, vol. 16, no. 1, pp. 1–11, Apr. 2019, doi: 10.1186/S12984- 019-0516-X/FIGURES/7.

[103] Y. Yu, C. Chen, X. Sheng, and X. Zhu, “A Motor Unit-specific Images Based Scheme for Continuous Estimation of Wrist Torques - A Pilot Study,” Conf Proc IEEE Int Conf Syst Man Cybern, vol. 2020-October, pp. 3182–3187, Oct. 2020, doi: 10.1109/SMC42975.2020.9283000.

[104] C. Chen, Y. Yu, X. Sheng, and X. Zhu, “Non-Invasive Analysis of Motor Unit Activation During Simultaneous and Continuous Wrist Movements,” IEEE J Biomed Health Inform, vol. 26, no. 5, pp. 2106–2115, May 2022, doi: 10.1109/JBHI.2021.3135575.

[105] C. Chen, Y. Yu, X. Sheng, J. Meng, and X. Zhu, “Mapping Individual Motor Unit Activity to Continuous Three-DoF Wrist Torques: Perspectives for Myoelectric Control,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 31, pp. 1807–1815, 2023, doi: 10.1109/TNSRE.2023.3260209.

[106] N. Jaquier and S. Calinon, “Gaussian mixture regression on symmetric positive definite matrices manifolds: Application to wrist motion estimation with sEMG,” IEEE International Conference on Intelligent Robots and Systems, vol. 2017-September, pp. 59–64, Dec. 2017, doi: 10.1109/IROS.2017.8202138.

[107] K. Bakshi, M. Manjunatha, and C. S. Kumar, “Estimation of continuous and constraint-free 3 DoF wrist movements from surface electromyogram signal using kernel recursive least square tracker,” Biomed Signal Process Control, vol. 46, pp. 104–115, Sep. 2018, doi: 10.1016/J.BSPC.2018.06.012.

[108] L. Pan, X. Sheng, D. Zhang, and X. Zhu, “Simultaneous and proportional estimation of finger joint angles from surface EMG signals during mirrored bilateral movements,” Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), vol. 8102 LNAI, no. PART 1, pp. 493–499, 2013, doi: 10.1007/978-3-642-40852- 6\_50/COVER.

[109] L. Pan, D. Zhang, J. Liu, X. Sheng, and X. Zhu, “Continuous estimation of finger joint angles under different static wrist motions from surface EMG signals,” Biomed Signal Process Control, vol. 14, pp. 265–271, Nov. 2014, doi: 10.1016/J.BSPC.2014.08.004.

[110] B. Dutra, A. Silveira, and A. Pereira, “Grasping force estimation using state-space model and Kalman Filter,” Biomed Signal Process Control, vol. 70, p. 103036, Sep. 2021, doi: 10.1016/J.BSPC.2021.103036.

[111] P. Liu, D. R. Brown, E. A. Clancy, F. Martel, and D. Rancourt, “EMG-force estimation for multiple fingers,” 2013 IEEE Signal Processing in Medicine and Biology Symposium, SPMB 2013, 2013, doi: 10.1109/SPMB.2013.6736772.

[112] A. Gijsberts et al., “Stable myoelectric control of a hand prosthesis using non-linear incremental learning,” Front Neurorobot, vol. 8, no. FEB, p. 74950, Feb. 2014, doi: 10.3389/FNBOT.2014.00008/ABSTRACT.

[113] A. Krasoulis, S. Vijayakumar, and K. Nazarpour, “Evaluation of regression methods for the continuous decoding of finger movement from surface EMG and accelerometry,” International IEEE/EMBS Conference on Neural Engineering, NER, vol. 2015-July, pp. 631– 634, Jul. 2015, doi: 10.1109/NER.2015.7146702.

[114] M. Xiloyannis, C. Gavriel, A. A. C. Thomik, and A. A. Faisal, “Dynamic forward prediction for prosthetic hand control by integration of EMG, MMG and kinematic signals,” International IEEE/EMBS Conference on Neural Engineering, NER, vol. 2015- July, pp. 611–614, Jul. 2015, doi: 10.1109/NER.2015.7146697.

[115] M. Xiloyannis, C. Gavriel, A. A. C. Thomik, and A. A. Faisal, “Gaussian Process Autoregression for Simultaneous Proportional Multi-Modal Prosthetic Control with Natural Hand Kinematics,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 25, no. 10, pp. 1785–1801, Oct. 2017, doi: 10.1109/TNSRE.2017.2699598.

[116] Q. Zhang, T. Pi, R. Liu, and C. Xiong, “Simultaneous and Proportional Estimation of Multijoint Kinematics from EMG Signals for Myocontrol of Robotic Hands,” IEEE/ASME Transactions on Mechatronics, vol. 25, no. 4, pp. 1953–1960, Aug. 2020, doi: 10.1109/TMECH.2020.2999532.

[117] J. Ngeo, T. Tamei, K. Ikeda, and T. Shibata, “Modeling dynamic high-DOF finger postures from surface EMG using nonlinear synergies in latent space representation,” Proceedings of the Annual International Conference of the IEEE Engineering in Medicine and Biology Society, EMBS, vol. 2015-November, pp. 2095–2098, Nov. 2015, doi: 10.1109/EMBC.2015.7318801.

[118] X. Li et al., “Simultaneous hand/wrist motion recognition and continuous grasp force estimation based on nonlinear spectral sEMG features for transradial amputees,” Biomed Signal Process Control, vol. 85, p. 105044, Aug. 2023, doi: 10.1016/J.BSPC.2023.105044.

[119] R. Ma, L. Zhang, G. Li, D. Jiang, S. Xu, and D. Chen, “Grasping force prediction based on sEMG signals,” Alexandria Engineering Journal, vol. 59, no. 3, pp. 1135–1147, Jun. 2020, doi: 10.1016/J.AEJ.2020.01.007.

[120] H. Mao, P. Fang, and G. Li, “Simultaneous estimation of multifinger forces by surface electromyography and accelerometry signals,” Biomed Signal Process Control, vol. 70, p. 103005, Sep. 2021, doi: 10.1016/J.BSPC.2021.103005.

[121] H. Mao et al., “Continuous grip force estimation from surface electromyography using generalized regression neural network,” Technology and Health Care, vol. 31, no. 2, pp. 675–689, Jan. 2023, doi: 10.3233/THC-220283.

[122] W. Batayneh, E. Abdulhay, and M. Alothman, “Comparing the efficiency of artificial neural networks in sEMG-based simultaneous and continuous estimation of hand kinematics,” Digital Communications and Networks, vol. 8, no. 2, pp. 162–173, Apr. 2022, doi: 10.1016/J.DCAN.2021.08.002.

[123] W. Hai, T. Qing, S. Na, and Z. Xiaodong, “Simultaneous Estimation of Hand Joints’ Angles Toward sEMG-Driven Human-Robot Interaction,” IEEE Access, vol. 10, pp. 109385–109394, 2022, doi: 10.1109/ACCESS.2022.3212046.

[124] J. Xue and K. W. C. Lai, “Dynamic gripping force estimation and reconstruction in EMG-based human-machine interaction,” Biomed Signal Process Control, vol. 80, p. 104216, Feb. 2023, doi: 10.1016/J.BSPC.2022.104216.

[125] Q. Li, S. Xu, and A. Zhang, “Continuous prediction of finger joint angles based on time series feature fusion CNN,” ACM International Conference Proceeding Series, pp. 902–906, Dec. 2022, doi: 10.1145/3584376.3584537.

[126] R. C. Simpetru, M. Osswald, D. I. Braun, D. S. Oliveira, A. L. Cakici, and A. Del Vecchio, “Accurate Continuous Prediction of 14 Degrees of Freedom of the Hand from Myoelectrical Signals through Convolutive Deep Learning,” Proceedings of the Annual International Conference of the IEEE Engineering in Medicine and Biology Society, EMBS, vol. 2022-July, pp. 702–706, 2022, doi: 10.1109/EMBC48229.2022.9870937.

[127] R. C. S. Sˆımpetru et al., “Sensing the Full Dynamics of the Human Hand with a Neural Interface and Deep Learning,” bioRxiv, p. 2022.07.29.502064, Aug. 2022, doi: 10.1101/2022.07.29.502064.

[128] J. Fan, L. Vargas, D. G. Kamper, and X. Hu, “Robust neural decoding for dexterous control of robotic hand kinematics,” Comput Biol Med, vol. 162, p. 107139, Aug. 2023, doi: 10.1016/J.COMPBIOMED.2023.107139.

[129] R. Roy, F. Xu, D. G. Kamper, and X. Hu, “A generic neural network model to estimate populational neural activity for robust neural decoding,” Comput Biol Med, vol. 144, p. 105359, May 2022, doi: 10.1016/J.COMPBIOMED.2022.105359.

[130] F. Xu, Y. Zheng, and X. Hu, “Real-time finger force prediction via parallel convolutional neural networks: A preliminary study,” Proceedings of the Annual International Conference of the IEEE Engineering in Medicine and Biology Society, EMBS, vol. 2020- July, pp. 3126–3129, Jul. 2020, doi: 10.1109/EMBC44109.2020.9175390.

[131] C. Ma, W. Guo, L. Xu, and G. Li, “Finger joint angle estimation based on sEMG signals and deep learning method,” 2021 IEEE International Conference on Real-Time Computing and Robotics, RCAR 2021, pp. 638–643, Jul. 2021, doi: 10.1109/RCAR52367.2021.9517643.

[132] C. Ma et al., “A Novel and Efficient Feature Extraction Method for Deep Learning Based Continuous Estimation,” IEEE Robot Autom Lett, vol. 6, no. 4, pp. 7341–7348, Oct. 2021, doi: 10.1109/LRA.2021.3097257.

[133] M. Zanghieri et al., “SEMG-based Regression of Hand Kinematics with Temporal Convolutional Networks on a Low-Power Edge Microcontroller,” 2021 IEEE International Conference on Omni-Layer Intelligent Systems, COINS 2021, Aug. 2021, doi: 10.1109/COINS51742.2021.9524188.

[134] C. Chen, W. Guo, C. Ma, Y. Yang, Z. Wang, and C. Lin, “sEMG-Based Continuous Estimation of Finger Kinematics via Large-Scale Temporal Convolutional Network,” Applied Sciences 2021, Vol. 11, Page 4678, vol. 11, no. 10, p. 4678, May 2021, doi: 10.3390/APP11104678.

[135] Z. Gao, R. Tang, Q. Huang, and J. He, “A Multi-DoF Prosthetic Hand Finger Joint Controller for Wearable sEMG Sensors by Nonlinear Autoregressive Exogenous Model,” Sensors 2021, Vol. 21, Page 2576, vol. 21, no. 8, p. 2576, Apr. 2021, doi: 10.3390/S21082576.

[136] F. Quivira, T. Koike-Akino, Y. Wang, and D. Erdogmus, “Translating sEMG signals to continuous hand poses using recurrent neural networks,” 2018 IEEE EMBS International Conference on Biomedical and Health Informatics, BHI 2018, vol. 2018-January, pp. 166–169, Apr. 2018, doi: 10.1109/BHI.2018.8333395.

[137] C. Wang, W. Guo, H. Zhang, L. Guo, C. Huang, and C. Lin, “sEMG-based continuous estimation of grasp movements by longshort term memory network,” Biomed Signal Process Control, vol. 59, p. 101774, May 2020, doi: 10.1016/J.BSPC.2019.101774.

[138] W. Guo, C. Wang, C. Lin, and C. Wang, “Long short term memory model based continuous estimation of human finger joint angles,” 2019 IEEE International Conference on Real-Time Computing and Robotics, RCAR 2019, vol. 2019-August, pp. 288–292, Aug. 2019, doi: 10.1109/RCAR47638.2019.9044013.

[139] Z. Ma, T. Pi, C. Xiong, and Q. Zhang, “Continuous Estimation of Multijoint Kinematics from Surface EMG during Daily Grasping Tasks,” ICARM 2022 - 2022 7th IEEE International Conference on Advanced Robotics and Mechatronics, pp. 655–659, 2022, doi: 10.1109/ICARM54641.2022.9959298.

[140] A. L. Delgado, A. F. Da Rocha, A. S. Leon, A. Ruiz-Olaya, K. R. Montero, and A. L. Delis, “Estimation of Joint Angle from sEMG and Inertial Measurements Based on Deep Learning Approach,” Proceedings of the Annual International Conference of the IEEE Engineering in Medicine and Biology Society, EMBS, pp. 700–703, 2021, doi: 10.1109/EMBC46164.2021.9630609.

[141] W. Guo et al., “Long exposure convolutional memory network for accurate estimation of finger kinematics from surface electromyographic signals,” J Neural Eng, vol. 18, no. 2, p. 026027, Mar. 2021, doi: 10.1088/1741-2552/ABD461.

[142] P. Zhao, C. Lin, J. Zhang, X. Niu, and Y. Liu, “Continuous Finger Kinematics Estimation Based on sEMG and Attention-ConvGRU Network,” Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), vol. 13458 LNAI, pp. 345–353, 2022, doi: 10.1007/978-3-031-13841-6\_32/FIGURES/7.

[143] P. Kim, K. S. Kim, and S. Kim, “Using common spatial pattern algorithm for unsupervised real-time estimation of fingertip forces

from sEMG signals,” IEEE International Conference on Intelligent Robots and Systems, vol. 2015-December, pp. 5039–5045, Dec. 2015, doi: 10.1109/IROS.2015.7354086.

[144] P. Kim, K. S. Kim, and S. Kim, “Modified Nonnegative Matrix Factorization Using the Hadamard Product to Estimate Real-Time Continuous Finger-Motion Intentions,” IEEE Trans Hum Mach Syst, vol. 47, no. 6, pp. 1089–1099, Dec. 2017, doi: 10.1109/THMS.2017.2751549.

[145] S. Zhou and K. Yin, “Continuous Finger Joint Angle Estimation with sEMG Signals,” NSENS 2019 - 2nd IEEE International Conference on Micro/Nano Sensors for Al, Healthcare, and Robotics, pp. 6–10, Oct. 2019, doi: 10.1109/NSENS49395.2019.9293963.

[146] N. M. Cole and A. B. Ajiboye, “Muscle synergies for predicting non-isometric complex hand function for commanding FES neuroprosthetic hand systems,” J Neural Eng, vol. 16, no. 5, p. 056018, Aug. 2019, doi: 10.1088/1741-2552/AB2D47.

[147] Z. Yu, H. Wang, and W. Yu, “Finger joint angle estimation based on sEMG signals by Attention-MLP,” J Phys Conf Ser, vol. 2113, no. 1, p. 012081, Nov. 2021, doi: 10.1088/1742- 6596/2113/1/012081.

[148] Y. Geng et al., “A CNN-Attention Network for Continuous Estimation of Finger Kinematics from Surface Electromyography,” IEEE Robot Autom Lett, vol. 7, no. 3, pp. 6297–6304, Jul. 2022, doi: 10.1109/LRA.2022.3169448.

[149] W. Guo et al., “Multi-Attention Feature Fusion Network for Accurate Estimation of Finger Kinematics from Surface Electromyographic Signals,” IEEE Trans Hum Mach Syst, vol. 53, no. 3, pp. 512–519, Jun. 2023, doi: 10.1109/THMS.2023.3252817.

[150] C. Lin, X. Chen, W. Guo, N. Jiang, D. Farina, and J. Su, “A BERT Based Method for Continuous Estimation of Cross-Subject Hand Kinematics From Surface Electromyographic Signals,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 31, pp. 87–96, 2023, doi: 10.1109/TNSRE.2022.3216528.

[151] Z. Chen, H. Wang, H. Chen, and T. Wei, “Continuous motion finger joint angle estimation utilizing hybrid sEMG-FMG modality driven transformer-based deep learning model,” Biomed Signal Process Control, vol. 85, p. 105030, Aug. 2023, doi: 10.1016/J.BSPC.2023.105030.

[152] Y. Long, Y. Geng, C. Dai, and G. Li, “A Transfer Learning Based Cross-Subject Generic Model for Continuous Estimation of Finger Joint Angles From a New User,” IEEE J Biomed Health Inform, vol. 27, no. 4, pp. 1914–1925, Apr. 2023, doi: 10.1109/JBHI.2023.3234989.

[153] H. Zhang, X. Zhou, Z. Yang, L. Tian, Y. Zheng, and G. Li, “Trace Finger Kinematics from Surface Electromyography by Using Kalman Decoding Method,” 2022 IEEE International Conference on Cyborg and Bionic Systems, CBS 2022, pp. 153–158, 2023, doi: 10.1109/CBS55922.2023.10115330.

[154] M. Kim, J. Lee, and K. Kim, “Tele-operation system with reliable grasping force estimation to compensate for the time-varying sEMG feature,” Proc IEEE Int Conf Robot Autom, vol. 2016-June, pp. 5561–5567, Jun. 2016, doi: 10.1109/ICRA.2016.7487773.

[155] I. J. R. Martinez, A. Mannini, F. Clemente, and C. Cipriani, “Online Grasp Force Estimation from the Transient EMG,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 28, no. 10, pp. 2333–2341, Oct. 2020, doi: 10.1109/TNSRE.2020.3022587.

[156] G. Liu, L. Wang, and J. Wang, “A novel energy-motion model for continuous sEMG decoding: from muscle energy to motor pattern,” J Neural Eng, vol. 18, no. 1, p. 016019, Feb. 2021, doi: 10.1088/1741-2552/ABBECE.

[157] S. Karimimehr, P. Ghaderi, and M. E. Andani, “Hand kinematics estimation using non-invasive surface sensors: A linear system identification approach,” 2015 22nd Iranian Conference on Biomedical Engineering, ICBME 2015, pp. 239–244, Feb. 2016, doi: 10.1109/ICBME.2015.7404149.

[158] R. Alazrai, D. Alabed, N. Alnuman, A. Khalifeh, and Y. Mowafi, “Continuous estimation of hand’s joint angles from sEMG using wavelet-based features and SVR,” ACM International Conference Proceeding Series, pp. 65–68, Oct. 2016, doi: 10.1145/3051488.3051498.

[159] R. Alazrai, D. Alabed, N. Alnuman, A. Khalifeh, and Y. Mowafi, “SEMG-based approach for estimating wrist and fingers joint angles using discrete wavelet transform,” IECBES 2016 - IEEE-EMBS

Conference on Biomedical Engineering and Sciences, pp. 596–599, 2016, doi: 10.1109/IECBES.2016.7843519.

[160] H. Mao, Y. Zheng, C. Ma, K. Wu, G. Li, and P. Fang, “Simultaneous estimation of grip force and wrist angles by surface electromyography and acceleration signals,” Biomed Signal Process Control, vol. 79, p. 104088, Jan. 2023, doi: 10.1016/J.BSPC.2022.104088.

[161] R. Alazrai, A. Khalifeh, N. Alnuman, D. Alabed, and Y. Mowafi, “An ensemble-based regression approach for continuous estimation of wrist and fingers movements from surface electromyography,” Proceedings of the Annual International Conference of the IEEE Engineering in Medicine and Biology Society, EMBS, vol. 2016- October, pp. 319–322, Oct. 2016, doi: 10.1109/EMBC.2016.7590704.

[162] H. Kawasaki, M. Kayukawa, H. Sakaeda, and T. Mouri, “Learning system for myoelectric prosthetic hand control by forearm amputees,” IEEE RO-MAN 2014 - 23rd IEEE International Symposium on Robot and Human Interactive Communication: Human-Robot Co-Existence: Adaptive Interfaces and Systems for Daily Life, Therapy, Assistance and Socially Engaging Interactions, pp. 899–904, Oct. 2014, doi: 10.1109/ROMAN.2014.6926367.

[163] Z. Qin, Z. He, Y. Li, S. Saetia, and Y. Koike, “A CW-CNN regression model-based real-time system for virtual hand control,” Front Neurorobot, vol. 16, p. 1072365, Dec. 2022, doi: 10.3389/FNBOT.2022.1072365/BIBTEX.

[164] A. E. Olsson, N. Malešević, A. Björkman, and C. Antfolk, “End-to-End Estimation of Hand- and Wrist Forces From Raw Intramuscular EMG Signals Using LSTM Networks,” Front Neurosci, vol. 15, p. 777329, Nov. 2021, doi: 10.3389/FNINS.2021.777329/BIBTEX.

[165] T. Bao, Y. Zhao, S. A. R. Zaidi, S. Xie, P. Yang, and Z. Zhang, “A deep Kalman filter network for hand kinematics estimation using sEMG,” Pattern Recognit Lett, vol. 143, pp. 88–94, Mar. 2021, doi: 10.1016/J.PATREC.2021.01.001.

[166] M. Meng, G. Zhou, Y. Ma, and X. Xi, “Continuous estimation of multi-DOF movement from sEMG based on non-negative matrix factorization and L2 regulation,” Med Biol Eng Comput, vol. 61, no. 7, pp. 1675–1686, Jul. 2023, doi: 10.1007/S11517-023-02807- 0/FIGURES/F.

[167] K. Zheng, S. Liu, J. Yang, M. Al-Selwi, and J. Li, “sEMG-Based Continuous Hand Action Prediction by Using Key State Transition and Model Pruning,” Sensors 2022, Vol. 22, Page 9949, vol. 22, no. 24, p. 9949, Dec. 2022, doi: 10.3390/S22249949.

[168] Z. Ruan, Q. Ai, K. Chen, L. Ma, Q. Liu, and W. Meng, “Simultaneous and Continuous Motion Estimation of Upper Limb Based on SEMG and LSTM,” Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), vol. 13013 LNAI, pp. 313–324, 2021, doi: 10.1007/978-3-030-89095-7\_31/FIGURES/9.

[169] G. Tang, J. Sheng, D. Wang, and S. Men, “Continuous Estimation of Human Upper Limb Joint Angles by Using PSO-LSTM Model,” IEEE Access, vol. 9, pp. 17986–17997, 2021, doi: 10.1109/ACCESS.2020.3047828.

[170] A. Zhang, Q. Li, Z. Li, and J. Li, “Upper Limb Movement Decoding Scheme Based on Surface Electromyography Using Attention-Based Kalman Filter Scheme,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 31, pp. 1878–1887, 2023, doi: 10.1109/TNSRE.2023.3262269.

[171] Q. Zhang, R. Liu, W. Chen, and C. Xiong, “Simultaneous and continuous estimation of shoulder and elbow kinematics from surface EMG signals,” Front Neurosci, vol. 11, no. MAY, p. 253504, May 2017, doi: 10.3389/FNINS.2017.00280/BIBTEX.

[172] Q. Zhang, C. Zheng, and C. Xiong, “EMG-based estimation of shoulder and elbow joint angles for intuitive myoelectric control,” 2015 IEEE International Conference on Cyber Technology in Automation, Control and Intelligent Systems, IEEE-CYBER 2015, pp. 1912–1916, Oct. 2015, doi: 10.1109/CYBER.2015.7288239.

[173] Y. M. Aung and A. Al-Jumaily, “Estimation of upper limb joint angle using surface EMG signal,” Int J Adv Robot Syst, vol. 10, Oct. 2013, doi: 10.5772/56717/ASSET/IMAGES/LARGE/10.5772\_56717- FIG15.JPEG.

[174] Y. Huang, K. Chen, X. Zhang, K. Wang, and J. Ota, “Joint torque estimation for the human arm from sEMG using backpropagation neural networks and autoencoders,” Biomed Signal Process Control, vol. 62, p. 102051, Sep. 2020, doi: 10.1016/J.BSPC.2020.102051.

[175] J. Wang et al., “Estimation of Continuous Joint Angles of Upper Limb Based on sEMG by Using GA-Elman Neural Network,” Math Probl Eng, vol. 2020, 2020, doi: 10.1155/2020/4065351.

[176] A. G. Feleke, L. Bi, and W. Fei, “EMG-Based 3D Hand Motor Intention Prediction for Information Transfer from Human to Robot,” Sensors 2021, Vol. 21, Page 1316, vol. 21, no. 4, p. 1316, Feb. 2021, doi: 10.3390/S21041316.

[177] X. Liu et al., “SE-TCN network for continuous estimation of upper limb joint angles,” Mathematical Biosciences and Engineering 2023 2:3237, vol. 20, no. 2, pp. 3237–3260, 2023, doi: 10.3934/MBE.2023152.

[178] Y. Chen et al., “A Continuous Estimation Model of Upper Limb Joint Angles by Using Surface Electromyography and Deep Learning Method,” IEEE Access, vol. 7, pp. 174940–174950, 2019, doi: 10.1109/ACCESS.2019.2956951.

[179] C. Ma et al., “A Bi-Directional LSTM Network for Estimating Continuous Upper Limb Movement from Surface Electromyography,” IEEE Robot Autom Lett, vol. 6, no. 4, pp. 7217–7224, Oct. 2021, doi: 10.1109/LRA.2021.3097272.

[180] L. Wen, J. Xu, D. Li, X. Pei, and J. Wang, “Continuous estimation of upper limb joint angle from sEMG based on multiple decomposition feature and BiLSTM network,” Biomed Signal Process Control, vol. 80, p. 104303, Feb. 2023, doi: 10.1016/J.BSPC.2022.104303.

[181] P. Xia, J. Hu, and Y. Peng, “EMG-Based Estimation of Limb Movement Using Deep Learning With Recurrent Convolutional Neural Networks,” Artif Organs, vol. 42, no. 5, pp. E67–E77, May 2018, doi: 10.1111/AOR.13004.

[182] C. Li, X. Zhang, H. Li, and H. Xu, “Continuous sEMG estimation method of upper limb shoulder elbow torque based on CNN-LSTM,” 2021 IEEE International Conference on Robotics and Biomimetics, ROBIO 2021, pp. 1390–1395, 2021, doi: 10.1109/ROBIO54168.2021.9739373.

[183] C. Ma, C. Lin, O. W. Samuel, L. Xu, and G. Li, “Continuous estimation of upper limb joint angle from sEMG signals based on SCA-LSTM deep learning approach,” Biomed Signal Process Control, vol. 61, p. 102024, Aug. 2020, doi: 10.1016/J.BSPC.2020.102024.

[184] Q. Ding, J. Han, and X. Zhao, “Continuous Estimation of Human Multi-Joint Angles from sEMG Using a State-Space Model,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 25, no. 9, pp. 1518–1528, Sep. 2017, doi: 10.1109/TNSRE.2016.2639527.

[185] H. ElMohandes, S. Eldawlatly, J. M. C. Audí, R. Ruff, and K. P. Hoffmann, “A multi-Kalman filter-based approach for decoding arm kinematics from EMG recordings,” Biomed Eng Online, vol. 21, no. 1, pp. 1–18, Dec. 2022, doi: 10.1186/S12938-022-01030- 6/FIGURES/14.

[186] C. Xie, Q. Yang, Y. Huang, S. Su, T. Xu, and R. Song, “A Hybrid Arm-Hand Rehabilitation Robot with EMG-Based Admittance Controller,” IEEE Trans Biomed Circuits Syst, vol. 15, no. 6, pp. 1332–1342, Dec. 2021, doi: 10.1109/TBCAS.2021.3130090.

[187] C. Wu, H. Zeng, A. Song, and B. Xu, “Grip force and 3D push-pull force estimation based on sEMG and GRNN,” Front Neurosci, vol. 11, no. JUN, p. 251824, Jun. 2017, doi: 10.3389/FNINS.2017.00343/BIBTEX.

[188] J. Liu, S. H. Kang, D. Xu, Y. Ren, S. J. Lee, and L. Q. Zhang, “EMG-Based continuous and simultaneous estimation of arm kinematics in able-bodied individuals and stroke survivors,” Front Neurosci, vol. 11, p. 241025, Aug. 2017, doi: 10.3389/FNINS.2017.00480/BIBTEX.

[189] Y. Liu, G. Wang, Z. Tian, K. Liu, and Z. Sun, “Upper Limb Multi-Joint Angle Estimation Based on Multichannel sEMG Signals Using Elman Neural Network,” 2022 IEEE International Conference on Real-Time Computing and Robotics, RCAR 2022, pp. 278–283, 2022, doi: 10.1109/RCAR54675.2022.9872197.

[190] Y. Chai, K. Liu, C. Li, Z. Sun, L. Jin, and T. Shi, “A novel method based on long short term memory network and discrete-time zeroing neural algorithm for upper-limb continuous estimation using sEMG signals,” Biomed Signal Process Control, vol. 67, p. 102416, May 2021, doi: 10.1016/J.BSPC.2021.102416.

[191] B. Zhang, X. Lan, Y. Liu, G. Wang, and Z. Sun, “sEMG-based continuous motion estimation of upper limb using a novel noisetolerant zeroing neurodynamic model combined with LSTM

network,” Digit Signal Process, vol. 133, p. 103828, Mar. 2023, doi: 10.1016/J.DSP.2022.103828.

[192] Y. Xing, C. Lv, Y. Liu, Y. Zhao, and D. Cao, “Continuous Driver Steering Intention Prediction Considering Neuromuscular Dynamics and Driving Postures,” Conf Proc IEEE Int Conf Syst Man Cybern, vol. 2020-October, pp. 4404–4409, Oct. 2020, doi: 10.1109/SMC42975.2020.9283327.

[193] Y. Xing, C. Lv, Y. hui Liu, Y. Zhao, D. Cao, and S. Kawahara, “Hybrid-Learning-Based Driver Steering Intention Prediction using Neuromuscular Dynamics,” IEEE Transactions on Industrial Electronics, 2021, doi: 10.1109/TIE.2021.3059537.

[194] J. Liu, Y. Ren, D. Xu, S. H. Kang, and L. Q. Zhang, “EMG-Based Real-Time Linear-Nonlinear Cascade Regression Decoding of Shoulder, Elbow, and Wrist Movements in Able-Bodied Persons and Stroke Survivors,” IEEE Trans Biomed Eng, vol. 67, no. 5, pp. 1272–1281, May 2020, doi: 10.1109/TBME.2019.2935182.