# Multijoint Continuous Motion Estimation for Human Lower Limb Based on Surface Electromyography - PMC

> **这是网页快照，不是同行评审论文。**
> 来源：<https://pmc.ncbi.nlm.nih.gov/articles/PMC11820049/>
> 抓取时间：2026-10-01 20:38
> HTTP 状态：200　｜　原始字节：206,207　｜　SHA256：`bc68ddd141c83e18`
> **用途**：本条在语料四中的证据等级为「第三方实测」。引用时必须标注为网页型证据，
> 不得与同行评审论文并列。抓取日期之后页面可能变化，引用前请核对原链接。

**页面描述**：The estimation of multijoint angles is of great significance in the fields of lower limb rehabilitation, motion control, and exoskeleton robotics. Accurate joint angle estimation helps assess joint function, assist in rehabilitation training, and ...


---

- 

 - 
- 

 - 
 - 
 - 
 - 
 - 
 - 

 Multijoint Continuous Motion Estimation for Human Lower Limb Based on Surface Electromyography - PMC

 - 

 - 

- 

 Skip to main content

 An official website of the United States government

 Here's how you know 

 Here's how you know 

 Official websites use .gov 

 A
 .gov website belongs to an official
 government organization in the United States.

 Secure .gov websites use HTTPS 

 A lock (

 ) or https:// means you've safely
 connected to the .gov website. Share sensitive
 information only on official, secure websites.

 Search

 Log in 

 - 

 Dashboard

 - 

 Publications

 - 

 Account settings

 - 

 Log out

 Search…

 Search NCBI

 Search PMC Full-Text Archive 

 Search in PMC 

 - 

 Journal List

 - 

 User Guide

 - 

 - 

 - 

 - 

 - 

 - 

## PERMALINK

 Copy 

 As a library, NLM provides access to scientific literature. Inclusion in an NLM database does not imply endorsement of, or agreement with,
 the contents by NLM or the National Institutes of Health.

 Learn more:
 PMC Disclaimer 
 |
 
 PMC Copyright Notice

 Sensors (Basel) 
. 2025 Jan 24;25(3):719. doi: 10.3390/s25030719 

# Multijoint Continuous Motion Estimation for Human Lower Limb Based on Surface Electromyography
 
 Yonglin Han 

### Yonglin Han 

 1 School of Mechanical Engineering, Xinjiang University, Urumqi 830047, China

 Conceptualization, Methodology, Software, Data curation, Writing – original draft

 Find articles by Yonglin Han 

 1 , Qing Tao 

### Qing Tao 

 1 School of Mechanical Engineering, Xinjiang University, Urumqi 830047, China

 Resources, Writing – review & editing, Supervision, Project administration, Funding acquisition

 Find articles by Qing Tao 

 1, * , Xiaodong Zhang 

### Xiaodong Zhang 

 2 School of Mechanical Engineering, Xi’an Jiaotong University, Xi’an 710049, China

 Writing – review & editing

 Find articles by Xiaodong Zhang 

 2 

 Editor: Luca Miceli 

- Author information 

- Article notes 

- Copyright and License information 

 1 School of Mechanical Engineering, Xinjiang University, Urumqi 830047, China

 2 School of Mechanical Engineering, Xi’an Jiaotong University, Xi’an 710049, China

 * Correspondence: taoqing@xju.edu.cn 

#### Roles

 Yonglin Han : Conceptualization, Methodology, Software, Data curation, Writing – original draft 

 Qing Tao : Resources, Writing – review & editing, Supervision, Project administration, Funding acquisition 

 Xiaodong Zhang : Writing – review & editing 

 Luca Miceli : Academic Editor 

 Received 2024 Dec 21; Revised 2025 Jan 21; Accepted 2025 Jan 21; Collection date 2025 Feb.

 © 2025 by the authors.

 Licensee MDPI, Basel, Switzerland. This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license ( https://creativecommons.org/licenses/by/4.0/ ).

 PMC Copyright notice 

 PMCID: PMC11820049 PMID: 39943357 

## Abstract

 The estimation of multijoint angles is of great significance in the fields of lower limb rehabilitation, motion control, and exoskeleton robotics. Accurate joint angle estimation helps assess joint function, assist in rehabilitation training, and optimize robotic control strategies. However, estimating multijoint angles in different movement patterns, such as walking, obstacle crossing, squatting, and knee flexion–extension, using surface electromyography (sEMG) signals remains a challenge. In this study, a model is proposed for the continuous motion estimation of multijoint angles in the lower limb (CB-TCN: temporal convolutional network + convolutional block attention module + temporal convolutional network). The model integrates temporal convolutional networks (TCNs) with convolutional block attention modules (CBAMs) to enhance feature extraction and improve prediction accuracy. The model effectively captures temporal features in lower limb movements, while enhancing attention to key features through the attention mechanism of CBAM. To enhance the model’s generalization ability, this study adopts a sliding window data augmentation method to expand the training samples and improve the model’s adaptability to different movement patterns. Through experimental validation on 8 subjects across four typical lower limb movements, walking, obstacle crossing, squatting, and knee flexion–extension, the results show that the CB-TCN model outperforms traditional models in terms of accuracy and robustness. Specifically, the model achieved R 2 values of up to 0.9718, RMSE as low as 1.2648°, and NRMSE values as low as 0.05234 for knee angle prediction during walking. These findings indicate that the model combining TCN and CBAM has significant advantages in predicting lower limb joint angles. The proposed approach shows great promise for enhancing lower limb rehabilitation and motion analysis.

 Keywords: sEMG, different movement patterns, TCN, multijoint angle estimation, TCN, CBAM

## 1. Introduction

 Surface electromyography (sEMG), a bioelectrical signal reflecting muscle activity, has significant application value in lower limb motion estimation [ 1 , 2 ]. It can capture dynamic muscle activity information, which is strongly correlated with joint movements, providing key technological support for lower limb rehabilitation [ 3 ], motion control [ 4 ], and exoskeleton robotics [ 5 ]. Daily activities such as walking, stair climbing, sitting, and standing hold significant importance for research [ 6 ]. The continuous motion estimation of these movement patterns not only helps understand the complexity of human motion control but also provides a theoretical foundation and technical support for robot design and human–robot interaction system development [ 7 ]. However, maintaining accuracy in lower limb multijoint continuous motion estimation based on surface electromyography (sEMG) signals, under varying movement patterns and individual differences, has always been a long-standing challenge.

 Currently, the research on lower limb motion intention recognition based on surface electromyography (sEMG) signals is mainly divided into two parts. One approach focuses on using sEMG signals to recognize discrete human movement patterns [ 8 , 9 , 10 ]. Many researchers focus on recognizing a wider range of movement patterns and achieving higher recognition accuracy using sEMG signals. The emphasis of these studies lies in feature extraction methods and classification algorithms. The other approach involves using sEMG signals to estimate the continuous motion of human joints [ 11 , 12 , 13 ]. Motion pattern classification based on sEMG signals can only predict predefined movement patterns and recognize a limited number of discrete postures. It cannot be directly used as input for a controller to achieve continuous control, which severely impacts the coordination and compliance of exoskeletons. As a result, researchers have gradually shifted the focus of theoretical and applied studies towards continuous motion estimation based on sEMG signals. In constructing regression models between sEMG signals and continuous lower limb joint motion, most researchers have focused on sEMG feature engineering and model selection to achieve better estimation performance. For example, L. F. Sommer et al. [ 14 ] proposed a Wiener model approach to estimate elbow joint angles using sEMG signals from arm muscles. Xie et al. [ 15 ] used a generalized regression neural network (GRNN) model optimized with the golden section algorithm (GS-GRNN), taking the root mean square features of sEMG signals from the thigh muscles and hip joint angles as inputs to predict lower limb joint angles. Liu et al. [ 16 ] proposed an improved feature-based convolutional neural network (CNN) model for the continuous estimation of knee joint angles. EMG signals and joint angles during normal walking were recorded using a 6-channel sEMG acquisition system and an optical motion capture system. The results show that the improved CNN model can predict knee joint angles with higher accuracy.

 In recent years, many researchers have used deep learning methods to estimate continuous joint angles of human motion based on sEMG signals [ 17 , 18 , 19 ]. The temporal convolutional network (TCN) is a neural network architecture suitable for processing sequential data. Compared to recurrent neural networks (RNNs), TCNs have significant advantages in handling long-term dependencies and parallel computation [ 20 ]. For example, Chen et al. [ 21 ] proposed a large-scale temporal convolutional network (LS-TCN), focusing on continuous motion estimation based on sEMG signals. The model was used to predict the angles of 10 joints during six types of grasping movements, achieving an accuracy of 71.6%. Wang et al. [ 22 ] proposed a subject-independent TCN-BiLSTM deep learning model. Compared to the long short-term memory (LSTM), artificial neural network (ANN), and gated recurrent unit (GRU), the temporal convolutional network with bidirectional long short-term memory (TCN-BiLSTM) model demonstrated higher accuracy in estimating the angles and torques of the hip, knee, and ankle joints, proving its robustness and practicality. Du et al. [ 23 ] proposed a joint angle prediction model based on sEMG signals for the continuous estimation of wrist motion angle changes. The model combines the advantages of TCN and LSTM networks; TCN is used to extract local features and deep information from the sEMG signals, while LSTM leverages its temporal memory capability to compensate for the limitations of TCN in capturing long-term dependencies, thus improving prediction performance.

 The main contributions of this study are as follows:

 (1) A CB-TCN network is proposed, which combines TCN and CBAM for estimating the angles of the hip, knee, and ankle joints in the lower limb. TCN captures the temporal features, while CBAM enhances the model’s focus on critical moments and features through its attention mechanism, thereby improving prediction accuracy.

 (2) The model’s performance is validated through four common lower limb movements, including gait, obstacle crossing, squatting, and knee flexion–extension. Its performance is tested under different movement scenarios and compared with the following four classic models: encoder–decoder temporal convolutional network (ED-TCN), TCN, LSTM, and Wiener filter. In this study, its advantages and performance improvements are comprehensively evaluated.

 In this study, the aim is to evaluate the CB-TCN model for estimating joint angles during various daily activities using the envelope extracted from sEMG signals. The structure of this paper is organized as follows: Section 2 presents the experimental setup; Section 3 provides a detailed discussion of the research methods; Section 4 discusses the results and their analysis; and Section 5 concludes with the findings and outlines potential directions for future research.

## 2. Experimental Setup

### 2.1. Experimental Design

 This study recruited 8 healthy volunteers (6 males and 2 females) with an average age of 23 ± 1.73 years, an average height of 173.75 ± 7.3 cm, and an average weight of 62.5 ± 11.46 kg. All participants signed informed consent forms and agreed to the use of experimental data and images for scientific research purposes. The experiment was approved by the Ethics Committee of Xi’an Jiaotong University, approval number: 2024YS239. All experimental procedures were conducted in accordance with the relevant provisions of the Declaration of Helsinki.

 The experimental design included the following four types of movement patterns: gait, obstacle crossing, squatting, and knee flexion–extension. The specific arrangements were as follows:

- Gait: Each participant completed this experiment in approximately 5.5 s, with a walking speed controlled at 2 m/s, performing a total of 5 trials. Gait involves the dynamic changes of lower limb joints during walking, which helps analyze the joint angle variations and gait cycle of normal walking. Although this speed (equivalent to 7.2 km/h) exceeds typical walking speeds (usually around 1.2–1.5 m/s), it was deliberately chosen to simulate brisk walking or rapid movement conditions, offering a more thorough evaluation of the model’s performance in such scenarios.

- Obstacle Crossing: Each participant completed this experiment in approximately 5 s, performing a total of 5 trials. Obstacle crossing simulates lower limb movements when stepping over an obstacle, providing angle change data for rapid movements within a short time.

- Squatting: Each participant completed this experiment in approximately 25 s, with 10 repetitions of each squatting action, performing a total of 5 trials. This movement pattern is designed to simulate the squatting and standing actions in daily life and is suitable for studying dynamic control and joint angle variations in the lower limbs.

- Knee Flexion–Extension: Each participant completed this experiment in approximately 20 s, with 10 repetitions of each knee flexion–extension action, performing a total of 5 trials. This movement pattern primarily focuses on the flexion–extension of the knee joint and effectively tests the continuous variation in the knee joint angle.

 During this experiment, participants were instructed to perform each task at their natural pace. A 5 min rest period was provided after each session to ensure recovery.

### 2.2. Data Collection

 The continuous walking capability of the human lower limb results from the coordinated action of multiple muscles that drive the lower limb to perform the required movements. Each side of the lower limb consists of at least 30 muscles, working together to enable the movement across 7 degrees of freedom.

 To better collect surface electromyography (sEMG) signals closely related to lower limb movements, in this study, 7 major muscles were selected as targets for signal acquisition, including rectus femoris (RF), tibialis anterior (TA), soleus (SOL), gastrocnemius (GAS), vastus medialis (VM), vastus lateralis (VL), and biceps femoris (BF) [ 24 ]. The placement of the electromyography sensors followed the Surface ElectroMyoGraphy for the Non-Invasive Assessment of Muscles (SENIAM) guidelines to ensure optimal signal quality [ 25 ]. SENIAM provides standardized protocols for electrode placement to ensure consistent and reliable measurements of muscle activity. Additionally, the specific locations of reflective markers and sEMG sensor placements during lower limb movements are shown in Figure 1 .

#### Figure 1.

 Open in a new tab 

 This shows the specific placement of the sensors, where ( a ) represents the arrangement of surface electromyography (sEMG) sensors and reflective markers on the right leg and ( b ) depicts the scenario of the participant performing the walking experiment on a treadmill.
 In this study, a VICON motion capture system consisting of 10 cameras was employed, primarily used to collect motion data from 16 motion capture markers placed on the lower limbs. The sampling frequency was set to 100 Hz. In this experiment, a total of seven surface electromyography sensors were used to record electromyographic signals, utilizing Noraxon wireless equipment with a sampling frequency of 2000 Hz. In this experiment, the MX-Giganet module was used to connect the host PC and MX cameras, serving as an interface between the VICON system and the sEMG system, ensuring synchronized data acquisition, recording, and analysis of both systems in the Vicon Nexus 2.7. This configuration maximized signal synchronization and minimized signal deviation caused by hardware and software errors. In this study, the placement of the cameras and markers followed standard motion capture protocols and complied with the operational specifications of the VICON system, ensuring the accuracy of data collection.

### 2.3. Data Processing

 (1) Preprocessing of sEMG Signals: In this study, a fourth-order Butterworth filter (20–450 Hz) was applied to preprocess the electromyographic signals, removing both high-frequency and low-frequency noise. Additionally, to eliminate the influence of power line interference on the sEMG signal quality, a 50 Hz notch filter (Q factor = 35) was used to filter out power line noise. Subsequently, a 10 Hz low-pass filter was applied to extract the envelope of the electromyographic signals.

 (2) Preprocessing of Motion Signals: The motion data, acquired from a high-precision motion capture system, were preprocessed using a fourth-order zero-phase Butterworth low-pass filter (cutoff frequency of 15 Hz) and a critically damped low-pass filter (cutoff frequency of 15 Hz) to remove high-frequency noise and smooth the signals [ 26 ]. The joint angles were then derived using the cosine formula, based on the motion capture data, ensuring accurate estimation of the joint angles from the positions of the relevant body segments.

 In addition, to address the issue of inconsistent sampling frequencies, resampling of the preprocessed sEMG signals was performed. The sampling frequency of the joint angle signals was 100 Hz, while that of the sEMG signals was 2000 Hz. Since the bandwidth of lower limb joint movements is relatively low, the requirements for the sampling frequency are less stringent. To reduce computational load, all signals were resampled to a unified sampling frequency of 20 Hz.

### 2.4. Sliding Window Data Expansion Strategy

 In this study, a sliding window data expansion strategy was employed, dividing the original time-series data into overlapping segments, each serving as an independent training sample [ 27 , 28 ]. The window size (W) was 10, with a step size (S) of 1, allowing each time point to be the center of multiple windows. This increases the number of training samples and helps the model capture fine-grained temporal patterns, enhancing prediction accuracy and generalization. The core idea of the above sliding window data expansion strategy can be expressed by the following formula:

 Assume the original time-series data are represented as X = x 1 , x 2 , ⋯ , x T , where x t represents the data at the t-th time step, where T is the total length of the data. It is defined the size of the sliding window as W (in this case, 10), and the step size as S (in this case, 1). The window w i represents a continuous set of W data points starting from the i -th time step, specifically, as follows:

 w i = x i , x i + 1 , ⋯ , x i + w − 1 
 (1) 

 For i = 1 , 2 , ⋯ , T − w + 1 , each window contains w consecutive data points. By setting the step size S = 1, the window slides one-time step at a time, generating multiple overlapping windows.

 Therefore, the set of windows is as follows:

 W = w 1 , w 2 , ⋯ w T − W + 1 
 (2) 

 Each window is used as an independent input to the model, allowing it to capture temporal dependencies and local features in the data.

## 3. Methods

### 3.1. TCN

 TCN is a deep learning model based on convolution operations, widely used for processing time-series data [ 29 ]. Compared to traditional RNNs, TCN provides an efficient alternative, excelling in time-series modelling tasks due to its parallel computation and longer memory capability. Its main structure includes causal convolutions, dilated convolutions, and residual blocks [ 30 ].

#### 3.1.1. Causal Convolution

 The core goal of causal convolution is to ensure that the model’s output only depends on the current and past input data, and not on future inputs, thus strictly adhering to the principle of causality, as shown in Figure 2 a.

##### Figure 2.

 Open in a new tab 

 The structure of the temporal convolutional network (TCN), including ( a ) causal convolution, ( b ) dilated convolution, and ( c ) residual module.
 Given that the convolution filter is represented as F = ( f 1 , f 2 , ⋯ , f k ) and the input sequence as X = ( x 1 , x 2 , ⋯ x T ) , the causal convolution at position t can be represented as follows:

 y t c a u s a l = ∑ k = 1 K f k x t − k 
 (3) 

 Here, k is explicitly defined as the kernel size, which determines the number of coefficients in the convolution filter.

#### 3.1.2. Dilated Convolution

 Dilated convolution can achieve a larger receptive field through interval sampling, compensating for the limitations of causal convolution, as shown in Figure 2 b.

 Given that the convolution filter is represented as F = ( f 1 , f 2 , ⋯ , f k ) and the input sequence is denoted as X = ( x 1 , x 2 , ⋯ x T ) , the dilated convolution at position t can be represented as follows:

 y t dilated = ∑ k = 1 K f k x t − d ⋅ k 
 (4) 

 Here, k is the number of coefficients in the convolution filter. d is the dilation rate, controlling the spacing between the elements of the input sequence used in the convolution.

#### 3.1.3. Residual Module

 TCN uses residual modules to mitigate the problems of vanishing and exploding gradients that occur as the network deepens, allowing information to be transmitted stably across layers and thereby improving generalization ability, as shown in Figure 2 c. The residual module in TCN includes causal convolution, batch normalization (BN) layers, and ReLU activation functions. In the residual module, a dropout layer is also added for regularization to prevent overfitting. The input of the residual module is added to the output after two convolution operations through an identity mapping (or a 1 × 1 convolution) to ensure that the dimensions are consistent.

### 3.2. CBAM

 CBAM is an integrated module that incorporates both spatial attention and channel attention mechanisms. By using the CBAM attention mechanism, it focuses on important features while suppressing irrelevant ones [ 31 ].

 Channel attention focuses on the most meaningful features by increasing the weights of effective channels and reducing the weights of less effective ones, as shown in Figure 3 . The input feature map is first processed by global max pooling and global average pooling based on width and height, respectively. Then, the outputs of the pooling operations are passed through an MLP. The features output by the MLP are then summed element-wise, followed by a sigmoid activation to generate the final channel attention feature map. This channel attention feature map is then element-wise multiplied with the input feature map to generate the input features required for the spatial attention module.

 F a v g c = A v g P o o l ( F ) ∈ R C , F max c = M a x P o o l ( F ) ∈ R C M c ( F ) = σ ( W 1 ( R eLU ( W 0 ( F avg c ) ) ) + W 1 ( R eLU ( W 0 ( F max c ) ) ) ) 
 (5) 

#### Figure 3.

 Open in a new tab 

 Channel attention module.
 Here, σ is the sigmoid activation function; W 0 ∈ R C × k and W 1 ∈ R k × 1 represent the weights; and k is the number of attention units. F ∈ R C × H × W denotes the feature map, and the channel attention weights M c ( F ) ∈ R C are within the range [0, 1], where C is the number of channels, and H and W represent the height and width of the feature map.

 Spatial attention obtains two different features by using max pooling and average pooling along the channel dimension, as shown in Figure 4 . The feature map output by the channel attention module is used as the input feature map for this module. First, a global max pooling and global average pooling are performed based on channels. The results of these two operations are then concatenated along the channel dimension. After that, a convolution operation is applied to reduce the dimensionality to a single channel. The resulting feature map is passed through a sigmoid function to generate the spatial attention feature. Finally, this feature is element-wise multiplied with the input feature map of the module to obtain the final generated feature.

 F a v g s = A v g P o o l ( F ′ ) ∈ R C , F max s = M a x P o o l ( F ′ ) ∈ R C 

 F cat s = [ F a v g s ; F max s ] , F cat s ∈ R 2 × T 
 (6) 

 M s ( F ′ ) = σ ( C o n v 1 D ( F cat s ) ) 
 (7) 

#### Figure 4.

 Open in a new tab 

 Spatial attention module.
 In the equation, σ is the sigmoid activation function; the size of the convolution kernel is typically 3; and M s ( F ′ ) ∈ R T represents the spatial attention weight.

### 3.3. CB-TCN

 The CB-TCN proposed in this paper is a deep learning model that combines the temporal convolutional networks (TCNs) and convolutional block attention module (CBAM). The structure is CB-TCN (temporal convolutional network + convolutional block attention module + temporal convolutional network). CBAM enhances the feature extraction capability, while the two TCN layers at the front and back efficiently model sequential data. This architecture is suitable for tasks such as continuous motion regression and other time-series prediction tasks.

 The overall framework of the CB-TCN model is shown in Figure 5 . The model adopts a cascaded structure of TCN → CBAM → TCN, where the initial TCN module extracts temporal features from the input time-series data. Then, the convolutional block attention module (CBAM) enhances the extracted features by emphasizing the most relevant temporal and spatial information. Finally, the second TCN module further models the enhanced features and outputs the predicted joint angles. In this architecture, the initial TCN module can be seen as the encoder, responsible for extracting temporal features from the original input sequence and transforming them into latent feature representations, while the second TCN module acts as the decoder, modeling the enhanced features and generating the final output prediction. This design fully leverages the advantages of TCN in temporal feature extraction and CBAM in feature refinement, ensuring accuracy and robustness when modeling complex temporal dependencies.

#### Figure 5.

 Open in a new tab 

 The complete framework of the proposed CB-TCN model for angle estimation.
 In this study, 500 iterations were set to observe the trend of the model’s loss curve. After extensive experimental testing, 50 training epochs were selected, which fully met the loss requirements. The mean squared error (MSE) loss function was used as the model’s loss function, and L2 regularization (weight_decay) along with the Adam optimizer were employed during the training process. MSE loss is used as the loss function to measure the difference between the predicted and actual values. MSE loss is calculated by averaging the squared differences between the predicted values and the actual ground truth values. L2 regularization helps prevent overfitting by penalizing large weight values, while the Adam optimizer combines the advantages of momentum and adaptive learning rates, enabling more efficient parameter updates, faster training, and stable convergence [ 32 ]. This approach also incorporates a learning rate scheduler, with the initial learning rate set to LR = 0.001, and the learning rate decayed to 50% of its original value every 10 epochs (i.e., gamma = 0.5) [ 33 ]. A batch size of 50 was used throughout the training process.

### 3.4. Evaluation Metrics

 In this paper, the root mean square error (RMSE), normalized root mean square error (NRMSE), and coefficient of determination (R 2 ) were introduced to evaluate the model’s estimation results. RMSE measures the average error between the predicted and actual values, with a lower RMSE indicating more accurate predictions. NRMSE normalizes RMSE according to the range of observed values, making it easier to compare errors across different datasets. R 2 represents the model’s ability to explain the variance in the observed data, with values close to 1 indicating a good fit and values close to 0 indicating poor prediction performance. The formulas for RMSE, NRMSE, and R 2 are as follows:

 R 2 = 1 − ∑ t = 1 N y ^ t − y t 2 ∑ t = 1 N y ¯ − y t 2 
 (8) 

 RMSE = 1 N ∑ t = 1 N ( y ^ t − y t ) 2 
 (9) 

 NRMSE = RMSE max ( y t ) − min ( y t ) 
 (10) 

where y ^ t is the predicted value at time t ; y t is the measured value at time t ; y ¯ is the mean of the observed values; and N is the number of samples.

### 3.5. Model Comparison

 - 
 (1) ED-TCN

 ED-TCN is an enhanced version of the traditional TCN that uses an encoder–decoder architecture to more effectively model sequential data, particularly for time-series prediction tasks. In this architecture, ED-TCN employs two separate TCN layers: one serves as the encoder to extract temporal features from the input, while the other acts as the decoder to transform these features into the final prediction results. This model shares the same hyperparameter settings as CB-TCN, including the kernel size, the number of convolution layers, and other parameters.

 - 
 (2) TCN

 TCN is a CNN-based model specifically designed to handle time-series data. By using causal convolutions and dilated convolutions, it effectively captures dependencies over long time horizons while avoiding an increase in computational load. TCN shares the same hyperparameter settings as CB-TCN, such as the kernel size, the number of convolution layers, and other parameters.

 - 
 (3) LSTM

 LSTM is a special type of recurrent neural network (RNN) that effectively captures long-term dependencies in sequences. Through gating mechanisms such as forget gates, input gates, and output gates, LSTM is able to retain important information in time series while ignoring irrelevant information, thereby improving the performance and stability when handling long sequences. This model shares the same hyperparameter settings as CB-TCN, including the kernel size, the number of layers, and other parameters.

 - 
 (4) Wiener Filter

 The Wiener filter is a classical linear filter widely used in signal processing and noise suppression. Its goal is to optimize the difference between the input signal and the output signal by minimizing the mean squared error (MSE). In time-series prediction, the Wiener filter is commonly used for denoising and smoothing signals, but due to its linear nature, it is generally not suitable for capturing complex nonlinear dependencies.

## 4. Results

 In this study, sEMG envelope signals and joint angles are used from four different motion modes (walking, obstacle crossing, squat, and knee flexion–extension) as inputs to the model. The evaluation metrics include R 2 , RMSE, and NRMSE. All metric values are obtained through 5-fold cross-validation and used as the final results. In the flexion–extension movement, the focus is primarily on the actions of the knee and ankle joints, as the range of motion and variations in the hip joint are relatively limited. Therefore, the hip joint is typically not analyzed in detail.

### 4.1. Angle Estimation Curve Analysis

 Figure 6 shows the estimation curves for the hip, knee, and ankle joints of the lower limb using five different models. The black solid line represents the actual joint curve, the blue dashed line represents the predictions of the CB-TCN model, the green dashed line represents the predictions of the ED-TCN model, the red dashed line represents the predictions of the TCN model, the yellow dashed line represents the predictions of the LSTM model, and the purple dashed line represents the predictions of the Wiener model. A comparison of the model structure parameters is provided in Table S1 .

#### Figure 6.

 Open in a new tab 

 ( a ) Gait motion; ( b ) obstacle crossing motion; ( c ) squat motion; ( d ) knee flexion–extension motion.
 As shown in Figure 6 , all models effectively capture the variations in joint angles. However, there are noticeable differences between the models. The proposed CB-TCN model demonstrates superior performance, with its predicted curve closely following the actual data, showing better fitting accuracy compared to the other models. The following is an analysis of the prediction curves for different motions:

 - 
 (1) Gait Motion

 In gait motion, the joint angles exhibit a periodic pattern due to the repetitive gait cycle, fluctuating in a regular manner. All models are able to capture these periodic patterns, but there are subtle differences in their accuracy in tracking the actual joint angle changes. Notably, the CB-TCN model aligns more precisely with the actual data, especially during transitional phases (e.g., from the stance phase to the swing phase), highlighting its exceptional ability to model the dynamic changes in the gait cycle. The subtle differences observed in the other models suggest that they may struggle to capture the fine-grained motion details that are crucial for walking.

 - 
 (2) Obstacle Crossing Motion

 In the obstacle crossing motion, the joint angles undergo sudden changes due to the dynamic shift in body posture during the stepping phase. While all models are able to capture the overall trend, the CB-TCN model excels in handling these sudden changes, closely following the actual data, particularly when the foot crosses the obstacle. In contrast, other models exhibit more noticeable jitter at these moments, indicating a lower prediction capability for such rapid posture adjustments.

 - 
 (3) Squat and Stand Motion

 As the body lowers and rises, the squat and stand motion causes significant changes in joint angles, especially in the knee and hip joints. The prediction curves of all models generally align well with the actual data, but the CB-TCN model stands out as it more smoothly and accurately depicts the rising and lowering movements, particularly the knee joint angle. The subtle differences observed in the other models may suggest difficulties in adapting to the complex nonlinear dynamic characteristics of the squat and stand motion.

 - 
 (4) Knee Flexion–Extension Motion

 The knee flexion–extension motion exhibits relatively simple but rapid fluctuations, especially during the knee flexion and extension phases. Due to the sharp transitions in this motion, accurately predicting these changes can be more challenging. During this experiment, as the subject was seated on a chair, the knee and ankle joints exhibited significant changes during knee flexion–extension, while the hip joint showed minimal changes, and the captured curve changes were irregular. Therefore, the model’s estimation of the hip joint motion during knee flexion–extension is not presented here. Similarly, the CB-TCN model performs better, closely following the actual data and accurately capturing the subtle variations in the knee joint angle. Other models tend to show some lag or less smooth transitions in their prediction curves, suggesting that their ability to capture rapid and fine motions may be limited.

 The CB-TCN model consistently demonstrates outstanding performance in tracking joint angle changes across various motions, particularly in motion transitions and certain high-dynamic scenarios. It excels at capturing fine motion details and rapid transitions, which are crucial for understanding joint behavior in activities such as walking, obstacle crossing, squatting, and knee flexion–extension. However, the CB-TCN model has some limitations when handling fast and complex movements, such as squatting and obstacle crossing. Specifically, the model performs poorly in capturing rapid changes in joint angles, particularly in situations with fast dynamic changes, making it difficult to accurately predict joint motion trajectories. Additionally, muscle activation patterns are typically nonlinear, and the model has not effectively handled this complex, time-varying muscle response, leading to larger prediction errors in certain complex movements. To improve the model’s accuracy and robustness, multi-task learning or multi-scale feature learning methods can be introduced to better capture the interactions and coordination between joints, enhancing the model’s ability to model multi-joint coordination in lower limb movements. Furthermore, increasing training data with more complex movement patterns can enhance the model’s generalization ability, ensuring that it maintains good predictive accuracy across different movement scenarios and practical applications.

### 4.2. Comparison Analysis of the CB-TCN Model with Other Models

 In the comparison of the five models (CB-TCN, ED-TCN, TCN, LSTM, Wiener) in terms of R 2 , the RMSE was performed using a one-way analysis of variance (ANOVA), followed by a post hoc Tukey test with a significance level of 0.05. As shown in Figure 7 , there are statistically significant differences ( p < 0.05) in the R 2 and the RMSE values for the hip, knee, and ankle joint motions among the five models. Figure 7 shows the mean values of R 2 , RMSE, and NRMS for the eight participants. Table S2 summarizes the average statistical metrics for all participants.

#### Figure 7.

 Open in a new tab 

 The R 2 , RMSE, and NRMSE of five models across four movements (hip, knee, ankle) show statistically significant differences in R 2 and RMSE ( p < 0.05; *: 0.05 ≥ p > 0.001; **: 0.001 ≥ p > 0.0001; ***: p < 0.0001), with the results averaged over 8 subjects. ( a ) Gait motion; ( b ) obstacle-crossing motion; ( c ) squat motion; ( d ) knee flexion–extension motion.
 - 
 (1) Gait Motion

 Hip Joint: The R 2 of the CB-TCN model is 0.9190, which is the best among all models. It outperforms the second-best model, ED-TCN (R 2 = 0.8965), by a margin of 0.0225 and shows a more significant improvement of 0.1148 compared to the Wiener model. The RMSE is 1.2648°, also lower than the other models, particularly the Wiener model (a difference of 1.6318°). For NRMSE, CB-TCN has a value of 0.08075, which is lower than the other models.

 Knee Joint: The R 2 of the CB-TCN model is 0.9718, significantly higher than the other models. It outperforms the second-best model, ED-TCN (R 2 = 0.9579), by 0.0139 and shows a notable improvement of 0.0667 compared to the Wiener model. In terms of RMSE, CB-TCN is 0.4991°, lower than the other models, particularly the Wiener model (a difference of 0.2234°). For NRMSE, CB-TCN has a value of 0.05234, which is lower than the other models.

 Ankle Joint: The R 2 of the CB-TCN model is 0.9701, which outperforms the other models. It shows an improvement of 0.0121 compared to the second-best model, ED-TCN (R 2 = 0.9580), and a notable difference of 0.0566 compared to the Wiener model. The RMSE is 1.0526°, outperforming the other models, particularly the Wiener model (a difference of 0.9147°). For NRMSE, CB-TCN has a value of 0.05531, which is also lower than the other models.

 - 
 (2) Obstacle Crossing Motion

 Hip Joint: The R 2 of the CB-TCN model is 0.8220, which is superior to the other models. It demonstrates an improvement of 0.0432 compared to the second-best model, ED-TCN (R 2 = 0.7788), and a significant difference of 0.2954 compared to the Wiener model. In terms of RMSE, CB-TCN is 3.2414°, lower than the other models, especially the Wiener model (a difference of 3.3052°). For NRMSE, CB-TCN has a value of 0.112, which is lower than the values of the other models.

 Knee Joint: The R 2 of the CB-TCN model is 0.8999, which outperforms the other models. It shows an improvement of 0.0485 compared to the second-best model, ED-TCN (R 2 = 0.8514), and a substantial difference of 0.2879 compared to the Wiener model. The RMSE is 2.4017°, lower than the other models, especially the Wiener model (a difference of 2.3222°). For NRMSE, CB-TCN has a value of 0.09267, which is lower than the other models.

 Ankle Joint: The R 2 of the CB-TCN model is 0.8315, which is higher than the other models. It shows an improvement of 0.11685 compared to the second-best model, ED-TCN (R 2 = 0.71465), and a substantial difference of 0.3200 compared to the Wiener model. The RMSE is 3.5349°, relatively low, especially compared to the Wiener model (a difference of 1.2344°). For NRMSE, CB-TCN has a value of 0.0945, which is comparatively lower than the values of the other models.

 - 
 (3) Squat and Stand Motion

 Hip Joint: The R 2 of the CB-TCN model is 0.9089, which outperforms the other models. It shows an improvement of 0.05605 compared to the second-best model, ED-TCN (R 2 = 0.85285), and a significant difference of 0.1689 compared to the Wiener model. The RMSE is 2.4053°, lower than the other models, particularly the Wiener model (a difference of 1.9422°). For NRMSE, CB-TCN has a value of 0.1008, which is lower than the values of the other models.

 Knee Joint: The R 2 of the CB-TCN model is 0.9406, which is significantly higher than the other models. It shows an improvement of 0.02535 compared to the second-best model, ED-TCN (R 2 = 0.91525), and a notable difference of 0.2308 compared to the Wiener model. The RMSE is 2.5121°, lower than the other models, particularly the Wiener model (a difference of 2.2364°). For NRMSE, CB-TCN has a value of 0.08422, which is lower than the other models.

 Ankle Joint: The R 2 of the CB-TCN model is 0.9092, which demonstrates a clear advantage over the other models, particularly the Wiener model, with a difference of 0.1803. Compared to the second-best model, ED-TCN (R 2 = 0.8587), CB-TCN also shows a notable improvement. The RMSE is 4.7484°, relatively low, especially compared to the Wiener model (a difference of 0.8446°). For NRMSE, CB-TCN has a value of 0.0932, which is lower than the other models.

 - 
 (4) Flexion–Extension Motion

 Knee Joint: The R 2 of the CB-TCN model is 0.9717, which outperforms the other models, particularly the Wiener model, with a difference of 0.1042. Compared to the second-best model, ED-TCN (R 2 = 0.9579), CB-TCN also demonstrates superior performance. The RMSE is 1.5891°, significantly lower than the other models, especially the Wiener model (a difference of 2.2909°). For NRMSE, CB-TCN has a value of 0.05583, which is lower than the other models.

 Ankle Joint: The R 2 of the CB-TCN model is 0.9459, significantly higher than the other models, especially the Wiener model, with a difference of 0.1042. Compared to the second-best model, ED-TCN (R 2 = 0.9114), CB-TCN also shows a notable performance advantage. The RMSE is 5.1833°, lower than the other models, particularly the Wiener model (a difference of 6.0915°). For NRMSE, CB-TCN has a value of 0.07585, which is lower than the other models.

 The CB-TCN model outperforms all other models across all motion types, particularly in terms of R 2 , RMSE, and NRMSE, demonstrating higher prediction accuracy and smaller errors. The combined analysis of p -values for R 2 and RMSE, along with the Tukey HSD test, reveals that the proposed CB-TCN model exhibits significant differences in both fitting ability and predictive performance compared to other models. In gait motion, CB-TCN shows particularly excellent performance in the hip, knee, and ankle joints, with high precision and small errors. These joints play a crucial role in normal walking, where hip flexion–extension and knee flexion provide primary motion, while the ankle joint plays a critical role in propulsion and shock absorption. The model’s high accuracy reflects its ability to accurately capture the coordinated motion of these joints in routine, low-dynamic movements.

 However, in more complex movements such as obstacle crossing, squatting, and flexion–extension motions, the model exhibits some deviations in prediction accuracy. These deviations are primarily due to the varying dynamic range and non-linear muscle activation patterns during these movements, leading to slightly higher RMSE and NRMSE values. From a physiological perspective, obstacle crossing and squatting involve more complex muscle coordination, particularly the rapid changes in body weight distribution and stability, requiring higher degrees of muscle activation and joint coordination. Specifically, during squatting, the rectus femoris (RF) and biceps femoris (BF) must work in high synchronization to control knee flexion and maintain a stable posture. The rapid fluctuations in joint angles, particularly during deep squats, challenge the model’s ability to capture these dynamic changes. Additionally, the gastrocnemius (GAS) and soleus (SOL) muscles work together to stabilize the ankle joint during squatting and obstacle crossing. Due to changes in load and posture, the activation patterns of these muscles are highly non-linear, further increasing the prediction difficulty. In flexion–extension movements, the model also faces challenges because muscle activation patterns are typically non-linear and vary significantly with changes in movement speed, load, and posture. Muscles like the tibialis anterior (TA), rectus femoris (RF), and biceps femoris (BF) undergo rapid changes during flexion–extension, which significantly increases the complexity of the model’s prediction capability, especially under rapidly changing joint angles and load conditions.

 Despite these challenges, the CB-TCN model still demonstrates overall lower RMSE and NRMSE compared to other models, with R 2 remaining high, indicating strong generalizability.

### 4.3. Real-Time Analysis

 Table 1 presents the computation time (in milliseconds) required per operation for the model proposed in this study across different joints (hip, knee, ankle) and movements (walking, obstacle crossing, squatting, knee flexion–extension). The data are presented as mean ± standard deviation, reflecting the average inference time and stability of the model across multiple experiments.

#### Table 1.

 Prediction time for joint movements under different types of exercises (unit: ms).

 Activity 

 Hip Joint (ms) 

 Knee Joint (ms) 

 Ankle Joint (ms) 

 Walking 
 13.20 ± 0.048 
 13.53 ± 0.124 
 13.46 ± 0.056 

 Obstacle Crossing 
 34.59 ± 0.068 
 34.31 ± 0.044 
 34.01 ± 0.072 

 Squatting 
 28.68 ± 0.064 
 28.68 ± 0.076 
 27.77 ± 0.060 

 Knee Flexion-Extension 
 27.17 ± 0.060 
 27.18 ± 0.048 
 27.18 ± 0.048 

 Open in a new tab 

 As shown in the table above, the inference time of the model proposed in this study is consistently below 50 ms under all test conditions, meeting the real-time requirements. In the walking motion, the model exhibits the lowest inference time, demonstrating its excellent performance in simple motion tasks. In more complex motions, such as obstacle crossing, the model still maintains a high level of efficiency. These results indicate that the model can adapt to various motion modes and meet the demands for real-time feedback. All networks in this study were implemented using Python 3.8 and PyTorch 2.4.1. The training and evaluation processes were conducted on a computer equipped with an NVIDIA GeForce RTX 4060 GPU, ensuring efficient computation and reliable performance.

## 5. Conclusions

 In this study, the focus was on predicting joint angles in lower-limb movements using the CB-TCM model, applying it to rehabilitation and motion analysis. Predicting joint angles is essential for advancing rehabilitation technologies, particularly for individuals with impaired motor function due to injury or neurological conditions. By accurately modeling joint movements, this approach facilitates the development of adaptive therapies, improves the design of assistive devices like exoskeletons and prosthetics, and enhances the understanding of human biomechanics. The method involves training the model with data from walking, obstacle crossing, squatting, and knee flexion–extension movements, and evaluating it using key metrics such as R 2 , RMSE, and NRMSE. The model performs excellently across all movements, particularly in walking and knee extension. During walking, the knee joint achieved an R 2 value of 0.9718, with an RMSE as low as 1.2648°. The model still performs well in squatting and obstacle crossing tasks, despite slight increases in RMSE and NRMSE.

 In future research, we aim to evaluate the practical feasibility of the proposed model in clinical environments, focusing on its applicability to different patient populations and ensuring user-friendliness. To enhance the generalizability of our findings, we plan to expand the participant pool to include individuals with varying health conditions, ages, genders, and body types. This will help verify the model’s broad applicability and explore its performance across different populations, ensuring its sustainability, safety, and usability in real-world applications. Additionally, we will assess the model’s performance across different clinical settings and patient groups, such as those with neurological impairments, movement disorders, or limb deficiencies, to explore its potential challenges and benefits. To ensure its clinical viability, we will address several critical aspects, including the compatibility of the technology with existing medical equipment and rehabilitation tools, the impact of various factors (such as skin condition and sensor placement) on signal quality, and the development of a user-friendly interface for both healthcare professionals and patients. Furthermore, safeguarding patient data privacy and ensuring secure data handling will be a priority to comply with ethical and regulatory standards.

## Acknowledgments

 We would like to thank X.Z. for their help in the experiments and express our gratitude to all the participants who took part in the experiments.

## Supplementary Materials

 The following supporting information can be downloaded at: https://www.mdpi.com/article/10.3390/s25030719/s1 , Table S1: Model structure parameters; Table S2: Statistical indicators.

 sensors-25-00719-s001.zip (21.4KB, zip) 

## Author Contributions

 Conceptualization, Y.H.; methodology, Y.H.; software, Y.H.; resources, Q.T.; data curation, Y.H.; writing—original draft preparation, Y.H.; writing—review and editing, Q.T. and X.Z.; visualization, Y.H.; supervision, Q.T.; project administration, Q.T.; funding acquisition, Q.T. All authors have read and agreed to the published version of the manuscript.

## Institutional Review Board Statement

 The experiment was approved by the Ethics Committee of Xi’an Jiaotong University, approval number: 2024YS239. All experimental procedures were conducted in accordance with the relevant provisions of the Declaration of Helsinki.

## Informed Consent Statement

 Informed consent was obtained from all subjects involved in this study.

## Data Availability Statement

 If data are needed, please contact the corresponding author.

## Conflicts of Interest

 The authors declare no conflicts of interest.

## Funding Statement

 This research was funded by the National Natural Science Foundation of China (52365039); the Tianshan Talent Training Program (2023TSYCLJ0051); and The Excellent Doctoral Graduate Innovation Program (XJU2024BS103).

## Footnotes

 Disclaimer/Publisher’s Note: The statements, opinions and data contained in all publications are solely those of the individual author(s) and contributor(s) and not of MDPI and/or the editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to people or property resulting from any ideas, methods, instructions or products referred to in the content.

## References

- 
 1. Sarajchi M., Al-Hares M.K., Sirlantzis K. Wearable lower-limb exoskeleton for children with cerebral palsy: A systematic review of mechanical design, actuation type, control strategy, and clinical evaluation. IEEE Trans. Neural Syst. Rehabil. Eng. 2021;29:2695–2720. doi: 10.1109/TNSRE.2021.3136088. [ DOI ] [ PubMed ] [ Google Scholar ]

- 
 2. Wang G., Jin L., Zhang J., Duan X., Yi J., Zhang M., Sun Z. Recurrent Neural Network Enabled Continuous Motion Estimation of Lower Limb Joints from Incomplete sEMG Signals. IEEE Trans. Neural Syst. Rehabil. Eng. 2024;32:3577–3589. doi: 10.1109/TNSRE.2024.3459924. [ DOI ] [ PubMed ] [ Google Scholar ]

- 
 3. Zhou J., Yang S., Xue Q. Lower limb rehabilitation exoskeleton robot: A review. Adv. Mech. Eng. 2021;13:16878140211011862. doi: 10.1177/16878140211011862. [ DOI ] [ Google Scholar ]

- 
 4. Li W.Z., Cao G.Z., Zhu A.B. Review on control strategies for lower limb rehabilitation exoskeletons. IEEE Access. 2021;9:123040–123060. doi: 10.1109/ACCESS.2021.3110595. [ DOI ] [ Google Scholar ]

- 
 5. Gordleeva S.Y., Lobov S.A., Grigorev N.A., Savosenkov A.O., Shamshin M.O., Lukoyanov M.V., Khoruzhko M.A., Kazantsev V.B. Real-time EEG–EMG human–machine interface-based control system for a lower-limb exoskeleton. IEEE Access. 2020;8:84070–84081. doi: 10.1109/ACCESS.2020.2991812. [ DOI ] [ Google Scholar ]

- 
 6. Chen D., Cai Y., Qian X., Ansari R., Xu W., Chu K.-C., Huang M.-C. Bring gait lab to everyday life: Gait analysis in terms of activities of daily living. IEEE Internet Things J. 2019;7:1298–1312. doi: 10.1109/JIOT.2019.2954387. [ DOI ] [ Google Scholar ]

- 
 7. Scherpereel K., Molinaro D., Inan O., Shepherd M., Young A. A human lower-limb biomechanics and wearable sensors dataset during cyclic and non-cyclic activities. Sci. Data. 2023;10:924. doi: 10.1038/s41597-023-02840-6. [ DOI ] [ PMC free article ] [ PubMed ] [ Google Scholar ]

- 
 8. Kuang Y., Wu Q., Shao J., Wu J., Wu X. Extreme learning machine classification method for lower limb movement recognition. Clust. Comput. 2017;20:3051–3059. doi: 10.1007/s10586-017-0985-2. [ DOI ] [ Google Scholar ]

- 
 9. Qin P., Shi X. Evaluation of feature extraction and classification for lower limb motion based on sEMG signal. Entropy. 2020;22:852. doi: 10.3390/e22080852. [ DOI ] [ PMC free article ] [ PubMed ] [ Google Scholar ]

- 
 10. Wu X., Yuan Y., Zhang X., Wang C., Xu T., Tao D. Gait phase classification for a lower limb exoskeleton system based on a graph convolutional network model. IEEE Trans. Ind. Electron. 2021;69:4999–5008. doi: 10.1109/TIE.2021.3082067. [ DOI ] [ Google Scholar ]

- 
 11. Sun Z., Zhang X., Liu K., Shi T., Wang J. A multi-joint continuous motion estimation method of lower limb using least squares support vector machine and zeroing neural network based on semg signals. Neural Process. Lett. 2023;55:2867–2884. doi: 10.1007/s11063-022-10988-2. [ DOI ] [ Google Scholar ]

- 
 12. Wang F., Lu J., Fan Z., Ren C., Geng X. Continuous motion estimation of lower limbs based on deep belief networks and random forest. Rev. Sci. Instrum. 2022;93:044106. doi: 10.1063/5.0057478. [ DOI ] [ PubMed ] [ Google Scholar ]

- 
 13. Li W., Liu K., Sun Z., Li C., Chai Y., Gu J. A neural network-based model for lower limb continuous estimation against the disturbance of uncertainty. Biomed. Signal Process. Control. 2022;71:103115. doi: 10.1016/j.bspc.2021.103115. [ DOI ] [ Google Scholar ]

- 
 14. Sommer L.F. EMG-Driven Exoskeleton Control. Universidade de São Paulo; São Paulo, Brazil: 2019. [ Google Scholar ]

- 
 15. Xie H., Li G., Zhao X., Li F. Prediction of limb joint angles based on multi-source signals by GS-GRNN for exoskeleton wearer. Sensors. 2020;20:1104. doi: 10.3390/s20041104. [ DOI ] [ PMC free article ] [ PubMed ] [ Google Scholar ]

- 
 16. Liu G., Zhang L., Han B., Zhang T., Wang Z., Wei P. sEMG-based continuous estimation of knee joint angle using deep learning with convolutional neural network; Proceedings of the 2019 IEEE 15th international conference on automation science and engineering (CASE); Vancouver, BC, Canada. 22–26 August 2019; pp. 140–145. [ Google Scholar ]

- 
 17. Song Q., Ma X., Liu Y. Continuous online prediction of lower limb joints angles based on sEMG signals by deep learning approach. Comput. Biol. Med. 2023;163:107124. doi: 10.1016/j.compbiomed.2023.107124. [ DOI ] [ PubMed ] [ Google Scholar ]

- 
 18. Ma C., Guo W., Zhang H., Samuel O.W., Ji X., Xu L., Li G. A novel and efficient feature extraction method for deep learning based continuous estimation. IEEE Robot. Autom. Lett. 2021;6:7341–7348. doi: 10.1109/LRA.2021.3097257. [ DOI ] [ Google Scholar ]

- 
 19. Ding G., Plummer A., Georgilas I. Deep learning with an attention mechanism for continuous biomechanical motion estimation across varied activities. Front. Bioeng. Biotechnol. 2022;10:1021505. doi: 10.3389/fbioe.2022.1021505. [ DOI ] [ PMC free article ] [ PubMed ] [ Google Scholar ]

- 
 20. Fan J., Zhang K., Huang Y., Zhu Y., Chen B. Parallel spatio-temporal attention-based TCN for multivariate time series prediction. Neural Comput. Appl. 2023;35:13109–13118. doi: 10.1007/s00521-021-05958-z. [ DOI ] [ Google Scholar ]

- 
 21. Chen C., Guo W., Ma C., Yang Y., Wang Z., Lin C. sEMG-based continuous estimation of finger kinematics via large-scale temporal convolutional network. Appl. Sci. 2021;11:4678. doi: 10.3390/app11104678. [ DOI ] [ Google Scholar ]

- 
 22. Wang F., Liang W., Afzal H.M.R., Fan A., Li W., Dai X., Liu S., Hu Y., Li Z., Yang P. Estimation of Lower Limb Joint Angles and Joint Moments during Different Locomotive Activities Using the Inertial Measurement Units and a Hybrid Deep Learning Model. Sensors. 2023;23:9039. doi: 10.3390/s23229039. [ DOI ] [ PMC free article ] [ PubMed ] [ Google Scholar ]

- 
 23. Du J., Liu Z., Dong W., Zhang W., Miao Z. A Novel TCN-LSTM Hybrid Model for sEMG-Based Continuous Estimation of Wrist Joint Angles. Sensors. 2024;24:5631. doi: 10.3390/s24175631. [ DOI ] [ PMC free article ] [ PubMed ] [ Google Scholar ]

- 
 24. Hamner S.R., Delp S.L. Muscle contributions to fore-aft and vertical body mass center accelerations over a range of running speeds. J. Biomech. 2013;46:780–787. doi: 10.1016/j.jbiomech.2012.11.024. [ DOI ] [ PMC free article ] [ PubMed ] [ Google Scholar ]

- 
 25. Hermens H., Frenks H.B. Surface electromyography application areas and parameters; Proceedings of the Third General SENIAM Workshop; Aachen, Germany. 15–16 May 1998. [ Google Scholar ]

- 
 26. Liang J., Shi Z., Zhu F., Chen W., Chen X., Li Y. Gaussian process autoregression for joint angle prediction based on sEMG signals. Front. Public Health. 2021;9:685596. doi: 10.3389/fpubh.2021.685596. [ DOI ] [ PMC free article ] [ PubMed ] [ Google Scholar ]

- 
 27. Wahid M.F., Tafreshi R., Langari R. A multi-window majority voting strategy to improve hand gesture recognition accuracies using electromyography signal. IEEE Trans. Neural Syst. Rehabil. Eng. 2019;28:427–436. doi: 10.1109/TNSRE.2019.2961706. [ DOI ] [ PubMed ] [ Google Scholar ]

- 
 28. Ma C., Li W., Cao J., Du J., Li Q., Gravina R. Adaptive sliding window based activity recognition for assisted livings. Inf. Fusion. 2020;53:55–65. doi: 10.1016/j.inffus.2019.06.013. [ DOI ] [ Google Scholar ]

- 
 29. Hewage P., Behera A., Trovati M., Pereira E., Ghahremani M., Palmieri F., Liu Y. Temporal convolutional neural (TCN) network for an effective weather forecasting using time-series data from the local weather station. Soft Comput. 2020;24:16453–16482. doi: 10.1007/s00500-020-04954-0. [ DOI ] [ Google Scholar ]

- 
 30. Tsinganos P., Jansen B., Cornelis J., Skodras A. Real-time analysis of hand gesture recognition with temporal convolutional networks. Sensors. 2022;22:1694. doi: 10.3390/s22051694. [ DOI ] [ PMC free article ] [ PubMed ] [ Google Scholar ]

- 
 31. Wang S., Huang L., Jiang D., Sun Y., Jiang G., Li J., Zou C., Fan H., Xie Y., Xiong H., et al. Improved multi-stream convolutional block attention module for sEMG-based gesture recognition. Front. Bioeng. Biotechnol. 2022;10:909023. doi: 10.3389/fbioe.2022.909023. [ DOI ] [ PMC free article ] [ PubMed ] [ Google Scholar ]

- 
 32. Zanghieri M., Benatti S., Burrello A., Kartsch V., Conti F., Benini L. Robust real-time embedded EMG recognition framework using temporal convolutional networks on a multicore IoT processor. IEEE Trans. Biomed. Circuits Syst. 2019;14:244–256. doi: 10.1109/TBCAS.2019.2959160. [ DOI ] [ PubMed ] [ Google Scholar ]

- 
 33. Kirti, Rajpal N. A multi-crop disease identification approach based on residual attention learning. J. Intell. Syst. 2023;32:20220248. doi: 10.1515/jisys-2022-0248. [ DOI ] [ Google Scholar ]

## Associated Data

 This section collects any data citations, data availability statements, or supplementary materials included in this article. 

### Supplementary Materials

 sensors-25-00719-s001.zip (21.4KB, zip) 

### Data Availability Statement

 If data are needed, please contact the corresponding author.

## ACTIONS

 - 

 View on publisher site 

 - 

 PDF (3.6 MB) 

 - 

 Cite 

 - 

 Collections 

 - 

 Permalink 

## PERMALINK

 Copy 

## RESOURCES

### 
 
 Similar articles

### 
 
 Cited by other articles

### 
 
 Links to NCBI Databases

## Cite

 - 

 Copy 

 - 

 Download .nbib 
 .nbib 

 - 

 Format: 

 AMA

 APA

 MLA

 NLM

## Add to Collections

 Create a new collection 

 Add to an existing collection 

 Name your collection
 * 

 Choose a collection

 Unable to load your collection due to an error

 Please try again 

 Add

 Cancel

 Back to Top 

- 

- 
- 
- 

-