<!-- source: https://doi.org/10.1109/tnsre.2023.3315373 -->
<!-- title: Estimation of Joint Torque by EMG-Driven Neuromusculoskeletal Models and LSTM Networks -->
<!-- fetched: 2026-09-30T13:46:08Z -->
<!-- via: local_paper_fulltext(_translation_outputs/Zhang_2023_TNSRE_JointTorqueLSTM/Zhang_2023_TNSRE_JointTorqueLSTM/auto/Zhang_2023_TNSRE_JointTorqueLSTM.md) -->
<!-- note: 原文 PDF 存于 09_章隆彬_合作团队/papers/，此为 PDF 解析后的全文 Markdown -->

# Estimation of Joint Torque by EMG-driven Neuromusculoskeletal Models and LSTM Networks

Longbin Zhang, Davit Soselia, Ruoli Wang, and Elena M. Gutierrez-Farewik

Abstract—Accurately predicting joint torque using wearable sensors is crucial for designing assist-as-needed exoskeleton controllers to assist muscle-generated torque and ensure successful task performance. In this paper, we estimated ankle dorsiflexion/plantarflexion, knee flexion/extension, hip flexion/extension, and hip abduction/adduction torques from electromyography (EMG) and kinematics during daily activities using neuromusculoskeletal (NMS) models and long short-term memory (LSTM) networks. The joint torque ground truth for model calibrating and training was obtained through inverse dynamics of captured motion data. A cluster approach that grouped movements based on characteristic similarity was implemented, and its ability to improve the estimation accuracy of both NMS and LSTM models was evaluated. We compared torque estimation accuracy of NMS and LSTM models in three cases: Pooled, Individual, and Clustered models. Pooled models used data from all 10 movements to calibrate or train one model, Individual models used data from each individual movement, and Clustered models used data from each cluster. Individual, Clustered and Pooled LSTM models all had relatively high joint torque estimation accuracy. Individual and Clustered NMS models had similarly good estimation performance whereas the Pooled model may be too generic to satisfy all movement patterns. While the cluster approach improved the estimation accuracy in NMS models in some movements, it made relatively little difference in the LSTM neural networks, which already had high estimation accuracy. Our study provides practical implications for designing assist-asneeded exoskeleton controllers by offering guidelines for selecting the appropriate model for different scenarios, and has potential to enhance the functionality of wearable exoskeletons and improve rehabilitation and assistance for individuals with motor disorders.

Index Terms—neural networks, joint torque prediction, neuromusculoskeletal modeling, cluster analysis, data-driven biomechanical models

## I. INTRODUCTION

Exoskeletons are investigated with increasing frequency due to their potential to facilitate movement for persons with disability, for instance, to facilitate neuromuscular rehabilitation in people with motor disorders [1]. Many recent studies have investigated exoskeletons’ potential for stimulating neuromuscular recovery during neurorehabilitation training by operating in a feedback loop that incorporates users’ biosignals, such as electromyography (EMG) [2], [3]. For example, Durandau et al. [4] developed an EMG-based controller that computes users’ joint torque to determine the exoskeleton assistance level, enabling individuals with incomplete spinal cord injury or post-stroke to voluntarily control the exoskeleton to follow the target positions. Consequently, research focus on exoskeleton control strategies that integrate the user’s torque capabilities continues to grow exponentially. The appropriate assistive torque is frequently predicted based on the user’s EMG signals, as they adapt to the user’s muscle function and can encourage users’ active participation, thereby facilitating motor recovery. This control strategy, often referred to as “assist-as-needed”, must sufficiently assist the musclegenerated torque to ensure that the user successfully performs the intended task. Achieving an accurate prediction of the user’s torque is the first step in achieving this goal.

To predict joint torque, neuromusculoskeletal (NMS) modelling and artificial neural networks (ANNs) are two common methods, and each has advantages and disadvantages [5]– [7]. Our recent study [8] was the first study that compared ankle joint torque estimation performance of a physics-based NMS model to a multilayer perceptron (MLP) model in gait and isokinetic movements. We highlighted the practicality of the NMS approach, which leverages domain knowledge to replicate the conversion process from EMGs to musculotendon forces and joint torque. However, setting up/calibrating the models in sophisticated steps can be labor-intensive, and it’s important to note that physics-based NMS models offer only an approximate estimate of joint dynamics. Furthermore, EMG quality can be influenced by issues such as cross-talk and the precise positioning of sensors. The MLP model could estimate joint torques better but requires a varied set of training data. MLP is a classical and simple type of neural network with no time-series or memory information. However, how torque predictions from other neural network structures compare to those from NMS models has not been thoroughly investigated. LSTMs, types of neural networks reported to be the most robust and effective method for time-series prediction [9]–[11], can selectively remember patterns that correspond to a period of time by incorporating memory structures into networks. The memory structures are able to update the information between time intervals with gates to select what to output, forget and remember. A similar memory structure is included in the physics-based NMS model which uses MTU neural activation information from two previous time steps. Thus, comparing physics-based NMS models and LSTM networks is intriguing. Despite the advances made in joint torque estimation using physics-based NMS and LSTM models, limited research has been conducted on torque estimation in multiple lower limb degrees of freedom (DOFs) during daily activities using NMS and LSTM models in parallel.

![](images/4e7dd924fbdaad269339624bb6d8687259f9184f5a0edd832976380dbf15dc80.jpg)

![](images/e72e750d23d1f1d0962cf82dcfc2a4866e440597767a371485a5293521c20b0b.jpg)  
Fig. 1. Experimental setup

Cluster approaches have been applied to classify gait and movement patterns in persons with and without motor disorders [12]–[14], muscle activation levels in non-pathological groups [15], and moment patterns in athletes with identical anterior cruciate ligament injury rates [16]. Cluster analysis is a statistical technique for grouping observations or patterns into a number of clusters based on similarity in their characteristics. Phinyomark et al. [13] clustered running kinematic patterns into two distinct groups; their results suggested that variability observations in running patterns may be attributable to different gait strategies. White et al. [15] applied cluster analysis to identify muscle activation patterns in abdominal and lumbar muscles during gait. In our recent study of ankle joint torque estimation by NMS models in gait and isokinetic movements [8], an NMS model calibrated with a varied set of movement trials had lower prediction accuracy than Individual models that were each calibrated with a specific movement. However, calibrating one model for each movement is timeconsuming and impractical in real-time applications. A logistic question that arises is whether calibration with a cluster approach in both EMG-driven NMS models and in LSTM models can result in high joint torque prediction accuracy.

The objective of this study was two-fold: (1) to estimate ankle dorsiflexion/plantarflexion (D/P), knee flexion/extension (F/E) torques, hip flexion/extension and hip abduction/adduction (Ab/Ad) by NMS and LSTM models with a cluster approach based on EMG signals and kinematics; and (2) to compare the prediction performance between Clustered NMS and LSTM models.

## II. METHODS

## A. Data Collection and Data Processing

Eight able-bodied participants (age: 29±4 years; sex: 4F/4M; weight: 65.2±17.8 kg; height: 168.1±9.4 cm) were enrolled in this study. Approval for this research was granted by the Swedish Ethical Review Authority (Reference Number: Dnr. 2020-02311). All participants gave their informed written consent. All participants were required to perform ten movement types at least 10 times each (Fig. 1), specifically normal walking, slow walking, fast walking, stand-to-sit, sitto-stand, squat, jump up to a 15-cm stair, jump down from

a 15-cm stair, vertical jump up, and land from vertical jump.   
The movements were arranged in a randomized sequence.

Surface EMG signals (Cometa aktos, Italy) were placed unilaterally on the tibialis anterior (TA), gastrocnemius medialis (GM), gastrocnemius lateralis (GL), soleus (SOL), peroneus longus (PL), vastus medialis (VM), vastus lateralis (VL), rectus femoris (RF), biceps femoris (BF), semitendinosus (ST), tensor fasciae latae (TFL), gluteus medius (GLME), and gluteus maximus (GLMA) and sampled at 1000 Hz. The selected leg was randomized. Raw EMG signals were bandpass filtered (30-300 Hz), rectified, low pass filtered (6 Hz), and normalized to the maximum EMG value among all recorded trials [5], [17], [18].

A 3D motion capture system (100 Hz, V16, Vicon, Oxford, UK) was used to capture marker trajectories, which were placed according to the CGM2.3 marker set model [19]. Three force plates (1000 Hz, AMTI, MA, USA) was used to record ground reaction forces.

Segment and joint kinematics were estimated via inverse kinematics by solving a weighted least square problem to minimize the error between modeled $x _ { i }$ and measured $\boldsymbol { x } _ { i } ^ { e x p }$ marker trajectories [20] (Eq. (1)). The modeled marker positions were placed at the same anatomical locations as the experimental marker positions (CGM2.3 marker set model).

$$
\operatorname* { m i n } _ { \boldsymbol { q } } ( \sum _ { i } ^ { N } \theta _ { i } \| \boldsymbol { x } _ { i } ^ { e x p } - \boldsymbol { x } _ { i } \| ^ { 2 } ) ,\tag{1}
$$

where $q$ is the generalized coordinates of the model and $\theta _ { i }$ is the weight of ith marker.

Kinetics, or specifically joint torques, were computed via inverse dynamics; these were considered as ground truth of joint torque for model calibrating or training. Kinetics were computed by solving the dynamic equations of motion [21] (Eq. (2)),

$$
M ( \mathbf { q } ) \ddot { \mathbf { q } } + \mathbf { C } ( \mathbf { q } , \dot { \mathbf { q } } ) + \mathbf { G } ( \mathbf { q } ) + R ( \mathbf { q } ) \mathbf { F } ^ { \mathbf { m t } } + \mathbf { F } _ { \mathbf { e } } = 0\tag{2}
$$

where ${ \bf q } , \dot { \bf q } ,$ ¨q represent the generalised position, velocity and acceleration vectors, respectively; $M ( \mathbf { q } ) \ddot { \mathbf { q } }$ represents the inertial forces and torques, and $M ( \mathbf { q } )$ denotes the mass matrix; $\mathbf { C } ( \mathbf { q } , \dot { \mathbf { q } } )$ represents the centripetal and Coriolis forces and torques; $\mathbf G ( \mathbf q )$ represents the gravitational forces and torques; $R ( \mathbf { q } )$ denotes the muscle moment arms; $\mathbf { F ^ { m t } }$ represents the musculotendon forces vector and $R ( \mathbf { q } ) \mathbf { F } ^ { \mathbf { m t } }$ denotes the musculotendon torques vector; $\mathbf { F _ { e } }$ represents the external force and torques. Joint kinematics and kinetics were filtered using a low-pass fourth-order zero-lag Butterworth filter (6 Hz). [22]– [24].

## B. Joint Torque Prediction Models

Both NMS models and LSTM neural networks were created to estimate ankle D/P, knee F/E, hip F/E, and hip Ab/Ad torques, using only input from the 13 captured EMG signals and joint kinematics (Fig. 2). The joint kinematics data included ankle D/P, knee F/E, hip F/E, and hip Ab/Ad joint angles.

(a) NMS Model

![](images/a8d4354e955092c43c747fac11e4cbb4903a11c906d142d3aaf47f9aefd9cb2b.jpg)  
Fig. 2. Schematic diagram of joint torque estimation using different NMS and LSTM models. Joint torques were estimated using Pooled models calibrated/trained by all movements, Individual models calibrated/trained by one single movement, and Clustered models calibrated/trained by one single cluster. The inputs for NMS and LSTM models were EMGs and joint angles, acquired and post-processed from a 3D motion capture system.

An open-source EMG-driven NMS model (CEINMS) [25] was used (Fig. 3 (a)), including muscle activation dynamics, musculotendon kinematics, muscle contraction dynamics, and joint dynamics [26]. Specifically, the muscle activation dynamics part is for computing muscle activation with EMG data. The relationship between EMG excitation e(t) and neural activation u(t) was formulated as Eq. (3) [27],

$$
u ( t ) = \alpha \cdot e ( t - d ) - \beta _ { 1 } \cdot u ( t - 1 ) - \beta _ { 2 } \cdot u ( t - 2 )\tag{3}
$$

where α represents the muscle gain variable; d corresponds to the electromechanical delay. $\beta _ { 1 }$ and $\beta _ { 2 }$ represent the recursive variables; they are constrained to $\beta _ { 1 } = C _ { 1 } + C _ { 2 } , \beta _ { 2 } = C _ { 1 } \cdot C _ { 2 }$ where $\left| C _ { 1 } \right| < 1 , \left| C _ { 2 } \right| < 1$ , and $\alpha - \beta _ { 1 } - \beta _ { 2 } = 1$ , in order to obtain a stable solution [25] [27] [28]. The muscle activation a(t) can be represented as Eq. (4),

$$
a ( t ) = { \frac { e ^ { A u ( t ) } - 1 } { e ^ { A } - 1 } }\tag{4}
$$

where A is the shape factor [18], [28].

The musculotendon kinematics part calculates the musculotendon lengths and moment arms. Muscle force is determined through the muscle contraction dynamics component using a Hill-type muscle model. The muscle force $F$ can be represented as Eq. (5),

$$
F = F _ { 0 } ^ { m } [ F _ { a l } ( l ) \cdot F _ { v } ( v ) \cdot a + F _ { p l } ( l ) + d _ { p } \cdot v ] c o s ( \theta )\tag{5}
$$

where $F _ { 0 } ^ { m }$ represents the muscle maximum isometric force; l is the fiber length; $F _ { a l } ( l )$ describes the muscle active force and fiber length relationship; v is the fiber contraction velocity; the $F _ { v } ( v )$ denotes the muscle force and fiber contraction velocity relationship; $F _ { p l } ( l )$ represents the muscle passive force and fiber length relationship; θ is the fiber pennation angle and $d _ { p }$ represents the muscle damping.

Finally, the joint dynamics part is for computing the joint torque based on muscle forces and moment arms.

Parameter calibration in the EMG-driven NMS model was performed according to recommendations by Pizzolato et al. [25]. For each participant, optimal fiber length $l _ { 0 } ^ { m }$ and tendon slack length $l _ { s } ^ { t }$ were bounded within a range of ±15% of their initials. The parameter A was limited to the interval (−3, 0) while the coefficients $C _ { 1 }$ and $C _ { 2 }$ were bounded in (−1, 1). To calibrate the maximum isometric force, a strength coefficient was employed, which was bounded within the range of 0.5 to 2.5. These parameters were determined by applying a simulated annealing algorithm, which minimized the discrepancy between estimated and measured joint torques (calculated by inverse dynamics) throughout the calibration process. The simulated annealing algorithm was set to run before the average alteration in the target function’s value was less than the specified tolerance of $1 0 ^ { - 5 }$

![](images/818873911b46007532319bdca44f9d69a902d858bc311e98fa15e2e60b1a2394.jpg)

![](images/35103a1a2e6f7ec07bc172cead7fc656e35f59ca6966c63766aee0fac8521129.jpg)  
Fig. 3. (a) Schematic of an NMS model structure including four parts: (1) muscle activation dynamics part, which calculates muscle activation with EMG data; (2) musculotendon kinematics part, that computes musculotendon lengths and moment arms; (3) muscle contraction dynamics part, which is for calculating muscle force; and (4) joint dynamics part, which is for computing joint torques. (b) Schematic of an LSTM network model structure. The inputs to the LSTM networks were the same as those to the NMS models, namely joint angles and EMG signals. The time-series inputs were converted into time-slice data for the next LSTM layer. After that, LSTM layers were added, then a dense layer and a dropout layer. Finally, the output layer predicted joint torques.

## (b) LSTM Networks

Using identical input data as the NMS model, namely EMGs and joint angles, LSTM networks were constructed to estimate joint torques (Fig. 3 (b)) [25]. The time-series inputs were converted into time-slice data for the next LSTM layer. After that, LSTM layers were added, then a dense layer and a dropout layer. The dropout layer was applied to avoid overfitting whereas the dense layer made use of the intermediates from the LSTM layers to assist in the joint torque estimation throughout the regression phase [29]. Finally, the output layer predicted joint torques.

To find the hyperparameters for the LSTM networks, we used a “Coarse-to-fine” random search method [30]. A Xavier weight initialization method with zero bias was applied, and a batch size of 50 was selected. Each LSTM layer was configured with 50 neurons and tanh activation function was applied. 64 neurons were selected in the dense layer. The dropout rate for the dropout layer was set to 0.4. Throughout the training process, the loss function was defined as the mean square error between predicted and measured joint torques. The learning rate for the Adam optimizer was set to $1 0 ^ { - 3 }$ . The optimum model was trained throughout 4000 epochs; however, it was stopped early if the accuracy of the predictions did not improve in a consecutive 50 epochs.

## C. Cluster Approach

Using EMG signals and joint angles as inputs, we applied a t-distributed stochastic neighbor embedding (t-SNE) [31] clustering approach to reduce data dimensionality. Subsequently, we employed a K-means algorithm [32] to classify different movement types into K distinct clusters.

t-SNE is a valuable technique as it transforms highdimensional data into a low-dimensional space while concurrently maintaining the data’s local structure. Suppose $\mathbf { x } _ { i }$ and $\mathbf { x } _ { j }$ denote two data points in high-dimensional space. Their similarity, defined as $p _ { j | i }$ , is given by the following equation:

$$
p _ { j | i } = \frac { \exp ( - | | \mathbf { x } _ { i } - \mathbf { x } _ { j } | | ^ { 2 } / 2 \sigma _ { i } ^ { 2 } ) } { \sum _ { k \neq i } \exp ( - | | \mathbf { x } _ { i } - \mathbf { x } _ { k } | | ^ { 2 } / 2 \sigma _ { i } ^ { 2 } ) }\tag{6}
$$

where $\sigma _ { i }$ symbolizes the variance of a Gaussian distribution with its center at $\mathbf { x } _ { i }$

The t-SNE algorithm minimizes the Kullback-Leibler divergence, denoted as $C ,$ , between the two aforementioned distributions. This minimization is accomplished through the application of gradient descent, as illustrated in Eq. (7),

$$
C = \sum _ { i } \sum _ { j } p _ { j | i } \log { \frac { p _ { j | i } } { q _ { j | i } } }\tag{7}
$$

## D. Evaluation Protocol

We investigated the lower limb joint torque prediction accuracy of both NMS and LSTM models in three cases: Pooled, Individual, and Clustered models. We compared the prediction accuracy between NMS and LSTM in clustered models.

• Pooled model (n = 1): used 80% of the data from all ten movements to train one Pooled LSTM model. Meanwhile, the Pooled NMS model was calibrated with one trial of each included movement and was subsequently evaluated on the identical dataset as the LSTM model. To be specific, 80% of the data corresponds to 8 out of the 10 trials used for model training.

• Individual models $( n = 1 0 )$ : used 80% of the data from each movement to train ten individual LSTM models. Meanwhile, NMS models were calibrated with three trials of each movement and evaluated on the identical dataset as the LSTM models.

• Clustered models (n = 3): used 80% of the data from each cluster to train three clustered LSTM models. Meanwhile, NMS models were calibrated with two trials of each cluster and evaluated on the identical dataset as the LSTM models.

Normalized root mean square error (NRMSE) $E _ { N R M S }$ was used to evaluate the estimation error of each model; low estimation error was regarded as high estimation accuracy. NRMSE was established as the root mean square error E<sub>RMS</sub> between the estimated and measured/actual torque, normalized by the range of observed joint torque during corresponding movements:

![](images/b0ff0b12802cfbd6608f0abb799cd118abddebfc1e6cb06214bc8679ec4c058c.jpg)  
Fig. 4. Illustration of $K \ : = \ : 3$ identified clusters in 2D space by the tdistributed stochastic neighbor embedding algorithm.

$$
\begin{array} { c c l } { \displaystyle E _ { R M S } } & { = } & { \displaystyle \sqrt { \frac { 1 } { N } \sum _ { n = 1 } ^ { N } ( y _ { e , n } - y _ { n } ) ^ { 2 } } } \end{array}\tag{8}
$$

$$
\ E _ { N R M S } \ = \ \frac { E _ { R M S } } { \left( y _ { m a x } - y _ { m i n } \right) } \times 1 0 0 \%\tag{9}
$$

where $y _ { e , n }$ and $y _ { n }$ represent the estimated and measured/actual torque, respectively; $y _ { m a x }$ and $y _ { m i n }$ are the maximum and minimum actual torque in corresponding motions, respectively. To assess data distribution normality, Shapiro-Wilk tests were conducted $( p < 0 . 0 5$ significance level). Subsequently, pairwise comparisons of the NRMSEs predicted by the three methods were conducted. Paired t-tests were applied to normally distributed data, and Wilcoxon signed-rank tests were used for data that did not follow a normal distribution, in both cases with Bonferroni correction $( p < 0 . 0 5$ significance level).

## III. RESULTS

## A. Movement Clusters

The t-SNE algorithm combined with K-means identified three clusters among the ten movements. The t-SNE algorithm reduced the dimensionality of the data, allowing for visualization in a 2D space (Fig. 4). Subsequently, K-means grouped the samples of each movement into three clusters: $K _ { 1 }$ comprising normal walking, slow walking, and fast walking; $K _ { 2 }$ consisting of stand-to-sit, land from vertical jump, and squat; and $K _ { 3 }$ including sit-to-stand, vertical jump up, jump up to a 15-cm stair, and jump down from a 15-cm stair. The median location of the movement samples was employed to assign them to their corresponding clusters based on relevance. The choice of the number K of clusters in K-means, as well as the naming of each cluster, was determined through visual inspection, ensuring meaningful and interpretable grouping of the movements.

![](images/799287459403ad7ae60e942b84d43e2cd43e33a99422ed9592ae4e8a28e9a82a.jpg)  
Fig. 5. The NRMSE distributions between predicted and actual torques during ten motions among participants in NMS models: Pooled, Individual, and Clustered, shown with violin plots. Each violin plot includes a kernel density plot with a box plot. Kernel density plots illustrate the NRMSE probability distributions. Box plots denote NRMSE’s minimum, lower quartile, median, upper quartile, and maximum values. ∗ indicated that there was a significant difference between two models, determined by pairwise comparisons.

## B. NMS models

Overall, the estimation accuracy was similar between Individual and Clustered models for walking movements and vertical jump up. The estimation accuracy by the Pooled model was much lower than Clustered and Individual models in most motions, i.e., walking movements, sit-to-stand, vertical jump up, and jump up to a stair (Fig. 5).

The ankle D/P joint torque prediction accuracy from both Individual and Clustered models was significantly higher than by Pooled models for gait at different speeds (Fig. 5, slow walking: $p < 0 . 0 1$ and $p = 0 . 0 1$ ; normal walking: $p < 0 . 0 1$ and $p \ < \ 0 . 0 1 ;$ ; fast walking: $p ~ < ~ 0 . 0 1$ and $p ~ < ~ 0 . 0 1 )$ In ankle D/P torque prediction for gait at different speeds, compared to Individual and Clustered models, Pooled models underestimated peak plantarflexion. In addition, a plantarflexion offset during the swing phase was predicted by all three models (Fig. 6). In other movements, compared to Pooled models, significantly higher estimation accuracy was found in Individual models for vertical jump up $( p = 0 . 0 4 )$ , jump up to a stair $( p = 0 . 0 4 )$ , jump down from a stair $( p < 0 . 0 1 )$ , and squat $( p = 0 . 0 5 )$ , and somewhat, though not significantly, higher in Clustered models.

The knee F/E joint torque prediction accuracy from both Individual and Clustered models was relatively higher than by Pooled models (Fig. 5) for gait at different speeds (NRMSE $P o o l e d \le 2 6 . 7 \%$ , Individual $\leq 1 3 . 9 \%$ and Clustered ≤ 14.3%), jump up to a stair (NRMSE P ooled $\leq ~ 1 9 . 7 \%$

Individual $\leq ~ 1 5 . 2 \%$ and Clustered $\leq ~ 1 7 . 3 \% )$ and sit to stand (NRMSE $P o o l e d \leq 1 0 . 5 \%$ $I n d i v i d u a l \ \leq \ 8 . 8 \%$ and $C l u s t e r e d \ \leq \ 6 . 8 \% )$ . Unlike Clustered and Individual models, Pooled models overestimated knee extension torque during preswing-early swing for gaits at different speeds and underestimated knee extension moment in sit-to-stand (Fig. 6). All models failed to predict a smooth knee extension moment during squat, and Individual models failed to predict knee extension moment accurately in the stand-to-sit motion.

The joint torque estimation accuracy in hip F/E and hip Ab/Ad from Individual and Clustered models for all movements was overall lower than that in knee F/E and in ankle D/P: hip F/E (median NRMSE across ten movements: Individual 14.4%; Clustered 16.1%) and hip Ab/Ad torque (Individual 14.9%, Clustered 19.3%) vs. knee F/E (Individual 12.2%, Clustered 12.7%) and ankle D/P (Individual 11.0%, Clustered 15.5%) for the ten movements. In ankle D/P joint torque prediction, Pooled models had a lower prediction accuracy (median NRMSE 20.6%) compared to the other three joints (knee F/E 14.4%, hip F/E: 18.9%, and hip Ab/Ad 18.9%).

It is worth noting that during squat and stand to sit (Fig. 5), the average prediction accuracy by Individual models in knee F/E torque was lower than both Pooled and Clustered models, but outliers with high prediction error were observed in the Individual models. Moreover, the predicted knee F/E and hip F/E torques by Individual models agreed poorly with measured torque compared to the Pooled and Clustered models (Fig. 6).

## C. LSTM models

Overall, the estimation accuracy was similar among all three models (Pooled, Individual, and Clustered, Fig. 7), and the estimated joint torques are all in good agreement with the actual torques during ten motions (Fig. 8).

Among the torque prediction models at different joints (Fig. 7), Pooled, Individual, and Clustered models all estimated joint torque accurately with relatively low error (NRMSE in hip F/E: Pooled $\leq 8 . 4 \%$ , Individual ≤ 9.2%, and Clustered ≤ 8.2%; hip Ab/Ad: Pooled $\leq ~ 9 . 3 \%$ , Individual $\leq ~ 8 . 9 \%$ and Clustered $\begin{array} { r l } { \leq } & { { } 8 . 6 \% ; } \end{array}$ knee F/E: Pooled $\leq ~ 5 . 5 \%$ Individual $\leq 6 . 0 \% ,$ and Clustered $\leq 6 . 1 \% ;$ ankle D/P: $P o o l e d \ \leq \ 9 . 4 \%$ , Individual $\leq 8 . 5 \%$ , and Clustered ≤ 9.0%), especially for gait movements (NRMSE ankle D/P: $\leq 4 . 2 \%$ ; knee $\mathrm { F / E } \colon \le 6 . 1 \% \ ;$ hip $\mathrm { F / E } \leq 3 . 8 \% $ ; and hip Ab/Ad $\leq 5 . 7 \% )$

## D. Comparison between NMS and LSTM Clustered models

Overall, compared to the NMS Cluster models, the torque predictions from LSTM Cluster models demonstrated a superior agreement with the torque calculated through inverse dynamics at all joints and movement tests (Fig. 9); median NRMSE across all ten movements was in hip F/E: NMS : 16.1%, LSTM : 4.9%, in hip Ab/Ad: NMS : 19.3%, LSTM : 4.7%, and in ankle D/P: NMS : 15.5%, $L S T M : 6 . 7 \%$ . Similarly, knee F/E torque by the LSTM models was significantly better agreed with the torque calculated through inverse dynamics than the NMS models for

![](images/47239a784f6fedd0b6634314fb8a4b7951347b6b115f8b4281c5d8b451d6463d.jpg)  
Fig. 6. One example trial of measured and predicted torque during ten movements via three NMS models: P ooled, Individual, and Clustered.

$$
( p = 0 . 1 6 ) .
$$

![](images/e3af1565f0ea4857ba4653dc9cfbe7ea1c42af9382c4f9dc9df8d4118ae14314.jpg)  
Fig. 7. The NRMSE distributions between predicted and actual torques during ten motions among participants in LSTM models: Pooled, Individual, and Clustered, shown with violin plots. Each violin plot includes a kernel density plot with a box plot. Kernel density plots illustrate the NRMSE probability distributions. Box plots denote NRMSE’s minimum, lower quartile, median, upper quartile, and maximum values. ∗ indicated that there was a significant difference between two models, determined by pairwise comparisons.

most movements, including slow walking $( p < 0 . 0 1 )$ , normal walking $( p < 0 . 0 1 )$ , fast walking $( p = 0 . 0 1 )$ , jump down from a stair $( p \ < \ 0 . 0 1 )$ , jump up to a stair $( p \ < \ 0 . 0 1 )$ , vertical jump up $( p \ < \ 0 . 0 1 )$ , land from vertical jump $( p \ < \ 0 . 0 1 )$ , squat $( p = 0 . 0 3 )$ , and stand-to-sit $( p < 0 . 0 1 )$ , and to some extent, though not significantly, better agreed for sit-to-stand

## IV. DISCUSSION

We estimated ankle, knee and hip joint sagittal plane torques, and hip joint frontal plane torques for ten movements using two frequently used prediction approaches – EMGdriven NMS models and LSTM neural networks – and evaluated whether a cluster approach that grouped movements based on feature similarities can improve prediction accuracy, as compared to models based on individual or on pooled movements. We observed that, for LSTM neural network models, the Cluster approach did not have higher prediction accuracy than Individual or Pooled models; all three approaches predicted torques that agreed very well with inverse dynamics. Overall, the prediction accuracy in NMS models was lower than in LSTM models. In the NMS models, the Cluster approach did have higher prediction accuracy than the Pooled models in some movements. Whereas Individual and Clustered NMS models had similar prediction performance, the Pooled models predicted with the least accuracy. To the best of our knowledge, this is the first study that describes multiple lower-limb joint torque predictions using both NMS models and LSTM neural networks with a cluster approach in daily activities.

It is an interesting finding that the cluster approach did not improve prediction accuracy in LSTM neural networks but did improve it in NMS Pooled models in some movements. Clustered and Individual NMS models had similar joint torque prediction accuracy, but Pooled models had a lower prediction accuracy and predicted higher peak knee extension and lower peak plantarflexion torques for gait at different speeds (Fig. 6). This confirms the concept that movements within a cluster have similar coordination patterns, and that the NMS models can capture these features for better prediction accuracy. Among Pooled, Individual and Clustered NMS models, muscle-tendon unit parameters, such as tendon slack length, optimal fiber length, shape factor, and strength coefficient, were calibrated with trials from different movement types. During the calibration, these parameters were optimized by minimizing the error between estimated and measured torque among all the included calibration trials. Therefore, the calibrated Pooled NMS may be too generic to satisfy all movement patterns. Practical implications of these findings are that, in some applications, creating and training individual NMS models for each movement may be impractical; clustering movements can contribute to a more practical workflow and is preferable to an approach that pools all movements.

![](images/797831aad1be008d88b6e7bfa02d442119104cfae0e41dfe0d32088bfd4ddc32.jpg)

Fig. 8. One example trial of measured and predicted torque during ten movements via three LSTM models: Pooled, Individual, and Clustered.  
![](images/41ea66cb9d5bd06eeff5830fb5805c942cdf766f318ce605d3afb91325ea2d8d.jpg)  
Fig. 9. The NRMSE distributions between predicted and actual torques during ten motions among participants in NMS and LSTM Clustered models, shown with violin plots. Each violin plot includes a kernel density plot with a box plot. Kernel density plots illustrate the NRMSE probability distributions. Box plots denote NRMSE’s minimum, lower quartile, median, upper quartile, and maximum values. ∗ indicated that there was a significant difference between two models, determined by pairwise comparisons.

In contrast, Individual and Pooled LSTM models already had very high torque prediction accuracy, and a cluster approach was generally neither more nor less accurate. All LSTM models were evaluated using the unseen, “new” data from the same type of motions that were employed to train models. Generally, when the same type of data is used for training and evaluating the model, deep learning is typically expected to demonstrate great prediction performance. The designed LSTM network, which has multiple neurons and layers, is able to selectively remember patterns that correspond to a time interval; and it can identify the sophisticated nonlinear associations between EMG data and joint torques. Therefore, regardless of whether the network is formed by a Pooled, Clustered, or Individual model, LSTM networks can accurately estimate joint torques in unseen, “new” data within the same type of motions.

It is important to note that the prediction error of Individual and Clustered NMS models in hip F/E and hip Ab/Ad torque was generally higher than in knee F/E and ankle D/P. One reason for this could be attributable to the difficulty in experimentally acquiring surface EMGs on the layered and deep muscles that surround the hip joint. This finding agrees with Pizzolato et al. [25] who attributed the lower prediction accuracy and underestimation of sagittal hip moments compared to knee and ankle moments to the difficulty of obtaining reliable surface EMGs on the iliopsoas. It is also worth noting that, while inverse dynamics is generally considered a gold standard in computing joint torques, it may not be entirely accurate either, particularly at more proximal joints. Estimation of any joint center is critical in the inverse dynamics approach, as it is used to compute the moment arm of the muscles and serves as the point about which the joint rotation occurs. However, accurately estimating the hip joint center is generally considered more challenging than estimating the knee and ankle joint centers, as the hip is deeper and more difficult to characterize with biomechanical models. It is also important to note that during stand-to-sit, outliers with higher prediction error were observed in the Individual models in hip F/E and knee F/E torque, compared to Pooled and Clustered models. This may be attributable to overfitting in the Individual calibrated models, wherein a slight input difference would result in a large output difference. Therefore, when using individual models, caution for overfitting is warranted.

The torque predictions by LSTM models were closer to the values calculated through inverse dynamics than that of NMS models in all movement tests. In LSTM networks, each node is viewed as an artificial neuron, simulating the human neural synapse system in the brain. This gives LSTM networks a great ability to comprehend complicated and non-linear information. However, it learns the correlations between inputs and outputs as a black box and does not explicitly model the relationships between physiological variables, for instance, joint angles-musculotendon kinematics, muscle force-length, and muscle force-velocity relationships. Therefore, only the designated outputs are appropriate to estimate. Although the NMS model had a lower prediction accuracy than the LSTM model, it analyzes and identifies participant-specific muscletendon parameters and relationships. As such, for applications in which participant-specific muscle-tendon parameters and relationships are desired, an NMS model-based solution is clearly more suitable. A model that combines relationships defined from NMS models with neural network structures may have the advantages of higher prediction accuracy than either of its derivatives, in addition to the benefit of defined muscletendon properties and relationships [7], [33], [34]. Therefore, in future work, it would be worthwhile to explore the potential of a physics-informed neural network approach that combines NMS features with an LSTM model to enhance the neural network method’s extrapolation capability.

As noted above, obtaining ground truth values of joint torques is not trivial. The joint torque estimated through inverse dynamics is not a direct measure of the actual joint torque occurring in the body. We implemented the inverse dynamics approach by solving the dynamic equation of motion (Eq. (2)) with kinematics. The kinematics (joint angles) were obtained through the inverse kinematics approach by solving a weighted least square problem to minimize the error between modeled and measured marker trajectories. This approach may lead to dynamic inconsistencies due to measurement and modeling errors, resulting in joint moments that do not follow muscle dynamics [35], [36]. Despite these limitations, the joint torque calculated by the inverse dynamics approach is currently the closest estimation to the real joint torque during dynamic tasks. It is also worth noting that model training processes in the NMS model are not directly comparable to those of the LSTM model, particularly in that the Pooled NMS model is only trained with one trial from each movement whereas the LSTM is trained with 80% of the data. Some of the prediction errors of the Pooled NMS models may be attributed to variability in EMG and kinetic measurements. Likewise, its performance improvement in Individualized models may be attributable to the varied training data.

While we have not evaluated the real-time computational cost of the LSTM network in this study, it is theoretically capable of predicting joint torque in real-time with low computational cost as it only requires forward evaluation [37]–[39]. It is important to note that for individual participants, there should be only one calibrated NMS model that covers the large ranges of force-length curves as much as possible. Ideally, a subset of movements that covers the full range would be identified, but our main focus was not on achieving this. We included all movements for calibration, which should work similarly to a subset, as seen in the Pooled model. However, we observed unsatisfactory performance, which may be due to the exclusion of other experimental measurements such as fiber length and pennation angle, that are important to calibrate muscle tendon parameters. Measuring them during dynamic movements is challenging, and the Pooled model may not be suitable to capture all muscle coordination patterns. Our primary goal was to obtain a highly accurate model for predicting joint torque, and to achieve this, we explored various mathematical approaches, including calibrating different models separately, clustering them, and pooling them. We also compared these approaches with LSTM models. Although this may not align with the physical perspective, it is a practical way to obtain a better-trained model from a mathematical perspective.

While our aim was to obtain a model that could predict joint torque with high accuracy, we have not yet implemented these models in practical settings, specifically in exoskeleton control. Further steps are required, such as selecting appropriate methods/sensors for capturing EMG and kinematic data, training models based on the application requirements, and optimizing the model for real-time processing. The developed model would then need to be integrated into the exoskeleton’s control system to provide assist-as-needed support for joint movements. Our study provides guidelines for selecting the appropriate model in different scenarios, which has the potential to improve the control strategies for exoskeletons and enhance the functionality of wearable exoskeletons.

We only investigated one NMS and one machine learning model approach, i.e. LSTM here, and our findings should be considered in this context. Other NMS models and even different parameters and levels of sophistication in the currentlyused NMS, such as electromechanical delay and tendon types, warrant investigation. There are also other different machine learning models with complicated architectures, such as convolutions neural networks, that could be evaluated. Aspects of the experimental protocol may influence the generalizability of the findings, such as adjusting the chair height in relation to each participant’s stature for tests involving sitting. Also, even though we used standardized instructions for all participants, there may have been subtle differences among participants in movement techniques for, e.g. squat and jump movements, and thus possible differences in muscle activation patterns. Additionally, this investigation exclusively involves able-bodied individuals, whereas the primary focus of interest, which is exoskeleton users, primarily comprises populations with motor disorders. Furthermore, the able-bodied participants were generally homogeneous in terms of age and overall physical condition.

## V. CONCLUSION

In this study, we predicted ankle, knee, and hip joint sagittal plane torques and hip joint frontal plane torques during several different motions using physics-based NMS models and LSTM neural networks with a cluster approach. The cluster approach did not improve the prediction accuracy in LSTM neural networks, which was already high, but did improve it in NMS Pooled models in some movements. Among NMS models, Individual and Clustered models had similar prediction performance but the Pooled model may be too generic to satisfy all movement patterns. In applications where individual NMS models for each movement are impractical, a clustered movement approach may be a good alternative to pooled movement models. In general, detailed comparative performance of physics-based neuromusculoskeletal and neural networks models to predict torques in multiple lower limb DOFs and in a range of movements can provide useful guidelines for paradigms on the preferred choice of a prediction model in specific circumstances, and thus holds great promise for application in, for instance, assist-as-needed exoskeleton control, by taking into account the physiological joint torque contribution of its users.

## REFERENCES

[1] Z. Li, G. Li, X. Wu, Z. Kan, H. Su, and Y. Liu, “Asymmetric cooperation control of dual-arm exoskeletons using human collaborative manipulation models,” IEEE Transactions on Cybernetics, vol. 52, no. 11, pp. 12 126–12 139, 2021.

[2] D. Coyle, “Neural network based auto association and time-series prediction for biosignal processing in brain-computer interfaces,” IEEE Computational Intelligence Magazine, vol. 4, no. 4, pp. 47–59, 2009.

[3] P. B. Jephil, P. Acharaya, L. Xu, K. Guo, H. Yu, M. Watsford, S. Rong, and S. Su, “Estimation of ankle joint torque and angle based on semg signal for assistive rehabilitation robots,” in Biomedical Signal Processing. Springer, 2020, pp. 31–47.

[4] G. Durandau, D. Farina, G. Asín-Prieto, I. Dimbwadyo-Terrer, S. Lerma-Lara, J. L. Pons, J. C. Moreno, and M. Sartori, “Voluntary control of wearable robotic exoskeletons by patients with paresis via neuromechanical modeling,” Journal ofNeuroengineering and Rehabilitation, vol. 16, no. 1, p. 91, 2019.

[5] M. Sartori, D. G. Lloyd, and D. Farina, “Neural data-driven musculoskeletal modeling for personalized neurorehabilitation technologies,” IEEE Transactions on Biomedical Engineering, vol. 63, no. 5, pp. 879– 893, 2016.

[6] C. Pizzolato, D. J. Saxby, D. Palipana, L. E. Diamond, R. S. Barrett, Y. D. Teng, and D. G. Lloyd, “Neuromusculoskeletal modeling-based prostheses for recovery after spinal cord injury,” Frontiers in Neurorobotics, vol. 13, pp. 1–10, 2019.

[7] M. Ezati, B. Ghannadi, and J. McPhee, “A review of simulation methods for human movement dynamics with emphasis on gait,” Multibody System Dynamics, vol. 47, no. 3, pp. 265–292, 2019.

[8] L. Zhang, Z. Li, Y. Hu, C. Smith, E. M. G. Farewik, and R. Wang, “Ankle joint torque estimation using an emg-driven neuromusculoskeletal model and an artificial neural network model,” IEEE Transactions on Automation Science and Engineering, vol. 18, no. 2, pp. 564–573, 2020.

[9] Y.-F. Chen, R. Fu, J. Wu, J. Song, R. Ma, Y.-C. Jiang, and M. Zhang, “Continuous bimanual trajectory decoding of coordinated movement from eeg signals,” IEEE Journal of Biomedical and Health Informatics, vol. 26, no. 12, pp. 6012–6023, 2022.

[10] Y. Hu, G. Chen, Z. Li, and A. Knoll, “Robot policy improvement with natural evolution strategies for stable nonlinear dynamical system,” IEEE Transactions on Cybernetics, vol. 53, no. 6, pp. 4002–4014, 2022.

[11] L. Zhang, D. Soselia, R. Wang, and E. M. Gutierrez-Farewik, “Lowerlimb joint torque prediction using lstm neural networks and transfer learning,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 30, pp. 600–609, 2022.

[12] E. B. Simonsen and T. Alkjær, “The variability problem of normal human walking,” Medical Engineering & Physics, vol. 34, no. 2, pp. 219–224, 2012.

[13] A. Phinyomark, S. Osis, B. A. Hettinga, and R. Ferber, “Kinematic gait patterns in healthy runners: A hierarchical cluster analysis,” Journal of Biomechanics, vol. 48, no. 14, pp. 3897–3904, 2015.

[14] S. Kinsella and K. Moran, “Gait pattern categorization of stroke participants with equinus deformity of the foot,” Gait & Posture, vol. 27, no. 1, pp. 144–151, 2008.

[15] S. G. White and P. J. McNair, “Abdominal and erector spinae muscle activity during gait: the use of cluster analysis to identify patterns of activity,” Clinical Biomechanics, vol. 17, no. 3, pp. 177–184, 2002.

[16] H. B. Sigurðsson and K. Briem, “Cluster analysis successfully identifies clinically meaningful knee valgus moment patterns: frequency of early peaks reflects sex-specific acl injury incidence,” Journal of Experimental Orthopaedics, vol. 6, no. 1, pp. 1–12, 2019.

[17] C. Pizzolato, M. Reggiani, L. Modenese, and D. Lloyd, “Real-time inverse kinematics and inverse dynamics for lower limb applications using opensim,” Computer Methods in Biomechanics and Biomedical Engineering, vol. 20, no. 4, pp. 436–445, 2017.

[18] H. X. Hoang, C. Pizzolato, L. E. Diamond, and D. G. Lloyd, “Subjectspecific calibration of neuromuscular parameters enables neuromusculoskeletal models to estimate physiologically plausible hip joint contact forces in healthy adults,” Journal ofBiomechanics, vol. 80, pp. 111–120, 2018.

[19] F. Leboeuf, R. Baker, A. Barré, J. Reay, R. Jones, and M. Sangeux, “The conventional gait model, an open-source implementation that reproduces the past but prepares for the future,” Gait & Posture, vol. 69, pp. 235– 241, 2019.

[20] T.-W. Lu and J. O’connor, “Bone position estimation from skin marker co-ordinates using global optimisation with joint constraints,” Journal of Biomechanics, vol. 32, no. 2, pp. 129–134, 1999.

[21] M. G. Pandy, “Computer modeling and simulation of human movement,” Annual Review of Biomedical Engineering, vol. 3, no. 1, pp. 245–273, 2001.

[22] D. A. Winter, H. G. Sidwall, and D. A. Hobson, “Measurement and reduction of noise in kinematics of locomotion,” Journal ofBiomechanics, vol. 7, no. 2, pp. 157–159, 1974.

[23] A. Mantoan, C. Pizzolato, M. Sartori, Z. Sawacha, C. Cobelli, and M. Reggiani, “Motonms: A matlab toolbox to process motion data for neuromusculoskeletal modeling and simulation,” Source Code for Biology and Medicine, vol. 10, no. 1, pp. 1–14, 2015.

[24] T. R. Derrick, A. J. van den Bogert, A. Cereatti, R. Dumas, S. Fantozzi, and A. Leardini, “Isb recommendations on the reporting of intersegmental forces and moments during human motion analysis,” Journal of Biomechanics, vol. 99, pp. 1–10, 2020.

[25] C. Pizzolato, D. G. Lloyd, M. Sartori, E. Ceseracciu, T. F. Besier, B. J. Fregly, and M. Reggiani, “Ceinms: A toolbox to investigate the influence of different neural control solutions on the prediction of muscle excitation and joint moments during dynamic motor tasks,” Journal of Biomechanics, vol. 48, no. 14, pp. 3929–3936, 2015.

[26] M. Sartori, M. Reggiani, D. G. Lloyd, and E. Pagello, “A neuromusculoskeletal model of the human lower limb: Towards emg-driven actuation of multiple joints in powered orthoses,” in 2011 IEEE International Conference on Rehabilitation Robotics. IEEE, 2011, pp. 1–6.

[27] D. G. Lloyd and T. F. Besier, “An emg-driven musculoskeletal model to estimate muscle forces and knee joint moments in vivo,” Journal of Biomechanics, vol. 36, no. 6, pp. 765–776, 2003.

[28] T. S. Buchanan, D. G. Lloyd, K. Manal, and T. F. Besier, “Neuromusculoskeletal modeling: estimation of muscle forces and joint moments and movements from measurements of neural command,” Journal ofApplied Biomechanics, vol. 20, no. 4, pp. 367–395, 2004.

[29] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, “Dropout: a simple way to prevent neural networks from overfitting,” The Journal of Machine Learning Research, vol. 15, no. 1, pp. 1929–1958, 2014.

[30] J. Bergstra and Y. Bengio, “Random search for hyper-parameter optimization.” Journal of machine learning research, vol. 13, no. 2, 2012.

[31] L. v. d. Maaten and G. Hinton, “Visualizing data using t-sne,” Journal of Machine Learning Research, vol. 9, no. Nov, pp. 2579–2605, 2008.

[32] L. Morissette and S. Chartier, “The k-means clustering technique: General considerations and implementation in mathematica,” Tutorials in Quantitative Methods for Psychology, vol. 9, no. 1, pp. 15–24, 2013.

![](images/b09d1437060f4ee7298a9516da65a93083d420b9b45c1357e369085b372c2ee4.jpg)

![](images/53eb34e27225b48cd88d1fe02c5a5fd1b2fd3220da3b4f9bfcd424742176a847.jpg)

[33] L. Zhang, X. Zhu, E. M. Gutierrez-Farewik, and R. Wang, “Ankle joint torque prediction using an nms solver informed-ann model and transfer learning,” IEEE Journal of Biomedical and Health Informatics, vol. 26, no. 12, pp. 5895–5906, 2022.

[34] D. J. Saxby, B. A. Killen, C. Pizzolato, C. Carty, L. Diamond, L. Modenese, J. Fernandez, G. Davico, M. Barzan, G. Lenton et al., “Machine learning methods to support personalized neuromusculoskeletal modelling,” Biomechanics and Modeling in Mechanobiology, vol. 19, no. 4, pp. 1169–1185, 2020.

[35] A. J. Van Den Bogert and A. Su, “A weighted least squares method for inverse dynamic analysis,” Computer Methods in Biomechanics and Biomedical Engineering, vol. 11, no. 1, pp. 3–9, 2008.

[36] A. D. Kuo, “A least-squares estimation approach to improving the precision of inverse dynamics computations,” Journal of Biomechanical Engineering, pp. 148–159, 1998.

[37] S. Sharma and S. Sen, “Real-time structural damage assessment using lstm networks: regression and classification approaches,” Neural Computing and Applications, vol. 35, no. 1, pp. 557–572, 2023.

[38] Y. Du, Y. Yan, S. Chen, and Y. Hua, “Object-adaptive lstm network for real-time visual tracking with adversarial data augmentation,” Neurocomputing, vol. 384, pp. 67–83, 2020.

## APPENDIX

## A. Trials for NMS Calibration

Some may raise concerns about the accuracy of the pooled NMS model calibration, as using only one trial might be insufficient. However, the NMS model is based on physics and utilizes muscle-tendon parameters and activation patterns specific to each movement type. Therefore, for a given movement type, the muscle is usually activated at a similar range of force-length relationship. Based on the physical structure of the NMS model, calibrating it with one trial is adequate. We also attempted to calibrate the NMS model using two trials from each movement type, but found that this did not result in a significant difference in prediction accuracy (Fig. A.1), while calibration time was considerably longer. Hence, considering the practical calibration time and the physical structure of the NMS model, we did not include multiple trials from the same movement type in Pooled models for calibration.

[39] H. Gao, C. Zhang, S. Pei, and X. Wu, “Lstm-based real-time signal quality assessment for blood volume pulse analysis,” Biomedical Optics Express, vol. 14, no. 3, pp. 1119–1136, 2023.

![](images/9dfafad4c63916d2395cb06b2e70a964b35d3399829ef1efd531a13759c5bb14.jpg)

![](images/7848b7a871c3a98b9b132794cdb08b47ae2605ea0945c98392aea8eb2d2ad2ba.jpg)  
Fig. A.1. The NRMSE distributions between predicted and actual torques during ten motions among participants in NMS Pooled models with as a comparison of one vs. two trials from each movement for calibration, shown with violin plots. Each violin plot includes a kernel density plot with a box plot. Kernel density plots illustrate the NRMSE probability distributions. Box plots denote NRMSE’s minimum, lower quartile, median, upper quartile, and maximum values. ∗ indicated that there was a significant difference between two models, determined by pairwise comparisons.