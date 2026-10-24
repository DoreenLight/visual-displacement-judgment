# An Adaptive Hammerstein Model for FES-Induced Torque Prediction Based on Variable Forgetting Factor Recursive Least Squares Algorithm

Qinlian Yang , Yingqi Li, You Li, Manxu Zheng , and Rong Song , Senior Member, IEEE

Abstract— Modeling the muscle response to functional electrical stimulation (FES) is an important step during model-based FES control system design. The Hammerstein structure is widely used in simulating this nonlinear biomechanical response. However, a fixed relationship cannot cope well with the time-varying property of muscles and muscle fatigue. In this paper, we proposed an adaptive Hammerstein model to predict ankle joint torque induced by electrical stimulation, which used variable forgetting factor recursive least squares (VFFRLS) method to update the model parameters. To validate the proposed model, ten healthy individuals were recruited for short-duration FES experiments, ten for long-duration FES experiments, and three stroke patients for both. The isometric ankle dorsiflexion torque induced by FES was measured, and then the test performance of the fixed-parameter Hammerstein model, the adaptive Hammerstein model based on fixed forgetting factor recursive least squares (FFFRLS) and the adaptive Hammerstein model based on VFFRLS was compared. The goodness of fit, root mean square error, peak error and success rate were applied to evaluate the accuracy and stability of the model. The results indicate a significant improvement in both the accuracy and stability of the proposed adaptive model compared to the fixedparameter model and the adaptive model based on FFFRLS. The proposed adaptive model enhances the ability of the model to cope with muscle changes.

Index Terms— Functional electrical stimulation, muscle model, system identification, Hammerstein model.

## I. INTRODUCTION

technique for neurologically impaired individuals to compensate or restore the lost motor function [1], [2]. In FES intervention, electrical currents are delivered to the excitable motor neurons of paralyzed muscle, in place of the central nervous system to induce muscle contractions, thereby generating force and torque [3]. Since FES can artificially induce muscle contraction, it has been widely utilized in the field of rehabilitation for various purposes, including the treatment of foot drop [4] and suppressing tremors [5], as well as assisting with cycling [6] and sit to stand [7] activities.

Due to the benefits of appropriate and adjustable FES in rehabilitation, extensive research has been dedicated to improving the control strategies of electrical stimulators based on closed-loop principles. Some studies have focused on modulating the electrical stimulation depending on patients’ motion performance, which utilize sensors to capture torque/angle signals as feedback and employ model-free methods, such as iterative learning control (ILC), to adjust the stimulation input [4], [8]. Model-based FES control has also been investigated, which aims to establish a mapping relationship between FES and torque/angle, so that the modulation can be performed based on the predicted output [9], [10]. In modelbased FES systems, a convenient model which can simulate the skeletal muscle dynamics precisely under FES is an important component. A common type is the physiological model such as the Hill model and Riener’s model, using different elements to simulate muscle contraction [11], [12]. Although efforts had been made in these models to accurately describe the actual muscle response, they were rarely applied in the FES control systems for the difficulty of obtaining the real values of the physiological parameters [13]. Another type of model is the empirical model. These models ignore the physiological structure of muscle but focus on their accuracy and feasibility in predicting outputs. A simple linear model was used to represent the response of electrically stimulated muscle in FES systems [14]. A nonlinear auto-regressive model with exogenous inputs-recurrent neural network (NARX-RNN) model was also proposed [15]. In addition, the Hammerstein model, including a static nonlinear function followed by a linear dynamic function, was also suggested for the prediction of FES-induced muscular dynamics [16]. According to Hunt’s study, the Hammerstein model can simulate the response of muscle to FES at different electrical stimulation intensities ideally under isometric contraction [17].

Since the two blocks of the Hammerstein model correspond to the recruitment of nerve fibers and the subsequent dynamics of muscle contraction [17], the model has been widely used for modeling biomechanical systems. In the past, most studies adopted the Hammerstein model with fixed parameters to predict the force and torque under FES [18], [19], where the model parameters did not change once the identification was finished. However, the time-varying property of muscles and the occurrence of muscle fatigue may lead to a decrease in the accuracy of models with fixed parameters. Therefore, adaptive methods were proposed. In Cai’s study [20], an adaptive Wiener-Hammerstein model was utilized to establish the relationship between the FES input and muscle force, which implemented an iterative relationship for model parameters adaption. In Zhang’s study [21], to describe the relationship between FES evoked-electromyography (eEMG) and ankle torque, a Kalman filter with a fixed forgetting factor was used to identify the Hammerstein model and adapt the parameters online. However, the former lacks forgetting factors, which may result in an excessive weighting of historical data, leading to a decline in the tracking performance of the model. In the meanwhile, the latter faces the challenge in selecting a suitable forgetting factor, as an inappropriate value may adversely affect the accuracy of the model and even lead to modelling failure [15], [22]. Therefore, it has motivated us to explore the adaptive Hammerstein model with a variable forgetting factor, as it may enhance the model’s adaptation ability to cope with the time-varying muscle property. However, to the best of our knowledge, this aspect has not been reported. In this paper, we aim to develop an improved approach for simulating ankle torque based on electrical stimulation amplitude. To achieve this, an adaptive Hammerstein model that utilized variable forgetting factor recursive least squares (VFFRLS) to update the parameters of the model based on the identification results of neural network was proposed. To validate the effectiveness of this model, we have recruited ten healthy individuals for short-duration FES experiments, ten for long-duration FES experiments, and three stroke patients for both short- and longduration experiments.

## II. MODELS

## A. Hammerstein Model Based on Neural Network

The discrete-time Hammerstein model, which consists of a static nonlinear block in series with a linear dynamic block, can be utilized to simulate muscles under FES. In this study, the stimulation to torque model is described as

![](images/d9f9fcdfeea9b1980fb567ea4ad2c265b37d1f5d10094f38d9973d644e980a6f.jpg)  
Fig. 1. Structure of the Hammerstein model based on neural network.

$$
\tau (k) = G \left(q ^ {- 1}\right) w (k) = \frac {B (q ^ {- 1})}{A (q ^ {- 1})} f (u (k))\tag{1}
$$

where model input u(k) is the electrical stimulation amplitude and the output τ(k) is the ankle joint torque. The static nonlinear function $f ( u ( k ) )$ maps $u ( k )$ to the internal unmeasurable variable $w ( k )$ , which stands for the relationship between electrical stimulation amplitude and muscle activation level. Then, $w ( k )$ is passed to a linear dynamic block described with the transfer function $G ( q ^ { - 1 } )$ . The linear dynamic block represents muscle contraction dynamics. The expressions of $B \left( q ^ { - 1 } \right)$ and $A \left( q ^ { - 1 } \right)$ can be shown as

$$
B \left(q ^ {- 1}\right) = b _ {0} q ^ {- d} + b _ {1} q ^ {- (d + 1)} + \dots + b _ {n _ {b}} q ^ {- (n _ {b} + d)}\tag{2}
$$

and

$$
A \left(q ^ {- 1}\right) = 1 + a _ {1} q ^ {- 1} + \dots + a _ {n _ {a}} q ^ {- n _ {a}}\tag{3}
$$

where $q ^ { - 1 }$ is the delay operator, and d, $n _ { a }$ and $n _ { b }$ are the time delay order, the number of poles and zeros in this dynamic transfer function, respectively. The values of $( n _ { a } , n _ { b } , d )$ in this paper are based on the work in [17] which are set to (2,1,1). The coefficients of the linear function are described in the vector $[ a _ { 1 } , a _ { 2 } \ldots , a _ { n _ { a } } , b _ { 0 } , b _ { 1 } , \ldots , b _ { n _ { b } } ] ^ { T }$

Hammerstein based on neural network structure is employed in this study due to its better nonlinear expressivity compared to polynomials [23], [24]. The use of it transforms the optimization of model parameters into the training of neural network. The type of neural network adopted here is multi-layer perceptron, which consists of two parts. Fig. 1 has shown the model structure. The first part is the static nonlinear block, which mainly consists of a single hidden layer feed-forward neural network. Neural network neurons in static nonlinear part are S-type neurons with activation function $g \left( k \right) = 2 / ( 1 + e ^ { - 2 k } ) - 1$ , which overcomes the shortcomings of non-zero mean output in the sigmoid activation function. M represents the number of static nonlinear hidden layer neurons, and was chosen as 10 because it achieved the highest accuracy among the tested values of 5, 10, and 15 in the pre-experiments. A single layer neural network with feedback hysteresis constitutes the second part of the Hammerstein model, the dynamic linear block. The expression for the internal unmeasurable variable w(k) can be obtained based on the connectivity of the neural network, shown as

$$
w (k) = \sum_ {m = 1} ^ {M} \beta_ {m} g (\omega_ {m} \cdot u (k) + \mu_ {m}).\tag{4}
$$

Substituting the (4) into the (1), and then obtain the (5):

$$
\begin{array}{l} \tau (k) = - \sum_ {i = 1} ^ {n} _ {a} a _ {i} \tau (k - i) \\ \qquad + \sum_ {j = 0} ^ {n} _ {b} \sum_ {m = 1} ^ {M} b _ {j} \beta_ {m} g (\omega_ {m} \cdot u (k - d - j) + \mu_ {m}) \end{array}\tag{5}
$$

All parameters that need to be determined in the model include the following: $\omega _ { m } , \ b _ { m } , \ a _ { i }$ , and $b _ { j }$ represent the connection weight of each layer. $\mu _ { m }$ represents the threshold of each layer. The model parameters mentioned above are obtained by training the neural network with experimental data using the Levenberg-Marquardt algorithm.

## B. Dynamic Linear Block Parameter Updating Based on VFFRLS

Considering the time-varying property of muscles is unknown in advance and subject-specific, and muscle fatigue may occur during FES interventions, it’s necessary to make the model adaptive. Hence, data from earlier times need to be discarded to track muscle changes in response to FES. In this process, the identification methods with a forgetting factor are usually adopted [15], [21]. Anyhow, it is difficult to select an optimal fixed forgetting factor [15], [22]. In order to avoid this problem, this study allows the system to use the VFFRLS method to update the dynamic linear block parameters based on the identification results of neural network. The VFFRLS algorithm maintains the simplicity of recursive least squares (RLS) while introducing adaptive forgetting factor adjustments to achieve dynamic data weighting. In RLS, define $\Phi _ { k } ~ = { }$ $\left[ \begin{array} { l } { \Phi _ { k - 1 } } \\ { \varphi ( k ) ^ { T } } \end{array} \right] , \mathbf { Y _ { k } } = \left[ \begin{array} { l } { \mathbf { Y } _ { k - 1 } } \\ { \boldsymbol { \tau } ( k ) } \end{array} \right]$ , (5) can be written as follows:

$$
\tau (k) = \varphi^ {T} (k) \hat {\theta} (k) + e (k)\tag{6}
$$

where $\begin{array} { r } { \varphi ( k ) \qquad = \qquad [ - \tau ( k \quad - \quad 1 ) , \ldots , - \tau ( k \quad - \quad n _ { a } ) } \end{array}$ $w ( k \textrm { -- } d ) , \ldots , \quad w ( k \textrm { -- } d \textrm { -- } n _ { b } ) ] ^ { T } \in R ^ { ( n _ { a } + n _ { b } + 1 ) \times 1 } , \theta \quad = \quad$ $[ a _ { 1 } , \dots , a _ { n _ { a } } , b _ { 0 } , \dots , b _ { n _ { b } } ] ^ { T } \in R ^ { ( n _ { a } + n _ { b } + 1 ) \times 1 }$ , and $\boldsymbol { \mathscr { e } } \left( \boldsymbol { k } \right)$ indicates the error. The recursive form of $\hat { \theta } \left( k \right)$ can be derived as follows [25]:

$$
\hat {\theta} (k) = \hat {\theta} (k - 1) + K (k) e (k)\tag{7}
$$

where K(k) is defined as the gain matrix, $K \left( k \right) = P \left( k \right) \varphi \left( k \right)$ with P(k) being defined as the prediction error covariance matrix, $P \left( k \right) = \mathbf { \bar { \Phi } } { \left( \Phi _ { k } ^ { T } \Phi _ { k } \right) } ^ { - 1 }$

To account for continuous and gradual changes in system parameters, it is important to reduce the impact of earlier data on identification results and emphasize the sensitivity to recent estimation errors. Therefore, $\boldsymbol { \mathscr { e } } \left( \boldsymbol { k } \right)$ is modified by introducing a forgetting factor, and the squared error can be described as

$$
J = \sum_ {k = 1} ^ {L} \lambda^ {L - k} \left[ \tau (k) - \varphi^ {T} (k) \hat {\theta} (k) \right] ^ {2}\tag{8}
$$

![](images/9d6ef1b39abe041af5450b1ec9455820412c260e925a586d54b44e5d66f35c23.jpg)  
Fig. 2. Forgetting factor λ  k curve with error e(k).

where L is the number of observations, and λ is the forgetting factor. The recursive formulas are deduced as [25]

$$
\hat {\theta} (k) = \hat {\theta} (k - 1) + K (k) (\tau (k) - \varphi^ {T} (k) \hat {\theta} (k - 1))\tag{9}
$$

$$
K (k) = \frac {P (k - 1) \varphi (k)}{\lambda + \varphi^ {T} (k) P (k - 1) \varphi (k)}
$$

$$
P (k) = \frac {(I - K (k) \varphi^ {T} (k)) P (k - 1)}{\lambda}\tag{10}
$$

(11)

The variable forgetting factor can be designed as the following smooth curve [26]:

$$
\lambda (k) = \lambda_ {m i n} + (1 - \lambda_ {m i n}) ^ {\alpha (k)}
$$

$$
\alpha (k) = 2 ^ {\rho e ^ {2} (k)}\tag{12}
$$

(13)

where $\lambda _ { m i n }$ and $\rho$ are fixed parameters which can flexibly control the shape and degree of variation of the $\lambda \left( k \right)$ and have relatively low requirements for parameter setting. Set $\lambda _ { m i n } =$ $0 . 9$ and $\rho = 1 0 ^ { 5 }$ . This is because the typical range of value for forgetting factor is [0.9, 1] [21], and setting the lower limit of 0.9 ensures that the system maintains relative attention to historical data to ensure system stability and reduce sensitivity to noise. Fig.2 shows the curve of $\lambda \left( k \right)$ varying with the error $\boldsymbol { \mathscr { e } } \left( \boldsymbol { k } \right)$ . The figure indicates that as e (k) approaches $0 ,$ $\lambda \left( k \right)$ approaches 1, and as $| e ( k ) |$ gradually increases, λ (k) approaches its minimum value.

## III. EXPERIMENTS

## A. Experimental Setup

The experimental setup is shown in Fig. 3. The apparatus mainly included a servo motor (DM1 B-045G, Yokogawa, Japan), a torque sensor (AKC-205, 701st Research Institute of China Aerospace, Science and Technology Corporation, China), a footplate, a data acquisition device (DAQ Multifunction NI USB-6341, National Instruments, USA), a customized functional electrical stimulator (P2-9632, Faisco, China), and a computer. The footplate could be set to a certain position by locking the servo motor. The torque sensor connecting with the footplate measured the torque of the interaction between human and footplate. For the functional electrical stimulator, the frequency and duration of electrical pulses could only be manipulated manually while the electrical stimulation amplitude ranging from 0 mA to 100 mA was controlled by pulse-width modulation (PWM) signal. The DAQ card generated standard PWM signal and delivered it to the electrical stimulator to regulate the amplitude of the electrical pulses. Besides, a LabVIEW-based program (National Instruments, USA) was applied to send instruction to DAQ to produce PWM signal and record the ankle torque signal.

Long-duration FES session

(b)

(a)  
![](images/fe17fbb0bfaac31aab74e353d41b0d222531170d0fa677658fded998b720dd70.jpg)

![](images/bf19b16c36bfff04bebdfaba63e8783a51474cdc1cfffc67acae75031b7933fc.jpg)  
Fig. 3. The experimental apparatus. (a) Schematic diagram of the experimental apparatus. (b) Photo of the experimental apparatus.

## B. Experimental Protocol

20 healthy subjects were recruited (aged: $2 4 \pm 1 . 2 3$ years; 10 females, 10 males) in this study and divided equally into two groups for long-duration and short-duration FES experiments, respectively. All of them were non-disabled young adults with no neurological or musculoskeletal disorders. Three stroke patients at the Department of Rehabilitation Medicine, Zhujiang Hospital of Southern Medical University, China, were also recruited for both short- and long-duration FES. The patient-related information is shown in Table I. The inclusion criteria for the stroke patients are: 1) stroke patients have hemiparesis caused by unilateral brain lesions; 2) stroke patients should be able to generate sufficient strength to maintain their sitting on a chair with a backrest and armrests; 3) stroke patients should have no significant restrictions in the passive motion range of their ankle joints; 3) stroke patients should have no impairment of visuospatial, cognitive, or attentional function that will prevent them from following instructions. The research was approved by the Ethics Committee of Zhujiang Hospital of Southern Medical University. Each subject gave written informed consent before taking part in this experiment.

Prior to the experiment, each subject was seated in a chair with their ankle (healthy subjects: right side; stroke patients: lesion side) fixed on the footplate at $9 0 ^ { \circ }$ synchronously, and the surface electrical stimulation electrodes (5 cm× 5 cm, M2223, 3M, USA) were placed on the tibialis anterior (TA) muscle of above sides in advance. The main function of the TA muscle is to make the ankle joint dorsiflex and produce dorsiflexion torque. Then, the electrical pulses were delivered to the surface electrodes at a constant duration (390 µs) and frequency (40 Hz). The amplitude of electrical pulses was gradually increased until the subjects reached their maximum tolerance or produced an ankle torque greater than 3 Nm. This process also allowed the subjects to become familiar with the electrical stimulation. Subjects were instructed to maintain relaxed throughout the experiment and provide no volitional effort.

TABLE I  
INFORMATION OF SUBJECTS WITH STROKE

<table><tr><td>Subject</td><td>Sex</td><td>Age</td><td>Months after stroke</td><td>Lesion side</td><td>FMA-LE</td></tr><tr><td>S1</td><td>F</td><td>34</td><td>6</td><td>Left</td><td>12/34</td></tr><tr><td>S2</td><td>M</td><td>40</td><td>6</td><td>Right</td><td>17/34</td></tr><tr><td>S3</td><td>F</td><td>65</td><td>2</td><td>Left</td><td>10/34</td></tr></table>

Abbreviations: F = Female; M = Male; FMA-LE = Fugl-Meyer Assessment for Lower Extremity.

![](images/b500d164e3cf51ac50862ac14f89a61ed3a5f0dc908b913fb1f18e4595cbc975.jpg)

![](images/b4b4e88a4f9b8785f11ca69cf5056624d745e74cb9f572bea7ea0ae2948e2031.jpg)

![](images/ee168ab17c871acf16522542f2b8d3bc6bfe227803cd495aa2be235dd08882cf.jpg)  
Fig. 4. The profile of the input electrical stimulation amplitude in (a) short-duration SP test; (b) short-duration TP test; (c) long-duration SP test; (d) long-duration TP test.

The formal experiment consisted of two test sessions, shortduration FES session and long-duration FES session. Each session contained two kinds of test, namely the sinusoidal profile (SP) test and the triangular profile (TP) test [27], [28]. Previously, muscle dynamic response modelling focused on analyzing the transient response of muscles to short input sequences, which could be influenced by voluntary muscle activation. Slowly varying stimulation signals are less likely to trigger an autonomic reflex such as the SP or TP stimulation adopted here [29], [30]. In short-duration session, the duration of electrical stimulation was short to avoid muscle fatigue. The two test signals are shown in the Fig. 4(a) and (b). In each test, the stimulation was repeated six trials on each subject. During the long-duration FES session, two tests were performed using the two electrical stimulation profiles described above. The electrical stimulation signals are shown in Fig. 4(c) and (d). Each signal lasted up to seven minutes and consisted of a sequence comprising seven electrical stimulation sequences, denoted as sequence0 to sequence6. Each subject underwent one trial for each profile. Between the two trials, subjects were given at least 48 hours of rest time and the electrodes placement was marked, to avoid the placement of the electrodes deviating significantly. In total, there were four types of electrical stimulation tests, namely short-duration SP test, short-duration TP test, long-duration SP test, and long-duration TP test. The measured ankle torque was collected on the DAQ card at a sample rate of 100Hz and finally saved and analyzed on the computer.

In this study, for each of the test, subject-specific modelling and evaluation were carried out.

In short-duration SP test, three trials of torque data from each subject were served as training set to identify the subjectspecific Hammerstein model based on neural network. For the remaining 3 trials, data from the first sinusoidal wave of each trial for each subject were used to perform parameter updates for dynamic linear block, and data from the remaining two sinusoidal waves were served as testing set.

In short-duration TP test, parameter updates were performed based on short-duration SP neural network identification results. Torque data from one trial in the short-duration TP test of each subject were used for parameter updating, and data from the other trials were served as test set.

In long-duration SP test, the data from sequence0 of each subject were used for training the subject-specific Hammerstein model. In the following sequences, data from the first 10 seconds of each sequence were used for updating model parameters, and the next 50 seconds were used for testing the performance of the model.

In long-duration TP test, data from sequence0 were also used for training the subject-specific Hammerstein model. Data from the first 20 seconds of each remaining sequence were used to update model parameters, and data from the next 40 seconds were used for model test.

In addition, fixed-parameter Hammerstein model and adaptive model based on fixed forgetting factor recursive least squares (FFFRLS) were also developed and the modeling results were compared with the adaptive Hammerstein model based on VFFRLS. Since the typical range of value for the fixed forgetting factor is [0.9, 1], it was set to 0.999 (high),0.95 (middle), and 0.9 (low) in this study. In shortduration sessions, in order to make the fixed-parameter model and the adaptive model witness the same range of data, data being used for parameter updates was also added to the training set of the fixed-parameter model.

## C. Data Analysis

The measured torque signals were passed through a 4th-order Butterworth low-pass filter with a cut-off frequency of 5 Hz. The MATLAB software (MATLAB R2018b,

MathWorks Inc., Natick, MA, USA) was applied during the procedure of data analysis.

In order to evaluate the performance of the models, goodness of fitness (GOF), root mean square error (RMSE) and peak error (PE) were proposed to use here, which could be calculated as

$$
G O F = 1 - \frac {\sum_ {k = 1} ^ {N} [ \tau_ {m} (k) - \tau_ {e s t} (k) ] ^ {2}}{\sum_ {k = 1} ^ {N} [ \tau_ {m} (k) - \bar {\tau} _ {m} (k) ] ^ {2}}\tag{14}
$$

$$
R M S E = \sqrt {\frac {\sum_ {k = 1} ^ {N} [ \tau_ {m} (k) - \tau_ {e s t} (k) ] ^ {2}}{N}}\tag{15}
$$

$$
P E = \max \left(\left| \tau_ {m} (k) - \tau_ {e s t} (k) \right|\right)\tag{16}
$$

where $\tau _ { m }$ is the actual measured output, $\bar { \tau } _ { m }$ is the average of $\tau _ { m }$ and $\tau _ { e s t }$ represents the output from simulated model. Results were examined to determine the proportion of successful model achieved by each method, defining modeling success when the GOF was above 0.7. Besides, the Wilcoxon signedrank test was conducted to examine if there were significant differences between the adaptive Hammerstein model based on VFFRLS, adaptive Hammerstein model based on FFFRLS and the fixed-parameter Hammerstein model which based on neural network only. Due to the limited number of stroke patient subjects, the significance analyses were only conducted on the results of healthy subjects. The statistical analyses were performed using SPSS 26.0 software (SPSS Inc., Chicago, IL, USA). The p value less than 0.05 was considered to be statistical significance.

## IV. RESULTS

## A. Short-Duration FES Session

Although sufficient rest was given to the subjects between each trial in the short-duration test, it is still necessary to update the model parameters. This is due to the time-varying property of muscles and the fact that the torque induced by FES may not be identical even under repeated electrical stimulation experimental conditions.

To evaluate the proposed method, the results of the fixedparameter model, the adaptive model based on FFFRLS, and the adaptive model based on VFFRLS were compared. The modelling success rates for the ten healthy subjects and three stroke patients are shown in Table II. Since the success rates of the adaptive model based on middle FFFRLS and low FFFRLS were extremely low, the modelling results of these two methods were not included in the following. Fig. 5 shows the fitting performance of the models for one healthy subject and one stroke patient (S3) on different occasions. As could be seen from the figure, the prediction torque of the model with updated parameters was closer to the real measured torque than the model with fixed parameters. Besides, the torque estimated by the proposed model was closer to the measured torque than high VFFRLS adaptive model in TP test.

Fig. 6 shows the statistical results of the models. It could be found that the adaptive model based on VFFRLS performed better than the fixed-parameter model either in SP and TP test in both healthy individuals and stroke patients. As can be seen from the significance analysis of healthy individuals’ model

(a)  
![](images/70bb036129042d77cfe905a4e2759a8e09b67751aac2ac48e65169642807556d.jpg)  
(b)

![](images/f78e3ddcab13f7470b6507e32df800681955d0c0fa5566a1e1f232b49ad617b8.jpg)

(c)  
![](images/f2ccd048c95d4a78bdfae56f05a5c645d4b8f11b67e2f90303a7f30c19bfc679.jpg)  
(d)

![](images/b31e909ec5dd2cae0abd901588b19908e95dc26f87b7ef6284e0e8bc8fa2561a.jpg)

![](images/2911d801cbc06ec8a6cdd2185c5ccbd2991cb56ce27ebd856e21a0a3e9d944dd.jpg)  
Torque estimated by the fixed-parameter model Torque estimated by the high FFFRLS model Torque estimated by the VFFRLS model

Fig. 5. The fit performance of the fixed-parameter model and adaptive models based on high FFFRLS and VFFRLS for a healthy subject and a stroke patient (S3). Fitting curve of a healthy subject under shortduration (a) SP test; (c) TP test. Fitting curve of S3 under short-duration (b) SP test; (d) TP test.  
TABLE II  
PERCENTAGE OF MODELLING SUCCESS UNDER SHORT-DURATION FES SESSION

<table><tr><td>Subjects</td><td>Test</td><td>Fixed</td><td>High</td><td>Middle</td><td>Low</td><td>VFFRLS</td></tr><tr><td rowspan="2">Healthy</td><td>SP</td><td>93.3</td><td>93.3</td><td>13.3</td><td>10.0</td><td>93.3</td></tr><tr><td>TP</td><td>94.0</td><td>56.0</td><td>10.0</td><td>8.0</td><td>98.0</td></tr><tr><td rowspan="2">Stroke</td><td>SP</td><td>88.9</td><td>88.9</td><td>0.1</td><td>0.0</td><td>100.0</td></tr><tr><td>TP</td><td>46.7</td><td>60.0</td><td>0.0</td><td>0.0</td><td>86.7</td></tr></table>

Abbreviations: Fixed = Fixed-parameter model; High = High FFFRLS model; Middle = Middle FFFRLS model; Low = Low FFFRLS model

results, there were highly significant differences between the proposed adaptive and the fixed model in TP test (p<0.01). Also, the GOF and RMSE results of proposed adaptive and fixed model had significant differences in SP test (p<0.05). Besides, the GOF and RMSE results of the proposed adaptive model and adaptive model based on high FFFRLS had significant differences in SP test and highly significant differences in TP test. However, the results of the adaptive model based on high FFFRLS did not show significant improvement compared to the fixed-parameter model. It is worth noting that the patients’ model results did not undergo significance analysis due to the limited sample size. However, the boxplot indicated a clear trend in the model accuracy comparison, with the VFFRLS model consistently exhibiting higher accuracy than the high FFFRLS model, which, in turn, outperformed the fixed-parameter model, especially in the TP test.

![](images/305f710ece10aa2c7f7324b03c089f83203d4bdbba4204c8449bc1b23edee166.jpg)

![](images/1e7eb5d89ceff9232a97f4ee1d8294e77220581d859875aeb6dde25f5316dd30.jpg)  
(d)

![](images/f22714d43f58b374db5b3b84aabf538438019c7fec8101c381e1496ba27c6f08.jpg)  
(e)

![](images/eac179b0eaeac4ef9c8e9d38b967c7710a0a31c142f0d0a87364ce2cc454ecb9.jpg)  
(f)

![](images/b9e68d8e7270a49cf0a9c3208acd96dfd4bdcfc54363ef36f09093970f2a85a9.jpg)

![](images/391991339fa45dd405c158304660c44c506fcb7e8e46185d9a56a7377d1091a0.jpg)  
Fig. 6. The statistical results of healthy subjects and stroke subjects. <sup>∗</sup> represents the significant difference level is at p<0.05, <sup>∗∗</sup> represents the significant difference level is at p<0.01, and <sup>∗∗∗</sup> represents the significant difference level is at p<0.001.

## B. Long-Duration FES Session

To validate the proposed adaptive Hammerstein model in long-duration FES, two 7-minutes tests were conducted on subjects. However, S1 received only two sets of electrical stimulation of about 6 minutes each due to the limited tolerance and S3 received only SP electrical stimulation due to scheduling constraints. The results of all subjects were analyzed. In long-duration SP test, except for two healthy subjects, the rest of the subjects showed a decreasing trend in torque as the experiment progressed, with an average decrease in torque to 87.4% for healthy subjects and 78.1% for stroke subjects of its original value. In long-duration TP sessions, excluding three healthy subjects, the average torque of the remaining healthy subjects decreased to 90.3% of its original value and the average torque of stroke patients decreased to 69.3% of its original value. The torque of ankle dorsiflexion gradually declined, which was consistent with the results of fatigue inducing experiments in the literature [21]. Therefore, it could be inferred that under the electrical stimulation induced in this experiment, most subjects have experienced muscle fatigue.

![](images/28f41cda3465027c963e8c8e7ab04f79d8d2c32c2069a55888e625d8f4c472ba.jpg)  
Measured torque — · – Torque estimated by the fixed-parameter model – - – Torque estimated by the high FFFRLS model – - -Torque estimated by the VFFRLS mode  
Fig. 7. The fit performance of the models for a healthy subject and a patient (S2). Fitting curve of a healthy subject under long-duration (a) $\mathtt { S P }$ test; (c) TP test. Fitting curve of S2 under long-duration (b) SP test; (d) TP test.

The model success rates for the healthy and stroke subjects are shown in Table III. It could be observed that the success rates of the VFFRLS model surpassed that of the other models. Since the success rates of the adaptive model based on middle FFFRLS and low FFFRLS were extremely low, these two models were omitted from the subsequent analysis. Fig. 7 shows the fitting performance of the models in predicting ankle torque for a healthy subject and a stroke patient (S2) under experimental conditions involving SP and TP FES of the TA muscles. To enhance visual clarity, the figure only included data from three specific time periods: sequence1, sequence4, and sequence6 and other sequences had been omitted from the figure. From the figure, it was evident that the torque generated by the muscle exhibited variability, even with the same electrical stimulation, and usually decreased as the time of electrical stimulation increased. However, the proposed model demonstrated good torque prediction capabilities, while the other two models might experience an increasing prediction error with prolonged electrical stimulation.

The model test statistical results of the models for the ten healthy subjects and two stroke patients are shown in Fig. 8. During the experimental period, the adaptive model based on VFFRLS consistently maintained a high level of accuracy while the other two models’ accuracy declined apparently, especially in patients’ experiments. According to the statistical results of healthy individuals, as the electrical stimulation continued, the prediction results of the VFFRLS model performed better than the fixed and high FFFRLS model. In SP test, from sequence 4 to 6, the results of the VFFRLS adaptive model showed a significant difference from the results of the fixedparameter model and high FFFRLS model $_ { ( \mathrm { p < 0 . 0 5 } ) }$ . In TP test, the GOF and RMSE results of the VFFRLS adaptive model showed a significant improvement from the results of the fixed-parameter model and high FFFRLS model in sequence 4 to 6 (p<0.05). However, the high FFFRLS model did not show significant improvement compared to the fixedparameter model.

TABLE III  
PERCENTAGE OF MODELLING SUCCESS UNDERLONG-DURATION FES SESSION

<table><tr><td>Subjects</td><td>Test</td><td>Fixed</td><td>High</td><td>Middle</td><td>Low</td><td>VFFRLS</td></tr><tr><td rowspan="4">Healthy Stroke</td><td>SP</td><td>86.7</td><td>85.0</td><td>28.3</td><td>16.7</td><td>95.0</td></tr><tr><td>TP</td><td>81.7</td><td>80.0</td><td>16.7</td><td>16.7</td><td>96.7</td></tr><tr><td>SP</td><td>64.7</td><td>94.1</td><td>0.0</td><td>0.0</td><td>100.0</td></tr><tr><td>TP</td><td>72.8</td><td>100.0</td><td>0.0</td><td>0.0</td><td>100.0</td></tr></table>

Abbreviations: Fixed = Fixed-parameter model; High = High FFFRLS model; Middle = Middle FFFRLS model: Low = Low FFFRLS model.

(a)  
![](images/3698022355b14465055a42fbd13de5f728c3e8d22123605b537613860f2c8199.jpg)

(b)  
![](images/3474ae3f724d0cdeb39ee07dc061bd810a8339427dd88dca204528ef9ab59855.jpg)

(d)  
![](images/24e42121db832aa02a78fdf0fe1693cc8a90156be07aa209ecb6921d27aec82f.jpg)

(c)  
![](images/d4c05fce2e510f08be8731800b5aa0a8036b4ddfa747f8d145334849f0e988ba.jpg)

![](images/71989b09411ced8fe5598ba528948fbafd2e5a800427d7a84a9cf7d4e79757c1.jpg)

(f)  
![](images/c313da9319146505a3a6d60ffcd0b0692ed28e9ca2a2f4364fb06ca63491f415.jpg)

(g)  
![](images/b10da5d1b58d11c56658b299f5bd3bb7e0e961668f36b67f26df105ff7fac7a2.jpg)

(h)  
![](images/ab8368fa44314042696c3edfe909b3b4bd978faafa725fdc1a1bd6061cd4ca2e.jpg)

(i)  
![](images/7ae15e5c500ed03b2baf3762dac3a3a4d7fecc5a29e0d6c23d0fff2e7dd095bb.jpg)

(i)  
![](images/9878103ad887d5f79e942ef8a131558ca91c2635f56610a2d37bbe656bcde680.jpg)

(k)  
![](images/d146a61faaf1f93506d2c2088fe3bf0d534988b2176a92c27a9ac995a4756b31.jpg)

![](images/6fc88bf599c86f716dfda16a39be7aa4b96c853840f8e27df985ed41d1719238.jpg)  
Fig. 8. The statistical test results. (a), (b) and (c) are the results of healthy subjects under long-duration SP test. (d), (e) and (f) are the results of stroke patients under long-duration SP test. (g), (h) and (i) are the results of healthy subjects under long-duration TP test. (j), (k) and (l) are the results of stroke patients under long-duration TP test. <sup>∗</sup> represents the significant difference level is at $\mathsf { \overline { { p } } } { < } 0 . 0 5 , \ { * } { * }$ represents the significant difference level is at p<0.01, and <sup>∗∗∗</sup> represents the significant difference level is at p<0.001.

## V. DISCUSSION

Identification of muscle torque or force induced by FES is an important part in the design of model-based FES controlled systems to assist patients to restore lost motor functions. In this study, an adaptive Hammerstein model which using VFFRLS to update the parameters of the model based on neural network was proposed to enhance the adaption ability of Hammerstein model, allowing the model to cope with muscle time-varying property and fatigue.

In previous modeling studies, the models were often fixedparameter, with low accuracy [19], [29], [31]. Due to the dynamic characteristics of muscles and their propensity for fatigue, the torque induced by FES exhibited variability and fluctuations [21], [32], which is consistent with the Fig. 5 and Fig.7. Hence, it became necessary to update the parameters. Previous studies have commonly employed a fixed forgetting factor to enhance the adaptive ability of the Hammerstein model [21], [33]. However, the selection of an appropriate forgetting factor value is not easy. Insufficient stability of the system occurs when the forgetting factor is too small, leading to modelling failures. Conversely, when the forgetting factor is too large, the algorithm’s tracking performance is compromised due to the persistent high weighting of historical data, resulting in cumulative error [26], [34], [35], [36]. When applying the VFFRLS method to update the linear block parameters of the Hammerstein model, the problem of forgetting factor selection can be avoided as it is dynamically adjusted according to the error variation at different time instances. When the error is large, it suggests significant parameter variations. So, reducing the forgetting factor can improve model tracking. Conversely, when the error is small, it indicates a closer parameter identification to the actual value, and the algorithm tends to be stable. In such cases, increasing the forgetting factor appropriately is recommended to enhance algorithmic stability. Besides, according to [26], the analysis of the sum of squared error surface revealed a notably flat region around the ideal $( \lambda _ { m i n } , \rho )$ parameters. This observation suggests that the cost function remains relatively stable when the $( \lambda _ { m i n } , \rho )$ is varied within a certain range and the VFFRLS algorithm can achieve good results. Therefore, when applying the proposed model to predict torque in other joints, there is no need for significant parameter modifications. Not only that, neural network was adopted to represent the nonlinear block, which had better nonlinear expressivity than linear polynomials [24], [28], [29]. Despite the limited amount of data used as the training set, the model still achieved a high level of accuracy.

The proposed model not only significantly enhanced accuracy and robustness in predicting FES-induced torque but also demonstrated promising potential for generalizability across different scenarios and is expected to be applicable to modelbased FES tracking tasks. The accuracy of the model was comparable to that of the eEMG to torque model [21], [33], while avoiding the problem of inconvenient measurement of eEMG, such as errors caused by electrode displacement, sweating, and stimulation artefacts that tend to contaminate the eEMG signal recorded from the target muscle. At the same time, the proposed adaptive model also effectively reduced the maximum error of prediction, which could avoid insufficient or excessive output intensity due to the large transient error of model prediction in the FES control system. Besides, the model’s general potential across different FES profiles was also preliminarily explored. Based on the neural network identification results obtained from short-duration SP test, it was possible to update the parameters using the experimental results from TP test, thereby achieving torque prediction under TP test. By leveraging the information obtained from both types of tests, the model might adapt and accurately predict torque in various scenarios. However, SP stimulation had better prediction results than TP stimulation, especially for long-duration test. In voluntary muscle contractions, it was speculated that compared to the staircase force mode, the sinusoidal mode was more similar to dynamic voluntary tasks, and therefore, it might contain more patterns of voluntary contraction force [37]. Hence, we hypothesize that FES with sinusoidal excitation may also provide more valid activation information and closely align with the dynamic characteristics of muscle activity. As for the FES control systems, although there have been studies applying ILC control for gait modulation [38], [39] and model-based repetitive control for hand tremor suppression [5], these methods are non-precise control. To address the uncertainties and muscle fatigue induced by electrical stimulation, hybrid systems combining FES and robots have been developed [40], [41]. These controllers primarily rely on robots for error compensation and fatigue regulation, which increases the technical complexity and hardware requirements. However, as FES is a rehabilitation method that can promote muscle active contraction, it is important to improve the model accuracy to achieve precise control of FES which is beneficial for the comprehensive modulation of movement, rather than solely focusing on a single issue, such as the maximum ankle angle of gait. And we believe that a more accurate model aids in the development of model predictive control FES tracking tasks.

To enhance the robustness of the Hammerstein model and its ability to handle time-varying property and fatigue, the VFFRLS was proposed. However, despite instructions given to subjects to refrain from voluntary movements, there was no direct evidence to confirm that, and even slight changes in posture could affect the measurement of torque. Besides, the current experiments have only been conducted on limited patients with paralysis, and more patients will be recruited in the future. Additionally, introducing the torque-joint angle function into the proposed model holds great promise for its application in FES rehabilitation including model-based control and combining with robots to create hybrid systems, which can leverage the advantages of robotic control while also providing physiological benefits of FES.

## VI. CONCLUSION

In this study, an adaptive Hammerstein model which employing VFFRLS to update the parameters of the model based on neural network was introduced to predict the ankle torque induced by electrical stimulation. The performance of the proposed adaptive model, the fixed-parameter model and FFFRLS model was compared. It was concluded that the modeling results had been significantly improved compared with the fixed model and FFFRLS adaptive model in short-duration FES session and long-duration FES session. Therefore, the VFFRLS adaptive model is suggested to simulate ankle torque induced by electrically stimulated muscle.

## REFERENCES

[1] P. H. Peckham and J. S. Knutson, “Functional electrical stimulation for neuromuscular applications,” Annu. Rev. Biomed. Eng., vol. 7, no. 1, pp. 327–360, Aug. 2005, doi: 10.1146/annurev.bioeng.6.040803.140103.

[2] T. Schauer and C. Freeman, “Advances in functional electrical stimulation modelling and control,” Med. Eng. Phys., vol. 38, no. 11, pp. 1157–1158, Nov. 2016, doi: 10.1016/j.medengphy.2016.09.004.

[3] C. L. Lynch and M. R. Popovic, “Functional electrical stimulation,” IEEE Control Syst. Mag., vol. 28, no. 2, pp. 40–50, Apr. 2008, doi: 10.1109/MCS.2007.914689.

[4] C. Jiang, M. Zheng, Y. Li, X. Wang, L. Li, and R. Song, “Iterative adjustment of stimulation timing and intensity during FES-assisted treadmill walking for patients after stroke,” IEEE Trans. Neural Syst. Rehabil. Eng., vol. 28, no. 6, pp. 1292–1298, Jun. 2020, doi: 10.1109/TNSRE.2020.2986295.

[5] Z. Zhang, B. Chu, Y. Liu, Z. Li, and D. H. Owens, “Multimuscle functional-electrical-stimulation-based wrist tremor suppression using repetitive control,” IEEE/ASME Trans. Mechatronics, vol. 27, no. 5, pp. 3988–3998, Oct. 2022, doi: 10.1109/TMECH.2022.3150301.

[6] B. C. Allen, C. A. Cousin, C. A. Rouse, and W. E. Dixon, “Robust cadence tracking for switched FES-cycling with an unknown timevarying input delay,” IEEE Trans. Control Syst. Technol., vol. 30, no. 2, pp. 827–834, Mar. 2022, doi: 10.1109/TCST.2021.3070189.

[7] T. Afzal, L. Khan, and M. O. Tokhi, “Simulation of a patient driven strategy for FES supported sit-to-stand movement,” in Proc. Int. Conf. Inf. Emerg. Technol., Jun. 2010, pp. 1–5, doi: 10.1109/ICIET.2010.5625680.

[8] X. Tu et al., “Iterative learning control applied to a hybrid rehabilitation exoskeleton system powered by PAM and FES,” Cluster Comput., vol. 20, no. 4, pp. 2855–2868, Dec. 2017, doi: 10.1007/s10586-017- 0880-x.

[9] N. Kirsch, N. Alibeji, and N. Sharma, “Nonlinear model predictive control of functional electrical stimulation,” Control Eng. Pract., vol. 58, pp. 319–331, Jan. 2017, doi: 10.1016/j.conengprac.2016.03.005.

[10] N. A. Kirsch, X. Bao, N. A. Alibeji, B. E. Dicianno, and N. Sharma, “Model-based dynamic control allocation in a hybrid neuroprosthesis,” IEEE Trans. Neural Syst. Rehabil. Eng., vol. 26, no. 1, pp. 224–232, Jan. 2018, doi: 10.1109/TNSRE.2017.2756023.

[11] A. V. Hill, “The heat of shortening and the dynamic constants of muscle,” Proc. Roy. Soc. London B, Biol. Sci., vol. 126, no. 843, pp. 136–195, Oct. 1938, doi: 10.1098/rspb.1938.0050.

[12] R. Riener, J. Quintern, and G. Schmidt, “Biomechanical model of the human knee evaluated by neuromuscular stimulation,” J. Biomechanics, vol. 29, no. 9, pp. 1157–1167, Sep. 1996, doi: 10.1016/0021- 9290(96)00012-7.

[13] H. Gollee, D. J. Murray-Smith, and J. C. Jarvis, “A nonlinear approach to modeling of electrically stimulated skeletal muscle,” IEEE Trans. Biomed. Eng., vol. 48, no. 4, pp. 406–415, Apr. 2001, doi: 10.1109/10.915705.

[14] A. H. Vette, K. Masani, J.-Y. Kim, and M. R. Popovic, “Closed-loop control of functional electrical stimulation-assisted arm-free standing in individuals with spinal cord injury: A feasibility study,” Neuromodulation, Technol. Neural Interface, vol. 12, no. 1, pp. 22–32, Jan. 2009, doi: 10.1111/j.1525-1403.2009.00184.x.

[15] Z. Li, M. Hayashibe, C. Fattal, and D. Guiraud, “Muscle fatigue tracking with evoked EMG via recurrent neural network: Toward personalized neuroprosthetics,” IEEE Comput. Intell. Mag., vol. 9, no. 2, pp. 38–46, May 2014, doi: 10.1109/MCI.2014.2307224.

[16] J. Bobet and R. B. Stein, “A simple model of force generation by skeletal muscle during dynamic isometric contractions,” IEEE Trans. Biomed. Eng., vol. 45, no. 8, pp. 1010–1016, 1998, doi: 10.1109/10.704869.

[17] K. J. Hunt, M. Munih, N. D. N. Donaldson, and F. M. D. Barr, “Investigation of the Hammerstein hypothesis in the modeling of electrically stimulated muscle,” IEEE Trans. Biomed. Eng., vol. 45, no. 8, pp. 998–1009, Aug. 1998, doi: 10.1109/10.704868.

[18] Y. Li, M. Zheng, Q. Yang, and R. Song, “Comparison of two models used to describe the ankle torque in response to functional electrical stimulation,” in Proc. IEEE 4th Int. Conf. Adv. Robot. Mechatronics (ICARM), Jul. 2019, pp. 894–898, doi: 10.1109/ICARM.2019. 8834236.

[19] H. Y. Zhou, L. K. Huang, Y. M. Gao, Ž. L. Vasic, M. Cifrek, and M. Du, “Estimating the ankle angle induced by FES via the neural network-based Hammerstein model,” IEEE Access, vol. 7, pp. 141277–141286, 2019, doi: 10.1109/ACCESS.2019.2943453.

[20] Z. Cai, E.-W. Bai, and R. K. Shields, “Fatigue and non-fatigue mathematical muscle models during functional electrical stimulation of paralyzed muscle,” Biomed. Signal Process. Control, vol. 5, no. 2, pp. 87–93, Apr. 2010, doi: 10.1016/j.bspc.2009.12.001.

[21] Q. Zhang, M. Hayashibe, P. Fraisse, and D. Guiraud, “FES-induced torque prediction with evoked EMG sensing for muscle fatigue tracking,” IEEE/ASME Trans. Mechatronics, vol. 16, no. 5, pp. 816–826, Oct. 2011, doi: 10.1109/TMECH.2011.2160809.

[22] Y. Cao, X. Chen, M. Zhang, and J. Huang, “Adaptive position constrained assist-as-needed control for rehabilitation robots,” IEEE Trans. Ind. Electron., vol. 71, no. 4, pp. 4059–4068, Apr. 2024, doi: 10.1109/TIE.2023.3273270.

[23] H. Al-Duwaish, M. N. Karim, and V. Chandrasekar, “Hammerstein model identification by multilayer feedforward neural networks,” Int. J. Syst. Sci., vol. 28, no. 1, pp. 49–54, Jan. 1997, doi: 10.1080/00207729708929362.

[24] A. Janczak, “Neural network approach for identification of Hammerstein systems,” Int. J. Control, vol. 76, no. 17, pp. 1749–1766, Nov. 2003, doi: 10.1080/00207170310001633259.

[25] T. Wang, Q. Li, Y. Qiu, L. Yin, L. Liu, and W. Chen, “Efficiency extreme point tracking strategy based on FFRLS online identification for PEMFC system,” IEEE Trans. Energy Convers., vol. 34, no. 2, pp. 952–963, Jun. 2019, doi: 10.1109/TEC.2018.2872861.

[26] Z. Lao, B. Xia, W. Wang, W. Sun, Y. Lai, and M. Wang, “A novel method for lithium-ion battery online parameter identification based on variable forgetting factor recursive least squares,” Energies, vol. 11, no. 6, p. 1358, May 2018, doi: 10.3390/en11061358.

[27] H. Rouhani, M. R. Popovic, M. Same, Y. Q. Li, and K. Masani, “Identification of ankle plantar-flexors dynamics in response to electrical stimulation,” Med. Eng. Phys., vol. 38, no. 11, pp. 1166–1171, Nov. 2016, doi: 10.1016/j.medengphy.2016.07.011.

[28] Y. Liu, Y. Qin, B. Huo, and Z. Wu, “Functional electrical stimulation based bicep force control via active disturbance rejection control,” in Proc. 5th Int. Conf. Adv. Robot. Mechatronics (ICARM), Dec. 2020, pp. 306–311, doi: 10.1109/ICARM49381.2020.9195304.

[29] F. Le, I. Markovsky, C. T. Freeman, and E. Rogers, “Identification of electrically stimulated muscle models of stroke patients,” Control Eng. Pract., vol. 18, no. 4, pp. 396–407, Apr. 2010, doi: 10.1016/j.conengprac.2009.12.007.

[30] J. Bobet, E. R. Gossen, and R. B. Stein, “A comparison of models of force production during stimulated isometric ankle dorsiflexion in humans,” IEEE Trans. Neural Syst. Rehabil. Eng., vol. 13, no. 4, pp. 444–451, Dec. 2005, doi: 10.1109/TNSRE.2005.858461.

[31] Y. Li, W. Chen, J. Chen, X. Chen, J. Liang, and M. Du, “Neural network based modeling and control of elbow joint motion under functional electrical stimulation,” Neurocomputing, vol. 340, pp. 171–179, May 2019, doi: 10.1016/j.neucom.2019.03.003.

[32] Z. Z. Karu, W. K. Durfee, and A. M. Barzilai, “Reducing muscle fatigue in FES applications by stimulating with N-let pulse trains,” IEEE Trans. Biomed. Eng., vol. 42, no. 8, pp. 809–817, Aug. 1995, doi: 10.1109/10.398642.

[33] Z. Li, D. Guiraud, D. Andreu, A. Gelis, C. Fattal, and M. Hayashibe, “Real-time closed-loop functional electrical stimulation control of muscle activation with evoked electromyography feedback for spinal cord injured patients,” Int. J. Neural Syst., vol. 28, no. 6, Aug. 2018, Art. no. 1750063, doi: 10.1142/s0129065717500630.

[34] Y. Cai, R. C. de Lamare, M. Zhao, and J. Zhong, “Low-complexity variable forgetting factor mechanism for blind adaptive constrained constant modulus algorithms,” IEEE Trans. Signal Process., vol. 60, no. 8, pp. 3988–4002, Aug. 2012, doi: 10.1109/TSP.2012.2199317.

[35] S. Song, J.-S. Lim, S. Joon Baek, and K.-M. Sung, “Variable forgetting factor linear least squares algorithm for frequency selective fading channel estimation,” IEEE Trans. Veh. Technol., vol. 51, no. 3, pp. 613–616, May 2002, doi: 10.1109/TVT.2002.1002509.

[36] H. Wang, Y. Zheng, and Y. Yu, “Lithium-ion battery SOC estimation based on adaptive forgetting factor least squares online identification and unscented Kalman filter,” Mathematics, vol. 9, no. 15, p. 1733, Jul. 2021, doi: 10.3390/math9151733.

[37] R. Hu, X. Chen, C. Huang, S. Cao, X. Zhang, and X. Chen, “Elbowflexion force estimation during arm posture dynamically changing between pronation and supination,” J. Neural Eng., vol. 16, no. 6, Oct. 2019, Art. no. 066005, doi: 10.1088/1741-2552/ab2e18.

[38] P. Müller, A. J. del Ama, J. C. Moreno, and T. Schauer, “Adaptive multichannel FES neuroprosthesis with learning control and automatic gait assessment,” J. NeuroEng. Rehabil., vol. 17, no. 1, p. 36, Dec. 2020, doi: 10.1186/s12984-020-0640-7.

[39] Y. Dong et al., “Hybrid and adaptive control of functional electrical stimulation to correct hemiplegic gait for patients after stroke,” Frontiers Bioeng. Biotechnol., vol. 11, Aug. 2023, Art. no. 1246014, doi: 10.3389/fbioe.2023.1246014.

[40] C. A. Rouse, C. A. Cousin, B. C. Allen, and W. E. Dixon, “Shared control for switched motorized FES-cycling on a split-crank cycle accounting for muscle control input saturation,” Automatica, vol. 123, Jan. 2021, Art. no. 109294, doi: 10.1016/j.automatica.2020.109294.

[41] A. J. Del-Ama, Á. Gil-Agudo, J. L. Pons, and J. C. Moreno, “Hybrid FES-robot cooperative control of ambulatory gait rehabilitation exoskeleton,” J. NeuroEng. Rehabil., vol. 11, no. 1, p. 27, 2014, doi: 10.1186/1743-0003-11-27.