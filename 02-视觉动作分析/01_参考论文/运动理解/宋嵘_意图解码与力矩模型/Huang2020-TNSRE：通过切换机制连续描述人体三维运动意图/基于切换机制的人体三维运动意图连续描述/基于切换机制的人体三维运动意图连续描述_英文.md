# Continuous description of human 3D motion intent through switching mechanism

Yao Huang, Rong Song, Ahmadreza Argha, Andrey V. Savkin, Branko G. Celler, and Steven W. Su\*

Abstract—Post-stroke motor recovery highly relies on voluntarily participating in active rehabilitation as early as possible for promoting the reorganization of the patient’s brain. In this paper, a new method is proposed which manipulates cablebased rehabilitation robots to assist multi-joint body motions. This uses an electromyography (EMG) decoder for continuous estimation of voluntary motion intention to establish a cooperative human-machine interface for promoting the participation in rehabilitation exercises. In particular, for multi-joint complex tasks in three-dimensional space, a switching mechanism has been developed which can carve up tasks into separate simple motions. For each simple motion, a linear six-inputs and threeoutputs time-invariant model is established respectively. The inputs are the processed muscle activations of six arm muscles, and the outputs are voluntary forces of participants when executing a multi-directional tracking task with visual feedback. The experiments for examining the decoder model and EMGbased controller include model training, testing and controller application phases with seven healthy participants. Experimental results demonstrate that the decoder model with the switching mechanism could effectively recognize arm movement intention and provide appropriate assistance to the participants. This study finds that the switching mechanism can improve both the model estimation accuracy and the completeness for executing complex tasks.

Index Terms—Rehabilitation Robotics, Myoelectrical control, EMG decoder, Multi-movement task, Model switching mechanism.

## I. INTRODUCTION

O regain lost motor functions, rehabilitation training with sufficient intensity and voluntary participation of poststroke patients is essential for brain reorganisation [1]. Due to the superiority of robotic system in terms of efficiency, precision, and controllability, a range of robots have been developed for post-stroke rehabilitation on the recovery of motor capacity [2], [3]. As a kind of parallel robot, cable-based robots show great advantages with respect to high accuracy and low inertia during continuous rehabilitation training and can provide proper assistance for upper limb rehabilitation and daily activity training while guaranteeing the safety of patients [4], [5].

The key characteristic of the robots used in different modes and phases of rehabilitation in neuro-rehabilitation is whether the robot with control strategies can provide appropriate assistance and reliable assessments for participants. The assistive mode as the most developed control algorithms is to help participants move the affected limbs in the desired therapy patterns [6]. Participants’ voluntary involvement is proved to be essential for provoking the motor plasticity [1] and inducing the brain plasticity [7]. For patients in the early stages of poststroke rehabilitation, the control strategies for providing passive assistance are applied. For example, an advanced proportional and differential controller [8] and a sliding mode controller [9] have been used to accurately control a rehabilitation robot to passively assist the patient to move to the desired end position. To increase and maintain the participants’ efforts, by increasing the compliance of passive controllers, the rehabilitation robot is expected to provide only partial assistance. To this end, the control techniques such as the impedance control and admittance control that can implement a law for compromising between training quality and robotic compliance are developed [10].

With the high requirements of the participants’ voluntary involvement during rehabilitation, the EMG signals which are highly related to the muscular forces and joint torque are valued [11], [12]. Furthermore, another reason for the popularity of EMG based control strategy is that the traditional electromechanical force/torque based control often involves electromechanical delay. For EMG based control, a nonlinear hill-based muscle model is derived for continuously predicting the limb moment from the muscle activities and joint kinematics [13]. However, due to the vast parameters needed for each muscle and nonlinearity of the model equations, the hill-based model is rarely applied in control involving multiple joints and muscles. An alternative approach is to train a decoder between the EMG signals and motor control variables by simply considering the relationship between them as a black box.

By applying binary or multiclass classification algorithms, the robot can be controlled in discrete mode. Dipietro et al. map the EMG signals into the binary output for determining the onset and end of the movement [14]. Ding et al. represent the relationship between the EMG patterns and predefined static postures or movements by the Artificial Neural Network algorithm [15]. For continuously controlling a single joint robot to assist participants, apart from the Hill model based approach [13], a linearly proportional EMG control strategy is proposed [16]. Based on the theory of muscle synergy [17] and movement primitives [18], Artemiadis et al. [19] proposed a state-space model with reduced dimensionality for continuously mapping the EMG signals to multi-joint movements. However, the performance of these models is mostly task-dependent and participant-dependent, which settles a big challenge in modeling accuracy with reduced model dimensionality.

Several research groups have developed switch schemes for decoding EMG signals to accomplish different tasks. Nizamis et al. [20] have developed an EMG-based control interface to detect the participant movement intention with an interface switching between two horizontal and one vertical tracing tasks. This switch regime for these tasks is based on the simple EMG-based proportional (direct) control and the counts of the wrist extensor contraction. Artemiadis et al. [21] have a switching decoding scheme where the switching variable was described by the participant to perform force tasks in ten different points in the 3D arm workspace. This switch regime can assist a participant to finish different reaching movement, but these movements are not continuously executed between every two movements. Artemiadis et al. [22] also put up a switching regime decoding model between multi-channel EMG signals and joint angles of arm movements in 3D space. They set up a discrete switching variable which controlled through a Bayesian classifier for choosing different decoding models for different movements.

Motivated by the switch model based approaches [20]–[22], in this paper, an EMG decoder is developed using switched linear system models to continuously estimate participants voluntary motion during multi-joint arm movements. Different from switching control based approach [20], the developed EMG decoder served as a human-machine interface rather than a controller. Although the functionality of the developed EMG decoder is similar to the one developed by Artemiadis et al. [22], the modeling strategy adopted in this paper is quite different. In [22], the model dimensionality reduction is applied for models of every task and the dimensions of all these models are reduced into two for both inputs and outputs. As discussed earlier, since the dimensionality-reduction might cause the loss of limb dynamics and detailed information included in the EMG signals, its applicability to the problem of cable rehabilitation robot used for multi-joint motion in 3D space needs to be further investigated. Moreover, these models are mostly task-dependent and participant-dependent approaches, limiting their application.

In this paper, a switched system model is established to continuously decode EMG signals during a multi-directional arm tracking task for the application of controlling a cablebased rehabilitation robot in active assistive mode. The EMG signals of six muscles are processed into muscle activations as the model input, while the voluntary forces of participants are processed as the model output. By dividing the complex task into several simple subtasks, we individually identify a model for each subtask with appropriate dimensionality. Three single-models with different orders for the whole complex task are also trained for comparison purposes. The efficiency and the feasibility of the switching mechanism have been demonstrated in both numerical analysis and experimental verification in terms of model fitness, tracking accuracy and muscle activations for human-robot cooperative manipulation.

It is well known that the human brain is a complex nonlinear time-varying system. Description of the motor intent, under a very specific condition, is feasible but very difficult. The research work in this paper demonstrates the effectiveness of using a switching mechanism to describe brain motion intent; just as the spline interpolation can be applied to approximate the static nonlinear functions, the dynamic switched system model can effectively approximate the time-variant nonlinear dynamical system.

The rest of this paper is organized as follows. The proposed system architecture and experiments are analyzed and reported in Section II, while Section III gives the results and Section IV discusses and concludes this paper.

## II. METHOD

## A. Platform

A cable-based upper limb rehabilitation robotic system as shown in Fig. 1(a) is used to provide assistance to participants. The robot consists of a cubic frame, a motor group, an arm splint as end-effector, cables connected with the prior three parts, a motion capture system, a computer with a wide screen and an EMG acquisition system. The details of the robotic system are described in [23]. The screen is placed directly in front of the participants, which shows the target and actual positions of the participants’ wrist using virtual cursors in realtime. Participants place their arms in the splint and secure them with straps. The frequency of the motion capture system is 100 Hz. The frequency of the EMG acquisition system is 1000 Hz, and amplifiers are with the magnitude of 5000.

![](images/f3f965d7e5cfe22c29b019d9298b115f59a1d7c67eec2f5e8a6aba47f325e1d0.jpg)  
Fig. 1. A cable-based upper limb rehabilitation robotic system and a multidirectional arm tracking task including four rectilinear movements: (a) The platform. (b) Task trajectory and direction.

## B. EMG-based human-machine cooperation controller

The human-machine cooperation controller based on an EMG decoder (Fig. 2 (a)) is used for controlling the robot. The EMG decoder (Fig. 2 (b)) is trained to estimate the EMGdriven forces along the cables from muscle activations. The muscle activations are obtained by basic EMG processing and EMG-activation model based on a previous research [16]. The arm and robot dynamic models are built for reckoning the voluntary forces driven by muscle’s EMG from the kinetics of arm together with the splint on the basis of arm and robot structure. The EMG-driven forces can be further transferred into a motor torque for driving the motor group.

![](images/437a8333e717383866e540c79c22eba756a769046bd142f8301512d186c25286.jpg)  
Fig. 2. The detailed architecture of the proposed EMG-based human-machine cooperation controller, the EMG decoder, and real-time decoding mechanism with the single-model and the switched system model. (a) Interfaces with a trained EMG decoder. The EMG-driven forces mapped from the EMG by the decoder is used as an interface to assist participants together with their voluntary motion intention. (b) Training Decoder Model. The decoder is trained to map the relationship between muscle activations from EMG signals and voluntary forces driven by muscles from arm dynamics. After being trained, the decoder is used to estimate the EMG-driven forces and are realized by the robot in real-time. (c) Two trained decoding models in the real-time control scheme: the switched system model and the single model.

1) EMG-activation model: The EMG to muscle activation model is built for calculating the muscle activated levels from the amplified EMG signals measured by the EMG acquisition system. First, the amplified EMG signals are full-wave rectified by a 4-th order Butterworth low-pass filter to obtain the envelope of amplified EMG signals [19]. Then, the envelop magnitude of each muscle is normalised to the values of its maximum voluntary isometric contraction [24].

The neural activations $u ( t )$ are then obtained from the normalized envelope $e ( t )$ of EMG signals in real-time by a second-order discrete-time linear model [25]:

$$
u (t) = \alpha e (t - d) - \beta_ {1} u (t - 1) - \beta_ {2} u (t - 2),\tag{1}
$$

where $\alpha = 1 + \beta _ { 1 } + \beta _ { 2 }$ is the gain coefficient, $\beta _ { 1 } = c _ { 1 } + c _ { 2 }$ and $\beta _ { 2 } = c _ { 1 } c _ { 2 }$ are the recursive coefficients, $c _ { 1 } , c _ { 2 }$ are adjustable parameters for each muscle based on previous studies [25] and d is the electromechanical delay, which is set as 80 ms in this study.

The muscle activations are then obtained by the coupled relationship between it and the neural activations. The relationship can be presented by first-order dynamics [26] and by a non-linear function at low levels of force [27]. A oneparameter segmented transformation model from the neural activation u(t) to the muscle activation a(t) is proposed by Manal and Buchanan:

$$
a (t) = \left\{ \begin{array}{l} \alpha^ {m a} \ln (\beta^ {m a} u (t) + 1), \quad 0 \leq u (t) <   u _ {0} \\ m u (t) + c, \quad u _ {0} \leq u (t) \leq 1 \end{array} \right.\tag{2}
$$

where the values of $u _ { 0 } , \alpha ^ { m a } , \beta ^ { m a }$ , m and c depend only on the shape factor of each muscle [27].

In order to match the frequency of muscle activations and the EMG-driven forces, the muscle activations are further decimated into 100 Hz.

2) Arm and Robot dynamics analysis: The dynamic model is used to calculate the EMG-driven forces on the wrist during the human-robot cooperation movements of the arm and splint. The wrist is the contact point between the robot and arm. During the modeling phase, as the assistance from the robot is unavailable. Therefore, the forces driven by EMG on the wrist consist of the motion-driven forces and the forces for resisting the effects of arm and splint gravities. During the verifying phase, the cables will drive the arm and the splint moving together for executing the task. The forces provided by the motor along the cables are estimated from the EMG decoder model based on the real-time EMG. The kinetics of the robot is applied for calculating the forces along the cables and further the driven torques of the motor group as described in [23].

3) EMG decoder model: A number of algorithms have previously been applied to decode human motion from EMG [28]. However, in most of these works, decoding is resolved using classification techniques, rather than a continuous description of the kinematics. As discussed in the introduction section, in our case, the goal is to develop an EMG decoder to continuously represent the voluntary motion forces using the measured muscle activation in real time.

The EMG decoder developed here is based on the identification of state-space model. There needs a process of model training phase to identify the parameters of the state-space model. Comparing with the approaches using the discrete-time model [28], the linear six-inputs and three-outputs continuous time-invariant model (3) is applied to map the relationship between the muscle activations and the voluntary motion forces. Although the data collection is discrete, here we use a continuous model, which might capture the key characteristics as the actual system is continuous:

$$
T _ {i}: \left\{ \begin{array}{l} \dot {x} = A _ {i} x + B _ {i} u \\ y = C _ {i} x + D _ {i} u \end{array} \right.\tag{3}
$$

where $x \in \mathbb { R } ^ { k }$ is the state vector and k is the order of the model, which can be pre-determined. In this study, k is selected between 3 and 10 depending on different scenarios. $u \in \mathbb { R } ^ { 6 }$ is the input vector of six muscle activations, $y \in \mathbb { R } ^ { 3 }$ is the force vector representing the model output. The matrices $A _ { i } \in \mathbb { R } ^ { k \times k } , B _ { i } \in \mathbb { R } ^ { k \times 6 }$ , and $C _ { i } \in \mathbb { R } ^ { 3 \times k }$ are the systematic matrix, input/control matrix, and output matrix, respectively. The matrix $D _ { i } \in \mathbb { R } ^ { 3 \times 6 }$ is the direct transfer matrix from the input u to output y and is a zero matrix in this study.

The state space model for a given task $T _ { i }$ can be established by identifying the matrices $\{ A _ { i } , B _ { i } , C _ { i } \}$ and the initial state $x _ { 0 _ { i } }$ using a model training dataset. These parameters are identified in the continuous time domain using one of the subspace model identification methods, the canonical variate analysis [29]. As the underlying system is stable, the stability of the model is added as a compulsory condition during model identification. In this study, in order to reduce transient behavior due to switching, the initial value $x _ { 0 _ { i } }$ for each subsystem $T _ { i }$ is adjusted within the range identified in the training phase. This adjustment can reduce the discontinuity of the output during switching so as to improve the transient switching dynamics.

4) A switching mechanism for complex tasks: To explore the best EMG decoder model during complex tasks, a switching mechanism for carving up the task into several simple subtasks and individually identifying a subsystem model for each subtask is set up as shown in Fig. 2 (c). The principle of the switching mechanism is to establish subsystem models for every subtask with appropriate dimensionality and construct a switched linear system model for the whole complex task to achieve high accuracy by switching among the subsystem models.

In this study, the complex task is the tracking of a square shape preset trajectory in a horizontal plane (Fig. 1(b)). This task contains four rectilinear movements (i.e. the four subtasks) with different directions in 3D space. The length of each rectilinear movement is ${ \sqrt { 0 . 0 2 } } m$ . Each rectilinear movement can be easily realised by setting the changes of X and $Y$ axes both as 0.1m in the developed trajectory setting module. Based on this task, the realization of the switching mechanism can be described as follows: 1) Divide the whole task into four subtasks; 2) Train a decoder model for each subtask, and achieve four linear time-invariant multivariable models $\{ A _ { i } , B _ { i } , C _ { i } \}$ with $x _ { 0 _ { i } } ~ ( i \in { 1 , 2 , 3 , 4 } ) ;$ 3) Switch the model according to the predefined subtasks.

To simplify the notations, in the following discussions, we use the prefix $\mathbf { \partial } ^ { \bullet } \mathbf { S } ^ { \bullet }$ to represent the case of a single model and ‘MM’ to the case the system switching among Multiple Models for a complex task. Based on the switching mechanism, a switched linear system model is trained. In this study, the MM is the integration of four best subsystem models for the four subtasks. For the subtasks, all subsystem models, whose orders could be between three to six (total four different options), are trained. The best subsystem model of each subtask is defined as the model whose average fitting errors of the three model outputs are the smallest among all four models with different orders. The order of each best subsystem model for each participant could be different. After selecting the best model for every subtask, the MM can be grouped and realized.

For comparisons, three single-models with different orders are also trained for the whole square shape task without considering the complexity of the task. For these singlemodels, their orders are chosen as three (S3), six (S6) and ten (S10) (please see Appendix A for details).

## C. Participants and Experimental protocol

Seven healthy participants (aged $2 5 . 3 \ \pm \ 0 . 7 \ \mathrm { y r s } )$ were recruited. All participants signed the written informed consent forms. This study was approved by the Human Ethics Committee of the first affiliated Hospital of Sun Yat-Sen University ([2013]C-096).

Based on biomechanics literature [30], the surface EMG signals of six muscles (Posterior of deltoid (DP), medial of deltoid (DM), anterior of deltoid (DA), triceps (TRI), biceps brachii (BIC) and brachioradialis (BR)), which are mainly responsible for analyzing the upper limb motions, were recorded by the EMG acquisition system. Two paired surface EMG electrodes were placed on the specified surface skin of participants. The reference electrodes were placed on the skin of the elbow or wrist bones area. To capture the kinematics characteristics of upper limb, three infrared-reflection markers were attached to the skin surface of the centre of shoulder, elbow, and wrist. The EMG signals and joint position signals were recorded simultaneously.

The efficiency of the proposed switching scheme was verified using both numerical analysis and real-time experiments with seven participants. Numerical analysis (mainly modeling) consists of the EMG decoder model training phase and model testing phase.

The datasets for the training phase and the testing phase were collected based on a procedure described as follows. Each participant was asked to track a moving cursor shown on the screen using his/her arm for 20 seconds in total without robot’s assistance. The robot provided zero assistance to the subjects, as the force feedback loop was open. The trajectory of the moving cursor is the preset square shape trajectory in the horizontal plane. The cursor moved for 5 seconds during each rectilinear movement. Each participant was asked to track the cursor which repeatedly moved along the square shape trajectory three times as 3 trials in one record. All participants were required to finish three records, resulting in a total of 9 trials whose data were collected for training and testing phases. Two records (i.e. 6 trials) were randomly selected for the training phase and one record (i.e. 3 trials) was for the testing phase. The data for model training was processed to train both the multi-model and three single-models.

In the testing phase, the EMG signals were also processed as the input, and the trained models were used to estimate the outputs. The estimated outputs were compared to the targeted output to assess the accuracy of the models. As noted earlier, during testing stage, the muscle activities did not response to the tracking error through human visual feedback.

In the experimental verification stage, the participants were instructed to place their forearms in the splint. All four models were applied in the EMG-based controller to realize the human-robot cooperation movements. Therefore, the robot provided assistance based on EMG signals as the force feedback loop was closed. During this phase, each participant was asked to track the cursor once in one record, indicating one trial in a record. All participants were required to finish three records (i.e. 3 trials) with robot assistance estimated by every model together with their muscle contributions. The performance of different models can be assessed in a realtime feedback loop. In this loop, the human 3D motion intent, estimated using the identified models, was reflected in the human visual feedback in real time to reduce the tracking error as indicated on the computer screen.

For safety reasons, the motor speed was constrained. To this end, the range of forces was limited to [−100N, 100N]. Specifically, if the forces were out of the range, the forces were set to −30N (if less than −100N) or 30N (if more than 100N).

## D. Evaluation parameters and statistical analysis

To evaluate the performance of different models, the model fitness, the trajectory tracking error, and the muscle activations were calculated, analyzed and compared.

1) Model Fitness: Based on the EMG decoder models, the target outputs from the joint positions and estimated outputs from the models with the same inputs can be calculated.

The model fitting error (MFE) was defined as the gap between the target outputs and estimated outputs. To evaluate the model accuracy, the root mean square (RMS) value of MFE was calculated as follows:

$$
M F E _ {R M S} = \sqrt {\frac {1}{N} \sum_ {i = 1} ^ {N} (\Delta F (i)) ^ {2}}\tag{4}
$$

where $\Delta F ( i )$ is the absolute value of MFE in each output at i-th sampling instant and N is the number of samples.

Furthermore, the Pearson Correlation Coefficient (PCC) between the targeted and estimated outputs was also calculated according to [31].

2) Trajectory Tracking Accuracy: The gap between the desired preset trajectory of the task and the actual trajectories completed by participants was defined as the tracking error (TE). The RMS values of TE were calculated to evaluate the trajectory tracking accuracy in X axis as follows:

$$
T E _ {R M S} = \sqrt {\frac {1}{N} \sum_ {i = 1} ^ {N} (\Delta x (i)) ^ {2}}\tag{5}
$$

where $\Delta x ( i )$ is the trajectory tracking error in X axis at the i-th sample. The $T E _ { R M S }$ of the other axes was calculated similarly. To further explore the effectiveness of the switching mechanism, when the human-robot assistance was available, the $T E _ { R M S } s$ in X and Y axes were calculated and compared separately.

3) Muscle Activation: The mean value of muscle activations (MMA) of the six selected muscles during every task execution was used to evaluate the effects on muscular effort by the EMG decoder models.

4) Statistical analysis: To analyze the influence of model order in model fitting accuracy, the difference of the MFE and PCC was evaluated using Kruskal-Wallis nonparametric test with pairwise multiple comparisons. To visually show the performance differences among different models in three phases, the mean and standard deviation (SD) values of these evaluation parameters were calculated for all seven subjects.

To analyze whether the robot’s participation affects the human motion intention, the difference of TE between the training and verifying phases was evaluated by one-way analysis of variance statistical method and the MME was evaluated by Kruskal-Wallis nonparametric test. The significance level of all statistical tests was 0.05. In the verifying phase, only the model that can support all participants to complete the task was compared. Statistical work was performed using SPSS 19.0 (SPSS Inc., USA).

It should be pointed that some of the participants could not complete the whole task when the single models S6 and S10 were used. Therefore, to analyze the effects of the switching mechanism during experimental verification, only the single model S3 and switched model MM were evaluated by the paired test in terms of the tracking error and muscle activations.

## III. RESULT

## A. Model Fitness

First, the estimation of the EMG-driven forces by using various EMG decoder models were presented in Fig.3. The data shown in this figure was captured and estimated from one subject as a sample.

The bold solid back lines and the thick grey lines illustrate the target outputs calculated by the dynamic model, while the green thin lines (single-models: S3, S6 and S10) and light orange solid line (switched model: MM) show the the EMGdriven forces estimated by the four different EMG decoder models. In the training and testing phases, all estimated outputs had similar trends with the target outputs during a 20-second trial, and the values were close. During the verifying phase of this sample, the outputs estimated by S6 and S10 went far away from the target, therefore these outputs are not included in this figure. Among all seven participants, the S6 of four participants and the S10 of six participants out of seven participants were not able to help the participants complete the verifying experiments with those outputs out of the range of motor limits. Therefore, there were three subjects who could complete the task with S6 and only one subject who could finish the task with S10.

The RMS values of MFE and the mean values of PCC were compared for each output among different models as shown in Fig. 4 and Table. I. In the training phase, the MFE and PCC of all three single-models, were significantly worse than those of the switched model except one output of S6. Moreover, the MFE and PCC of S6 in two outputs were significantly better than those of S10. In the testing phase, the MFE and PCC of all three single-models were found significantly worse than the switched model while no significant difference was found among single-models. In the verifying phase, the MFE values significantly increased with the increment the order of singlemodels. Based on the significance test, the performance of MM was significantly better than the S3 except for one output. Also, it is quite clear that the performance of S6/S10 is not as good as the switched model. It should be noted not all the subjects could finish the task with the two single models S6 and S10, as reported in Table. I.

![](images/b2950bed9fee219cc1b8e3875c0d66e63529171b014d48a471d6f4ba5fd0e21a.jpg)  
Fig. 3. The three outputs of the multi-model (MM) and three single-models (S3, S6, S10) during the training ((1)-(3)), testing ((4)-(6)) and verifying phases ((7)-(9)). The thick black lines in ((1)-(6)) indicate the target outputs from the dynamic model during the training and testing phases. The thick black lines and the thick grey lines in ((7)-(9)) indicate the two target outputs of two experiments during the verifying phase. The light orange solid lines in ((1)-(9)) denote the outputs of MM. The dark green dotted lines in ((1)-(9)) denote the outputs of S3. The medium green dash-dotted lines in ((1)-(6)) denote the outputs of S6. The light green dashed lines denote in ((1)-(6)) the outputs of S10.

## B. Tracking Accuracy

The preset trajectory of the task and actual trajectories by a participant conducting the task in training and verifying phases with different models are shown in Fig.5. Notice that, as shown in these trajectories, this participant could not finish the task with the S6 model and S10 model (their trajectories stop at the red circle points in Fig.5).

Fig.6 presents the mean RMS values of the TEs in X and Y axes during participants tracking the whole task. The values of S6 and S10 (in dashed edges) show great SD because some participants could not finish the tasks. The TE values in X-axis of MM is significantly lower than the values of S3 $( p < 0 . 0 5 )$ .

## C. Muscle Activation

The MMAs of the seven muscles are displayed in Fig.7. All muscle activations decreased when participants tracked the task with assistance from the robot. The activation of TRI and DA during the verifying phase significantly decreased in comparison with the training phase. No significant difference was found between S3 and MM during the real-time verifying phase.

![](images/48b91af7d48dacc0d08fa2b6d2bcbf0257f2c6fb02c5fca442d4786e16213989.jpg)  
Fig. 4. The mean RMS model fitting error values among all participants in training, testing and verifying phases.

## IV. DISCUSSION

In this study, an EMG decoder has been constructed using the state space model without dimensionality-reduction [19] to estimate the forces using the EMG signals of the six arm muscles that contribute primarily and can reveal voluntary intention.

TABLE I  
THE COMPARISON BETWEEN THE RMS MFE AND PCC VALUES OF THREE SINGLE-MODELS AND THE SWITCHED SYSTEM MODEL DURING TRAINING, TESTING AND VERIFYING PHASES

<table><tr><td rowspan="2">Phase</td><td rowspan="2">Output</td><td colspan="4">MFE (Mean±SD) (N)</td><td rowspan="2">Significance</td></tr><tr><td>S3</td><td>S6</td><td>S10</td><td>MM</td></tr><tr><td rowspan="3">Training</td><td>O1</td><td>1.01±0.72</td><td>0.74±0.32</td><td>1.11±0.53</td><td>0.41±0.30</td><td>a,b,c</td></tr><tr><td>O2</td><td>1.19±0.85</td><td>0.70±0.41</td><td>1.30±0.60</td><td>0.43±0.28</td><td>a,c,d</td></tr><tr><td>O3</td><td>0.85±0.65</td><td>0.62±0.14</td><td>1.02±0.50</td><td>0.33±0.17</td><td>a,b,c,d</td></tr><tr><td rowspan="3">Testing</td><td>O1</td><td>1.71±0.97</td><td>2.28±0.89</td><td>2.40±0.64</td><td>0.83±0.26</td><td>a,b,c</td></tr><tr><td>O2</td><td>1.91±1.26</td><td>2.52±1.23</td><td>2.46±1.05</td><td>0.96±0.45</td><td>a,b,c</td></tr><tr><td>O3</td><td>1.65±0.96</td><td>2.00±0.67</td><td>2.10±0.69</td><td>0.69±0.20</td><td>a,b,c</td></tr><tr><td rowspan="3">Verifying</td><td>O1</td><td>4.02±2.74</td><td>(NF) 4.88±3.26</td><td>(NF) 8.80±0.36</td><td>3.59±1.72</td><td></td></tr><tr><td>O2</td><td>4.71±2.25</td><td>(NF) 3.77±1.65</td><td>(NF) 4.43±0.28</td><td>3.23±1.90</td><td>a</td></tr><tr><td>O3</td><td>4.93±3.61</td><td>(NF) 3.94±1.60</td><td>(NF) 9.01±0.13</td><td>2.71±1.32</td><td>a</td></tr><tr><td rowspan="2">Phase</td><td rowspan="2">Output</td><td colspan="4">PCC (Mean±SD)</td><td rowspan="2">Significance</td></tr><tr><td>S3</td><td>S6</td><td>S10</td><td>MM</td></tr><tr><td rowspan="3">Training</td><td>O1</td><td>0.81±0.33</td><td>0.95±0.06</td><td>0.86±0.12</td><td>0.98±0.02</td><td>a,b,c</td></tr><tr><td>O2</td><td>0.80±0.32</td><td>0.94±0.09</td><td>0.83±0.16</td><td>0.98±0.02</td><td>a,c,d</td></tr><tr><td>O3</td><td>0.82±0.27</td><td>0.96±0.02</td><td>0.79±0.21</td><td>0.98±0.02</td><td>a,b,c,d</td></tr><tr><td rowspan="3">Testing</td><td>O1</td><td>0.76±0.26</td><td>0.78±0.28</td><td>0.56±0.41</td><td>0.96±0.02</td><td>a,b,c</td></tr><tr><td>O2</td><td>0.68±0.48</td><td>0.75±0.18</td><td>0.56±0.43</td><td>0.93±0.08</td><td>a,b,c</td></tr><tr><td>O3</td><td>0.76±0.27</td><td>0.77±0.21</td><td>0.73±0.15</td><td>0.96±0.03</td><td>a,b,c</td></tr><tr><td rowspan="3">Verifying</td><td>O1</td><td>0.17±0.59</td><td>(NF) 0.03±0.62</td><td>(NF) 0.01±0.02</td><td>0.49±0.51</td><td>a</td></tr><tr><td>O2</td><td>0.23±0.48</td><td>(NF) 0.19±0.46</td><td>(NF) -0.17±0.05</td><td>0.62±0.20</td><td>a</td></tr><tr><td>O3</td><td>0.17±0.43</td><td>(NF) 0.02±0.65</td><td>(NF) 0.13±0.04</td><td>0.70±0.12</td><td>a</td></tr></table>

Note: Significant differences between these models are represented by the following alphabets. a: ‘S3-MM’, b: ‘S6-MM’, c: ‘S10-MM’, d: ‘S6-S10’. NF: Not all participants have completed the tasks. For those models which cannot support all subjects to complete the task, the mean and SD values of MFE and PCC are only calculated from those subjects who successfully completed the task.

![](images/566d23e7fa104373bcd1d019a67e843736061b64a1c08201ac0618d49910c7b8.jpg)  
Fig. 5. The trajectories of the training and verifying phases. The two red circles ‘O’ show the end position of the unfinished trajectories. The plus markers ‘+’ show the end position of the finished trajectories. The star markers $\cdot _ { * } \cdot$ show the half position of the trajectories.

![](images/bc5f95ee3a3f23ef138a638e46687bb268c63c5e14e4b556240ddbcbe0cf9db3.jpg)  
Fig. 6. The mean RMS tracking error values in X and Y axes during the training and verifying phases with different models. The dashed edge indicated tasks were not totally finished by all participants.

![](images/9d087353a53e133e15af1456d9c9ce54bce0c02d15a624513cd47af37aaec5a9.jpg)  
Fig. 7. The mean muscle activations of six muscles during the training and verifying phases with different models. The dashed edge indicated tasks were not totally finished by all participants.

Furthermore, a switching mechanism has been applied for complex motion tracking by integrating simple motion models. The results indicated that the proposed switching mechanism can well estimate the forces when performing complex tasks. It is demonstrated that the proposed approach can be applied in a cable-based rehabilitation robot.

## A. The novelty of the EMG decoder model without dimensionality reduction in myoelectrical control

Most EMG decoders for myoelectrical control strategies during human-robot cooperation are either represented in a discrete mode or restricted in one-dimensional space/single joint. Although the Hill-based model [24] is built and frequently used for continuously decoding arm motion from EMG signals [32], the nonlinearity of the decoder model and vast parameters of each muscle needed to be measured make the analysis rather difficult for multiple muscles and joints cooperation movements. The linear state-space model is reported to be able to continuously decode upper limb motion from EMG signals for controlling an anthropomorphic robot in the 3D space by Artemiadis et al. [19], while an extra dimensionality-reduction technique is needed before modeling. The dimensionality-reduction can reduce the difficulty of the modeling, but it might cause the loss of limb dynamics and detailed information included in EMG signals. Moreover, by tuning the parameters of the state-space model, the participantspecific relationship between inputs and outputs can be clearly identified without dimensionality-reduction. A vital part of the modeling scheme is the initial value of the state x, which highly affects the accuracy of the model. However, neither the dimension nor the values of x was fully considered in previous researches.

In this study, the linear state-space model without dimensionality reduction has been developed for decoding EMG signals and further applied to control a cable-based rehabilitation robot in real-time. By training multiple-input (six muscle activations) and multiple-output (three forces) models without reductions, the EMG-driven forces of the upper limb during rehabilitation tasks in 3D space can be estimated and applied to assist the participants based on their voluntary efforts. The results of MFE and PCC analysis in training phases showed that the tracking error of the single-model S6 is the smallest among the three single models. However, the results in testing phase showed that the performance of the lowest order model S3 is the best among all three single-models and is close to the training phase. It indicated that the model with lower order is more robust than the higher order models and it is in-line with the previous research [19].

However, non single-models can maintain its performance in real-time verification. The difference between the testing and verifying phases is whether the human visual feedback system participates in the control loop of the human brain cooperative motion. In other words, different with testing phase, the verifying phase is the assessment of the model performance in the closed loop manner. Due to the robustness of the closed loop control system, the stability can be maintained and the performance degradation is acceptable when the modeling error is small (in the case of MM) or not big (in the case of S3). However, if the modeling error is big, out of the tolerance of the participants, then the closed loop will become unstable, which is the cases of S6 and S10, i.e., for S6 and S10, some participants could not complete the whole tracking task. This indicates the closed loop performance of different models is also related to the motion adaptability of participants. As the motion adaptability of a stroke patient is often much lower than healthy participants, the EMG decoding model needs to be more accurate even for the testing phase. Therefore, the proposed switched model is the most suitable candidature for the active rehabilitation of post-stroke patients.

## B. Superiority of the switching mechanism for modeling the human-related system

The proposed switching mechanism aims to address current issues of active rehabilitation, and achieved good performance in terms of model accuracy, the required training data size, and the individualization of models.

By considering the initial state of each subsystem model, the best model is trained for each subtask, and the switching system model exhibits excellent performance in both the test and verifying phases. Therefore, the switching mechanism has high robustness and can decode EMG signals to assist participants during human-robot cooperation movements. This mechanism is able to improve model accuracy and replace the extra dimensionality-reduction technique without losing information.

The switching mechanism for carving up the complex task into simple subtasks and then training for each subtask is similar to the brain’s muscle control strategy, i.e., by incorporating different muscle synergies to complete a complex movement. During the real-time robot-aided control strategy based on physiology signals, due to the involvement of human subjects, both the experimental time and its data size are often limited. During the training phase of the single-models (i.e., without switching), the complex task needs to be finished perfectly by the participants; otherwise the stimulation would not be enough to identify the model parameters. These requirements certainly increase the difficulty in completing the task and further reduce the universality of the model.

On the other hand, to train the model, in each individual subtask, the best model can be quickly trained with limited data. Even if the participant does not complete the entire task but completes two or three subtasks, data can still be used to train the models for these particular subtasks. Therefore, the switching mechanism is more suitable for the modeling of human involved processes, in which, due to safety reasons, the volume of the data is often limited.

During the experiments, when the high order single models were utilized, an unusual situation was observed that those participants who easily feel tired (with the oral report), can often only complete the first two or three subtasks. The reason is that the high order single models (S6 and S10) have bigger modeling errors (see Table I). This requests the participant spending more energy to counteract the error; then in the late stage, the participants often become tired and the muscle fatigue contaminates the normal EMG signals [33]. It thus appears that the switching mechanism can improve modeling accuracy and support the participants to complete complex tasks without muscle fatigue.

## C. Movement performance of EMG-based human-robot cooperation and its clinical significance

Overall, although both S3 and MM have training and testing errors, the healthy participants could all complete the tasks with similar tracking accuracy during verifying phase, demonstrating the motion adaptability of healthy people. The improvement of tracking accuracy with the switching mechanism especially the significance in X-axis indicated the superiority of the switched model in motion control accuracy. It also indicates that for experimental verification, the arm movement performance is less affected when using appropriate EMG decoder models. However, since there exist bump transfers between the switching of two subsystems, a further conjecture to be verified is, the bump change can be decreased by tuning initial states of the models, and the participants’ movement performance and control efficiency can be improved.

Under the organization of motor control by the human brain, healthy participants naturally have the ability to adaptively track an intuitional task without [34] or with appropriate assistance. However, when considering the unfinished movements with the two models: S6 and S10, the ‘assistance’ from the robot biassed the motion intention, which prevents the human limb to complete complex tasks.

The decreased muscle activations of the six muscles match well with the previous studies [35]–[37]. This can be explained as healthy participants can voluntarily adjust their muscle activation and force when external assistance are provided [38]. Therefore, compared with the movement during the training phase, the physical effort of the six muscles are reduced when the participants are performing the same movement with the assistance from the cable-based robot.

The effectiveness of continuous assistance provided by myoelectrical controlled rehabilitation robot is confirmed for the elbow and wrist joint rehabilitation of patients after stroke [16], [39]. The newly proposed EMG decoder model with switching mechanism can be generalized to other multi-joint rehabilitation robots for patients who suffer from neurological diseases, such as stroke, cerebral palsy and so on. As the next step, we aim to apply the proposed method to young patients (20-30 yrs) who had a stroke in the previous year and perform high scores in basic motion ability assessments after passive rehabilitation. In addition, these patients should be able to sit on a chair for at least 30 minutes and hold their affected arms for at least 1 minute.

## D. Summary of contribution

The major contribution of this study is the proposed new switching based approach for continuously decoding motion intention from multiple EMG signals and further actively supporting the subject’s movement by taking human participation into account. By introducing the switching mechanism, an EMG decoder model has been built up with multiple EMG signals and multiple robot control signals for a single but complicated task.

The philosophy behind this is to separate the complex motion task into multiple simple subtasks so that each subtask is simple enough to be modeled by a simple linear state-space model. Then, through the help of the switching mechanism, the complex rehabilitation movement can be implemented. Although we separated the complex task into multiple simple subtasks, we still treated the overall task as a whole task and tried to minimize the transient response during switching (i.e., minimized the jerk during switching).

Theoretically, due to the involvement of human central nervous system, decoding a complex task by using a single model is quite complex. Also, due to the complexity of the multiple-input and multiple-output relationship recorded in our study, the decoder model needs to be a high dimensional nonlinear dynamic model and may be time-variant. An alternative solution is to separate the whole complex tracking task into subtasks and build simple models for subtasks. This is similar to spline interpolation, in which a complex input-output relationship is approximated by a set of piecewise polynomial functions.

Furthermore, motivated by the continuity/smoothness requirement at the knots of the spline interpolation, we considered the problem of reducing discontinuity during switching. To the best of the authors knowledge, no report exists in the literature which considers the continuity of output during switching for the decoding of human motion intention.

## V. CONCLUSION

This study investigated the active robot rehabilitation by using multiple-channel EMG signals. EMG decoders based on linear time-invariant state space models, without necessarily having dimension-reduction, were proposed. By introducing a switching mechanism, this approach carves up the complex tracking task into simple subtasks. Then, a switching system model composed of four subsystems were trained. This paper found that the low-order simple models of the EMG decoder based on the switching mechanism are able to cope with the variation of the underlying system while reducing the complexity of the model, and improve the accuracy during model testing and real-time experimental verification. Overall, the decoder with the switching mechanism can predict the voluntary motion intention with high accuracy from multiplechannel EMG signals and help the participants to finish the human-robot cooperation movement with lower muscle efforts and higher task completion rate.

## APPENDIX A

## MODEL ORDER SELECTION CRITERIA

The model order selection is based on preliminary experiments for three subjects in the training and testing phases under the following four criteria: 1) the percentage of the output variations that is reproduced by the model and a higher value means a better model, 2) the singular values (SVs) of a certain covariance matrix constructed from the observed data as a function of the model order which is a measure of how much the n-th component of the state vector contributes to the input-output behavior of the model, 3) considering there are three outputs, if the overall transfer function is strictly proper, the order cannot be lower than 3, and 4) the model order cannot be too high, because it will lead to overfitting with a relatively short period of stimulation (around 5 seconds for each sub-trajectory).

Our findings from the pre-experiments are, 1) with the order growing from 2 to 10, the percentage of the output variations grows, 2) the SVs of models whose order is 1/2/3 are close, and they are all higher than the SVs of models whose order is 4/5/6, and the SVs of models whose order is 7/8/9/10 reveal a similar situation, i.e. $S V _ { 1 - 3 } > S V _ { 4 - 6 } > S V _ { 7 - 1 0 }$ . The SVs of the models whose order is over 10, are lower than 0.1. This result means the components with a higher order have smaller contributions to the behavior of the model. Therefore, the low, middle and high order representatives that we chose are 3, 6 and 10, respectively.

## ACKNOWLEDGMENT

This work was supported by the Australia-China Joint Institute for Health Technology and Innovation established by Sun Yat-sen University and University of Technology Sydney. We also thank the anonymous reviewers whose comments/suggestions helped us improve and clarify this manuscript.

## REFERENCES

[1] M. Lotze, C. Braun, N. Birbaumer, S. Anders, and L. G. Cohen, “Motor learning elicited by voluntary drive,” Brain, vol. 126, no. 4, pp. 866–872, 2003.

[2] R. Riener, M. Guidali, U. Keller, A. Duschau-Wicke, V. Klamroth, and T. Nef, “Transferring armin to the clinics and industry,” Topics in spinal cord injury rehabilitation, vol. 17, no. 1, pp. 54–59, 2011.

[3] S. J. Ball, I. E. Brown, and S. H. Scott, “Medarm: a rehabilitation robot with 5dof at the shoulder complex,” in Advanced intelligent mechatronics, 2007 IEEE/ASME international conference on. IEEE, 2007, pp. 1–6.

[4] S. Masiero, A. Celia, G. Rosati, and M. Armani, “Robotic-assisted rehabilitation of the upper limb after acute stroke,” Archives ofphysical medicine and rehabilitation, vol. 88, no. 2, pp. 142–149, 2007.

[5] D. Zanotto, G. Rosati, S. Minto, and A. Rossi, “Sophia-3: A semiadaptive cable-driven rehabilitation device with a tilting working plane,” IEEE Transactions on Robotics, vol. 30, no. 4, pp. 974–979, 2014.

[6] L. Marchal-Crespo and D. J. Reinkensmeyer, “Review of control strategies for robotic movement training after neurologic injury,” Journal of neuroengineering and rehabilitation, vol. 6, no. 1, p. 20, 2009.

[7] P. M. Rossini and G. Dal Forno, “Integrated technology for evaluation of brain function and neural plasticity,” Physical Medicine and Rehabilitation Clinics, vol. 15, no. 1, pp. 263–306, 2004.

[8] G. Xu, A. Song, and H. Li, “Control system design for an upper-limb rehabilitation robot,” Advanced Robotics, vol. 25, no. 1-2, pp. 229–251, 2011.

[9] M.-K. Chang, “An adaptive self-organizing fuzzy sliding mode controller for a 2-dof rehabilitation robot actuated by pneumatic muscle actuators,” Control Engineering Practice, vol. 18, no. 1, pp. 13–22, 2010.

[10] T. Proietti, V. Crocher, A. Roby-Brami, and N. Jarrasse, “Upper-´ limb robotic exoskeletons for neurorehabilitation: a review on control strategies,” IEEE reviews in biomedical engineering, vol. 9, pp. 4–14, 2016.

[11] C. Loconsole, S. Dettori, A. Frisoli, C. A. Avizzano, and M. Bergamasco, “An EMG-based approach for on-line predicted torque control in robotic-assisted rehabilitation,” in Haptics Symposium (HAPTICS), 2014 IEEE. IEEE, 2014, pp. 181–186.

[12] M. H. Rahman, C. Ochoa-Luna, M. Saad, and P. Archambault, “EMG based control of a robotic exoskeleton for shoulder and elbow motion assist,” J Aut Con Eng, vol. 3, no. 4, 2015.

[13] J. Potvin, R. Norman, and S. McGill, “Mechanically corrected EMG for the continuous estimation of erector spinae muscle loading during repetitive lifting,” European journal of applied physiology and occupational physiology, vol. 74, no. 1-2, pp. 119–132, 1996.

[14] L. Dipietro, M. Ferraro, J. J. Palazzolo, H. I. Krebs, B. T. Volpe, and N. Hogan, “Customized interactive robotic treatment for stroke: EMG-triggered therapy,” IEEE transactions on neural systems and rehabilitation engineering, vol. 13, no. 3, pp. 325–334, 2005.

[15] Y. M. Aung and A. Al-Jumaily, “sEMG based ann for shoulder angle prediction,” Procedia Engineering, vol. 41, pp. 1009–1015, 2012.

[16] R. Song, K.-y. Tong, X. Hu, L. Li et al., “Assistive control system using continuous myoelectric signal in robot-aided arm training for patients after stroke,” IEEE transactions on neural systems and rehabilitation engineering, vol. 16, no. 4, pp. 371–379, 2008.

[17] A. d’Avella, A. Portone, L. Fernandez, and F. Lacquaniti, “Control of fast-reaching movements by muscle synergy combinations,” Journal of Neuroscience, vol. 26, no. 30, pp. 7791–7810, 2006.

[18] B. Lim, S. Ra, and F. C. Park, “Movement primitives, principal component analysis, and the efficient generation of natural motions,” in Robotics and Automation, 2005. ICRA 2005. Proceedings of the 2005 IEEE International Conference on. IEEE, 2005, pp. 4630–4635.

[19] P. Artemiadis and K. Kyriakopoulos, “EMG-based control of a robot arm using low-dimensional embeddings,” IEEE Transactions on Robotics, vol. 2, no. 26, pp. 393–398, 2010.

[20] K. Nizamis, J. Lobo-Prat, A. Q. Keemink, R. Carloni, A. H. Stienen, and B. F. Koopman, “Switching proportional EMG control of a 3d endpoint arm support for people with duchenne muscular dystrophy,” in Rehabilitation Robotics (ICORR), 2015 IEEE International Conference on. IEEE, 2015, pp. 235–240.

[21] P. K. Artemiadis, P. T. Katsiaris, M. V. Liarokapis, and K. J. Kyriakopoulos, “On the effect of human arm manipulability in 3d force tasks: Towards force-controlled exoskeletons,” in Robotics and Automation (ICRA), 2011 IEEE International Conference on. IEEE, 2011, pp. 3784–3789.

[22] P. K. Artemiadis and K. J. Kyriakopoulos, “A switching regime model for the EMG-based control of a robot arm,” IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), vol. 41, no. 1, pp. 53–63, 2011.

[23] J. Yang, H. Su, Z. Li, D. Ao, and R. Song, “Adaptive control with a fuzzy tuner for cable-based rehabilitation robot,” International Journal of Control, Automation and Systems, vol. 14, no. 3, pp. 865–875, 2016.

[24] F. E. Zajac, “Muscle and tendon properties models scaling and application to biomechanics and motor,” Critical reviews in biomedical engineering, vol. 17, no. 4, pp. 359–411, 1989.

[25] D. G. Lloyd and T. F. Besier, “An EMG-driven musculoskeletal model to estimate muscle forces and knee joint moments in vivo,” Journal of biomechanics, vol. 36, no. 6, pp. 765–776, 2003.

[26] A. Huxley, “Muscular contraction.” The Journal ofphysiology, vol. 243, no. 1, pp. 1–43, 1974.

[27] K. Manal and T. S. Buchanan, “A one-parameter neural activation to muscle activation model: estimating isometric joint moments from electromyograms,” Journal of biomechanics, vol. 36, no. 8, pp. 1197– 1202, 2003.

[28] P. K. Artemiadis and K. J. Kyriakopoulos, “EMG-based teleoperation of a robot arm using low-dimensional representation,” in Intelligent Robots and Systems, 2007. IROS 2007. IEEE/RSJ International Conference on. IEEE, 2007, pp. 489–495.

[29] W. E. Larimore, “Canonical variate analysis in identification, filtering, and adaptive control,” in 29th IEEE Conference on Decision and control. IEEE, 1990, pp. 596–604.

[30] J. R. Cram, Cram’s introduction to surface electromyography. Jones & Bartlett Learning, 2011.

[31] M. M. Mukaka, “A guide to appropriate use of correlation coefficient in medical research,” Malawi Medical Journal, vol. 24, no. 3, pp. 69–71, 2012.

[32] E. Cavallaro, J. Rosen, J. C. Perry, S. Burns, and B. Hannaford, “Hill-based model as a myoprocessor for a neural controlled powered exoskeleton arm-parameters optimization,” in Robotics and Automation, 2005. ICRA 2005. Proceedings of the 2005 IEEE International Conference on. IEEE, 2005, pp. 4514–4519.

[33] T. Oberg, “Muscle fatigue and calibration of EMG measurements,”<sup>¨</sup> Journal of Electromyography and Kinesiology, vol. 5, no. 4, pp. 239– 243, 1995.

[34] K. Nazarpour, A. Barnard, and A. Jackson, “Flexible cortical control of task-specific muscle synergies,” Journal ofNeuroscience, vol. 32, no. 36, pp. 12 349–12 360, 2012.

[35] T. Lenzi, S. M. M. De Rossi, N. Vitiello, and M. C. Carrozza, “Intentionbased EMG control for powered exoskeletons,” IEEE transactions on biomedical engineering, vol. 59, no. 8, pp. 2180–2190, 2012.

[36] S. Kwon, Y. Kim, and J. Kim, “Movement stability analysis of surface electromyography-based elbow power assistance,” IEEE Transactions on Biomedical Engineering, vol. 61, no. 4, pp. 1134–1142, 2014.

[37] E. Mastinu, P. Doguet, Y. Botquin, B. Hakansson, and M. Ortiz-Catalan,˚ “Embedded system for prosthetic control using implanted neuromuscular interfaces accessed via an osseointegrated implant,” IEEE transactions on biomedical circuits and systems, vol. 11, no. 4, pp. 867–877, 2017.

[38] K. Suzuki, G. Mito, H. Kawamoto, Y. Hasegawa, and Y. Sankai, “Intention-based walking support for paraplegia patients with robot suit hal,” Advanced Robotics, vol. 21, no. 12, pp. 1441–1469, 2007.

[39] R. Song, K.-y. Tong, X. Hu, and W. Zhou, “Myoelectrically controlled wrist robot for stroke rehabilitation,” Journal of neuroengineering and rehabilitation, vol. 10, no. 1, p. 52, 2013.