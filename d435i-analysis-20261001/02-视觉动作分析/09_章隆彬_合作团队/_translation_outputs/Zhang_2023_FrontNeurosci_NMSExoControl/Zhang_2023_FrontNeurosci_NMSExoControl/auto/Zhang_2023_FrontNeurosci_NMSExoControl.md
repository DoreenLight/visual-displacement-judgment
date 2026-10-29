Check for updates

## OPEN ACCESS

Alois C. Knoll, Technical University of Munich, Germany

Yisen Huang,   
The Chinese University of Hong Kong, China   
Keli Shen,   
National Institute of Advanced Industrial   
Science and Technology (AIST), Japan

Elena M. Gutierrez-Farewik lanie@kth.se

RECEIVED 06 July 2023   
ACCEPTED 11 August 2023   
PUBLISHED 30 August 2023

CITATION   
Zhang L, Zhang X, Zhu X, Wang R and   
Gutierrez-Farewik EM (2023)   
Neuromusculoskeletal model-informed   
machine learning-based control of a knee exoskeleton with uncertainties quantification. Front. Neurosci.   
doi: 10.3389/fnins,2023.1254088

## COPYRIGHT

© 2023 Zhang, Zhang, Zhu, Wang and Gutierrez-Farewik. This is an open-access article distributed under the terms of the Creative Commons Attribution License (CC BY). The use, distribution or reproduction in other forums is permitted, provided the original author(s) and the copyright owner(s) are credited and that the original publication in this journal is cited, in accordance with accepted academic practice. No use, distribution or reproduction is permitted which does not comply with these terms.

# Neuromusculoskeletal model-informed machine learning-based control of a knee exoskeleton with uncertainties quantification

Longbin Zhang<sup>1</sup>, Xiaochen Zhang<sup>1</sup>, Xueyu Zhu<sup>2</sup>, Ruoli Wang<sup>1</sup> and Elena M. Gutierrez-Farewik<sup>1</sup>\*

<sup>1</sup>KTH MoveAbility Lab, Department of Engineering Mechanics, KTH Royal Institute of Technology, Stockholm, Sweden, <sup>2</sup>Department of Mathematics, University of Iowa, Iowa City, IA, United States

Introduction: Research interest in exoskeleton assistance strategies that incorporate the user’s torque capacity is growing rapidly. However, the predicted torque capacity from users often includes uncertainty from various sources, which can have a significant impact on the safety of the exoskeleton-user interface.

Methods: To address this challenge, this paper proposes an adaptive control framework for a knee exoskeleton that uses muscle electromyography (EMG) signals and joint kinematics. The framework predicted the user’s knee flexion/extension torque with confidence bounds to quantify the uncertainty based on a neuromusculoskeletal (NMS) solver-informed Bayesian Neural Network (NMS-BNN). The predicted torque, with a specified confidence level, controlled the assistive torque provided by the exoskeleton through a TCP/IP stream. The performance of the NMS-BNN model was also compared to that of the Gaussian process (NMS-GP) model.

Results: Our findings showed that both the NMS-BNN and NMS-GP models accurately predicted knee joint torque with low error, surpassing traditional NMS models. High uncertainties were observed at the beginning of each movement, and at terminal stance and terminal swing in self-selected speed walking in both NMS-BNN and NMS-GP models. The knee exoskeleton provided the desired assistive torque with a low error, although lower torque was observed during terminal stance of fast walking compared to self-selected walking speed.

Discussion: The framework developed in this study was able to predict knee flexion/extension torque with quantifiable uncertainty and to provide adaptive assistive torque to the user. This holds significant potential for the development of exoskeletons that provide assistance as needed, with a focus on the safety of the exoskeleton-user interface.

##

machine learning, data-driven biomechanical models, inverse dynamics, neuromusculoskeletal modeling, uncertainty quantification

## 1. Introduction

Exoskeletons have enormous potential to enhance movement and to contribute to neuromuscular rehabilitation in persons with motor disorders such as stroke, cerebral palsy, and spinal cord injury (Sartori et al., 2016; Li et al., 2019, 2021; Liu et al., 2019; Zhang et al., 2019). Exoskeleton-assisted rehabilitation training involves the use of control algorithms aimed at improving muscle strength, neuroplasticity, and movement enhancement in users (Fujii et al., 2017). These control strategies can be classified into three types: passive control, triggered passive control, and assist-as-needed control (Marchal-Crespo and Reinkensmeyer, 2009; Meng et al., 2015; Proietti et al., 2016). Passive control refers to a technique in which the exoskeleton is in charge and guides the user to follow predefined trajectories or assistive forces/torques that have been extracted from healthy populations. The user is passive in the movement and does not actively control the exoskeleton. This type of control is often used in the initial stages of therapy to re-acquaint a limb to movement. Triggered passive control is a variant of passive control, where the user initiates the exoskeleton’s assistance. Once activated, the user is again passive in the movement as the exoskeleton moves along pre determined trajectories. This technique is often used to incorporate the brain-machine interface into the control process, providing assistance to individuals with irreversible impairments, such as tetraplegia (Proietti et al., 2016). Assist-as-needed control, also known as “user-in-charge” or “active control,” empowers the user to perform daily tasks with the aid of an exoskeleton. The exoskeleton provides assistance based on the user’s ability and intention to generate torque, with the aim of promoting neuroplasticity and user autonomy. This active control technique is typically applied in persons with residual motor function (Chen et al., 2016; Durandau et al., 2017; Li et al., 2018; Yao et al., 2018). The primary focus of this paper is on active control techniques that seek to supplement the user’s insuficient muscle contributions with assistance from an exoskeleton. Providing torque assistance based on the user’s movement intention requires precise and robust decoding ofmotor function, which can be achieved through recording of underlying neuromuscular activities, such as brain and nerve signals and muscle electromyography (EMG) signals. EMG signals, which capture the electrical excitation of muscles, are a commonly used method for predicting joint torques, as they are easy to obtain and ofer crucial insights into human motion (Sartori et al., 2018; Huang et al., 2019; Mounis et al., 2019).

Joint torque prediction is crucial in the control of exoskeletonassisted rehabilitation and has frequently been achieved through two methods: physics-based neuromusculoskeletal (NMS) modeling and artificial neural networks (ANNs) (Pizzolato et al., 2015, 2019; Zhang et al., 2020, 2021). To improve prediction accuracy, ANNs have been integrated into NMS models in recent research. In our recent study (Zhang et al., 2022), an NMS solver-informed ANN model was developed to estimate ankle joint torque by combining features from an NMS model with a standard ANN, based on measured joint angles and muscle EMG signals during gait and isokinetic motions. This hybrid model was overall more accurate than the NMS or standard ANN models alone, but still showed poor prediction performance in one subject during gait, possibly due to incorporating less informative or misleading input features from the NMS model. This highlights the necessity of quantifying the uncertainty of joint torque predictions for safe and eficient human-exoskeleton interaction; accurate estimation of joint torque is crucial for determining the appropriate level of assistance from an exoskeleton.

A Bayesian Neural Network (BNN) is a well-established type of ANN for making predictions with uncertainties and has great potential in safe and eficient exoskeleton control (Cursi et al., 2021; Wei et al., 2021; Zhong et al., 2021). Unlike conventional ANNs (Cao et al., 2022b; Hu et al., 2022), BNNs incorporate probability distributions to represent prediction uncertainty and provide a probability distribution indicating the likelihood of diferent outcomes (Cursi et al., 2021). This characteristic makes BNNs useful for decision-making in various fields, including biomechanics, meteorology, and robotics. For instance, Zhong et al. (2021) developed a BNN-based framework for predicting the environmental context of lower limb prostheses. The quantified prediction uncertainty could lead to context recognition strategies that enhance reliable decisionmaking, eficient sensor fusion, and improved design of intelligent systems for various applications. Another popular technique for making predictions with uncertainties is the Gaussian Process (GP) (Chen et al., 2013; Yun et al., 2014; Maritz et al., 2018; Guo et al., 2019; Cao et al., 2022a). GP models the output as a Gaussian distribution with mean and covariance parameters, wherein the uncertainty is expressed by the covariance. Liang et al. (2021), for example, developed a GP model to estimate knee joint angles and uncertainties from EMG signals during walking and running movements. As both BNNs and GPs can estimate prediction uncertainty, it is of interest to compare the two methods in the context of safe and eficient human-exoskeleton interaction.

The objective of this study was thus to develop an NMS uncertainty-informed adaptive control framework for a knee exoskeleton. The framework aimed to provide accurate predictions of the user’s knee flexion/extension (F/E) physiological torque, while also quantifying the level of estimation uncertainty. To achieve this, an NMS solver was employed to inform the machine learning models, which would subsequently adjust the assistance level based on the level of uncertainty. Another aim was to compare the predictions with uncertainties from the NMS solver-informed BNN (NMS-BNN) model with those from the NMS solver-informed GP (NMS-GP) model.

## 2. Methods

We developed an adaptive control framework for a knee exoskeleton based on an NMS-BNN model (Figure 1). The NMS-BNN takes two types of inputs: (1) experimental measurements— muscle signals and joint angles, and (2) informative physical features extracted from the underlying NMS solver, such as individual muscle force and joint torque. The NMS-BNN outputs knee joint torque with uncertainty quantification in the form of confidence bounds. The predicted torque with a specified confidence level is then used to control the assistive torque provided by the knee exoskeleton through a TCP/IP data stream. The study results consist of two key components: (1) an analysis of the NMS-BNN model’s prediction accuracy and uncertainty, compared to the traditional NMS model and to the NMS-GP model; (2) an evaluation of the tracking performance of the assistive torque provided by the knee exoskeleton.

![](images/db7040418fb9b66742b6fe9ead04f6dbb861f4be987dd2f2e67445ea2d9ba3fa.jpg)  
Schematic of the adaptive control framework for a knee exoskeleton based on an NMS solver-informed BNN (NMS-BNN) model. The inputs to the NMS-BNN include observed muscle signals and joint angles, as well as physical features derived from the NMS solver such as individual muscle force and joint torque. The NMS-BNN outputs knee joint torque with uncertainty quantification in the form of confidence bounds. The predicted torque with a specified confidence level is then used to control the assistive torque provided by the knee exoskeleton through a TCP/IP data stream.

![](images/8f1582ba9b24732cbb94f9558e70fd4b6e32f6462c92df95e0eb7d86fee0c108.jpg)

![](images/a8e008311273f6f2b040af1a3d79b5afc4ccbdba903e5016d525ef8583eed06f.jpg)  
Experimental setup: subjects equipped with EMG sensors and markers, performed movements in an instrumented motion lab.

## 2.1. Data collection and processing

Eight able-bodied subjects (sex: 4F/4M; height: 168.1 9.4 cm; weight: 65.2 17.8 kg; age: 29 4 years) were recruited. The Swedish Ethical Review Authority (Dnr. 2020-02311) approved this study, and all subjects provided informed written consent documents. All participants were asked to do five movement types (Figure 2), specifically slow walking, normal walking, fast walking, sit-to-stand, and stand-to-sit. During the experiments, each movement was repeated at least ten times. The sequence of movements was randomized.

Surface EMG signals (aktos nano, myon, Schwarzenberg, Switzerland) from vastus medialis (VM), vastus lateralis (VL), rectus femoris (RF), semitendinosus (ST), biceps femoris (BF), gastrocnemius medialis (GM), and gastrocnemius lateralis (GL) of each participant’s one randomly-selected leg were measured at 1,000 Hz. EMGs were post-processed by bandpass filtering (30–300 Hz), rectifying, low pass filtering (6 Hz), and normalizing to the maximum EMG value among all movement trials (Sartori et al., 2016; Pizzolato et al., 2017; Hoang et al., 2018).

Marker trajectories were recorded at 100 Hz using a 3D motion capture system (V16, Vicon, Oxford, UK), with marker placement based on the CGM2.3 model (Leboeuf et al., 2019). Ground reaction forces (GRFs) were measured at 100 Hz with three force plates (AMTI, MA, USA). Kinematics were calculated via inverse kinematics by solving a weighted least square optimization problem to minimize the discrepancy between virtual $x _ { i }$ and measured $x _ { i } ^ { e x p }$ marker trajectories (Lu and O’connor, 1999), Equation (1).

$$
\operatorname* { m i n } _ { \boldsymbol q } ( \sum _ { i } ^ { N } \theta _ { i } \| \boldsymbol x _ { i } ^ { e x p } - \boldsymbol x _ { i } \| ^ { 2 } ) ,\tag{1}
$$

where q represents the generalized coordinates of the model and $\theta _ { i }$ is the weight of ith marker. Kinetics were computed via inverse dynamics by solving for joint torques in the dynamic equations of motion (Pandy, 2001) (Equation 2),

$$
M ( \mathbf { q } ) \ddot { q } + C ( q , \dot { q } ) + \mathbf { G ( q ) } + R ( \mathbf { q } ) \mathbf { F } ^ { \mathbf { m t } } + \mathbf { F _ { e } } = 0\tag{2}
$$

where ${ \bf q } , \dot { q } , \ddot { q }$ are the vector of generalized position, velocity and acceleration, respectively; $M ( \mathbf { q } )$ is mass matrix and $M ( \mathbf { q } ) \ddot { q }$ is a vector of inertial forces and torques; $C ( q , \dot { q } )$ is the vector of centripetal and Coriolis forces and torques; $\mathbf { G } ( \mathbf { q } )$ is the vector of gravitational forces and torques; $R ( \mathbf { q } )$ is the matrix of muscle moment arms; $\mathbf { F } ^ { \mathbf { m t } }$ is a vector of musculotendon forces and $R ( \mathbf { q } ) \mathbf { F } ^ { \mathbf { m t } }$ is the vector of musculotendon torques; $\mathbf { F _ { e } }$ is the vector of external force and torques (i.e., GRFs in this paper). A low-pass fourth-order zero-lag Butterworth filter (6 Hz) was used to filter joint kinematics and kinetics (Winter et al., 1974; Mantoan et al., 2015; Derrick et al., 2020).

## 2.2. EMG-driven neuromusculoskeletal model

The EMG-driven NMS model used in this study was the open-source CEINMS model (Pizzolato et al., 2015) (Figure 3). This model includes four components: musculotendon kinematics, muscle activation dynamics, muscle contraction dynamics, and joint dynamics relationships (Sartori et al., 2011). The musculotendon kinematics component calculates moment arms and musculotendon lengths, while the muscle activation dynamics component computes muscle activation based on the available EMG information. The relationship between EMG excitation, e(t), and neural activation, u(t), is expressed in Equation (3) (Lloyd and Besier, 2003):

$$
u ( t ) = \alpha \cdot e ( t - \tau ) - \beta _ { 1 } \cdot u ( t - 1 ) - \beta _ { 2 } \cdot u ( t - 2 )\tag{3}
$$

where α is the muscle gain parameter, $\beta _ { 1 }$ and $\beta _ { 2 }$ are the recursive parameters $[ \beta _ { 1 } = C _ { 1 } + C _ { 2 } , \beta _ { 2 } = C _ { 1 } \cdot C _ { 2 } ,$ with $| C _ { 1 } | < 1 , | C _ { 2 } | < 1 ,$ and $\alpha - \beta _ { 1 } - \beta _ { 2 } = 1$ for a stable solution (Lloyd and Besier, 2003; Buchanan et al., 2004; Pizzolato et al., 2015)], and τ is the electromechanical delay. Muscle activation, a(t), is described by Equation (4):

$$
a ( t ) = { \frac { e ^ { A u ( t ) } - 1 } { e ^ { A } - 1 } }\tag{4}
$$

where A is the shape factor (Buchanan et al., 2004; Hoang et al., 2018).

The muscle contraction dynamics component calculates the muscle force with a Hill-type muscle model, represented by Equation (5):

$$
F = F _ { 0 } ^ { m } [ F _ { a l } ( l ) \cdot F _ { \nu } ( \nu ) \cdot a + F _ { p l } ( l ) + d _ { p } \cdot \nu ] c o s ( \theta )\tag{5}
$$

where $F _ { 0 } ^ { m }$ is the muscle’s maximum isometric force, $F _ { a l } ( l )$ describes the relationship between active muscle force and fiber length $l , F _ { p l } ( l )$ describes the relationship between passive muscle force and fiber length, $F _ { \nu } ( \nu )$ describes the relationship between muscle force and fiber contraction velocity $\nu , \theta$ is the fiber pennation angle, and $d _ { p }$ is the muscle damping parameter.

Finally, the joint dynamics component computes joint torque by multiplying muscle forces and moment arms.

The parameters were calibrated as outlined by Pizzolato et al. (2015), with optimal fiber length and tendon slack length adjusted within 15% of initial values, coeficients $C _ { 1 }$ and $C _ { 2 }$ limited to values between 1 and 1, and parameter A bounded between $( - 3 , 0 )$ . The maximum isometric force was determined using a strength coeficient with a range of 0.5 to 2.5. The optimization process focused on minimizing the error between predicted and actual joint torques (computed via inverse dynamics) during the calibration procedure. This optimization task was achieved by employing a simulated annealing algorithm, which iteratives to refine the parameter values. The algorithm was executed until the average change in the objective function’s value reached a tolerance level of $1 0 ^ { - 5 }$

## 2.3. NMS-BNN model

The NMS-BNN models consist of an input neural layer, 3 hidden layers, and an output neural layer. The inputs, $\begin{array} { r l } { \mathbf { x } } & { { } = } \end{array}$ $[ x _ { 1 } , x _ { 2 } , \ldots , x _ { m } ] ^ { T }$ where $m = 2 1$ , were augmented with two types of features (Figure 4): (1) Muscle EMG signals and joint kinematics (knee F/E angle) from a 3D motion capture system, and (2) physical features such as muscle forces and NMS torque from an underlying NMS solver, to increase the model’s accuracy by providing more information about the system being modeled. Each hidden layer has 40 neurons. The estimated knee torque with uncertainties bound was determined in the output layer.

In BNNs, weights are treated as probability distributions rather than as single point estimates, as in standard neural networks (Figure 4). These distributions are used to reflect the uncertainty in weights and predictions. The posterior probability of weights, $P ( W | X )$ , is computed using Bayes theorem as follows (Equation 6):

![](images/917c3f745db97baea11e2cc002ff35fac2c168375046e2c92b6fe6ea858b2f66.jpg)  
Schematic structure of an EMG-driven neuromusculoskeletal model with four components: the musculotendon kinematics component calculates musculotendon lengths and moment arms; the muscle activation dynamics component uses the EMG information to compute muscle activation; the muscle contraction dynamics component, predicts musculotendon force using musculotendon length and muscle activation based on a Hill-type muscle model; and finally, the joint dynamics component computes joint torques with musculotendon forces and moment arms as inputs.

![](images/2fae4cd863a38bf905ab290bf57b00b25b0ebb251c511bf13f3ee214f2ebe853.jpg)  
Architecture for the NMS-BNN model. The NMS-BNN models consist of an input neural layer, 3 hidden layers, and an output neural layer. The inputs, $\mathbf { x } = [ x _ { 1 } , x _ { 2 } , \ldots , x _ { m } ] ^ { T }$ where $m = 2 1$ , were augmented with two types of features: (1) Muscle EMG signals and joint kinematics from a 3D motion capture system, and (2) Physical features such as muscle forces and NMS torque from an underlying NMS solver, to increase the model’s accuracy by providing more information about the system being modeled. Each hidden layer has 40 neurons. The estimated knee torque with uncertainties bound was determined in the output layer. Weights are treated as probability distributions rather than as single-point estimates as in standard neural networks. These distributions are used to reflect the uncertainty in weights and predictions

$$
P ( W | X ) = { \frac { P ( X | W ) P ( W ) } { P ( X ) } }\tag{6}
$$

where X is the data, P(X W) represents the likelihood of the data given weights W, and P(W) is the prior probability of the weights.

The denominator, P(X), represents the probability of the data, which is obtained by integrating the likelihood of the data given weights and the prior probability of weights over all possible values of weights, represented by Equation (7):

$$
P ( X ) = \int P ( X \vert W ) P ( W ) d W\tag{7}
$$

The BNN package TensorBNN, developed by Kronheim et al. (2022), was used in this study. The hyper-parameters of the BNN models were determined using a “coarse-to-fine” random search method (Bergstra and Bengio, 2012). During training, the mean square error was used as the loss function and a batch size of 32 was applied. Three hidden layers were included. Each hidden layer has 40 neurons with a tanh activation function. A Gaussian likelihood with a standard deviation of 0.1 was employed. Prior to sampling, the model was pre-trained using the AMSGrad optimizer with learning rates of 0.01, 0.001, and 0.0001, with a patience of 10. To obtain a point cloud of the posterior density of neural network parameters, Hamiltonian Monte Carlo (HMC) sampling was used to compute the likelihood function. HMC is a Markov chain Monte Carlo method that leverages a fictitious potential energy function derived from the posterior density of the neural network parameters. Numerical approximation was conducted using the leapfrog method, with the number of leapfrog steps and step size determining the distance traveled to the next proposed point. The number of steps for the HMC hyper-parameter sampler remained constant, while the step size was adapted using the Dual-Averaging algorithm based on the acceptance rate of the sample during 80% of the burn-in period. It is worth noting that selecting suitable values for the number of steps and step size can be challenging, and TensorBNN incorporates the parameter adapter algorithm to automatically optimize these parameters (Wang et al., 2013; Kronheim et al., 2022).

## 2.4. NMS-GP model

The NMS-GP model was developed using the same input data as the NMS-BNN model, which comprised experimentally obtained muscle signals and joint kinematics, as well as physical features such as muscle forces and NMS torque extracted from the underlying NMS solver. The NMS-GP modelf(x) was specified by a mean function µ(x) and a covariance function $k ( \mathbf { x } , \mathbf { x ^ { ' } } )$ ), as expressed in equation (8) (Rasmussen, 2004).

$$
f ( \mathbf { x } ) \sim G P ( \mu ( \mathbf { x } ) , k ( \mathbf { x } , \mathbf { x ^ { ' } } ) )\tag{8}
$$

where the mean function µ(x) provides an estimate of the expected value of the model at a given input, while the covariance function, also referred to as the kernel function, quantifies the similarity between two inputs. The Gaussian process model ofers various kernel functions to capture the underlying structure of data. Among them, the radial basis function kernel (RBF) is widely used due to its smoothness and infinite diferentiability, as shown in Equation (9),

$$
k ( \mathbf { x } , \mathbf { x } ^ { \prime } ) = \exp ( - \frac { | \mathbf { x } - \mathbf { x } ^ { \prime } | ^ { 2 } } { 2 l ^ { 2 } } )\tag{9}
$$

where l controls the length-scale of the kernel, and $| { \bf x } - { \bf x } ^ { \prime } |$ is the Euclidean distance between inputs x and x′.

Another popular kernel function is the Matern kernel, which is a generalization of the RBF kernel and is defined as Equation (10),

$$
k ( \mathbf x , \mathbf x ^ { \prime } ) = \frac { 2 ^ { 1 - \nu } } { \Gamma ( \nu ) } \left( \frac { \sqrt { 2 \nu } } l | \mathbf x - \mathbf x ^ { \prime } | \right) ^ { \nu } K _ { \nu } \left( \frac { \sqrt { 2 \nu } } l | \mathbf x - \mathbf x ^ { \prime } | \right)\tag{10}
$$

where ν determines the smoothness of the kernel, $K _ { \nu }$ is the modified Bessel function, Ŵ(ν) is the Gamma function, and l controls the scale of the kernel.

For modeling noise in data, the White Noise Kernel, as shown in Equation (11), is commonly used,

$$
k ( \mathbf { x } , \mathbf { x } ^ { \prime } ) = \sigma ^ { 2 } \delta ( \mathbf { x } - \mathbf { x } ^ { \prime } )\tag{11}
$$

where $\sigma ^ { 2 }$ is the noise variance parameter that determines the amplitude of the noise, and $\delta ( { \bf x } - { \bf x } ^ { \prime } )$ is the Dirac delta function. This function equals one when $\mathbf { x } = \mathbf { x } ^ { \prime }$ and zero otherwise, ensuring that the kernel function is non-zero only at the diagonal ofthe input space.

The Linear kernel is another widely used kernel that models a linear relationship between the input and output variables. Specifically, it can be formulated as depicted in Equation (12):

$$
k ( \mathbf { x } , \mathbf { x } ^ { \prime } ) = \sigma ^ { 2 } \mathbf { x } ^ { T } \mathbf { x } ^ { \prime }\tag{12}
$$

where $\sigma ^ { 2 }$ represents the variance parameter.

In Gaussian process modeling, the combination of diferent kernel functions can improve the performance of the model. In this regard, we selected a combination of Matern, White Noise, and Linear kernels after conducting extensive testing to obtain the most accurate predictions. The Matern kernel was used to capture non-linear patterns, while the White Noise kernel accounted for measurement errors and uncertainty, and the Linear kernel modeled linear relationships between the input and output variables. Hyperparameters such as length scale, signal variance, noise variance, and others were optimized during the training process to enhance the model’s performance. In the Matern kernel, we set ν to 3/2, and the length scale was bounded between the range of (0.01, 200), with variance confined within the range of $( 1 0 ^ { - 3 }$ , 10<sup>5</sup>). Similarly, the White Noise kernel had a noise variance bounded between (0.03, 100), while the Linear kernel had a variance range of $( 1 0 ^ { - 3 }$ , 10<sup>5</sup>).

## 2.5. Knee exoskeleton

The knee exoskeleton hardware consists of a drive unit (Gen.1, Maxon, Switzerland), a 3D-printed thigh-shank frame, and thigh and shank straps (Orliman 94260, Spain). The drive unit features a brushless DC motor (EC90 flat), a MILE encoder with 4,096 counts per turn, a three-stage planetary gearbox with an 18-bit SSI absolute encoder, and an EPOS4 position controller. The drive unit is capable of providing a continuous torque of 54 Nm and a maximum torque of 120 Nm on a 20% duty cycle. The system can operate on a DC power supply ranging from 10 to 50vV and its actuation speed can reach up to 22 rpm.

## 2.6. Evaluation protocol

## 2.6.1. Joint torque prediction

The prediction accuracy of the knee joint torque for NMS, NMS-GP, and NMS-BNN models was investigated. The uncertainty of the predicted torque by NMS-GP and NMS-BNN models was also analyzed. The prediction accuracy and uncertainty quantification was compared in five cases: $G a i t _ { s l o w } , G a i t _ { s e l f } , G a i t _ { f a s t } ,$ SitToStand, and StandToSit, which were trained using data from each movement separately. NMS-GP and NMS-BNN models were trained using 80% of the data and evaluated on the remaining data, while NMS models were calibrated using three trials of each movement and tested on the same data as NMS-GP and NMS-BNN models. The input data from each trial consisted of approximately 100 time-series data points and 21 dimensions.

![](images/c28c47c3259942688497c2283f7aae8fa3e91dafa517daccef3e37f3f7dbdc80.jpg)  
The distributions of NRMSE between estimated and measured/actual knee joint torques across subjects in NMS, NMS-GP, and NMS-BNN models during five daily activities. The violin plots depict the probability distributions of NRMSE using kernel density plots, and the box plots represent the minimum, lower quartile, median, upper quartile, and maximum values of NRMSE. A significant di erence between any two models is indicated by an asterisk ( ), based on paired t-test (for normally distributed data) or Wilcoxon signed-rank test (for non-normally distributed data) with Bonferroni correction.

Two prediction error metrics were evaluated: the Normalized Root Mean Square Error (NRMSE, E ) and the Root Mean Square Error (RMSE, E<sub>RMS</sub>). A low prediction error indicated a high prediction accuracy. NRMSE was calculated by dividing the RMSE (between the predicted and actual torque) by the range of joint torque observed during the corresponding motion:

$$
E _ { R M S } = \sqrt { \frac { 1 } { N } \sum _ { n = 1 } ^ { N } ( y _ { P , n } - y _ { n } ) ^ { 2 } }\tag{13}
$$

$$
E _ { N R M S } = \frac { E _ { R M S } } { ( y _ { m a x } - y _ { m i n } ) } \times 1 0 0 \%\tag{14}
$$

where $y _ { n }$ and $y _ { P } , n$ are the measured/actual and predicted torque respectively; and $y _ { m i n }$ and $y _ { m a x }$ are the minimum and maximum measured torque in corresponding movements. The RMSE and NRMSE were calculated for each subject and the average was obtained across eight subjects. The results section presents the average values of RMSE and NRMSE.

The normality of the data distribution was evaluated using Shapiro-Wilk tests $( p < 0 . 0 5$ significance level). To determine the diferences among the NRMSEs and RMSEs estimated by the three approaches, pairwise comparisons were performed using either paired t-tests for normally distributed data or Wilcoxon signedrank tests for non-normally distributed data, both with Bonferroni correction applied and significance level of $\dot { p } < 0 . 0 5$

![](images/bed5379ffbed88320d77e36ca0f3d468a19d0865c5a53c28f93fb0697a379c89.jpg)  
The distributions of RMSE between estimated and measured/actual knee joint torques across subjects in NMS, NMS-GP, and NMS-BNN models during five daily activities. The violin plots depict the probability distributions of NRMSE using kernel density plots, and the box plots represent the minimum, lower quartile, median, upper quartile, and maximum values of NRMSE. A significant di erence between any two models is indicated by an asterisk ( ), based on paired t-test (for normally distributed data) or Wilcoxon signed-rank test (for non-normally distributed data) with Bonferroni correction.

The uncertainty ofthe predicted torque by NMS-GP and NMS-BNN models was quantified by using a 95% confidence level (CL), which means that there is a 95% probability that the true value of the function being modeled falls within the predicted interval. A high uncertainty value indicates low confidence in the predicted value.

## 2.6.2. Exoskeleton assistive torque tracking performance

We also evaluate the tracking performance of the knee exoskeleton’s assistive torque provided by the adaptive control framework during five daily activities by using the two metrics: NRMSE and RMSE (between desired and actual torque provided by the knee exoskeleton). The assistance level $A _ { L }$ of the knee torque provided by the adaptive control framework is adapted/determined by the uncertainties U quantified by the NMS-BNN model, as described by the following equation:

$$
A _ { L } = \left\{ \begin{array} { l l } { 0 . 8 } & { \mathrm { i f ~ } U < 0 . 0 5 } \\ { 0 . 6 } & { \mathrm { i f ~ } 0 . 0 5 \le U < 0 . 1 } \\ { 0 . 4 } & { \mathrm { i f ~ } 0 . 1 \le U < 0 . 1 5 } \\ { 0 . 3 } & { \mathrm { i f ~ } 0 . 1 5 \le U < 0 . 2 } \\ { 0 . 1 } & { \mathrm { i f ~ } U \ge 0 . 2 } \end{array} \right.\tag{15}
$$

![](images/f445af1b03fd661d3fa7150bddec739e59bb21e409bd8c81b6e21c8d730d178b.jpg)

A  
(A) The uncertainty quantification of predicted knee joint torque by the NMS-GP and NMS-BNN models across subjects during five daily activities, as the mean 1 standard deviation of all subjects. The uncertainty was quantified using a 95% confidence level, meaning that there is a 95% probability that the true value falls within the predicted interval. A high uncertainty value indicates low confidence in the prediction. (B) One example of measured knee flex/extension torques (Nm/kg) by inverse dynamics (ID) and predicted values by both NMS and NMS-GP models during five daily activities. The standard deviation in the NMS-GP models highlights the uncertainties from the expected mean value. (C) One example of predicted knee flex/extension torques by NMS-BNN models was presented and compared with the same example from ID and NMS models in (B). The standard deviation in the NMS-BNN models also highlights the uncertainties from the expected mean value.  
A  
![](images/9cfab67e4e03bbb8bf8cd08b05d50e3e29680e814f272671fdcb5b4a4ea027f1.jpg)

B  
![](images/e40a732b43fe82bc19bee7c6944a9848505404c9a0c7da00ee571796c91e3850.jpg)  
The tracking error of the knee exoskeleton’s assistive torque provided by the adaptive control framework during five daily activities, presented as (A) NRMSE and (B) RMSE. The violin plots illustrate the probability distributions of prediction error through kernel density plots, and the box plots depict the minimum, first quartile, median, third quartile, and maximum values of the prediction error.

## 3. Results

## 3.1. Joint torque prediction

Overall, both NMS-BNN and NMS-GP models accurately predicted knee joint torque with relatively low error (RMSE: NMS-$\mathrm { G P } \leq 0 . 0 5 \mathrm { N m / k g } ,$ $\mathrm { N M S - B N N } \leq 0 . 0 7$ Nm/kg; NRMSE: NMS-GP 5.9%, $\mathrm { N M S - B N N } \leq 6 . 8 \% )$ . The errors were considerably lower than those of NMS models (RMSE: 0.14 Nm/kg, $\mathrm { N R M S E } \colon \leq \ 1 8 . 3 \% .$ Figures 5, 6).

The NRMSE prediction error for the NMS-GP and NMS-BNN models was significantly lower than that of the NMS models in all cases, except the StandToSit case $( G a i t _ { s l o w } \colon p < 0 . 0 1$ and $p < 0 . 0 1 ;$ $G a i t _ { s e l f } \colon p < 0 . 0 1$ and $p \ < \ 0 . 0 1 ; G a i t _ { f a s t } \colon p \ < \ 0 . 0 1$ and $\mathit { p } \prec \ : 0 . 0 1$ SitToStand: $\textit { p } < \ 0 . 0 1$ and $\textit { p } < \ 0 . 0 1$ , StandToSit: $p \ = \ 0 . 0 8$ and $\begin{array} { r } { p \ = \ 1 . 4 5 ; } \end{array}$ Figure 5). Similar findings were also observed in the RMSE.

Among the NRMSE predicted by NMS models in five cases, the NRMSE in the StandToSit case was the lowest ( 7.2%). No significant diferences were observed in the StandToSit case among NMS, NMS-GP, and NMS-BNN models (NMS: 7.2%, NMS-GP: $\leq 5 . 5 \% ; \mathrm { N M S - B N N } ; \leq 6 . 8 \% )$

Both the NMS-GP and NMS-BNN models had relatively high uncertainties in the predicted knee torque at the beginning of each movement, particularly in the $G a i t _ { s e l f }$ case (Figure 7A). In the NMS-GP model, high uncertainties were observed during terminal stance and terminal swing in the $G a i t _ { s e l f }$ case. On the other hand, the NMS-BNN model had high uncertainties during terminal stance, initial swing, and terminal swing in all gait speeds.

The predicted torque by the NMS models had a poorer agreement with the measured torque compared to the NMS-GP and NMS-BNN models (Figures 7B, C). Relatively high ofsets were observed at the beginning of each movement in NMS models.

## 3.2. Exoskeleton assistive torque tracking performance

Overall, the knee exoskeleton accurately provided the required assistive torque with relatively low error (RMSE: 0.06 Nm/kg, $\mathrm { { N R M S E } \colon \leq \quad 5 . 6 \% }$ , Figure 8). Among the five movements, the NRMSE was evenly distributed among all subjects for walking movements, while one outlier was noted in both sit-to-stand and stand-to-sit movements. The sit-to-stand movement had the highest tracking error among the five movements.

Generally, the actual assistive torque provided by the knee exoskeleton matched the desired torque well (Figure 9). However, it is important to note that limited torque was provided at the start of the sit-to-stand movement. Additionally, relatively low assistive torque was observed during the terminal stance of fast walking compared to self-selected speed walking.

## 4. Discussion

We developed an NMS-BNN-based adaptive control framework for a knee exoskeleton using muscle EMG signals and joint kinematics. We also compared the predictions with uncertainties from the NMS-BNN model with those from the NMS-GP model. We observed both NMS-BNN and NMS-GP models showed accurate predictions of knee joint torque with low error, outperforming traditional NMS models, indicating the benefits of incorporating NMS features into machine learning models. High uncertainties, however, were observed at the beginning of each movement and at terminal stance and terminal swing in the self-selected speed walking in both NMS-BNN and NMS-GP models. The knee exoskeleton provided the desired assistive torque accurately, with a relatively low error. Lower levels of torque were observed during terminal stance in fast walking compared to self-selected walking speed. The level of assistive torque was determined and adjusted based on the uncertainty in the NMS-BNN predictions, promoting the safety of the exoskeleton-user interface.

Incorporating the user’s physiological joint torque into exoskeleton control strategies has in recent years become feasible, and has vast potential to improve task performance and rehabilitation outcomes. Among common techniques for predicting joint torque, EMG-driven NMS models require expertise and complex calibration, whereas machine learning models are more accessible but considered as black boxes (Hoang et al., 2018; Ezati et al., 2019; Soleimani and Nazerfard, 2021). To improve prediction accuracy, ANNs have been integrated into NMS models, allowing the advantages of both approaches to be leveraged. However, ensuring the safety and eficiency of exoskeleton control is also crucial, particularly when using predicted torque as inputs for the exoskeleton. To address this, this study integrated NMS with machine learning models with uncertainty quantification for joint torque prediction. As mentioned earlier, Both BNNs and GP models can provide predictions with associated uncertainties. While BNN models incorporate Bayesian methods to quantify the uncertainty in predictions, GP models are based on Gaussian processes and provide a probabilistic model for predictions with uncertainties. In the current study, we also compared the predictions with uncertainties between NMS-BNN and NMS-GP models. We found both NMS-BNN and NMS-GP models accurately predicted knee joint torque with relatively low error (RMSE: NMS- $\cdot \mathrm { G P } ~ \leq ~ 0 . 0 5$ Nm/kg, NMS-BNN $\le ~ 0 . 0 7$ Nm/kg; NRMSE: $\mathrm { { N M S - G P } \leq 5 . 9 \% }$ , NMS-BNN 6.8%), and were found to be superior to traditional NMS models $( \mathrm { R M S E } \leq 0 . 1 4 ~ \mathrm { N m / k g } ,$ NRMSE 18.3%). The results are attributed to the addition of machine learning layers, which further train the model by minimizing the error between measured and predicted joint torque.

The quantification of uncertainty by either the NMS-BNN or NMS-GP models can supply the exoskeleton controller with valuable data for decision-making, which could enhance safety in the exoskeleton-user interaction. For instance, we observed high uncertainties at the beginning of each movement in both NMS-BNN and NMS-GP models (Figure 7). This is likely due to the physical characteristics adopted from NMS models, which show a noticeable ofset at the start ofeach movement. In NMS models, two prior time steps of neural activation from each MTU are required to calculate muscle neural activation (Zhang et al., 2022). At the beginning of a cycle, these past two neural activation values are not yet obtainable and are approximated using EMG signals from two previous time steps, potentially leading to initial inaccuracies in predicted torque. Furthermore, high uncertainties were observed during the terminal stance and terminal swing in self-selected speed walking in both NMS-BNN and NMS-GP models. This may be attributed to the of transitions between the stance and swing phases of gait.

![](images/8d9f04d0361452b9b09ef043f303ec2631d0ba7da5ac3965e3fd7ae0567068e5.jpg)  
One example of the desired and actual assistive torque provided by the knee exoskeleton during five daily activities.

The knee exoskeleton provided the desired assistive torque accurately, with a relatively low error (RMSE: 0.06 Nm/kg, NRMSE: 5.6%, Figure 8). The assistive torque was achieved through current control in the motor, a widely used closed loop control technique (Zhang et al., 2018; Azocar et al., 2020; Nuckols et al., 2021). The current control system aims to maintain a consistent current in the motor, even as its speed and load conditions vary. Precise control over the motor’s torque production can be achieved through current control, though accuracy may be influenced by factors such as the quality of current sensing and the speed of the control loop’s response. To estimate output torque, the control system uses the measured current as feedback, as the current is proportional to the torque produced by the motor (Azocar et al., 2020). This allows the control system to determine the amount of torque produced and adjust the exoskeleton accordingly. We observed lower levels oftorque during the terminal stance of fast walking compared to self-selected walking speed (Figure 9). This discrepancy may be due to the increased uncertainties present during fast walking, which in turn led to a lower level of assistance torque being assigned according to our control strategy (Equation 15). It is worth noting that an outlier was observed in the sit-to-stand and stand-to-sit movements (Figure 8). This deviation may be attributed to the limited torque capacity of the exoskeleton at the beginning of the sit-to-stand movement and at the end of the stand-to-sit movement (Figure 9).

This study focused on evaluating the feasibility of the NMS-BNN framework by implementing a basic current control strategy. The objective was to assess the overall viability of the framework. However, future studies are necessary to investigate more advanced control techniques, such as impedance control. The current control strategy may result in less smooth assistive torque. Therefore, in future studies, we recommend incorporating an improved control strategy that takes into account both uncertainties and the closest points of predicted torque to enhance the smoothness and improve the user-exoskeleton interface. Furthermore, while our current study centers on the knee joint, it is important to note that the approach can be adapted and extended to other joints as well. Additionally, it is worth mentioning that the maximum torque that can be generated by the system is 54 Nm, which may also impact the smoothness of the assistive torque. Thus, this should be considered in future control strategies. It should be noted that this study did not involve testing the performance of the NMS-BNN-based adaptive framework on real users for practical applications. Further research is essential to address this issue and ascertain the practicality of the framework.

## 5. Conclusion

In this study, we proposed an NMS-BNN-based adaptive control framework for a knee exoskeleton that uses muscle EMG signals and joint kinematics. The NMS-BNN model combines a traditional NMS model with modern machine learning techniques and includes uncertainty quantification. The proposed framework also measures uncertainty in predictions and incorporates it into the control design to ensure safety of the exoskeleton-user interface. We also compared the performance of the NMS-BNN model to an NMS-GP model, which also predicts uncertainties. Detailed information relating to how to combine traditional models with machine learning models with uncertainties can provide useful guidance for designing exoskeleton control strategies.

## Data availability statement

The raw data supporting the conclusions of this article will be made available by the authors, without undue reservation.

## Ethics statement

The studies involving humans were approved by Swedish Ethical Review Authority. The studies were conducted in accordance with the local legislation and institutional requirements. The participants provided their written informed consent to participate in this study. Written informed consent was obtained from the individual(s) for the publication of any potentially identifiable images or data included in this article.

## Author contributions

LZ: Conceptualization, Data curation, Formal analysis, Investigation, Methodology, Software, Validation, Visualization, Writing—original draft, Writing—review and editing. XZha: Writing—review and editing, Formal analysis, Methodology, Investigation. XZhu: Methodology, Supervision, Writing—review and editing. RW: Methodology, Supervision, Writing—review and editing. EG-F: Conceptualization, Data curation, Funding acquisition, Methodology, Project administration, Resources, Supervision, Writing—review and editing.

## Funding

We would like to acknowledge financial support from Promobilia Foundation (A22078, 18014, 21302, and 18200), Swedish Research

Azocar, A. F., Mooney, L. M., Duval, J. F., Simon, A. M., Hargrove, L. J., and Rouse, E. J. (2020). Design and clinical implementation of an open-source bionic leg. Nat. Biomed. Eng. 4, 941–953. doi: 10.1038/s41551-020-00619-3

Bergstra, J., and Bengio, Y. (2012). Random search for hyper-parameter optimization. J. Mach. Learn. Res. 13, 1–25.

## References

Buchanan, T. S., Lloyd, D. G., Manal, K., and Besier, T. F. (2004). Neuromusculoskeletal modeling: estimation of muscle forces and joint moments and movements from measurements of neural command. J. Appl. Biomech. 20, 367–395. doi: 10.1123/jab.20.4.367

Cao, H., Chen, G., Li, Z., Feng, Q., Lin, J., and Knoll, A. (2022a). Eficient grasp detection network with Gaussian-based grasp representation for robotic manipulation. IEEE/ASME Trans. Mechatron. 28, 1384–1394. doi: 10.1109/TMECH.2022.3224314

Cao, H., Chen, G., Li, Z., Hu, Y., and Knoll, A. (2022b). Neurograsp: multimodal neural network with Euler region regression for neuromorphic vision-based grasp pose estimation. IEEE Trans. Instrument. Meas. 71, 1–11. doi: 10.1109/TIM.2022.3179469

Chen, B., Ma, H., Qin, L.-Y., Gao, F., Chan, K.-M., Law, S.-W., et al. (2016). Recent developments and challenges of lower extremity exoskeletons. J. Orthopaed. Transl. 5, 26–37. doi: 10.1016/j.jot.2015.09.007

All claims expressed in this article are solely those of the authors and do not necessarily represent those of their afiliated organizations, or those of the publisher, the editors and the reviewers. Any product that may be evaluated in this article, or claim that may be made by its manufacturer, is not guaranteed or endorsed by the publisher.

Chen, N., Qian, Z., Nabney, I. T., and Meng, X. (2013). Wind power forecasts using Gaussian processes and numerical weather prediction. IEEE Trans. Power Syst. 29, 656–665. doi: 10.1109/TPWRS.2013.2282366

Cursi, F., Modugno, V., Lanari, L., Oriolo, G., and Kormushev, P. (2021). Bayesian neural network modeling and hierarchical MPC for a tendon-driven surgical robot with uncertainty minimization. IEEE Robot. Automat. Lett. 6, 2642–2649. doi: 10.1109/LRA.2021.3062339

Derrick, T. R., van den Bogert, A. J., Cereatti, A., Dumas, R., Fantozzi, S., and Leardini, A. (2020). ISB recommendations on the reporting of intersegmental forces and moments during human motion analysis. J. Biomech. 99, 1–10. doi: 10.1016/j.jbiomech.2019.109533

Durandau, G., Farina, D., and Sartori, M. (2017). Robust real-time musculoskeletal modeling driven by electromyograms. IEEE Trans. Biomed. Eng. 65, 556–564. doi: 10.1109/TBME.2017.2704085

Ezati, M., Ghannadi, B., and McPhee, J. (2019). A review of simulation methods for human movement dynamics with emphasis on gait. Multibody Syst. Dyn. 47, 265–292. doi: 10.1007/s11044-019-09685-1

## Publisher’s note

Fujii, K., Abe, T., Kubota, S., Marushima, A., Kawamoto, H., Ueno, T., et al. (2017). The voluntary driven exoskeleton hybrid assistive limb (HAL) for postoperative training of thoracic ossification of the posterior longitudinal ligament: a case report. J. Spinal Cord Med. 40, 361–367. doi: 10.1080/10790268.2016.1142056

Council (2018-04902 and 2018-00750), and Simons Foundation (504054).

Guo, H., Meng, Z., Huang, Z., Kang, L. W., Chen, Z., Meghjani, M., et al. (2019). “Safe path planning with gaussian process regulated risk map,” in 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) (Macau: IEEE), 2044–2051.

The authors declare that the research was conducted in the absence of any commercial or financial relationships that could be construed as a potential conflict of interest.

## Conflict of interest

Hoang, H. X., Pizzolato, C., Diamond, L. E., and Lloyd, D. G. (2018). Subject-specific calibration of neuromuscular parameters enables neuromusculoskeletal models to estimate physiologically plausible hip joint contact forces in healthy adults. J. Biomech. 80, 111–120. doi: 10.1016/j.jbiomech.2018.08.023

Hu, Y., Chen, G., Li, Z., and Knoll, A. (2022). Robot policy improvement with natural evolution strategies for stable nonlinear dynamical system. IEEE Trans. Cybernet.53, 4002-4014. doi: 10.1109/TCYB.2022.3192049

Huang, Y., Song, R., Argha, A., Savkin, A. V., Celler, B. G., and Su, S. W. (2019). Continuous description of human 3d motion intent through switching mechanism. IEEE Trans. Neural Syst. Rehabil. Eng. 28, 277–286. doi: 10.1109/TNSRE.2019.2949203

Kronheim, B., Kuchera, M. P., and Prosper, H. B. (2022). Tensorbnn: Bayesian inference for neural networks using tensorflow. Comput. Phys. Commun. 270, 1–8. doi: 10.1016/j.cpc.2021.108168

Leboeuf, F., Baker, R., Barré, A., Reay, J., Jones, R., and Sangeux, M. (2019). The conventional gait model, an open-source implementation that reproduces the past but prepares for the future. Gait Post. 69, 235–241. doi: 10.1016/j.gaitpost.2019.04.015

Li, Z., Huang, B., Ye, Z., Deng, M., and Yang, C. (2018). Physical human–robot interaction of a robotic exoskeleton by admittance control. IEEE Trans. Indus. Electron. 65, 9614–9624. doi: 10.1109/TIE.2018.282 1649

Li, Z., Li, G., Wu, X., Kan, Z., Su, H., and Liu, Y. (2021). Asymmetric cooperation control of dual-arm exoskeletons using human collaborative manipulation models. JEEE Trans, Cybernet, 52, 12126–12139, doi: 10.1109/TCYB.2021.3113709

Li, Z., Yuan, Y., Luo, L., Su, W., Zhao, K., Xu, C., et al. (2019). Hybrid brain/muscle signals powered wearable walking exoskeleton enhancing motor ability in climbing stairs activity. IEEE Trans. Med. Robot. Bionics 1, 218–227. doi: 10.1109/TMRB.2019.2949865

Liang, J., Shi, Z., Zhu, F., Chen, W., Chen, X., and Li, Y. (2021). Gaussian process autoregression for joint angle prediction based on sEMG signals. Front. Public Health 9, 685596. doi: 10.3389/fpubh.2021.685596

Liu, Z., Zhong, B., Zhong, W., Guo, K., and Zhang, M. (2019). “A new trajectory determination method for robot-assisted ankle ligament rehabilitation,” in 2019 41st Annual International Conference of the IEEE Engineering in Medicine and Biology Society (Berlin: IEEE), 5390–5393.

Lloyd, D. G., and Besier, T. F. (2003). An EMG-driven musculoskeletal model to estimate muscle forces and knee joint moments in vivo. J. Biomech. 36, 765–776. doi: 10.1016/S0021-9290(03)00010-1

Lu, T.-W., and O’connor, J. (1999). Bone position estimation from skin marker co-ordinates using global optimisation with joint constraints. J. Biomech. 32, 129–134.

Mantoan, A., Pizzolato, C., Sartori, M., Sawacha, Z., Cobelli, C., and Reggiani, M. (2015). MOtoNMS: a MATLAB toolbox to process motion data for neuromusculoskeletal modeling and simulation. Source Code Biol. Med. 10, 1–14. doi: 10.1186/s13029-015-0044-4

Marchal-Crespo, L., and Reinkensmeyer, D. J. (2009). Review of control strategies for robotic movement training after neurologic injury. J. Neuroeng. Rehabil. 6, 1–15. doi: 10.1186/1743-0003-6-20

Maritz, J., Lubbe, F., and Lagrange, L. (2018). A practical guide to Gaussian process regression for energy measurement and verification within the Bayesian framework. Energies 11, 1–12. doi: 10.3390/en11040935

Meng, W., Liu, Q., Zhou, Z., Ai, Q., Sheng, B., and Xie, S. S. (2015). Recent development of mechanisms and control strategies for robot-assisted lower limb rehabilitation. Mechatronics 31, 132–145. doi: 10.1016/j.mechatronics.2015.04.005

Mounis, S. Y. A., Azlan, N. Z., and Sado, F. (2019). Assist-as-needed control strategy for upper-limb rehabilitation based on subject's functional ability. Meas. Control 52, 1354–1361. doi: 10.1177/0020294019866844

Nuckols, R. W., Lee, S., Swaminathan, K., Orzel, D., Howe, R. D., and Walsh, C. J. (2021). Individualization of exosuit assistance based on measured muscle dynamics during versatile walking. Sci. Robot. 6, 1–11. doi: 10.1126/scirobotics.abj1362

Pandy, M. G. (2001). Computer modeling and simulation of human movement. Annu. Rev. Biomed. Eng. 3, 245–273. doi: 10.1146/annurev.bioeng.3.1.245

Pizzolato, C., Lloyd, D. G., Sartori, M., Ceseracciu, E., Besier, T. F., Fregly, B. J., et al. (2015). CEINMS: a toolbox to investigate the influence of diferent neural control solutions on the prediction of muscle excitation and joint moments during dynamic motor tasks. J. Biomech. 48, 3929–3936. doi: 10.1016/j.jbiomech.2015.09.021

Pizzolato, C., Reggiani, M., Modenese, L., and Lloyd, D. (2017). Real-time inverse kinematics and inverse dynamics for lower limb applications using opensim. Comput. Methods Biomech. Biomed. Eng. 20, 436–445. doi: 10.1080/10255842.2016.1240789

Pizzolato, C., Saxby, D. J., Palipana, D., Diamond, L. E., Barrett, R. S., Teng, Y. D., et al. (2019). Neuromusculoskeletal modeling-based prostheses for recovery after spinal cord injury. Front. Neurorobot. 13, 97. doi: 10.3389/fnbot.2019.00097

Proietti, T., Crocher, V., Roby-Brami, A., and Jarrasse, N. (2016). Upper-limb robotic exoskeletons for neurorehabilitation: a review on control strategies. IEEE Rev Biomed. Eng. 9, 4–14. doi: 10.1109/RBME.2016.2552201

Rasmussen, C. E. (2004). Gaussian processes in machine learning. Adv. Lectures Mach. Learn. 3176, 63–71. doi: 10.1007/978-3-540-28650-9\_4

Sartori, M., Durandau, G., Došen, S., and Farina, D. (2018). Robust simultaneous myoelectric control of multiple degrees of freedom in wrist-hand prostheses by real-time neuromusculoskeletal modeling. J. Neural Eng. 15, 1–15. doi: 10.1088/1741-2552/aae26b

Sartori, M., Lloyd, D. G., and Farina, D. (2016). Neural data-driven musculoskeletal modeling for personalized neurorehabilitation technologies. IEEE Trans. Biomed. Eng. 63, 879–893. doi: 10.1109/TBME.2016.25 38296

Sartori, M., Reggiani, M., Lloyd, D. G., and Pagello, E. (2011). “A neuromusculoskeletal model of the human lower limb: towards EMG-driven actuation

of multiple joints in powered orthoses,” in 2011 IEEE International Conference on Rehabilitation Robotics (Zürich: IEEE), 1–6.

Soleimani, E., and Nazerfard, E. (2021). Cross-subject transfer learning in human activity recognition systems using generative adversarial networks. Neurocomputing 426, 26–34. doi: 10.1016/j.neucom.2020.10.056

Wang, Z., Mohamed, S., and Freitas, N. (2013). “Adaptive Hamiltonian and Riemann manifold Monte Carlo,” in International Conference on Machine Learning (Georgia: PMLR), 1462–1470.

Wei, W., Kaiming, Y., Yu, Z., Yuyang, Q., Chenhui, W., and Min, L. (2021). Walking speed estimation from a wearable insole pressure system embedded with an accelerometer using Bayesian neural network. J. Eng. Sci. Med. Diagnost. Ther. 4, 1–7. doi: 10.1115/1.4049964

Winter, D. A., Sidwall, H. G., and Hobson, D. A. (1974). Measurement and reduction of noise in kinematics of locomotion. J. Biomech. 7, 157–159.

Yao, S., Zhuang, Y., Li, Z., and Song, R. (2018). Adaptive admittance control for an ankle exoskeleton using an emg-driven musculoskeletal model. Front. Neurorobot. 12, 16. doi: 10.3389/fnbot,2018.00016

Yun, Y., Kim, H.-C., Shin, S. Y., Lee, J., Deshpande, A. D., and Kim, C. (2014). Statistical method for prediction of gait kinematics with Gaussian process regression. J. Biomech. 47, 186–192. doi: 10.1016/j.jbiomech.2013.09.032

Zhang, L., Li, Z., Hu, Y., Smith, C., Farewik, E. M. G., and Wang, R. (2020). Ankle joint torque estimation using an EMG-driven neuromusculoskeletal model and an artificial neural network model. IEEE Trans. Automat. Sci. Eng. 18, 564–573. doi: 10.1109/TASE.2020.3033664

Zhang, L., Zhu, X., Farewik, E. M. G., and Wang, R. (2021). “Estimation of ankle dynamic joint torque by a neuromusculoskeletal solver-informed NN model,” in 2021 6th IEEE International Conference on Advanced Robotics and Mechatronics (ICARM) (Chongqing: IEEE), 75–80.

Zhang, L., Zhu, X., Gutierrez-Farewik, E. M., and Wang, R. (2022). Ankle joint torque prediction using an NMS solver informed-ANN model and transfer learning. IEEE J. Biomed. Health Inform. 26, 5895–5906. doi: 10.1109/JBHI.2022.3207313

Zhang, M., McDaid, A., Veale, A. J., Peng, Y., and Xie, S. Q. (2019). Adaptive trajectory tracking control of a parallel ankle rehabilitation robot with joint-space force distribution. IEEE Access 7, 85812–85820. doi: 10.1109/ACCESS.2019.292 5182

Zhang, T., Tran, M., and Huang, H. (2018). Design and experimental verification of hip exoskeleton with balance capacities for walking assistance. IEEE/ASME Trans. Mechatron. 23, 274–285. doi: 10.1109/TMECH.2018.279 0358

Zhong, B., Silva, R. L., Li, M., Huang, H., and Lobaton, E. (2021). Environmental context prediction for lower limb prostheses with uncertainty quantification. IEEE Trans. Automat. Sci. Eng. 18, 458–470. doi: 10.1109/TASE.2020.2993399