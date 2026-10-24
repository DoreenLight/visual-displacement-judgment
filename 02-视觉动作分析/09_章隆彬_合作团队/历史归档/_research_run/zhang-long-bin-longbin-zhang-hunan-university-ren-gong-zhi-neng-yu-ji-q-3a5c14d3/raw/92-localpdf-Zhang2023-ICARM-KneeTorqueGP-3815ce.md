<!-- source: https://doi.org/10.1109/icarm58088.2023.10218934 -->
<!-- title: Knee Joint Torque Prediction with Uncertainties by a Neuromusculoskeletal Solver-informed Gaussian Process Model -->
<!-- fetched: 2026-10-01T03:49:39Z -->
<!-- via: local_paper_fulltext(pdftotext -layout) -->
<!-- note: 用户于 2026-10-01 从学校数据库下载的原文 PDF：papers/Zhang2023-ICARM：NMS 求解器与高斯过程的膝关节力矩预测/NMS_Solver_Informed_GP_Knee_Joint_Torque_2023_ICARM.pdf -->

2023 8th International Conference on
                                                                                                                                                               Advanced Robotics and Mechatronics
                                                                                                                                                               July 8-10, Sanya, China

                                                                                                                                                                           Knee Joint Torque Prediction with Uncertainties by a
                                                                                                                                                                       Neuromusculoskeletal Solver-informed Gaussian Process Model
                                                                                                                                                                         Longbin Zhang, Xiaochen Zhang, Xueyu Zhu, Ruoli Wang, and Elena M. Gutierrez-Farewik

                                                                                                                                                                  Abstract— Research interest in exoskeleton assistance strate-                and where it is needed, thus improving the patient’s ability to
                                                                                                                                                               gies that incorporate the user’s torque capacity is rapidly                     perform specific tasks and movements, and has demonstrated

2023 International Conference on Advanced Robotics and Mechatronics (ICARM) | 979-8-3503-0017-8/23/$31.00 ©2023 IEEE | DOI: 10.1109/ICARM58088.2023.10218934
                                                                                                                                                               growing, yet uncertainty in predicted torque capacity can                       potential to improve rehabilitation outcomes [13]–[16]. One
                                                                                                                                                               significantly impact the user-exoskeleton interface safety. In
                                                                                                                                                               this paper, we estimated knee flexion/extension torques by                      common assistance-as-needed strategy estimates the user’s
                                                                                                                                                               using a neuromusculoskeletal (NMS) solver-informed Gaussian                     physiological joint torques from EMG signals and uses them
                                                                                                                                                               process (NMS-GP) model with muscle electromyography signals                     as controller input.
                                                                                                                                                               and joint kinematics as model inputs. The NMS-GP model                             EMG-driven neuromusculoskeletal (NMS) model-based
                                                                                                                                                               combined the NMS and GP models by integrating valuable                          methods and machine learning are two popular techniques to
                                                                                                                                                               features from an NMS solver into a GP model. The NMS-
                                                                                                                                                               GP model was used to predict knee joint torque in daily                         predict a user’s physiological joint torques based on muscle
                                                                                                                                                               activities with uncertainty quantification. The activities included             excitation signals from EMG sensors [17]–[19]. Generally,
                                                                                                                                                               slow walking, self-selected speed walking, fast walking, sit-to-                NMS models require domain expertise to establish rela-
                                                                                                                                                               stand, and stand-to-sit. Model performance, defined as low                      tionships between various variables, for instance, muscle
                                                                                                                                                               prediction error between the model’s predicted torque and                       excitation vs. activation, muscle force vs. length, muscle
                                                                                                                                                               measured torques from inverse dynamics computations, of both
                                                                                                                                                               the NMS-GP and NMS models was analyzed. We found that                           force vs. velocity, and muscle-tendon kinematics, and may
                                                                                                                                                               prediction error was significantly lower in NMS-GP models                       involve complex calibration procedures [19]–[22]. On the
                                                                                                                                                               than in NMS models. We observed relatively high uncertainties                   contrary, machine learning can estimate joint torques without
                                                                                                                                                               in the predicted knee torque at the beginning of each movement,                 explicitly describing the complex relationships between vari-
                                                                                                                                                               particularly in self-selected speed walking. High uncertainties                 ables. However, machine learning is generally considered a
                                                                                                                                                               were also found during terminal stance and swing in self-
                                                                                                                                                               selected speed walking. Compared to other torque prediction                     black-box approach, in that it does not provide understanding
                                                                                                                                                               methods, the proposed NMS-GP model not only provides an                         of the underlying mechanisms of the physical system. This
                                                                                                                                                               accurate joint torque prediction but also a measure of the                      may make it unreliable when predicting data that lies outside
                                                                                                                                                               uncertainty. Our study showed that the NMS-GP model has                         the training distribution, especially in cases where the train-
                                                                                                                                                               a large potential in control strategy design for rehabilitation                 ing data is limited [23]. Physics-based NMS methods and
                                                                                                                                                               exoskeletons and to enhance the overall user experience.
                                                                                                                                                                                                                                               machine learning methods each have their own advantages
                                                                                                                                                                                     I. I NTRODUCTION                                          and disadvantages. We recently proposed a hybrid model
                                                                                                                                                                  Robotic exoskeletons are used more and more often in                         that combines physical features from an NMS solver with
                                                                                                                                                               the rehabilitation of individuals with motor disorders such                     a standard artificial neural network (ANN) [24] to improve
                                                                                                                                                               as stroke, cerebral palsy, and spinal cord injury [1]–[7].                      the accuracy of the ANN alone to predict ankle torques
                                                                                                                                                               These conditions often cause muscle weakness, coordination                      during gait and isokinetic motions. We found that the hybrid
                                                                                                                                                               impairment, and limited mobility, making it challenging for                     model had superior performance compared to both the NMS
                                                                                                                                                               individuals to perform daily activities. Robotic exoskele-                      and ANN models, thus highlighting the advantages gained
                                                                                                                                                               tons can provide assistance during rehabilitation training,                     by integrating informative features from the NMS model
                                                                                                                                                               allowing individuals with motor impairment to perform daily                     into an ANN. However, there was one exception to this
                                                                                                                                                               activities and/or exercises that would otherwise be chal-                       finding; we observed that our hybrid-ANN model performed
                                                                                                                                                               lenging or impossible [8]–[10]. During exoskeleton-assisted                     worse than the standard ANN during self-selected speed
                                                                                                                                                               rehabilitation training, incorporating users’ biosignals such as                walking. It is possible that the incorporated input features
                                                                                                                                                               electromyography (EMG) into control strategies has gained                       from the NMS model were representative or even misleading
                                                                                                                                                               significant attention, as it actively involves the user and                     due to differences in muscle coordination patterns between
                                                                                                                                                               thereby stimulates motor relearning [11], [12]. The strategy                    calibration and testing motions or uncertainties in EMG input
                                                                                                                                                               of this type of rehabilitation training, called assistance-as-                  signals. It is thus important to quantify the uncertainty of a
                                                                                                                                                               needed control strategies, is to provide assistance only when                   model’s predictions, particularly when it is to be used for the
                                                                                                                                                                                                                                               control of exoskeletons.
                                                                                                                                                                  *This work was supported by Promobilia Foundation (A22078, 18014,               Gaussian processes (GPs) have been shown to be powerful
                                                                                                                                                               18200 and 21302), Swedish Research Council (2018-00750 and 2018-
                                                                                                                                                               04902), and Simons Foundation (504054).                                         tools for making predictions with uncertainties in a variety
                                                                                                                                                                  L. Zhang, X. Zhang, R. Wang and E.M. Gutierrez-Farewik are with the          of applications, such as joint kinematics prediction [25],
                                                                                                                                                               KTH MoveAbility Lab, Department of Engineering Mechanics, KTH Royal             weather forecasting [26], and robot motion planning [27].
                                                                                                                                                               Institute of Technology, Stockholm, Sweden (e-mail: lanie@kth.se)
                                                                                                                                                                  X. Zhu is with Department of Mathematics, University of Iowa, Iowa           A GP is a type of probabilistic model that can provide a
                                                                                                                                                               City, Iowa.                                                                     measure of uncertainty in its predictions [28]. Liang et al.

                                                                                                                                                               979-8-3503-0017-8/23/$31.00 ©2023 IEEE                                    1035
                                                                                                                                                                 Authorized licensed use limited to: South-Central Minzu University. Downloaded on October 01,2026 at 03:45:09 UTC from IEEE Xplore. Restrictions apply.

[25] proposed a GP model to predict knee joint angles during                         Joint Angle     Musculotendon Kinematics     Moment Arms
                                                                                                                                                  Joint Dynamics
                                                                                                      (Musculoskeletal Model)
walking and running based on EMG signals. They demon-
strated the ability of the GP model to accurately predict knee                                     Musculotendon Length                                              Joint Torque

angles while providing uncertainty information. Similarly,                                                                        Musculotendon
                                                                                                    Muscle Contraction Dynamics
Yun et al. [29] applied a GP model to predict human gait                                             (Hill-type muscle Model)        Forces

patterns using only body parameters as input. They found the                                              Muscle Activation

GP model to be a cost-effective and subject-specific method                          EMG Signal      Muscle Activation Dynamics
for estimating gait kinematics. The GP model allowed for
a comprehensive understanding of the correlation between
                                                                                Fig. 1. Schematic illustration of an NMS model structure. The muscle
and uncertainty among anthropometric parameters and gait                        activation dynamics element computes muscle activation based on EMG
kinematics. To the best of our knowledge, Gaussian processes                    signals (muscle excitation). The musculotendon kinematics element calcu-
have not been applied to predict knee joint torques using                       lates musculotendon lengths and moment arms. Musculotendon forces are
                                                                                then computed by using the muscle contraction dynamics element and a
measured EMG signals as inputs. However, this appears to                        Hill-type muscle model, with muscle activation and musculotendon lengths
be a promising technique for quantifying the uncertainty                        as inputs. Finally, the joint dynamics element computes joint torques with
associated with joint torque prediction.                                        musculotendon forces and moment arms as inputs.

   Our aim was to estimate knee flexion/extension (F/E)
torques during different everyday movements by using an
NMS solver-informed Gaussian process (NMS-GP) model
that incorporates measured muscle electromyography signals                                                                    Muscle
                                                                                                                              Force
and knee joint angles as model inputs. We hypothesized that
                                                                                                                               NMS
the NMS-GP model would yield a more accurate knee joint                                                                       Torque     Gaussian
                                                                                                       NMS Model
torque prediction than the NMS model alone. A secondary                                                                                  Process                   Joint Torque
aim was to quantify the model’s uncertainty in predicted knee                                                                            Models
joint torques during different movements.                                                                        EMG Signal

                                                                                                                 Kinematics
                            II. M ETHODS
A. Data Collection and Processing
   Eight able-bodied subjects (sex: 4F/4M; height: 168.1±9.4
                                                                                Fig. 2.     Overview of the joint torque estimation using NMS-informed
cm; weight: 65.2±17.8 kg; age: 29±4 years) were recruited.                      Gaussian process models. The inputs of NMS-informed GP models were
The Swedish Ethical Review Authority approved this study                        augmented with two kinds of features: (1) experimentally measured muscle
(Dnr. 2020-02311), and all subjects provided written in-                        signals and 3D joint kinematics, and (2) informative physical features,
                                                                                specifically muscle forces and NMS torque, from the underlying NMS
formed consents. All participants were equipped with sensors                    solver.
for 3D movement capture and asked to perform five differ-
ent everyday movements, specifically slow walking, normal
walking at a self-selected pace, fast walking, sit-to-stand, and                motion [32], [33]. A low-pass fourth-order zero-lag Butter-
stand-to-sit. Each recorded movement was repeated at least                      worth filter (6 Hz) was used to filter joint kinematics and
ten times.                                                                      kinetics [34]–[36].
   Surface EMG signals (Myon aktos/Cometa systems, Mi-
lan, Italy) of rectus femoris (RF), vastus medialis (VM),                       B. Neuromusculoskeletal Solver-informed Gaussian Process
vastus lateralis (VL), biceps femoris (BF), semitendinosus                      Regression Model
(ST), gastrocnemius medialis (GM), and gastrocnemius lat-                          In our study, we used an open-source EMG-informed
eralis (GL) of each participant’s one randomly selected leg                     NMS model developed by Pizzolato et al. [21] consist-
were measured at 1000 Hz. The raw EMGs signals were                             ing of four elements (Fig. 1): muscle activation dynamics,
post-processed as follows: they were bandpass filtered (30-                     musculotendon kinematics, muscle contraction dynamics,
300 Hz), rectified, low pass filtered (6 Hz), and normalized                    and joint dynamics [37]. The muscle activation dynamics
to the maximum EMG value among all movement trials [2],                         element computes muscle activation based on EMG signals
[20], [30].                                                                     (muscle excitation). The musculotendon kinematics element
   Marker trajectories were recorded using a 3D motion                          calculates musculotendon lengths and moment arms. The
capture system (100 Hz, V16, Vicon, Oxford, UK), with                           muscle contraction dynamics element computes musculo-
marker placement according to the CGM2.3 model [31].                            tendon forces using a Hill-type muscle model with muscle
Ground reaction forces were measured with three force plates                    activation and musculotendon lengths as inputs. Finally,
(1000 Hz, AMTI, MA, USA). Kinematics (joint angles)                             the joint dynamics element computes joint torques with
were obtained using an inverse kinematics approach, which                       musculotendon forces and moment arms as inputs.
involved minimizing the discrepancy between modeled and                            The inputs x = [x1 , x2 , . . . , xn ]T (n = 21) of NMS-
measured marker trajectories. Inverse dynamics was used to                      informed GP models were augmented with two types of
calculate joint torques that satisfy the dynamic equations of                   features (Fig. 1): (1) experimentally measured muscle signals

  Authorized licensed use limited to: South-Central Minzu University. Downloaded on October 01,2026 at 03:45:09 UTC from IEEE Xplore. Restrictions apply.

and joint kinematics from a 3D motion capture system,                           where σ 2 is the variance parameter.
and (2) informative physical features - muscle forces and                          In practice, it is common to combine different kernel
NMS-based torques, from the underlying NMS solver. These                        functions to improve the performance of a Gaussian process
features were used to enhance the accuracy of the model per-                    model. This is known as kernel composition and can be
formance by providing more information about the modeled                        achieved by adding or multiplying different kernels together.
system. The NMS-GP model f (x) is characterized by a mean                       This allows the model to capture various underlying struc-
                                                ′
function µ(x) and a covariance function k(x, x ), as shown                      tures in the data and improve the overall accuracy of the
in Eq. (1) [38].                                                                predictions. After experimenting with various kernel func-
                                               ′                                tions, we adopted a combination of Matern, White Noise,
                 f (x) ∼ GP (µ(x), k(x, x ))                           (1)
                                                                                and Linear kernels in our GP model, based on trial and
The mean function µ(x) is used to estimate the expected                         error testing; this combination produced the most accurate
value of the function at a given input, and can be defined as                   predictions for the data set. The Matern kernel allowed us
Eq. (2):                                                                        to capture any underlying non-linear patterns in the data,
                                                                                while the White Noise kernel helped to account for any
                        µ(x) = E[f (x)]                                (2)      measurement errors or other sources of uncertainty. The
The covariance function, also known as the kernel function,                     Linear kernel, on the other hand, helped to model the linear
represents the similarity between two input variables. It is                    relationship between the inputs and outputs.
used to estimate the uncertainty of the prediction and is                          Gaussian process regression was performed in Python
defined as Eq. (3) [39]:                                                        using the GaussianProcessRegressor library and kernel func-
                                                                                tions from the gaussian process module in the scikit-learn
      k(x, x′ ) = E[(f (x) − µ(x))(f (x′ ) − µ(x′ ))]                  (3)      package. During the training process, hyperparameters asso-
   There are various kernel functions available in the Gaus-                    ciated with the GP model, including length scale, signal vari-
sian process model. One of the most commonly used func-                         ance, noise variance, and others, were adjusted to optimize its
tions is the radial basis function kernel (RBF), a smooth                       performance. Specifically, in the Matern kernel, the positive
and infinitely differentiable function that can capture the                     parameter ν was set to 3/2. The length scale was bounded
underlying structure of the data well. It is defined as Eq.                     between (0.01, 200) and variance was bounded between
(4)                                                                             (10−3 , 105 ). For the White Noise kernel, noise variance was
                                                                                bounded between (0.03, 100) and for the Linear kernel,
                                 |x − x′ |2                                     variance was bounded between (10−3 , 105 ).
                k(x, x′ ) = exp(−           )               (4)
                                    2l2
where l is a hyperparameter that controls the length-scale of                   C. Evaluation Protocol
the kernel, and |x − x′ | is the Euclidean distance between                        We investigated the accuracy of knee joint torque pre-
two inputs x and x′ .                                                           diction based on both NMS and NMS-GP models, defined
   Another popular kernel function is the Matern kernel,                        here as the error between predicted torque and measured
which is a generalization of the RBF kernel and is defined                      torques from inverse dynamics, and analyzed the uncertainty
as Eq. (5):                                                                     quantification by NMS-GP models. The prediction accuracy
                   √             !ν         √             !
                                                                                with uncertainty quantification was evaluated in five cases:
      ′     21−ν      2ν       ′              2ν       ′
k(x, x ) =               |x − x | Kν             |x − x |   (5)                 Gaitslow , Gaitself , Gaitf ast , SitT oSta, and StaT oSit;
            Γ(ν)      l                       l
                                                                                where a separate NMS-GP model was trained for each move-
where ν is a positive parameter that determines the smooth-                     ment. In all cases, NMS-GP models were trained using 80%
ness of the kernel, Kν is the modified Bessel function, Γ(ν)                    of the recorded data and were subsequently evaluated using
is the Gamma function, and l is the length-scale parameter                      the remaining data, whereas NMS models were calibrated
that controls the scale of the kernel.                                          using three trials of each movement and tested on the same
   The White Noise Kernel is another popular kernel, used                       trials as the NMS-GP testing data.
here for modeling the noise in the data, as shown in Eq. (6):                      The torque estimation error of each model was evaluated
                                                                                using the normalized root mean square error (NRMSE)
                   k(x, x′ ) = σ 2 δ(x − x′ )                          (6)      EN RM S , wherein a low estimation error implies high predic-
where σ 2 is the noise variance parameter that controls the                     tion accuracy. NRMSE was defined as the root mean square
amplitude of the noise, and δ(x − x′ ) is the Dirac delta                       error ERM S (between the predicted and measured torque)
function, which is equal to one when x = x′ and zero                            divided by the range of observed joint torque from inverse
otherwise. It ensures that the kernel function is non-zero only                 dynamics in corresponding motions:
at the diagonal of the input space, where x = x′ .                                                         v
                                                                                                           u
                                                                                                           u1 X   N
   The Linear kernel is another commonly used kernel that
                                                                                             ERM S = t              (yp,n − yn )2         (8)
models a linear relationship between the inputs and the                                                      N n=1
outputs, as shown in Eq. (7):
                                                                                                                   ERM S
                                                                                             EN RM S      =                    × 100%                   (9)
                      k(x, x′ ) = σ 2 xT x′                            (7)                                      (ymax − ymin )

  Authorized licensed use limited to: South-Central Minzu University. Downloaded on October 01,2026 at 03:45:09 UTC from IEEE Xplore. Restrictions apply.

                                                                    NMS                                                                             NMS
                                *                                   NMS-GP                                           *                              NMS-GP

 Prediction Error of Knee F/E                                                         Prediction Error of Knee F/E
                                      *                                                                                     *

                                                 *                                                                                  *
                                                              *                                                                              *           *
                                                                          *

     Torque, NRMSE (%)                                                                    Torque, NRMSE (%)

                                                                                     Fig. 4. RMSE of the predicted knee joint torques across subjects using
Fig. 3. NRMSE of predicted knee joint torque across subjects using NMS               NMS and NMS-GP models. The violin plots show the probability distri-
and NMS-GP models. The violin plots show the probability distributions               butions of NRMSE using kernel density plots, and the box plots represent
of NRMSE using kernel density plots, and the box plots represent the                 the minimum, lower quartile, median, upper quartile, and maximum values
minimum, lower quartile, median, upper quartile, and maximum values                  of RMSE during five daily activities. An asterisk (*) indicates a significant
of NRMSE in five daily activities. An asterisk (*) indicates a significant           difference between the two models.
difference between the two models.

where yn and yp,n are the measured/actual and predicted                              than in the other movements using NMS models (SitT oSta:
torque respectively; ymin is the minimum and ymax is the                             ≤ 9.7% and StaT oSit: ≤ 7.2%) (Fig. 3). The NRMSEs
maximum measured torque in corresponding movements.                                  predicted by NMS-GP models were generally lower in all
Each subject’s NRMSE was calculated, and the median                                  five movements (≤ 5.9%). Similar trends were also observed
value across eight participants was then determined. Data                            in the RMSE (Fig. 4).
distribution normality was evaluated using Shapiro-Wilk                                 It is worth noting that the uncertainties of predicted
tests (p < 0.05 significance level). Pairwise comparisons                            knee torque were relatively high at the beginning of each
between NRMSEs of torque estimation obtained from the                                movement, particularly in self-selected speed walking (Fig.
two approaches were then performed, using paired t-tests                             5 (a)). High uncertainties were also observed during terminal
for data with a normal distribution or Wilcoxon signed-rank                          stance and swing in self-selected speed walking.
tests for non-normally distributed data (p < 0.05).
                                                                                                                         IV. D ISCUSSION
   The uncertainty of the predicted torque was quantified by
using a 95% confidence level (CL), indicating that there                                We proposed an NMS-GP model to estimate knee flex-
is a 95% probability that the true value of the function                             ion/extension torques using measured muscle electromyog-
being modeled falls within the predicted interval. A high                            raphy signals and joint kinematics. Our results showed that
uncertainty value indicates a low confidence in the predicted                        the prediction error was significantly lower in the NMS-
value.                                                                               GP model than in the traditional NMS model, supporting
                                                                                     the hypothesis that integrating features of NMS model into
                                    III. R ESULTS                                    machine learning models can leverage the benefits of the
   Overall, compared to NMS models, the NRMSE of knee                                individual model. Additionally, the NMS-GP model quanti-
torque prediction was significantly lower in NMS-GP models                           fied uncertainty in torque predictions. The torque prediction
(Gaitslow : p < 0.01, Gaitself : p < 0.01, Gaitf ast : p <                           uncertainty was relatively high at the beginning of each
0.01, SitT oSta: p < 0.01, and StaT oSit:p = 0.03; Fig. 3).                          movement, particularly in self-selected speed walking. High
Similar findings were also observed in the RMSE (Gaitslow :                          uncertainties were also observed during terminal stance and
p < 0.01, Gaitself : p < 0.01, Gaitf ast : p < 0.01,                                 swing in self-selected speed walking. The ability to not
SitT oSta: p < 0.01, and StaT oSit: p = 0.04; Fig. 4).                               only predict joint torques accurately but also to compute
Compared to NMS-GP models, the predicted torque with                                 the uncertainties of those predictions has very important
NMS models had a worse agreement with measured torque                                potential implications in certain applications, such as en-
(Fig. 5 (b)) and relatively high offsets at the beginning of                         suring both the efficacy and the safety of exoskeletons. By
each movement.                                                                       incorporating these uncertainties into the control strategy, we
   Among the five movements, the NRMSE of knee torque                                can better manage risks and ensure a safe and comfortable
prediction in walking movements (Gaitslow : ≤ 18.3%,                                 user-exoskeleton interface.
Gaitself : ≤ 17.5%, and Gaitf ast : ≤ 14.2%) was higher                                 NMS-GP models with uncertainty quantification were

       Authorized licensed use limited to: South-Central Minzu University. Downloaded on October 01,2026 at 03:45:09 UTC from IEEE Xplore. Restrictions apply.

 Uncertainties Quantification of
                                         Normal Walking   Self Selected Speed Walking    Fast Walking            Sit-to-stand                  Stand-to-sit

 Predicted Knee Torque (95% CL)            %Gait                     %Gait                     %Gait                  %Cycle                        %Cycle
                                                                                                (a)

          Knee Torque (Nm/Kg)
                                   Ext

                                   Flex

                                           %Gait                    %Gait                      %Gait                 %Cycle                         %Cycle
                                                     ID                                         NMS                                     NMS-GP
                                                                                                (b)

Fig. 5. (a) Computed uncertainty of predicted knee joint torque by the NMS-GP model across subjects during five daily activities, shown as mean ±
1 standard deviation of all subjects and trials. The prediction uncertainty was quantified with a 95% confidence level (CL, illustrated by colored shade),
indicating that there is a 95% probability that the true value of the modelled function falls within the predicted interval. A high uncertainty value indicates
low confidence in the predicted value. (b) One example of measured (by inverse dynamics (ID)) and predicted knee flex/extension torques (Nm/kg) with
both NMS and NMS-GP models during five daily activities. The shaded area in NMS-GP models illustrates the uncertainties in the expected mean value.

found to accurately predict knee joint torques with relatively                              account in exoskeleton controller design.
low error (RMSE ≤ 0.05 Nm/kg, NRMSE ≤ 5.9%), and                                               We used only a single machine learning model and NMS
performed better than conventional NMS models (RMSE ≤                                       model in this study; further research involving other ma-
0.14 Nm/kg, NRMSE ≤ 18.3%). These findings corroborate                                      chine learning approaches and NMS models is necessary to
those from our previous study [22] in which we applied                                      generalize our findings. In addition, future studies should
a hybrid NMS-ANN model to predict ankle joint torques                                       focus on applying joint torque prediction with uncertainty
during gait and isokinetic motions. In that study, we com-                                  quantification in exoskeleton control, which is crucial for
bined physical features from an NMS model with a standard                                   achieving realistic implementation. Our approach was tested
ANN and found that the NMS-ANN model performed well                                         on gait at different speeds, sit-to-stand, and stand-to-sit
in most cases, with a few exceptions. However, the hybrid                                   movements; further testing and validation are required to
NMS-ANN model was only evaluated at the ankle joint.                                        determine its applicability in a wider range of activities.
In this study, we further explored other machine learning                                   Other study limitations are that the subjects recruited were
models, specifically Gaussian process models that include                                   relatively homogeneous in terms of age and overall health
uncertainty quantification in predictions. We observed that                                 status. To generalize these findings to other study popula-
NMS-GP models tend to have high prediction uncertainties                                    tions, it is essential to conduct further research involving a
at the beginning of each movement. This could be attributed                                 diverse group of subjects with a broader range of ages and
to the physical features adopted from NMS models at the                                     neuromuscular functions.
beginning of each movement, where a large offset was also
observed. The NMS model requires the neural activation                                                             V. C ONCLUSION
from two previous time steps to compute the muscle neural
                                                                                                In this study, we proposed an NMS-GP model to estimate
activation of each musculotendon unit (MTU) [22]. However,
                                                                                            knee flexion/extension torques using measured muscle elec-
this data is not available at the beginning of a cycle. They are
                                                                                            tromyography signals and joint kinematics. The NMS-GP
instead approximated using EMGs from two previous time
                                                                                            model combines the traditional NMS model with modern
steps. This approximation may introduce initial offsets in the
                                                                                            machine learning techniques that include uncertainty quan-
predicted torque. Additionally, relatively high uncertainties
                                                                                            tification. The proposed NMS-GP model not only offers a
were observed during terminal stance and terminal swing
                                                                                            higher prediction accuracy but also provides a measure of
in self-selected speed walking. This could be attributed to
                                                                                            uncertainty in predictions. This approach can be applied to
the transition between stance and swing phases of gait,
                                                                                            enhance the performance of rehabilitation exoskeletons and
during which different muscle coordination patterns may
                                                                                            demonstrates great potential for various applications such as
exist. These periods of high uncertainty should be taken into
                                                                                            exoskeleton control and real-time evaluation of interventions.

          Authorized licensed use limited to: South-Central Minzu University. Downloaded on October 01,2026 at 03:45:09 UTC from IEEE Xplore. Restrictions apply.

                              R EFERENCES                                              model and an artificial neural network model,” IEEE Transactions on
                                                                                       Automation Science and Engineering, vol. 18, no. 2, pp. 564–573,
 [1] Z. Li, Y. Yuan, L. Luo, W. Su, K. Zhao, C. Xu, J. Huang, and M. Pi,               2020.
     “Hybrid brain/muscle signals powered wearable walking exoskeleton            [20] H. X. Hoang, C. Pizzolato, L. E. Diamond, and D. G. Lloyd, “Subject-
     enhancing motor ability in climbing stairs activity,” IEEE Transactions           specific calibration of neuromuscular parameters enables neuromus-
     on Medical Robotics and Bionics, vol. 1, no. 4, pp. 218–227, 2019.                culoskeletal models to estimate physiologically plausible hip joint
 [2] M. Sartori, D. G. Lloyd, and D. Farina, “Neural data-driven muscu-                contact forces in healthy adults,” Journal of Biomechanics, vol. 80,
     loskeletal modeling for personalized neurorehabilitation technologies,”           pp. 111–120, 2018.
     879–893, 2016.                                                                    B. J. Fregly, and M. Reggiani, “Ceinms: A toolbox to investigate the
 [3] L. Li, M. J. Foo, J. Chen, K. Y. Tan, J. Cai, R. Swaminathan, K. S. G.            influence of different neural control solutions on the prediction of
     Chua, S. K. Wee, C. W. K. Kuah, H. Zhuo et al., “Mobile robotic                   muscle excitation and joint moments during dynamic motor tasks,”
     balance assistant (mrba): a gait assistive and fall intervention robot for        Journal of Biomechanics, vol. 48, no. 14, pp. 3929–3936, 2015.
     daily living,” Journal of NeuroEngineering and Rehabilitation, vol. 20,      [22] L. Zhang, X. Zhu, E. M. Gutierrez-Farewik, and R. Wang, “Ankle
     no. 1, pp. 1–17, 2023.                                                            joint torque prediction using an nms solver informed-ann model and
 [4] L. Luo, M. J. Foo, M. Ramanathan, J. K. Er, C. H. Chiam, L. Li,                   transfer learning,” IEEE journal of biomedical and health informatics,
     W. Y. Yau, and W. T. Ang, “Trajectory generation and control of a                 vol. 26, no. 12, pp. 5895–5906, 2022.
     lower limb exoskeleton for gait assistance,” Journal of Intelligent &        [23] E. Soleimani and E. Nazerfard, “Cross-subject transfer learning in
     Robotic Systems, vol. 106, no. 3, p. 64, 2022.                                    human activity recognition systems using generative adversarial net-
 [5] Z. Li, G. Li, X. Wu, Z. Kan, H. Su, and Y. Liu, “Asymmetric coop-                 works,” Neurocomputing, vol. 426, pp. 26–34, 2021.
     eration control of dual-arm exoskeletons using human collaborative           [24] L. Zhang, X. Zhu, E. M. G. Farewik, and R. Wang, “Estimation of
     no. 11, pp. 12 126–12 139, 2021.                                                  nn model,” in 2021 6th IEEE International Conference on Advanced
 [6] S. Yang, N. P. Garg, R. Gao, M. Yuan, B. Noronha, W. T. Ang, and                  Robotics and Mechatronics (ICARM). IEEE, 2021, pp. 75–80.
     D. Accoto, “Learning-based motion-intention prediction for end-point         [25] J. Liang, Z. Shi, F. Zhu, W. Chen, X. Chen, and Y. Li, “Gaussian
     control of upper-limb-assistive robots,” Sensors, vol. 23, no. 6, p. 2998,        process autoregression for joint angle prediction based on semg
     2023.                                                                             signals,” Frontiers in Public Health, vol. 9, pp. 1–15, 2021.
 [7] Z. Li, K. Zhao, L. Zhang, X. Wu, T. Zhang, Q. Li, X. Li, and C.-Y. Su,       [26] N. Chen, Z. Qian, I. T. Nabney, and X. Meng, “Wind power forecasts
     “Human-in-the-loop control of a wearable lower limb exoskeleton for               using gaussian processes and numerical weather prediction,” IEEE
     stable dynamic walking,” IEEE/ASME Transactions on Mechatronics,                  Transactions on Power Systems, vol. 29, no. 2, pp. 656–665, 2013.
     vol. 26, no. 5, pp. 2700–2711, 2020.                                         [27] H. Guo, Z. Meng, Z. Huang, L. W. Kang, Z. Chen, M. Meghjani,
 [8] B. Chen, H. Ma, L.-Y. Qin, F. Gao, K.-M. Chan, S.-W. Law, L. Qin,                 M. Ang, and D. Rus, “Safe path planning with gaussian process
     and W.-H. Liao, “Recent developments and challenges of lower                      regulated risk map,” in 2019 IEEE/RSJ International Conference on
     extremity exoskeletons,” Journal of Orthopaedic Translation, vol. 5,              Intelligent Robots and Systems (IROS). IEEE, 2019, pp. 2044–2051.
     pp. 26–37, 2016.                                                             [28] J. Maritz, F. Lubbe, and L. Lagrange, “A practical guide to gaussian
 [9] Z. Li, B. Huang, Z. Ye, M. Deng, and C. Yang, “Physical human–                    process regression for energy measurement and verification within the
     robot interaction of a robotic exoskeleton by admittance control,” IEEE           bayesian framework,” Energies, vol. 11, no. 4, pp. 1–12, 2018.
     Transactions on Industrial Electronics, vol. 65, no. 12, pp. 9614–9624,      [29] Y. Yun, H.-C. Kim, S. Y. Shin, J. Lee, A. D. Deshpande, and C. Kim,
     2018.                                                                             “Statistical method for prediction of gait kinematics with gaussian
[10] M. Zhang, A. McDaid, A. J. Veale, Y. Peng, and S. Q. Xie, “Adaptive               process regression,” Journal of Biomechanics, vol. 47, no. 1, pp. 186–
     trajectory tracking control of a parallel ankle rehabilitation robot with         192, 2014.
     joint-space force distribution,” IEEE Access, vol. 7, pp. 85 812–85 820,
                                                                                  [30] C. Pizzolato, M. Reggiani, L. Modenese, and D. Lloyd, “Real-time
     2019.
                                                                                       inverse kinematics and inverse dynamics for lower limb applications
[11] S. Yao, Y. Zhuang, Z. Li, and R. Song, “Adaptive admittance control               using opensim,” Computer methods in biomechanics and biomedical
     for an ankle exoskeleton using an emg-driven musculoskeletal model,”              engineering, vol. 20, no. 4, pp. 436–445, 2017.
     Frontiers in Neurorobotics, vol. 12, pp. 1–12, 2018.
                                                                                  [31] F. Leboeuf, R. Baker, A. Barré, J. Reay, R. Jones, and M. Sangeux,
[12] G. Durandau, D. Farina, and M. Sartori, “Robust real-time muscu-
                                                                                       “The conventional gait model, an open-source implementation that
     loskeletal modeling driven by electromyograms,” IEEE Transactions
                                                                                       reproduces the past but prepares for the future,” Gait & Posture,
     on Biomedical Engineering, vol. 65, no. 3, pp. 556–564, 2017.
                                                                                       vol. 69, pp. 235–241, 2019.
[13] K. Fujii, T. Abe, S. Kubota, A. Marushima, H. Kawamoto, T. Ueno,
                                                                                  [32] M. G. Pandy, “Computer modeling and simulation of human move-
     A. Matsushita, K. Nakai, K. Saotome, H. Kadone et al., “The voluntary
                                                                                       ment,” Annual Review of Biomedical Engineering, vol. 3, no. 1, pp.
     driven exoskeleton hybrid assistive limb (hal) for postoperative training
                                                                                       245–273, 2001.
     of thoracic ossification of the posterior longitudinal ligament: a case
     report,” The journal of spinal cord medicine, vol. 40, no. 3, pp. 361–       [33] R. Wang, “Biomechanical consequences of gait impairment at the
     367, 2017.                                                                        ankle and foot: Injury, malalignment, and co-contraction,” Ph.D.
                                                                                       dissertation, KTH Royal Institute of Technology, 2012.
[14] L. Marchal-Crespo and D. J. Reinkensmeyer, “Review of control
     strategies for robotic movement training after neurologic injury,”           [34] D. A. Winter, H. G. Sidwall, and D. A. Hobson, “Measurement
     Journal of Neuroengineering and Rehabilitation, vol. 6, no. 1, pp.                and reduction of noise in kinematics of locomotion,” Journal of
     1–15, 2009.                                                                       Biomechanics, vol. 7, no. 2, pp. 157–159, 1974.
[15] Y. Huang, R. Song, A. Argha, A. V. Savkin, B. G. Celler, and S. W. Su,       [35] A. Mantoan, C. Pizzolato, M. Sartori, Z. Sawacha, C. Cobelli, and
     “Continuous description of human 3d motion intent through switching               M. Reggiani, “Motonms: A matlab toolbox to process motion data
     mechanism,” IEEE Transactions on Neural Systems and Rehabilitation                for neuromusculoskeletal modeling and simulation,” Source Code for
     Engineering, vol. 28, no. 1, pp. 277–286, 2019.                                   Biology and Medicine, vol. 10, no. 1, pp. 1–14, 2015.
[16] S. Y. A. Mounis, N. Z. Azlan, and F. Sado, “Assist-as-needed control         [36] T. R. Derrick, A. J. van den Bogert, A. Cereatti, R. Dumas, S. Fantozzi,
     strategy for upper-limb rehabilitation based on subject’s functional              and A. Leardini, “Isb recommendations on the reporting of interseg-
     ability,” Measurement and Control, vol. 52, no. 9-10, pp. 1354–1361,              mental forces and moments during human motion analysis,” Journal
     2019.                                                                             of Biomechanics, vol. 99, pp. 1–10, 2020.
[17] C. Pizzolato, D. J. Saxby, D. Palipana, L. E. Diamond, R. S. Barrett,        [37] M. Sartori, M. Reggiani, D. G. Lloyd, and E. Pagello, “A neu-
     Y. D. Teng, and D. G. Lloyd, “Neuromusculoskeletal modeling-                      romusculoskeletal model of the human lower limb: Towards emg-
     based prostheses for recovery after spinal cord injury,” Frontiers in             driven actuation of multiple joints in powered orthoses,” in 2011 IEEE
[18] M. Ezati, B. Ghannadi, and J. McPhee, “A review of simulation                     1–6.
     methods for human movement dynamics with emphasis on gait,”                  [38] C. E. Rasmussen, Gaussian processes in machine learning. Springer,
     Multibody System Dynamics, vol. 47, no. 3, pp. 265–292, 2019.                     2004.
[19] L. Zhang, Z. Li, Y. Hu, C. Smith, E. M. G. Farewik, and R. Wang, “An-        [39] T. Beckers, “An introduction to gaussian process models,” arXiv
     kle joint torque estimation using an emg-driven neuromusculoskeletal              preprint arXiv:2102.05497, 2021.

  Authorized licensed use limited to: South-Central Minzu University. Downloaded on October 01,2026 at 03:45:09 UTC from IEEE Xplore. Restrictions apply.