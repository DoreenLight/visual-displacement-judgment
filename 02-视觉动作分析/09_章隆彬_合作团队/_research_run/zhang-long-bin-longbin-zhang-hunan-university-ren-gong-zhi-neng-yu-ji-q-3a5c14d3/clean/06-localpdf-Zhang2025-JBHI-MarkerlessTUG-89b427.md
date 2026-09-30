<!-- source: https://doi.org/10.1109/jbhi.2025.3543095 -->
<!-- title: Towards Clinical Application of Enhanced Timed Up and Go With Markerless Motion Capture and Machine Learning for Balance and Gait Assessment -->
<!-- fetched: 2026-09-30T13:46:22Z -->
<!-- via: local_paper_fulltext(pdftotext -layout) -->
<!-- note: 原文 PDF 存于 09_章隆彬_合作团队/papers/，此为 pdftotext 抽取全文 -->

Abstract—Balance and gait impairments play a key role                         we included TUG datasets of 70 participants with varying
in falls among the elderly. Traditional clinical scales such                     degrees of fall risk based on the BBS scores. We seg-
as the Berg Balance Scale (BBS) to assess fall risk are                          mented TUG trials into five phases automatically using data
often subjective, time consuming, and do not assess gait                         from the MoCap system and extracted features from the
performance. Shorter assessments such as Timed Up and                            phases. These features were then analyzed to identify those
Go (TUG) are available, but most clinicians only look into                       that significantly discriminate between high and low fall
the completion time. This study aimed to develop a fast,                         risk groups. Using the identified features, various machine
low-cost, and automated framework for balance function                           learning models were tested to estimate the BBS scores.
assessment and comprehensive gait analysis by enhancing                          The markers obtained from the markerless MoCap system
the traditional TUG test with a markerless motion capture                        were used for detailed gait analysis, and lower limb kine-
(MoCap) system and machine learning models. In total,                            matics were compared between the markerless and marker-
                                                                                 based methods. Our findings indicate that individuals at
   Received 1 August 2024; revised 16 January 2025; accepted 9 Febru-            high risk of falling had longer completion times, lower per-
ary 2025. Date of publication 18 February 2025; date of current ver-             formance velocities, and smaller ranges of motion in lower-
sion 5 August 2026. This work was supported in part by the Nanyang               limb joints. Among the tested machine learning models,
Technological University (NTU), in part by the Agency for Science Tech-          random forest demonstrated the best performance in pre-
nology and Research (A*STAR), in part by the National Healthcare                 dicting BBS scores (RMSE: 0.98, R2 : 0.94). Additionally, our
Group (NHG), and in part by the Jian Xiang Hospital Group, China.                markerless MoCap system showed comparable accuracy
(Corresponding author: Ananda Sidarta.)                                          to state-of-the-art systems, eliminating the need to attach
   This work involved human subjects or animals in its research. Ap-
                                                                                 markers or sensors. The findings could help develop a
proval of all ethical and experimental procedures and protocols was
granted by the Nanyang Technological University Institutional Review             quick and objective tool for balance and gait assessment in
Board under Application No. IRB-2022-382, the Human Subjects Ethics              older adults, providing quantitative data to improve screen-
Subcommittee of The Hong Kong Polytechnic University under Applica-              ing and intervention planning.
tion No. HSEARS20210715003-01, and by Foshan Jianxiang Nursing
Home Medical Ethics Committee under Application No. IRB-2023-02.                   Index Terms—Berg balance scale, gait analysis,
All procedures performed were in line with the Declaration of Helsinki.          kinematics, machine learning, markerless motion capture
   Longbin Zhang is with the School of Robotics, Hunan University,               system.
Changsha 410012, China, also with the Rehabilitation Research Insti-
tute of Singapore, Lee Kong Chian School of Medicine, Nanyang Tech-
nological University, Singapore 308232, and also with the Guangdong                                          I. INTRODUCTION
Zhongxin Intelligent Rehabilitation Institute affiliated with Jian Xiang
Hospital Group, 528000, China.                                                           FALL is defined as ‘an unexpected loss of balance re-
   Hao Wang and Lei Li are with the Rehabilitation Research Institute
of Singapore, Lee Kong Chian School of Medicine, Nanyang Techno-
logical University, Singapore 308232, and also with the Guangdong
                                                                                 A      sulting in coming to rest on the floor, the ground, or other
                                                                                 lower level’ [1]. Falls in older adults are a global health and
Zhongxin Intelligent Rehabilitation Institute affiliated with Jian Xiang         societal challenge. According to the World Health Organization,
Hospital Group, 528000, China.
   Ananda Sidarta, Tsung-Lin Wu, and Prayook Jatesiktat are with the             approximately one in three adults aged 65 and above experiences
Rehabilitation Research Institute of Singapore, Lee Kong Chian School            a fall each year [2]. Furthermore, approximately 30% of these
of Medicine, Nanyang Technological University, Singapore 308232                  falls result in moderate to severe injury, such as fractures and
(e-mail: ananda.sidarta@ntu.edu.sg).
   Patrick Wai-Hang Kwong is with the Department of Rehabilitation               head trauma, with long-lasting and debilitating effects. Falls can
Sciences, The Hong Kong Polytechnic University, 528000, China.                   lead to hospitalization, decreased mobility, reduced quality of
   Aoyang Long and Xiangyu Long are with the Jian Xiang Hospital                 life, and a subsequent loss of independence, incurring substantial
Group, 528000, China.
   Wei Tech Ang is with the School of Mechanical and Aerospace En-               financial burdens on the healthcare system. The causes of falls
gineering, Nanyang Technological University, Singapore 639798, and               among older adults are multifaceted and complex. Among these,
also with the Rehabilitation Research Institute of Singapore, Lee Kong           balance function and gait performance are critical indicators of
Chian School of Medicine, Nanyang Technological University, Singapore
308232.                                                                          fall risk. We posited requirements for a desirable balance and
2168-2194 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and similar technologies.
   Personal use is permitted, but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html
                                                                 for more information.

  Authorized licensed use limited to: South-Central Minzu University. Downloaded on September 30,2026 at 08:36:00 UTC from IEEE Xplore. Restrictions apply.

ZHANG et al.: TOWARDS CLINICAL APPLICATION OF ENHANCED TUG WITH MOCAP AND MACHINE LEARNING                                                                7073

comprehensive, probing balance and mobility functions. The                       retroreflective markers, the need for trained operators, and the
outcomes not only identify older adults at increased risk of                     potential interference of markers with natural movement [25].
falling but also serve as the foundation for developing early and                Although wearable sensors are also widely used in balance
tailored interventions to mitigate their vulnerability [3], [4].                 function prediction and gait analysis [8], their placement can
   In clinical practice, the Berg Balance Scale (BBS) and Timed                  introduce uncertainty, causing discomfort and affecting natural
Up and Go (TUG) test are widely used to assess an individ-                       movement [9]. As a result, there is increasing demand for fast,
ual’s balance function [5], [6], [7], [8], [9]. The BBS is often                 non-invasive, and user-friendly alternatives to overcome these
regarded as the “gold standard” for evaluating balance due to its                challenges.
comprehensive assessment of multiple dimensions of postural                         In recent years, markerless MoCap systems have transformed
control, including static and dynamic balance. It consists of 14                 motion analysis, offering precise assessments of human move-
tasks designed to evaluate an individual’s ability to maintain                   ment without the need for physical markers or sensors [25].
stability during different activities. Each task is scored on a                  Among these, OpenPose can detect 25 2D body keypoints for
scale from 0 to 4 (where 0 indicates an inability to complete                    basic movement analysis, but cannot directly determine body
the task and 4 indicates smooth execution). The total BBS score                  segment orientations, has limited precision for distal joints,
can reach a maximum of 56, with clinical BBS scores below 50                     and struggles with occlusion or complex environments [26],
generally indicating a high risk of falls [10], [11], [12]. Despite              [27]. OpenCap enhances this by incorporating Long Short-Term
its widespread use and rigor, the process is time-consuming,                     Memory networks, trained on a large MoCap dataset, to infer
with a full protocol taking approximately 30 minutes [8], [9].                   and “augment” the sparse keypoints with 43 detailed anatomical
Additionally, BBS does not directly assess gait performance.                     markers. However, because these markers are inferred rather
In contrast, the TUG test is a quicker and simpler alternative,                  than directly extracted from the user’s images, they may intro-
measuring the time required to complete a sequence of func-                      duce biases toward average movement patterns from the training
tional movements: rising from a chair, walking, turning, and                     data. This limits the system’s generalizability, particularly for
sitting back down. By incorporating elements of both balance                     individuals with unique or pathological movement profiles [28].
and mobility, the TUG provides valuable insights into fall risk                  Another notable system, Theia3D, predicts 3D joint positions
with a shorter time and minimal equipment. Importantly, studies                  from users’ video data by using a customized dataset of anno-
have demonstrated a strong correlation between TUG and BBS                       tated anatomical landmarks and scaling an articulated multibody
scores [13], [14], [15], suggesting that the TUG test could                      model to match subject-specific landmarks in 3D space [29].
serve as a practical proxy for predicting BBS outcomes while                     However, it relies on proprietary landmarks and outputs only
providing a quicker alternative. However, the reliability of the                 joint centers, limiting the accuracy of landmark detection and the
TUG can vary due to its dependence on the clinician’s judgment                   analysis of full-body segment orientations. In contrast, our re-
and environmental conditions [16]. Moreover, the threshold for                   cent work integrates anatomical priors and data-driven methods
discriminating between low and high fall risk individuals using                  to capture human movement from multi-view RGB videos [30].
TUG range widely in the literature, from 10 to 25 seconds [17],                  This markerless system is trained using the RRIS40 dataset,
[18], [19]. Given these challenges, there is a growing demand                    which uses 2D annotations of 40 surface anatomical landmarks
for innovative, technology-driven solutions to enhance the ob-                   projected from marker-based motion capture at pixel-level ac-
jectivity and efficiency of balance function assessments. The                    curacy. With a multi-view weighted triangulation, this system
integration of automated approaches into the TUG test may                        predicts 3D marker positions with an average error of 13.23mm.
enhance its utility by not only measuring completion time but                    Unlike Theia3D, our system directly outputs virtual marker
also extracting various kinematic parameters from each section                   positions corresponding to bone landmarks, which facilitates
of the TUG movements, which have a high overlap with most                        the application of well-established biomechanical formulas de-
components of the BBS test, such as sitting balance, sitting                     veloped over decades. This unique capability enables compre-
to stand, standing to sit, and turning, thus bridging the gap                    hensive parameter derivation from movement data, enhancing
between simplicity and comprehensiveness in balance and gait                     biomechanical analysis. Therefore, a question arises whether an
evaluation.                                                                      enhanced TUG test, integrating a markerless MoCap system and
   Clinical gait analysis is commonly used to provide key in-                    machine learning, could effectively predict the BBS score with
sights into mobility, stability, and functional performance [20],                high accuracy while concurrently providing a comprehensive
[21], [22], [23]. Compared to younger adults, older individuals                  analysis of gait and balance.
require greater motor control to maintain stability while walking,                  Our objectives are twofold: (1) to evaluate the agreement
which makes gait impairments one of the strongest predictors                     of lower limb joint kinematics during the walking phase of
of falls. Past studies have identified changes in specific gait                  the TUG using the markerless MoCap system, compared to
parameters, such as stride length, gait speed, and variability,                  a marker-based MoCap system; (2) to develop a quantitative
as independent risk factors for falls. These subtle gait changes                 balance assessment framework for older adults by implementing
are often imperceptible through visual observation, necessitating                an enhanced TUG test integrating a markerless MoCap system
the use of objective and precise analytical tools [24]. While                    and machine learning techniques. This framework aimed to
highly accurate, gait analysis has traditionally relied on marker-               estimate BBS scores and simultaneously offer comprehensive
based 3D motion capture (MoCap) systems, which have signif-                      gait analysis, including spatiotemporal parameters and joint
icant limitations, such as the time-intensive process of placing                 kinematics.

   Authorized licensed use limited to: South-Central Minzu University. Downloaded on September 30,2026 at 08:36:00 UTC from IEEE Xplore. Restrictions apply.

Fig. 1. Schematic of the balance and gait assessment framework using an enhanced TUG test with a markerless motion capture system and
machine learning. This system can predict BBS scores and simultaneously offer comprehensive gait analysis, including spatiotemporal (ST)
parameters and joint kinematics.

                             II. METHODS                                        All BBS assessments were performed by a professional physical
                                                                                therapist.
   We developed a quantitative framework for balance function
                                                                                   TUG trials were recorded using the Vicon (200Hz) system
assessment and gait analysis using an enhanced TUG test,
                                                                                in Hongkong and OptiTrack (200Hz) system in Foshan. TUG
integrating a markerless MoCap system and machine learning
                                                                                trials were recorded in Singapore simultaneously by a marker-
techniques (Fig. 1). TUG motions were recorded with a 3D
                                                                                based Miqus M3 Qualisys (200Hz) system and a markerless
markerless MoCap system. Using the TUG auto-segmentation
                                                                                MoCap system [30]. The markerless MoCap system comprises
approach, we obtained selected features and motion trajectories
                                                                                8 cameras and uses computer vision to automatically capture
during the TUG test. These features were then used as inputs
                                                                                3D data of human movement. The system can obtain kinematic
for machine learning models to predict BBS outcome, which is
                                                                                data by automatically extracting the 2D position of each virtual
more widely accepted as the ’gold standard’. Walking motions
                                                                                anatomical marker from video captured by the multi-camera
were extracted from the motion trajectories, and gait analysis,
                                                                                system and triangulate them. All trials were performed inside
including spatiotemporal parameters and joint kinematics, was
                                                                                the capture volume of the markerless system.
conducted. The study results consist of three parts: (1) TUG
                                                                                   In each trial, participants performed TUG for 3 repetitions. In
features extracted from high and low fall-risk groups; (2) BBS
                                                                                each repetition, they were initially seated on a height-adjustable
prediction performance using commonly used regression ma-
                                                                                chair such that the knee was at a 90-degree angle. After the cue
chine learning models; and (3) a pilot study comparing lower
                                                                                from the experimenter, participants then stood up and walked
limb joint kinematics captured by marker-based and markerless
                                                                                for a distance of 3 meters towards a line marked on the floor.
MoCap systems.
                                                                                They were asked to walk as fast as possible but safely. Upon
                                                                                passing the line, the participants then made a 180-degree turn to
A. Experimental Setup                                                           go back towards the chair. Upon reaching the chair, they would
   In this multicentre study, we included 70 participants com-                  then be required to sit back, and that repetition was considered
prising three parts: 44 able-bodied subjects (age: 45.8 ± 20.0                  complete. The experimenter continued to cue the participants
years; sex: 26F/18M; weight: 60.6 ± 12.5 kg; height: 163.7 ±                    for subsequent repetitions. The repetition was repeated if there
10.0 cm) from the Rehabilitation Research Institute of Singapore                was a false start or failure to comply with the required TUG
(approved by the Nanyang Technological University Institu-                      performance.
tional Review Board, IRB-2022-382), 7 post-stroke individuals
(age: 65.6 ± 5.6 years; sex: 3F/4M; weight: 57.9 ± 5.1 kg;
height: 167.4 ± 4.1 cm) from Hong Kong Polytechnic University
(approved by Human Subjects Ethics Subcommittee of Hong                         B. TUG Segmentation
Kong Polytechnic University, HSEARS20210715003-01), and                           We segmented TUG into five phases: sit-to-stand, forward
19 older adults with knee osteoarthritis (age: 62.3 ± 9.2 years;                walking, turning, return walking, and stand-to-sit (Fig. 2(a)).
sex: 17F/2M; weight: 57.4 ± 9.7 kg; height: 156.3 ± 10.0cm)                        r Sit-to-stand phase: The marker placed on the C7 vertebra
from the Guangdong China-Singapore Intelligent Rehabilitation                        was used to detect the start and standing event of the sit-to-
Institute (approved by Foshan Jianxiang Nursing Home Medi-                           stand phase. The movement of the C7 marker in the vertical
cal Ethics Committee, IRB-2023-02). These participants were                          Z-direction transitions from stable to moving when the
instructed to perform TUG movements within a 3D MoCap                                person begins to stand, which was detected as the start
system. All participants were able to complete the experimental                      event. After standing up, the C7 marker stops increasing
procedures as directed by the instructors and willingly consented                    in the Z direction, which was detected as the standing event
to participate in the study by signing informed consent forms.                       (Fig. 2(b)).

 Authorized licensed use limited to: South-Central Minzu University. Downloaded on September 30,2026 at 08:36:00 UTC from IEEE Xplore. Restrictions apply.

ZHANG et al.: TOWARDS CLINICAL APPLICATION OF ENHANCED TUG WITH MOCAP AND MACHINE LEARNING                                                                7075

Fig. 2. Illustration of TUG phases with segmentation approaches. (a) Overview of TUG phases: sit-to-stand, forward walking, turning, return
walking, and stand-to-sit. (b) Sit-to-stand phase segmentation using C7 marker’s vertical Z-direction position (c) Heel strike and toe-off gait events
using the heel and toe markers relative to the pelvis coordination system in the anteroposterior direction. (d) Turning phase segmentation using
pelvis rotation angle. (e) Stand-to-Sit phase segmentation using C7 marker’s vertical Z-direction position.

    r Forward walking phase: After standing up from the chair,                         at turn start and turn end), turn arc length (the length of
      the subject walked 3 meters and made a turn. The heel                            the curved path traveled by the pelvis during the turn), and
      strike (right heel strike, RHS; left heel strike, LHS) and                       turn speed (turn arc length divided by the turning time).
      toe-off (right toe-off, RTO; left toe-off, LTO) gait events                      4 features were extracted.
      during the gait cycle were detected using the heel and toe                     r Gait spatiotemporal features extracted from gait cycles,
      markers relative to the pelvis coordination system in the                        strides, and steps for each leg, including gait speed, ca-
      anteroposterior direction (Fig. 2(c)) [31].                                      dence, stride length, step length, step time, step width, as
    r Turning phase: While turning, the body rotates around                            well as gait cycle time, stance time, swing time, initial
      180 degrees. The start and end of the turning phase were                         double support time, terminal double support time, total
      detected when the body began to rotate and complete the                          double support time, and their proportions relative to the
      rotation. The pelvis rotation angle was used to segment                          gait cycle time. Additionally, the average, ratio, and gait
      this phase (Fig. 2(d)).                                                          symmetry index of these left and right leg features were
    r Return walking phase: After turning back, the person                             calculated. 85 features were extracted. Gait features were
      walked 3 meters back to the starting point of the TUG                            extracted from the forward and return walking phases of
      test. The gait cycle was detected using the same method                          the TUG test.
      as in the forward walking phase.                                               r Gait kinematics features extracted from lower limb joints,
    r Stand-to-sit phase: Similar to the sit-to-stand phase, the                       including the minimum, maximum, and range of motion
      standing event was detected when the C7 marker in the                            of hip flexion/extension (F/E), knee flexion/extension, and
      Z direction started to decrease, and the sitting event was                       ankle dorsiflexion/plantarflexion (D/P). Then, 18 features
      detected when the Z direction of the C7 marker stabilized                        were extracted.
      in its position after decreasing (Fig. 2(e)).                                In total, 115 features were extracted from three repetitions of
                                                                                 TUG. Gait spatiotemporal and kinematic features were averaged
                                                                                 across gait cycles of all repetitions, while the remaining features
C. Feature Extraction                                                            were averaged across repetitions.
   Based on the five phases extracted from the TUG test, we
selected the following parameters:
    r Total duration and duration of each phase. 6 features were                 D. Machine Learning Models/Regression Models
      extracted.                                                                    Linear regression (LR), least absolute shrinkage and selec-
    r Maximum vertical velocity to stand and sit down. 2 fea-                    tion operator regression (Lasso), random forest (RF), gradi-
      tures were extracted.                                                      ent boosted decision tree (GBDT), extreme gradient boosting
    r Several features during the turn phase, including turn                     (XGBoost), k-nearest neighbors regression (KNN), and support
      steps (the number of steps taken during the turn), turn                    vector regression (SVR) were used to estimate BBS scores with
      width (the Euclidean distance between the pelvis positions                 selected features. These models were chosen to include both

   Authorized licensed use limited to: South-Central Minzu University. Downloaded on September 30,2026 at 08:36:00 UTC from IEEE Xplore. Restrictions apply.

linear and non-linear approaches, ensuring a comprehensive                      fits the data while maintaining a maximum margin (epsilon-
evaluation of different modeling paradigms, and their selection                 insensitive tube) around the predicted values. The objective
was further motivated by their frequent application in similar                  function is formulated as follows:
studies in the literature [8], [32].                                                                               n
                                                                                                                                  
   LR is a linear approach to modeling the relationship between                                        1           
                                                                                                             2                ∗
                                                                                              Fobj =     w + C       (ζi + ζi )        (5)
a target variable and one or more independent input variables.                                         2           i=1
The objective function is formulated as:
                                 1
                                     n                                          subject to: yi − wT φ(xi ) − b ≤  + ζi , wT φ(xi ) + b − yi ≤
                      Fobj =           (yi − ŷi )2                      (1)     + ζi∗ , and ζi , ζi∗ ≥ 0. Where w is the weight vector, b is the
                                 n i=1
                                                                                bias term, φ(xi ) is the feature mapping function, ζi and ζi∗ are
where yi and ŷi are the actual and predicted value, and n is the               slack variables,  is the margin of tolerance, and C is the penalty
number of samples.                                                              parameter for errors.
   Lasso regression, a variant of linear regression, automatically
selects features by shrinking less important coefficients to zero,              E. Feature Selection
making it useful when dealing with many features and selecting
only the relevant ones. The objective function can be written as:                  The subjects were categorized into two groups using a cutoff
                                                                                score of 50 [10], [11], [12]. Participants with a BBS score > 50
                           n                    p
                        1                                                     were classified as low-fall-risk individuals, while those with a
             Fobj =           (yi − ŷi )2 + α     |βj |                (2)
                       2n i=1                                                   score ≤ 50 were classified as high-fall-risk individuals. This
                                               j=1
                                                                                classification resulted in 55 low fall-risk individuals and 15
where p is the number of features, βj is the coefficient of feature             high fall-risk individuals. The high fall-risk group included 7
j, and α is the regularization parameter.                                       older post-stroke individuals and 8 older individuals with knee
   RF is an ensemble method that effectively models non-linear                  osteoarthritis.
relationships between inputs and target variables by constructing                  To assess the normality of TUG features, Shapiro-Wilk tests
multiple decision trees and combining their predictions. Each                   were used. Subsequently, two-sample t-tests and Mann-Whitney
decision tree partitions the dataset into smaller, more predictable             U tests were used to compare features between the high- and
subsets based on input features, with internal nodes representing               low-fall-risk groups for features with normal and non-normal
feature-based decisions and leaf nodes representing outcomes                    distributions, respectively. Statistical results are considered sig-
or predictions. Compared to a single decision tree, RF reduces                  nificant for p < 0.05. To optimize the feature set, redundant fea-
overfitting and improves prediction accuracy through ensemble                   tures were identified through correlation analysis. Specifically,
averaging. GBDT, on the other hand, is also an ensemble method                  pairs of features with a correlation coefficient exceeding 90%
but builds trees sequentially. Each new tree corrects errors made               were considered redundant. Within each pair, the feature demon-
by previously trained trees, unlike RF, which trains each tree                  strating the most significant statistical distinction between the
independently using a random sample of the data. XGBoost                        high- and low-fall-risk groups (i.e., the lowest p-value) was
is a more regularized form of gradient boosting that enhances                   retained to ensure the selection of features most relevant to the
model generalization by incorporating advanced regularization                   analysis [32], [33].
techniques. The objective function for XGBoost can be written
as:
                                                                                F. Evaluation Protocol
                          n
                                                K
                                                 
                Fobj =          L(yi , ŷi ) +         Ω(fk )            (3)       We evaluated the BBS score prediction accuracy of these
                          i=1                    k=1                            machine learning models: LR, Lasso, RF, GBDT, XGBoost,
                                                                                KNN, and SVR. The data was split into 80% for training and
where L is the loss function that measures the difference between
                                                                                20% for testing. To mitigate the risk of overfitting, five-fold
the true target values yi and the predicted values ŷi ; Ω is the
                                                                                cross-validation was conducted, which is a widely used method
regularization term that penalizes the complexity of the model;
                                                                                to ensure a thorough assessment of the model’s generalizabil-
fk represents the individual decision trees in the ensemble.
                                                                                ity [34]. Root mean square error (RMSE), absolute mean error
   KNN regression makes predictions by finding the K nearest
                                                                                (MAE), and R-squared (R2 ) were used to evaluate the estimation
data points to a given input and averaging their target values.
                                                                                performance of each model. A low RMSE and MAE indicate
Mathematically, the prediction ŷ for a new data point x in
                                                                                high prediction accuracy. R2 returns a value between 0 and 1,
regression is given by:
                                                                                providing a measure of how much of the variability in the actual
                                         k
                                    1                                          BBS scores is explained by the model’s predictions. A value of 0
                             ŷ =         yi                             (4)    indicates no explanatory power, while a value of 1 indicates that
                                    k i=1
                                                                                the model perfectly explains all of the variability in the actual
where yi represents the target value of the ith nearest neighbor                BBS scores.
to x, and k is the number of neighbors to consider.                                Simultaneous markerless and marker-based data were col-
   SVR is a variant of Support Vector Machines used for regres-                 lected for analysis from eighteen subjects during walking, ex-
sion analysis. Its goal is to find an optimal hyperplane that best              tracted from TUG trials. The RMSE and intraclass correlation

 Authorized licensed use limited to: South-Central Minzu University. Downloaded on September 30,2026 at 08:36:00 UTC from IEEE Xplore. Restrictions apply.

ZHANG et al.: TOWARDS CLINICAL APPLICATION OF ENHANCED TUG WITH MOCAP AND MACHINE LEARNING                                                                7077

coefficients (ICC) of hip, knee, and ankle joint kinematics, calcu-
lated using both marker-based and markerless MoCap systems,
were analyzed.

                             III. RESULTS
A. TUG Features
   We extracted 115 features from each TUG trial. Overall,
significant differences were found in most features between the
high- and low-fall-risk groups, including those related to the total
duration and duration of each phase, maximum vertical velocity,
turn phase, gait spatiotemporal parameters, and gait kinematics
(Fig. 3). However, no significant differences were observed in
the gait symmetry ratio and step width.
   In terms of time duration, the high-fall-risk group exhibited
longer total TUG times, longer duration in each phase, and
longer times in each gait phase (all p < 0.01, Fig. 3(a), (b)).
Regarding the fraction of gait cycle time, the high-fall-risk group
had higher values for stance time, initial double (ID) support
time, terminal double (TD) support time, and total double sup-
port time fractions (all p < 0.01, Fig. 3(c)).
   In terms of velocity/speed performance, the high-fall-risk
group had lower velocities/speeds for sit-to-stand, turning, gait,
and stand-to-sit phases, as well as lower gait cadence (all p <
0.01, Fig. 3(d), (e)). Additionally, the high-fall-risk group took
more steps during turning phases (p < 0.01, Fig. 3(f)). During
gait and turning, this group also had smaller turn widths, turn arc
lengths, stride lengths, and step lengths (all p < 0.01). However,
no significant differences were observed in step width (p = 0.37,
Fig. 3(g)).
   Regarding gait symmetry features, no significant differences
were observed in all gait symmetry ratios (Fig. 3(h)). However,
most gait symmetry index features showed significant differ-
ences, except for initial and terminal double support times and
their fractions, as well as stride length (Fig. 3(i)).
   Among gait kinematics metrics, the high-fall-risk group had
significantly lower ranges of motion and higher minimum values
in hip F/E, knee F/E, and ankle D/P (all p < 0.01). No significant
differences in maximum hip F/E were observed (Left: p = 0.82,
Right: p = 0.38, Fig. 3(j)).
   After removing redundant features, we retained 28 optimal
features, including sit-to-stand time, turn time, stand-to-sit time,
sit-to-stand maximal vertical velocity, turn steps, turn width, turn
arc length, left and right stance time fraction, left and right swing
time fraction, right initial double support time fraction, right
terminal double support time fraction, right gait speed, average
terminal double support time, double support time asymmetry
index, gait speed asymmetry index, cadence, cadence asymme-
try index, step width asymmetry index, right hip F/E range of
motion, left and right knee F/E range of motion and minimum                      Fig. 3. Distributions of features between the high and low fall-risk
                                                                                 groups. The features include (a) TUG Phase Duration, (b) Gait Phase
value, left ankle D/P range of motion and minimum value, and                     Duration, (c) Gait Phase Time Fraction of Gait Cycle Total Time, (d) TUG
right ankle D/P maximum value. These features served as inputs                   Phase Velocity/Speed, (e) Gait Cadence, (f) Turn Steps, (g) Width and
for the regression models developed to predict BBS scores.                       Length Metrics, (h) Gait Symmetry Ratio, (i) Gait Symmetry Index, and
                                                                                 (j) Gait Kinematics Metrics (minimum, maximum, and range of motion of
                                                                                 hip, knee, and ankle joints). ∗ indicated a significant difference between
B. BBS Score Prediction                                                          the two groups, determined using a two-sample t-test for normally dis-
                                                                                 tributed data or the Mann-Whitney U test for non-normally distributed
  Overall, all regression model has high prediction accuracy                     data.
(RMSE ≤ 1.62, MAE ≤ 1.38, R2 ≥ 0.82, Fig. 4). Among the

   Authorized licensed use limited to: South-Central Minzu University. Downloaded on September 30,2026 at 08:36:00 UTC from IEEE Xplore. Restrictions apply.

Fig. 4. Root mean square error (RMSE), absolute mean error (MAE),
and R-squared (R2 ) between predicted by machine learning models and
actual BBS scores.

models, RF, GBDT, and KNN exhibit low prediction error and
high explanatory power. In contrast, LR and XGboost demon-
strate high prediction error and low explanatory power.
   Specifically, LR was observed with the highest prediction
error (both RMSE and MAE) with low explanatory power R2 .
Lasso and SVR show similar prediction performance, with
somewhat improved prediction error and higher explanatory
power compared to LR.
   RF demonstrates further improvement in prediction perfor-
mance, with the lowest MAE among all models (RMSE: 0.98,
MAE: 0.38, R2 : 0.94). GBDT shows similar prediction perfor-
mance to RF, while XGBoost exhibits higher prediction errors
and lower explanatory power.

C. Markerless vs. Marker-Based MoCap Kinematics
   Overall, the lower limb kinematics obtained by the markerless
system exhibited a similar pattern to the marker-based system                    Fig. 5. Lower limb joint kinematics across 163 gait cycles measured
(average RMSE: 5.06◦ , Fig. 5). Hip F/E displayed an extension                   by the markerless and marker-based motion capture system from 18
                                                                                 participants. Average RMSE values over gait cycles are inset in each
offset throughout the gait cycle, while knee F/E showed a similar                joint angle plot.
extension offset. Ankle D/P measured by the markerless system
showed good agreement with the marker-based system (average
RMSE: 3.03◦ , standard deviation (STD): 1.26◦ , ICC: 0.98),                      risk of falling had longer completion times, slower performance
compared to hip F/E (average RMSE: 9.14◦ , STD: 3.55◦ , ICC:                     velocities, and reduced ranges of motion in lower-limb joints.
0.99) and knee F/E (average RMSE: 7.72◦ , STD: 2.45◦ , ICC:                      Among the tested machine learning models, Random Forest
0.99).                                                                           demonstrated the highest accuracy in predicting BBS scores.
                                                                                 Additionally, our markerless MoCap system achieved accuracy
                                                                                 comparable to state-of-the-art systems. By leveraging this com-
                            IV. DISCUSSION                                       prehensive assessment framework, therapists can develop tai-
   We proposed a quantitative framework for balance function                     lored interventions that precisely target stability improvement,
assessment and gait analysis by enhancing the traditional TUG                    risk reduction, and overall mobility enhancement.
test with a markerless MoCap system and machine learning mod-                       Individuals with a high fall risk exhibited longer total TUG
els, eliminating the need for sensors. This system provides more                 times, longer duration in each phase, and longer times in each
clinically relevant outcome measures than the traditional TUG                    gait phase. This finding aligns with the study by Kim et al. [4],
test, which primarily relies on completion time. Additionally, our               which used a Kinect camera-based test battery to assess fall
system predicts BBS scores and offers gait analysis, including                   risk in older adults. They found a significant increase in turning
spatiotemporal parameters and joint kinematics. Our system is                    time but no significant difference in total completion time. They
more time-efficient compared to the gold standard BBS, re-                       attributed this to the limited trackable range of the Kinect, which
quiring significantly less time from clinicians, with assessments                resulted in a shorter TUG walking distance (2 meters) compared
typically completed within 5 minutes. It also eliminates the need                to the original 3 meters, thereby reducing its predictive power.
for specialized clinicians, making it more accessible and cost-                  Additionally, the discontinuation of the Kinect V2 camera by
effective for widespread use. We found that individuals at high                  Microsoft in October 2017 may impact the reproducibility of
  Authorized licensed use limited to: South-Central Minzu University. Downloaded on September 30,2026 at 08:36:00 UTC from IEEE Xplore. Restrictions apply.

ZHANG et al.: TOWARDS CLINICAL APPLICATION OF ENHANCED TUG WITH MOCAP AND MACHINE LEARNING                                                                7079

their assessment system in other settings. In contrast, our sys-                 the knee joint angle calculation also relies on the marker on the
tem provides more comprehensive information, including BBS                       pelvis, resulting in a slightly higher RMSE. The slightly smaller
scores and gait analysis. We also observed shorter stride length,                standard deviation in the markerless mocap system (Fig. 5)
step length, cadence, and gait speed as well as longer gait time                 may be attributed to the machine learning model’s training to
and step time during gait, which aligns with the findings of                     minimize prediction error. This process may inherently align
Dubois et al. [17], who reported significantly shorter step length,              outputs closer to the average patterns in the training data, thereby
step time, cadence, and gait speed. They did not find significant                reducing variability. In contrast, the marker-based system is
differences in turn steps and turn width, which may be related                   more prone to variability from marker placement errors.
to participants’ turning habits. In our study, we did not find                      In this proof-of-concept study, features were derived from
significant differences in step width, likely due to participants’               both marker-based and markerless MoCap systems due to the
walking habits during the TUG test where turning is involved.                    limited number of subjects we were able to recruit in Singapore.
Compared to their study, our approach provides additional in-                    The markerless system was trained on a marker-based dataset
formation, including BBS scores and detailed gait analysis that                  that followed the same markerset and marker placement protocol
encompasses lower limb joint kinematics. In the high-fall risk                   and included over 200 subjects performing various activities.
group, most gait symmetry index features showed significant                      This dataset is part of an Asian-centric movement database
differences. However, no significant differences were observed                   collected over five years [37]. A comparison of lower limb joint
in all gait symmetry ratios. This may be because gait symmetry                   kinematics between the two systems showed similar patterns
ratios are not as sensitive as gait symmetry indices.                            and ranges of motion, supporting the reliability of our approach.
   Machine learning models can provide high accuracy in pre-                     Based on this evidence, we expect minimal differences in predic-
dicting BBS scores. Even a linear regression model using fea-                    tive performance between the two systems. Nonetheless, further
tures extracted from TUG trials demonstrated good prediction                     investigations with a larger sample size are needed to strengthen
accuracy (RMSE: 1.62, MAE: 1.38, R2 : 0.82). Non-linear mod-                     the robustness and broaden the applicability of the markerless
els, such as RF, showed excellent prediction accuracy (MAE:                      MoCap system.
0.38). This aligns with the findings of Lin et al. [8], who applied                 We acknowledge that falls among older adults have diverse
various machine learning models, including bagging regression,                   and complex causes, requiring a multidimensional approach to
KNN, Lasso, LR, RF, ridge regression, and SVR, to predict BBS                    evaluating fall risk factors. While our study provides balance
scores using features from IMU sensors. They also found the                      function assessment and detailed gait analysis with a focus on
RF model had the lowest MAE (MAE: 1.19). Similarly, Badura                       modifiable physical factors, it does not encompass the entire
et al. [35], using five IMUs, achieved an MAE of 1.47, suggesting                spectrum of fall risk factors, such as cognition. Furthermore,
that our model’s accuracy meets the expected standards. In our                   some studies have demonstrated the limited predictive per-
study, RF and GBDT exhibited similar prediction performance,                     formance of the BBS and TUG tests for fall risk, and they
as both are ensemble methods capable of modeling non-linear                      are not recommended for use as standalone predictors despite

<!-- [clean: truncated to max_content_length] -->