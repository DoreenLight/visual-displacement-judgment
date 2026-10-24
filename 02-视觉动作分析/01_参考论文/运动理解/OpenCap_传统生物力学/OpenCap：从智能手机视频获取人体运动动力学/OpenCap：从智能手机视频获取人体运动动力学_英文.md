![](images/887f135c2df2a3c7ee6ef4813f682fa7274529da7c0019586b38beb32040ee89.jpg)

RESEARCH ARTICLE

# OpenCap: Human movement dynamics from smartphone videos

Scott D. Uhlrich<sub>ID</sub><sup>1☯</sup>\*, Antoine Falisse<sub>ID</sub><sup>1☯</sup>\*, Łukasz Kidziński<sup>1☯</sup>, Julie Muccini<sub>I</sub> 2 D Michael Ko<sup>2</sup>, Akshay S. Chaudhari<sup>2,3</sup>, Jennifer L. Hicks<sup>1</sup>, Scott L. Delp<sup>1,4,5</sup>

![](images/655a212352aec9251d3244bef37796bb1f110a260297aaf48d79783692d26852.jpg)

## OPEN ACCESS

Citation: Uhlrich SD, Falisse A, Kidziński Ł, Muccini J, Ko M, Chaudhari AS, et al. (2023) OpenCap: Human movement dynamics from smartphone videos. PLoS Comput Biol 19(10): e1011462. https://doi.org/10.1371/journal.pcbi.1011462

1 Departments of Bioengineering, Stanford University, Stanford, California, United States of America, 2 Radiology, Stanford University, Stanford, California, United States of America, 3 Biomedical Data Science, Stanford University, Stanford, California, United States of America, 4 Mechanical Engineering, Stanford University, Stanford, California, United States of America, 5 Orthopaedic Surgery, Stanford University, Stanford, California, United States of America

Editor: Alison L. Marsden, Stanford University, UNITED STATES

Received: February 1, 2023

Accepted: August 24, 2023

Published: October 19, 2023

Peer Review History: PLOS recognizes the benefits of transparency in the peer review process; therefore, we enable the publication of all of the content of peer review and author responses alongside final, published articles. The editorial history of this article is available here: https://doi.org/10.1371/journal.pcbi.1011462

Copyright: © 2023 Uhlrich et al. This is an open access article distributed under the terms of the Creative Commons Attribution License, which permits unrestricted use, distribution, and reproduction in any medium, provided the original author and source are credited.

Data Availability Statement: The data from the laboratory validation and field studies are available at https://simtk.org/projects/opencap. Identifiable video data is shared for the laboratory validation study. Participants whose identifiable videos are ☯ These authors contributed equally to this work.

\* suhlrich@stanford.edu (SDU); afalisse@stanford.edu (AF)

## Abstract

Measures of human movement dynamics can predict outcomes like injury risk or musculoskeletal disease progression. However, these measures are rarely quantified in large-scale research studies or clinical practice due to the prohibitive cost, time, and expertise required. Here we present and validate OpenCap, an open-source platform for computing both the kinematics (i.e., motion) and dynamics (i.e., forces) of human movement using videos captured from two or more smartphones. OpenCap leverages pose estimation algorithms to identify body landmarks from videos; deep learning and biomechanical models to estimate three-dimensional kinematics; and physics-based simulations to estimate muscle activations and musculoskeletal dynamics. OpenCap’s web application enables users to collect synchronous videos and visualize movement data that is automatically processed in the cloud, thereby eliminating the need for specialized hardware, software, and expertise. We show that OpenCap accurately predicts dynamic measures, like muscle activations, joint loads, and joint moments, which can be used to screen for disease risk, evaluate intervention efficacy, assess between-group movement differences, and inform rehabilitation decisions. Additionally, we demonstrate OpenCap’s practical utility through a 100-subject field study, where a clinician using OpenCap estimated musculoskeletal dynamics 25 times faster than a laboratory-based approach at less than 1% of the cost. By democratizing access to human movement analysis, OpenCap can accelerate the incorporation of biomechanical metrics into large-scale research studies, clinical trials, and clinical practice.

## Author summary

Analyzing how humans move, how we coordinate our muscles, and what forces act on the musculoskeletal system is important for studying neuro-musculoskeletal conditions. Traditionally, measuring these quantities requires expensive laboratory equipment, a trained expert, and hours of analysis. Thus, high-quality measures of human movement are rarely

shared provided written consent to the release of these videos. The source code is available at https://github.com/stanfordnmbl/opencap-core (archived in Zenodo: https://doi.org/10.5281/ zenodo.7419967) and https://github.com/ stanfordnmbl/opencap-processing (archived in Zenodo: https://doi.org/10.5281/zenodo.7419973) under a permissive open-source license (Apache 2.0), and the software can be used through our web and mobile applications by visiting https:// opencap.ai. These applications and the cloud computing that enables them will be freely available to the research community for the foreseeable future.

Funding: SDU, AF, LK, JM, ASC, JLH, and SLD were supported by the National Institutes of Health (https://www.nih.gov; grant 1P41EB027060-01A1) and the Wu Tsai Human Performance Alliance (https://humanperformancealliance.org). ASC and MK were supported by Philips Healthcare (https:// www.usa.philips.com/healthcare) and the National Institutes of Health (https://www.nih.gov; grant 1R01AR077604-01). The funders had no role in study design, data collection and analysis, decision to publish, or preparation of the manuscript.

Competing interests: I have read the journal’s policy and the authors of this manuscript have the following competing interests: SDU and AF are cofounders of Model Health, Inc., which supports the non-academic, commercial use of the open-source software described here. OpenCap, the clouddeployed academic version of this open-source software is hosted at Stanford University and will remain freely available to the academic research community for the foreseeable future. No other authors have competing interests to declare.

incorporated into clinical practice and large-scale research studies. The advent of computer vision methods for locating human joints from standard videos offers a promising alternative to laboratory-based movement analysis. However, it is unclear whether these methods provide sufficient information for informing biomedical research and clinical practice. Here, we introduce OpenCap, an open-source, web-based software tool for computing the motion (e.g., joint angles) and the musculoskeletal forces underlying human movement (e.g., joint forces) from smartphone videos. OpenCap combines advances in computer vision, machine learning, and musculoskeletal simulation to make movement analysis widely available without specialized hardware, software, or expertise. We validate OpenCap against laboratory-based measurements and show its usefulness for applications including screening for disease risk, evaluating intervention efficacy, and informing rehabilitation decisions. Finally, we highlight how OpenCap enables large-scale human studies of human movement in real-world settings.

## Introduction

Evaluating the dynamics (i.e., musculoskeletal forces) and control of human movement is important for understanding and managing musculoskeletal and neuromuscular diseases. For example, the loading in osteoarthritic joints predicts osteoarthritis progression [1], the distribution of moments generated by muscles about lower-extremity joints when rising from a chair relates to falling in older adults [2–4], and the between-limb asymmetry of muscle and ground reaction forces while performing demanding tasks relates to functional outcomes after joint surgery [5–7]. Despite their utility, metrics of movement dynamics are rarely measured in clinical practice. Instead, visual movement evaluations or general functional tests that require basic instruments, like a stopwatch or goniometer, are used to inform clinical decisions and as outcomes for clinical trials.

The quantitative analysis of movement dynamics can provide deeper and more reproducible insights than visual evaluations and simple functional tests; however, this analysis is resource intensive, which has impeded its use in large-scale studies and clinical practice. Traditionally, motion analysis requires a fixed lab space with more than \$150,000 of equipment (Fig 1, top row). Kinematics (e.g., joint angles) are measured with a marker-based motion capture system that uses eight or more specialized cameras to capture the three-dimensional (3D) trajectories of markers placed on a subject. Dynamic measures (e.g., joint moments and powers), also referred to as kinetics, can be estimated with the additional measurement of ground reaction forces from force plates mounted beneath the floor. Musculoskeletal modeling and simulation tools [8–10] combine measures of kinematics, kinetics, and muscle activation from electromyography to enable deeper investigations of motor control and musculoskeletal loading (e.g., muscle coordination and joint forces). This comprehensive analysis of movement is infrequently used outside of small-scale research studies because collecting data on a single participant, processing it, and, optionally, generating dynamic musculoskeletal simulations typically takes several days for a trained expert.

Studies of movement dynamics with hundreds of participants have elucidated biomechanical markers that predict injury risk or surgical outcomes [11–13]. However, studies of this scale are expensive and rare—the median number of subjects included in biomechanics studies is between 12 and 21 [14,15]. There is a need for inexpensive, scalable, and accurate tools for estimating movement dynamics on orders of magnitude more individuals in their natural environments. Modern data science techniques could then leverage these large datasets to

## Mocap: marker-based motion analysis

![](images/0013c81ebda6996061c4cb3b1a34999bf229873bc3d92a1de9f855472680c685.jpg)  
Fig 1. Marker-based motion capture (Mocap) versus video-based (OpenCap) analysis of human movement dynamics. (Top row) Marker-based movement analysis usually occurs in a motion capture laboratory, and a comprehensive study of musculoskeletal dynamics typically requires more than two days of an expert’s time and equipment worth more than \$150,000. (Bottom row) OpenCap enables the study of musculoskeletal dynamics in less than 10 minutes of hands-on time and with equipment worth less than \$700 (assuming users need to purchase new mobile devices). OpenCap can be used anywhere with internet access and requires a minimum of two iOS devices (e.g., iPhones or iPads). (Right panel) OpenCap enables the estimation of kinematic, dynamic, and musculotendon parameters, many of which were previously only accessible using marker-based motion capture and force plate analysis.  
https://doi.org/10.1371/journal.pcbi.1011462.g001

explore the role of movement in health and disease, facilitating the identification and clinical translation of quantitative movement biomarkers.

Mobile tools for estimating kinematics have been developed, but most are still too expensive and time consuming for large-scale research studies and clinical translation, and none enable full-body analysis of movement dynamics. Inertial measurement units, the most widely used of these tools, can accurately estimate kinematics [16], but commercially available sensors remain expensive, time-consuming to don and doff, and utilize proprietary algorithms. Recent advances in physics-based simulation of the musculoskeletal system have estimated dynamics from inertial measurement unit–based motion capture [17,18], but these algorithms are not publicly available and have not been translated beyond small-scale feasibility studies.

Measuring kinematics with video cameras is another promising approach made possible by recent advancements in human pose estimation algorithms [19,20]. Open-source, two-dimensional (2D) pose estimation algorithms (e.g., OpenPose [21]) have enabled 2D kinematic analyses [22] and can generate inputs for machine learning models that predict kinematic and dynamic measures [23,24]. While these machine learning models are useful for specific applications, they may not generalize to other measures, tasks, and populations not represented in their training data. Another potentially more generalizable approach is to triangulate the body keypoints (e.g., joint centers) identified by pose estimation algorithms on multiple videos [25– 30] and track these 3D positions with a musculoskeletal model and physics-based simulation. However, the sparse set of 3D keypoints identified by these algorithms does not fully characterize the translations and rotations of all body segments; thus, it is unclear whether these key points are expressive and accurate enough to inform movement research. Commercial markerless motion capture systems accurately estimate kinematics [31], but they typically

require many wired cameras, proprietary software, and specialized computing resources. The ubiquity of smartphone cameras could enable video-based motion capture without the need to purchase specialized equipment, but it is unclear whether kinematics can be accurately estimated from a small number of devices that lack hardware synchronization. If the challenges of computing accurate kinematics and dynamics from smartphone video can be addressed, smartphone-based analysis of musculoskeletal dynamics has the potential to overcome the translational barriers faced by current movement analysis technologies.

Here we introduce OpenCap, open-source, web-based software that is freely available to the research community for estimating the 3D kinematics and dynamics of human movement from videos captured with two or more smartphones (Fig 1, bottom row). OpenCap brings together decades of advances in computer vision and musculoskeletal simulation to make the analysis of movement dynamics available without specialized hardware, software, or expertise. We first validate kinematic and dynamic measures estimated with OpenCap against gold standard measures computed with marker-based motion capture and force plates. Next, we explore whether OpenCap estimates dynamic measures with sufficient accuracy to be used for disease risk screening, evaluating intervention efficacy, studying between-group movement differences, and tracking rehabilitation progress. After validating these measures in the laboratory, we highlight how OpenCap enables clinicians to measure movement dynamics in large cohorts in real-world settings.

## Results

## Data collection with OpenCap

Setting up a data collection with OpenCap takes under five minutes and requires two iOS devices (iPhone, iPad, or iPod), two tripods, a calibration checkerboard (printed with a standard printer), and another device to run OpenCap’s web application (e.g., a laptop). After pairing the iOS devices to the web application, users are guided through camera calibration, data collection, and visualization of 3D kinematics. Kinematics are estimated from video using deep learning models and inverse kinematics in OpenSim [8,10], and dynamics are estimated using a physics-based musculoskeletal simulation approach (Fig 2, S1 Movie). OpenCap leverages cloud computing for data processing using a scalable server architecture.

## Validation against the lab-based gold standard

We validated OpenCap using two iPhones against marker-based motion capture and force plate analysis in a cohort of ten healthy individuals for several activities (walking, squatting, rising from a chair, and drop jumps). OpenCap estimated joint angles with a mean absolute error (MAE) of 4.5˚, ground reaction forces with an MAE of 6.2% bodyweight, and joint moments with an MAE of 1.2% bodyweight\*height (Table 1; additional validation in Methods: Validation; S1–S4 Tables and Figs A-L in S1 Appendix).

## Disease risk screening: Knee loading during walking

We then explored whether OpenCap is sufficiently accurate to estimate measures of joint loading that could be used to screen for individuals at risk of rapid progression of medial knee osteoarthritis and to evaluate the efficacy of a non-surgical intervention. We first evaluated how accurately OpenCap estimates the early-stance peak knee adduction moment, which predicts rapid progression of medial knee osteoarthritis [1]. The ten healthy individuals walked naturally (i.e., with a self-selected strategy) and with a trunk sway gait modification that typically reduces the knee adduction moment [32]. OpenCap predicted the early-stance peak knee

Ground reaction forces (MAE)  
![](images/26b9f4cd078e478fb9e3fd9058ddc664c7e3e4441392582a08adbcd74ff6d015.jpg)  
iOS application simultaneous video capture

![](images/c2bb7049a8205f27487072b6cb9e2048086c2464c34f65700c8eb2ffb5d41f70.jpg)  
Web application data collection and visualization

![](images/d90981a44e224083ac2579a3ff0dd98545dc1ac7ff971f34bea4d344db9f3da1.jpg)  
Scalable cloud computing biomechanical analysis from video

Fig 2. OpenCap comprises a smartphone application, a web application, and cloud computing. To collect data, users open an application on two or more iOS devices and pair them with the OpenCap web application. The web application enables users to record videos simultaneously on the iOS devices and to visualize the resulting 3-dimensional (3D) kinematics. In the cloud, 2D keypoints are extracted from multi-view videos using open-source pose estimation algorithms. The videos are time synchronized using cross-correlations of keypoint velocities, and 3D keypoints are computed by triangulating these synchronized 2D keypoints. These 3D keypoints are converted into a more comprehensive 3D anatomical marker set using a recurrent neural network (LSTM) trained on a large motion capture dataset. 3D kinematics are then computed from marker trajectories using inverse kinematics and a musculoskeletal model with biomechanical constraints. Finally, dynamic measures are estimated using muscle-driven dynamic simulations that track 3D kinematics.

https://doi.org/10.1371/journal.pcbi.1011462.g002

Table 1. Mean absolute error (MAE) in kinematics and kinetics from OpenCap compared to laboratory-based motion capture and force plates.
<table><tr><td rowspan=1 colspan=1>Kinematics (MAE)</td><td rowspan=1 colspan=1>Walking</td><td rowspan=1 colspan=1>Squat</td><td rowspan=1 colspan=1>Sit-to-stand</td><td rowspan=1 colspan=1>Drop jump</td><td rowspan=1 colspan=1>Mean</td></tr><tr><td rowspan=1 colspan=1>Rotations (n = 18) [°]</td><td rowspan=1 colspan=1>4.1 (2.3–6.6)</td><td rowspan=1 colspan=1>4.1 (1.8–7.2)</td><td rowspan=1 colspan=1>4.7 (1.7–10.3)</td><td rowspan=1 colspan=1>5.1 (2.3–8.6)</td><td rowspan=1 colspan=1>4.5</td></tr><tr><td rowspan=1 colspan=1>Translations (n = 3) [mm]</td><td rowspan=1 colspan=1>12.3 (6.8–19.6)</td><td rowspan=1 colspan=1>12.3 (5.8–18.4)</td><td rowspan=1 colspan=1>13.2 (5–20.3)</td><td rowspan=1 colspan=1>11.5 (6.3–16.5)</td><td rowspan=1 colspan=1>12.3</td></tr></table>

<table><tr><td rowspan=1 colspan=1>Vertical [%BW]</td><td rowspan=1 colspan=1>8.2 (7.5%)</td><td rowspan=1 colspan=1>6.4 (20.0%)</td><td rowspan=1 colspan=1>5.7 (13.4%)</td><td rowspan=1 colspan=1>25.2 (13.8%)</td><td rowspan=1 colspan=1>11.4 (13.7%)</td></tr><tr><td rowspan=1 colspan=1>Anterior-posterior [%BW]</td><td rowspan=1 colspan=1>2.1 (6.7%)</td><td rowspan=1 colspan=1>1.3 (37.5%)</td><td rowspan=1 colspan=1>1.9 (31.0%)</td><td rowspan=1 colspan=1>8.9 (17.3%)</td><td rowspan=1 colspan=1>3.5 (23.1%)</td></tr><tr><td rowspan=1 colspan=1>Medio-lateral [%BW]</td><td rowspan=1 colspan=1>1.1 (17.1%)</td><td rowspan=1 colspan=1>5.7 (85.4%)</td><td rowspan=1 colspan=1>3.2 (110.5%)</td><td rowspan=1 colspan=1>5.3 (29%)</td><td rowspan=1 colspan=1>3.8 (60.5%)</td></tr></table>

<table><tr><td rowspan=1 colspan=1>All degrees of freedom (n = 15) [%BW*ht]</td><td rowspan=1 colspan=1>0.75 (0.20–1.32, 19%)</td><td rowspan=1 colspan=1>0.97 (0.11–1.93, 45%)</td><td rowspan=1 colspan=1>0.68 (0.13–1.09, 60%)</td><td rowspan=1 colspan=1>2.50 (1.15–5.90, 25%)</td><td rowspan=1 colspan=1>1.22 (37%)</td></tr></table>

Errors for each activity were averaged over trials and participants (n = 10), and the reported mean is an average over activities and degrees of freedom (six for pelvis position and orientation [kinematics only], three for the lumbar, three per hip, one per knee, and two per ankle). Forces are expressed in percent bodyweight (BW) and moments in percent BW times height (ht). Kinematic and joint moment errors are presented as the mean and range over the degrees of freedom, and kinetic errors are additionally presented as the MAE as a percentage of the range. Root mean squared error in kinematics and kinetics are available in S2–S4 Tables. Average kinematic, ground reaction force, and joint moment waveforms estimated using OpenCap and Mocap are presented in Figs A–L in S1 Appendix.

![](images/fe3d5b7173eff357394312d49b076befe6ac9070959ca41856eee5e898af6e82.jpg)

![](images/582359c404c74eefa76fc492c5a8ef9c23dc9f6899a884c865caad9d271cbeff.jpg)

![](images/738e6af90071182df80e83fdba2544a637eab6cf369d4356cd938675a94bf96f.jpg)  
Fig 3. Medial knee loading during walking. We evaluated how accurately OpenCap estimates the knee adduction moment (KAM), a measure of medial knee loading that predicts knee osteoarthritis progression, and how knee loading changes with a modified walking pattern. Participants (n = 10) walked naturally and with a trunk sway gait modification. (A) OpenCap estimated the early-stance peak KAM with $r ^ { 2 } = 0 . 8 0$ , compared to an analysis using marker-based motion capture and force plates (Mocap). The KAM is normalized by bodyweight (BW) and height (ht). (B) The mean (bar) and standard deviation (error bar) across participants (circles) are shown for the changes in the peak KAM and peak medial contact force (MCF), which is a more comprehensive measure of medial knee loading, from natural to trunk sway walking $( ^ { * } P < . 0 5 ) .$ OpenCap detected the reductions in peak KAM and MCF (P < .006, t test and Wilcoxon signed rank test) that were measured with Mocap (P < .016, t tests). Furthermore, OpenCap correctly identified the one individual who did not reduce KAM or MCF as estimated by Mocap (filled circles).

https://doi.org/10.1371/journal.pcbi.1011462.g003

adduction moment with $r ^ { 2 } = 0 . 8 0$ (r: Pearson correlation coefficient) and an MAE of 0.30% bodyweight\*height compared to marker-based motion capture and force plates (Fig 3). This error is smaller than a range of thresholds for detecting knee osteoarthritis symptoms and progression (0.5–2.2% bodyweight\*height [1,33–35]). We then evaluated whether OpenCap could estimate changes induced by the trunk sway modification in the peak knee adduction moment as well as the peak medial contact force, which is a more comprehensive loading metric that is often targeted by knee osteoarthritis interventions [36,37]. At the group level, OpenCap captured expected reductions in the early-stance peak knee adduction moment and peak medial contact force from the trunk sway gait modification (16–33% reductions, $P < . 0 0 6 ;$ t test and Wilcoxon signed rank test, n = 10, Fig 3B). Significant changes in the same direction were also detected with motion capture and force plates (21–46% reductions, $P < . 0 1 6 ;$ t tests, n = 10); further details about these statistical tests can be found in Table A in S1 Appendix. For this sample size, OpenCap had a 92% chance (post-hoc power averaged across tests) of detecting these expected group differences at the significance level alpha = .05, compared to the 77% chance from motion capture and force plates. At the individual level, OpenCap correctly predicted the directional change in both peak loading measures (decrease for nine individuals and increase for one individual) induced by trunk sway. OpenCap’s ability to accurately estimate knee loading and changes in loading during walking suggests that it could be used to identify individuals with medial knee osteoarthritis who may be at risk of rapid disease progression and to evaluate the effect of a gait modification on individual and group levels [38,39].

## Detecting between-group differences: Kinetic differences during sit-tostand

We then explored whether OpenCap is useful for studying differences in movement dynamics that commonly exist between young and older adults. Strategies for rising from a chair vary with age and are associated with different muscle force requirements [2]. Older adults often use a rising strategy with increased trunk flexion, which shifts the muscular demand from the

![](images/14b286a74e0ed1b3d20818b4fd14f3efa76ab3596b817de7ae57d156a1a6181d.jpg)

![](images/aea261301d869783af9ff7618d0c5e07290fe4ef136b6efab58e0cd7f1bba665.jpg)

![](images/e3235b8e91474e78f69d70e2ab03817beb41b8d9bfd58df0dc59348728b747c3.jpg)  
Fig 4. Distribution of lower-extremity joint moments when rising from a chair. To evaluate OpenCap’s ability to detect between-group differences in dynamics, we computed differences in lower-extremity joint moments while rising from a chair that commonly exist between young and older adults. Individuals (n = 10) stood naturally and with increased trunk flexion, a strategy used by individuals with knee extensor weakness that shifts muscle demand to the hip extensors and ankle plantarflexors. (A) The mean (bar) and standard deviation (error bar) across participants (open circles) are shown for the changes in knee extension, hip extension, and ankle plantarflexion moments, averaged over the rising phase, from the natural to trunk flexion condition $( ^ { * } P < . 0 5 )$ Moments are normalized to percent bodyweight (BW) times height (ht). OpenCap identified the changes in joint moments (P = .004–.024, t tests) that were identified with motion capture and force plates (Mocap, P = .001–.002, t tests). (B) The rising-phase-averaged knee extension moment values for each participant and condition are shown. OpenCap estimated the knee extension moment with $\stackrel { \smile } { r ^ { 2 } } = 0 . 6 5$ compared to simulations that used motion capture and force plate data as input (Mocap).

https://doi.org/10.1371/journal.pcbi.1011462.g004

knee extensors to the hip extensors and ankle plantarflexors [40]; this strategy is associated with low functional muscle strength [4], which relates to fall risk [3]. We simulated differences in rising strategies between age groups by instructing ten healthy individuals to rise from a chair five times naturally, then five times with increased trunk flexion (Fig 4). At the group level, OpenCap estimated the expected reduction in the knee extension moment $( P = . 0 2 4 , \mathrm { t }$ test, n = 10) and increase in the hip extension (P = .020, t test, n = 10) and ankle plantarflexion moments $\left( P = . 0 0 4 , \mathrm { t } \mathrm { t e s t } , \mathrm { n } = 1 0 \right)$ , averaged over the rising phase, from the natural to the increased trunk flexion condition. The direction of these changes matched what was measured with motion capture and force plates (P = .002–.003, t tests, n = 10); further details about these statistical tests can be found in Table B in S1 Appendix. For this sample size, OpenCap had a 65% chance (post-hoc power averaged across tests) of detecting these expected between-condition differences at the significance level alpha = .05, compared to the 89% chance from motion capture and force plates. OpenCap also predicted the peak knee extension moment with $r ^ { 2 } = { }$ 0.65 and MAE = 5.5Nm (0.47%bodyweight\*height) compared to marker-based motion capture and force plates. This error is similar to the average loss in strength that occurs over six years in middle-aged adults (0.93Nm/year [41], see S1 Appendix for further discussion). Together, these findings suggest that OpenCap can be used to study differences in movement dynamics between young and older adults and can identify individuals with low knee extensor strength who may benefit from muscle strengthening interventions [2].

## Informing rehabilitation: Muscle activation asymmetry during squatting

Finally, we explored whether OpenCap can accurately estimate measures of muscle force associated with rehabilitation progress. Restoring between-limb symmetry in knee extensor muscle force generation is often a goal of rehabilitation following knee surgeries, and identifying persistent asymmetry prior to rehabilitation discharge can prevent poor functional outcomes [5,6,42]. To simulate post-surgical asymmetries, we instructed the ten healthy individuals to perform five squats naturally, then asymmetrically by reducing the force under their left foot

![](images/88717d0ffa101135d781b6003b8527d570b030d46cd0003437fa1e53ff4cc8b6.jpg)

![](images/f7488b85f17b5c809599c71653768c0d99f575af0b9a93dbf0c7528342a41c45.jpg)

![](images/e9c0368835b16d5301cf4f5f90af934c5ce580aa622f84a561e3353750c7aa8a.jpg)

![](images/703156d2b13c7fe67701ae5f5e065888813b4c48803f0d59fdfc996b8a8b1c26.jpg)  
Fig 5. Asymmetry in vasti muscle activation during squatting. To assess the utility of OpenCap for informing rehabilitation decisions, we sought to identify between-limb asymmetries in knee extensor muscle (vasti) function that indicate incomplete rehabilitation and relate to poor post-surgical functional outcomes. Participants (n = 10) performed squats naturally, then asymmetrically, where they were instructed to reduce the force under the left foot. (A, B) The mean (line) and standard deviation (shading) across participants are shown for the vasti muscle activation of the left (unweighted) leg measured with electromyography (EMG) and estimated using OpenCap. Muscle activations are normalized by the maximum value for each participant and measurement modality. (C) OpenCap identified peak vasti activation asymmetry between the left and right leg (asymmetry defined from EMG and clinically relevant symmetry threshold), with area under the receiver operator characteristic curve (AUC) of 0.83 and accuracy of 75%. This was similar to the performance of simulations that used marker-based motion capture and force plate data as input (Mocap sim., AUC = 0.82, accuracy = 70%).  
https://doi.org/10.1371/journal.pcbi.1011462.g005

(Fig 5). Since muscle activation can be measured more directly than muscle force, we compared vasti (knee extensor) muscle activation measured with electromyography to activation estimated with OpenCap. We defined ground truth activation asymmetry using electromyography and a clinically relevant symmetry index threshold [43] of 1.15. OpenCap classified squats as being symmetric or asymmetric with an area under the receiver operator characteristic curve (AUC) of 0.83 and an accuracy of 75% at the optimal symmetry index threshold of 1.13 (Fig 5), which was similar to the performance of simulations that used motion capture and force plate data (AUC = 0.82, accuracy = 70%).

## Out-of-lab dynamic analysis

To demonstrate OpenCap’s utility in real-world conditions, we extended this analysis of rehabilitation tracking to a field study. A clinician, who was not an expert in movement analysis, used OpenCap to evaluate knee extension moment symmetry in 100 individuals performing natural and asymmetric squats in the community. On average, set up and data collection took five minutes per participant, and for a single squat, kinematics and kinetics were computed automatically in two and 35 minutes on a single server, respectively. In total, data collection took eight hours for 100 subjects, and computation took 31 hours on a 32-thread CPU (kinetic computation was parallelized). OpenCap’s peak knee extension moment estimates could discriminate between the symmetric and asymmetric conditions with AUC = 0.90 and accuracy = 85% at the optimal symmetry index threshold of 1.33 when using the condition instruction (i.e., natural or asymmetric) as ground truth (Fig 6A and 6B). OpenCap also detected within-subject improvements in peak knee extension moment symmetry from the asymmetric to the natural condition with AUC = 0.93 and accuracy = 89% at the optimal threshold of 0.26 (Fig 6C and 6D). Together, our lab and field studies demonstrate that Open-Cap can detect asymmetries in vasti force generation that may be useful for guiding rehabilitation decisions and can track improvements in symmetry expected to occur over the course of rehabilitation.

![](images/b7ca3a7489a52eb608518b6722593a1e6b1838e5a9ada045ebd3dff583b4e5c8.jpg)

![](images/fc47df3ffecf0c243500d1f7678b20b27940c8c5e73fc10298d839c3169c7c18.jpg)

![](images/7ee419a51fbc43d0492ace6efcbea660a40ef4f5a195ddce7e994b09316769e0.jpg)

![](images/19200c1c40459c1134a81d357de90ac69078fa1af84665fd6e012e0739100ffd.jpg)

![](images/6b844e81eae7af380f115b62427f7f51898e21b001c55933cf6ac41aecb0d084.jpg)  
Fig 6. Out-of-lab dynamic analysis. To demonstrate the practical utility of OpenCap for tracking rehabilitation progress, we enrolled 100 participants in a clinician-led field experiment. Participants performed symmetric squats and asymmetric squats, where they were instructed to reduce the force under the left foot, which likely resulted in an asymmetry between the left and right knee extension moments. We first evaluated the utility of OpenCap as a screening tool to detect peak knee extension moment asymmetries. (A) The distributions of knee extension moment symmetry indice for both squat conditions are shown, with a symmetry index larger than one indicating a lower peak knee extension moment for the left (unweighted) leg compared to the right leg. (B) OpenCap’s symmetry index estimates classified between natural and asymmetric squats with an area under the receiver operator characteristic curve (AUC) of 0.90 and accuracy of 85%. We then evaluated the utility of OpenCap for detecting changes in peak knee extension moment symmetry that would be expected to occur over time during rehabilitation. (C) The distributions of the average difference in the symmetry index between the asymmetric and natural conditions (i.e., hypothetical improved symmetry over time; red) and the average difference in the symmetry index between the three trials in the asymmetric condition (i.e., hypothetical unchanged symmetry over time; gray) are shown. (D) OpenCap detected improvements in symmetry from the asymmetric to the natural condition with $\Delta \mathrm { U C } = \dot { 0 } . 9 3$ (improved compared to unchanged distributions from (C) and accuracy = 89%.

https://doi.org/10.1371/journal.pcbi.1011462.g006

## Discussion

This study describes OpenCap, a platform that combines computer vision and musculoskeletal simulation to quantify human movement dynamics from smartphone videos. We showed that OpenCap is sufficiently accurate for several research and clinical applications. OpenCap estimated changes in dynamic measures between conditions with similar statistical power (0.65– 0.92) as the gold standard technique (0.77–0.89), it estimated dynamic measures that predict adverse outcomes related to osteoarthritis and fall risk with $r ^ { 2 } = 0 . 6 5  – 0 . 8 0$ , and it estimated dynamic measures that can inform rehabilitation decision making with classification accuracies of 75–89%. Our field study demonstrated how OpenCap enables clinicians and researchers alike to analyze movement dynamics in the field and in large populations.

OpenCap reduces the cost, time, and expertise barriers to analyzing movement dynamics. OpenCap’s hardware can be acquired for between \$40 and \$700, depending on whether users need to purchase new iOS devices (Fig 1). This is about 215 times cheaper than traditional motion capture laboratory equipment, and it does not require a dedicated laboratory space.

Hands-on time for measuring movement dynamics with OpenCap is the time for setting up the mobile devices and the time to perform the movements (Fig 1, bottom row). During our in-field data collection, the five minutes of hands-on time per participant was about 25 times less than a comparable analysis in a motion capture laboratory (about 2 hours). OpenCap does not require specialized software or expertise, which bridges gaps between the computer vision, biomechanics, and clinical movement science communities. Most computer vision algorithms require computer science knowledge to run and most simulation tools require biomechanics knowledge to operate, but OpenCap automates these processes, making advancements in these fields more accessible to clinicians and researchers. In the year since releasing OpenCap, over 2,000 researchers from a variety of fields have used it to collect tens of thousands of motion trials. OpenCap also meets Stanford University’s security requirements for cloud based systems using high-risk data (e.g., protected health information) and ensures end-toend data encryption. To further facilitate ease of use, we provide tutorials and examples on a companion website (https://opencap.ai/).

Our results demonstrate OpenCap’s potential clinical utility as a screening tool and for informing rehabilitation decisions. Future studies could test OpenCap’s ability to screen for risk of non-contact ligament injury in athletes [11,44], or to predict efficacy of surgery in individuals with cerebral palsy [12,13]. OpenCap assessments may also be fast enough to enable movement screens to become part of routine clinical care, allowing clinicians to track function over time, and following an injury or surgery, to benchmark rehabilitation status against preinjury measures [45]. We investigated OpenCap’s accuracy for estimating several clinically meaningful metrics; we encourage further validation studies for other activities, populations, and metrics of interest to evaluate whether OpenCap is sufficiently accurate to inform decisions in use cases beyond those that we explored.

By enabling large-scale, out-of-lab studies, OpenCap can accelerate movement research. OpenCap detected between-condition differences with similar statistical power as motion capture and force plate analysis, but in substantially less time. This accuracy and efficiency makes prospective injury risk studies that require hundreds of participants more feasible, enables the incorporation of movement dynamics into population-scale health studies that typically only use pedometry (e.g., the Osteoarthritis Initiative [46] or the UK Biobank [47]), and facilitates the development of more sensitive functional outcome measures for clinical trials. By automatically computing kinematics, OpenCap is not susceptible to errors introduced by betweenexperimenter variance in motion capture marker placement [48]. This could reduce variability in multi-center studies [49] and enable movement data to be compiled into homogeneous, sharable datasets that are useful to the machine learning community. Importantly, OpenCap’s portability will enable studies of populations that are often underrepresented in movement research due to time and geographic constraints.

OpenCap uses machine learning to improve the fidelity of video-based kinematic estimates and physics-based modeling to maintain generalizability (Fig 2). We combined a deep learning model suited for time series data with a constrained biomechanical model to estimate 3D kinematics from video keypoints that, alone, are insufficient to characterize 3D kinematics. Our deep learning model predicts a comprehensive set of anatomical markers from the sparse video keypoints labeled in common computer vision datasets (e.g., COCO [50]). Using the predicted anatomical markers, instead of video keypoints, with our biomechanical model improved kinematic accuracy, averaged across all degrees of freedom, by 3.4˚ (see Methods and S2 Table). These improvements were greatest for the hip flexion, pelvic tilt, and lumbar flexion degrees of freedom (4.9–32.6˚ improvement), which are susceptible to large angular errors due to the sparsity of video keypoints between the hips and shoulders. Additionally, the deep learning model architecture provides temporal consistency, making OpenCap more

robust to brief occlusions or mis-identified keypoints. Finally, despite some task-dependent tuning of the problem formulation, muscle-driven dynamic simulations are a more generalizable approach for estimating dynamics than an end-to-end machine learning approach. This enabled us to study the dynamics of different movements without training data for each activity.

The accuracy of OpenCap’s kinematic and kinetic estimates is similar to state-of-the art markerless motion capture solutions. OpenCap’s kinematic error (range of root mean squared error [RMSE] across lower-extremity degrees of freedom: 2.0–10.2˚) is similar to errors reported for inertial measurement unit–based approaches (RMSE: 2.0–12˚ for walking, running, and daily living activities [18,51–57]) and commercial and academic video-based systems with eight cameras (RMSE: 2.6–11˚ for walking, running, and cycling activities [31,58]). Additionally, the joint angle errors between OpenCap and marker-based motion capture are similar to errors induced by skin motion artifact in marker-based motion capture. For example, the knee flexion angle computed with skin-mounted markers can have a root-mean-square error of up to 8˚ compared to the angle measured with bone pin–mounted markers [59], which exceeds the 4.2˚ root-mean-square error in knee flexion angle between OpenCap and skin marker–based motion capture (S2 Table). Dramatic improvements in the concordance between markerless and marker-based motion capture systems may be challenging to demonstrate due to the errors in the skin marker–based motion capture approach used as ground truth. Furthermore, in contrast with most inertial measurement unit–based approaches, OpenCap estimates global translations (e.g., pelvis displacement), enabling estimation of whole-body measures like center-of-mass trajectory. Interestingly, kinematic estimates did not substantially improve when using more than two cameras (see Methods and S2 Table), suggesting that two cameras are sufficient for analyzing activities like those included in this study. To our knowledge, there is no previous example of computing whole-body kinetics from video alone; however, OpenCap’s kinetic estimates are comparable to inertial measurement unit– based approaches. For example, OpenCap’s root mean square errors in ground reaction force (1.5–11.1% bodyweight) and lower-extremity joint moment (0.3–1.7% bodyweight\*height) predictions during walking (S3 and S4 Tables) are comparable to those resulting from using a 17-sensor inertial measurement unit suit (1.7–9.3% bodyweight and 0.5–2.2% bodyweight\*height, respectively [18,60]). OpenCap also predicted the first peak knee adduction moment during walking with 44% higher accuracy than a machine learning model trained specifically to predict this measure from marker positions that could be extracted from video [24].

OpenCap transforms the outputs of pose detection algorithms into valuable insights for studying human movement. We designed OpenCap to integrate different pose detection algorithms, and we found only minor differences in kinematics when testing different algorithms (see Methods and S1 and S2 Tables). With the recent advances in joint center estimation from single-view [26,30,61,62] and multi-view [26–30] video, we expect OpenCap’s accuracy in estimating kinematics and kinetics to improve as more accurate pose estimation algorithms are released. By sharing our data and source code, we encourage researchers to benchmark their models using our data and to contribute to OpenCap’s development by adding support for their models.

Our study has several limitations. First, we tested OpenCap’s ability to estimate informative dynamic measures by having healthy individuals simulate different movement patterns associated with pathology or treatment. While the simulated movements were similar to those reported in the populations of interest (see Methods: Applications and Statistics), and Open-Cap could distinguish differences in dynamics between these simulated conditions, future work is needed to validate these measures in the populations of interest. Furthermore, the relationship between some kinetic measures and clinical outcomes has been established in the

literature; however, the level of accuracy at which these metrics need to be estimated to inform clinical decisions is unknown in most cases because establishing these thresholds requires resource-intensive longitudinal studies. Where possible, we compared OpenCap’s accuracy to clinically meaningful thresholds (e.g., the knee adduction moment). We hope that OpenCap’s ease of use will enable more longitudinal studies that can establish the clinical utility of kinetic metrics and the required measurement accuracy. Second, we did not measure the repeatability of measurements obtained with OpenCap. However, based on previous repeatability studies using markerless motion capture and employing similar methodologies [49], we anticipate OpenCap to measure kinematics with similar repeatability to, if not better than, marker-based motion capture. Future studies are necessary to confirm this claim and assess repeatability for dynamics assessments. Third, the deep learning model that augments our 3D marker set may not generalize to activities outside of the distribution of activities that it was trained on. We generated the training data for this model using standard OpenSim kinematics data, so additional datasets could be added to the training set in the future. Additionally, estimating dynam ics requires some task-dependent user inputs, which is a limitation of any optimization-based muscle-driven simulation. We have provided optimization problem formulations that work well for several activities. Overall, if future applications require high accuracy rather than generalizability, OpenCap’s accuracy could likely be improved with task-specific tuning of the deep learning model and optimization problem formulation.

In conclusion, OpenCap allows non-experts to analyze human movement dynamics in an order of magnitude less time and for several orders of magnitude less money than was previously possible with marker-based motion capture and force plates. We expect that OpenCap will catalyze large-scale studies of human movement, the sharing of motion datasets, and the translation of movement biomarkers into clinical practice.

## Methods

## Ethics statement

The protocol for this study was approved and overseen by the Institutional Review Board of Stanford University (IRB00000351). We conducted the experiment in accordance with this approved protocol and relevant guidelines and regulations. All participants provided written informed consent before participation. Participants whose identifiable videos are shared publicly consented to sharing these videos.

## Design

OpenCap comprises several steps to estimate movement dynamics from videos. These steps include calibrating cameras, collecting and processing videos, estimating marker positions, estimating kinematics, and generating physics-based dynamic simulations of movements. This pipeline is implemented in Python (v3.7.10). OpenCap’s web application guides users through each step, and cloud instances are used for computing (Fig 2).

Camera calibration. OpenCap models the iOS device cameras using a fifteen-parameter pinhole camera model (https://github.com/smidm/camera.py) and computes parameters using OpenCV [63]. At the beginning of a data collection, OpenCap loads the pre-computed intrinsic parameters related to each device’s camera hardware and recording settings (principal point, focal length, and distortion parameters) from a database that we created of recent iOS devices. Next, the web application guides users to place a checkerboard in view of all cameras, and OpenCap automatically computes the extrinsic parameters (camera transformation relative to the global frame) from a single image of a checkerboard. We used a precision-manufactured 720x540 mm checkerboard to pre-compute the intrinsic camera parameters for each device in our database (see S1 Appendix for details about intra- and inter-phone intrinsic parameter testing). A 210x175 mm checkerboard printed on A4 paper and mounted to a flat surface is sufficient for computing extrinsic camera parameters during each data collection. We found minimal kinematic differences when using the printed checkerboard, compared to the precision-manufactured checkerboard, to calibrate the cameras (see S1 Appendix).

Video collection and pose estimation. After calibration, users can proceed with simultaneously recording videos on all devices through the web application. Videos are recorded at a resolution of 720x1280 pixels, a frame rate of 60 Hz, and with the camera focus distance set to a fixed value.

Recorded videos are then processed using video pose detection algorithms. OpenCap currently supports two algorithms: OpenPose [21] and HRNet [64–67]. These algorithms were selected due to performance and the inclusion of foot keypoints. For each video, and at each time frame, both algorithms return the two-dimensional (2D) position of body keypoints as well as a confidence score (between 0 and 1) indicating the confidence of the algorithm in the keypoint position. Twenty body keypoints are included for further analysis (neck, mid hip, left and right shoulders, hips, knees, ankles, heels, small and big toes, elbows, and wrists). Open-Cap implements custom algorithms for processing 2D keypoint positions (e.g., handling keypoint occlusion) and time synchronizing them across videos using cross-correlations of keypoint velocities (see S1 Appendix for details).

Triangulation and marker-set augmentation. OpenCap triangulates the synchronized 2D video keypoint positions to compute 3D positions. OpenCap uses a Direct Linear Transformation algorithm for triangulation [68], and weights the contribution of individual cameras in the least-squares problem with the corresponding keypoint confidence score [58]. There are two major limitations of using 3D keypoint positions triangulated from video for biomechanical analysis. First, the video keypoint set is not sufficient to fully define the kinematics of all degrees-of-freedom of the body segments. Tracking these limited keypoints using a model with biomechanical joint constraints mitigates this issue for some, but not all body segments. For example, keypoints at the hips and shoulders are insufficient for robustly determining sagittal-plane hip, pelvis, and lumbar kinematics. Second, most pose estimation algorithms identify keypoints on a frame-by-frame basis, so the resulting 3D keypoint trajectories are often physically unrealistic, especially in the presence of misidentified or occluded keypoints.

To overcome these limitations, we trained two long short-term memory (LSTM) networks to predict the 3D positions of 43 anatomical markers from the 3D positions of the 20 triangulated video keypoints. The set of anatomical markers corresponds to what is commonly used for marker-based motion capture [69] to robustly determine 3D joint kinematics. We chose LSTM networks as they leverage time series data, which may improve the temporal consistency of the output marker position trajectories. We trained two LSTM networks: an arm model to predict the positions of eight arm markers from the positions of seven arm and torso keypoints, and a body model to predict the positions of 35 body markers from the positions of 15 lower-limb and torso keypoints. Both models also use height and weight as inputs. To train the networks, we synthesized corresponding pairs of 3D video keypoints and 3D anatomical markers from 108 hours motion capture data processed in OpenSim from published biomechanics studies [70–79] (see S1 Appendix for details on dataset generation). We split the data into a training set (\~80%), validation set (\~10%), and test set (\~10%). Prior to training, we expressed the 3D positions of each marker with respect to a root marker (the midpoint of the hip keypoints), normalized the 3D positions by the subject’s height, sampled at 60 Hz, split the data into non-overlapping time-sequences of 0.5 s, and added Gaussian noise (standard deviation: 18 mm) to each time step of the video keypoint positions based on a range of previously reported keypoint errors [24,26,29]. For both models, we tuned hyperparameters using a

random search. The RMSEs on the test set were 8.0 and 15.2 mm for the body and arm model, respectively (see S1 Appendix for details about model architecture and training). In practice, OpenCap uses both LSTM networks to predict root-centered arm and body anatomical marker positions from root-centered 3D video keypoints. It then adds the root keypoint position to all predicted positions.

Physics-based modeling and simulation. After calibration, OpenCap’s web application guides users to record the participant in a standing neutral pose. OpenCap uses the anatomical marker positions estimated from the neutral pose to scale a musculoskeletal model to the participant’s anthropometry using OpenSim’s Scale tool. OpenCap uses the musculoskeletal model from Lai et al. [69,80] with modified hip abductor muscle paths according to Uhlrich et al. [79]. The musculoskeletal model comprises 33 degrees of freedom (pelvis in the ground frame [6], hips [2x3], knees [2x1], ankles [2x2], metatarsophalangeal joints [2x1], lumbar [3], shoulders [2x3], and elbows [2x2]). Note that since no markers are attached to the toes, no reli able estimates of metatarsophalangeal joint kinematics can be obtained. The metatarsophalangeal joint is nevertheless included when generating tracking simulations, since modeling that joint improves knee mechanics in muscle-driven simulations [81]. The musculoskeletal model is driven by 80 muscles actuating the lower-limb coordinates and 13 ideal torque motors actuating the lumbar, shoulder, and elbow coordinates. Ground reaction forces $( \mathrm { i . e . , }$ external forces) are modeled through six foot-ground contact spheres attached to the foot segments of the model [82,83]. Raasch’s model [84,85] is used to describe muscle excitation-activation coupling, and a Hill-type muscle model [86,87] is used to describe muscle-tendon dynamics and the dependence of muscle force on muscle fiber length and velocity. Skeletal motion is mod eled with Newtonian rigid body dynamics and smooth approximations of compliant Hunt-Crossley foot-ground contacts [88,89]. The dynamics of the ideal torque motors are described using linear first-order approximations of a time delay [83]. To increase computational speed, muscle-tendon lengths and velocities, and moment arms are defined as a polynomial function ofjoint positions and velocities [90]. The polynomial coefficients are fit to the output from OpenSim’s Muscle Analysis tool applied to 5000 randomly varied lower limb postures. Muscles are represented by ninth-order or lower polynomials, with RMSE of muscle-tendon length and moment arm lower than 1.5 mm compared to the original model.

After scaling, users can record any movement through OpenCap’s web application. Open-Cap then uses the anatomical marker positions estimated from the recorded videos and LSTM network to compute joint kinematics using OpenSim’s Inverse Kinematics tool and the scaled musculoskeletal model. Users can visualize the resulting 3D kinematics in the web application.

Finally, OpenCap can estimate dynamics using muscle-driven tracking simulations ofjoint kinematics. The tracking simulations are formulated as optimal control problems that aim to identify muscle excitations that minimize a cost function subject to constraints describing muscle and skeleton dynamics. The cost function $J ( \mathbb { E } \mathbb { q } 1 )$ includes squared terms for muscle activations (a) and excitations of the ideal torque motors at the lumbar, shoulder, and elbow joints $( e _ { t m } )$ . It also includes tracking terms (squared difference between simulated and reference data), namely tracking of experimental joint positions $( \tilde { q } ) .$ , joint velocities $( \tilde { \dot { q } } ) .$ , and joint accelerations $( \tilde { \ddot { q } } )$

$$
J = \int _ { t _ { 0 } } ^ { t _ { f } } w _ { 1 } a ^ { 2 } + w _ { 2 } e _ { t m } ^ { 2 } + w _ { 3 } \| \tilde { q } - q \| _ { 2 } ^ { 2 } + w _ { 4 } \| \tilde { \dot { q } } - \dot { q } \| _ { 2 } ^ { 2 } + w _ { 5 } \| \tilde { \dot { q } } - \ddot { q } \| _ { 2 } ^ { 2 } d t ,\tag{ð1Þ}
$$

where $t _ { 0 }$ and $t _ { f }$ are initial and final times, $w _ { i }$ with $i = 1 , . . . , 5$ are weights, and t is time. Experimental joint positions, velocities, and accelerations are low-pass filtered using fourth-order, zero-lag Butterworth filters (default cutoff frequencies are 12 Hz for gait trials and 30 Hz for non-gait trials). Each cost term is scaled with empirically determined weights. To avoid singular arcs [91], a penalty function is appended to the cost function with the remaining control variables [83,92]. Note that the optimal control problem formulation can be tailored to the activity of interest to incorporate activity-based knowledge by, for instance, adjusting the cost function, constraints, and filter settings (see S1 Appendix). The optimal control problems are formulated in Python with CasADi [93] (v3.5), using direct collocation and implicit formulations of the muscle and skeleton dynamics [83]. Algorithmic differentiation is used to compute derivatives [92], and IPOPT is used to solve the resulting nonlinear programming problems [94] with a convergence tolerance of 1e-4 (all other settings are kept to default).

Practical considerations. The outcomes of OpenCap can be influenced by environmental and experimental factors. To maximize the accuracy of results, we recommend users follow several best practices. First, the pose estimation models perform best when the participant is clearly visible in the video. Enhancing visibility can be achieved by wearing clothing articles that contrast with each other and the background. For example, blue pants and white shoes on a black floor would yield better results than black pants and black shoes on a black floor. It is also advisable for the participant to wear tight-fitting clothing to facilitate the detection of their joints. While these considerations likely improve OpenCap’s performance, to best represent out-of-lab data-collection conditions, we did not ask participants in this study to wear specific colors of clothing. Participant visibility can also be enhanced by avoiding excessive brightness within the field of view and choosing an uncluttered background.

Second, OpenCap performs best when all body segments are visible by at least two cameras at all times. OpenCap can handle temporary occlusions of body segments, but it is advisable to minimize occlusions by optimizing the position of the cameras. When using two cameras for activities with minimal subject displacement (e.g., squats, sit-to-stands, and drop jumps), we recommend positioning the cameras at approximately ±45˚ from the subject’s forward-facing direction. For activities like walking, placing the two cameras at about ±30˚ from the subject’s walking direction can reduce occlusions and increase the length of the capture volume.

Third, when using two cameras, the size of the capture volume is maximized when the subject enters the field of view of both cameras at the same location in the volume and exits the field of view of both cameras at the same location. For example, when positioning cameras for walking, the participant can stand where the capture volume should begin, and both cameras can be adjusted such that the participant would just enter the field of view of both cameras at this location.

Fourth, the distance between the subject and the cameras affects the accuracy of pose estimation models. We recommend selecting a distance such that the subject’s joints are clearly discernible in the recorded videos. Performing movements 2–10m from the cameras works well; however, higher resolution pose estimation settings may be needed for accurate kinematics at 10m (i.e., HRNet or OpenPose high-accuracy settings). Furthermore, extrinsic calibration errors increase as distance increases from the calibration board’s position. Performing activities near where the board was placed during calibration or using a larger calibration board can mitigate these problems. We also recommend not positioning the cameras less than 2m from the subject, since some pose estimation models commonly fail when the subject takes up a significant portion of the field of view.

Finally, we recommend users frequently visit our website opencap.ai, where we provide best practices and updates about new features. In the year following its release, we have added multiple features to OpenCap including support for different musculoskeletal models, recording videos at different frame rates and in landscape mode, and interactive data plotting.

## Validation

Participants and experiment. To validate OpenCap against gold standard kinematic and kinetic measures, we measured ten healthy adults (sex: 6 female and 4 male; age = 27.7±3.8 [23-25] years; body mass = 69.2±11.6 [59.0–92.9] kg; height = 1.74±0.12 [1.60–1.96] m; mean ± standard deviation [range]) performing multiple activities in a motion capture laboratory. Participants were instructed to perform four activities in a natural (i.e., self-selected) and modified way during data collection: i) walking naturally and with a trunk sway modification (trunk leaned laterally over stance leg), ii) performing five squats naturally and then asymmetrically (reduced force under the left foot), iii) performing five sit-to-stands naturally and then with increased trunk flexion (forward lean when rising), and iv) performing three drop jumps naturally and then asymmetrically (reduced force under the left foot when landing).

Experimental data. We measured ground truth kinematics, ground reaction forces, and muscle activity with optical motion capture, force plates, and electromyography. An eightcamera motion capture system (Motion Analysis Corp., Santa Rosa, CA, USA) tracked the positions (100 Hz) of 31 retroreflective markers placed bilaterally on the 2nd and 5th metatarsal heads, calcanei, medial and lateral malleoli, medial and lateral femoral epicondyles, anterior and posterior superior iliac spines, sternoclavicular joints, acromia, medial and lateral epicondyles of the humerus, radial and ulnar styloid processes, and the C7 vertebrae. Twenty additional markers were used to aid in segment tracking. Ground reaction forces were synchronously measured (2000 Hz) using three in-ground force plates (Bertec Corp., Columbus, OH, USA). Wireless electromyography electrodes (Delsys Corp., Natick, MA, USA) measured muscle activity (2000 Hz) from the vastus lateralis and medialis (electromyography data from 14 other lower-extremity muscles are shared with the dataset but not analyzed here). We used OpenCap to record video from five smartphones (iPhone 12 Pro, Apple Inc., Cupertino, CA, USA). The phones were positioned 1.5 m off the ground, 3 m from the center of the force plates, and at ±70˚, ±45˚, and 0˚, where 0˚ faces the participant. Unless otherwise noted, the validation results used only the two ±45˚ cameras. A precision-manufactured, 720x540 mm checkerboard was used for computing the extrinsic parameters during OpenCap’s camera calibration step.

Marker, force, and electromyography data were filtered using a fourth-order, zero-lag Butterworth filter. Marker and force data were low-pass filtered (walking: 6 Hz, squat: 4 Hz, sit-tostand: 4 Hz, and drop jump: 30 Hz). These frequencies were selected as the frequency that retained 99.7% of the cumulative signal power of the Fourier-transformed marker trajectories [95]. Electromyography data were band-pass filtered (30–500 Hz), rectified, and low-pass filtered (6 Hz). Electromyography data were normalized to maximum activation trials including maximum height jumps, sprinting, and isometric and isokinetic ankle dorsiflexion, knee flexion, hip abduction exercises [96].

Kinematics and kinetics. Laboratory-based (later referred to as Mocap) kinematic and kinetic data were estimated from measured marker and force plate data using OpenSim 4.3. We used the same modeling and simulations pipeline as OpenCap to scale the musculoskeletal models and estimate joint kinematics from measured marker data (see Methods: Design: Physics-based modeling and simulation). Joint kinetics were then estimated from joint kinematics (filtered at same frequencies as force plate data) and force plate data using OpenSim’s Inverse Dynamics tool.

OpenCap kinematic and kinetic data were estimated using the two 45˚ cameras and the HRNet pose detection algorithm. This setup combines simplicity, performance, and a permissible open-source software license. It was selected after conducting a sensitivity analysis studying the effect of using different camera configurations (two, three, and five cameras) and pose detection algorithms (OpenPose with default settings, OpenPose with high-accuracy settings, and HRNet) on predicted anatomical marker positions and joint kinematics. See S1 Appendix and Methods: Validation: Validation Results for details about the sensitivity analysis and pose detection algorithm settings.

Error analysis. We evaluated the performance of OpenCap against Mocap by quantifying errors in anatomical marker positions, joint kinematics, ground reaction forces, and joint kinetics.

We quantified errors in 3D anatomical marker positions using mean per marker error (Euclidean distance). We report errors for 17 anatomical markers (the C7 vertebrae and the left and right acromia, anterior and posterior superior iliac spines, medial and lateral femoral epicondyles, medial and lateral malleoli, calcanei, and second and fifth metatarsal heads). Prior to error analysis, we synchronized and aligned Mocap and OpenCap position data by removing the time delay that minimized the mean difference between marker positions (averaged over all markers and time steps), then subtracting this average position offset from the OpenCap positions.

We quantified errors in 3D joint kinematics using MAE. We report errors for 18 rotational degrees of freedom (pelvis rotations [3], hips [2x3], knees [2x1], ankles [2x2], and lumbar [3]) and three translational degrees of freedom (pelvis translations).

We quantified errors in 3D ground reaction forces using MAE normalized by bodyweight. We also expressed errors as percent of range of the measured signal over each trial. Prior to quantifying errors, we filtered ground reaction forces from OpenCap using the same filters as for the measured ground reaction forces (see Methods: Validation: Experimental data).

We quantified errors in 3D joint kinetics using MAE normalized by bodyweight times body height. We report errors for 15 rotational degrees of freedom (hips [2x3], knees [2x1], ankle [2x2], and lumbar [3]). It is important to note that while joint moments estimated from inverse dynamics are considered gold standard, they include non-physical pelvis residual forces and moments to compensate for the inconsistency between model-based kinematics and measured ground reaction forces. In contrast, muscle-driven simulations are dynamically consistent and do not include pelvis residuals. Thus, the differences between Inverse Dynamics and OpenCap-estimated joint moments are not entirely attributable to error in the OpenCap pipeline.

Validation results. The marker error, averaged across markers and activities, was 32 mm using the two-camera HRNet setup. Our sensitivity analysis demonstrated that OpenCap’s accuracy remained consistent across different pose detectors and additional cameras. Marker error was 31 and 35 mm when using OpenPose with high-accuracy settings and default setting, respectively. Using three cameras did not improve accuracy, but using five cameras mildly reduced error (29 mm for HRNet). Marker error was larger for the upper extremity (39 mm) and pelvis (38 mm) than for the lower extremity (27 mm) using the two-camera HRNet setup. Detailed results of the sensitivity analyses are presented in S1 Table.

The kinematic MAE for the two-camera HRNet setup, averaged across degrees of freedom and activities, was 4.5˚ (range = 1.7–10.3˚) and 12.3 mm (range = 5–20.3 mm) for the 18 rotational and three translational degrees of freedom, respectively (Table 1). Our sensitivity analysis showed that kinematic errors were similar when using the high-accuracy settings and default OpenPose settings (4.3˚ and 4.7˚, respectively), and when adding cameras (improvement of less than 0.3˚). We also investigated the effect of using video keypoints instead of anatomical markers to estimate joint kinematics. Kinematic errors were 3.4˚ worse on average for the two-camera HRNet setup when using the video keypoints instead of the anatomical markers. This was primarily due to 12.1–39.2˚ errors at the lumbar extension, pelvic tilt, and hip flexion degrees of freedom, due to the limited information in the video keypoint marker set for distinguishing between rotations at these joints. Detailed results of the sensitivity analyses are presented in S2 Table. Average kinematic waveforms estimated using OpenCap and Mocap are presented in Figs A–D in S1 Appendix.

The ground reaction force MAE, averaged across directions and activities was 6.2% bodyweight. It was 11.4% bodyweight in the vertical direction, 3.5% bodyweight in the anteriorposterior direction, and 3.8% bodyweight in the medio-lateral direction (Table 1). The joint moment MAE, averaged across degrees of freedom and activities, was 1.2% bodyweight\*height (Table 1). Detailed results are presented in S3 and S4 Tables, and average ground reaction force and joint moment waveforms estimated using OpenCap and Mocap are presented in Figs E–L ofS1 Appendix.

## Applications and statistics

We assessed OpenCap’s ability to estimate kinetic measures related to musculoskeletal pathology in three applications that represent clinical use cases. Unless otherwise noted, these analyses were performed on the 10-subject dataset described in Methods: Validation. All statistical analyses were performed in Python (v3.7.10) using the scipy [97] (v1.5.4), statsmodels [98] (v0.13.2), and pingouin [99] (v0.5.2) packages. We compared conditions within and between measurement modalities using r<sup>2</sup>, MAE, two-sided paired t tests (alpha = .05), and two-sided Wilcoxon signed rank tests. Prior to conducting a t test, we tested for normality using a Shapiro Wilkes test, and we used a Wilcoxon signed rank test to compare non-normally distributed data. To prevent inflated Type 1 error from multiple comparisons, we report corrected Pvalues after controlling for the false discovery rate using the Benjamini Hochberg procedure [100]. We evaluated the post-hoc power of t tests and Wilcoxon signed rank tests using the sample size, alpha = .05, and the observed effect size. We evaluated performance on classification tasks using AUC and binary classification accuracy at the threshold that maximized the true positive rate minus the false positive rate. Unless otherwise noted, values are reported as mean ± standard deviation.

In the first application, we assessed the peak knee adduction moment and peak medial knee contact force during walking. Participants walked naturally and with a trunk sway modification, which typically alters medial knee loading [32]. Participants walked with 15˚ more trunk sway on average during the trunk sway compared to the natural condition, which is similar to the 10–13˚ of trunk sway reported in gait modification studies [32,101]. We computed peaks of both loading measures during the first half of the stance phase using the Joint Reaction Analysis tool in OpenSim (see S1 Appendix for details), which uses kinematics, ground reaction forces, and muscle forces as inputs. For OpenCap, we used the outputs of the muscledriven dynamic simulation for this analysis, and for Mocap, we used the OpenSim Static Optimization tool to estimate muscle forces. We used static optimization for the Mocap data as it is commonly used to estimate knee contact forces during low-frequency tasks like walking, and it is sensitive to changes in contact force induced by gait modifications [102]. We first determined how accurately OpenCap could estimate the peak knee adduction moment and how it varies among gait patterns and individuals. For each walking condition, we averaged the peak knee adduction moment across the three trials for each individual and compared between OpenCap and Mocap using r<sup>2</sup> and MAE. We then determined whether OpenCap could detect group changes in both loading measures from a gait modification similarly to Mocap. For each measurement modality, we used either a two-sided paired t test or a Wilcoxon signed rank test to evaluate the changes from baseline, and we computed the post-hoc power of each test. Finally, we evaluated whether OpenCap correctly identified an increase or decrease in peak knee loading measures for each individual, using the Mocap estimate as ground truth.

In the second application, we evaluated lower-extremity joint moments while rising from a 40 cm chair. Participants stood naturally and with increased trunk flexion, which can shift the muscle force demand from the knee extensors to the hip extensors and ankle plantarflexors [40]. During the increased trunk flexion condition, participants stood with $1 4 2 { \pm } 8 ^ { \circ }$ oftrunk flexion, which is similar to the $4 7 ^ { \circ }$ reported in a cohort of older adults with functional limitations [103]. For three repetitions per condition, we averaged the hip extension, knee extension, and ankle plantarflexion moments over the rising phase, then averaged these values across repetitions. To evaluate OpenCap’s ability to detect group changes between conditions, we compared the moment changes from baseline to trunk-lean using two-sided paired t tests for both OpenCap and Mocap. We then conducted a post-hoc power analysis for each measurement modality. To determine OpenCap’s ability to identify individuals with low knee extensor moments during this motion, we compared each participant’s average knee extension moment for each condition between OpenCap and Mocap using $\hat { r } ^ { 2 }$ and MAE.

In the third application, we assessed the between-limb symmetry of knee extensor muscle activation while squatting. Participants squatted naturally and asymmetrically, which can elicit asymmetrical knee extensor force generation [7]. We first performed an in-lab experiment to compare peak vasti muscle (knee extensors) activation measured with electromyography to peak activation estimated with OpenCap and Mocap. Since there is no change in muscle strength between these conditions, a change in muscle activation between conditions is a more easily measured surrogate for a change in muscle force. For OpenCap, muscle activations were outputs of the muscle-driven tracking simulations, whereas for Mocap, muscle activations were estimated using OpenSim’s Static Optimization tool. We first averaged the activation of the vastus medialis and vastus lateralis, then extracted the peak value over a squat (standing to standing again). We calculated the peak vasti activation symmetry index between the left and right leg (Eq 2) and averaged across three repetitions in each condition:

$$
s y m m e t r y i n d e x = 1 - \frac { a _ { \mathrm { i n v o l v e d } } - a _ { \mathrm { u n i n v o l v e d } } } { a _ { \mathrm { u n i n v o l v e d } } } ,\tag{ð2Þ}
$$

where $a _ { \mathrm { i n v o l v e d } }$ is the peak activation of the left vasti (reduced force under left foot during asymmetric condition) and $a _ { \mathrm { u n i n v o l v e d } }$ is the peak activation of the right vasti. The symmetry index is larger than one when the left peak vasti activation is lower than the right peak vasti activation, which would be expected in the asymmetric condition. On average, during the asymmetric condition compared to the natural condition, our participants squatted with a 0.53±0.32 greater symmetry index measured by electromyography; this is similar to the 0.51 greater asymmetry in vasti strength reported in individuals one month after a total knee replacement [104]. We determined OpenCap’s ability to classify symmetric vs. asymmetric squats using AUC and classification accuracy, with ground truth symmetry labels determined from electromyography based on a symmetry index threshold (1.15) that predicts functional deficits following anterior cruciate ligament surgery [43]. We also computed the AUC and accuracy for simulated muscle activations from Mocap.

Finally, we performed a field study where a clinician used OpenCap to evaluate knee extension moment symmetry in 100 individuals outside of the laboratory (sex: 41 female and 59 male; age = 29.6±9.2 [18-67] years, body mass = 69.2±12.0 [50–109] kg; height = 1.74±0.09 [1.45–1.97] m; mean ± standard deviation [range]). We used a 210x175 mm checkerboard printed on A4 paper and mounted to plexiglass for camera calibration. Participants performed natural squats and asymmetric squats. All participants provided written informed consent before participation. The study protocol was approved and overseen by the Institutional Review Board ofStanford University (IRB00000351). We conducted the experiment in accordance with this approved protocol and relevant guidelines and regulations. First, we evaluated OpenCap’s ability to detect a squat with a between-limb asymmetry in the peak knee extension moment. For each participant, we computed the peak knee extension moment for three repetitions per condition, computed the peak knee extension moment symmetry index (Eq 2), and averaged across the repetitions in each condition. To determine the classification performance, we computed AUC and accuracy, with ground truth labels being the instructed condition (i.e., natural [assumed to be symmetric] vs. asymmetric squats). Second, we evaluated OpenCap’s ability to detect between-condition changes in knee extension moment symmetry, simulating the ability to detect improvements in symmetry that would be expected to occur over time. To simulate improved symmetry, we subtracted each participant’s symmetry index averaged over the repetitions of the natural condition from their symmetry index averaged over repetitions of the asymmetric condition (a positive value indicates an improvement in symmetry). To simulate unchanged symmetry, we averaged the difference in symmetry index between each combination of the asymmetric squat repetitions. We computed the AUC and accuracy of this change in symmetry measure using the known class (i.e., improved symmetry or unchanged symmetry) as ground truth.

## Supporting information

S1 Appendix. Supplementary methods and results. (PDF)

S1 Table. Errors in each marker position between OpenCap and motion capture. The mean per-marker error is shown for each marker, activity, camera combination, and pose detection algorithm.

(XLSX)

S2 Table. Errors in kinematics between OpenCap and motion capture. The mean absolute error (MAE) and root mean square error (RMSE) are shown for each degree of freedom, activity, camera combination, and pose detection algorithm.

(XLSX)

S3 Table. Errors in ground reaction forces between OpenCap and force plates. The mean absolute error (MAE), root mean square error (RMSE), and mean absolute error as a percent age of the range (MAPE) are shown for each activity using the two-camera HRNet setup. (XLSX)

S4 Table. Errors in joint moments between OpenCap and inverse dynamics using motion capture and force plates. The mean absolute error (MAE), root mean square error (RMSE), and mean absolute error as a percentage of the range (MAPE) are shown for each activity and degree of freedom using the two-camera HRNet setup.

(XLSX)

S1 Movie. Overview of the motivation, technology, and impact of OpenCap. We introduce the value and limitations of the current lab-based approach for estimating human movement dynamics. We then show an OpenCap data collection and demonstrate how it overcomes the cost, time, and expertise limitations of the lab-based approach. Finally, we demonstrate Open-Cap’s accuracy and discuss how it can catalyze large-scale research studies and the clinical translation of movement biomarkers.

(MP4)

## Author Contributions

Conceptualization: Scott D. Uhlrich, Antoine Falisse, Łukasz Kidziński, Jennifer L. Hicks, Scott L. Delp.

Data curation: Scott D. Uhlrich, Antoine Falisse, Łukasz Kidziński, Julie Muccini.

Formal analysis: Scott D. Uhlrich, Antoine Falisse, Łukasz Kidziński.

Funding acquisition: Akshay S. Chaudhari, Jennifer L. Hicks, Scott L. Delp.

Investigation: Scott D. Uhlrich, Antoine Falisse, Łukasz Kidziński, Julie Muccini.

Methodology: Scott D. Uhlrich, Antoine Falisse, Łukasz Kidziński, Jennifer L. Hicks, Scott L. Delp.

Project administration: Jennifer L. Hicks, Scott L. Delp.

Resources: Jennifer L. Hicks, Scott L. Delp.

Software: Scott D. Uhlrich, Antoine Falisse, Łukasz Kidziński, Michael Ko.

Supervision: Jennifer L. Hicks, Scott L. Delp.

Validation: Scott D. Uhlrich, Antoine Falisse.

Visualization: Scott D. Uhlrich, Antoine Falisse, Łukasz Kidziński.

Writing – original draft: Scott D. Uhlrich, Antoine Falisse.

Writing – review & editing: Scott D. Uhlrich, Antoine Falisse, Łukasz Kidziński, Julie Muccini, Michael Ko, Akshay S. Chaudhari, Jennifer L. Hicks, Scott L. Delp.

## References

1. Miyazaki T, Wada M, Kawahara H, Sato M, Baba H, Shimada S. Dynamic load at baseline can predict radiographic disease progression in medial compartment knee osteoarthritis. Ann Rheum Dis. 2002; 61: 617–622. https://doi.org/10.1136/ard.61.7.617 PMID: 12079903

2. Smith SHL, Reilly P, Bull AMJ. A musculoskeletal modelling approach to explain sit-to-stand difficulties in older people due to changes in muscle recruitment and movement strategies. J Biomech. 2020; 98: 109451. https://doi.org/10.1016/j.jbiomech.2019.109451 PMID: 31685221

3. Moreland JD, Richardson JA, Goldsmith CH, Clase CM. Muscle weakness and falls in older adults: a systematic review and meta-analysis. J Am Geriatr Soc. 2004; 52: 1121–1129. https://doi.org/10. 1111/j.1532-5415.2004.52310.x PMID: 15209650

4. Van der heijden MMP, Meijer K, Willems PJB, Savelberg HHCM. Muscles limiting the sit-to-stand movement: an experimental simulation of muscle weakness. Gait Posture. 2009; 30: 110–114. https:// doi.org/10.1016/j.gaitpost.2009.04.002 PMID: 19419871

5. Christiansen CL, Bade MJ, Judd DL, Stevens-Lapsley JE. Weight-bearing asymmetry during sit-stand transitions related to impairment and functional mobility after total knee arthroplasty. Arch Phys Med Rehabil. 2011; 92: 1624–1629. https://doi.org/10.1016/j.apmr.2011.05.010 PMID: 21839986

6. Briggs RA, Houck JR, Drummond MJ, Fritz JM, Lastayo PC, Marcus RL. Asymmetries Identified in Sitto-Stand Task Explain Physical Function after Hip Fracture. J Geriatr Phys Ther. 2018; 41: 210–217. https://doi.org/10.1519/JPT.0000000000000122 PMID: 28252470

7. Dai B, Butler RJ, Garrett WE, Queen RM. Using ground reaction force to predict knee kinetic asymmetry following anterior cruciate ligament reconstruction. Scand J Med Sci Sport. 2014; 24: 974–981. https://doi.org/10.1111/sms.12118 PMID: 24118495

8. Seth A, Hicks JL, Uchida TK, Habib A, Dembia CL, Dunne JJ, et al. OpenSim: Simulating musculoskeletal dynamics and neuromuscular control to study human and animal movement. PLoS Comput Biol. 2018; 14: e1006223. https://doi.org/10.1371/journal.pcbi.1006223 PMID: 30048444

9. Pizzolato C, Lloyd DG, Sartori M, Ceseracciu E, Besier TF, Fregly BJ, et al. CEINMS: A toolbox to investigate the influence of different neural control solutions on the prediction of muscle excitation and joint moments during dynamic motor tasks. J Biomech. 2015; 48: 3929–3936. https://doi.org/10.1016/ j.jbiomech.2015.09.021 PMID: 26522621

10. Delp SL, Anderson FC, Arnold AS, Loan P, Habib A, John CT, et al. OpenSim: Open-source software to create and analyze dynamic simulations of movement. IEEE Trans Biomed Eng. 2007; 54: 1940– 1950. https://doi.org/10.1109/TBME.2007.901024 PMID: 18018689

11. Hewett T, Myer G, Ford K, Heidt R, Colosimo A, McLean S, et al. Biomechanical measures of neuromuscular control and valgus loading of the knee predict anterior cruciate ligament injury risk in female athletes a prospective study. Am J Sports Med. 2005; 33: 492–501. https://doi.org/10.1177/ 0363546504269591 PMID: 15722287

12. Rajagopal A, Kidziński Ł, McGlaughlin AS, Hicks JL, Delp SL, Schwartz MH. Estimating the effect size of surgery to improve walking in children with cerebral palsy from retrospective observational clinical data. Sci Rep. 2018; 8: 16344. https://doi.org/10.1038/s41598-018-33962-2 PMID: 30397268

13. Rajagopal A, Kidziński Ł, McGlaughlin AS, Hicks JL, Delp SL, Schwartz MH. Pre-operative gastrocnemius lengths in gait predict outcomes following gastrocnemius lengthening surgery in children with cerebral palsy. PLoS One. 2020; 15: e0233706. https://doi.org/10.1371/journal.pone.0233706 PMID: 32502157

14. Knudson D V. Authorship and sampling practice in selected biomechanics and sports science journals. Percept Mot Skills. 2011; 112: 838–844. https://doi.org/10.2466/17.PMS.112.3.838-844 PMID: 21853773

15. Oliveira AS, Pirscoveanu CI. Implications of sample size and acquired number of steps to investigate running biomechanics. Sci Rep. 2021; 11: 3083. https://doi.org/10.1038/s41598-021-82876-z PMID: 33542463

16. Weygers I, Kok M, Konings M, Hallez H, De Vroey H, Claeys K. Inertial Sensor-Based Lower Limb Joint Kinematics: A Methodological Systematic Review. Sensors. 2020; 20: 673. https://doi.org/10. 3390/s20030673 PMID: 31991862

17. Dorschky E, Nitschke M, Seifer AK, van den Bogert AJ, Eskofier BM. Estimation of gait kinematics and kinetics from inertial sensor data using optimal control of musculoskeletal models. J Biomech. 2019; 95: 111147. https://doi.org/10.1016/j.jbiomech.2019.07.022 PMID: 31472970

18. Karatsidis A, Jung M, Schepers HM, Bellusci G, de Zee M, Veltink PH, et al. Musculoskeletal modelbased inverse dynamic analysis under ambulatory conditions using inertial motion capture. Med Eng Phys. 2019; 65: 68–77. https://doi.org/10.1016/j.medengphy.2018.12.021 PMID: 30737118

19. Desmarais Y, Mottet D, Slangen P, Montesinos P. A review of 3D human pose estimation algorithms for markerless motion capture. Comput Vis Image Underst. 2021; 212: 103275. https://doi.org/10. 1016/J.CVIU.2021.103275

20. Mathis A, Mamidanna P, Cury KM, Abe T, Murthy VN, Mathis MW, et al. DeepLabCut: markerless pose estimation of user-defined body parts with deep learning. Nat Neurosci. 2018; 21: 1281–1289. https://doi.org/10.1038/s41593-018-0209-y PMID: 30127430

21. Cao Z, Hidalgo G, Simon T, Wei S, Sheikh Y. OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields. IEEE Trans Pattern Anal Mach Intell. 2021; 43: 172–186. https://doi.org/10. 1109/TPAMI.2019.2929257 PMID: 31331883

22. Stenum J, Rossi C, Roemmich RT. Two-dimensional video-based analysis of human gait using pose estimation. PLoS Comput Biol. 2021; 17: e1008935. https://doi.org/10.1371/journal.pcbi.1008935 PMID: 33891585

23. Kidziński Ł, Yang B, Hicks JL, Rajagopal A, Delp SL, Schwartz MH. Deep neural networks enable quantitative movement analysis using single-camera videos. Nat Commun. 2020; 11: 4054. https:// doi.org/10.1038/s41467-020-17807-z PMID: 32792511

24. Boswell MA, Uhlrich SD, Kidziński Ł, Thomas K, Kolesar JA, Gold GE, et al. A neural network to predict the knee adduction moment in patients with osteoarthritis using anatomical landmarks obtainable from 2D video analysis. Osteoarthr Cartil. 2021; 29: 346–356. https://doi.org/10.1016/j.joca.2020.12. 017 PMID: 33422707

25. Joo H, Simon T, Li X, Liu H, Tan L, Gui L, et al. Panoptic Studio: A Massively Multiview System for Social Motion Capture. 2015 IEEE International Conference on Computer Vision (ICCV). IEEE; 2015. pp. 3334–3342. https://doi.org/10.1109/ICCV.2015.381

26. Iskakov K, Burkov E, Lempitsky V, Malkov Y. Learnable triangulation of human pose. 2019 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE; 2019. pp. 7717–7726. https://doi.org/10. 1109/ICCV.2019.00781

27. Tu H, Wang C, Zeng W. VoxelPose: Towards Multi-camera 3D Human Pose Estimation in Wild Environment. European Conference on Computer Vision (ECCV). Springer-Verlag; 2020. pp. 197–212. https://doi.org/10.1007/978-3-030-58452-8\_12

28. He Y, Yan R, Fragkiadaki K, Yu SI. Epipolar Transformers. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE; 2020. pp. 7776–7785. https://doi.org/10.1109/ CVPR42600.2020.00780

29. Nakano N, Sakura T, Ueda K, Omura L, Kimura A, Iino Y, et al. Evaluation of 3D Markerless Motion Capture Accuracy Using OpenPose With Multiple Video Cameras. Front Sport Act Living. 2020; 2: 50. https://doi.org/10.3389/fspor.2020.00050 PMID: 33345042

30. Reddy ND, Guigues L, Pishchulin L, Eledath J, Narasimhan SG. TesseTrack: End-to-end learnable multi-person articulated 3D pose tracking. 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE; 2021. pp. 15185–15195. https://doi.org/10.1109/CVPR46437.2021. 01494

31. Kanko RM, Laende EK, Davis EM, Selbie WS, Deluzio KJ. Concurrent assessment of gait kinematics using marker-based and markerless motion capture. J Biomech. 2021; 127: 110665. https://doi.org/ 10.1016/j.jbiomech.2021.110665 PMID: 34380101

32. Mu¨ndermann A, Asay JL, Mu¨ndermann L, Andriacchi TP. Implications of increased medio-lateral trunk sway for ambulatory mechanics. J Biomech. 2008; 41: 165–170. https://doi.org/10.1016/j.jbiomech. 2007.07.001 PMID: 17678933

33. Mu¨ndermann A, Dyrby CO, Hurwitz DE, Sharma L, Andriacchi TP. Potential strategies to reduce medial compartment loading in patients with knee osteoarthritis of varying severity: reduced walking speed. Arthritis Rheum. 2004; 50: 1172–1178. https://doi.org/10.1002/art.20132 PMID: 15077299

34. Mu¨ndermann A, Dyrby CO, Andriacchi TP. Secondary gait changes in patients with medial compartment knee osteoarthritis: Increased load at the ankle, knee, and hip during walking. Arthritis Rheum. 2005; 52: 2835–2844. https://doi.org/10.1002/art.21262 PMID: 16145666

35. Amin S, Luepongsak N, McGibbon CA, LaValley MP, Krebs DE, Felson DT. Knee adduction moment and development of chronic knee pain in elders. Arthritis Care Res. 2004; 51: 371–376. https://doi.org 10.1002/art.20396 PMID: 15188321

36. Walter JP D’Lima DD, Colwell CW, Fregly BJ. Decreased knee adduction moment does not guarantee decreased medial contact force during gait. J Orthop Res. 2010; 28: 1348–1354. https://doi.org/10. 1002/jor.21142 PMID: 20839320

37. Brandon SCE, Brown MJ, Clouthier AL, Campbell A, Richards JD, Deluzio KJ. Contributions of muscles and external forces to medial knee load reduction due to osteoarthritis braces. Knee. 2019; 26: 564–577. https://doi.org/10.1016/j.knee.2019.04.006 PMID: 31097362

38. Uhlrich SD, Silder A, Beaupre GS, Shull PB, Delp SL. Subject-specific toe-in or toe-out gait modifications reduce the larger knee adduction moment peak more than a non-personalized approach. J Biomech. 2018; 66: 103–110. https://doi.org/10.1016/j.jbiomech.2017.11.003 PMID: 29174534

39. Felson DT, Parkes M, Carter S, Liu A, Callaghan MJ, Hodgson R, et al. The Efficacy of a Lateral Wedge Insole for Painful Medial Knee Osteoarthritis After Prescreening: A Randomized Clinical Trial. Arthritis Rheumatol. 2019; 71: 908–915. https://doi.org/10.1002/art.40808 PMID: 30615299

40. van der Kruk E, Silverman AK, Reilly P, Bull AMJ. Compensation due to age-related decline in sit-tostand and sit-to-walk. J Biomech. 2021; 122: 110411. https://doi.org/10.1016/j.jbiomech.2021.110411 PMID: 33915476

41. Larsson L, Karlsson J. Isometric and dynamic endurance as a function of age and skeletal muscle characteristics. Acta Physiol Scand. 1978; 104: 129–136. https://doi.org/10.1111/j.1748-1716.1978. tb06259.x PMID: 152565

42. Myer GD, Sugimoto D, Thomas S, Hewett TE. The influence of age on the effectiveness of neuromuscular training to reduce anterior cruciate ligament injury in female athletes: A meta-analysis. Am J Sports Med. 2013; 41: 203–215. https://doi.org/10.1177/0363546512460637 PMID: 23048042

43. Schmitt LC, Paterno MV, Hewett TE. The impact of quadriceps femoris strength asymmetry on functional performance at return to sport following anterior cruciate ligament reconstruction. J Orthop Sports Phys Ther. 2012; 42: 750–759. https://doi.org/10.2519/jospt.2012.4194 PMID: 22813542

44. Anz AW, Bushnell BD, Griffin LP, Noonan TJ, Torry MR, Hawkins RJ. Correlation of torque and elbow injury in professional baseball pitchers. Am J Sports Med. 2010; 38: 1368–1374. https://doi.org/10. 1177/0363546510363402 PMID: 20400752

45. Wellsandt E, Failla MJ, Snyder-Mackler L. Limb symmetry indexes can overestimate knee function after anterior cruciate ligament injury. J Orthop Sports Phys Ther. 2017; 47: 334–338. https://doi.org/ 10.2519/jospt.2017.7285 PMID: 28355978

46. Song J, Semanik P, Sharma L, Chang RW, Hochberg MC, Mysiw WJ, et al. Assessing Physical Activity in Persons With Knee Osteoarthritis Using Accelerometers: Data From the Osteoarthritis Initiative. Arthritis Care Res. 2010; 62: 1724–1732. https://doi.org/10.1002/acr.20305 PMID: 20806273

47. Sudlow C, Gallacher J, Allen N, Beral V, Burton P, Danesh J, et al. UK Biobank: An Open Access Resource for Identifying the Causes of a Wide Range of Complex Diseases of Middle and Old Age. PLOS Med. 2015; 12: e1001779. https://doi.org/10.1371/journal.pmed.1001779 PMID: 25826379

48. Gorton GE, Hebert DA, Gannotti ME. Assessment of the kinematic variability among 12 motion analysis laboratories. Gait Posture. 2009; 29: 398–402. https://doi.org/10.1016/j.gaitpost.2008.10.060 PMID: 19056271

49. Kanko RM, Laende E, Selbie WS, Deluzio KJ. Inter-session repeatability of markerless motion capture gait kinematics. J Biomech. 2021; 121: 110422. https://doi.org/10.1016/j.jbiomech.2021.110422 PMID: 33873117

50. Lin T-Y, Maire M, Belongie S, Hays J, Perona P, Ramanan D, et al. Microsoft COCO: Common Objects in Context. Computer Vision–ECCV 2014. Springer; 2014. pp. 740–755. https://doi.org/10. 48550/arXiv.1506.01497

51. Bolink SAAN, Naisas H, Senden R, Essers H, Heyligers IC, Meijer K, et al. Validity of an inertial measurement unit to assess pelvic orientation angles during gait, sit-stand transfers and step-up transfers: Comparison with an optoelectronic motion capture system. Med Eng Phys. 2016; 38: 225–231. https:// doi.org/10.1016/j.medengphy.2015.11.009 PMID: 26711470

52. Nu¨esch C, Roos E, Pagenstert G, Mu¨ndermann A. Measuring joint kinematics of treadmill walking and running: Comparison between an inertial sensor based system and a camera-based system. J Biomech. 2017; 57: 32–38. https://doi.org/10.1016/j.jbiomech.2017.03.015 PMID: 28366438

53. Schepers M, Giuberti M, Bellusci G. Xsens MVN: Consistent Tracking of Human Motion Using Inertial Sensing. Xsens Technol Tech Rep. 2018. https://doi.org/10.13140/RG.2.2.22099.07205

54. Konrath J, Karatsidis A, Schepers H, Bellusci G, de Zee M, Andersen M. Estimation of the Knee Adduction Moment and Joint Contact Force during Daily Living Activities Using Inertial Motion Capture. Sensors. 2019; 19: 1681. https://doi.org/10.3390/s19071681 PMID: 30970538

55. Mavor MP, Ross GB, Clouthier AL, Karakolis T, Graham RB. Validation of an IMU Suit for Military-Based Tasks. Sensors. 2020; 20: 4280. https://doi.org/10.3390/s20154280 PMID: 32751920

56. Al Borno MO’Day J, Ibarra V, Dunne J, Seth A, Habib A, et al. OpenSense: An open-source toolbox for inertial-measurement-unit-based measurement of lower extremity kinematics over long durations. J Neuroeng Rehabil. 2022; 19: 22. https://doi.org/10.1186/s12984-022-01001-x PMID: 35184727

57. Slade P, Habib A, Hicks JL, Delp SL. An open-source and wearable system for measuring 3D human motion in real-time. IEEE Trans Biomed Eng. 2022; 69: 678–688. https://doi.org/10.1109/TBME.2021. 3103201 PMID: 34383640

58. Pagnon D, Domalain M, Reveret L. Pose2Sim: An End-to-End Workflow for 3D Markerless Sports Kinematics—Part 2: Accuracy. Sensors. 2022; 22: 2712. https://doi.org/10.3390/s22072712 PMID: 35408326

59. Cappozzo A, Catani F, Leardini A, Benedetti MG, Della Croce U. Position and orientation in space of bones during movement: Experimental artefacts. Clin Biomech. 1996; 11: 90–100. https://doi.org/10. 1016/0268-0033(95)00046-1 PMID: 11415604

60. Karatsidis A, Bellusci G, Schepers HM, de Zee M, Andersen MS, Veltink PH. Estimation of ground reaction forces and moments during gait using only inertial motion capture. Sensors. 2017; 17: 75. https://doi.org/10.3390/s17010075 PMID: 28042857

61. Hu W, Zhang C, Zhan F, Zhang L, Wong TT. Conditional Directed Graph Convolution for 3D Human Pose Estimation. Proceedings of the 29th ACM International Conference on Multimedia. Association for Computing Machinery, Inc; 2021. pp. 602–611. https://doi.org/10.1145/3474085.3475219

62. Shan W, Lu H, Wang S, Zhang X, Gao W. Improving Robustness and Accuracy via Relative Information Encoding in 3D Human Pose Estimation. Proceedings of the 29th ACM International Conference on Multimedia. Association for Computing Machinery; 2021. pp. 3446–3454.

63. Bradski G. The OpenCV Library. Dr Dobb’s J Softw Tools. 2000.

64. Ren S, He K, Girshick R, Sun J. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. Advances in Neural Information Processing Systems. Curran Associates, Inc.; 2015.

65. Sun K, Xiao B, Liu D, Wang J. Deep High-Resolution Representation Learning for Human Pose Estimation. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE; 2019. pp. 5693–5703. https://doi.org/10.48550/arXiv.1902.09212

66. Jin S, Xu L, Xu J, Wang C, Liu W, Qian C, et al. Whole-Body Human Pose Estimation in the Wild. Computer Vision–ECCV 2020. Springer-Verlag; 2020. pp. 196–214. https://doi.org/10.48550/arXiv.2007. 11858

67. Zhang F, Zhu X, Dai H, Ye M, Zhu C. Distribution-Aware Coordinate Representation for Human Pose Estimation. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2020. pp. 7091–7100. https://doi.org/10.1109/CVPR42600.2020.00712

68. Hartley RI, Sturm P. Triangulation. Comput Vis Image Underst. 1997; 68: 146–157. https://doi.org/10. 1006/CVIU.1997.0547

69. Rajagopal A, Dembia CL, DeMers MS, Delp DD, Hicks JL, Delp SL. Full-Body Musculoskeletal Model for Muscle-Driven Simulation of Human Gait. IEEE Trans Biomed Eng. 2016; 63: 2068–2079. https:// doi.org/10.1109/TBME.2016.2586891 PMID: 27392337

70. Arnold EM, Hamner SR, Seth A, Millard M, Delp SL. How muscle fiber lengths and velocities affect muscle force generation as humans walk and run at different speeds. J Exp Biol. 2013; 216: 2150– 2160. https://doi.org/10.1242/jeb.075697 PMID: 23470656

71. Dembia CL, Silder A, Uchida TK, Hicks JL, Delp SL. Simulating ideal assistive devices to reduce the metabolic cost of walking with heavy loads. PLoS One. 2017; 12: e0180320. https://doi.org/10.1371/ journal.pone.0180320 PMID: 28700630

72. Afschrift M, van Deursen R, De Groote F, Jonkers I. Increased use of stepping strategy in response to medio-lateral perturbations in the elderly relates to altered reactive tibialis anterior activity. Gait Posture. 2019; 68: 575–582. https://doi.org/10.1016/j.gaitpost.2019.01.010 PMID: 30654320

73. Swinnen W, Mylle I, Hoogkamer W, DE Groote F, Vanwanseele B. Changing Stride Frequency Alters Average Joint Power and Power Distributions during Ground Contact and Leg Swing in Running. Med Sci Sports Exerc. 2021; 53: 2111–2118. https://doi.org/10.1249/MSS.0000000000002692 PMID: 33935233

74. Swinnen W, Hoogkamer W, De Groote F, Vanwanseele B. Habitual foot strike pattern does not affect simulated triceps surae muscle metabolic energy consumption during running. J Exp Biol. 2019; 222: jeb212449. https://doi.org/10.1242/jeb.212449 PMID: 31704899

75. Thompson-Kolesar JA, Gatewood CT, Tran AA, Silder A, Shultz R, Delp SL, et al. Age Influences Biomechanical Changes After Participation in an Anterior Cruciate Ligament Injury Prevention Program. Am J Sports Med. 2018; 46: 598–606. https://doi.org/10.1177/0363546517744313 PMID: 29281799

76. Thompson JA, Tran AA, Gatewood CT, Shultz R, Silder A, Delp SL, et al. Biomechanical Effects of an Injury Prevention Program in Preadolescent Female Soccer Athletes. Am J Sports Med. 2017; 45: 294–301. https://doi.org/10.1177/0363546516669326 PMID: 27793803

77. Falisse A, Van Rossom S, Jonkers I, De Groote F. EMG-Driven Optimal Estimation of Subject-SPE-CIFIC Hill Model Muscle-Tendon Parameters of the Knee Joint Actuators. IEEE Trans Biomed Eng. 2017; 64: 2253–2262. https://doi.org/10.1109/TBME.2016.2630009 PMID: 27875132

78. Uhlrich SD, Kolesar JA, Kidziński Ł, Boswell MA, Silder A, Gold GE, et al. Personalization improves the biomechanical efficacy of foot progression angle modifications in individuals with medial knee osteoarthritis. J Biomech. 2022;144. https://doi.org/10.1016/j.jbiomech.2022.111312 PMID: 36191434

79. Uhlrich SD, Jackson RW, Seth A, Kolesar JA, Delp SL. Muscle coordination retraining inspired by musculoskeletal simulations reduces knee contact force. Sci Rep. 2022; 12: 9842. https://doi.org/10. 1038/s41598-022-13386-9 PMID: 35798755

80. Lai AKM, Arnold AS, Wakeling JM. Why are Antagonist Muscles Co-activated in My Simulation? A Musculoskeletal Model for Analysing Human Locomotor Tasks. Ann Biomed Eng. 2017; 45: 2762– 2774. https://doi.org/10.1007/s10439-017-1920-7 PMID: 28900782

81. Falisse A, Afschrift M, De Groote F. Modeling toes contributes to realistic stance knee mechanics in three-dimensional predictive simulations of walking. PLoS One. 2022; 17: e0256311. https://doi.org/ 10.1371/journal.pone.0256311 PMID: 35077455

82. Lin YC, Pandy MG. Three-dimensional data-tracking dynamic optimization simulations of human locomotion generated by direct collocation. J Biomech. 2017; 59: 1–8. https://doi.org/10.1016/j.jbiomech. 2017.04.038 PMID: 28583674

83. Falisse A, Serrancolı´ G, Dembia CL, Gillis J, Jonkers I, De Groote F. Rapid predictive simulations with complex musculoskeletal models suggest that diverse healthy and pathological human gaits can emerge from similar control strategies. J R Soc Interface. 2019; 16: 20190402. https://doi.org/10. 1098/rsif.2019.0402 PMID: 31431186

84. Raasch CC, Zajac FE, Ma B, Levine WS. Muscle coordination of maximum-speed pedaling. J Biomech. 1997; 30: 595–602. https://doi.org/10.1016/s0021-9290(96)00188-1 PMID: 9165393

85. De Groote F, Pipeleers G, Jonkers I, Demeulenaere B, Patten C, Swevers J, et al. A physiology based inverse dynamic analysis of human gait: potential and perspectives. Comput Methods Biomech Biomed Engin. 2009; 12: 563–574. https://doi.org/10.1080/10255840902788587 PMID: 19319704

86. Zajac FE. Muscle and tendon: properties, models, scaling, and application to biomechanics and motor control. Crit Rev Biomed Eng. 1989; 17: 359–411. PMID: 2676342

87. De Groote F, Kinney AL, Rao A V., Fregly BJ. Evaluation of Direct Collocation Optimal Control Problem Formulations for Solving the Muscle Redundancy Problem. Ann Biomed Eng. 2016; 44: 2922– 2936. https://doi.org/10.1007/S10439-016-1591-9 PMID: 27001399

88. Sherman MA, Seth A, Delp SL. Simbody: multibody dynamics for biomedical research. Procedia IUTAM. 2011; 2: 241–261. https://doi.org/10.1016/j.piutam.2011.04.023 PMID: 25866705

89. Serrancoli G, Falisse A, Dembia C, Vantilt J, Tanghe K, Lefeber D, et al. Subject-Exoskeleton Contact Model Calibration Leads to Accurate Interaction Force Predictions. IEEE Trans Neural Syst Rehabil Eng. 2019; 27: 1597–1605. https://doi.org/10.1109/TNSRE.2019.2924536 PMID: 31247556

90. Van Den Bogert AJ, Geijtenbeek T, Even-Zohar O, Steenbrink F, Hardin EC. A real-time system for biomechanical analysis of human movement and muscle function. Med Biol Eng Comput. 2013; 51: 1069–1077. https://doi.org/10.1007/s11517-013-1076-z PMID: 23884905

91. Betts JT. Practical Methods for Optimal Control and Estimation Using Nonlinear Programming. 2nd ed. Philadelphia, PA: SIAM; 2010.

92. Falisse A, Serrancolı´ G, Dembia CL, Gillis J, De Groote F. Algorithmic differentiation improves the computational efficiency of OpenSim-based trajectory optimization of human movement. PLoS One. 2019; 14: e0217730. https://doi.org/10.1371/journal.pone.0217730 PMID: 31622352

93. Andersson JAE, Gillis J, Horn G, Rawlings JB, Diehl M. CasADi: a software framework for nonlinear optimization and optimal control. Math Program Comput. 2019; 11: 1–36. https://doi.org/10.1007/ S12532-018-0139-4/TABLES/3

94. Wa¨chter A, Biegler LT. On the implementation of an interior-point filter line-search algorithm for largescale nonlinear programming. Math Program. 2006; 106: 25–57. https://doi.org/10.1007/S10107-004- 0559-Y

95. Winter DA. Biomechanics and Motor Control of Human Movement. 3rd ed. Hoboken, NJ: John Wiley and Sons Inc.; 2005.

96. Suydam SM, Manal K, Buchanan TS. The advantages of normalizing electromyography to ballistic rather than isometric or isokinetic tasks. J Appl Biomech. 2017; 33: 189–196. https://doi.org/10.1123/ jab.2016-0146 PMID: 27918690

97. Virtanen P, Gommers R, Oliphant TE, Haberland M, Reddy T, Cournapeau D, et al. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nat Methods. 2020; 17: 261–272. https://doi. org/10.1038/s41592-019-0686-2 PMID: 32015543

98. Seabold S, Perktold J. Statsmodels: Econometric and Statistical Modeling with Python. Proceedings of the 9th Python in Science Conference. 2010. pp. 92–96. https://doi.org/10.25080/majora-92bf1922- 011

99. Vallat R. Pingouin: statistics in Python. J Open Source Softw. 2018; 3: 1026.

100. Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. J R Stat Soc B. 1995; 57: 289–300. https://doi.org/10.2307/2346101

101. Hunt MA, Simic M, Hinman RS, Bennell KL, Wrigley T V. Feasibility of a gait retraining strategy for reducing knee joint loading: increased trunk lean guided by real-time biofeedback. J Biomech. 2011; 44: 943–947. https://doi.org/10.1016/j.jbiomech.2010.11.027 PMID: 21144522

102. Kaneda JM, Seagers KA, Uhlrich SD, Kolesar JA, Thomas KA, Delp SL. Can static optimization detect changes in peak medial knee contact forces induced by gait modifications? J Biomech. 2023;152. https://doi.org/10.1016/j.jbiomech.2023.111569 PMID: 37058768

103. Scarborough DM, McGibbon CA, Krebs DE. Chair rise strategies in older adults with functional limitations. J Rehabil Res Dev. 2007; 44: 33–41. https://doi.org/10.1682/jrrd.2005.08.0134 PMID: 17551856

104. Christiansen CL, Bade MJ, Weitzenkamp DA, Stevens-Lapsley JE. Factors predicting weight-bearing asymmetry 1 month after unilateral total knee arthroplasty: A cross-sectional study. Gait Posture. 2013; 37: 363–367. https://doi.org/10.1016/j.gaitpost.2012.08.006 PMID: 22980137