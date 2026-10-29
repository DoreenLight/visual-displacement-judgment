<!-- source: https://arxiv.org/abs/2503.00794 -->
<!-- title:  -->
<!-- fetched: 2026-09-30T14:14:37Z -->
<!-- via: expert_seed_search("章隆彬 Longbin Zhang Hunan University 人工智能与机器人学院 rehabilitation robotics exoskeleton neuromusculoskeletal model surface EMG markerless motion capture brain computer interface muscle tendon estimation joint torque prediction gait balance assessment 实验室团队 合作网络") -->

Detecting Heel Strike and toe off Events Using Kinematic Methods and LSTM Models

  
    
      Report GitHub Issue

      ×
    

    
      Title:
      

      Content selection saved. Describe the issue below:

      Description:
      
    

    
      Submit without GitHub
      Submit in GitHub
    
  

    
    arXiv is now an independent nonprofit!
    Learn more
    ×
  

  
    
      Back to arXiv
    
  

  
  
    Why HTML?
    
      
        
      
      Report Issue
    
    
      
        
        
      
      Back to Abstract
    
    
    
      
        
      
      Download PDF
    
    
      
        
        
      
    
    
    
      
        
      
    
    
    
      
        
          
          
        
      
      
        
          
          
        
      
      
        
          
          
        
      
    
  

Abstract

I Introduction

II Methods

II-A Experimental Setup

II-B Gait Event Identification Models

II-C Evaluation Protocol

III Results

IV Discussion

V Conclusion

References

  
    License: arXiv.org perpetual non-exclusive license
  
  
arXiv:2503.00794v3 [cs.RO] 24 Aug 2026

Detecting Heel Strike and toe off Events Using Kinematic Methods and LSTM Models

Longbin Zhang

  
Zhizhang Li

  
Xinyi Fu

  
Yi Xie

  
Zekun Liu

  
Xiaoyue Yan

  
Suiyuan Wang

  
Te ZhangHui Zhang, Kailun Yang, Tsung-Lin Wu, Prayook Jatesiktat, Ananda Sidarta, and Wei Tech Ang

Thanks: This work was supported by the National Natural Science Foundation of China under Grant 62503165, Hunan Provincial Research and Development Project under Grant 2026QK3021, Yuelushan Center for Industrial Innovation, and Health Commission of Hunan Province. The authors also acknowledge the support from Nanyang Technological University, the Agency for Science Technology and Research (A*STAR), and the National Healthcare Group (NHG). (Corresponding author: ZhiZhang Li.)

Thanks: L. Zhang is with the School of Artificial Intelligence and Robotics, Hunan University, Changsha, China and the School of Mechanical and Aerospace Engineering, Nanyang Technological University, Singapore.

Thanks: Z. Li, X. Fu, Y. Xie, Z. Liu, H. Zhang and K. Yang are with the School of Artificial Intelligence and Robotics, Hunan University, Changsha, China.

Thanks: X. Yan, T. Wu, P. Jatesiktat, and W. T. Ang are with the School of Mechanical and Aerospace Engineering, Nanyang Technological University.

Thanks: S. Wang is with the Fourth Hospital of Changsha, Hunan Normal University, Changsha, China.

Thanks: T. Zhang is with the Department of Stomatology, Hunan Provincial People’s Hospital, Hunan Normal University, Changsha, China.

Thanks: A. Sidarta is with the Rehabilitation Research Institute of Singapore, Lee Kong Chian School of Medicine, Nanyang Technological University, Singapore.

Abstract

    
Accurate gait event detection is crucial for gait analysis, rehabilitation, and assistive technology, particularly in exoskeleton control, where precise identification of stance and swing phases is essential. This study evaluated the performance of seven kinematics-based methods and a Long Short-Term Memory (LSTM) model for detecting heel strike and toe-off events across 4363 gait cycles from 588 able-bodied subjects. The results indicated that while the Zeni et al. method achieved the highest accuracy among kinematics-based approaches, other methods exhibited systematic biases or required dataset-specific tuning. The LSTM model performed comparably to Zeni et al., providing a data-driven alternative without systematic bias. These findings highlight the potential of deep learning-based approaches for gait event detection while emphasizing the need for further validation in clinical populations and across diverse gait conditions. Future research will explore the generalizability of these methods in pathological populations, such as individuals with post-stroke conditions and knee osteoarthritis, as well as their robustness across varied gait conditions and data collection settings to enhance their applicability in rehabilitation and exoskeleton control.

  

I Introduction

Exoskeletons are emerging as powerful assistive technologies designed to enhance mobility, restore gait function, and reduce the physical burden on users in both rehabilitation and everyday settings [15, 14, 29]. A key challenge in exoskeleton control is the precise timing of assistance, which relies on accurate gait event detection to differentiate between stance and swing phases [16, 17].
Misidentification of these events can lead to improper assistance timing, reducing efficiency, increasing metabolic cost, and potentially compromising user safety. Gait event detection is, therefore, a fundamental component of gait analysis, ensuring the precise identification of key moments within the gait cycle, such as heel strike (HS) and toe off (TO) [12]. Accurate detection is essential for calculating spatiotemporal parameters and kinematic variables, which are critical for optimizing exoskeleton control strategies. Errors in gait event identification can propagate through the system, leading to inaccurate joint angle estimations and impaired motion synchronization [5, 26]. In clinical settings, precise gait event detection is vital for diagnosing gait abnormalities, monitoring rehabilitation progress, and tailoring interventions [1]. It also plays a crucial role in providing real-time feedback to patients, preventing secondary complications, and customizing assistive devices [19]. Beyond clinical applications, accurate gait event detection supports the optimization of prosthetics and orthotics, enhances athletic performance, and enables predictive modeling for fall risk [20]. Given its significance in both rehabilitation and assistive robotics [27], improving gait event detection is essential for advancing exoskeleton technology, ensuring smooth user-exoskeleton interaction, and enhancing mobility for individuals with gait impairments.

The gold standard for identifying HS and TO in gait analysis involves using force plates with predefined vertical ground reaction force (GRF) thresholds, typically ranging from 5 to 20 N [25, 21], in a laboratory setup where subjects walk across a walkway embedded with force plates [3, 13, 2].
However, the availability of force plates and the ability to achieve sufficient clean foot contacts can be significant challenges, particularly in cases with pathological characteristics such as foot drag, small step size, or the use of walking aids [5]. Additionally, employing force plates often necessitates a force-sensitive walkway or multiple force plates, which are expensive and impractical, requiring each footstep to occur on a single force plate and ensuring only one foot contacts the plate at a time [9]. Portable alternatives, such as pressure insoles or foot switches, offer some flexibility but require additional equipment and modifications to the subject’s footwear, limiting their use in populations with abnormal gait patterns, especially those lacking sufficient foot clearance during the swing phase. Furthermore, the accuracy of these portable methods heavily depends on the precise placement of sensors [26]. Despite the challenges, force plate thresholding remains the gold standard, but there is a continuous need for advancements in gait event detection methods to improve accuracy and applicability across diverse clinical and research settings.

Kinematics-based models and deep learning approaches are widely used for gait event detection in both healthy and pathological populations [26, 4, 8, 18, 11, 10, 13, 12]. These models often rely on foot or pelvis segments captured through motion capture systems or wearable sensors like IMUs [7, 22, 2]. French et al. [7] found that horizontal ankle-heel distance was most accurate for detecting heel strike during overground walking, while sacral-heel distance was best for toe off (TO). Voisard et al. [22] used inertial measurement units (IMUs) to detect gait events, showing high accuracy in healthy subjects (TO: 8 ms, HS: 7 ms) and reasonable accuracy in multiple sclerosis patients (TO: 23 ms, HS: 15 ms). Bonci et al. [2] combined marker position and velocity, achieving over 99% sensitivity but limited by small sample sizes and marker requirements.
Deep learning models, including Long Short-Term Memory (LSTM) networks, have emerged as alternatives. Lempereur et al. [13] used LSTM to detect gait events in children with gait disorders, achieving high accuracy with HS detected within 5.5 ms and TO within 10.7 ms. Kim et al. [12] achieved 89.7% detection for HS and 71.6% for TO in children with cerebral palsy. Filtjens et al. [6] employed Temporal Convolutional Networks for Parkinson’s disease, achieving F1-scores exceeding 0.99. Dumphart et al. [5] used LSTM to detect events in both healthy individuals and those with pathologies, reporting high detection rates but emphasizing the influence of marker placement and variability in laboratory setup. Despite advances, few studies have compared kinematics-based and deep learning models across data centers and diverse subject groups.

The objective of this study was to detect heel strike and toe off events using kinematics-based methods and an LSTM model and to evaluate their prediction accuracy. We hypothesized that the LSTM model would achieve comparable or superior accuracy to traditional kinematics-based methods while maintaining consistency across trials.

Fig. 1: Schematic diagram of gait event identification using kinematics-based or machine learning models. The inputs for these models consist of key marker trajectories from the toe, heel, or pelvis, acquired and post-processed using a 3D motion capture system.

II Methods

II-A Experimental Setup

A total of 4363 gait cycles from 588 able-bodied subjects (age: 45 ±\pm 16.2 years; sex: 317F/271M; weight: 62.7 ±\pm 13.5 kg; height: 164.4 ±\pm 9.1 cm) were collected from the Rehabilitation Research Institute of Singapore. The study was approved by the Nanyang Technological University Institutional Review Board (IRB-2022-382). Marker trajectories were recorded using a marker-based Miqus M3 Qualisys system (200 Hz) in Singapore. Ground reaction forces were simultaneously measured and used to identify heel strike and toe off events using a 20 N threshold, serving as the ground truth for both kinematics-based and machine learning models (Fig. 1).

TABLE I: The Kinematics-based approaches used to detect HS and TO

Method
Markers
HS Detection Features
TO Detection Features

Foot trajectories relative to pelvis
(Zeni et al. 2008)

Heel, toe, and pelvis

Local maximum horizontal position of the heel marker relative to the pelvis

Local minimal horizontal position of the toe marker relative to the pelvis

High-pass filtered foot markers
(Desailly et al. 2009)

Heel and toe

First maximum between high pass filtered (0.5 * gait frequency) horizontal heel marker trajectory

Last minimum between high pass filtered (1.1 * gait frequency) horizontal toe marker trajectory

Foot markers velocity extreme (O’Connor et al. 2007)

Heel and toe

Local minimal vertical velocity of the mid-point of heel and toe markers

Local maximum vertical velocity of the mid-point of heel and toe markers

Foot markers Velocity Threshold
(Ghoussayni et al. 2004)

Heel and toe

Sagittal velocity of heel marker lower than 50 cm/s

Sagittal velocity of toe marker higher than 50 cm/s

Foot markers acceleration extreme (Hreljac et al. 2000)

Heel and toe

Local maximum vertical acceleration of the heel marker, with the condition that the derivative of the acceleration (jerk) was zero.

Local maximum horizontal acceleration of the toe marker, also with the constraint that the relevant jerk was zero

Foot markers acceleration extreme (Hsue et al. 2007)

Heel and toe

Local minimal horizontal acceleration of heel marker

Local maximum horizontal acceleration of toe marker

Position and Velocity combined
(Bonci et al. 2022)

Heel, Toe and Pelvis

Initially detected using the method from Zeni et al. (2008) and then refined based on the 3D velocity of heel marker: lower than 0.5 * walking speed for rearfoot contacts and 0.8 * walking speed for forefoot contacts

Initially detected using the method from Zeni et al. and then refined when the 3D velocity of the toe marker exceeded 0.8 times the walking speed, and further refined when the 3D velocity of the heel marker decreased after its local peak

II-B Gait Event Identification Models

(a) Kinematics-based Models

To identify heel strike and toe off gait events, we evaluated seven kinematics-based methods derived from peer-reviewed literature. These methods employed different biomechanical principles, utilizing positional, velocity, or acceleration-based criteria applied to key anatomical markers, including those placed on the heel, toe, and pelvis (Table I). Each approach had unique advantages and limitations, particularly in terms of sensitivity to gait variations and robustness across different walking conditions.

• 

Positional-Based Methods:

Zeni et al. (2008) proposed a widely used method for detecting HS and TO based on the local extrema of heel and toe marker positions relative to the pelvis. Specifically, HS was identified when the heel marker reached its maximum horizontal displacement, while TO was detected when the toe marker reached its minimum horizontal displacement, both measured relative to the pelvis (Table I).

Desailly et al. (2009) introduced a filtering-based approach in which high-pass filtering was applied to horizontal heel and toe marker trajectories to eliminate low-frequency drift. HS was detected at the first maximum in the high-pass filtered (0.5 * gait frequency) horizontal heel marker trajectory, while TO was identified at the last minimum in the high-pass filtered (1.1 * gait frequency) horizontal toe marker trajectory. This method was designed to enhance accuracy by mitigating the effects of marker noise and soft tissue artifacts.

• 

Velocity-Based Methods:

O’Connor et al. (2007) employed a velocity-based detection method, defining HS and TO based on vertical velocity extrema. Specifically, HS was identified at the local minimum and TO at the local maximum of the vertical velocity of the midpoint between the heel and toe markers.

Ghoussayni et al. (2004) introduced a threshold-based velocity approach, where HS was detected when the sagittal velocity of the heel marker fell below 5 cm/s, and TO was identified when the sagittal velocity of the toe marker exceeded 5 cm/s. However, the velocity threshold recommended by Ghoussayni et al. was later found to be too low and was increased to 50 cm/s by Bruening et al. [3]. In this study, we adopt the revised threshold of 50 cm/s.

• 

Acceleration-Based Methods:

Hreljac et al. (2000) used acceleration profiles to detect HS and TO, incorporating vertical and horizontal acceleration components to capture gait event dynamics. HS was identified at the local maximum vertical acceleration of the heel marker, provided that the derivative of acceleration (jerk) was zero. Similarly, TO was detected at the local maximum horizontal acceleration of the toe marker, also under the condition that the corresponding jerk was zero.

Hsue et al. (2007) extended acceleration-based detection by introducing jerk constraints (rate of change of acceleration). HS was identified at the local minimum of the heel marker’s horizontal acceleration, while TO was detected at the local maximum of the toe marker’s horizontal acceleration.

• 

Hybrid Approaches:

Bonci et al. (2022) refined Zeni’s approach by incorporating 3D velocity thresholds relative to walking speed. HS was initially detected using the method from Zeni et al. (2008) and then adjusted based on the 3D velocity of the heel marker: below 0.5 * walking speed for rearfoot contacts and 0.8 * walking speed for forefoot contacts. TO was first identified using Zeni et al.’s (2008) method, then refined when the 3D velocity of the toe marker exceeded 0.8 * walking speed, with a further adjustment when the 3D velocity of the heel marker decreased after reaching its local peak.

Fig. 2: The boxplot of prediction errors (time differences) between the measured and calculated heel strike (left) and toe off (right) events using various kinematics-based methods and the LSTM model.

Fig. 3: The histogram of time differences between the measured and calculated heel strike (left) and toe off (right) events using various kinematics-based methods and the LSTM model.

(b) LSTM Networks

LSTM models were developed to identify gait events using the position and velocity of pelvic, heel, and toe markers. These markers included those placed on the left anterior superior iliac spine (LASIS), left posterior superior iliac spine (LPSIS), left heel (LFCC), left second metatarsal head (LFMT2), right anterior superior iliac spine (RASIS), right posterior superior iliac spine (RPSIS), right heel (RFCC), and right second metatarsal head (RFMT2). Additionally, the heel position relative to the pelvis was used as input. The model begins with a Masking layer to ignore padded values (zeros) in the input. It includes a single LSTM layer with 128 units and a 30% dropout rate, followed by a Dense layer with 32 neurons using ReLU activation to capture high-level feature representations. Finally, a time-distributed Dense layer with a softmax activation function outputs probabilities across five nodes, corresponding to no gait event, left heel HS/TO, and right HS/TO events. Gait events were detected using a simple peak detection algorithm with a threshold of 0.01, as proposed by Lempereur et al. [13].

Hyperparameter optimization was conducted using Keras Tuner’s RandomSearch approach. The search space included key parameters such as the number of LSTM units (32–128, in steps of 32), dropout rates (0.2–0.5, in steps of 0.1), batch size (16, 32, 64, 128), and optimizer type (Adam or RMSprop). A total of 20 trials were evaluated, each testing a unique combination of hyperparameters, with validation accuracy serving as the optimization metric. An early stopping mechanism was applied to ensure efficient tuning without overfitting. After completing the search, the best-performing hyperparameters were used to build and train the final model.

II-C Evaluation Protocol

Gait event identification accuracy was evaluated by computing the temporal error (time difference) between the estimated and actual event times across both kinematics-based and machine learning-based models. The resulting discrepancies were analyzed to assess the performance of each approach in detecting key gait events.

TABLE II: Mean and standard deviation (STD) of time differences for heel strike and toe off events across kinematics-based methods and the LSTM model.

Method
Heel Strike
Toe off

Mean (ms)
STD (ms)
Mean (ms)
STD (ms)

Zeni et al.
4.78
9.56
-8.66
6.05

Desailly et al.
22.17
8.66
7.45
7.62

O’Connor et al.
46.73
7.72
-16.27
8.35

Ghoussayni et al.
50.49
11.28
-61.96
8.67

Hreljac et al.
32.28
31.56
4.70
8.14

Hsue et al.
-11.86
11.50
4.70
8.14

Bonci et al.
36.31
9.15
-16.63
10.34

LSTM
-0.72
9.56
-6.52
10.34

III Results

Among the kinematics-based methods, the Zeni et al. approach had the smallest time difference for heel strike (4.78 ±\pm 9.56 ms, Table II), aligning most closely with the zero time difference line, followed by the methods of Hsue et al. and Desailly et al. (Fig. 2 and 3). In contrast, the methods of Desailly et al., O’Connor et al., Ghoussayni et al., and Bonci et al. detected heel strike later than the measured values, while the Hreljac et al. method exhibited both early and late detections. The highest variability was observed in the Hreljac et al. method (31.56 ms), indicating less consistency across trials (Fig. 2).

For toe off detection, the methods of Desailly et al., Hreljac et al., and Bonci et al. had time differences close to zero, whereas the Ghoussayni et al. method detected toe off events considerably earlier (-61.96 ±\pm 8.67 ms), suggesting a systematic bias toward early detections. O’Connor et al. and Bonci et al. also showed a tendency for earlier detections, but with smaller deviations (-16.27 ms and -16.63 ms, respectively). Variability among the methods was also evident, with some methods, such as O’Connor et al., showing lower variability (8.35 ms), while others, such as Hreljac et al., exhibited wider spreads in heel strike detection.

The LSTM-based approach demonstrated low mean time differences for both heel strike (-0.72 ±\pm 9.56 ms) and toe off (-6.52 ±\pm 10.34 ms), with values centered around zero. However, its performance was comparable to that of Zeni et al., which also showed relatively small deviations in heel strike detection.

IV Discussion

In this study, we evaluated gait event detection using seven kinematics-based computational models and one LSTM model. Our results showed that the Zeni et al. method achieved the highest accuracy among the kinematics-based approaches, while other methods exhibited systematic biases or required specific adjustments for different datasets. The LSTM model performed similarly to the Zeni et al. method, providing a data-driven alternative that avoided systematic biases. These findings hold potential implications for gait analysis research and applications such as assistive exoskeletons, where accurate identification of stance and swing phases is crucial for controlling algorithms in varied environments. By enhancing the accuracy and robustness of gait event detection, these advancements could play a vital role in supporting rehabilitation efforts, improving clinical assessments, and advancing the development of assistive technologies.

Our results indicated that while certain kinematics-based methods, such as Zeni et al., provided relatively accurate gait event detection, others exhibited systematic shifts or greater variability. Kinematics-based approaches are widely used for gait analysis, particularly when force plates are unavailable. Among the evaluated methods, Zeni et al. demonstrated the smallest time difference for heel strike (4.78 ± 9.56 ms), likely due to the alignment of its reference dataset with our study population. Velocity threshold-based methods, such as those by O’Connor et al. and Ghoussayni et al., required careful tuning of thresholds across different datasets to maintain accuracy. Acceleration-based methods, including those by Hreljac et al. and Hsue et al., exhibited good predictive accuracy for horizontal components but were less reliable for vertical components, which appeared to be influenced by individual walking patterns. Interestingly, the position- and velocity-based method by Bonci et al. did not achieve the highest accuracy in our study. This may be attributed to the dataset used in its original validation. Bonci et al. initially tested their method against existing algorithms using data from healthy young adults (n = 20) and later assessed it in a small cohort of 10 individuals across five groups: older adults, individuals with chronic obstructive pulmonary disease, multiple sclerosis, Parkinson’s disease, and proximal femur fractures. In contrast, our study, which only had normative data, demonstrated that Zeni et al.’s method achieved strong accuracy without requiring extensive dataset-specific tuning.

The LSTM model demonstrated promising results, with low mean time differences and no systematic bias. Its performance was comparable to that of Zeni et al., suggesting that deep learning-based approaches can serve as viable alternatives to traditional rule-based methods. Unlike conventional approaches that rely on predefined thresholds, the LSTM model leverages its ability to learn temporal dependencies in gait kinematics, capturing complex relationships between movement phases [28, 23, 24]. However, while the LSTM model performed well on unseen data within the same dataset, its generalizability to broader populations, particularly those with pathological gait patterns, remains uncertain. Deep learning models typically excel when evaluated on datasets similar to their training data, and further studies are required to assess their robustness across diverse movement patterns. Expanding the evaluation to individuals with gait impairments will provide valuable insights into the adaptability and potential clinical utility of LSTM-based gait event detection.

This study has several limitations. The evaluation was conducted solely on a normative dataset, limiting its applicability to populations with altered gait patterns. Future work should include individuals with conditions such as knee osteoarthritis or post-stroke impairments to assess the generalizability of the tested methods. Also, only one machine learning model (LSTM) was evaluated. Exploring alternative deep learning architectures, such as transformers, could provide further insights into model adaptability across different datasets. Lastly, cross-center validation with data from multiple research facilities would help determine the reproducibility and robustness of these approaches in diverse clinical and real-world settings.

V Conclusion

This study assessed the performance of seven kinematics-based methods and one LSTM-based approach for gait event detection. While some kinematics-based methods, particularly Zeni et al., demonstrated strong accuracy, others exhibited systematic shifts or required dataset-specific tuning. The LSTM model showed promising results, performing comparably to the most accurate kinematics-based method without systematic bias. However, its applicability to individuals with gait impairments remains to be explored. Future research should extend these evaluations to clinical populations and explore additional machine learning techniques, such as transformer models, to enhance robustness and generalizability across diverse gait conditions and data collection settings.

References

[1]
R. Baker and H. M. Hart (2013)

Measuring walking: a handbook of clinical gait analysis.

Vol. 1,  Mac Keith Press London.

Cited by: §I.

[2]
T. Bonci, F. Salis, K. Scott, L. Alcock, C. Becker, S. Bertuletti, E. Buckley, M. Caruso, A. Cereatti, S. Del Din, et al. (2022)

An algorithm for accurate marker-based gait event detection in healthy and pathological populations during complex motor tasks.

Frontiers in Bioengineering and Biotechnology 10, pp. 1–14.

Cited by: §I,
§I.

[3]
D. A. Bruening and S. T. Ridge (2014)

Automated event detection algorithms in pathological gait.

Gait & Posture 39 (1), pp. 472–477.

Cited by: §I,
§II-B.

[4]
E. Desailly, Y. Daniel, P. Sardain, and P. Lacouture (2009)

Foot contact event detection using kinematic data in cerebral palsy children and normal adults gait.

Gait & Posture 29 (1), pp. 76–80.

Cited by: §I.

[5]
B. Dumphart, D. Slijepcevic, M. Zeppelzauer, A. Kranzl, F. Unglaube, A. Baca, and B. Horsak (2023)

Robust deep learning-based gait event detection across various pathologies.

Plos One 18 (8), pp. 1–16.

Cited by: §I,
§I,
§I.

[6]
B. Filtjens, A. Nieuwboer, N. D’cruz, J. Spildooren, P. Slaets, and B. Vanrumste (2020)

A data-driven approach for detecting gait events during turning in people with parkinson’s disease and freezing of gait.

Gait & Posture 80, pp. 130–136.

Cited by: §I.

[7]
M. A. French, C. Koller, and E. S. Arch (2020)

Comparison of three kinematic gait event detection methods during overground and treadmill walking for individuals post stroke.

Journal of Biomechanics 99, pp. 1–7.

Cited by: §I.

[8]
S. Ghoussayni, C. Stevens, S. Durham, and D. Ewins (2004)

Assessment and validation of a simple automated method for the detection of gait events and intervals.

Gait & Posture 20 (3), pp. 266–272.

Cited by: §I.

[9]
A. H. Hansen, D. S. Childress, and M. R. Meier (2002)

A simple method for determination of gait events.

Journal of Biomechanics 35 (1), pp. 135–138.

Cited by: §I.

[10]
A. Hreljac and R. N. Marshall (2000)

Algorithms to determine event timing during normal walking using kinematic data.

Journal of Biomechanics 33 (6), pp. 783–786.

Cited by: §I.

[11]
B. Hsue, F. Miller, F. Su, J. Henley, and C. Church (2007)

Gait timing event determination using kinematic data for the toe walking children with cerebreal palsy.

Journal of Biomechanics 40 (2), pp. S529.

Cited by: §I.

[12]
Y. K. Kim, R. M. Visscher, E. Viehweger, N. B. Singh, W. R. Taylor, and F. Vogl (2022)

A deep-learning approach for automatically detecting gait-events based on foot-marker kinematics in children with cerebral palsy—which markers work best for which gait patterns?.

Plos One 17 (10), pp. 1–16.

Cited by: §I,
§I.

[13]
M. Lempereur, F. Rousseau, O. Rémy-Néris, C. Pons, L. Houx, G. Quellec, and S. Brochard (2020)

A new deep learning-based method for the detection of gait events in children with gait disorders: proof-of-concept and concurrent validity.

Journal of Biomechanics 98, pp. 1–4.

Cited by: §I,
§I,
§II-B.

[14]
Z. Li, B. Huang, A. Ajoudani, C. Yang, C. Su, and A. Bicchi (2017)

Asymmetric bimanual control of dual-arm exoskeletons for human-cooperative manipulations.

IEEE Transactions on Robotics 34 (1), pp. 264–271.

Cited by: §I.

[15]
Z. Li, B. Huang, Z. Ye, M. Deng, and C. Yang (2018)

Physical human–robot interaction of a robotic exoskeleton by admittance control.

IEEE Transactions on Industrial Electronics 65 (12), pp. 9614–9624.

Cited by: §I.

[16]
Z. Li, Y. Yuan, L. Luo, W. Su, K. Zhao, C. Xu, J. Huang, and M. Pi (2019)

Hybrid brain/muscle signals powered wearable walking exoskeleton enhancing motor ability in climbing stairs activity.

IEEE Transactions on Medical Robotics and Bionics 1 (4), pp. 218–227.

Cited by: §I.

[17]
L. Luo, M. J. Foo, M. Ramanathan, J. K. Er, C. H. Chiam, L. Li, W. Y. Yau, and W. T. Ang (2022)

Trajectory generation and control of a lower limb exoskeleton for gait assistance.

Journal of Intelligent & Robotic Systems 106 (3), pp. 64.

Cited by: §I.

[18]
C. M. O’Connor, S. K. Thorpe, M. J. O’Malley, and C. L. Vaughan (2007)

Automatic detection of gait events using kinematic data.

Gait & Posture 25 (3), pp. 469–474.

Cited by: §I.

[19]
S. R. Simon (2004)

Quantification of human motion: gait analysis—benefits and limitations to its application to clinical problems.

Journal of Biomechanics 37 (12), pp. 1869–1880.

Cited by: §I.

[20]
D. H. Sutherland (2002)

The evolution of clinical gait analysis: part ii kinematics.

Gait & Posture 16 (2), pp. 159–179.

Cited by: §I.

[21]
O. Tirosh and W. Sparrow (2003)

Identifying heel contact and toe-off using forceplate thresholds with a range of digital-filter cutoff frequencies.

Journal of Applied Biomechanics 19 (2), pp. 178–184.

Cited by: §I.

[22]
C. Voisard, N. de l’Escalopier, D. Ricard, and L. Oudre (2024)

Automatic gait events detection with inertial measurement units: healthy subjects and moderate to severe impaired patients.

Journal of NeuroEngineering and Rehabilitation 21 (1), pp. 1–17.

Cited by: §I.

[23]
Z. Xia, B. M. Cornish, D. Devaprakash, R. S. Barrett, D. G. Lloyd, A. H. Hams, and C. Pizzolato (2024)

Prediction of achilles tendon force during common motor tasks from markerless video.

IEEE Transactions on Neural Systems and Rehabilitation Engineering.

Cited by: §IV.

[24]
L. Xiang, Y. Gu, Z. Gao, P. Yu, V. Shim, A. Wang, and J. Fernandez (2024)

Integrating an lstm framework for predicting ankle joint biomechanics during gait using inertial sensors.

Computers in Biology and Medicine 170, pp. 108016.

Cited by: §IV.

[25]
N. Zahradka, K. Verma, A. Behboodi, B. Bodt, H. Wright, and S. C. Lee (2020)

An evaluation of three kinematic methods for gait event detection compared to the kinetic-based ‘gold standard’.

Sensors 20 (18), pp. 1–15.

Cited by: §I.

[26]
J. Zeni Jr, J. Richards, and J. Higginson (2008)

Two simple methods for determining gait events during treadmill and overground walking using kinematic data.

Gait & Posture 27 (4), pp. 710–714.

Cited by: §I,
§I,
§I.

[27]
L. Zhang, Y. Liu, R. Wang, C. Smith, and E. M. Gutierrez-Farewik (2021)

Modeling and simulation of a human knee exoskeleton’s assistive strategies and interaction.

Frontiers in Neurorobotics 15, pp. 1–12.

Cited by: §I.

[28]
L. Zhang, D. Soselia, R. Wang, and E. M. Gutierrez-Farewik (2022)

Lower-limb joint torque prediction using lstm neural networks and transfer learning.

IEEE Transactions on Neural Systems and Rehabilitation Engineering 30, pp. 600–609.

Cited by: §IV.

[29]
L. Zhang, X. Zhu, E. M. Gutierrez-Farewik, and R. Wang (2022)

Ankle joint torque prediction using an nms solver informed-ann model and transfer learning.

IEEE Journal of Biomedical and Health Informatics 26 (12), pp. 5895–5906.

Cited by: §I.

  
    Experimental support, please
    view the build logs
    for errors. Generated by
    
      
        L
        A
        T
        E
      
      xml
      
    .
  

  
    Instructions for reporting errors

    We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile
      support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the
      methods listed below:

    
      Click the "Report Issue" (
            
          ) button, located in the page header.

    
    Tip: You can select the relevant text first, to include it in your report.

    Our team has already identified the following issues. We appreciate your time reviewing and reporting rendering errors we
      may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability
      should not be a barrier to accessing research. Thank you for your continued support in championing open access for
      all.

    Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a list of packages that need conversion, and welcome developer contributions.

  

  
    
      
        We gratefully acknowledge support from
        our major funders,
        member institutions, ,
        and all contributors.
      

      
        About
        ·
        Help
        ·
        Contact
        ·
        Subscribe
        ·
        Copyright
        ·
        Privacy
        ·
        Accessibility
        ·
        Operational Status (opens in new tab)
      
    

    
      Major funding support from