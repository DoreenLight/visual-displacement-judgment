# Detecting Heel Strike and toe off Events Using Kinematic Methods and LSTM Models

Longbin Zhang, Zhizhang Li<sup>\*</sup>, Xinyi Fu, Yi Xie, Zekun Liu, Xiaoyue Yan, Suiyuan Wang, Te Zhang, Hui Zhang, Kailun Yang, Tsung-Lin Wu, Prayook Jatesiktat, Ananda Sidarta, and Wei Tech Ang

Abstract— Accurate gait event detection is crucial for gait analysis, rehabilitation, and assistive technology, particularly in exoskeleton control, where precise identification of stance and swing phases is essential. This study evaluated the performance of seven kinematics-based methods and a Long Short-Term Memory (LSTM) model for detecting heel strike and toe-off events across 4363 gait cycles from 588 able-bodied subjects. The results indicated that while the Zeni et al. method achieved the highest accuracy among kinematics-based approaches, other methods exhibited systematic biases or required dataset-specific tuning. The LSTM model performed comparably to Zeni et al., providing a data-driven alternative without systematic bias. These findings highlight the potential of deep learning-based approaches for gait event detection while emphasizing the need for further validation in clinical populations and across diverse gait conditions. Future research will explore the generalizability of these methods in pathological populations, such as individuals with post-stroke conditions and knee osteoarthritis, as well as their robustness across varied gait conditions and data collection settings to enhance their applicability in rehabilitation and exoskeleton control.

## I. INTRODUCTION

Exoskeletons are emerging as powerful assistive technologies designed to enhance mobility, restore gait function, and reduce the physical burden on users in both rehabilitation and everyday settings [1]–[3]. A key challenge in exoskeleton control is the precise timing of assistance, which relies on accurate gait event detection to differentiate between stance and swing phases [4], [5]. Misidentification of these events can lead to improper assistance timing, reducing efficiency, increasing metabolic cost, and potentially compromising user safety. Gait event detection is, therefore, a fundamental component of gait analysis, ensuring the precise identification of key moments within the gait cycle, such as heel strike (HS) and toe off (TO) [6]. Accurate detection is essential for calculating spatiotemporal parameters and kinematic variables, which are critical for optimizing exoskeleton control strategies. Errors in gait event identification can propagate through the system, leading to inaccurate joint angle estimations and impaired motion synchronization [7], [8]. In clinical settings, precise gait event detection is vital for diagnosing gait abnormalities, monitoring rehabilitation progress, and tailoring interventions [9]. It also plays a crucial role in providing real-time feedback to patients, preventing secondary complications, and customizing assistive devices [10]. Beyond clinical applications, accurate gait event detection supports the optimization of prosthetics and orthotics, enhances athletic performance, and enables predictive modeling for fall risk [11]. Given its significance in both rehabilitation and assistive robotics [12], improving gait event detection is essential for advancing exoskeleton technology, ensuring smooth user-exoskeleton interaction, and enhancing mobility for individuals with gait impairments.

The gold standard for identifying HS and TO in gait analysis involves using force plates with predefined vertical ground reaction force (GRF) thresholds, typically ranging from 5 to 20 N [13], [14], in a laboratory setup where subjects walk across a walkway embedded with force plates [15]–[17]. However, the availability of force plates and the ability to achieve sufficient clean foot contacts can be significant challenges, particularly in cases with pathological characteristics such as foot drag, small step size, or the use of walking aids [7]. Additionally, employing force plates often necessitates a force-sensitive walkway or multiple force plates, which are expensive and impractical, requiring each footstep to occur on a single force plate and ensuring only one foot contacts the plate at a time [18]. Portable alterna tives, such as pressure insoles or foot switches, offer some flexibility but require additional equipment and modifications to the subject’s footwear, limiting their use in populations with abnormal gait patterns, especially those lacking sufficient foot clearance during the swing phase. Furthermore, the accuracy of these portable methods heavily depends on the precise placement of sensors [8]. Despite the challenges, force plate thresholding remains the gold standard, but there is a continuous need for advancements in gait event detection methods to improve accuracy and applicability across diverse clinical and research settings.

Kinematics-based models and deep learning approaches are widely used for gait event detection in both healthy and pathological populations [6], [8], [16], [19]–[23]. These models often rely on foot or pelvis segments captured through motion capture systems or wearable sensors like IMUs [17], [24], [25]. French et al. [24] found that horizontal ankle-heel distance was most accurate for detecting heel strike during overground walking, while sacral-heel distance was best for toe off (TO). Voisard et al. [25] used inertial measurement units (IMUs) to detect gait events, showing high accuracy in healthy subjects (TO: 8 ms, HS: 7 ms) and reasonable accuracy in multiple sclerosis patients (TO: 23 ms, HS: 15 ms). Bonci et al. [17] combined marker position and velocity, achieving over 99% sensitivity but limited by small sample sizes and marker requirements. Deep learning models, including Long Short-Term Memory (LSTM) networks, have emerged as alternatives. Lempereur et al. [16] used LSTM to detect gait events in children with gait disorders, achieving high accuracy with HS detected within 5.5 ms and TO within 10.7 ms. Kim et al. [6] achieved 89.7% detection for HS and 71.6% for TO in children with cerebral palsy. Filtjens et al. [26] employed Temporal Convolutional Networks for Parkinson’s disease, achieving F1-scores exceeding 0.99. Dumphart et al. [7] used LSTM to detect events in both healthy individuals and those with pathologies, reporting high detection rates but emphasizing the influence of marker placement and variability in laboratory setup. Despite advances, few studies have compared kinematics-based and deep learning models across data centers and diverse subject groups.

The objective of this study was to detect heel strike and toe off events using kinematics-based methods and an LSTM model and to evaluate their prediction accuracy. We hypothesized that the LSTM model would achieve comparable or superior accuracy to traditional kinematics-based methods while maintaining consistency across trials.

## II. METHODS

## A. Experimental Setup

A total of 4363 gait cycles from 588 able-bodied subjects (age: $4 5 ~ \pm ~ 1 6 . 2$ years; sex: 317F/271M; weight: 62.7 ± 13.5 kg; height: 164.4 ± 9.1 cm) were collected from the Rehabilitation Research Institute of Singapore. The study was approved by the Nanyang Technological University Institutional Review Board (IRB-2022-382). Marker trajectories were recorded using a marker-based Miqus M3 Qualisys system (200 Hz) in Singapore. Ground reaction forces were simultaneously measured and used to identify heel strike and toe off events using a 20 N threshold, serving as the ground truth for both kinematics-based and machine learning models (Fig. 1).

## B. Gait Event Identification Models

## (a) Kinematics-based Models

To identify heel strike and toe off gait events, we evaluated seven kinematics-based methods derived from peer-reviewed literature. These methods employed different biomechanical principles, utilizing positional, velocity, or acceleration-based criteria applied to key anatomical markers, including those placed on the heel, toe, and pelvis (Table I). Each approach had unique advantages and limitations, particularly in terms of sensitivity to gait variations and robustness across different walking conditions.

## • Positional-Based Methods:

Zeni et al. (2008) proposed a widely used method for detecting HS and TO based on the local extrema of heel and toe marker positions relative to the pelvis. Specifically, HS was identified when the heel marker reached its maximum horizontal displacement, while TO was detected when the toe marker reached its minimum horizontal displacement, both measured relative to the pelvis (Table I).

Desailly et al. (2009) introduced a filtering-based approach in which high-pass filtering was applied to horizontal heel and toe marker trajectories to eliminate low-frequency drift. HS was detected at the first maximum in the highpass filtered (0.5 \* gait frequency) horizontal heel marker trajectory, while TO was identified at the last minimum in the high-pass filtered (1.1 \* gait frequency) horizontal toe marker trajectory. This method was designed to enhance accuracy by mitigating the effects of marker noise and soft tissue artifacts.

## • Velocity-Based Methods:

O’Connor et al. (2007) employed a velocity-based detection method, defining HS and TO based on vertical velocity extrema. Specifically, HS was identified at the local minimum and TO at the local maximum of the vertical velocity of the midpoint between the heel and toe markers.

Ghoussayni et al. (2004) introduced a threshold-based velocity approach, where HS was detected when the sagittal velocity of the heel marker fell below 5 cm/s, and TO was identified when the sagittal velocity of the toe marker exceeded 5 cm/s. However, the velocity threshold recommended by Ghoussayni et al. was later found to be too low and was increased to 50 cm/s by Bruening et al. [15]. In this study, we adopt the revised threshold of 50 cm/s.

## • Acceleration-Based Methods:

Hreljac et al. (2000) used acceleration profiles to detect HS and TO, incorporating vertical and horizontal acceleration components to capture gait event dynamics. HS was identified at the local maximum vertical acceleration of the heel marker, provided that the derivative of acceleration (jerk) was zero. Similarly, TO was detected at the local maximum horizontal acceleration of the toe marker, also under the condition that the corresponding jerk was zero.

Hsue et al. (2007) extended acceleration-based detection by introducing jerk constraints (rate of change of acceleration). HS was identified at the local minimum of the heel marker’s horizontal acceleration, while TO was detected at the local maximum of the toe marker’s horizontal acceleration.

## • Hybrid Approaches:

Bonci et al. (2022) refined Zeni’s approach by incorporating 3D velocity thresholds relative to walking speed. HS was initially detected using the method from Zeni et al. (2008) and then adjusted based on the 3D velocity of the heel marker: below 0.5 \* walking speed for rearfoot contacts and 0.8 \* walking speed for forefoot contacts. TO was first identified using Zeni et al.’s (2008) method, then refined when the 3D velocity of the toe marker exceeded 0.8 \* walking speed, with a further adjustment when the 3D velocity of the heel marker decreased after reaching its local peak.

![](images/81b83919f51e27cae4e005cc0a3e72825b79a9b924c0a7bc4072164a4d8756c4.jpg)  
Fig. 1. Schematic diagram of gait event identification using kinematics-based or machine learning models. The inputs for these models consist of ke marker trajectories from the toe, heel, or pelvis, acquired and post-processed using a 3D motion capture system.

TABLE I  
THE KINEMATICS-BASED APPROACHES USED TO DETECT HS AND TO

<table><tr><td>Method</td><td>Markers</td><td>HS Detection Features</td><td>TO Detection Features</td></tr><tr><td>Foot trajectories relative to pelvis (Zeni et al. 2008)</td><td>Heel, toe, and pelvis</td><td>Local maximum horizontal position of the heel marker relative to the pelvis</td><td>Local minimal horizontal position of the toe marker relative to the pelvis</td></tr><tr><td>High-pass filtered foot markers (Desailly et al. 2009)</td><td>Heel and toe</td><td>First maximum between high pass filtered (0.5 * gait frequency) horizontal heel marker trajectory</td><td>Last minimum between high pass filtered (1.1 * gait frequency) horizontal toe marker trajectory</td></tr><tr><td>Foot markers velocity extreme (O&#x27;Connor et al. 2007)</td><td>Heel and toe</td><td>Local minimal vertical velocity of the mid-point of heel and toe markers</td><td>Local maximum vertical velocity of the mid-point of heel and toe markers</td></tr><tr><td>Foot markers Velocity Threshold (Ghoussayni et al. 2004)</td><td>Heel and toe</td><td>Sagittal velocity of heel marker lower than 50 cm/s</td><td>Sagittal velocity of toe marker higher than 50 cm/s</td></tr><tr><td>Foot markers acceleration extreme (Hreljac et al. 2000)</td><td>Heel and toe</td><td>Local maximum vertical acceleration of the heel marker, with the condition that the derivative of the acceleration (jerk) was zero.</td><td>Local maximum horizontal acceleration of the toe marker, also with the constraint that the relevant jerk was zero</td></tr><tr><td>Foot markers acceleration extreme (Hsue et al. 2007)</td><td>Heel and toe</td><td>Local minimal horizontal acceleration of heel marker</td><td>Local maximum horizontal acceleration of toe marker</td></tr><tr><td>Position and Velocity combined (Bonci et al. 2022)</td><td>Heel, Toe and Pelvis</td><td>Initially detected using the method from Zeni et al. (2008) and then refined based on the 3D velocity of heel marker: lower than 0.5 * walking speed for rearfoot contacts and 0.8 * walking speed for forefoot contacts</td><td>Initially detected using the method from Zeni et al. and then refined when the 3D velocity of the toe marker exceeded 0.8 times the walking speed, and further refined when the 3D velocity of the heel marker decreased after its local peak</td></tr></table>

## (b) LSTM Networks

LSTM models were developed to identify gait events using the position and velocity of pelvic, heel, and toe markers. These markers included those placed on the left anterior superior iliac spine (LASIS), left posterior superior iliac spine (LPSIS), left heel (LFCC), left second metatarsal head (LFMT2), right anterior superior iliac spine (RASIS), right posterior superior iliac spine (RPSIS), right heel (RFCC), and right second metatarsal head (RFMT2). Additionally, the heel position relative to the pelvis was used as input. The model begins with a Masking layer to ignore padded values (zeros) in the input. It includes a single LSTM layer with 128 units and a 30% dropout rate, followed by a Dense layer with 32 neurons using ReLU activation to capture high-level feature representations. Finally, a time-distributed Dense layer with a softmax activation function outputs probabilities across five nodes, corresponding to no gait event, left heel HS/TO, and right HS/TO events. Gait events were detected using a simple peak detection algorithm with a threshold of 0.01, as proposed by Lempereur et al. [16].

Hyperparameter optimization was conducted using Keras Tuner’s RandomSearch approach. The search space included key parameters such as the number of LSTM units (32–128, in steps of 32), dropout rates (0.2–0.5, in steps of 0.1), batch size (16, 32, 64, 128), and optimizer type (Adam or RMSprop). A total of 20 trials were evaluated, each testing a unique combination of hyperparameters, with validation accuracy serving as the optimization metric. An early stopping mechanism was applied to ensure efficient tuning without overfitting. After completing the search, the best-performing hyperparameters were used to build and train the final model.

![](images/43a85911990a609b3dc6695dcde6539c565366059811edd049153fc7655ddca9.jpg)

![](images/68ee84483a0cd5c49ade2bd95408be5d66b339e60283a60da684628274d93c6c.jpg)  
Fig. 2. The boxplot of prediction errors (time differences) between the measured and calculated heel strike (left) and toe off (right) events using various kinematics-based methods and the LSTM model.

## C. Evaluation Protocol

Gait event identification accuracy was evaluated by computing the temporal error (time difference) between the estimated and actual event times across both kinematicsbased and machine learning-based models. The resulting discrepancies were analyzed to assess the performance of each approach in detecting key gait events.

TABLE II  
MEAN AND STANDARD DEVIATION (STD) OF TIME DIFFERENCES FOR HEEL STRIKE AND TOE OFF EVENTS ACROSS KINEMATICS-BASED METHODS AND THE LSTM MODEL.

<table><tr><td rowspan="2">Method</td><td colspan="2">Heel Strike</td><td colspan="2">Toe off</td></tr><tr><td>Mean (ms)</td><td>STD (ms)</td><td>Mean (ms)</td><td>STD (ms)</td></tr><tr><td>Zeni et al.</td><td>4.78</td><td>9.56</td><td>-8.66</td><td>6.05</td></tr><tr><td>Desailly et al.</td><td>22.17</td><td>8.66</td><td>7.45</td><td>7.62</td></tr><tr><td>O&#x27;Connor et al.</td><td>46.73</td><td>7.72</td><td>-16.27</td><td>8.35</td></tr><tr><td>Ghoussayni et al.</td><td>50.49</td><td>11.28</td><td>-61.96</td><td>8.67</td></tr><tr><td>Hreljac et al.</td><td>32.28</td><td>31.56</td><td>4.70</td><td>8.14</td></tr><tr><td>Hsue et al.</td><td>-11.86</td><td>11.50</td><td>4.70</td><td>8.14</td></tr><tr><td>Bonci et al.</td><td>36.31</td><td>9.15</td><td>-16.63</td><td>10.34</td></tr><tr><td>LSTM</td><td>-0.72</td><td>9.56</td><td>-6.52</td><td>10.34</td></tr></table>

## III. RESULTS

Among the kinematics-based methods, the Zeni et al. approach had the smallest time difference for heel strike $( 4 . 7 8 \pm 9 . 5 6 ~ \mathrm { m s } ,$ , Table II), aligning most closely with the zero time difference line, followed by the methods of Hsue et al. and Desailly et al. (Fig. 2 and 3). In contrast, the methods of Desailly et al., O’Connor et al., Ghoussayni et al., and Bonci et al. detected heel strike later than the measured values, while the Hreljac et al. method exhibited both early and late detections. The highest variability was observed in the Hreljac et al. method (31.56 ms), indicating less consistency across trials (Fig. 2).

For toe off detection, the methods of Desailly et al., Hreljac et al., and Bonci et al. had time differences close to zero, whereas the Ghoussayni et al. method detected toe off events considerably earlier $( - 6 1 . 9 6 \pm 8 . 6 7 ~ \mathrm { m s } )$ , suggesting a systematic bias toward early detections. O’Connor et al. and Bonci et al. also showed a tendency for earlier detections, but with smaller deviations (-16.27 ms and -16.63 ms, respectively). Variability among the methods was also evident, with some methods, such as O’Connor et al., showing lower variability (8.35 ms), while others, such as Hreljac et al., exhibited wider spreads in heel strike detection.

The LSTM-based approach demonstrated low mean time differences for both heel strike $( - 0 . 7 2 \pm 9 . 5 6$ ms) and toe off $( - 6 . 5 2 \pm 1 0 . 3 4 ~ \mathrm { m s } )$ , with values centered around zero. However, its performance was comparable to that of Zeni et al., which also showed relatively small deviations in heel strike detection.

## IV. DISCUSSION

In this study, we evaluated gait event detection using seven kinematics-based computational models and one LSTM model. Our results showed that the Zeni et al. method achieved the highest accuracy among the kinematics-based approaches, while other methods exhibited systematic biases or required specific adjustments for different datasets. The LSTM model performed similarly to the Zeni et al. method, providing a data-driven alternative that avoided systematic biases. These findings hold potential implications for gait analysis research and applications such as assistive exoskeletons, where accurate identification of stance and swing phases is crucial for controlling algorithms in varied environments. By enhancing the accuracy and robustness of gait event detection, these advancements could play a vital role in supporting rehabilitation efforts, improving clinical assessments, and advancing the development of assistive technologies.

![](images/c9e98e40f2273c24ef151ce2c808b5aa54768b407f4efb9a0ad6833f07b27f7d.jpg)  
Fig. 3. The histogram of time differences between the measured and calculated heel strike (left) and toe off (right) events using various kinematicsbased methods and the LSTM model.

Our results indicated that while certain kinematics-based methods, such as Zeni et al., provided relatively accurate gait event detection, others exhibited systematic shifts or greater variability. Kinematics-based approaches are widely used for gait analysis, particularly when force plates are unavailable. Among the evaluated methods, Zeni et al. demonstrated the smallest time difference for heel strike $( 4 . 7 8 ~ \pm ~ 9 . 5 6$ ms), likely due to the alignment of its reference dataset with our study population. Velocity threshold-based methods, such as those by O’Connor et al. and Ghoussayni et al., required careful tuning of thresholds across different datasets to maintain accuracy. Acceleration-based methods, including those by Hreljac et al. and Hsue et al., exhibited good predictive accuracy for horizontal components but were less reliable for vertical components, which appeared to be influenced by individual walking patterns. Interestingly, the position- and velocity-based method by Bonci et al. did not achieve the highest accuracy in our study. This may be attributed to the dataset used in its original validation. Bonci et al. initially tested their method against existing algorithms using data from healthy young adults (n = 20) and later assessed it in a small cohort of 10 individuals across five groups: older adults, individuals with chronic obstructive pulmonary disease, multiple sclerosis, Parkinson’s disease, and proximal femur fractures. In contrast, our study, which only had normative data, demonstrated that Zeni et al.’s method achieved strong accuracy without requiring extensive dataset-specific tuning.

The LSTM model demonstrated promising results, with low mean time differences and no systematic bias. Its performance was comparable to that of Zeni et al., suggesting that deep learning-based approaches can serve as viable alternatives to traditional rule-based methods. Unlike conventional approaches that rely on predefined thresholds, the LSTM model leverages its ability to learn temporal dependencies in gait kinematics, capturing complex relationships between movement phases [27]–[29]. However, while the LSTM model performed well on unseen data within the same dataset, its generalizability to broader populations, particularly those with pathological gait patterns, remains uncertain. Deep learning models typically excel when evaluated on datasets similar to their training data, and further studies are required to assess their robustness across diverse movement patterns. Expanding the evaluation to individuals with gait impairments will provide valuable insights into the adaptability and potential clinical utility of LSTM-based gait event detection.

This study has several limitations. The evaluation was conducted solely on a normative dataset, limiting its applicability to populations with altered gait patterns. Future work should include individuals with conditions such as knee osteoarthritis or post-stroke impairments to assess the generalizability of the tested methods. Also, only one machine learning model (LSTM) was evaluated. Exploring alternative deep learning architectures, such as transformers, could provide further insights into model adaptability across different datasets. Lastly, cross-center validation with data from multiple research facilities would help determine the reproducibility and robustness of these approaches in diverse clinical and real-world settings.

## V. CONCLUSION

This study assessed the performance of seven kinematicsbased methods and one LSTM-based approach for gait event detection. While some kinematics-based methods, particularly Zeni et al., demonstrated strong accuracy, others exhibited systematic shifts or required dataset-specific tuning. The LSTM model showed promising results, performing comparably to the most accurate kinematics-based method without systematic bias. However, its applicability to individuals with gait impairments remains to be explored. Future research should extend these evaluations to clinical populations and explore additional machine learning techniques, such as transformer models, to enhance robustness and generalizability across diverse gait conditions and data collection settings.

## REFERENCES

[1] Z. Li, B. Huang, Z. Ye, M. Deng, and C. Yang, “Physical human– robot interaction of a robotic exoskeleton by admittance control,” IEEE Transactions on Industrial Electronics, vol. 65, no. 12, pp. 9614–9624, 2018.

[2] Z. Li, B. Huang, A. Ajoudani, C. Yang, C.-Y. Su, and A. Bicchi, “Asymmetric bimanual control of dual-arm exoskeletons for humancooperative manipulations,” IEEE Transactions on Robotics, vol. 34, no. 1, pp. 264–271, 2017.

[3] L. Zhang, X. Zhu, E. M. Gutierrez-Farewik, and R. Wang, “Ankle joint torque prediction using an nms solver informed-ann model and transfer learning,” IEEE Journal ofBiomedical and Health Informatics, vol. 26, no. 12, pp. 5895–5906, 2022.

[4] Z. Li, Y. Yuan, L. Luo, W. Su, K. Zhao, C. Xu, J. Huang, and M. Pi, “Hybrid brain/muscle signals powered wearable walking exoskeleton enhancing motor ability in climbing stairs activity,” IEEE Transactions on Medical Robotics and Bionics, vol. 1, no. 4, pp. 218–227, 2019.

[5] L. Luo, M. J. Foo, M. Ramanathan, J. K. Er, C. H. Chiam, L. Li, W. Y. Yau, and W. T. Ang, “Trajectory generation and control of a lower limb exoskeleton for gait assistance,” Journal of Intelligent & Robotic Systems, vol. 106, no. 3, p. 64, 2022.

[6] Y. K. Kim, R. M. Visscher, E. Viehweger, N. B. Singh, W. R. Taylor, and F. Vogl, “A deep-learning approach for automatically detecting gait-events based on foot-marker kinematics in children with cerebral palsy—which markers work best for which gait patterns?” Plos One, vol. 17, no. 10, pp. 1–16, 2022.

[7] B. Dumphart, D. Slijepcevic, M. Zeppelzauer, A. Kranzl, F. Unglaube, A. Baca, and B. Horsak, “Robust deep learning-based gait event detection across various pathologies,” Plos One, vol. 18, no. 8, pp. 1–16, 2023.

[8] J. Zeni Jr, J. Richards, and J. Higginson, “Two simple methods for determining gait events during treadmill and overground walking using kinematic data,” Gait & Posture, vol. 27, no. 4, pp. 710–714, 2008.

[9] R. Baker and H. M. Hart, Measuring walking: a handbook of clinical gait analysis. Mac Keith Press London, 2013, vol. 1.

[10] S. R. Simon, “Quantification of human motion: gait analysis—benefits and limitations to its application to clinical problems,” Journal of Biomechanics, vol. 37, no. 12, pp. 1869–1880, 2004.

[11] D. H. Sutherland, “The evolution of clinical gait analysis: Part ii kinematics,” Gait & Posture, vol. 16, no. 2, pp. 159–179, 2002.

[12] L. Zhang, Y. Liu, R. Wang, C. Smith, and E. M. Gutierrez-Farewik, “Modeling and simulation of a human knee exoskeleton’s assistive strategies and interaction,” Frontiers in Neurorobotics, vol. 15, pp. 1– 12, 2021.

[13] N. Zahradka, K. Verma, A. Behboodi, B. Bodt, H. Wright, and S. C. Lee, “An evaluation of three kinematic methods for gait event detection compared to the kinetic-based ‘gold standard’,” Sensors, vol. 20, no. 18, pp. 1–15, 2020.

[14] O. Tirosh and W. Sparrow, “Identifying heel contact and toe-off using forceplate thresholds with a range of digital-filter cutoff frequencies,” Journal of Applied Biomechanics, vol. 19, no. 2, pp. 178–184, 2003.

[15] D. A. Bruening and S. T. Ridge, “Automated event detection algorithms in pathological gait,” Gait & Posture, vol. 39, no. 1, pp. 472– 477, 2014.

[16] M. Lempereur, F. Rousseau, O. Remy-N ´ eris, C. Pons, L. Houx, ´ G. Quellec, and S. Brochard, “A new deep learning-based method for the detection of gait events in children with gait disorders: Proofof-concept and concurrent validity,” Journal of Biomechanics, vol. 98, pp. 1–4, 2020.

[17] T. Bonci, F. Salis, K. Scott, L. Alcock, C. Becker, S. Bertuletti, E. Buckley, M. Caruso, A. Cereatti, S. Del Din et al., “An algorithm for accurate marker-based gait event detection in healthy and pathological populations during complex motor tasks,” Frontiers in Bioengineering and Biotechnology, vol. 10, pp. 1–14, 2022.

[18] A. H. Hansen, D. S. Childress, and M. R. Meier, “A simple method for determination of gait events,” Journal of Biomechanics, vol. 35, no. 1, pp. 135–138, 2002.

[19] E. Desailly, Y. Daniel, P. Sardain, and P. Lacouture, “Foot contact event detection using kinematic data in cerebral palsy children and normal adults gait,” Gait & Posture, vol. 29, no. 1, pp. 76–80, 2009.

[20] S. Ghoussayni, C. Stevens, S. Durham, and D. Ewins, “Assessment and validation of a simple automated method for the detection of gait events and intervals,” Gait & Posture, vol. 20, no. 3, pp. 266–272, 2004.

[21] C. M. O’Connor, S. K. Thorpe, M. J. O’Malley, and C. L. Vaughan, “Automatic detection of gait events using kinematic data,” Gait & Posture, vol. 25, no. 3, pp. 469–474, 2007.

[22] B. Hsue, F. Miller, F. Su, J. Henley, and C. Church, “Gait timing event determination using kinematic data for the toe walking children with cerebreal palsy,” Journal of Biomechanics, vol. 40, no. 2, p. S529, 2007.

[23] A. Hreljac and R. N. Marshall, “Algorithms to determine event timing during normal walking using kinematic data,” Journal of Biomechanics, vol. 33, no. 6, pp. 783–786, 2000.

[24] M. A. French, C. Koller, and E. S. Arch, “Comparison of three kinematic gait event detection methods during overground and treadmill walking for individuals post stroke,” Journal of Biomechanics, vol. 99, pp. 1–7, 2020.

[25] C. Voisard, N. de l’Escalopier, D. Ricard, and L. Oudre, “Automatic gait events detection with inertial measurement units: healthy subjects and moderate to severe impaired patients,” Journal of NeuroEngineering and Rehabilitation, vol. 21, no. 1, pp. 1–17, 2024.

[26] B. Filtjens, A. Nieuwboer, N. D’cruz, J. Spildooren, P. Slaets, and B. Vanrumste, “A data-driven approach for detecting gait events during turning in people with parkinson’s disease and freezing of gait,” Gait & Posture, vol. 80, pp. 130–136, 2020.

[27] L. Zhang, D. Soselia, R. Wang, and E. M. Gutierrez-Farewik, “Lowerlimb joint torque prediction using lstm neural networks and transfer learning,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 30, pp. 600–609, 2022.

[28] Z. Xia, B. M. Cornish, D. Devaprakash, R. S. Barrett, D. G. Lloyd, A. H. Hams, and C. Pizzolato, “Prediction of achilles tendon force during common motor tasks from markerless video,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2024.

[29] L. Xiang, Y. Gu, Z. Gao, P. Yu, V. Shim, A. Wang, and J. Fernandez, “Integrating an lstm framework for predicting ankle joint biomechanics during gait using inertial sensors,” Computers in Biology and Medicine, vol. 170, p. 108016, 2024.