Published in final edited form as: Physiotherapy. 2015 December ; 101(4): 389–393. doi:10.1016/j.physio.2015.02.002.

# Validity and reliability of Kinect skeleton for measuring shoulder joint angles: a feasibility study

M. Huber<sup>a</sup>, A.L. Seitz<sup>b</sup>, M. Leeser<sup>c</sup>, and D. Sternad<sup>c,d,e</sup>

<sup>a</sup>Department of Bioengineering, Northeastern University, Boston, MA, USA

<sup>b</sup>Department of Physical Therapy and Human Movement Sciences, Northwestern University, Chicago, IL, USA

<sup>c</sup>Department of Electrical and Computer Engineering, Northeastern University, Boston, MA, USA

<sup>d</sup>Department of Biology, Northeastern University, Boston, MA, USA

<sup>e</sup>Department of Physics, Northeastern University, Boston, MA, USA

## Abstract

Objective—To test the reliability and validity of shoulder joint angle measurements from the Microsoft Kinect<sup>™</sup> for virtual rehabilitation.

Design—Test–retest reliability and concurrent validity, feasibility study.

Setting—Motion analysis laboratory.

Participants—A convenience sample of 10 healthy adults.

Methods—Shoulder joint angle was assessed concurrently with two trials in four static poses using: (1) the Kinect; (2) a three-dimensional (3D) motion analysis system; and (3) a clinical goniometer. All poses were captured with the Kinect from the frontal view. The two poses of shoulder flexion were also captured with the Kinect from the sagittal view.

Main outcome measures—Absolute and relative test–retest reliability of the Kinect for the measurement of shoulder angle was determined in each pose with intraclass correlation coefficients (ICCs), standard error of the measure and minimal detectable change. The 95% limits of agreement (LOA) between the Kinect and the standard methods for measuring shoulder angle were computed to determine concurrent validity.

Results—While the Kinect provided to be highly reliable (ICC 0.76 to 0.98) for measuring shoulder angle from the frontal view, the 95% LOA between the Kinect and the two measurement standards were greater than ±5° in all poses for both views.

Conclusions—Before the Kinect is used to measure movements for virtual rehabilitation applications, it is imperative to understand its limitations in precision and accuracy for the measurement of specific joint motions.

## Keywords

Microsoft Kinect; Motion analysis; Skeleton tracking; Virtual reality; Measurement accuracy; Shoulder rehabilitation

## Introduction

The use of virtual reality technology for rehabilitation, or virtual rehabilitation (VR), provides several advantages over conventional therapy, including increased capacity for quantitative measurement of motor performance, delivery of real-time performance feedback, and enhancement of patient motivation. By exploiting the latest commercial game technologies [1–4], VR systems are being developed at increasingly low costs, making them particularly useful for in-home therapy.

Much of the current research using in-home VR is aimed at patients with neurological disorders [1–5]. However, these systems also have the potential to improve physical therapy for patients with musculoskeletal disorders. With an in-home VR system, a clinician can ensure that the patient is performing exercises correctly and reaching targeted functional goals in specific postoperative timeframes that allow proper joint healing. However, despite these advantages, the use of VR for postoperative joint therapy is very limited at present.

The Kinect<sup>™</sup> (Microsoft Corp., Redmond, WA, USA), one of the more popular gaming sensors, is an ideal sensor for a VR system designed for postoperative joint rehabilitation. However, as for all gaming technology, the Kinect was not developed with the intention of clinical use. As such, the accuracy of Kinect measurements must be evaluated thoroughly for movements of interest before clinical application. It is disconcerting for a study using the Kinect for VR to claim that the validity and reliability of the sensor have been established previously, with the citation of studies that have assessed the validity and reliability of different Kinect measurements [5]. For instance, it is misleading to cite the accuracy of the depth image to assure accuracy of skeletal data from the Microsoft Kinect for Windows<sup>™</sup> Software Development Kit (SDK) [6].

With the ultimate goal of developing a VR system for postoperative shoulder therapy, this study aimed to assess the reliability and validity of skeletal data from the Kinect for Windows SDK for the measurement of precise shoulder angles. Previous work has found that skeletal data from the Kinect SDK can be used for accurate measurement of shoulder range of motion (ROM) [7]; however, its accuracy for the measurement of exact shoulder angles has not been investigated. A prior study by Fernández-Baena et al. [8], which examined accuracy of the Kinect for the measurement of shoulder ROM, observed average errors between 8 and $1 4 ^ { \circ }$ in shoulder angle trajectories. Similarly, Chang et al. [9] observed large errors in tracking the shoulder position with the Kinect. However, no formal analyses of validity and reliability were conducted in these studies. Furthermore, these studies used skeletal data from OpenNI (Primesense), which differ from skeletal data from the Kinect

SDK. Nonetheless, the findings of Fernández-Baena et al. [8] and Chang et al. [9] identify the need to assess the accuracy of measurement of exact shoulder angles using the Kinect skeletal data.

The present feasibility study measured various shoulder angles while participants held a series of static poses. These poses consisted of shoulder configurations commonly used in postoperative shoulder rehabilitation, including one pose where the shoulder was occluded from the view of the Kinect. The shoulder angle measurements from three data acquisition systems – the Kinect, a three-dimensional (3D) motion analysis system (gold standard; trakSTAR, Ascension Technology Corp., Shelburne, VT, USA) and a goniometer (clinical standard) – were compared.

## Methods

## Participants

A convenience sample of 10 asymptomatic adults with no known shoulder pathology {six females and four males, mean age 22.1 [standard deviation (SD) 0.9] years} participated in the study. All participants gave informed written consent before the study. The study protocol was approved by the Institutional Review Board of xxxxxxxxxxx.

## Procedure

With the Kinect in the frontal view, each participant held the following static poses, two repetitions each, in a random order: flexion to $9 0 ^ { \circ }$ , flexion to max, abduction to $9 0 ^ { \circ }$ , and external rotation to max at $0 ^ { \circ }$ abduction (Fig. A, see online supplementary material). Two additional repetitions of the flexion to $9 0 ^ { \circ }$ and flexion to max poses were measured with the Kinect from the sagittal view. The sagittal view poses were added as pilot work revealed that the shoulder joint was occluded from the Kinect in the frontal view during the flexion to $9 0 ^ { \circ }$ pose. Thus, the two poses of shoulder flexion were repeated from the sagittal view to determine if shoulder flexion measurements were reliable and valid from this vantage point. Half of the participants performed the shoulder motion with their dominant arm, and the other half performed the shoulder motion with their non-dominant arm; this was also randomised.

The $9 0 ^ { \circ }$ poses (flexion to $9 0 ^ { \circ }$ and abduction to $9 0 ^ { \circ } )$ were set using the goniometer. For the max poses (flexion to max and external rotation to max at $0 ^ { \circ }$ abduction), participants were instructed to rotate to their maximum capability. Once the pose was set, measurements were recorded simultaneously using a Kinect, a 3D motion analysis system and a blinded goniometer. The pose was reset for each repetition. It is feasible that there were differences between repetitions in the max values. However, this procedure was consistent with current practice. The likelihood of variation in ROM for each pose between trials was reduced by the recruitment of healthy, pain-free subjects.

## Data capture and processing

Kinect—The skeletal data captured from the Kinect for Windows SDK for each pose consisted of the 3D positions of 20 joints. The positions of shoulder and elbow joints relative to the trunk were used to measure the angles of shoulder flexion and abduction (in degrees), while the positions of the elbow and hand relative to the trunk were used to measure the angle of external rotation. Skeletal data from the Kinect for Windows SDK were accessed and analysed using MATLAB (Mathworks, Natick, MA, USA).

Goniometer—A standard 12-inch goniometer was modified so that the examiner was blinded to the measures. Goniometric measurements of shoulder joint motions were performed using standardised methods [10]. Once the goniometer was aligned to the shoulder motion by the examiner, a second examiner read and recorded the measurement (in degrees). Goniometric measures of the shoulder have demonstrated excellent reliability [11,12].

3D motion analysis—The Trakstar electromagnetic-based motion analysis system with a sampling rate of 240 Hz was used with Motion Monitor software (Innovative Sports Training, Inc., Chicago, IL, USA) to collect 3D kinematic data of the humerus and trunk. Electromagnetic receivers were secured with tape on the thorax over the spinous process of the T3 vertebrae and the posterior aspect of the distal humerus of the arm. Local coordinate axes systems for each segment were created using digitised anatomical landmarks on each segment, in accordance with the recommendations of the International Society of Biomechanics [13]. Euler angle sequences for humeral rotations were used to describe motion of the humerus relative to the thorax. Shoulder movements in the directions of abduction and flexion were described using elevation angles and external rotation with long axis rotation. All motions in flexion, abduction and external rotation were defined as positive for direct comparison with clinical goniometry measures. Root mean square accuracy of this system has been reported to be <1° [14].

## Statistical analysis

Intraclass correlation coefficient (ICC) model 3,2 [ICC(3,2)] was used to determine the relative test–retest reliability of the Kinect for the measurement of shoulder angle [15]. Six ICC(3,2) values, representing the agreement of two trials for each pose from the respective views, were computed. The ICC(3,2) values were defined as ‘poor’ when below 0.20, ‘fair for values between 0.21 and 0.40, ‘moderate’ for values between 0.41 and 0.60, ‘good’ for values between 0.61 and 0.80, and ‘very good’ for values between 0.81 and 1.0 [16]. The standard error of the measure (SEM) and the minimal detectable change (MDC) were calculated to establish absolute reliability. SEM was defined as SD multiplied by the square root of ICC subtracted from 1 [17], and MDC was calculated by multiplying SEM by the square root of 2 [18].

The 95% limits of agreement (LOA) between the Kinect and the two measurement standards for the shoulder were computed for each pose to determine validity [18]. To obtain the 95% LOA in each pose, the mean of the two shoulder angle measurements from each method was calculated. Next, the mean and SD of differences between (1) the Kinect and the goniometer and (2) the Kinect and the 3D motion analysis measurements were computed. The 95% LOA were defined as the mean difference ±1.96 SD of the difference, such that 95% of differences lay within these limits. If the 95% LOA were greater than ±5°, the discrepancies between measurement systems were considered to be clinically significant [19]. All statistical analyses were conducted using IBM SPSS Statistics software (IBM Corp., Armonk, NY, USA).

## Results

## Test–retest reliability

Results of test–retest reliability of the Kinect for the measurement of shoulder angle with ICC, SEM, and MDC values are shown in Table 1. From the frontal view, the Kinect had good to very good relative reliability for the measurement of shoulder angle in all four poses, as indicated by the high $\mathrm { I C C } ( 3 , 2 )$ values. Small SEM and MDC values, indicating good absolute reliability, were observed for all poses from the frontal view except flexion to $9 0 ^ { \circ }$ , where the shoulder joint was occluded from the Kinect camera. From the sagittal view, the Kinect had very good relative and absolute reliability in the flexion to $9 0 ^ { \circ }$ pose, but only fair relative reliability and poor absolute reliability in the flexion to max pose.

## Concurrent validity

The 95% LOA between the Kinect and the goniometer and between the Kinect and the 3D motion analysis system are shown in Table 2. Abduction to $9 0 ^ { \circ }$ was the only pose in which the Kinect measures of shoulder angle were reasonably accurate compared with either the goniometer or the 3D motion tracking system. Like the other poses, however, the 95% LOA for the discrepancy between systems exceeded $\pm 5 ^ { \circ } .$ , which was defined as clinically significant.

## Discussion

Using the skeletal data from the Kinect for Windows SDK, the Kinect was found to be highly reliability for the measurement of shoulder angle in most poses. Highest accuracy was achieved for the measurement of shoulder angle in the abduction to $9 0 ^ { \circ }$ pose. This is consistent with a previous finding that the Kinect was accurate for the measurement of ROM during shoulder abduction [7]. However, the measurement discrepancies between the Kinect and the two measurement standards were clinically significant in all poses.

From the frontal view, the Kinect had poor results for the measurement of shoulder angle in the transverse plane (external rotation to max at $0 ^ { \circ }$ abduction) and the sagittal plane (flexion to $9 0 ^ { \circ }$ and flexion to max). Bonnechère et al. [7] reported similarly poor results using the Kinect to measure ROM during elbow flexion and knee flexion, which are both movements in the sagittal plane. Inaccurate measurements in the sagittal plane can be explained, in part, by errors introduced from estimating the position of occluded joints; a problem typical of vision-based motion tracking systems. This was most evident from the poor absolute reliability and validity of the Kinect measurements in the flexion to $9 0 ^ { \circ }$ pose from the frontal view, where the shoulder joint was occluded by the arm.

In an attempt to circumvent this problem, the flexion to $9 0 ^ { \circ }$ and flexion to Max poses were also measured from the sagittal view, but this did not prove a viable solution. In the external rotation to max at $0 ^ { \circ }$ abduction and flexion to max poses, joint occlusion was not an issue, but the shoulder angles measured by the Kinect were still inaccurate. This major limitation poses a problem for the measurement of simple shoulder movements, as well as complex, full-body movements.

## Conclusions

While the skeletal data of Kinect for Windows SDK may be accurate for commercial gaming purposes, this study revealed significant concerns about the use of these data to measure shoulder motion when precise shoulder angle measurements are required. Only a limited number of shoulder configurations were examined in this study. However, the large discrepancies in shoulder angle measurements should not be taken lightly. In fact, these results identify the need for further assessment of the accuracy of the Kinect for measurement of a wider range of shoulder angles and in impaired movements.

VR for postoperative shoulder therapy is only one example of a case where accurate shoulder angle measurements are needed. In fact, accurate shoulder angle measurements are required for most musculoskeletal and neurological rehabilitation protocols [4,8,9,20]. Before the Kinect can be used to measure movements in VR, it is imperative to understand its limitations in precision and accuracy for the measurement of specific joint motions.

## Supplementary Material

Refer to Web version on PubMed Central for supplementary material.

## Acknowledgments

Funding: This study was funded by a seed grant from The Mathworks to M.E. Huber, and a Northeastern University Provost Tier 1 grant.

## References

1. Deutsch JE, Borbely M, Filler J, Huhn K, Guarrera-Bowlby P. Use of a low-cost, commercially available gaming console (Wii) for rehabilitation of an adolescent with cerebral palsy. Phys Ther. 2008; 88:1196–207. [PubMed: 18689607]

2. Morrow, K.; Docan, C.; Burdea, G.; Merians, A. Low-cost virtual rehabilitation of the hand for patients post-stroke. IEEE 2006 International Workshop on Virtual Rehabilitation; New York, NY, USA. p. 6-10.

3. Huber M, Rabin B, Docan C, Burdea GC, AbdelBaky M, Golomb MR. Feasibility of modified remotely monitored in-home gaming technology for improving hand function in adolescents with cerebral palsy. IEEE Trans Inf Technol Biomed. 2010; 14:526–34. [PubMed: 20071262]

4. Golomb MR, McDonald BC, Warden SJ, Yonkman J, Saykin AJ, Shirley B, et al. In-home virtual reality videogame telerehabilitation in adolescents with hemiplegic cerebral palsy. Arch Phys Med Rehabil. 2010; 91:1–8. e1. [PubMed: 20103390]

5. Pompeu JE, Arduini LA, Botelho AR, Fonseca MBF, Pompeu SMAA, Torriani-Pasin C, et al. Feasibility, safety and outcomes of playing Kinect Adventures!<sup>™</sup> for people with Parkinson’s disease: a pilot study. Physiotherapy. 2014; 100:162–8. [PubMed: 24703891]

6. Shotton, J.; Fitzgibbon, A.; Cook, M.; Sharp, T.; Finocchio, M.; Moore, R., et al. Real-time human pose recognition in parts from single depth images. 2011 IEEE Conference on Computer Vision and Pattern Recognition; Colorado Springs, CO, USA. 20–25 June 2011; p. 1297-304.

7. Bonnechère B, Jansen B, Salvia P, Bouzahouene H, Omelina L, Moiseev F, et al. Validity and reliability of the Kinect within functional assessment activities: comparison with standard stereophotogrammetry. Gait Posture. 2014; 39:593–8. [PubMed: 24269523]

8. Fernández-Baena, A.; Susín, A.; Lligadas, X. Biomechanical validation of upper-body and lowerbody joint movements of Kinect motion capture data for rehabilitation treatments. 4th IEEE International Conference on Intelligent Networking and Collaborative Systems; 19–21 September 2012; Bucharest, Romania. p. 656-61.

9. Chang CY, Lange B, Zhang M, Koenig S, Requejo P, Somboon N, et al. Towards pervasive physical rehabilitation using Microsoft Kinect. Pervasive Health. 2012:159–62.

10. Jain NB, Wilcox RB, Katz JN, Higgins LD. Clinical examination of the rotator cuff. PM R. 2013; 5:45–56. [PubMed: 23332909]

11. Riddle DL, Rothstein JM, Lamb RL. Goniometric reliability in a clinical setting. Shoulder measurements. Phys Ther. 1987; 67:668–73. [PubMed: 3575423]

12. Hayes K, Walton JR, Szomor ZR, Murrell GA. Reliability of five methods for assessing shoulder range of motion. Aust J Physiother. 2001; 47:289–94. [PubMed: 11722295]

13. Wu G, van der Helm FCT, Veeger HEJ, Makhsous M, Van Roy P, Anglin C, et al. ISB recommendation on definitions of joint coordinate systems of various joints for the reporting of human joint motion. Part II: Shoulder, elbow, wrist and hand. J Biomech. 2005; 38:981–92. [PubMed: 15844264]

14. Ludewig PM, Phadke V, Braman JP, Hassett DR, Cieminski CJ, LaPrade RF. Motion of the shoulder complex during multiplanar humeral elevation. J Bone Joint Surg Am. 2009; 91:378–89. [PubMed: 19181982]

15. Shrout PE, Fleiss JL. Intraclass correlations: uses in assessing rater reliability. Psychol Bull. 1979; 86:420–8. [PubMed: 18839484]

16. Landis JR, Koch GG. The measurement of observer agreement for categorical data. Biometrics. 1977; 33:159–174. [PubMed: 843571]

17. Stratford PW, Binkley JM, Riddle DL. Health status measures: strategies and analytic methods for assessing change scores. Phys Ther. 1996; 76:1109–23. [PubMed: 8863764]

18. Stratford PW, Binkley J, Solomon P, Finch E, Gill C, Moreland J. Defining the minimum level of detectable change for the Roland–Morris questionnaire. Phys Ther. 1996; 76:359–65. discussion 366–8. [PubMed: 8606899]

19. Martin Bland J, Altman D. Statistical methods for assessing agreement between two methods of clinical measurement. Lancet. 1986; 327:307–10.

20. Lange B, Chang C-Y, Suma E, Newman B, Rizzo AS, Bolas M. Development and evaluation of low cost game-based balance rehabilitation tool using the Microsoft Kinect sensor. Annu Int Conf IEEE Eng Med Biol Soc. 2011; 2011:1831–4.

ICC, intraclass correlation coefficient; SEM, standard error of the measure; MDC, minimal detectable change.  
<sub>a</sub>b<sup>le</sup>  
Test–retest reliability results of the Kinect for the measurement of shoulder angle.

<table><tr><td>Kinect view</td><td>Pose</td><td>ICC</td><td>Mean</td><td>SEM</td><td>MDC</td></tr><tr><td rowspan="4">Front</td><td>Abduction to 90°</td><td>0.76</td><td>90.1°</td><td>2.5°</td><td>3.5°</td></tr><tr><td>External rotation to max at 0° abduction</td><td>0.98</td><td>65.8°</td><td>3.7°</td><td>5.2°</td></tr><tr><td>Flexion to 90°</td><td>0.85</td><td>73.7°</td><td>12.2°</td><td>17.3°</td></tr><tr><td>Flexion to max</td><td>0.95</td><td>162.2°</td><td>4.0°</td><td>5.6°</td></tr><tr><td rowspan="2">Sagittal</td><td>Flexion to 90°</td><td>0.84</td><td>86.7°</td><td>4.4°</td><td>6.2°</td></tr><tr><td>Flexion to max</td><td>0.37</td><td>161.5°</td><td>24.2°</td><td>34.1°</td></tr></table>

<sub>a</sub>b<sup>le</sup>

angle obtained using the Kinect, a goniometer and a three-dimensional (3D) magnetic tracking system. Bland and Altman analysis and limits of agreement, calculated as ±1.96 standard deviation of the difference, between measurements of shoulder joint

<table><tr><td rowspan="2">Kinect view</td><td rowspan="2">Pose</td><td colspan="2">Kinect vs goniometer</td><td colspan="2">Kinect vs 3D magnetic tracker</td></tr><tr><td>Mean bias</td><td>95% Limits of agreement</td><td>Mean bias</td><td>95% Limits of agreement</td></tr><tr><td>Front</td><td>Abduction to 90°</td><td>-1.5°</td><td>-7.0 to 4.1°</td><td>3.0°</td><td>-10.6 to 16.7°</td></tr><tr><td></td><td>External rotation to max at 0° abduction</td><td>1.3°</td><td>-14.3 to 16.9°</td><td>16.0°</td><td>-36.4 to 68.4°</td></tr><tr><td></td><td>Flexion to 90°</td><td>-16.6°</td><td>-52.1 to 18.9°</td><td>-6.1°</td><td>-44.6 to 32.4°</td></tr><tr><td></td><td>Flexion to max</td><td>-15.9°</td><td>-36.5 to 4.7°</td><td>10.6°</td><td>-15.4 to 36.7°</td></tr><tr><td>Sagittal</td><td>Flexion to 90°</td><td>-6.9°</td><td>-17.7 to 3.8°</td><td>6.1°</td><td>-8.7 to 20.9°</td></tr><tr><td></td><td>Flexion to max</td><td>-15.0°</td><td>-44.5 to 14.5°</td><td>10.6°</td><td>-25.0 to 46.3°</td></tr></table>