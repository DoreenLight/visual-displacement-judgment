WILEY

# Accuracy evaluation of two markerless motion capture systems for measurement of upper extremities: Kinect V2 and Captiv

Tim Steinebach<sup>1</sup> | Eric H. Grosse<sup>2</sup> | Christoph H. Glock<sup>2</sup> | Jurij Wakula<sup>1</sup> | Alexander Lunin<sup>2</sup>

<sup>1</sup>Institute for Ergonomics and Human Factors, Technische Universität Darmstadt, Darmstadt, Germany

<sup>2</sup>Institute of Production and Supply Chain Management, Technische Universität Darmstadt, Darmstadt, Germany

## Correspondence

Tim Steinebach, Institute for Ergonomics and Human Factors, Technische Universität Darmstadt, Otto‐Bernd‐Straße 2, 64287 Darmstadt, Germany. Email: t.steinebach@jad.tu-darmstadt.de

## Abstract

Motion capturing is a promising method to assess working postures and human movements and, therewith, the risk of musculoskeletal injuries that could occur while performing manual tasks in industrial settings. To obtain a reliable risk assessment, the motion capture system used has to accurately measure body postures adopted by the worker during the task. This study evaluates the accuracy to measure joint angles of upper extremities of two different motion capture systems, namely the Microsoft Kinect V2 and the Captiv system, for angles of upper extremities. For this purpose, an experimental study was conducted involving 12 subjects performing preset static postures and basic movements, including elbow flexion, shoulder flexion, and shoulder abduction. In addition, to examine whether self‐occlusion or occlusion of body parts by work equipment has an impact on the accuracy of the Kinect V2, the subjects handled boxes during some of the tests. As a gold standard, a goniometer for static and an angle scale for dynamic exercises was used. The Captiv system shows high correlation coefficients (r > .93) and small mean absolute errors (<5°) for all movements except for elbow flexion. The Kinect V2 has sufficient results for joint angles captured without occlusion as well, but the accuracy significantly decreases when occlusion occurs.

K E Y W O R D S

accuracy, Captiv, Kinect V2, occlusion, upper extremities

## 1 | INTRODUCTION

Despite advances in technology and automation, manual human work is still a key factor in many industries. In the logistics sector, for example, around 80% of the warehouses in Western Europe are still mainly operated with manual labor (De Koster, Le‐duc, & Roodbergen, 2007). Manual material handling is often performed in awkward postures including stretching and bending of the back with repetitive movements. This can lead to work‐related musculoskeletal disorders (MSD), which are one of the main causes for sick‐leave and incapacity for work in the industry (BKK Gesundheitsreport, 2013). In Germany, for example, the cost of sick‐leave add up to approxi mately €133 billion per year due to the loss of labor productivity About 22.8% of these absences are due to MSDs (BAuA, Federa Institute for Occupational Safety & Health, 2018). To reduce the risk for developing MSDs, an ergonomic analysis of industrial workplaces is recommended (and in many countries required by law) to create an appropriate workplace design preventing high physical stresses or overexertions, especially for jobs characterized by frequent manual handling of loads.

Recently, motion capture technologies have been used for an ergonomic assessment of physical work loads on industrial workplaces more frequently. These systems allow the automatic recording of postures and movements during manual working tasks. The collected data can be digitized, graphically visualized and analyzed using statistical computing methods. Although marker‐based systems are considered to be the gold standard in laboratory environments (Lai et al., 2013), markerless systems are much better suited for practitioners because they can be used with less elaborate preparation and are significantly cheaper (e.g., Corazza, Mündermann, Gambaretto, Ferrigno, & Andriacchi, 2010). This study evaluates the suitability of two markerless motion capture systems for an ergonomic assessment of working postures and movements of upper extremities: the optical sensor Microsoft Kinect V2 and the body‐mounted inertial sensors of the Captiv L7000 system provided by TEA (http://teaergo.com). This study presents an important first step in understanding the applicability of these technologies for upper limb movements and provides insights into the accuracy of the Captiv system since there is little evidence on the system's reliability in the literature so far.

In the past, the Kinect system has frequently been studied for its accuracy in the assessment of postures and specific movements, usually in laboratories. In the maiority of studies. the parameters measured by the Kinect were compared with the data of a gold standard. Most often, marker‐based optical systems were used as a gold standard (e.g.. Bonnechère et al, 2014: Clark et al., 2015 Dutta, 2012; Kuster, Heinlein, Bauer, & Graf, 2016; Plantard, Muller et al., 2017). Body-mounted sensors (e.g., Huber, Seitz. Leeser, & Sternad, 2015; Romero et al., 2017), goniometers (e.g., Hawi et al., 2014; Lee et al., 2015) and orthogonal reference photographs (e.g., Matsen, Lauder, Rector, Keeling, & Cherones, 2016) have been used as gold standards for postures and motions as well. Building on these findings, the Kinect has already been used for the automation of ergonomic assessments by calculating joint angles, for example using the Ovako Working Posture Assessment (Diego‐Mas & Alcaide‐Marzal, 2014) or the Rapid Upper Limb Assessment (RULA; Manghis et al., 2017). The results of these tools calculated by the Kinect data, compared with expert ratings, are sufficient with over 70% correctly assessed RULA scores for the upper body, even under suboptimal conditions in real working environments (Plantard, Shum Le Pierres, & Multon, 2017). Moreover, some researchers used raw data from multiple Kinect systems to capture and consolidate movements from different perspectives. Most of them focused on the analysis of human gait (Dolatabadi, Taati, & Mihailidis. 2016: Geerse Coolen, & Roerdink, 2015: Müller, Ilg, Giese, & Ludolph, 2017), Chen Lee, and Lin (2015), for example, developed an approach with multiple Kinect V1 sensors to evaluate the accuracy in range‐of‐motion experiments of the upper extremities. However, researchers agree that the newer generation (Kinect V2) is more accurate than the Kinect V1 for motion capture of upper extremities (Mishra, Skubic, & Abbott, 2015: Wang, Kurillo, Ofli, & Baicsy, 2015). Therefore, it i recommended to use the current Kinect V2 model for further studies. One major disadvantage of the Kinect V2 that has been mentioned i the literature is the capturing of work processes in environments where occlusion occurs. Occlusion, in this context, could either be caused by an object (e.g., a box) handled by the subject or by the subject itself, since the field of view of the camera is blocked. This may reduce the accuracy of the optical system (i.e., the Kinect V2). Especially, this is important for practical cases in real work settings, as such occlusion situations often occur (Dzeng, Hsueh, & Ho, 2017; Kuster et al., 2016; Müller et al., 2017). Although Plantard, Shum et al. (2017) indicated that the accuracy of the Kinect V2 is sufficient to calculate RULA values even in situations where occlusion occurs, an accuracy evaluation of joint angle measurement in authentic working environments using multiple Kinect V2 sensors is still lacking. To the best of the authors’ knowledge, prior research has so far not used multiple Kinect V2 systems to capture human motions from different perspectives to reduce the influence of occlusion of upper extremities and to improve the accuracy of the system in real working tasks.

In contrast to the Kinect V1 and V2, the Captiv system has not yet been analyzed for its accuracy at all. So far, this system has only been used to investigate other topics and thus some degree of system accuracy has been assumed. Bartnicka, Ziętkiewicz, and Kowalski (2015), for example, used Captiv to capture body postures of surgeons during surgeries. Furthermore, Bartnicka et al. (2017) examined the motions of orthopedists and medical staff to improve ergonomic working conditions. Taber, Sweeney, Bishop, and Boute (2017) used Captiv to measure the movements of limb, torso, and head to jettison a helicopter push‐out window. In another study, Vignais, Bernard, Touvenot, and Sagot (2017) established a continuous RULA computation by supplying a biomechanical model with motion data from Captiv. Furthermore, Captiv is based on inertial sensors, and hence occlusion does not influence the accuracy of the system.

The aim of this study is to quantitatively evaluate and compare the accuracy of a motion capture system consisting of two Kinect V2 sensors and the Captiv system using a subject study including a defined working task (the lifting of boxes). Here, the measured joint angles of the upper extremities are compared with a gold standard for static postures (a goniometer, e.g., Hawi et al., 2014) and movements (an angle scale). The following research questions are addressed in the work at hand.

(1) How accurate are the Captiv system and a Kinect V2 system with two sensors in capturing basic postures and movements, both with respect to a gold standard and to one another?

(2) How is the accuracy of the two‐sensor Kinect V2 system influenced by occlusion of upper extremities?

## 2 | MATERIALS AND METHODS

## 2.1 | Analyzed systems

The Kinect V2 is the second generation of an inexpensive, depthsensing device provided by Microsoft and originally developed for interaction with the Xbox (see Figure 1 left). The time‐of‐flight technology included in the Kinect V2 allows capturing 30 frames per second with a depth resolution of 512 × 426 pixels. An algorithm is able to detect a total of 25 joints of the analyzed subject. In this study, the software provided by iPi Soft (http://ipisoft.com/) was used for the data integration of both Kinect sensors and the transformation of the raw data. This program was analyzed and rated more accurate than similar software in earlier studies (e.g., Choppin, Lane, & Wheat, 2014; Choppin & Wheat, 2012). Moreover, other researchers frequently based their findings on calculations obtained with iPi Soft (e.g., Cha et al., 2015; Fornaser, Tomasin, De Cecco, Tavernini, & Zanetti, 2017; Skals, Rasmussen, Bendtsen, Yang, & Andersen, 2017; Zhou, Li, & Bai, 2017).

![](images/2d057974a5989f53ee5f257786749f2f7212c7fabbe876354d73c3143133ed36.jpg)  
F I G U R E 1 Kinect V2 (left), Captiv sensor (middle), and Captiv recording device (right)

The second motion capture system used in this study is the Captiv L7000 system provided by the TEA group (see Figure 1). This system uses inertial sensors with integrated magnetometers, accelerometers, and gyroscopes for motion detection. The sensors are fixed on the subject with the help of straps, where a correct attachment of the sensors according to the recommendations of the manufacturer is mandatory for an accurate measurement. A total of 15 sensors are placed on the right and left upper and lower arm, hand (fixed by gloves), thigh and lower leg, foot and centrally on the forehead, back and hip. It is also possible to use fewer sensors if only certain joints are examined—only the sensor of the back is to be applied with each measurement. The battery‐powered sensors are wireless and communicate with a recording device worn on the subject's body. The sampling rate is either 32, 64, or 128 Hz. With the help of the corresponding software provided by TEA, the data from the accelerometers, gyroscopes, magnetometers are transferred into quaternions and then combined by an algorithm into joint angles.

These two systems were chosen for the purpose of our study because, on the one hand, Kinect is a quite inexpensive and widely available system that has often been evaluated in the literature as having sufficient performance. Therefore, it could be especially suitable for companies that do not have the financial capacity for high‐priced systems. In addition, unlike many alternative motion capture techniques, the operator is unaffected by markers or cables in his/her work. Captiv, on the other hand, is evaluated because it uses a different approach to capture motion than the Kinect and since it has not been validated yet. In addition, compared with optical motion capture systems, inertial measurement unit‐based systems are more suitable for field studies due to their quick setup, which makes the Captiv system attractive for practical analyses.

Both systems are thus well‐established complete solutions consisting of hardware and software. Our focus is not on improving the algorithms used by the systems. Instead, the whole system is understood as an integrated measurement tool whose output we analyze. By comparing both systems, we can evaluate two fundamentally different motion capture technologies to provide recommendations for action regarding which system is suited for capturing postures and motion for industrial working tasks.

## 2.2 | Subjects

Twelve young and apparently healthy individuals (age: 23.8 ± 2.6 years, height: 177.3 ± 9.4 cm, weight: 70.9 ± 12.3 kg, seven males and five females) participated in the study, which is comparable to similar studies (e.g., Huber et al.. 2015: Plantard. Muller et al.. 2017: Plantard, Shum et al., 2017). All subjects read an ethics statement and signed an informed written consent before the experiments. The subjects wore tight‐fitting clothes that allowed an appropriate placement of inertial sensors.

## 2.3 | Experimental procedure

The experiment was set up by placing two Kinect V2 sensors 2.5 and 2.7 m, respectively, in front of the subject and 1.5 m above the ground; see Figure 2. One of the sensors was facing the subject, as this has often been described as an optimal sensor setup (e.g., Xu, Robertson, Chen, Lin, & McGorry, 2017). In preliminary tests, the second sensor was placed at an angle of 80° to the first sensor (iPi Soft, 2019). This setup was also used in previous studies (e.g. Skals et al., 2017; Stone & Skubic, 2011) and found to be particularly robust before our tests, even for complex movements, and therefore adopted for the experiments. For all experiments in this study, attention was paid to the best possible exclusion of all potential influences, such as ferromagnetic metals, electromagnetic radiation, or inaccurate alignment of the sensors.

To assess the accuracy of the two systems, static postures and basic movements (with and without holding a box in both hands) of 12 volunteers were recorded. The focus was on the joint angles of upper extremities: (a) shoulder abduction/adduction, (b) shoulder flexion/extension as well as (c) flexion/extension of the elbow, since shoulder and elbow are commonly associated with work‐related MSDs within the upper limbs (Aptel, Aublet‐Cuvelier, & Cnockaert, 2002; Buckle & Devereux, 2002). During the experiments, the joint angles were measured simultaneously with both systems, and the raw data of both systems were synchronized temporally using a light signal. The procedure was divided into three phases as described in Table 1.

![](images/d96b9a6174c3c4ed630de6407ee69344e464623462d2c4fe8bd8c7062f5feaaf.jpg)  
F I GU R E 2 Experimental setup

Phase 1 includes four preset, static postures related to upper extremities (see Figure 3). Subsequently, joint angles were measured on the subject using the motion capture systems and a goniometer. An optical marker‐based system that has often been used as a gold standard could not be employed because of resource limitations; a goniometer, which is a standard tool for determining joint angles, was used instead (Lee et al., 2015; Milanese et al., 2014). It has high validity (Rothstein, Miller, & Roettger, 1983) as well as intra‐ and interpersonal reliability (Brosseau et al., 2001). Thus, goniometers have already been used successfully as a gold standard in similar studies for simple movements, for example, Lee et al. (2015) and Hawi et al. (2014). The chosen landmarks for flexion/extension of the elbow, for example, are the lateral epicondyle, the tip of the acromion, and midline of the wrist (Chapleau, Canet, Petit, Laflamme, & Rouleau, 2011).

In Phases 2 and 3, the accuracy of both systems was evaluated for basic movements of the upper extremities. Therefore, abduction and flexion of the shoulder, as well as a flexion of the elbow, were carried out (right and left side). The subjects stood straight, without moving the legs or bending the back during the tests. The movements in Phase 2 were performed without the handling of objects. In Phase 3, a box (dimensions: 35 cm × 25 cm × 25 cm) was held using both hands during the movements. In this case, the view of one Kinect sensor on one of the arms was occluded. Hence, it was possible to determine the impact of box handling on the angle measurement accuracy of the Kinect V2.

An angle scale (dimensions 2 m × 1 m) was installed as a gold standard for the movements, because the angles cannot be recorded continuously with the goniometer. The upper extremities were rotated around the origin of the angle scale to allow accurate reading of angles. For this purpose, the scale was mounted on the ceiling with ropes to permit adjusting its height. The origin of the angle scale has to be exactly at the height of the examined joint. A video camera was used to determine the angles from the scale for the evaluation of the movements. The camera was aligned at this height with the help of a laser to prevent perspective distortion. A similar procedure was utilized by Matsen et al. (2016), who measured the actual joint angles on photographs.

## 2.4 | Raw data processing

First, the motion data of both systems were exported. The Captiv L7000 software provides a function to automatically calculate join angles (abduction/flexion of the shoulder and flexion of the elbow), which could directly be used for evaluation. The frequency of the dataset was down‐sampled from 32 to 30 Hz to be in accordance with the Kinect data.

In the iPi Soft software for the Kinect V2, in contrast, only the Cartesian coordinates of the joints can be exported. To calculate the joint angles φ from the coordinates, vectors $\vec { v } _ { \mathrm { i } }$ and anatomical planes were computed by linear algebra. The mathematical models were based on those of Lee et al. (2015) and Diego‐Mas and Alcaide‐Marzal (2014). The angle of the elbow, for example, was calculated using the vectors of the upper arm and forearm.

$$
\varphi_ {\mathrm{elbow}} = \cos^ {- 1} \left(\frac {\langle \overrightarrow {v _ {\mathrm{upperarm}}} \cdot \overrightarrow {v _ {\mathrm{forearm}}} \rangle}{| \overrightarrow {v _ {\mathrm{upperarm}}} | \cdot | \overrightarrow {v _ {\mathrm{forearm}}} |}\right).\tag{1}
$$

Anatomical planes were determined to calculate shoulder angles. For the frontal plane, the connecting vector of the right and left shoulder and the trunk vector, which consists of the connection between the spine base and neck, were selected. Based on this, the sagittal plane was defined. The first support vector was also described via the trunk vector. The second support vector was defined as the normal vector n⃗ of the frontal plane (Diego‐Mas &

T A B L E 1 Overview of the individual phases of the study

<table><tr><td>Phase</td><td>Activity</td><td>Repetition</td><td>Gold standard</td></tr><tr><td>1</td><td>Four different static postures related to upper extremities</td><td>Once</td><td>Goniometer</td></tr><tr><td>2</td><td>Shoulder abduction, adduction, and flexion of the elbow up to 90° without box</td><td>Each three times</td><td>Angle scale</td></tr><tr><td>3</td><td>Shoulder abduction, adduction, and flexion of the elbow up to 90° with box</td><td>Each three times</td><td>Angle scale</td></tr></table>

![](images/8337f57106c5a089a24477090371ffc3a7c4ca3d4e6aac65641d89ef6f0d8200.jpg)

![](images/8c9f68aa8cfe5fb41442e51f9227e08dde4799e5fcc3e84d4baac8b51505cf40.jpg)

![](images/06280be72e44dc3757776fbde878c4a0d5851d37c05ab7aaacd5ee34b1e5c4e2.jpg)

![](images/149574e9cfbd147cfe8faa5def51e2b98355c38027575bbb24df89bd0e2c022c.jpg)  
F I G U R E 3 Static postures in Phase 1 (the following joint angles were measured: (a) abduction shoulder right and left; (b) abduction shoulde and flexion elbow right and left; (c) flexion shoulder right and left; (d) flexion shoulder and flexion elbow right and left)

Alcaide‐Marzal, 2014). The following approaches were used to determine the abduction or flexion angle of the shoulder.

(1) The angle between the projected vector of the upper arm on the frontal plane and the sagittal plane is consistent with the abduction angle of the shoulder.

(2) The angle between the projected vector of the upper arm on the sagittal plane and the frontal plane is consistent with the flexion angle of the shoulder (Lee et al., 2015).

For automation of the described procedure, a visual basic for applications macro was developed. Once the angles measured by the two motion capture systems had been calculated, synchronized and collected in a dataset, the corresponding values of the gold standard were manually added. For this purpose, the values measured with the goniometer in Phase 1 were used. For Phases 2 and 3, time‐synchronous measuring points per movement and test person were read off the angle scale and added to the dataset.

## 2.5 | Statistical analysis

To determine the deviation of the two motion capture systems from the gold standard, the mean absolute error (MAE) was calculated. Assuming that the gold standard represents the true value, the MAE indicates the measurement error of the systems. This type of error calculation has frequently been used for evaluating the Kinect V2 sensor in the literature (e.g., Cippitelli et al., 2015; Eltoukhy, Kuenze, Oh, Wooten, & Signorile, 2017; Nixon, Howard, & Chen, 2013; Schmitz et al., 2015).

The MAE was calculated as follows in this study, where n is the number of measuring values.

$$
\text { MAE } = \frac {\sum_ {i = 1} ^ {n} \left| \varphi_ {\text { KinectV2 / Captiv }} - \varphi_ {\text { gold   standard }} \right|}{n}.\tag{2}
$$

We followed the recommendation of Willmott and Matsuura (2005) by choosing the MAE instead of the root mean square error (RMSE) as a measure of deviation, because RMSE gives a relatively high weight to large errors and is therefore hard to interpret. However, the RMSE is also reported below to facilitate comparing our results to those of earlier studies that relied on the RMSE. To analyze the differences between the two systems, the Wilcoxon signed‐rank tests, was performed (α = .05) because the corresponding data were not normally distributed.

In addition, correlation coefficients between the two systems and the gold standard were calculated by comparing the maximum angle, minimum angle and the midpoint angle (angle that results at 0.5 × t, with $t = t _ { \mathrm { m a x a n g l e } } - t _ { \mathrm { m i n a n g l e } } )$ of each movement repetition. It should be noted that a high correlation does not necessarily imply a high quantitative agreement of the measured values (Dolatabadi et al., 2016). Two different methods were applied, depending on whether or not the dataset was normally distributed, namely Pearson's and Spearman's correlation coefficients. All necessary calculations were performed using the IBM SPSS. Bland–Altman plots were generated for illustration purposes. They investigate the presence of absolute agreement between the measures of the two systems and the gold standard (Bland & Altman, 1986). The average values of the gold standard and Kinect or Captiv x x<sub>gold</sub> <sub>standard</sub> − <sub>Kinect Captiv/</sub> measurement are plotted on the x‐axi 2 and the differences between the two measured values $x _ { \mathrm { K i n e c t / C a p t i v } } - x _ { \mathrm { g o l d s t a n d a r d } }$ are plotted on the y‐axis.

## 3 | RESULTS

This section presents the descriptive results of the subject study concerning the accuracy of both systems for each phase.

## 3.1 | Results for Phase 1

The MAEs of the systems’ measurements compared with the goniometer are between 2.5° and 12.9° for the Kinect V2 and between 1.9° and 14.9° for the Captiv for static postures (see Table 2). Especially for the elbow, the Captiv system shows high errors, while the angles of the shoulder, were captured very accurately in all postures with deviations <4°.

In general, the correlation coefficients for both systems are very high and significant, except for the Kinect's measurements of abduction (Posture 1) and flexion (Posture 3) or the angle of the elbow in Captiv, respectively.

## 3.2 | Results for Phases 2 and 3

After manually evaluating the joint angles during movements using the angle scale, the MAE, the RMSE and the correlation coefficients were calculated again. For Phase 2 (without box) and Phase 3 (with box), these values are summarized in Tables 3 and 4, respectively.

The measurement errors during the abduction of the shoulder are relatively low for the Kinect V2 sensor as well as for the Captiv system with ${ \mathsf { M A E } } < 3 . 9 ^ { \circ }$ in Phase 2. Adding the box in Phase 3 results in significantly higher values for this movement for the Kinect V2 in contrast to Phase 2 (the Wilcoxon signed‐rank test; left: $p = . 0 1 )$ . Overall, however, the mean errors of both systems remain below 5.0° for shoulder abduction. Consequently, the correlations with r > .89 in the experiments with and without a box are very high and statistically significant. Looking at the abduction in Phase 2, the correlation with coefficients r > .97 is almost perfect for both motion capture systems.

The flexion of the shoulder results in a notably higher MAE for the Kinect V2 than for the Captiv L7000. Errors of the Captiv system remain below 4.2°. The Kinect V2 has higher deviations, in particular for the left arm that shows larger mean absolute errors during flexion than the right arm in Phase $2 \ ( \mathsf { M A E } _ { \mathsf { r i g h t } } = 9 . 5 ^ { \circ } , \mathsf { M A E } _ { \mathsf { l e f t } } = 2 0 . 9 ^ { \circ }$ , the Wilcoxon signed‐rank test; p = .068) and Phase 3 $\mathsf { \Gamma } ( \mathsf { M A E } _ { \mathrm { r i g h t } } = 1 0 . 2 ^ { \circ }$ $\mathsf { M A E } _ { \mathsf { l e f t } } = 2 6 . 7 ^ { \circ }$ , the Wilcoxon signed‐rank test; p < .001). A discussion on this follows in Section 4.

For flexion of the elbow, the Kinect system also has higher MAEs than Captiv, especially for the detection of the left arm. This is also reflected in the correlations, which are lower for Kinect (r < .72) than for Captiv (r > .93).

Finally, it can be observed that the measurement deviations of the Kinect system in Phase 3 compared to those of Phase 2 for the flexion of the shoulders and the elbows are significantly higher. The MAE of the left elbow, for example, increases from a deviation of 22.5° in Phase 2 to a very high error of 40.2° in Phase 3 (the Wilcoxon signed‐rank test; p < .001). The differences in the Captiv measurements between those two phases, however, are much smaller or negligible.

On the basis of the calculated data, Bland–Altman plots are generated, as shown in Figures 4–6.

## 4 | DISCUSSION

## 4.1 | Static postures

As can be seen in Table 2, especially the static abduction of the shoulders was captured with very small mean absolute errors by both systems. This applies to both smaller and larger abduction angles,

T A B L E 2 MAE, RMSE, and correlation coefficients—Phase 1

<table><tr><td rowspan="2" colspan="3">Phase 1</td><td colspan="2">MAE</td><td colspan="2">RMSE</td><td colspan="2">Correlation</td></tr><tr><td>Kinect V2</td><td>Captiv</td><td>Kinect V2</td><td>Captiv</td><td>Kinect V2</td><td>Captiv</td></tr><tr><td rowspan="4">Abduction/adduction shoulder</td><td rowspan="2">Posture 1</td><td>Right</td><td>6.5°</td><td>3.3°</td><td>10.0°</td><td>4.5°</td><td>0.146</td><td>0.984**</td></tr><tr><td>Left</td><td>4.9°</td><td>2.8°</td><td>8.2°</td><td>3.4°</td><td>0.120</td><td>0.947**</td></tr><tr><td rowspan="2">Posture 2</td><td>Right</td><td>2.5°</td><td>3.1°</td><td>3.2°</td><td>4.9°</td><td>0.815**</td><td>0.660*</td></tr><tr><td>Left</td><td>2.9°</td><td>2.1°</td><td>4.7°</td><td>2.7°</td><td>0.658*</td><td>0.945**</td></tr><tr><td rowspan="4">Flexion/extension shoulder</td><td rowspan="2">Posture 3</td><td>Right</td><td>12.9°</td><td>2.1°</td><td>15.0°</td><td>2.7°</td><td>0.300</td><td>0.957**</td></tr><tr><td>Left</td><td>7.9°</td><td>2.1°</td><td>10.0°</td><td>2.6°</td><td>0.554</td><td>0.967**</td></tr><tr><td rowspan="2">Posture 4</td><td>Right</td><td>10.9°</td><td>1.8°</td><td>13.3°</td><td>2.5°</td><td>0.848**</td><td>0.989**</td></tr><tr><td>Left</td><td>8.8°</td><td>1.9°</td><td>11.1°</td><td>2.5°</td><td>0.854**</td><td>0.989**</td></tr><tr><td rowspan="4">Flexion/extension elbow</td><td rowspan="2">Posture 2</td><td>Right</td><td>7.0°</td><td>13.3°</td><td>8.5°</td><td>15.1°</td><td>0.861**</td><td>0.737**</td></tr><tr><td>Left</td><td>4.4°</td><td>12.1°</td><td>7.0°</td><td>14.0°</td><td>0.751**</td><td>0.494</td></tr><tr><td rowspan="2">Posture 4</td><td>Right</td><td>4.4°</td><td>13.8°</td><td>5.6°</td><td>16.8°</td><td>0.923** (S)</td><td>0.650* (S)</td></tr><tr><td>Left</td><td>4.6°</td><td>14.9°</td><td>5.0°</td><td>15.9°</td><td>0.979**</td><td>-0.343</td></tr></table>

\*\*p < .01.  
Note: (S): Spearman's correlation coefficient, otherwise Pearson's correlation coefficient.  
\*p < .05.  
Abbreviations: MAE, mean absolute error; RMSE, root mean square error.

T A B L E 3 MAE, RMSE, and correlation coefficients without box—Phase 2

<table><tr><td rowspan="2" colspan="2">Phase 2</td><td colspan="2">MAE</td><td colspan="2">RMSE</td><td colspan="2">Correlation</td></tr><tr><td>Kinect V2</td><td>Captiv</td><td>Kinect V2</td><td>Captiv</td><td>Kinect V2</td><td>Captiv</td></tr><tr><td rowspan="2">Abduction/adduction shoulder</td><td>Right</td><td>3.9°</td><td>2.6°</td><td>5.1°</td><td>3.1°</td><td>0.978**</td><td>0.992**</td></tr><tr><td>Left</td><td>2.8°</td><td>2.8°</td><td>4.1°</td><td>4.0°</td><td>0.972** (S)</td><td>0.977** (S)</td></tr><tr><td rowspan="2">Flexion/extension shoulder</td><td>Right</td><td>9.5°</td><td>4.1°</td><td>10.9°</td><td>5.0°</td><td>0.972**</td><td>0.991**</td></tr><tr><td>Left</td><td>20.9°</td><td>3.0°</td><td>31.6°</td><td>3.9°</td><td>0.494**</td><td>0.990**</td></tr><tr><td rowspan="2">Flexion/extension elbow</td><td>Right</td><td>14.8°</td><td>6.9°</td><td>32.8°</td><td>8.9°</td><td>0.645**</td><td>0.979**</td></tr><tr><td>Left</td><td>22.5°</td><td>7.2°</td><td>40.7°</td><td>8.9°</td><td>0.519** (S)</td><td>0.975**</td></tr></table>

Note: (S): Spearman's correlation coefficient, otherwise Pearson's correlation coefficient.  
Abbreviations: MAE, mean absolute error; RMSE, root mean square error.

$$
^ {* *} p <  . 0 1.
$$

with the errors of Captiv being slightly smaller at ≤3.3°. These values are consistent with the studies of Chen et al. (2015), who obtained a mean error of 4.4° in abduction using two Kinect sensors. It should be noted that the arms of one subject in Posture 1 were not correctly recognized by the Kinect V2. For this reason, the errors in this exercise are higher than for Posture 2, which also has an impact on the correlation coefficients.

Looking at the corresponding Bland–Altman plots (Figure 4), the mean difference is close to zero for Captiv in many cases. Furthermore, the two dashed lines (95% limit of agreement) are closer to each other for the Captiv measurement. The narrower the limit of the agreement is, the more practical is the use of this measuring method (Dolatabadi et al., 2016).

For the flexion of the shoulders. however, we observed a significant accuracy difference (the Wilcoxon signed‐rank test) between the two systems. While Captiv delivered very accurate values with $M A E \leq 2 . 1 ^ { \circ } ,$ average errors between 7.9° and 12.9° for the individual postures were found for the Kinect V2. Fernandez‐Baena, Susin, and Lligadas (2012) observed similar mean errors for the shoulder flexion (7°‐10°) measured with Kinect V1. Huber et al. (2015), on the other hand, report mean deviations >15° for the flexion when detected with a V1 model placed in front of the subject. Especially the right side, which can only be captured by one of the two Kinect sensors because it is occluded by the body of the subject, shows very high errors. The Bland–Altman plots (Figure 5) show that the mean difference between the Kinect and the goniometer is negative. Thus, the Kinect V2 underestimates the flexion angle of the shoulder.

Contrary results were found for the flexion of the elbow in static postures. The mean absolute errors of the Kinect measurement were below 7° for both sides and in both postures. The errors of the Captiv system compared with the goniometer measurements, however were significantly higher with errors ≥12°.

This error may have been caused by the initialization (manua alignment of the sensors to minimize the impact of environmental influences) of the Captiv system, which only had a “moderate” quality according to the software, although the initialization was carried out exactly as recommended by the manufacturer. A moderate initialization quality may not be sufficient for the highest accuracy requirements and has to be taken into account when interpreting the results.

In sum, the Kinect V2 is able to deliver results that are almost as accurate as those obtained by Captiv L7000 in very simple, static situations without occlusion of body parts

## 4.2 | Basic movements

This subsection evaluates dynamic experiments. The following type of occlusion are of particular interest.

T A B L E 4 MAE, RMSE, and correlation coefficients with box—Phase 3

<table><tr><td rowspan="2" colspan="2">Phase 3</td><td colspan="2">MAE</td><td colspan="2">RMSE</td><td colspan="2">Correlation</td></tr><tr><td>Kinect V2</td><td>Captiv</td><td>Kinect V2</td><td>Captiv</td><td>Kinect V2</td><td>Captiv</td></tr><tr><td rowspan="2">Abduction/adduction shoulder</td><td>Right</td><td>5.0°</td><td>2.0°</td><td>6.5°</td><td>2.5°</td><td>0.891** (S)</td><td>0.996**</td></tr><tr><td>Left</td><td>4.3°</td><td>2.8°</td><td>5.2°</td><td>3.6°</td><td>0.977**</td><td>0.993**</td></tr><tr><td rowspan="2">Flexion/extension shoulder</td><td>Right</td><td>10.2°</td><td>4.2°</td><td>11.0°</td><td>5.2°</td><td>0.926** (S)</td><td>0.940** (S)</td></tr><tr><td>Left</td><td>26.7°</td><td>3.4°</td><td>35.7°</td><td>4.4°</td><td>0.161 (S)</td><td>0.931** (S)</td></tr><tr><td rowspan="2">Flexion/extension elbow</td><td>Right</td><td>13.8°</td><td>10.7°</td><td>20.5°</td><td>14.1°</td><td>0.721**</td><td>0.939**</td></tr><tr><td>Left</td><td>40.2°</td><td>9.4°</td><td>49.1°</td><td>12.5°</td><td>0.013 (S)</td><td>0.952**</td></tr></table>

Note: (S): Spearman's correlation coefficient, otherwise Pearson's correlation coefficient.  
Abbreviations: MAE, mean absolute error; RMSE, root mean square error.

$$
^ {* *} p <  . 0 1.
$$

![](images/ef46dd22777183e3b92fc95d66fc265810cd0b73b85a2633ba7e6d921b25e179.jpg)  
F I GU R E 4 Bland–Altman plot—Phase 1: Shoulder abduction

(1) Self‐occlusion (occlusion of the left arm by the right arm).

(2) Occlusion by tools/equipment (body parts covered by the box held between the hands).

As mentioned in the previous section, the mean absolute deviations of both systems for the abduction of the shoulders in Phase 2 with MAE < 4.4° are very low and the correlation coefficients with r > .9 are high. For these movements without occlusion, both systems can be recommended for use.

In Phase 3, MAEs were higher for the Kinect system. A statistical comparison of the averages shows that the mean absolute errors of abduction in the left shoulder were significantly higher in Phase 3 than in Phase 2 (the Wilcoxon signed‐rank test; p = .01). This could be due to the fact that the view of the laterally placed sensor during the abduction or flexion is occluded by the box. Although the camera facing the subject should allow a good detection of the movement, the errors increased when the box is held by a subject. The second Kinect sensor (with the software iPi Soft) seems not to be able to fully compensate the other sensor's occluded view.

With respect to the capturing of the shoulder flexion, Captiv has a maximum MAE of 4.2° in Phases 2 and 3. The significant correlation coefficients of r > .93 indicate almost perfect accordance to the gold standard. In contrast, the errors of the Kinect system are, similar to the static postures, significantly higher.

To gain insights into the effects of self‐occlusion on the results obtained by the Kinect sensors, the errors of the right and left joints of the subjects are compared. The left shoulder has an MAE of 20.9° in Phase 2, while the right arm only has an MAE of 9.5°. Nevertheless, these differences are barely significant for Phase 2 with p = .068 (p < .001 in Phase 3). This effect can be explained by the fact that during the simultaneous shoulder flexion of both sides, the left arm is occluded by the right one. The second Kinect sensor, which can capture both arms uncovered due to the different positioning, is not able to compensate the occlusion of the first sensor—neither in the phase with nor without box.

Bland–Altman plots in Figure 6 graphically illustrate these errors. The dotted limit‐of‐agreement lines in the right plots (no self‐occlusion) are much closer to each other than in the left plots. Self‐occlusion also becomes evident from the high number of measurements of the left shoulder located in the negative range of the differences (y‐axis) of the left diagrams. These high negative differences occurred when the Kinect was unable to detect the joints of the occluded arm. The corresponding iPi Soft visualizations show that, in these cases, the arm points down parallel to the upper body, even though the arm is actually flexed at 90°. In addition, the influence of the occlusion of the box can also be recognized for shoulder flexion. The MAEs in the third phase is again higher than in the second phase.

Kinect V2 mean of differences   +1,96 SD … - 1,96 SD

![](images/f148c32fdd0ac74135bf1fd5fa843613da4615fe16ed6f504b328a63f5dd4e24.jpg)

![](images/5ddbf9f20b0d56e3b602427fb767ac46f82a9b422ee40ab9b3014074fe6afe88.jpg)  
F I GU R E 5 Bland–Altman plot—Phase 1: Shoulder flexion

The detection of joint angles during elbow flexion is not completely satisfactory for both systems. The problem of the initialization of Captiv, which has already been described in the static phase, is also recognizable during movements. Mean absolute errors between 6.9° and 10.7° could be insufficient for specific clinical studies or ergonomic analyses. Whether a system can still be used in investigations depends ultimately on the context of the analysis and on whether or not the conclusion of an analysis changes due to the measurement error (Myles & Cui, 2007). The detection of the joint angles of the elbow by the Kinect V2 sensor has even higher errors than the measurement with Captiv. In contrast to the relatively accurate results of the static motion capture of the elbow flexion, the detection of dynamic movements by the Kinect sensor failed for some subjects. Again, the self‐occlusion of body parts seems to be problematic. The mean absolute error of the left elbow in Phase 2 is higher (MAE = 22.5°) than the one obtained for the right elbow (MAE = 14.8°).

In addition to the mathematical error calculations presented in this section, qualitative observations were made repeatedly with Kinect. A systematic error occurred in nearly half of all movements where a box was handled. The iPi Soft visualization shows that Kinect falsely identifies the box that was held by the subjects between the hands as a leg of the subject (see Figure 7). So, Kinect may incorrectly identify a work item as a body part and thus record erroneous movements. The error pattern described here had only a marginal influence on the joint angle of the upper body, as only one of the legs was affected. Nevertheless, this misinterpretation of the system makes examinations difficult or impossible in industrial practice, especially if the angles of the lower body are of relevance.

![](images/92b25986616b99714a4a9e7da1930e2d1735d4252e60abbaf1d02f574a8c73d4.jpg)  
Cptiv mean ofdifferences   +1,96 SD  - 1,96SD

![](images/d840280898a6ceb33ee269b97d6f9d2b458f07499271aea4a7ecbeb063c6a6c1.jpg)

The results of this study are subject to limitations. First, it has to be taken into account that the two motion capture systems were only evaluated in this study with respect to three different joint angles. Furthermore, the Kinect system can face further challenges in a real work environment. Daylight, moving objects in the background, suboptimal sensor placement and larger areas of the subject's movements may further reduce the accuracy of the motion capturing. In addition, the experiments were carried out only with relatively young participants (20–27 years), which does not necessarily corre spond to the average age of persons performing manual operations for example, in logistics.

## 5 | CONCLUSION

The aim of this study was to quantitatively compare the accuracy of the Kinect V2 and the Captiv L7000 motion capture systems in a subject experiment study. The joint angles of the upper extremitie measured by both systems were compared with a reference system for static postures and movements. A central objective of thi experimental study was to investigate if a second Kinect V2 sensor, which detects the motions from another angle, is able to compensate the problem of occlusion that results, for example, from the handling of a box. This disadvantage has been described earlier in the literature. The results of the study showed that the angles of the upper extremities have significantly higher measurement errors when occluded by either a work item or the subjects themselves. This especially applies for the flexion of the shoulders and elbows.

Kinect V2 mean ofdifferences – – -+1,95 SD ......- 1,96 SD  
![](images/a8902f36a7b57389c4bb90a4e850880824ab39375b082cc3b3450927a7a14f9b.jpg)

![](images/7ceec45a1c055f5ed6591b6ced52b861be4d2eb62ab9f4b165b44bbaca88f5d2.jpg)  
F I GU R E 6 Bland–Altman‐plot—Phases 2 and 3: Shoulder flexion

![](images/1d6e0e6b15cc0e2d597f6d892aabc307e73e356a0c399f8c74bebb1e7c8d5b93.jpg)

F I GU R E 7 Error pattern for movements with box  
![](images/332899e521af39c9bbcd73e2052bf906dde4e6c8ec4a8b5acd8002dd641f1eaf.jpg)

![](images/52e1d89bc7d254a7c1617a666742e7721d4c89bb44c3af3c9ea9f6a32d24d4cf.jpg)

Occlusion, therefore, remains a major disadvantage even when using two Kinect sensors. For complex movements, especially in industrial work environments with occlusions, the Captiv system is therefore preferable for ergonomic analyses in terms of accuracy in the majority of cases.

We conclude that the Kinect V2 sensor and the Captiv L7000 system are both able to obtain accurate results in situations withou occlusion for joint angles of upper extremities. However, the flexion angle of the shoulder is underestimated during measurement with the Kinect V2. In addition, Captiv is less accurate in determining elbow angles; in the case of this study possibly due to an insufficient initialization quality. For many clinical studies, these characteristics are sufficient as long as there are no occluded extremities while capturing with Kinect. In the next step, motion data captured by Captiv in a real manufacturing environment could be integrated in ergonomic risk assessment, for example, RULA or the European Assembly Worksheet.

Moreover, it may be worthwhile to test the Captiv system in further studies for its accuracy in more complex movement sequences to further demonstrate its applicability to real working environments. In addition, it would be desirable to assess the effec of initialization quality on the accuracy of the Captiv system. Also, the capturing with more than two Kinect sensors or the integration of raw data with another software than iPi Soft could be evaluated in an extension of this study to analyze if this leads to higher accuracy.

## CONFLICT OF INTERESTS

The authors declare that there are no conflict of interests.

## ORCID

Tim Steinebach http://orcid.org/0000-0003-1224-9121

## REFERENCES

Aptel, M., Aublet‐Cuvelier, A., & Cnockaert, J. C. (2002). Work‐related musculoskeletal disorders of the upper limb. Joint, Bone, Spine, 69 546–555.

Bartnicka, J., Piedrabuena, A., Portilla, R., Mleczko, K., Moyano, C., Pagador, J., … Tokarczyk, J. (2017). Ergonomic education in orthopedic surgery. LIFE: International Journal of Health and Life‐Sciences, 3, 194–215.

Bartnicka, J., Ziętkiewicz, A., & Kowalski, G. (2015). Ergonomic analysis of surgeries with the use of wireless body postures measurement system. Innovations in Manufacturing Management and Engineering, 2, 361–371.

BAuA–Federal Institute for Occupational Safety and Health (2018). Economic Costs due to Incapacity for Work 2016. Retrieved from https://www.baua.de/DE/Themen/Arbeitswelt-und-Arbeitsschutz-im-Wandel/Arbeitsweltberichterstattung/Kosten‐der‐AU/pdf/Kosten‐2016. pdf? blob=publicationFile&y=4

BKK Gesundheitsreport (2013). BKK Gesundheitsreport 2013—Gesundheit in Bewegung. Berlin, Germany: BKK Dachverband e. V.

Bland, J. M., & Altman, D. G. (1986). Statistical methods for assessing agreement between two methods of clinical measurement. Lancet, 1, 307–310.

Bonnechère, B., Jansen, B., Salvia, P., Bouzahouene, H., Sholukha, V., & Cornelis, J. (2014). Determination of the precision and accuracy of morphological measurements using the Kinect™ sensor. Comparison with standard stereophotogrammetry. Ergonomics, 57, 622–631.

Brosseau, L., Balmer, S., Tousignant, M., O'Sullivan, J. P., Goudreault, C., Goudreault, M., & Gringras, S. (2001). Intra‐ and intertester reliability and criterion validity of the parallelogram and universal goniometers for measuring maximum active knee flexion and extension of patients with knee restrictions. Archives of Physical Medicine and Rehabilitation 82, 396–402.

Buckle, P. W., & Devereux, J. J. (2002). The nature of work‐related neck and upper limb musculoskeletal disorders. Applied Ergonomics, 33, 207–217.

Cha, K., Lee, E. Y., Heo, M. H., Shin, K. C., Son, J., & Kim, D. (2015). Analysis of climbing postures and movements in sport climbing for realistic 3d climbing animations. Procedia Engineering, 112, 52–57.

Chapleau, J., Canet, F., Petit, Y., Laflamme, G. Y., & Rouleau, D. M. (2011) Validity of goniometric elbow measurements: Comparative study with a radiographic method. Clinical Orthopaedics and Related Research, 469 3134-3140.

Chen, Y. C., Lee, H. J., & Lin, K. H. (2015). Measurement of body joint angles for physical therapy based on mean shift tracking using two low cost Kinect images. 37th Annual International Conference of the IEEE Engineering in Medicine and Biology Society, 703–706.

Choppin, S., Lane, B., & Wheat, J. (2014). The accuracy of the Microsoft Kinect in joint ang‐le measurement. Sports Technology, 7, 98–105.

Choppin, S., & Wheat, J. (2012). Marker‐less tracking of human movement using Microsoft Kinect. 30th Annual Conference of Biomechanics in Sports—Conference Proceedings Archive, 86, 231–234.

Cippitelli, E., Gasparrini, S., Spinsante, S., Gambi, E., Verdini, F., Burattini, L., & Fioretti, S. (2015). Validation of an optimized algorithm to use Kinect in a non‐structured environment for Sit‐to‐Stand analysis. Engineering in Medicine and Biology Society (EMBC), 5078–5081.

Clark, R. A., Pua, Y.‐H., Oliveira, C. C., Bower, K. J., Thilarajah, S., & McGaw, R. (2015). Reliability and concurrent validity of the Microsoft

Xbox One Kinect for assessment of standing balance and postura control. Gait & Posture, 42, 210–213

Corazza, S., Mündermann, L., Gambaretto, E., Ferrigno, G., & Andriacchi, T. P. (2010). Markerless motion capture through visual hull articulated ICP and subject specific model generation. International Journal of Computer Vision, 87, 156–169.

De Koster, R., Le‐duc, T., & Roodbergen, K. J. (2007). Design and control of warehouse order picking: A literature review. European Journal of Operational Research, 182, 481–501.

Diego‐Mas, J. A., & Alcaide‐Marzal, J. (2014). Using Kinect™ sensor in observational methods for assessing postures at work. Applied Ergonomics, 45, 976–985.

Dolatabadi, E., Taati, B., & Mihailidis, A. (2016). Concurrent validity of the Microsoft Kinect for Windows v2 for measuring spatiotemporal gait parameters. Medical Engineering and Physics, 38, 952–958.

Dutta, T. (2012). Evaluation of the Kinect™ sensor for 3‐D kinematic measurement in the workplace. Applied Ergonomics, 43, 645–649.

Dzeng, R. J., Hsueh, H. H., & Ho, C. W. (2017). Automated Posture Assessment for construction workers. 40th International Convention on Information and Communication Technology, Electronics and Microelectronics, 1027–1031.

Eltoukhy, M., Kuenze, C., Oh, J., Wooten, S., & Signorile, J. (2017). Kinect based assessment of lower limb kinematics and dynamic postural control during the star excursion balance test. Gait & Posture, 58 421–427.

Fernandez‐Baena, A., Susin, A., & Lligadas, X. (2012). Biomechanical Validation of Upper‐Body and Lower‐Body Joint Movements of Kinect Motion Capture Data for Rehabilitation Treatments. 4th International Conference on Intelligent Net‐working and Collaborative Systems, 656–661.

Fornaser, A., Tomasin, P., De Cecco, M., Tavernini, M., & Zanetti, M (2017). Automatic graph based spatiotemporal extrinsic calibration of multiple Kinect V2 ToF cameras. Robotics and Autonomous Systems, 98, 105–125.

Geerse, D. J., Coolen, B. H., & Roerdink, M. (2015). Kinematic validation of a multi‐Kinect v2 instrumented 10‐meter walkway for quantitative gait assessments. PLOS One, 10, 1–15.

Hawi, N., Liodakis, E., Musolli, D., Suero, E. M., Stuebig, T., Claassen, L., & Citak, M. (2014). Range of motion assessment of the shoulder and elbow joints using a motion sensing input device: A pilot study Technology and Health Care, 22, 289–295.

Huber, M. E., Seitz, A. L., Leeser, M., & Sternad, D. (2015). Validity and reliability of Kinect skeleton for measuring shoulder joint angles: A feasibility study. Physiotherapy, 101, 389–393.

iPi Soft (2019). Dual Depth Sensors Configurations. Retrieved from http:/ docs.ipisoft.com/User\_Guide\_for\_Multiple\_Depth\_Sensors\_Configuration# Recording\_Video\_from\_Multiple\_Depth\_Sensors

Kuster, R. P., Heinlein, B., Bauer, C. M., & Graf, E. S. (2016). Accuracy of KinectOne to quantify kinematics of the upper body. Gait & Posture, 47, 80–85.

Lai, C. L., Tseng, S. Y., Huang, C. H., Pei, C., Chi, W. M., Hsu, L. C., & Sun, T. L. (2013). Fun and accurate static balance training to enhance fall prevention ability of aged adults: A preliminary study. Human Factors and Ergonomics in Manufacturing & Service Industries, 23, 517–527.

Lee, S. H., Yoon, C., Chung, S. G., Kim, H. C., Kwak, Y., Park, H. W., & Kim, K. (2015). Measurement of shoulder range of motion in patients with adhesive capsulitis using a kinect. PLOS One, 10, 1–12.

Manghisi, V. M., Uva, A. E., Fiorentino, M., Bevilacqua, V., Trotta, G. F., & Monno, G. (2017). Real time RULA assessment using Kinect v2 sensor. Applied Ergonomics, 65, 481–491.

Matsen, F. A., Lauder, A., Rector, K., Keeling, P., & Cherones, A. L. (2016) Measurement of active shoulder motion using the Kinect, a commercially available infrared position detection system. Journal of Shoulder and Elbow Surgery, 25, 216–223.

Milanese, S., Gordon, S., Buettner, P., Flavell, C., Ruston, S., & Coe, D. (2014). Reliability and concurrent validity of knee angle measurement. Smart phone app versus universal goniometer used by experienced and novice clinicians. Manual Therapy, 19, 569–574.

Mishra, A. K., Skubic, M., & Abbott, C. (2015). Development and preliminary validation of an interactive remote physical therapy system. 37th Annual International Conference of the IEEE Engineering in Medicine and Biology Society, 190–193.

Müller, B., Ilg, W., Giese, M. A., & Ludolph, N. (2017). Validation of enhanced kinect sensor based motion capturing for gait assessment. PLOS One, 12, 1–18.

Myles, P. S., & Cui, J. (2007). Using the Bland‐Altman method to measure agreement with repeated measures. British Journal of Anesthesia, 99, 309–311.

Nixon, M. E., Howard, A. M., & Chen, Y. (2013). Quantitative evaluation of the Microsoft Kinect for use in an upper extremity virtual rehabilitation environment. 2013 International Conference on Virtual Rehabilitation, 222–228.

Plantard, P., Muller, A., Pontonnier, C., Dumont, G., Shum, H. P., & Multon, F. (2017). Inverse dynamics based on occlusion‐resistant Kinect data: Is it usable for ergonomics? International Journal of Industrial Ergonomics, 61, 71–80.

Plantard, P., Shum, H. P., Le Pierres, A. S., & Multon, F. (2017). Validation of an ergonomic assessment method using Kinect data in real workplace conditions. Applied Ergonomics, 65, 562–569.

Romero, V., Amaral, J., Fitzpatrick, P., Schmidt, R. C., Duncan, A. W., & Richardson, M. J. (2017). Can low‐cost motion‐tracking systems substitute a Polhemus system when researching social motor coordination in children? Behavior Research Methods, 49(2), 588–601.

Rothstein, J. M., Miller, P. J., & Roettger, R. F. (1983). Goniometric reliability in a clinical setting. Physical Therapy, 63, 1611–1615.

Schmitz, A., Ye, M., Boggess, G., Shapiro, R., Yang, R., & Noehren, B. (2015). The measurement of in vivo joint angles during a squat using a single camera markerless motion capture system as compared to a marker based system. Gait & Posture, 41, 694–698.

Skals, S., Rasmussen, K. P., Bendtsen, K. M., Yang, J., & Andersen, M. S. (2017). A musculoskeletal model driven by dual Microsoft Kinect Sensor data. Multibody System Dynamics, 41, 297–316.

Stone, E. E., & Skubic, M. (2011). Evaluation of an inexpensive depth camera for passive in‐home fall risk assessment. 5th International ICST Conference on Pervasive Computing Technologies for Healthcare, 71–77.

Taber, M. J., Sweeney, D., Bishop, N., & Boute, R. (2017). Factor effecting the capability to jettison an S92 push‐out window. Internationa Journal of Industrial Ergonomics, 58, 79–89.

Vignais, N., Bernard, F., Touvenot, G., & Sagot, J. C. (2017). Physical risk factors identification based on body sensor network combined to videotaping. Applied Ergonomics, 65, 410–417.

Wang, Q., Kurillo, G., Ofli, F., & Bajcsy, R. (2015). Evaluation of pose tracking accuracy in the first and second generations of Microsoft Kinect. Proceedings of the 2015 International Conference on Healthcare Informatics, 380‐389.

Willmott, C. J., & Matsuura, K. (2005). Advantages of the mean absolute error (MAE) over the root mean square error (RMSE) in assessing average model performance. Climate Research, 30, 79–82.

Xu, X., Robertson, M., Chen, K. B., Lin, J. H., & McGorry, R. W. (2017). Using the Microsoft Kinect™ to assess 3‐D shoulder kinematics during computer use. Applied Ergonomics, 65, 418–423.

Zhou, L., Li, Y., & Bai, S. (2017). A human‐centered design optimization approach for robotic exoskeletons through biomechanical simulation. Robotics and Autonomous Systems, 91, 337–347.

How to cite this article: Steinebach T, Grosse EH, Glock CH, Wakula J, Lunin A. Accuracy evaluation of two markerless motion capture systems for measurement of upper extremities: Kinect V2 and Captiv. Hum. Factors Man. 2020;30:291–302. https://doi.org/10.1002/hfm.20840