![](images/9a2a356a354b7a007252e61f5aaea3a3026985bfaf807c8568cbada8a5112420.jpg)

RESEARCH ARTICLE

# Finger Muscle Attachments for an OpenSim Upper-Extremity Model

Jong Hwa Lee<sup>1</sup>, Deanna S. Asakawa<sup>2</sup>, Jack T. Dennerlein<sup>3</sup>, Devin L. Jindrich<sup>2</sup>\*

1 Department of Mechanical and Aerospace Engineering, Arizona State University, Tempe, Arizona, United States of America, 2 Department of Kinesiology, California State University, San Marcos, California, United States of America, 3 Department of Physical Therapy, Movement, and Rehabilitation Sciences, Bouvé College of Health Sciences, Northeastern University, Boston, Massachusetts, United States of America

\* djindrich@csusm.edu

CrossMark click for updates

![](images/53576e2482faefc095d86ebb37493645dc99825bb640059d69ff69c5a29c287d.jpg)

## OPEN ACCESS

Citation: Lee JH, Asakawa DS, Dennerlein JT, Jindrich DL (2015) Finger Muscle Attachments for an OpenSim Upper-Extremity Model. PLoS ONE 10(4): e0121712. doi:10.1371/journal.pone.0121712

## Abstract

Academic Editor: David Carrier, University of Utah, UNITED STATES

Received: November 8, 2014

Accepted: February 14, 2015

Published: April 8, 2015

Copyright: © 2015 Lee et al. This is an open access article distributed under the terms of the Creative Commons Attribution License, which permits unrestricted use, distribution, and reproduction in any medium, provided the original author and source are credited.

Data Availability Statement: All relevant data are within the paper.

Funding: This work was supported by the National Science Foundation (NSF 0964220). The funders had no role in study design, data collection and analysis, decision to publish, or preparation of the manuscript.

Competing Interests: The authors have declared that no competing interests exist.

We determined muscle attachment points for the index, middle, ring and little fingers in an OpenSim upper-extremity model. Attachment points were selected to match both experimentally measured locations and mechanical function (moment arms). Although experimental measurements of finger muscle attachments have been made, models differ from specimens in many respects such as bone segment ratio, joint kinematics and coordinate system. Likewise, moment arms are not available for all intrinsic finger muscles. Therefore, it was necessary to scale and translate muscle attachments from one experimental or model environment to another while preserving mechanical function. We used a two-step process. First, we estimated muscle function by calculating moment arms for all intrinsic and extrinsic muscles using the partial velocity method. Second, optimization using Simulated Annealing and Hooke-Jeeves algorithms found muscle-tendon paths that minimized root mean square (RMS) differences between experimental and modeled moment arms. The partial velocity method resulted in variance accounted for (VAF) between measured and calculated moment arms of 75.5% on average (range from 48.5% to 99.5%) for intrinsic and extrinsic index finger muscles where measured data were available. RMS error between experimental and optimized values was within one standard deviation (S.D) of measured moment arm (mean RMS error = 1.5 mm < measured S.D = 2.5 mm). Validation of both steps of the technique allowed for estimation of muscle attachment points for muscles whose moment arms have not been measured. Differences between modeled and experimentally measured muscle attachments, averaged over all finger joints, were less than 4.9 mm (within 7.1% of the average length of the muscle-tendon paths). The resulting non-proprietary musculoskeletal model of the human fingers could be useful for many applications, including better understanding of complex multi-touch and gestural movements.

## Introduction

Dexterous manipulation often involves complex movements of several fingers. For example, grasping or pinching can involve the coordinated activations of many hand muscles [1]. Movements similar to pinching and grasping are commonly used as inputs for human-computer interfaces (HCIs) such as a smart phone and tablet computer touch screen. The “multitouch” interfaces of these devices often require complex, multi-finger gestures or gesture sequences on the touch screen [2]. However, we have little understanding of whether the cumulative effects of long-term exposures can lead to injuries such as musculoskeletal disorders (MSDs). One way to identify exposure to forces that may lead to injury is to estimate muscle and tendon forces during repetitive activities. However, individual muscle tension or stress is difficult to measure in vivo [3,4].

Musculoskeletal models can be helpful for understanding many aspects of biomechanics and motor control. Biomechanical models can help to understand hand anatomy [5,6]. Models have also contributed to understanding upper-extremity biomechanics, including internal muscle loading during motor tasks [7–10]. Musculoskeletal models have also helped to understand neural control of hand and arm movements. For example, models helped to identify the function of the intrinsic hand [11,12] and thumb and index finger muscles [13,14], and to study movement coordination of the interphalangeal joints [15,16].

However, existing models have several limitations. First, many models are developed with proprietary or commercial software, and are not generally available for use. Second, many models are two-dimensional (2D), whereas movements are often in three-dimensional (3D) space. Third, many models focus on specific joints or sets of joints and do not include the entire upper extremity. Finally, even models that incorporate all of the major skeletal segments of the hand and arm do not currently include all of the muscles potentially responsible for actuating complex finger motions.

Muscle attachment points have been experimentally measured for both intrinsic and extrinsic muscles of the hand [5]. However, published attachments cannot be directly used in the OpenSim model [9]. The 50th percentile male used for the OpenSim model differs from the specimens used for experimental measurements. For example, the OpenSim model and experimental specimens have different sized segments, segment proportions, joint center positions, rotational axis orientations and anatomical coordinate systems. Therefore, directly adding experimentally measured muscle attachments to the OpenSim model results in different modeled moment arms than those experimentally measured for the same muscles. Muscle moment arm is an important functional measure because it determines the joint torques that result from muscle forces. Therefore, to create a musculoskeletal model capable of evaluating muscle function, it is necessary to determine sets of muscle attachments that result in moment arms that are representative of human subjects.

Several studies have measured moment arm in vivo [17–19] and in situ [20–24]. However, to our knowledge, moment arms for all muscles of the middle, ring and little fingers have not been reported. Moreover, even when moment arms are known, the specific muscle attachments are unknown and indeterminate: many potential muscle attachments can result in similar moment arms [25].

To overcome some of these limitations, we seek to add intrinsic and extrinsic finger muscles to a non-proprietary, 3D musculoskeletal model of the upper extremity [9,26]. The OpenSim upper-extremity model currently incorporates muscles at the shoulder and elbow. Adding finger muscles is a necessary component of using the model to better understand dexterous finger movements. The OpenSim model is particularly suited to help understand the role of segment interactions and the consequences of multi-joint articulation of finger muscles.

Therefore, we sought to identify muscle attachments for the intrinsic and extrinsic finger muscles within the OpenSim upper-extremity model that resulted in the functionally important characteristic of matching experimentally measured moment arms.

The procedure we used to identify muscle attachments involved two steps. First, we determined muscle moment arms from experimentally measured tendon locations [5]. We used the “partial velocity” method calculation [27,28] to reproduce anatomical moment arms from experimentally measured muscle attachment points [5]. Successful prediction was considered to be calculated moment arms within one standard deviation (S.D) of experimental measurements (S.D = 2.5 mm; [6,20]). Second, given a moment arm curve, we identified muscle-tendon paths using a data-driven optimization that is capable of determining muscle attachments whose resulting moment arms matched experimental measurements [25]. We considered successful approximation to be optimized moment arms within 10% of experimentally derived values [29,30].

## Methods

We expanded an existing musculoskeletal model of upper extremity on the OpenSim platform to include finger muscles (2.3.2, Simbios, Stanford, CA; [9,31]). Currently, the shoulder, elbow, forearm, wrist, thumb and index finger are modeled with 15 joint degrees of freedom (DOFs) and 50 muscle elements. The existing model is actuated by 50 muscle compartments including extrinsic muscles of the thumb and index finger. The rotational axes and centers for the thumb joints are based on measured values [32,33], and those for the index finger joints are modeled as the long axis of cylinders fit to the articular surfaces of the metacarpal and phalangeal bones.

## Musculoskeletal Model

We added custom joints to the middle, ring and little fingers of the OpenSim model because only the thumb and index finger currently have active joints [9]. Each finger was modeled to have four DOFs linking four successive bones: metacarpal bone, proximal phalange, middle phalange and distal phalange. These four bone segments were linked with three joints: metacarpophalangeal (MCP), proximal interphalangeal (PIP) and distal interphalangeal (DIP). The MCP joint was capable of flex/extension and ab/adduction as a universal joint (2 DOFs). The PIP and DIP joints were limited to flex/extension as revolute joints (1 DOF). We constrained joint angles to be within their physiological range of motion (RoM) from experimental measurements [20,23,34]. The MCP joints were modeled with a RoM of 0° (extension:-) to 90° (flexion: +) as well as 0° (adduction:-) to 30° (abduction: +). The PIP and DIP joints were also modeled with a RoM of 0° to 90° (flexion: +) and 50° (flexion: +), respectively.

For all fingers, we modeled the following muscles: terminal extensor (TE), extensor slip (ES), radial band (RB), ulnar band (UB), dorsal interosseous or radial interosseous (RI), lumbricals (LU), palmar interosseous or ulnar interosseous (UI), flexor digitorum profundus (FDP), flexor digitorum superficialis (FDS) and extensor digitorum communis or long extensor (EDC). The wrapping surfaces for all fingers were based on previous studies [9,25,31,35]. We modified the wrapping surface parameters (e.g., dimension, orientation and position) to prevent the muscles passing into the bone for the entire range of movements.

## Scaling Model and Moment Arms

To normalize for differences among data sets and reproduce hand skin surface from the bony segments in the OpenSim model, we assumed that all linear dimensions scaled isometrically, as found for the ratio among the length of phalanx, the width and thickness of each joint [5,25,36–39]. We scaled measured anthropometric data to the OpenSim model dimensions to

Table 1. Anthropometric index finger dimensions of cadaveric specimens [20] and OpenSim model (mm).

<table><tr><td></td><td>Specimen bony dimensions</td><td>OpenSim bony dimensions</td><td>Skin surface scaled</td></tr><tr><td>Distal phalanx length</td><td>19.67±1.03</td><td>19.10(Δ0.57)</td><td>30.65</td></tr><tr><td>Middle phalanx length</td><td>24.67±0.98</td><td>25.10(Δ0.43)</td><td>27.22</td></tr><tr><td>Proximal phalanx length</td><td>43.57±0.98</td><td>42.60(Δ0.97)</td><td>50.86</td></tr><tr><td>DIP joint thickness</td><td>5.58±0.92</td><td>4.95(Δ0.63)</td><td>14.38</td></tr><tr><td>PIP joint thickness</td><td>7.57±0.45</td><td>7.31(Δ0.26)</td><td>18.86</td></tr><tr><td>MCP joint thickness</td><td>15.57±0.84</td><td>17.08(Δ1.51)</td><td>27.80</td></tr></table>

Symbol (±) indicates standard deviation in interspecimen variation. Lengths of the phalanges in OpenSim model are calculated by the distance between the origins of two coordinate systems in three-dimensional (3D) Cartesian space, e.g., the center of rotation at MCP and the center of rotation at PIP. Parentheses (Δ) in OpenSim bony dimensions express difference between model dimension and specimen dimension. Skin surface set is scaled in threedimensions to preserve measured anatomical proportions [36–39]. These skin surface (external dimensions) function as upper boundary constraints during optimization.

doi:10.1371/journal.pone.0121712.t001

describe muscle-tendon paths within OpenSim (Table 1), then normalized moment arms to the length of the middle phalanx [5,25].

## Reproducing Moment Arms from Experimentally Measured Attachments

Because experimental moment arm curves are available only for the index finger at the MCP joint, it was necessary to calculate moment arm values from measured muscle attachment points for the middle, ring and little fingers. We calculated moment arms from tendon attachment locations [5] using the “partial velocity” method [28]. The partial velocity method provides a consistent technique to compute the moment arms of muscles crossing many types of joints [27].

To validate the moment arms calculated using the partial velocity method, we compared moment arms calculated from muscle attachments for the index finger [5] to measured moment arms [20]. Experimentally measured moment arms as a function of joint angle were only available for the MCP joint of the index finger, with the exception of the FDS muscle [40]. Therefore, we limited comparisons of moment arm curves to the index finger MCP joint and the FDS muscle for all of the fingers.

## Muscle Attachment Determination

We used a data-driven optimization approach to identify muscle attachment sites that resulted in moment arms that most closely matched the experimentally derived curves over the joint range of motion. We performed optimizations using the same methods that were used to find attachment locations of the index finger [25]. The optimization consists of three parts: 1) an objective function, 2) boundary conditions and 3) an inequality constraint. We defined the objective function as the root mean square (RMS) error between the experimentally derived moment arms, $\Gamma _ { \mathrm { j } } ( \vec { \bf q } _ { \mathrm { i } } )$ and the model-estimated moment arms, $\hat { \bf r } _ { \mathrm { j } } ( \vec { \bf q } _ { \mathrm { i } } , \vec { \bf x } )$ . The optimization searched for the minimum values of RMS error $( f ( \vec { x } ) )$ over the domain of attachment points (x<sup>!</sup>) and joint angles (q<sup>!</sup>) that satisfy both flex/extension and ab/adduction moment arm relationships $( { \bf g } _ { \mathrm { j } } \left( { \bf x } \right) )$ . Moment arm curves are different in magnitude and shape at every joint. However, we do not currently have criteria to weight errors at different joints. Therefore, we summed RMS errors for each muscle over all joints in the objective function.

Minimize

$$
f (\vec {x}) = \sqrt {\sum_ {i = 1} ^ {m} \frac {\left[ r _ {j} (\vec {q} _ {i}) - \hat {r} _ {j} (\vec {q} _ {i} , \vec {x}) \right] ^ {2}}{m}}
$$

Subject

$$
l b _ {j} \leq x _ {j} \leq u b _ {j}
$$

$$
\mathbf {g} _ {\mathrm{j}} (\vec {x}) - \varepsilon_ {\mathrm{j}} \leq 0
$$

We defined the optimization parameters and variables as:

x<sup>!</sup> Muscle attachment points (distal and proximal; $\vec { x } \in \mathcal { R } ^ { 6 } )$

q<sup>!</sup> Joint angle (resolution (i) of 100 increments (m) covering the RoM)

$r _ { j } ( \vec { q } _ { i } )$ Experimentally measured moment arms

$\hat { r } _ { j } ( \vec { q } _ { i } , \vec { \bf x } )$ Model-estimated moment arms

i Joint motions (flex/extension or ab/adduction)

j Individual muscles $( j = 4 0 )$

$l b _ { j }$ Lower bound (bone surface)

ub<sub>j</sub> Upper bound (skin surface)

$\varepsilon _ { \mathrm { j } }$ Maximum standard deviation of experimental moment arms

$\mathrm { g } _ { \mathrm { j } } ( \vec { x } )$ RMS error of ab/adduction moment arms during flexion/extension movements

Second, the boundary conditions constrained muscle attachments to be between the bone (the lower bound, $l b _ { j } \left[ \underline { { 9 } } \right] )$ and the skin surface (the upper bound, $u b _ { j } [ { \underline { { 3 6 } } } , { \underline { { 3 8 } } } , { \underline { { 3 9 } } } ] )$

Third, we imposed an inequality constraint on ab/adduction $( g _ { j } ( \vec { x } ) \leq \varepsilon _ { \mathrm { j } } )$ . Muscle-tendon paths have moment arms for both flex/extension and ab/adduction at the MCP joint. However, measured moment arms for ab/adduction are more uncertain than those for flex/extension. For example, ab/adduction moment arms have higher standard deviations than flex/extension moment arms, reflecting either measurement or anatomical variability [5,20,21,23]. Moreover, ab/adduction moment arm values depend on joint angles of the MCP, PIP and DIP joints [41]. However, the specific postures used to measure ab/adduction moment arm values are not available. Therefore, we included ab/adduction in the optimization by using an inequality constraint $( g _ { j } ( \vec { x } ) \leq \varepsilon _ { \mathrm { j } } )$ to find solutions where muscle attachments $( \vec { x } )$ resulted in moment arms within the standard deviations about experimentally measured moment arms, $( g _ { j } ( \vec { x } ) \leq \mathfrak { s } _ { \mathrm { j } } )$ . The standard deviations for ab/adduction, $\varepsilon _ { \mathrm { j } } ,$ were 2.5 mm for extrinsic muscles and 1.7 mm for intrinsic muscles [20]. Our optimization was implemented using Simulated Annealing [42] and Hooke-Jeeves [43] algorithms. Both algorithms were chosen because they have been used successfully for musculoskeletal modeling [25].

Initial parameter selection can influence optimization in complex search spaces [13,25,43]. Therefore, we performed sensitivity analyses for all attachments (x<sup>!</sup><sub>j</sub>). We performed optimizations using 26 different starting points $( \overrightarrow { x } _ { 0 } )$ . The starting points included experimental estimates attachment locations [5] and where possible existing OpenSim model attachments [9]. Optimizations from different initial conditions can discover multiple solutions, which require a criterion to select a single set of attachment points [25]. We assumed that the optimized attachment points that were nearest to the experimentally measured locations and resulted in smooth tendon paths were the most anatomically reasonable.

To determine the locations of experimentally measured attachments in the OpenSim coordinate systems, we transformed attachments from the An et al. (1979) reference frames to OpenSim. We constructed a transformation matrix that mapped translation and rotation in three dimensional (3D) Cartesian coordinates (3D Euclidean space). We approximated kinematic parameters: segment rotation (θ), segment distance (d), joint twist (α) and joint offset (a)

from the OpenSim model [9]. X coordinate axes were assigned to the joint axes (-: ulnar to +: radial deviation), Y coordinate axes were assigned to the phalangeal bone (-: distal to +: proxi mal phalanx), and Z coordinate axes were determined by the right hand rule (-: palmar to +: dorsal direction) in the OpenSim coordinate system. Primary coordinate systems of An et al. (1979), located at the approximate center of rotation of the distal, proximal phalangeal and metacarpal heads (2, 4 and 6 reference frames), were translated to OpenSim PIP, MCP joints and middle of metacarpal bone coordinate systems, respectively. Secondary coordinate systems of An et al. (1979), a translation of the proximal systems to the centers of the concave articular surface (1, 3 and 5 reference frames) were translated to OpenSim DIP, PIP and MCP joint coordinate systems, respectively.

Attachment points closest to experimentally measured values were chosen for analysis and presentation. If several candidate attachment points had Euclidean distances to the experimentally measured point within the standard deviation of experimental measurements (28% of middle phalanx length [5]), we chose the path with the largest radius of curvature (the “smoothest” path). Curvature is the rate of change of tendon path direction (i.e. deviation of the tendon path from a straight line). The radius of curvature was determined from three successive attachments, e.g., origin, via and insertion points [25]. The tendon path with the largest radius of curvature was considered the smoothest path and chosen for analysis and presentation. To measure the agreement between model-optimized and experiment-measured moment arm curves, we estimated the variance accounted for (VAF) using the equation:

$$
\mathrm{VAF} = 1 0 0 \times \left[ 1 - \sqrt {\sum (\vec {r} - \vec {r}) ^ {2}} / \sqrt {\sum \vec {r} ^ {2}} \right]
$$

We implemented optimizations in Matlab (2012a, Mathworks, Natick, MA) using the OpenSim Application Programming Interface (API; OpenSim 2.0 Doxygen) to compute moment arms.

## Results

## Moment Arms Calculated from Muscle Attachments Fitted Experimentally Measured Values for the Index Finger

Moment arms calculated using the partial velocity method fitted those measured at the MCP joint of the index finger (Fig 1; Table 2). Over the entire range of motion, calculated moment arms derived from anatomical muscle attachments [5] were within one standard deviation of experimentally measured moment arms [20]. Variance accounted for (VAF) averaged 75.5% across all index finger muscles, ranging from minimum (min.) 48.2% (UI) to maximum (max.) 99.5% (FDS). VAF for the UI muscle was low because of the small value for the moment arms of this muscle. However, RMS error for UI muscle was within one standard deviation (S.D) of experimentally measured moment arms (RMS error = 2.0 mm < measured S.D = 2.1 mm), implying that the calculated moment arm was reasonable. VAF were 96.5% for extrinsic tendons and 86.9% for intrinsic muscles (flex/extension).

## Moment Arms Calculated from Muscle Attachments were Reasonable for All Finger Muscles

Calculated moment arms for the middle, ring, and little finger muscles were reasonable based on moment arm magnitude and variation with joint angle. Specifically, all calculated flex/extension moment arms had an ordering of magnitude that was comparable to the ordering of finger dimensions. Moment arms for the middle finger were largest, ring or index second and little smallest (Figs 2–4). Moreover, calculated moment arms were consistent functions ofjoint angle. Calculated moment arms of FDP, FDS, RI, UI and LU increased with finger flexion, while those of TE, ES and EDC decreased with flexion for the index, middle, ring and little fingers. Moment arm variation with flex/extension was consistent across all finger joints.

![](images/26b79e04d399936fd98e3907f93becb06ef073fba62095aabbf580b57f91b667.jpg)  
Fig 1. Measured and derived flex/extension moment arms (mm) as a function of flexion (+)/ extension (-) at the MCP joint of the index finger. Dotted moment arm values are derived from experimental muscle attachments [5], and solid moment arm values are direct measurements (n = 7 specimens with mean and standard deviation (error bar); [20]). Positive values indicate flexion moment arms, negative values indicate extension moment arms, and 0° is full extension.  
doi:10.1371/journal.pone.0121712.g001

Unlike flex/extension moment arms, ab/adduction moment arms did not show ordering that paralleled finger dimensions. For example, calculated FDS and EDC ab/adduction moment arms for the little finger were larger than those for the middle finger (Fig 3). However,

Table 2. Derived mean moment arms (MA) from experimentally measured muscle attachments [5].

<table><tr><td>Joint</td><td>RoM</td><td></td><td>FDP</td><td>FDS</td><td>EDC</td><td>LU</td><td>RI</td><td>UI</td></tr><tr><td rowspan="8">MCP2</td><td></td><td>MA</td><td>11.5</td><td>12.4</td><td>-7.4</td><td>3.5</td><td>6.6</td><td>2.0</td></tr><tr><td> $0^{\circ} \sim 90^{\circ}$ </td><td>(VAF)</td><td>(98.7%)</td><td>(99.5%)</td><td>(91.0%)</td><td>(77.0%)</td><td>(95.9%)</td><td>(87.9%)</td></tr><tr><td>Flex/Ext</td><td>RMS</td><td> $\Delta 0.2$ </td><td> $\Delta 0.1$ </td><td> $\Delta 0.7$ </td><td> $\Delta 0.8$ </td><td> $\Delta 0.3$ </td><td> $\Delta 0.3$ </td></tr><tr><td></td><td>(Error)</td><td>(1.2%)</td><td>(0.5%)</td><td>(8.0%)</td><td>(23.4%)</td><td>(4.3%)</td><td>(73.9%)</td></tr><tr><td></td><td>MA</td><td>-2.8</td><td>2.2</td><td>2.5</td><td>4.3</td><td>4.1</td><td>-6.8</td></tr><tr><td> $0^{\circ} \sim 30^{\circ}$ </td><td>(VAF)</td><td>(68.2%)</td><td>(63.0%)</td><td>(53.2%)</td><td>(68.1%)</td><td>(56.8%)</td><td>(48.2%)</td></tr><tr><td>Ab/Add</td><td>RMS</td><td> $\Delta 0.8$ </td><td> $\Delta 3.4$ </td><td> $\Delta 2.6$ </td><td> $\Delta 1.5$ </td><td> $\Delta 3.1$ </td><td> $\Delta 2.0$ </td></tr><tr><td></td><td>(Error)</td><td>(31.2%)</td><td>(41.6%)</td><td>(99.4%)</td><td>(10.7%)</td><td>(79.6%)</td><td>(43.0%)</td></tr></table>

All are expressed in millimeters (mm) with flexion (+)/ extension (-) and abduction (+)/ adduction (-) at the MCP joint (degree) of the index finger. (VAF) represents Variance Accounted For, RMS represents root mean square error (Δ) between derived moment arms [5] and experimentally measured values [20], and (Error) represents percentage error, respectively.

doi:10.1371/journal.pone.0121712.t002

![](images/5719df7eeee239c3a0cc5c0c388f4c2912f67eebb18b001f2c5ce6bd3652da42.jpg)

![](images/13a92e8a778d873ba56c96cfe955ee713961c79571da73135536b4ece622c03c.jpg)  
Fig 2. Flex/extension moment arms (mm) as a function of flexion (+)/ extension (-) at the MCP joint of the all fingers. Dotted moment arm values are derived from experimentally measured muscle attachments [5], and solid moment arm values are direct measurements (n = 7 specimens with mean and standard deviation; [20]). Positive values indicate flexion moment arms, negative values indicate extension moment arms, and 0° is full extension. Blue, green, red and cyan colors represent index, middle, ring and little finger moment arms, respectively.  
doi:10.1371/journal.pone.0121712.g002

![](images/456b2ca6e1bbb15053ec2e2fcbcea98c86c6951e6b77b0541084bbdeba568254.jpg)  
Fig 3. Ab/adduction moment arms (mm) as a function of abduction (+)/ adduction (-) at the MCP joint of the all fingers. Symbols are the same as in Fig 2.

![](images/b08427039efb71d34a258d2060d759167a1eb5afc17b90911f43f8ffd023ebdb.jpg)  
Fig 4. Flex/extension moment arms (mm) as a function of flexion (+)/ extension (-) at the PIP and DIP joints of the all fingers. Symbols are the same as in Fig 2.

little finger moment arms were consistent with FDS and EDC muscle locations, which were longer than measured middle finger muscle locations from the MCP joint (Euclidean distance, $d _ { l i t t l e } = 0 . 5 2 > d _ { m i d d l e } = 0 . 4 0 >$ middle phalanx length). Therefore, the partial velocity calculation predicted moment arm curves consistent with expectations based on finger anatomy.

## Optimized Moment Arms Matched Calculated Values Derived from Anatomical Attachments

Optimization identified multiple muscle attachments that locally minimized RMS error (Fig 5). The average Euclidean distance from experimentally measured attachment points to the closest, most smooth optimized muscle attachments (i.e., those selected for analysis and presentation) was 4.9 mm, or 7.1% of the average length along the muscle-tendon pathways.

Model-optimized moment arms agreed with experimentally derived values for the index, middle, ring and little fingers at the MCP joint (Figs 6–9). VAF averaged 84.3, 80.4, 86.6 and 84.1% for the index, middle, ring and little fingers, respectively (Table 3, 4, 5, 6). At the middle finger (min. VAF = 80.4%), RMS error ranged from min. 0.3 mm (UI) to max. 2.4 mm (EDC). Because S.D of moment arms has not been reported for any finger but the index finger, we compared the RMS errors to the S.D of index finger moment arms. Max. RMS error between computationally optimized and experimentally derived moment arm values was within one S. D of the experimentally measured index finger moment arm for all muscles (mean RMS error = 1.5 mm < measured S.D = 2.5 mm; [20]).

For the PIP and DIP joints, model-optimized moment arms also matched experimentally derived data (Figs 6–9). VAF averaged 67.4, 81.8, 92.2 and 62.6% for muscles at the index,

![](images/444a98401dfc321ab19d03ed195a29c10e9f9cea334d2183f56218f18085c40d.jpg)  
Fig 5. Muscle attachment points for modeled FDP muscles at the MCP, PIP and DIP joints. Red circles and blue squares denote proximal and dista points, respectively.

doi:10.1371/journal.pone.0121712.g005

![](images/b0ad98afd2e1ea331830d548dc2af0be3018d438e4636821015cbfd9bdc3eed8.jpg)

Fig 6. Index finger moment arm values (mm). Left MCP curves, PIP and DIP curves represent flex/extension moment arms as a function of flexion (+)/ extension (-). Right MCP curves represent ab/adduction moment arms as a function of abduction (+)/ adduction (-). Solid curves (with plot markers) represent experimentally derived moment arms from anatomical attachment locations, and dotted curves represent optimally estimated moment arms from data-driven optimizations. Positive values indicate flexion moment arms, negative values indicate extension moment arms, and 0° flexion is full extension, respectively.

doi:10.1371/journal.pone.0121712.g006

middle, ring and little finger PIP joints, respectively. For the DIP joints, VAF averaged 73.5, 81.4, 87.1 and 89.6% across all muscles for the index, middle, ring and little fingers, respectively. RMS error between computationally optimized and experimentally derived moment arm values was within one S.D of the experimentally measured index finger moment arm for all muscles (mean RMS error = 0.7 mm < measured S.D = 1.3 mm; [20]).

## Discussion

The goal of this study was to determine attachment points for muscles of the index, middle, ring and little fingers in the OpenSim upper-extremity model. Moment arms calculated from experimentally measured muscle attachments using the partial velocity method matched experimentally measured moment arms for the index finger, where both attachments and moment arm data were available. Optimization found OpenSim muscle attachments resulting in moment arms that matched moment arms derived from experimental measurements.

## Limitations

Our approach had several limitations. First, the muscle wrapping objects that we used during optimization were simple shapes that in some circumstances can result in discontinuous moment arm curves [9]. Muscle wrapping can be used to better describe muscle-tendon geometry [44,45]. However, to our knowledge quantitative, experimental data for wrapping surfaces are not available for finger muscles. Therefore, we chose to use minimal wrapping surfaces that prevented the muscle-tendon paths from penetrating the bone, but did not otherwise change the muscle-tendon paths.

Second, compiling data drawn from different reference frames and sources into a common model could introduce errors. We attempted to mitigate these potential sources of error by primarily basing our calculations on data collected from all fingers from the same specimens [5,20]. Moreover, we normalized not only by middle phalanx length for flex/extension moment arms at the PIP and DIP joints, but also normalized by MCP thickness for flex/extension moment arms and MCP width for ab/adduction values.

Third, although it was possible to estimate experimentally derived muscle attachment locations in OpenSim, the coordinate transformations involved several assumptions or approximations. Attachment estimations required both scaling and coordinate transformations from 2D to 3D. Coordinate transformations are complicated by the absence of quantitative information for both 3D rotation and translation in the 2D model. We were therefore required to make assumptions about both 3D reference frame orientations and origins. However, the discovery of muscle attachments that were similar to finger anatomy suggests that our assumptions were reasonable (i.e., low ΔR = 4.9 mm, which was less than the 6.9 mm interspecimen phalanx variation observed experimentally; Tables 7–10). Moreover, optimization resulted in muscle moment arms that matched experimentally measured values. Therefore, even though the attachment sites are for a hypothetical 50<sup>th</sup> percentile male and not for a specific specimen, muscle function is as close to experimentally measured values as feasible.

Finally, we did not conduct experiments to validate the model at a functional level [13]. Full model validation was outside of the scope of the present study because it would require the determination of muscle physiological parameters and measurements of muscle activation during

![](images/8bc906d7fb73e7db6e3b1b1c1001d9b894d0fa26e6108f0a7efaea63b3d61e43.jpg)

Fig 7. Middle finger moment arm values (mm). Left MCP curves, PIP and DIP curves represent flex/extension moment arms as a function of flexion (+)/ extension (-). Right MCP curves represent ab/adduction moment arms as a function of abduction (+)/ adduction (-). Solid curves (with plot markers) represent experimentally derived moment arms from anatomical attachment locations, and dotted curves represent optimally estimated moment arms from datadriven optimizations.

doi:10.1371/journal.pone.0121712.g007

diverse tasks. Moreover, modeling potentially complex structures such as the extensor mechanism may require additional data, assumptions, and validation [46]. We therefore limited model validation to the anatomical level of muscle attachment locations. Future experiments will seek to functionally validate the model and resolve discrepancies among experimental measurements [20,21,40].

## Moment Arms and Muscle Attachments were Reasonable

Moment arms calculated using the partial velocity method for the middle, ring and little finger muscles (whose experimentally measured moment arms are not available) were reasonable based on moment arm magnitude and variation with joint angle. Flex/extension moment arms had the same ordering of magnitude as the finger dimensions, e.g., moment arms for the middle finger largest, ring or index second, and little smallest [40]. Joint thickness or phalanx lengths influenced moment arms, e.g., moment arm magnitudes increased with phalanx lengths [20,21,35,40]. Moreover, calculated moment arm variation with joint angle was consis tent among muscles at all finger joints. For example, calculated flexor moment arms increased with finger flexion; calculated extensor values decreased. Experimentally measured moment arms for extrinsic (FDP/FDS) and intrinsic (RI/UI/LU) flexors increased with flexion for the index, middle, ring and little fingers, whereas extensors (TE/ES/EDC) increased with extension as observed in previous studies [5,6,20,23,24,36]. Therefore, the partial velocity method predicted moment arm values whose relationships to joint angle agreed with experimental data.

Muscle attachments found by optimization were also reasonable. Optimization identified multiple muscle attachments that locally minimized RMS error (Fig 5). However, at least one set of muscle attachments were close to experimentally measured attachments. The average distance between modeled and measured muscle attachments fell within 7.1% of the average length of the muscle paths and 71.0% of one standard deviation of specimens’ phalanx length. Moreover, the agreement of the OpenSim moment arms with both directly measured moment arms [20] and moment arms calculated from muscle attachments [5] suggested that the Open-Sim muscle attachments are reasonable within the OpenSim coordinate system.

Although optimization found moment arm curves that closely matched experimental values for most muscles, some muscles such as UI showed poorer fits. Differences in VAF may in part be related to variability in experimentally measured data. An et al. (1983) reported more variability in ab/adduction moment arm (S.D = 2.5 mm) than that of flex/extension (S.D = 2.1 mm). Ab/adduction moment arms could also be influenced by finger postures to a greater extent than flex/extension moment arms [41]. Moreover, although RMS error was the same for ab/adduction and flex/extension, VAF was lower for ab/adduction because of smaller overall moment arms than flex/extension. Ab/adduction moment arms (mean MA = 3.4 mm) were less than half the magnitude of flex/extension values (mean MA = 7.2 mm). Therefore, low VAF values for some muscles may reflect variability, experimental uncertainty, and small overall moment arm values.

![](images/39fe5efb488438bdad3179c15882a3755a43d63577184c1d9fc2c3a2a57ea6db.jpg)

Fig 8. Ring finger moment arm values (mm). Left MCP curves, PIP and DIP curves represent flex/extension moment arms as a function of flexion (+)/ extension (-). Right MCP curves represent ab/adduction moment arms as a function of abduction (+)/ adduction (-). Solid curves (with plot markers) represent experimentally derived moment arms from anatomical attachment locations, and dotted curves represent optimally estimated moment arms from datadriven optimizations.

doi:10.1371/journal.pone.0121712.g008

## Objective Methods could Facilitate Transformation among Musculoskeletal Models

The optimization procedure we proposed could potentially help to translate data from one musculoskeletal model to another. We demonstrated that reasonable muscle attachments can be determined using muscle function (e.g., moment arms) as a constraint. Scaling or individualizing musculoskeletal models can be difficult. Often, scaling is based on body segment lengths or limb circumferences. For example, scaling has been based on finger middle phalanx length [5], scaling elbow moment arm by cross-sectional area [47], or other linear dimensions [48]. However, scaling alone does not account for other differences among models such as different axes of rotation [34]. Using both scaling and data-driven optimization of functional parameters can identify muscle-tendon paths that both maintain anatomical proportionality and muscle function at the same time.

## Muscle Parameters for a Dynamic Model

We added only anatomical information to the OpenSim model. Musculoskeletal anatomy alone can be helpful for understanding some aspects of function such as joint stability [49]. However, to dynamically simulate movement, modeling additional aspects of muscle anatomy, physiology, and motor control will be necessary. Muscle anatomical properties can be drawn from literature values, including physiological cross-sectional area and optimal fiber length [22], pennation angle [50–53], and tendon slack length [54,55]. Likewise, physiological properties such as passive and active force-length, series elasticity, force-velocity, force-activation, maximum isometric muscle force, and optimal muscle fiber length can be modeled based on experimental measurements [9,56,57]. Muscle activity from experimental measurements can be used to predict joint loading and to better understand joint function during specific tasks [58,59]. The muscle attachments we report will allow for dynamic models of tasks involving many fingers or the entire hand.

Addition of finger muscles to the OpenSim upper-extremity model will facilitate efforts to use the model to better characterize dexterous finger movements. Coupled with experimental data, the model will allow estimates of internal musculoskeletal loading during multi-touch tasks involving many fingers. The model will therefore aid understanding of complex multitouch and gestural movements, and potentially guide the design of technologies that reduce injury risk.

## Acknowledgments

We thank Aaron Kociolek and Alexander MacIntosh (McMaster University) and Christopher Gatti, (Rensselaer Polytechnic Institute) for sharing their OpenSim models. We wish to express our deep gratitude to Dr. Kai-Nan An (Mayo Clinic College of Medicine) and Dr. John Z. Wu (National Institute for Occupational Safety) for helpful discussions regarding coordinate transformation. We thank Dr. Joel Martin (George Mason University) for help calculate FDS moment arms. Moreover, we are grateful to Dr. Panagiotis Artemiadis, Patrick Phelan, Veronica Santos and Huei-Ping Huang for their valuable comments on this study.

![](images/3cdfc3d25a2474033b3900043114516ab9c7dc28045ed826a7baed4b30bbb612.jpg)

Fig 9. Little finger moment arm values (mm). Left MCP curves, PIP and DIP curves represent flex/extension moment arms as a function of flexion (+)/ extension (-). Right MCP curves represent ab/adduction moment arms as a function of abduction (+)/ adduction (-). Solid curves (with plot markers) represent experimentally derived moment arms from anatomical attachment locations, and dotted curves represent optimally estimated moment arms from datadriven optimizations.

doi:10.1371/journal.pone.0121712.g009

Table 3. Mean moment arms (MA) of the index finger.

<table><tr><td>Joint</td><td>RoM</td><td></td><td>FDP</td><td>FDS</td><td>EDC(ES)</td><td>LU(RB)</td><td>RI</td><td>UI(UB)</td></tr><tr><td rowspan="8">MCP2</td><td></td><td>MA</td><td>11.4</td><td>12.4</td><td>-7.4</td><td>3.8</td><td>5.5</td><td>1.9</td></tr><tr><td> $0^{\circ} \sim 90^{\circ}$ </td><td>(VAF)</td><td>(98.9%)</td><td>(99.2%)</td><td>(91.2%)</td><td>(73.0%)</td><td>(57.4%)</td><td>(86.1%)</td></tr><tr><td>Flex/Ext</td><td>RMS</td><td>Δ0.1</td><td>Δ0.1</td><td>Δ0.7</td><td>Δ1.0</td><td>Δ2.4</td><td>Δ0.3</td></tr><tr><td></td><td>(Error)</td><td>(1.0%)</td><td>(0.7%)</td><td>(7.7%)</td><td>(19.8%)</td><td>(42.0%)</td><td>(28.6%)</td></tr><tr><td></td><td>MA</td><td>1.3</td><td>0.7</td><td>-1.0</td><td>6.5</td><td>5.6</td><td>-6.7</td></tr><tr><td> $0^{\circ} \sim 30^{\circ}$ </td><td>(VAF)</td><td>(5.3%)</td><td>(12.7%)</td><td>(89.9%)</td><td>(66.5%)</td><td>(76.6%)</td><td>(88.9%)</td></tr><tr><td>Ab/Add</td><td>RMS</td><td>Δ1.7</td><td>Δ0.9</td><td>Δ0.1</td><td>Δ2.2</td><td>Δ1.3</td><td>Δ0.7</td></tr><tr><td></td><td>(Error)</td><td>(59.5%)</td><td>(53.2%)</td><td>(8.6%)</td><td>(50.4%)</td><td>(31.7%)</td><td>(9.3%)</td></tr><tr><td rowspan="4">PIP2</td><td></td><td>MA</td><td>8.5</td><td>7.4</td><td>-3.6</td><td>-2.6</td><td></td><td>0.6</td></tr><tr><td> $0^{\circ} \sim 90^{\circ}$ </td><td>(VAF)</td><td>(97.7%)</td><td>(89.5%)</td><td>(44.5%)</td><td>(33.5%)</td><td></td><td>(71.8%)</td></tr><tr><td>Flex/Ext</td><td>RMS</td><td>Δ0.2</td><td>Δ0.8</td><td>Δ2.0</td><td>Δ2.2</td><td></td><td>Δ0.3</td></tr><tr><td></td><td>(Error)</td><td>(0.5%)</td><td>(9.8%)</td><td>(32.8%)</td><td>(14.3%)</td><td></td><td>(10.8%)</td></tr><tr><td rowspan="4">DIP2</td><td></td><td>MA</td><td>5.0</td><td></td><td>-3.1</td><td></td><td></td><td></td></tr><tr><td> $0^{\circ} \sim 50^{\circ}$ </td><td>(VAF)</td><td>(88.9%)</td><td></td><td>(58.2%)</td><td></td><td></td><td></td></tr><tr><td>Flex/Ext</td><td>RMS</td><td>Δ0.6</td><td></td><td>Δ1.3</td><td></td><td></td><td></td></tr><tr><td></td><td>(Error)</td><td>(10.7%)</td><td></td><td>(37.4%)</td><td></td><td></td><td></td></tr></table>

All are expressed in millimeters (mm) in OpenSim: flexion(+)/extension(-) and abduction(+)/adduction(-) for the MCP, PIP and DIP joints. (VAF) represents Variance Accounted For, RMS represents root mean square error (Δ) between OpenSim model and experimentally measured values [20], and (Error) represents percentage error, respectively.

doi:10.1371/journal.pone.0121712.t003

Table 4. Mean moment arms (MA) of the middle finger.

<table><tr><td>Joint</td><td>RoM</td><td></td><td>FDP</td><td>FDS</td><td>EDC(ES)</td><td>LU(RB)</td><td>RI</td><td>UI(UB)</td></tr><tr><td rowspan="8">MCP3</td><td></td><td>MA</td><td>11.8</td><td>15.7</td><td>-6.7</td><td>4.0</td><td>8.0</td><td>2.2</td></tr><tr><td> $0^{\circ} \sim 90^{\circ}$ </td><td>(VAF)</td><td>(83.6%)</td><td>(88.7%)</td><td>(65.3%)</td><td>(75.3%)</td><td>(81.5%)</td><td>(88.0%)</td></tr><tr><td>Flex/Ext</td><td>RMS</td><td> $\Delta 2.0$ </td><td> $\Delta 1.8$ </td><td> $\Delta 2.4$ </td><td> $\Delta 1.0$ </td><td> $\Delta 1.5$ </td><td> $\Delta 0.3$ </td></tr><tr><td></td><td>(Error)</td><td>(10.8%)</td><td>(6.9%)</td><td>(23.1%)</td><td>(23.0%)</td><td>(24.7%)</td><td>(6.2%)</td></tr><tr><td></td><td>MA</td><td>-0.1</td><td>0.3</td><td>0.4</td><td>3.9</td><td>8.1</td><td>-3.8</td></tr><tr><td> $0^{\circ} \sim 30^{\circ}$ </td><td>(VAF)</td><td>(8.9%)</td><td>(4.9%)</td><td>(29.8%)</td><td>(61.6%)</td><td>(63.2%)</td><td>(28.8%)</td></tr><tr><td>Ab/Add</td><td>RMS</td><td> $\Delta 2.6$ </td><td> $\Delta 2.0$ </td><td> $\Delta 0.6$ </td><td> $\Delta 1.5$ </td><td> $\Delta 3.0$ </td><td> $\Delta 2.7$ </td></tr><tr><td></td><td>(Error)</td><td>(84.8%)</td><td>(36.7%)</td><td>(51.6%)</td><td>(54.2%)</td><td>(56.4%)</td><td>(69.4%)</td></tr><tr><td rowspan="4">PIP3</td><td></td><td>MA</td><td>7.9</td><td>6.5</td><td>-4.2</td><td>-1.1</td><td></td><td>-0.2</td></tr><tr><td> $0^{\circ} \sim 90^{\circ}$ </td><td>(VAF)</td><td>(94.5%)</td><td>(98.8%)</td><td>(42.6%)</td><td>(87.3%)</td><td></td><td>(86.0%)</td></tr><tr><td>Flex/Ext</td><td>RMS</td><td> $\Delta 0.4$ </td><td> $\Delta 0.1$ </td><td> $\Delta 2.4$ </td><td> $\Delta 0.1$ </td><td></td><td> $\Delta 0.3$ </td></tr><tr><td></td><td>(Error)</td><td>(3.2%)</td><td>(0.7%)</td><td>(52.5%)</td><td>(1.1%)</td><td></td><td>(67.3%)</td></tr><tr><td rowspan="4">DIP3</td><td></td><td>MA</td><td>5.2</td><td></td><td>-2.8</td><td></td><td></td><td></td></tr><tr><td> $0^{\circ} \sim 50^{\circ}$ </td><td>(VAF)</td><td>(90.6%)</td><td></td><td>(72.2%)</td><td></td><td></td><td></td></tr><tr><td>Flex/Ext</td><td>RMS</td><td> $\Delta 0.5$ </td><td></td><td> $\Delta 0.8$ </td><td></td><td></td><td></td></tr><tr><td></td><td>(Error)</td><td>(9.7%)</td><td></td><td>(22.6%)</td><td></td><td></td><td></td></tr></table>

All are expressed in millimeters (mm) in OpenSim: flexion(+)/extension(-) and abduction(+)/adduction(-) for the MCP, PIP and DIP joints. (VAF) represents Variance Accounted For, RMS represents root mean square error (Δ) between OpenSim model and experimentally measured values [20], and (Error) represents percentage error, respectively.  
doi:10.1371/journal.pone.0121712.t004

Table 5. Mean moment arms (MA) of the ring finger.

<table><tr><td>Joint</td><td>RoM</td><td></td><td>FDP</td><td>FDS</td><td>EDC(ES)</td><td>LU(RB)</td><td>RI</td><td>UI(UB)</td></tr><tr><td rowspan="8">MCP4</td><td></td><td>MA</td><td>9.1</td><td>9.7</td><td>-5.8</td><td>3.0</td><td>5.4</td><td>1.7</td></tr><tr><td> $0^{\circ} \sim 90^{\circ}$ </td><td>(VAF)</td><td>(83.0%)</td><td>(87.2%)</td><td>(93.3%)</td><td>(75.3%)</td><td>(91.5%)</td><td>(89.1%)</td></tr><tr><td>Flex/Ext</td><td>RMS</td><td>Δ1.6</td><td>Δ1.3</td><td>Δ0.4</td><td>Δ0.7</td><td>Δ0.5</td><td>Δ0.2</td></tr><tr><td></td><td>(Error)</td><td>(9.6%)</td><td>(11.3%)</td><td>(1.1%)</td><td>(1.9%)</td><td>(3.8%)</td><td>(2.9%)</td></tr><tr><td></td><td>MA</td><td>-2.1</td><td>0.1</td><td>3.2</td><td>4.0</td><td>2.8</td><td>-3.8</td></tr><tr><td> $0^{\circ} \sim 30^{\circ}$ </td><td>(VAF)</td><td>(30.0%)</td><td>(84.1%)</td><td>(27.3%)</td><td>(1.6%)</td><td>(65.6%)</td><td>(53.4%)</td></tr><tr><td>Ab/Add</td><td>RMS</td><td>Δ1.6</td><td>Δ1.6</td><td>Δ2.4</td><td>Δ4.0</td><td>Δ1.0</td><td>Δ1.8</td></tr><tr><td></td><td>(Error)</td><td>(38.8%)</td><td>(75.1%)</td><td>(71.8%)</td><td>(96.8%)</td><td>(5.1%)</td><td>(60.2%)</td></tr><tr><td rowspan="4">PIP4</td><td></td><td>MA</td><td>6.2</td><td>5.0</td><td>-2.9</td><td>-0.4</td><td></td><td>-0.7</td></tr><tr><td> $0^{\circ} \sim 90^{\circ}$ </td><td>(VAF)</td><td>(95.5%)</td><td>(95.0%)</td><td>(93.4%)</td><td>(84.1%)</td><td></td><td>(93.2%)</td></tr><tr><td>Flex/Ext</td><td>RMS</td><td>Δ0.3</td><td>Δ0.3</td><td>Δ1.4</td><td>Δ0.1</td><td></td><td>Δ0.0</td></tr><tr><td></td><td>(Error)</td><td>(2.0%)</td><td>(4.6%)</td><td>(48.2%)</td><td>(16.3%)</td><td></td><td>(6.2%)</td></tr><tr><td rowspan="4">DIP4</td><td></td><td>MA</td><td>3.3</td><td></td><td>-1.7</td><td></td><td></td><td></td></tr><tr><td> $0^{\circ} \sim 50^{\circ}$ </td><td>(VAF)</td><td>(96.4%)</td><td></td><td>(77.8%)</td><td></td><td></td><td></td></tr><tr><td>Flex/Ext</td><td>RMS</td><td>Δ0.1</td><td></td><td>Δ0.4</td><td></td><td></td><td></td></tr><tr><td></td><td>(Error)</td><td>(3.5%)</td><td></td><td>(3.0%)</td><td></td><td></td><td></td></tr></table>

All are expressed in millimeters (mm) in OpenSim: flexion(+)/extension(-) and abduction(+)/adduction(-) for the MCP, PIP and DIP joints. (VAF) represents Variance Accounted For, RMS represents root mean square error (Δ) between OpenSim model and experimentally measured values [20], and (Error) represents percentage error, respectively.  
doi:10.1371/journal.pone.0121712.t005

Table 7. Index finger muscle-tendon locations, expressed in OpenSim frame (mm).

<table><tr><td>Joint</td><td>Muscle</td><td>x</td><td>y</td><td>z</td><td>ΔR</td><td>x</td><td>y</td><td>z</td><td>ΔR</td></tr><tr><td></td><td></td><td colspan="3">proximal point (secondmc)</td><td></td><td colspan="3">distal point (proxph2)</td><td></td></tr><tr><td></td><td>FDP</td><td>4.11(2.76)</td><td>-16.00(-21.47)</td><td>-4.21(-4.02)</td><td>5.6</td><td>3.34(1.00)</td><td>-20.24(-13.35)</td><td>-5.03(-9.54)</td><td>9.0</td></tr><tr><td></td><td>FDS</td><td>4.86(1.65)</td><td>-13.77(-21.47)</td><td>-0.66(-5.83)</td><td>9.8</td><td>1.32(-0.83)</td><td>-8.41(-13.35)</td><td>-12.01(-12.12)</td><td>5.4</td></tr><tr><td></td><td>RI</td><td>9.07(10.45)</td><td>-19.95(-19.21)</td><td>-4.88(-1.11)</td><td>4.1</td><td>7.04(7.20)</td><td>-6.72(-18.37)</td><td>-0.15(0.50)</td><td>11.7</td></tr><tr><td>MCP2</td><td>LU</td><td>10.17(10.63)</td><td>-26.47(-19.21)</td><td>-0.01(-7.10)</td><td>10.2</td><td>8.38(6.69)</td><td>-8.29(-18.37)</td><td>0.04(-3.66)</td><td>10.9</td></tr><tr><td></td><td>UI</td><td>-3.32(-4.36)</td><td>-29.39(-19.21)</td><td>-0.12(-0.39)</td><td>10.2</td><td>-4.31(-6.96)</td><td>-15.93(-18.37)</td><td>2.41(-1.68)</td><td>5.5</td></tr><tr><td></td><td>EDC(LE)</td><td>3.05(2.96)</td><td>-29.51(-28.24)</td><td>12.43(15.05)</td><td>2.9</td><td>3.31(0.33)</td><td>-7.11(-10.84)</td><td>11.64(11.42)</td><td>4.8</td></tr><tr><td></td><td></td><td colspan="3">proximal point (proxph2)</td><td></td><td colspan="3">distal point (midph2)</td><td></td></tr><tr><td></td><td>FDP</td><td>6.33(8.01)</td><td>-34.33(-36.25)</td><td>0.61(9.35)</td><td>3.3</td><td>0.61(0.68)</td><td>-8.81(-8.89)</td><td>-5.84(-7.66)</td><td>1.8</td></tr><tr><td></td><td>RB(LU)</td><td>11.30(10.79)</td><td>-33.93(-40.02)</td><td>9.35(9.86)</td><td>6.1</td><td>5.80(5.20)</td><td>-6.86(-6.38)</td><td>8.79(5.50)</td><td>3.3</td></tr><tr><td></td><td>UB</td><td>1.00</td><td>-39.44</td><td>11.14</td><td>3.3</td><td>-4.08</td><td>-6.55</td><td>3.75</td><td>3.5</td></tr><tr><td>PIP2</td><td>(UI)</td><td>(3.16)</td><td>(-40.02)</td><td>(8.78)</td><td></td><td>(-7.61)</td><td>(-6.38)</td><td>(3.74)</td><td></td></tr><tr><td></td><td>FDS</td><td>6.35(7.55)</td><td>-34.33(-36.25)</td><td>1.61(3.64)</td><td>3.0</td><td>-0.24(0.13)</td><td>-8.81(-8.89)</td><td>-4.08(-6.02)</td><td>2.0</td></tr><tr><td></td><td>ES(EDC)</td><td>6.88(7.02)</td><td>-38.72(-41.27)</td><td>10.95(10.84)</td><td>2.6</td><td>1.86(-0.90)</td><td>-0.10(-4.62)</td><td>7.41(7.35)</td><td>5.3</td></tr><tr><td></td><td></td><td colspan="3">proximal point (midph2)</td><td></td><td colspan="3">distal point (distph2)</td><td></td></tr><tr><td></td><td>TE</td><td>2.83</td><td>-24.89</td><td>4.82</td><td>0.0</td><td>-0.48</td><td>-5.62</td><td>3.45</td><td>2.0</td></tr><tr><td>DIP2</td><td>(EDC)</td><td>(2.83)</td><td>(-24.89)</td><td>(4.82)</td><td></td><td>(-0.48)</td><td>(-5.62)</td><td>(5.45)</td><td></td></tr><tr><td></td><td>FDP</td><td>4.32(4.32)</td><td>-19.62(-19.62)</td><td>-3.52(-3.52)</td><td></td><td>-0.70(-0.70)</td><td>-5.62(-5.62)</td><td>-3.99(-3.99)</td><td>0.0</td></tr></table>

The coordinate system of the OpenSim model is attached to metacarpal (secondmc), proximal (proxph2), middle (midph2) and distal (distph2) phalanges. x, y and z components indicate radioulnar (+ points out, perpendicular to the palm plan), axial (+ points from distal to proximal side) and dorsolar (+ points up, from palm to hand side), respectively. Optimized muscle-tendon locations of OpenSim are compared with those of experimental measurements (numbers in parentheses). ΔR is the Euclidean distance between OpenSim model and experimentally measured attachment sites [5].

Table 8. Middle finger muscle-tendon locations, expressed in OpenSim frame (mm).

<table><tr><td>Joint</td><td>Muscle</td><td>x</td><td>y</td><td>z</td><td>ΔR</td><td>x</td><td>y</td><td>z</td><td>ΔR</td></tr><tr><td></td><td></td><td colspan="3">proximal point (thirdmc)</td><td></td><td colspan="3">distal point (proxph3)</td><td></td></tr><tr><td></td><td>FDP</td><td>1.86(0.53)</td><td>-15.09(-19.14)</td><td>-6.77(-3.07)</td><td>5.6</td><td>0.69(0.61)</td><td>-15.83(-14.58)</td><td>-8.10(-8.75)</td><td>1.4</td></tr><tr><td></td><td>FDS</td><td>2.24(0.69)</td><td>-10.87(-19.14)</td><td>-13.45(-5.55)</td><td>11.5</td><td>2.44(-1.03)</td><td>-19.19(-14.58)</td><td>-8.01(-10.41)</td><td>6.2</td></tr><tr><td></td><td>RI</td><td>9.91(6.88)</td><td>-24.00(-16.75)</td><td>-4.96(-5.34)</td><td>7.9</td><td>7.70(5.26)</td><td>-19.91(-19.88)</td><td>0.05(-1.01)</td><td>2.7</td></tr><tr><td>MCP3</td><td>LU</td><td>10.25(6.18)</td><td>-23.17(-16.75)</td><td>-0.83(-5.34)</td><td>9.6</td><td>6.84(5.22)</td><td>-21.96(-19.88)</td><td>7.26(-3.71)</td><td>11.3</td></tr><tr><td></td><td>UI</td><td>-3.11(-4.98)</td><td>-24.84(-16.75)</td><td>-3.45(-14.43)</td><td>13.8</td><td>-1.19(-5.68)</td><td>-18.29(-19.88)</td><td>-9.33(0.95)</td><td>11.3</td></tr><tr><td></td><td>EDC(LE)</td><td>-0.69(-0.69)</td><td>-26.29(-26.29)</td><td>10.04(11.15)</td><td>1.1</td><td>-1.18(-0.48)</td><td>-12.22(-11.93)</td><td>10.74(8.53)</td><td>2.3</td></tr><tr><td></td><td></td><td colspan="3">proximal point (proxph3)</td><td></td><td colspan="3">distal point (midph3)</td><td></td></tr><tr><td></td><td>FDP</td><td>2.38(1.60)</td><td>-36.43(-38.91)</td><td>0.54(2.71)</td><td>3.4</td><td>1.58(-0.11)</td><td>-11.49(-10.34)</td><td>-5.12(-7.23)</td><td>2.9</td></tr><tr><td></td><td>RB(LU)</td><td>5.79(4.86)</td><td>-44.69(-42.89)</td><td>7.56(7.56)</td><td>2.0</td><td>4.92(6.20)</td><td>-11.61(-7.69)</td><td>3.54(4.56)</td><td>4.5</td></tr><tr><td></td><td>UB</td><td>-4.26(-2.05)</td><td>-42.83(-42.89)</td><td>6.94(7.24)</td><td>2.2</td><td>-5.36(-6.55)</td><td>-7.74(-7.69)</td><td>5.97(4.27)</td><td>1.2</td></tr><tr><td>PIP3</td><td>(UI)</td><td>4.54(1.44)</td><td>-37.60(-38.91)</td><td>2.13(3.35)</td><td>3.6</td><td>0.23(0.03)</td><td>-10.75(-10.34)</td><td>-5.41(-5.49)</td><td>4.6</td></tr><tr><td></td><td>ES(EDC)</td><td>1.69(1.40)</td><td>-41.39(-44.21)</td><td>11.27(9.36)</td><td>3.4</td><td>0.39(-0.64)</td><td>-4.23(-3.71)</td><td>6.25(6.55)</td><td>3.9</td></tr><tr><td></td><td></td><td colspan="3">proximal point (midph3)</td><td></td><td colspan="3">distal point (distph3)</td><td></td></tr><tr><td></td><td>TE</td><td>1.09(1.09)</td><td>-29.05(-29.05)</td><td>5.07(5.07)</td><td>0.0</td><td>-0.58(-0.58)</td><td>-6.94(-5.83)</td><td>3.08(4.08)</td><td>1.5</td></tr><tr><td>DIP3</td><td>(EDC)</td><td>2.13(1.98)</td><td>-21.44(-23.48)</td><td>-2.13(-2.76)</td><td>0.0</td><td>0.06(-0.85)</td><td>-5.01(-3.18)</td><td>-4.19(-3.74)</td><td>2.0</td></tr></table>

Parentheses () values indicate measured muscle-tendon locations [5], transformed to OpenSim coordinate. ΔR is the Euclidean distance between OpenSim model and experimentally measured attachment sites [5].

doi:10.1371/journal.pone.0121712.t008

Table 9. Ring finger muscle-tendon locations, expressed in OpenSim frame (mm).

<table><tr><td>Joint</td><td>Muscle</td><td>X</td><td>y</td><td>z</td><td>ΔR</td><td>x</td><td>y</td><td>z</td><td>ΔR</td></tr><tr><td></td><td></td><td colspan="3">proximal point (fourthmc)</td><td></td><td colspan="3">distal point (proxph4)</td><td></td></tr><tr><td></td><td>FDP</td><td>-0.62(-2.57)</td><td>-13.64(-16.97)</td><td>-6.29(-7.76)</td><td>4.1</td><td>-1.22(1.00)</td><td>-11.97(-10.96)</td><td>-8.80(-9.46)</td><td>2.5</td></tr><tr><td></td><td>FDS</td><td>2.02(-3.67)</td><td>-13.83(-16.97)</td><td>-6.02(-9.55)</td><td>6.7</td><td>0.08(-0.82)</td><td>-12.68(-10.96)</td><td>-7.47(-12.03)</td><td>5.0</td></tr><tr><td></td><td>RI</td><td>1.63(7.18)</td><td>-17.90(-14.73)</td><td>-9.78(-4.86)</td><td>8.1</td><td>9.87(7.14)</td><td>-16.29(-15.94)</td><td>0.53(0.50)</td><td>2.8</td></tr><tr><td>MCP4</td><td>LU</td><td>3.17(7.36)</td><td>-22.42(-14.73)</td><td>-0.81(-10.80)</td><td>13.3</td><td>0.17(6.63)</td><td>-14.60(-15.94)</td><td>0.62(-3.64)</td><td>7.9</td></tr><tr><td></td><td>UI</td><td>-4.85(-7.51)</td><td>-24.09(-14.73)</td><td>-4.39(-4.15)</td><td>9.7</td><td>-4.15(-6.90)</td><td>-21.18(-15.94)</td><td>-5.52(-1.67)</td><td>9.3</td></tr><tr><td></td><td>EDC(LE)</td><td>-0.42(-2.37)</td><td>-20.69(-23.69)</td><td>9.47(11.17)</td><td>4.0</td><td>-1.99(-0.32)</td><td>-8.36(-8.47)</td><td>8.76(11.33)</td><td>3.1</td></tr><tr><td></td><td></td><td colspan="3">proximal point (proxph4)</td><td></td><td colspan="3">distal point (midph4)</td><td></td></tr><tr><td></td><td>FDP</td><td>-3.44(-3.04)</td><td>-33.34(-35.28)</td><td>-1.92(-3.32)</td><td>2.4</td><td>2.32(0.67)</td><td>-7.12(-7.84)</td><td>-6.00(-7.59)</td><td>2.4</td></tr><tr><td></td><td>RB(LU)</td><td>1.28(-0.28)</td><td>-40.33(-39.01)</td><td>4.10(3.86)</td><td>2.1</td><td>4.27(5.15)</td><td>-4.51(-5.35)</td><td>4.30(5.45)</td><td>1.6</td></tr><tr><td></td><td>UB</td><td>-8.87(-7.85)</td><td>-38.84(-39.01)</td><td>1.83(2.79)</td><td>1.6</td><td>-5.76(-7.55)</td><td>-4.01(-5.35)</td><td>3.39(3.71)</td><td>2.3</td></tr><tr><td>PIP4</td><td>(UI)</td><td>0.56(-3.49)</td><td>-36.10(-35.28)</td><td>-0.99(2.32)</td><td>4.3</td><td>0.04(0.12)</td><td>-11.94(-7.84)</td><td>-1.77(-5.98)</td><td>5.9</td></tr><tr><td></td><td>ES(EDC)</td><td>-3.85(-4.03)</td><td>-38.00(-40.26)</td><td>4.84(4.83)</td><td>2.5</td><td>-1.94(-0.90)</td><td>-5.82(-3.61)</td><td>8.82(7.30)</td><td>3.5</td></tr><tr><td></td><td></td><td colspan="3">proximal point (midph4)</td><td></td><td colspan="3">distal point (distph4)</td><td></td></tr><tr><td></td><td>TE</td><td>0.49(-2.69)</td><td>-23.94(-24.79)</td><td>3.56(4.14)</td><td>3.3</td><td>-0.39(-0.47)</td><td>-2.72(-3.74)</td><td>3.58(5.40)</td><td>1.0</td></tr><tr><td>DIP4</td><td>(EDC)</td><td>-0.70(-1.34)</td><td>-19.70(-19.56)</td><td>-1.31(-4.14)</td><td>2.9</td><td>0.33(0.70)</td><td>-4.04(-3.74)</td><td>-3.05(-3.96)</td><td>1.4</td></tr></table>

Parentheses () values indicate measured muscle-tendon locations [5], transformed to OpenSim coordinate. ΔR is the Euclidean distance between OpenSim model and experimentally measured attachment sites [5].

doi:10.1371/journal.pone.0121712.t009

Table 10. Little finger muscle-tendon locations, expressed in OpenSim frame (mm).

<table><tr><td>Joint</td><td>Muscle</td><td>x</td><td>y</td><td>z</td><td>ΔR</td><td>x</td><td>y</td><td>z</td><td>ΔR</td></tr><tr><td></td><td></td><td colspan="3">proximal point (fifthmc)</td><td></td><td colspan="3">distal point (proxph5)</td><td></td></tr><tr><td></td><td>FDP</td><td>0.95(-2.22)</td><td>-13.05(-15.76)</td><td>-8.70(-7.89)</td><td>4.3</td><td>-0.95(0.84)</td><td>-13.66(-12.39)</td><td>-5.11(-7.98)</td><td>3.6</td></tr><tr><td></td><td>FDS</td><td>-0.10(-3.15)</td><td>-13.08(-15.76)</td><td>-9.52(-9.40)</td><td>4.1</td><td>-1.67(-0.69)</td><td>-14.93(-12.39)</td><td>-8.12(-10.14)</td><td>3.4</td></tr><tr><td></td><td>RI</td><td>6.11(6.03)</td><td>-14.87(-13.87)</td><td>-6.60(-5.45)</td><td>1.5</td><td>4.72(6.02)</td><td>-14.34(-16.59)</td><td>2.33(0.42)</td><td>3.2</td></tr><tr><td>MCP5</td><td>LU</td><td>6.06(6.17)</td><td>-19.91(-13.87)</td><td>-0.17(-10.46)</td><td>11.7</td><td>3.51(5.59)</td><td>-13.13(-16.59)</td><td>1.96(-3.07)</td><td>5.2</td></tr><tr><td></td><td>UI</td><td>-5.13(-6.37)</td><td>-19.67(-13.87)</td><td>-7.63(-4.85)</td><td>6.5</td><td>-3.07(-5.82)</td><td>-17.71(-16.59)</td><td>-7.93(-1.41)</td><td>7.2</td></tr><tr><td></td><td>EDC(LE)</td><td>-1.26(-2.05)</td><td>-18.18(-21.43)</td><td>5.66(8.07)</td><td>4.1</td><td>-4.45(-0.27)</td><td>-12.53(-10.29)</td><td>6.29(9.56)</td><td>5.8</td></tr><tr><td></td><td></td><td colspan="3">proximal point (proxph5)</td><td></td><td colspan="3">distal point (midph5)</td><td></td></tr><tr><td></td><td>FDP</td><td>-0.46(-4.87)</td><td>-29.14(-31.36)</td><td>-8.05(-6.82)</td><td>5.1</td><td>0.87(0.57)</td><td>-8.31(-7.46)</td><td>-7.69(-6.41)</td><td>1.6</td></tr><tr><td></td><td>RB(LU)</td><td>-3.16(-2.54)</td><td>-37.20(-34.51)</td><td>3.14(-0.76)</td><td>4.9</td><td>1.39(4.35)</td><td>-7.41(-5.36)</td><td>2.74(4.60)</td><td>4.1</td></tr><tr><td></td><td>UB</td><td>-10.71(-8.92)</td><td>-35.88(-34.51)</td><td>-3.16(-1.66)</td><td>2.7</td><td>-3.89(-6.36)</td><td>-7.41(-5.36)</td><td>2.74(3.13)</td><td>7.2</td></tr><tr><td>PIP5</td><td>(UI)</td><td>-2.51(-5.25)</td><td>-25.87(-31.36)</td><td>-5.41(-5.97)</td><td>6.2</td><td>2.11(0.11)</td><td>-5.50(-7.46)</td><td>-6.96(-5.04)</td><td>3.4</td></tr><tr><td></td><td>ES(EDC)</td><td>-7.94(-5.70)</td><td>-32.24(-35.56)</td><td>2.51(0.06)</td><td>4.7</td><td>-2.68(-0.76)</td><td>-7.36(-3.89)</td><td>4.66(6.15)</td><td>4.2</td></tr><tr><td></td><td></td><td colspan="3">proximal point (midph5)</td><td></td><td colspan="3">distal point (distph5)</td><td></td></tr><tr><td></td><td>TE</td><td>0.20(-3.17)</td><td>-20.00(-20.68)</td><td>6.42(0.77)</td><td>6.6</td><td>1.48(-0.39)</td><td>-13.95(-4.41)</td><td>4.70(4.56)</td><td>9.7</td></tr><tr><td>DIP5</td><td>(EDC)</td><td>-1.01(-2.05)</td><td>-16.01(-16.27)</td><td>-4.87(-6.21)</td><td>3.3</td><td>0.19(0.59)</td><td>-11.35(-4.41)</td><td>-4.68(-3.34)</td><td>7.1</td></tr></table>

Parentheses () values indicate measured muscle-tendon locations [5], transformed to OpenSim coordinate. ΔR is the Euclidean distance between OpenSim model and experimentally measured attachment sites [5].

doi:10.1371/journal.pone.0121712.t010

## Author Contributions

Conceived and designed the experiments: JHL DLJ DSA JTD. Performed the experiments: JHL. Analyzed the data: JHL DLJ DSA. Contributed reagents/materials/analysis tools: DLJ. Wrote the paper: JHL DLJ DSA JTD.

## References

1. Brochier T, Spinks RL, Umilta MA, Lemon RN (2004) Patterns of muscle activity underlying object-specific grasp by the macaque monkey. J Neurophysiol 92: 1770–1782. PMID: 15163676

2. Shin G, Zhu X (2011) User discomfort, work posture and muscle activity while using a touchscreen in a desktop PC setting. Ergonomics 54: 733–744. doi: 10.1080/00140139.2011.592604 PMID: 21846311

3. Dennerlein JT, Diao E, Mote CD Jr., Rempel DM (1998) Tensions of the flexor digitorum superficialis are higher than a current model predicts. J Biomech 31: 295–301. PMID: 9672082

4. Dennerlein JT, Diao E, Mote CD Jr., Rempel DM (1999) in vivo flexor tendon force while tapping on a keyswitch. Journal of orthopaedic research 17: 178–184. PMID: 10221833

5. An KN, Chao EY, Cooney WP, Linscheid RL (1979) Normative model of human hand for biomechanica analysis. J Biomech 12: 775–788. PMID: 489628

6. Biggs J, Horch K (1999) A three-dimensional kinematic model of the human long finger and the muscles that actuate it. Medical Engineering & Physics 21: 625–639.

7. Sancho-Bru JL, Pèrez-González A, Vergara-Monedoro M, Giurintano D (2001) A 3-D dynamic model of human finger for studying free movements. Journal of Biomechanics 34: 1491–1500. PMID: 11672724

8. Brook N, Mizrahi J, Shoham M, Dayan J (1995) A biomechanical model of index finger dynamics. Med Eng Phys 17: 54–63. PMID: 7704345

9. Holzbaur KR, Murray WM, Delp SL (2005) A model of the upper extremity for simulating musculoskele tal surgery and analyzing neuromuscular control. Ann Biomed Eng 33: 829–840. PMID: 16078622

10. Cholewicki J, McGill SM (1994) EMG assisted optimization: a hybrid approach for estimating muscle forces in an indeterminate biomechanical model, J Biomech 27: 1287–1289, PMID: 7962016

11. Spoor CW, Landsmeer JMF (1976) Analysis of the zigzag movement of the human finger under influence of the extensor digitorum tendon and the deep flexor tendon. J Biomech 9: 561–566. PMID: 965421

12. Leijnse JNAL (1995) A two-dimensional kinematic model of the lumbrical in the human finger. J Biomech 28: 237–249. PMID: 7730384

13. Valero-Cuevas FJ, Johanson ME, Towles JD (2003) Towards a realistic biomechanical model of the thumb:the choice of kinematic description may be more critical than the solution method or the variability/uncertainty of musculoskeletal parameters. J Biomech 36: 1019–1030. PMID: 12757811

14. Valero-Cuevas FJ, Zajac FE, Burgar CG (1998) Large index-fingertip forces are produced by subjectindependent patterns of muscle excitation. J Biomech 31: 693–703. PMID: 9796669

15. Buchner HJ, Hines MJ, Hemami H (1988) A dynamic model for finger interphalangeal coordination. J Biomech 21: 459–468. PMID: 3209591

16. Lee JW, Rim K (1990) Maximum finger force prediction using a planar simulation of the middle finger. Proc Inst Mech Eng H 204: 169–178. PMID: 2133783

17. Lopes MM, Lawson W, Scott T, Keir PJ (2011) Tendon and nerve excursion in the carpal tunnel in healthy and CTD wrists. Clin Biomech (Bristol, Avon) 26: 930–936. doi: 10.1016/j.clinbiomech.2011. 03.014 PMID: 21550703

18. Oh S, Belohlavek M, Zhao C, Osamura N, Zobitz ME, An KN, et al. (2007) Detection of differential gliding characteristics of the flexor digitorum superficialis tendon and subsynovial connective tissue using color Doppler sonographic imaging. J Ultrasound Med 26: 149–155. PMID: 17255175

19. Yoshii Y, Henderson J, Villarraga HR, Zhao C, An KN, Amadio PC (2011) Ultrasound assessment of the motion patterns of human flexor digitorum superficialis and profundus tendons with speckle track ing. J Orthop Res 29: 1465–1469. doi: 10.1002/jor.21428 PMID: 21469183

20. An KN, Ueba Y, Chao EY, Cooney WP, Linscheid RL (1983) Tendon excursion and moment arm of index finger muscles. J Biomech 16: 419–425. PMID: 6619158

21. Armstrong TJ, Chaffin DB (1978) An investigation of the relationship between displacements of the finger and wrist joints and the extrinsic finger flexor tendons. J Biomech 11: 119–128. PMID: 670246

22. Brand PW, Hollister AM (1999) Clinical mechanics of the hand. St. Louis: Mosby Year Book. 369 p.

23. Chao EYS, An K-N, Clooney WP III, Linscheid RL (1989) Biomechanics of the hand: a basic research study. Singapore: World Scientific.

24. Franko OI, Winters TM, Tirrell TF, Hentzen ER, Lieber RL (2011) Moment arms of the human digital flexors. J Biomech 44: 1987–1990. doi: 10.1016/j.jbiomech.2011.04.025 PMID: 21561624

25. Lee JH, Asakawa DS, Dennerlein JT, Jindrich DL (2014) Extrinsic and Intrinsic Index Finger Muscle Attachments in an OpenSim Upper-Extremity Model. Ann Biomed Eng.

26. Delp SL, Anderson FC, Arnold AS, Loan P, Habib A, John CT, et al. (2007) OpenSim: open-source software to create and analyze dynamic simulations of movement. IEEE Trans Biomed Eng 54: 1940– 1950. PMID: 18018689

27. Delp SL, Loan JP (1995) A software system to develop and analyze models of musculoskeletal structures. Computers in Biology and Medicine 25: 21–34. PMID: 7600758

28. Kane TR, Levinson DA (1985) Dynamics, Theory and Applications: McGraw Hill.

29. Arnold AS, Salinas S, Asakawa DJ, Delp SL (2000) Accuracy of muscle moment arms estimated from MRI-based musculoskeletal models of the lower extremity. Comput Aided Surg 5: 108–119. PMID: 10862133

30. Rankin JW, Neptune RR (2012) Musculotendon lengths and moment arms for a three-dimensional upper-extremity model. J Biomech 45: 1739–1744. doi: 10.1016/j.jbiomech.2012.03.010 PMID: 22520587

31. Saul KR, Hu X, Goehler CM, Vidt ME, Daly M, Velisar A, et al. (2014) Benchmarking of dynamic simulation predictions in two software platforms using an upper limb musculoskeletal model. Comput Methods Biomech Biomed Engin: 1–14.

32. Hollister A, Buford WL, Myers LM, Giurintano DJ, Novick A (1992) The axes of rotation of the thumb carpometacarpal joint. J Orthop Res 10: 454–460. PMID: 1569508

33. Hollister A, Giurintano DJ, Buford WL, Myers LM, Novick A (1995) The axes of rotation of the thumb interphalangeal and metacarpophalangeal joints. Clin Orthop Relat Res: 188–193.

34. Wohlman SJ, Murray WM (2013) Bridging the gap between cadaveric and in vivo experiments: a biomechanical model evaluating thumb-tip endpoint forces. J Biomech 46: 1014–1020. doi: 10.1016/j. jbiomech.2012.10.044 PMID: 23332233

35. Kociolek AM, Keir PJ (2011) Modelling tendon excursions and moment arms of the finger flexors: anatomic fidelity versus function. J Biomech 44: 1967–1973. doi: 10.1016/j.jbiomech.2011.05.002 PMID: 21596382

36. Fowler NK, Nicol AC, Condon B, Hadley D (2001) Method of determination of three dimensional index finger moment arms and tendon lines of action using high resolution MRI scans. J Biomech 34: 791– 797. PMID: 11470117

37. Gorden CC, Churchill T, Clauser CE, Bradtmiller B, McConville JT, Tebbetts I, et al. (1989) Anthropometic survey of U.S. army personnel: Summary statistics interim report. Natick, MA: United State Army Natick Research, Development and Engineering Center.

38. Greiner TM (1991) Hand Anthropometry of US Army Personnel. Natick, MA: United States Natick Research, Development and Engineering Center.

39. Li Z, Chang CC, Dempsey PG, Ouyang L, Duan J (2008) Validation of a three-dimensional hand scanning and dimension extraction method with dimension data. Ergonomics 51: 1672–1692. doi: 10.1080/ 00140130802287280 PMID: 18941974

40. Martin JR, Latash ML, Zatsiorsky VM (2012) Effects of the index finger position and force production on the flexor digitorum superficialis moment arms at the metacarpophalangeal joints—a magnetic resonance imaging study. Clin Biomech (Bristol, Avon) 27: 453–459. doi: 10.1016/j.clinbiomech.2011.11. 008 PMID: 22192658

41. Kamper DG, Fischer HC, Cruz EG (2006) Impact of finger posture on mapping from muscle activation to joint torque. Clin Biomech (Bristol, Avon) 21: 361–369. PMID: 16406192

42. Kirkpatrick S, Gelatt CD Jr., Vecchi MP (1983) Optimization by simulated annealing. Science 220: 671–680. PMID: 17813860

43. Kelley CT (1999) Iterative Methods for Optimization. Philadelphia: SIAM.

44. Garner BA, Pandy MG (2000) The Obstacle-Set Method for Representing Muscle Paths in Musculoskeletal Models. Comput Methods Biomech Biomed Engin 3: 1–30. PMID: 11264836

45. Gatti CJ, Hughes RE (2009) Optimization of muscle wrapping objects using simulated annealing. Ann Biomed Eng 37: 1342–1347. doi: 10.1007/s10439-009-9710-5 PMID: 19434495

46. Hu D, Howard D, Ren L (2014) Biomechanical analysis of the human finger extensor mechanism during isometric pressing. PLoS One 9: e94533. doi: 10.1371/journal.pone.0094533 PMID: 24732789

47. An KN, Hui FC, Morrey BF, Linscheid RL, Chao EY (1981) Muscles across the elbow joint: a biomechanical analysis. J Biomech 14: 659–669. PMID: 7334026

48. Murray WM, Buchanan TS, Delp SL (2002) Scaling of peak moment arms of elbow muscles with upper extremity bone dimensions. J Biomech 35: 19–26. PMID: 11747879

49. Johnson WL, Jindrich DL, Roy RR, Reggie Edgerton V (2008) A three-dimensional model of the rat hindlimb: musculoskeletal geometry and muscle moment arms. J Biomech 41: 610–619. PMID: 18061600

50. Lieber RL, Fazeli BM, Botte MJ (1990) Architecture of selected wrist flexor and extensor muscles. J Hand Surg Am 15: 244–250. PMID: 2324452

51. Lieber RL, Jacobson MD, Fazeli BM, Abrams RA, Botte MJ (1992) Architecture of selected muscles of the arm and forearm: anatomy and implications for tendon transfer. J Hand Surg Am 17: 787–798. PMID: 1401782

52. Jacobson MD, Raab R, Fazeli BM, Abrams RA, Botte MJ, Lieber RL (1992) Architectural design of the human intrinsic hand muscles. J Hand Surg Am 17: 804–809. PMID: 1401784

53. Wang K, McGlinn EP, Chung KC (2014) A biomechanical and evolutionary perspective on the function of the lumbrical muscle. J Hand Surg Am 39: 149–155. doi: 10.1016/j.jhsa.2013.06.029 PMID: 24369943

54. Garner BA, Pandy MG (2003) Estimation of musculotendon properties in the human upper limb. Ann Biomed Eng 31: 207–220. PMID: 12627828

55. Vilimek M (2007) Musculotendon forces derived by different muscle models. Acta Bioeng Biomech 9: 41–47. PMID: 18421942

56. Hu X, Murray WM, Perreault EJ (2011) Muscle short-range stiffness can be used to estimate the endpoint stiffness of the human arm. J Neurophysiol 105: 1633–1641. doi: 10.1152/jn.00537.2010 PMID: 21289133

57. Wu JZ, An KN, Cutlip RG, Krajnak K, Welcome D, Dong RG (2008) Analysis of musculoskeletal loading in an index finger during tapping. J Biomech 41: 668–676. PMID: 17991473

58. Manal K, Gonzalez RV, Lloyd DG, Buchanan TS (2002) A real-time EMG-driven virtual arm. Comput Biol Med 32: 25–36. PMID: 11738638

59. Kinney AL, Besier TF, D'Lima DD, Fregly BJ (2013) Update on grand challenge competition to predict in vivo knee loads. J Biomech Eng 135: 021012. doi: 10.1115/1.4023255 PMID: 23445057