# A Real-time, 3D Musculoskeletal Model for Dynamic Simulation of Arm Movements - PMC

> **这是网页快照，不是同行评审论文。**
> 来源：<https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2971671>
> 抓取时间：2026-10-01 20:38
> HTTP 状态：200　｜　原始字节：180,515　｜　SHA256：`969799a57da2e70f`
> **用途**：本条在语料四中的证据等级为「第三方实测」。引用时必须标注为网页型证据，
> 不得与同行评审论文并列。抓取日期之后页面可能变化，引用前请核对原链接。

**页面描述**：Neuroprostheses can be used to restore movement of the upper limb in individuals with high-level spinal cord injury. Development and evaluation of command and control schemes for such devices typically requires real-time, “patient-in-the-loop” ...


---

- 

 - 
- 

 - 
 - 
 - 
 - 
 - 
 - 

 A Real-time, 3D Musculoskeletal Model for Dynamic Simulation of Arm Movements - PMC

 - 

 - 

- 

 Skip to main content

 An official website of the United States government

 Here's how you know 

 Here's how you know 

 Official websites use .gov 

 A
 .gov website belongs to an official
 government organization in the United States.

 Secure .gov websites use HTTPS 

 A lock (

 ) or https:// means you've safely
 connected to the .gov website. Share sensitive
 information only on official, secure websites.

 Search

 Log in 

 - 

 Dashboard

 - 

 Publications

 - 

 Account settings

 - 

 Log out

 Search…

 Search NCBI

 Search PMC Full-Text Archive 

 Search in PMC 

 - 

 Journal List

 - 

 User Guide

 - 

 - 

 - 

 - 

 - 

 - 

## PERMALINK

 Copy 

 As a library, NLM provides access to scientific literature. Inclusion in an NLM database does not imply endorsement of, or agreement with,
 the contents by NLM or the National Institutes of Health.

 Learn more:
 PMC Disclaimer 
 |
 
 PMC Copyright Notice

 IEEE Trans Biomed Eng 
. Author manuscript; available in PMC: 2010 Nov 3.

 Published in final edited form as: IEEE Trans Biomed Eng. 2008 Sep 26;56(4):941–948. doi: 10.1109/TBME.2008.2005946 

# A Real-time, 3D Musculoskeletal Model for Dynamic Simulation of Arm Movements
 
 Edward K Chadwick 

### Edward K Chadwick 

 1 Biomedical Engineering department of Case Western Reserve University

 Find articles by Edward K Chadwick 

 1 , Dimitra Blana 

### Dimitra Blana 

 2 Biomedical Engineering department of Case Western Reserve University

 Find articles by Dimitra Blana 

 2 , Antonie J van den Bogert 

### Antonie J van den Bogert 

 3 Biomedical Engineering department of The Cleveland Clinic

 Find articles by Antonie J van den Bogert 

 3 , Robert F Kirsch 

### Robert F Kirsch 

 4 Biomedical Engineering department of Case Western Reserve University; Research Career Scientist, Dept. of Veterans Affairs Rehabilitation Research and Development Service

 Member, IEEE

 Find articles by Robert F Kirsch 

 4 

- Author information 

- Article notes 

- Copyright and License information 

 1 Biomedical Engineering department of Case Western Reserve University

 2 Biomedical Engineering department of Case Western Reserve University

 3 Biomedical Engineering department of The Cleveland Clinic

 4 Biomedical Engineering department of Case Western Reserve University; Research Career Scientist, Dept. of Veterans Affairs Rehabilitation Research and Development Service

#### Roles

 Robert F Kirsch : Member, IEEE 

 Issue date 2009 Apr.

 PMC Copyright notice 

 PMCID: PMC2971671 NIHMSID: NIHMS240588 PMID: 19272926 

 The publisher's version of this article is available at IEEE Trans Biomed Eng 

## Abstract

 Neuroprostheses can be used to restore movement of the upper limb in individuals with high-level spinal cord injury. Development and evaluation of command and control schemes for such devices typically requires real-time, “patient-in-the-loop” experimentation. A real-time, three-dimensional, musculoskeletal model of the upper limb has been developed for use in a simulation environment to allow such testing to be carried out non-invasively. The model provides real-time feedback of human arm dynamics that can be displayed to the user in a virtual reality environment. The model has a three degree-of-freedom gleno-humeral joint as well as elbow flexion/extension and pronation/supination, and contains 22 muscles of the shoulder and elbow divided into multiple elements. The model is able to run in real time on modest desktop hardware and demonstrates that a large-scale, 3D model can be made to run in real time. This is a prerequisite for a real-time, whole arm model that will form part of a dynamic arm simulator for use in the development, testing and user training of neural prosthesis systems.

 Index Terms: Functional electrical stimulation, biomechanics, musculoskeletal modelling, shoulder, upper limb, simulation

## I. Introduction

 Functional electrical stimulation (FES) can be used to restore upper limb function in people with high-level spinal cord injury (SCI) through the electrical stimulation of nerves and muscles to generate movement in the paralysed limb. Upper limb FES systems have been described by numerous groups [ 1 ]–[ 5 ], but have more commonly been applied to people with mid-cervical level (C5-C6 SCI) or thoracic-level injuries [ 6 ]. People with high-level (C1-C4 SCI) injuries have fewer options for command sources for FES than those with lower level injuries, who can often make use of retained voluntary function to control an FES-based neuroprosthesis.

 Operating an FES system requires sophisticated control of a multiple degree-of-freedom, kinematically coupled system of joints using redundant, non-linear actuators from a limited, artificial command source. For goal-directed movements in the upper extremity, feedback control involving user input is essential and the command interface for the user becomes an integral part of the control system. Musculoskeletal models have been used successfully to help design control systems for FES by a number of authors [ 7 ]–[ 10 ]. These modelling studies have consisted of off-line analyses of model and controller behaviour, not real-time simulations. They did not generate real-time feedback for the user and did not allow the adaption of control signals in real time. The execution of “patient-in-the-loop” experiments in which it is possible to give the subject real-time feedback of simulated arm movement requires a virtual reality environment with real-time simulation of arm dynamics.

 A brain-machine interface (BMI) is an example of such a command interface, and offers a promising source of commands for neuroprosthetic control, but to date BMI research has tended to focus on the control of non-dynamic tasks such as 2D computer mouse movement [ 11 ] or in some cases simulated 3D target reaching tasks [ 12 ]. For the use of brain signals for real arm control (i.e. with a neuroprosthesis), it may be essential to consider the effect of arm and muscle dynamics on the nature of the extracted signal. Carmena et al. [ 13 ], for example, showed movement control with a BMI by decoding multiple movement parameters including hand force and muscle activations, not just movement trajectories.

 In order to help answer the question of whether arm dynamics affect the nature of the required movement command, and to facilitate FES controller development, a tool for the real-time, dynamic simulation of arm movement would be very useful. A number of large-scale models of the upper limb have been described in the literature [ 14 ]–[ 17 ] but the focus of these models has always been anatomical fidelity, and not simulation speed. Real-time simulation was not a goal of these studies, and no data are given on simulation speeds. Other groups have described very promising simulation environments that do feature real-time simulation, incorporating musculoskeletal modelling and visualisation [ 18 ], but to date details on the complexity of models that can be simulated in real time are not available. Davoodi et al. [ 19 ] described the virtual reality aspects of the simulation environment in some detail, but did not provide many details on the biomechanical modelling aspects.

 The long-term goal of this work, therefore, was to develop a complex, three-dimensional (3D) biomechanical model of the upper limb that runs in real time. The model described simulates realistic arm dynamics (inertia, kinematic coupling, muscle dynamics), for use in the so-called Dynamic Arm Simulator (DAS). The real-time dynamics model interfaces with a visualisation environment developed using Game Studio (Conitec Data Systems, Inc.). This gives the user of the simulator direct visual feedback regarding the movement of their virtual arm, and allows the development of neural prosthesis controllers and brain-machine interfaces, as well as providing a training environment for potential users of such systems. The model is fully customisable, e.g. allowing compromises in muscle strength associated with spinal cord injury to be accurately modelled. The clinical relevance of this work, then, is that it allows rapid development of rehabilitation technologies for people with spinal cord injury or other neurological impairment.

 The aim of the current study was to develop a dynamic model of the human upper limb that would run in real time and be of sufficient complexity to allow realistic simulations of arm movement. This real-time model is a pre-requisite for the Dynamic Arm Simulator that could be used in the development, testing and user training of neural prosthesis systems.

## II. Methods

### A. Musculoskeletal model in SIMM

 The structure of the model was initially built using SIMM (MusculoGraphics Inc.), a graphical musculoskeletal modelling package. The model has five degrees of freedom: three at the gleno-humeral joint, one for elbow flexion/extension and one for pronation/supination. The scapula was considered to be the base of this model and was fixed. The gleno-humeral joint was modelled as three orthogonal hinges, or pin joints as they are known in SIMM. The elbow flexion axis and the forearm pronation axis were also modelled as pin joints, with direction vectors determined by cadaver kinematic measurement [ 20 ]. Limits on the range of motion of the joints were imposed based on data from Gûnal et al. [ 21 ], who measured the range of motion of the upper limb in a large number of male subjects. These angles were restricted slightly where necessary in order to ensure the correct wrapping of muscles in the SIMM model, and the final values are shown in Table I . The range of motion of the humerus with respect to the thorax is of course reduced compared to a full upper limb model due to the fixed scapula.

#### TABLE I. Angular limits for each degree of freedom (degrees).

 Degree of freedom 
 Min angle 
 Max angle 

 Plane of elevation 
 -90 
 90 

 Angle of elevation 
 5 
 90 

 Internal rotation 
 -55 
 70 

 Elbow flexion 
 5 
 140 

 Forearm pronation 
 5 
 160 

 Open in a new tab 

 Muscles crossing joints proximal to the gleno-humeral joint have the correct line of action and wrapping, but are considered to originate from the scapular rigid body. A total of 22 muscles and muscle parts are included in the real-time model, divided into 102 muscle elements that are modelled independently. Muscles are modelled using the minimum number of elements needed to accurately model the mechanical line of action of each part. In the case of widely diverging muscles such as the deltoids, this was as many as 11 elements. The number of elements used for each muscle and the degrees of freedom crossed by the muscles are shown in Table II .

#### TABLE II. Muscles included in the real-time model, showing the joints crossed by each muscle (gleno-humeral: GH, humero-ulnar: HU or radio-ulnar: RU) and the number of elements used to model the muscle.

 Muscle 
 Joints 
 No. of elements 

 Deltoid, scapular part 
 GH 
 11 

 Deltoid, clavicular part 
 GH 
 4 

 Coracobrachialis 
 GH 
 3 

 Infraspinatus 
 GH 
 6 

 Teres minor 
 GH 
 3 

 Teres major 
 GH 
 4 

 Supraspinatus 
 GH 
 4 

 Subscapularis 
 GH 
 11 

 Biceps, long head 
 GH, HU and RU 
 1 

 Biceps, short head 
 GH, HU and RU 
 2 

 Triceps, long head 
 GH and HU 
 4 

 Latissimus dorsi 
 GH 
 6 

 Pectoralis major, thoracic part 
 GH 
 6 

 Pectoralis major, clavicular part 
 GH 
 2 

 Triceps, medial head 
 HU 
 5 

 Brachialis 
 HU 
 7 

 Brachioradialis 
 HU and RU 
 3 

 Pronator teres 
 HU and RU 
 2 

 Supinator 
 HU and RU 
 5 

 Pronator quadratus 
 RU 
 3 

 Triceps, lateral head 
 HU 
 5 

 Anconeus 
 HU 
 5 

 Open in a new tab 

 Geometrical data for the real-time model were taken from the cadaver studies of Klein-Breteler et al. [ 20 ]. In these studies the muscles and muscle parts controlling the movement of the shoulder and elbow were divided into multiple elements, and their origins and insertions measured. Joint surfaces and other bony contours were digitised for modelling using geometrical forms, and an extensive set of muscle architecture parameters was measured. This included tendon length, physiological cross-sectional area (PCSA), pennation angle, sarcomere length and fibre length. All these data are available via the International Shoulder Group website ( http://internationalshouldergroup.org ).

### B. Equations of motion

 The equations of motion were derived using SDFAST (Symbolic Dynamics and Parametric Technology Corporation, Needham, MA, USA). The dynamics export feature of SIMM was used to form the model description file for SDFAST, model.sd. This file was processed by SDFAST to generate the equations of motion as C code for use in the simulation. The equations of motion are expressed as second-order differential equations of the following form:

 M ( q ) q ¨ = Q M ( q , q ˙ , t ) + Q E ( q , q ˙ ) + Q C ( q , q ˙ ) 
 (1) 

 where M is the mass matrix, q are the generalised coordinates and Q M , Q E and Q C are the generalised force terms due to muscles, external forces and coriolis and centrifugal forces.

### C. Muscle wrapping and lines of action

 Muscle wrapping objects were defined in the SIMM file to allow the calculation of the lines of action and the moment arms of the muscles in all positions of the model. These were based on the geometrical structures measured in the cadaver studies described above [ 20 ] and used in [ 22 ], and included spheres for the humeral head, an ellipsoid for the thorax and cylinders for the humerus, ulna and radius. The run-time calculation of muscle wrapping, however, is too time-consuming for a real-time application. In this study, preprocessing of the muscle moment arms and lines of action was carried out to allow faster operation at run-time.

 Muscle moment arms were exported from SIMM for each muscle element crossing each degree of freedom over the whole range of model motions. In the real-time model, muscle-skeleton coupling was represented by a polynomial model for muscle length as a function of kinematic degrees of freedom. This was done in preference to fitting the moment arms directly as it reduces the number of model coefficients needed, therefore decreasing simulation time. This also avoids over fitting the data, which can happen when fitting the moment arms directly, resulting in the model not being mechanically consistent. The moment arm data from the SIMM model were then used to assess the fitting accuracy. Polynomial terms were added to the model until the moment arm error for each element was less than 10% of the maximum moment arm value or 2mm, whichever was the greater. This gave a good compromise between accurately describing the moment arms and limiting the number of polynomial terms needed for that description. A similar procedure was followed to approximate the muscle lines of action. In the case of the lines of action, one polynomial was required for each orthogonal component of the path. The lines of action of the muscles were needed for the calculation of the joint reaction force and subsequently the assessment of gleno-humeral stability described in Section II-E.

### D. Muscle model

 A three-component, Hill-type muscle model was used to model muscle force generation. The model consists of a contractile element for active force generation, a series elastic element (SEE) representing tendon and other series stiffnesses, and a parallel elastic element (PEE) representing the passive stiffness of the muscle belly. Activation dynamics were governed by the first order differential equation:

 a ˙ ( t ) = ( c 1 u ( t ) + c 2 ) ( u ( t ) − a ( t ) ) 
 (2) 

 where a is the active state, u is the neural command and the terms c 1 and c 2 are chosen to give activation and deactivation time constants derived from the proportions of fast and slow twitch fibres in the muscle (derived from [ 23 ]).

 The contractile element produces a force, F ce , which is dependent on the shortening velocity, V ce , fibre length, L ce , and active state, a :

 F ce = f ( L ce ) ⋅ g ( V ce , a ) ⋅ a 
 (3) 

 where f describes the force-length relationship and g describes the Hill shortening equation (for details see McLean et al. [ 24 ]). In order to prevent infinite muscle velocities at zero activation, a passive damper of 10 N/m/s was modelled in parallel with the contractile element for low activations (less than 0.02). This value was within the range found by Kirsch et al. [ 25 ] and allowed the arm to fall at a realistic rate.

 Stiffness properties for the series and parallel elastic components were given by quadratic relationships between force and elongation (where L slack is the element slack length normalised to the optimal fibre length):

 F = k ( L − L slack ) 2 
 (4) 

 The stiffness parameter k for the SEE was set such that 4% elongation was achieved when the maximal isometric force was applied (from [ 24 ]). L slack for the series elastic element was approximated by the tendon slack length measured in the cadaver studies. L slack for the PEE is a harder parameter to estimate, as data for doing this are not available. Following the example of McLean et al. [ 24 ] a default value of 1.0 was used (normalised to optimum fibre length), except for a small number of muscle elements for which this led to prohibitively high passive forces. For these elements the value of L slack was estimated based on the desired range of motion of the joint that the muscle crossed. There are insufficient data in the literature to obtain this value in any other way. These values are shown in Table V in the Appendix .

### E. Computation of GH stability

 The gleno-humeral joint is minimally stabilised by passive structures around the joint, and requires active stabilisation from the rotator cuff muscles during movement. The ability to monitor the stability of that joint is essential in the design of controllers for FES systems, a major application of this model.

 At each major time step of the simulation, the resultant force vector between the humeral head and the glenoid was calculated. This was the vector sum of all contributions to the joint reaction force: external forces on the limb as well as the sum of all muscle forces for muscles crossing the gleno-humeral joint. The stability of the gleno-humeral joint is defined as a function of the angle of the resultant force in the glenoid relative to the maximum angle that can be reached before dislocation of the joint. Specifically, the stability value is defined as:

 G H stab = ( θ / θ a ) 2 + ( φ / φ a ) 2 
 (5) 

 where θ and ϕ are the angles of the vector away from the normal to the glenoid along the major and minor axes of the ellipse, and θ a and ϕ a are the angles of that vector as it reaches the rim of the glenoid. The value of GH stab thus reaches 1 as the reaction force vector reaches the edge of the ellipsoid defining the glenoid fossa and has a value of zero when the vector is exactly in the centre of the fossa. Values of less than 1 indicate that the force is inside the rim of the fossa and the joint is stable, and values over 1 indicate that the reaction force vector would point outside the fossa, tending to cause dislocation of the joint.

### F. Simulink and xPC target

 In order to guarantee real-time operation of the model, Matlab's xPC Target operating system was used. This is a specialised application of the Real-Time Workshop allowing fast development of real-time applications on x86 hardware. The model is developed in Simulink in simulation mode before being compiled as a real-time executable for execution on the real-time operating system (RTOS). This approach allows rapid development of code, maximising model performance on given hardware as well as guaranteeing real-time operation.

 When running a model on the real-time system, xPC Target, it is required to use a fixed-step solver to guarantee real-time operation. The time taken for one complete integration of the system is reported as the Task Execution Time (TET) by xPC Target. The minimum step size that can be used with a given model is this TET (plus a small amount to allow for fluctuations in the execution time). If the simulation is stable with that step size for a given integrator, or solver, then real-time operation is possible.

 In this study, two solvers were compared for their performance and stability: the 1 st order Euler solver and the 4 th order Runge-Kutte solver.

### G. The muscle stimulation experiments

 Simulations of simple muscle stimulation experiments were carried out to test the model response to activation of the various muscle groups. Groups of muscles were activated sequentially in the model, and the resulting motions were observed. During these simulations, the Task Execution Time for the model was recorded in order to assess the real-time capability of the model during simulated motions.

 Table III shows the baseline activations used during the simulations. These are the minimum activations necessary to ensure stability of the gleno-humeral joint in the resting position. Only the rotator cuff muscles (infraspinatus, teres minor, supraspinatus and subscapularis) are active in this state, and the maximum of these activations is 0.07. These values were found by iteratively increasing the values of the activations until a combination was found that reduced the value of the gleno-humeral stability constraint ( GH stab ) to less than 0.5, giving a factor of safety of two, and allowed the arm to hang in a neutral position (i.e. without excessive internal or external rotation). As defined previously, GH stab must be less than one to ensure stability of the joint.

#### TABLE III. Minimum activations necessary in the rotator cuff muscles to ensure stability of the gleno-humeral joint with a factor of safety of 2. ( GH stab < 0.5).

 Infraspinatus 
 Teres minor 
 Supraspinatus 
 Subscapularis 

 0.04 
 0.02 
 0.02 
 0.07 

 Open in a new tab 

 Activation of the various muscle groups to produce movement was then applied on top of these baseline activations. Three sets of activations were defined. The first tested the elbow flexion/extension and forearm pro/supination response of the model. The second tested the humeral abduction response by elevating the arm in the scapular plane, and the third repeated this motion, but increased the rotator cuff activity to stabilise the gleno-humeral joint.

## III. Results

 Fig. 1 shows the effect on the model of activation of combinations of elbow flexors (biceps brachii, brachialis, brachioradialis), extensors (triceps) and pronators (pronator teres and pronator quadratus). Part A shows activation of the elbow flexor muscles (at 4 s), co-contraction of the flexors and extensors (at 10 s) and activation of the flexors, extensors and pronators (at 14 s). Part B shows the movements produced by the activations applied in Part A: flexion of the elbow to 140° combined with full supination, then extension back to 100°, and then pronation of the forearm to 160°. Finally, deactivation of all muscles allows the model to return to its initial position of 5° flexion and 70° pronation.

### Fig. 1.

 Open in a new tab 

 Part A shows activation of the elbow flexor muscles (at 4 s), co-contraction of the flexors and extensors (at 10 s) and activation of the flexors, extensors and pronators (at 14 s). Part B shows flexion of the elbow to 140° combined with full supination, extension back to 100° followed by pronation to 160°. Finally, deactivation of all muscles allows the model to return to its initial position of 5° flexion and 70° pronation.
 Fig. 2 shows the effect of activating humeral abductors (the middle and anterior parts of the deltoids) (part A) in addition to the constant rotator cuff muscle baseline activations. The resulting motion of the model (part B) shows abduction of the humerus in approximately the scapular plane (part way between lateral abduction and forward flexion). The activity of the rotator cuff muscles was not adjusted during these motions, and part C of the figure shows the increase in the value of GH stab to over 1, indicating instability of the gleno-humeral joint.

### Fig. 2.

 Open in a new tab 

 Part A shows the activation of the middle and anterior parts of the deltoid muscles, and Part B the model motions in response to those activations. Rotator cuff activity was held at the baseline level throughout the movement, and Part C shows the increase in the value of GH stab to over 1, indicating potential dislocation of the joint.
 Fig. 3 shows the same deltoid activity as the previous figure, but this time with a concurrent increase in the rotator cuff activity. This level of activation was chosen merely to be sufficient to maintain gleno-humeral stability. The resulting motion is quite similar, with slightly increased humeral elevation provided by the abduction tendency of these muscles. Part C, though, shows a much reduced value of GH stab , to less than 0.5, indicating that the stability of the gleno-humeral joint is maintained throughout the motion (with a factor of safety of two).

### Fig. 3.

 Open in a new tab 

 Activation of the middle and anterior parts of the deltoid muscles, with increased stimulation of the rotator cuff muscles are shown in Part A. The resulting model motions are shown in Part B. Note the increased stability of the gleno-humeral joint, shown by the stability value of < 0.5 in Part C.
 Fig. 4 shows the effect of two different solvers on the stability of the model during the simulation. Part A shows that globally the two solvers result in the same motion given the same input activations. Part B shows a magnified y-axis to illustrate the more stable performance of the 4 th order Runge-Kutta solver as compared to the 1 st order Euler solver.

### Fig. 4.

 Open in a new tab 

 Stability of the simulation with two different solvers at their minimum step sizes. Part A shows that the model response to the given input is extremely similar with the two solvers. Part B shows the increased stability of the Runge-Kutta 4 th order solver over the Euler 1 st order.
 Table IV shows the performance of the model on the real-time system. The task execution time (TET) is the computation time required by the processor to complete one integration of the system. If the largest stable step size for the model is larger than this TET, then real-time operation is possible. The table shows that both solvers are capable of real-time operation with this model.

### TABLE IV. Performance of model on real-time system (1.8GHz Pentium 4), averaged across three tasks.

 Solver 
 Task Execution Time 
 Stable Step Size 

 Runge-Kutta 4th order 
 0.51ms 
 1.0ms 

 Euler 1st order 
 0.13ms 
 0.5ms 

 Open in a new tab 

## IV. Discussion

### A. Model evaluation

 The goal of this study was to develop a real-time musculoskeletal model of the human upper limb for use in a dynamic arm simulator. The real-time model described in this study is based on anatomical parameters and structures described in [ 20 ] and previously implemented in a large-scale model using SIMM [ 22 ]. Verification of the model response to muscle activation indicates that the model still behaves as expected, even after conversion of the data to the format suitable for real-time execution. The primary outcome of this work is thus the real-time performance allowing the model to be used in “patient-in-the-loop” testing of neural prosthesis systems.

 The muscle stimulation experiments showed that the model responds in a predictable way to various levels of muscle stimulation. The model is stable in the resting position with only low-level baseline activation of the rotator cuff muscles to stabilise the gleno-humeral joint, and is also stable in arbitrary intermediate positions of elbow flexion, forearm pronation and humeral abduction ( Fig. 1 and 2 ). Furthermore, gleno-humeral stability is maintained during humeral abduction by activation of the rotator cuff muscles, and is seen to be compromised when these muscles are not concurrently activated ( Fig. 2 and 3 ). The main outcome of these experiments, though, is that all these movements were successfully simulated in real time.

### B. Computational performance

 Simulink offers a range of fixed-step solvers for use with xPC Target, from 1 st to 5 th order. The 1 st order Euler solver has the lowest computational cost, but also requires the smallest step size to maintain a given level of accuracy. The Runge-Kutta 4 th order solver, in contrast, has a higher computational cost but produces smaller errors for a given step size. This means that the higher order solver is able to perform the simulation with a larger step size, permitting a larger task execution time (TET) while still running in real time. This is the trade-off that must be assessed in the evaluation of the solvers. The solvers at intermediate orders to those described were not found to offer any performance benefit and so were not used. The best solver choice is also influenced by the range of stiffnesses encountered in the system.

 In our system, high stiffness is introduced by the combination of high-stiffness muscle elements (those with short series elastic components, i.e. short tendon slack lengths such as pronator quadratus) and low-inertia degrees of freedom, such as forearm pronation/supination. This stiffness requires the use of the small step sizes seen in Table IV to ensure a stable solution. The Runge-Kutta 4 th order solver was found to be the best compromise between speed and accuracy for the integration of this system. This solver allowed real-time operation of the model on a 1.8GHz Pentium 4 with a solver step size of 1ms. The Euler 1 st order solver was also able to run in real time, with a step size of 0.5ms, but the simulation was slightly less stable in that case, as seen in Fig. 4 .

 In our experience, calculation of the muscle elements is the most time-consuming part of the simulation, and additional degrees of freedom in the kinematics are less significant. We therefore expect some increase in simulation time with a more complex model comprising more muscle elements, but this tends to increase linearly with the number of polynomial terms used to model the muscles and the number of muscle elements. We are therefore confident that the whole arm model which is the long-term goal of this work will also be suitable for real-time simulation, since the increase in complexity should be a factor of around two or less.

### C. Model limitations

 The model is a fully three-dimensional representation of the gleno-humeral joint and elbow that includes all the relevant muscles and degrees of freedom. As far as can be seen from the literature, this is the only model of this scale whose ability to run in real time has been tested. However, our model does have the limitation of the fixed scapula. This simplification was used to allow us to develop the real-time framework without jumping straight into the complexity of the closed-chain shoulder girdle and focus on necessary developments such as monitoring and ensuring gleno-humeral stability. This limits the range of motion of the arm somewhat, as the shoulder girdle would normally contribute to the range of motion of the arm. This also has the effect of increasing the length range over which some of the muscles have to operate, as the gleno-humeral joint must provide all the movement that would otherwise be provided by the combination of scapular and humeral motion. The model still operates under these conditions, providing a real-time, 3D simulation of arm movement, but the motions at individual joints may be slightly less realistic than they would be with a full model including a shoulder girdle.

 The anatomical parameters in the model are taken from measurements based on a single cadaver, but our model can use any internally consistent parameter set describing the shoulder and elbow mechanism. Many other models described in the literature are based on the averaging of all available data (for example, [ 16 ]) or parameter optimisation (for example, [ 26 ]), both of which have their merits. Using single cadaver measurements maintains relationships between structures, such as muscle attachment points, moment arms, optimum fibre lengths etc., which may be lost using other methods. For real-time, “patient-in-the-loop” experiments, matching the model to the patient through the use of actual patient data such as muscle points of origin and insertion would be very attractive. This is not really feasible at this time, but may become so in the future with improvements in imaging modalities. Despite differences between the actual subject and the model, the real-time simulator still provides an extremely useful tool which allows us to examine how well a user can control a complex neuroprosthesis and what the effects of different command interfaces and controllers are on that control.

### D. Applications

 A realistic 3D model of the human upper limb that provides real-time feedback on arm motion allows us to perform experiments with the patient in the loop in a way that would not otherwise be possible. For example, in the development of control algorithms for implanted FES systems, a variety of virtual systems can be investigated before any actual FES intervention is applied to the subject. Simulation of different muscle groups can be tried, stimulation levels can be tested and controller designs can be developed. The level of control that a potential user would have with that system can be assessed with minimal risk and inconvenience to the subject. In the field of brain-machine interfaces, decoding algorithms can be tested to assess the feasibility of multi-degree of freedom control and fundamental questions about control mechanisms can be answered for conditions that would not otherwise be possible.

## V. Conclusions

 We have developed a real-time, 5 DOF, dynamic model of the human shoulder and elbow. This model differs from previous models of these joints in that it can compute the dynamics for an appropriate time duration faster than that time duration, i.e., in “real time”. This was accomplished by replacing computationally demanding calculations (muscle moment arms and muscle lines of action) by efficient, pre-computed polynomial fits. Our model achieved real-time performance using very modest computing hardware, demonstrating the utility of our approach and providing ample opportunities for increasing the complexity of the model in the future.

 This study demonstrated the feasibility of producing a model of sufficient complexity that would run in real time, a prerequisite for the Dynamic Arm Simulator which will allow the execution of “patient-in-the-loop” experiments for FES command and control development. Future work will focus on the integration of the shoulder girdle and hand to the model. That is, additional degrees of freedom will be added to model the sterno-clavicular and acromio-clavicular joints and a simple hand model will be added. This will form a complete upper arm simulator for the design and testing of neural prosthesis controllers and brain-machine interfaces.

## Acknowledgments

 This work is supported by the National Institutes of Health under contracts N01-HD-5-3403 and N01-NS-5-2365.

## Biographies

 Edward Chadwick received the B.Eng. degree in mechanical engineering from The University of Nottingham, U.K., in 1993, and was awarded his Ph.D. in bioengineering for work on biomechanical modelling of the upper limb by the University of Strathclyde, U.K., in 1999. From there he went on to complete a four-year post-doc in the area of shoulder biomechanics at the Technical University of Delft in the Netherlands. Since 2003 he has been employed as a Senior Research Associate in the Biomedical Engineering Department of Case Western Reserve University in Cleveland, Ohio. His research interests are biomechanical modelling of the upper limb, movement analysis and motor control, and the application of modelling and biomechanics to the rehabilitation of people with neuromuscular deficit.

 Dimitra Blana received the B.S. degree in electrical and computer engineering from the National Technical University of Athens, Greece, in 2001. She then went on to earn the M.S. degree in biomedical engineering from Case Western Reserve University, Cleveland, OH, in 2003, and a PhD from the same institute in 2008. She is currently a Research Associate in Biomedical Engineering at Case Western Reserve University. Her research interests are in biomechanical modelling and controller design for upper extremity neuroprostheses.

 Antonie J. (Ton) van den Bogert received his B.S. and M.S. degrees in physics and applied mathematics at the University of Utrecht, and in 1989 a Ph.D. degree in veterinary sciences from the same university for computational modeling of locomotion in horses. After postdoctoral work in sports biomechanics at the University of Calgary, he became a faculty member in Kinesiology in 1993. He moved to Cleveland in 1998 where he currently is Associate Staff in the Lerner Research Institute of the Cleveland Clinic, and Associate Professor at Case Western Reserve University. His current research interest is in human biomechanics and motor control, with applications in the prevention and rehabilitation of musculoskeletal injuries and the development of assistive technology for persons with disabilities. This work is supported by innovative methods for analysis of human movement and computational modeling of musculoskeletal system dynamics. He is a founding member of the Technical Group Computer Simulation (TGCS) in the International Society of Biomechanics, and since 1988 the moderator of the internet discussion group BIOMCH-L.

 Robert Kirsch (M'82) received the B.S. degree in electrical engineering from the University of Cincinnati, OH, in 1982, and the M.S. and Ph.D. degrees in biomedical engineering from Northwestern University, in 1986 and 1990, respectively. He was a postdoctoral fellow in the Department of Biomedical Engineering, McGill University, Montréal, QC, Canada, from 1990 to 1993. He is currently a Professor of Biomedical Engineering at Case Western Reserve University, Cleveland, OH, and Associate Director for Research in the Cleveland VA FES Center. His research focuses on restoring movement to disabled individuals using functional electrical stimulation (FES) and controlling FES actions via natural neural commands. Computer-based models of the human upper extremity are used to develop new FES approaches. FES user interfaces, including ones based on brain recordings, are being developed to provide FES users with the ability to command movements of their own arm.

## Appendix: Muscle Stiffness Parameters

 Table V shows the values of PEE slack (parallel elastic element slack length) which were modified from the default value of 1.0 to give reasonable resting lengths for the muscles based on limb resting position.

### TABLE V. The values of PEE slack for muscle elements in which it differed from the default value of 1.0 (normalised to optimum fibre length).

 Muscle name 
 Element no. 
 PEE slack 

 Infraspinatus 
 [4 5 6] 
 [1.1 1.2 1.1] 

 Teres minor 
 [1 2 3] 
 [1.3 1.4 1.4] 

 Supraspinatus 
 [1] 
 [1.1] 

 Biceps, long head 
 [1] 
 [1.3] 

 Biceps, short head 
 [1 2] 
 [1.5 1.6] 

 Triceps, long head 
 [1 2 3 4] 
 [1.5 1.5 1.4 1.2] 

 Triceps, medial head 
 [1 2 3 4 5] 
 [1.3 1.3 1.4 1.3 1.2] 

 Brachialis 
 [1 2 3 4 5 6 7] 
 [1.2 1.2 1.1 1.4 1.3 1.2 1.3] 

 Brachioradialis 
 [1 2 3] 
 [1.2 1.2 1.3] 

 Pronator teres 
 [1] 
 [1.3] 

 Supinator 
 [2 3 4 5] 
 [1.2 1.3 1.2 1.3] 

 Pronator quadratus 
 [1] 
 [1.1] 

 Triceps, lateral head 
 [1 2 3 4] 
 [1.2 1.2 1.2 1.2] 

 Open in a new tab 

## Contributor Information

 Edward K. Chadwick, Biomedical Engineering department of Case Western Reserve University

 Dimitra Blana, Biomedical Engineering department of Case Western Reserve University.

 Antonie J. van den Bogert, Biomedical Engineering department of The Cleveland Clinic

 Robert F. Kirsch, Biomedical Engineering department of Case Western Reserve University; Research Career Scientist, Dept. of Veterans Affairs Rehabilitation Research and Development Service.

## References

- 
 1. Keith MW, Peckham PH, Thrope GB, Stroh KC, Smith B, Buckett JR, Kilgore KL, Jatich JW. Implantable functional neuromuscular stimulation in the tetraplegic hand. J Hand Surg [Am] 1989 May;14:524–530. doi: 10.1016/s0363-5023(89)80017-6. [ DOI ] [ PubMed ] [ Google Scholar ]

- 
 2. Bryden AM, Memberg WD, Crago PE. Electrically stimulated elbow extension in persons with c5/c6 tetraplegia: a functional and physiological evaluation. Arch Phys Med Rehabil. 2000 Jan;81:80–88. doi: 10.1016/s0003-9993(00)90226-0. [ DOI ] [ PubMed ] [ Google Scholar ]

- 
 3. Thorsen R, Spadone R, Ferrarin M. A pilot study of myoelectrically controlled fes of upper extremity. IEEE Trans Neural Syst Rehabil Eng. 2001 Jun;9:161–168. doi: 10.1109/7333.928576. [ DOI ] [ PubMed ] [ Google Scholar ]

- 
 4. Popovic MR, Popovic DB, Keller T. Neuroprostheses for grasping. Neurol Res. 2002 Jul;24:443–452. doi: 10.1179/016164102101200311. [ DOI ] [ PubMed ] [ Google Scholar ]

- 
 5. Ragnarsson KT. Functional electrical stimulation after spinal cord injury: current use, therapeutic effects and future directions. Spinal Cord. 2007 Sep; doi: 10.1038/sj.sc.3102091. [ DOI ] [ PubMed ] [ Google Scholar ]

- 
 6. Rupp R, Gerner HJ. Neuroprosthetics of the upper extremity–clinical application in spinal cord injury and challenges for the future. Acta Neurochir Suppl. 2007;97:419–426. doi: 10.1007/978-3-211-33079-1_55. [ DOI ] [ PubMed ] [ Google Scholar ]

- 
 7. Kirsch R, Acosta A, van der Helm F, Rotteveel R, Cash L. Model-based development of neuroprostheses for restoring proximal arm function. J Rehabil Res Dev. 2001 Dec;38:619–626. [ PubMed ] [ Google Scholar ]

- 
 8. Ferrarin M, Palazzo F, Riener R, Quintern J. Model-based control of fes-induced single joint movements. IEEE Trans Neural Syst Rehabil Eng. 2001 Sep;9:245–257. doi: 10.1109/7333.948452. [ DOI ] [ PubMed ] [ Google Scholar ]

- 
 9. Giuffrida JP, Crago PE. Functional restoration of elbow extension after spinal-cord injury using a neural network-based synergistic fes controller. IEEE Trans Neural Syst Rehabil Eng. 2005 Jun;13:147–152. doi: 10.1109/TNSRE.2005.847375. [ DOI ] [ PubMed ] [ Google Scholar ]

- 
 10. Heilman BP, Kirsch RF. Model-based development of a user control algorithm for postural control via a fes-based standing neuroprosthesis. Conf Proc IEEE Eng Med Biol Soc; 2004. pp. 4614–4617. xx. [ DOI ] [ PubMed ] [ Google Scholar ]

- 
 11. Pistohl T, Ball T, Schulze-Bonhage A, Aertsen A, Mehring C. Prediction of arm movement trajectories from ecog-recordings in humans. J Neurosci Methods. 2008 Jan;167:105–114. doi: 10.1016/j.jneumeth.2007.10.001. [ DOI ] [ PubMed ] [ Google Scholar ]

- 
 12. Taylor DM, Tillery SIH, Schwartz AB. Direct cortical control of 3d neuroprosthetic devices. Science. 2002 Jun;296:1829–1832. doi: 10.1126/science.1070291. [ DOI ] [ PubMed ] [ Google Scholar ]

- 
 13. Carmena J, Lebedev M, Crist R, O'Doherty J, Santucci D, Dimitrov D, Patil P, Henriquez C, Nicolelis M. Learning to control a brainmachine interface for reaching and grasping by primates. PLoS Biol. 2003 Nov;1:e42. doi: 10.1371/journal.pbio.0000042. [ DOI ] [ PMC free article ] [ PubMed ] [ Google Scholar ]

- 
 14. van der Helm FC. A finite element musculoskeletal model of the shoulder mechanism. J Biomech. 1994 May;27:551–569. doi: 10.1016/0021-9290(94)90065-5. [ DOI ] [ PubMed ] [ Google Scholar ]

- 
 15. Garner BA, Pandy MG. Musculoskeletal model of the upper limb based on the visible human male dataset. Comput Methods Biomech Biomed Engin. 2001 Feb;4:93–126. doi: 10.1080/10255840008908000. [ DOI ] [ PubMed ] [ Google Scholar ]

- 
 16. Holzbaur KRS, Murray WM, Delp SL. A model of the upper extremity for simulating musculoskeletal surgery and analyzing neuromuscular control. Ann Biomed Eng. 2005 Jun;33:829–840. doi: 10.1007/s10439-005-3320-7. [ DOI ] [ PubMed ] [ Google Scholar ]

- 
 17. Charlton IW, Johnson GR. A model for the prediction of the forces at the glenohumeral joint. Proc Inst Mech Eng [H] 2006 Nov;220:801–812. doi: 10.1243/09544119JEIM147. [ DOI ] [ PubMed ] [ Google Scholar ]

- 
 18. Hauschild M, Davoodi R, Loeb GE. A virtual reality environment for designing and fitting neural prosthetic limbs. IEEE Trans Neural Syst Rehabil Eng. 2007 Mar;15:9–15. doi: 10.1109/TNSRE.2007.891369. [ DOI ] [ PubMed ] [ Google Scholar ]

- 
 19. Davoodi R, Urata C, Hauschild M, Khachani M, Loeb GE. Model-based development of neural prostheses for movement. IEEE Trans Biomed Eng. 2007 Nov;54:1909–1918. doi: 10.1109/TBME.2007.902252. [ DOI ] [ PubMed ] [ Google Scholar ]

- 
 20. Klein Breteler MD, Spoor CW, Van der Helm FC. Measuring muscle and joint geometry parameters of a shoulder for modeling purposes. J Biomech. 1999 Nov;32:1191–1197. doi: 10.1016/s0021-9290(99)00122-0. [ DOI ] [ PubMed ] [ Google Scholar ]

- 
 21. Gûnal I, Köse N, Erdogan O, Göktürk E, Seber S. Normal range of motion of the joints of the upper extremity in male subjects, with special reference to side. J Bone Joint Surg Am. 1996 Sep;78:1401–1404. doi: 10.2106/00004623-199609000-00017. [ DOI ] [ PubMed ] [ Google Scholar ]

- 
 22. Blana D, Hincapie JG, Chadwick EK, Kirsch RF. A musculoskeletal model of the upper extremity for use in the development of neuroprosthetic systems. J Biomech. 2008;41:1714–1721. doi: 10.1016/j.jbiomech.2008.03.001. [ DOI ] [ PMC free article ] [ PubMed ] [ Google Scholar ]

- 
 23. Winters JM, Stark L. Analysis of fundamental human movement patterns through the use of in-depth antagonistic muscle models. IEEE Trans Biomed Eng. 1985 Oct;32:826–839. doi: 10.1109/TBME.1985.325498. [ DOI ] [ PubMed ] [ Google Scholar ]

- 
 24. McLean SG, Su A, van den Bogert AJ. Development and validation of a 3-d model to predict knee joint loading during dynamic movement. J Biomech Eng. 2003 Dec;125:864–874. doi: 10.1115/1.1634282. [ DOI ] [ PubMed ] [ Google Scholar ]

- 
 25. Kirsch RF, Boskov D, Rymer WZ. Muscle stiffness during transient and continuous movements of cat muscle: perturbation characteristics and physiological relevance. IEEE Trans Biomed Eng. 1994 Aug;41:758–770. doi: 10.1109/10.310091. [ DOI ] [ PubMed ] [ Google Scholar ]

- 
 26. Garner BA, Pandy MG. Estimation of musculotendon properties in the human upper limb. Ann Biomed Eng. 2003 Feb;31:207–220. doi: 10.1114/1.1540105. [ DOI ] [ PubMed ] [ Google Scholar ]

## ACTIONS

 - 

 View on publisher site 

 - 

 PDF (798.8 KB) 

 - 

 Cite 

 - 

 Collections 

 - 

 Permalink 

## PERMALINK

 Copy 

## RESOURCES

### 
 
 Similar articles

### 
 
 Cited by other articles

### 
 
 Links to NCBI Databases

## Cite

 - 

 Copy 

 - 

 Download .nbib 
 .nbib 

 - 

 Format: 

 AMA

 APA

 MLA

 NLM

## Add to Collections

 Create a new collection 

 Add to an existing collection 

 Name your collection
 * 

 Choose a collection

 Unable to load your collection due to an error

 Please try again 

 Add

 Cancel

 Back to Top 

- 

- 
- 
- 

-