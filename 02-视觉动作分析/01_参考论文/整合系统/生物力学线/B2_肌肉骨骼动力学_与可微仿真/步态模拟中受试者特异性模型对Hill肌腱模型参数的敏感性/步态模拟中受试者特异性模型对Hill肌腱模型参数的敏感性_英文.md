# Sensitivity of subject-specific models to Hill muscle–tendon model parameters in simulations of gait

V. Carbone <sup>a,n</sup>, M.M. van der Krogt <sup>a,b</sup>, H.F.J.M. Koopman <sup>a</sup>, N. Verdonschot <sup>a,c</sup>

<sup>a</sup> Laboratory of Biomechanical Engineering, MIRA Institute, University of Twente, Enschede, The Netherlands

<sup>b</sup> Department of Rehabilitation Medicine, MOVE Research Institute Amsterdam, VU University Medical Center, Amsterdam, The Netherlands

<sup>c</sup> Orthopedic Research Laboratory, Radboud University Medical Centre, Nijmegen, The Netherlands

## a r t i c l e i n f o

Article history: Accepted 2 April 2016

Keywords: Sensitivity analysis Muscle–tendon parameters Subject-specific modeling Lower extremity Gait

## a b s t r a c t

Subject-specific musculoskeletal (MS) models of the lower extremity are essential for applications such as predicting the effects of orthopedic surgery. We performed an extensive sensitivity analysis to assess the effects of potential errors in Hill muscle–tendon (MT) model parameters for each of the 56 MT parts contained in a state-of-the-art MS model. We used two metrics, namely a Local Sensitivity Index (LSI) and an Overall Sensitivity Index (OSI), to distinguish the effect of the perturbation on the predicted force produced by the perturbed MT parts and by all the remaining MT parts, respectively, during a simulated gait cycle. Results indicated that sensitivity of the model depended on the specific role of each MT part during gait, and not merely on its size and length. Tendon slack length was the most sensitive parameter, followed by maximal isometric muscle force and optimal muscle fiber length, while nominal pennation angle showed very low sensitivity. The highest sensitivity values were found for the MT parts that act as prime movers of gait (Soleus: average OSI 5.27%, Rectus Femoris: average OSI 4.47%, Gastrocnemius: average OSI 3.77%, Vastus Lateralis: average OSI 1.36%, Biceps Femoris Caput Longum: average OSI 1.06%) and hip stabilizers (Gluteus Medius: average OSI 3.10%, Obturator Internus: average OSI 1.96%, Gluteus Minimus: average OSI 1.40%, Piriformis: average OSI 0.98%), followed by the Peroneal muscles (average OSI 2.20%) and Tibialis Anterior (average OSI 1.78%) some of which were not included in previous sensitivity studies. Finally, the proposed priority list provides quantitative information to indicate which MT parts and which MT parameters should be estimated most accurately to create detailed and reliable subject-specific MS models.

& 2016 Elsevier Ltd. All rights reserved.

## 1. Introduction

Musculoskeletal (MS) models of the lower extremity represent a promising tool to explore complex biomechanical problems, such as predicting the functional outcome of orthopedic surgery (Piazza et al., 2003; Reinbolt et al., 2009; Taddei et al., 2012). For these applications, subject-specific models are essential to achieve reliable model predictions (Lenaerts et al., 2009; Scheys et al., 2011; Gerus et al., 2013). In particular, muscle–tendon (MT) parameters, which greatly affect model outcomes (Brand et al., 1986; Hoy et al., 1990; Out et al., 1996), are difficult to measure directly, and are known to vary with age, gender and activity (Friederich and Brand, 1990).

Developing efficient methods that allow estimating subjectspecific MT parameters represents a significant challenge (Garner and Pandy, 2003; Winby et al., 2008; Van Campen et al., 2014). However, when using detailed state-of-the-art models (Klein Horsman et al., 2007; Arnold et al., 2010; Modenese et al., 2011), it is fundamental to clarify which muscles need to be described with greater detail, and what the required accuracy of the measurement techniques should be. Previous studies have investigated the sensitivity of MS model predictions to variations in MT parameters, unfortunately these analyses used simplified models (Hoy et al., 1990), reported results averaged over the muscles analyzed (Scovil and Ronsky, 2006), or focused only on a single joint (Out et al., 1996; De Groote et al., 2010) or on a small group of representative muscles (Redl et al., 2007; Xiao and Higginson, 2010; Ackland et al., 2012). To our knowledge, the only extensive sensitivity studies of MS model predictions analyzed uncertainties in MS geometry parameters (Carbone et al., 2012; Valente et al., 2014), but no comprehensive analysis has been performed to study individual sensitivity in muscles to MT parameters perturbations.

The aim of this study was to assess the sensitivity of a detailed state-of-the-art MS model to potential errors in Hill MT model parameters. We performed an extensive analysis to quantify the effect of perturbations on tendon slack length, optimal muscle fiber length, maximal isometric muscle force and nominal pennation angle for each muscle present in the model. The results provided quantitative information to indicate which MT parameters for which muscles should be estimated most accurately to create a reliable subject-specific MS model that satisfies the required accuracy of the specific application.

## 2. Methods

We used the Twente Lower Extremity Model (TLEM) (Klein Horsman et $\mathrm { a l . }$ 2007) implemented in the AnyBody Modeling System™ ver. 4.2.1 (AnyBody Technology A/S, Aalborg, Denmark) (Damsgaard et al., 2006). The model consisted of 12 body segments, 11 joints, and 21 Degrees of Freedom (Fig. 1A). To describe accurately the functionality of the MS system, each leg contained 56 MT parts whose mechanical effect was described by 159 three-element Hill-type MT elements (Zajac, 1989) (Fig. 1B).

Inverse dynamic simulations were based on 3D motion analysis and force-plate data recorded during a gait trial on a level walkway. Age, height and mass of one male subject were 26 years, 1.73 cm and 63 kg respectively (AnyBody Managed Model Repository Ver. 1.3). The model was linearly scaled to match the anthropometry of the subject, based on relative position of optical markers. A static optimization problem was solved to calculate MT forces, minimizing the sum of the cubes of muscle activations at each time step (Crowninshield and Brand, 1981). MT forces were constrained to the dynamic equilibrium equations, taking into account for each MT element the physiological bounds imposed by force-length and forcevelocity characteristics of the contractile element CE, and the elasticity of the passive element PE. The effect of the elasticity of the tendon T was approximated by allowing the length and the force of the tendon to vary with the muscle activity. Assuming relatively small tendon elongations, the tendon force dependency on muscle activity was approximated with a linear relationship that could be incorporated efficiently into the muscle recruitment solver implemented in the AnyBody Modeling System™ (AnyBody Technology A/S, Aalborg, Denmark).

Our sensitivity analysis focused on four MT parameters that can be potentially measured in vivo (Narici et al., 1996: Fukunaga et al., 1997) or estimated (Garner and Pandy 2003: Van Campen et al. 2014) when creating a subiect-specific model: tendon slack length $L _ { T } ^ { s } ,$ optimal muscle fiber length ${ \overline { { L _ { f } } } } ,$ maximal isometric muscle force $\overline { { F ^ { M } } }$ and nominal pennation angle γ. For each MT part and each of the four MT parameters, two perturbations from its nominal value were applied, based on the average anatomical variability calculated by Ward et al. (2009): specifically, 20% and 20% for L<sup>s</sup> , L<sub>f</sub> and $\overline { { F ^ { M } } }$ , and 10° and 10° for γ (only 10° for not-pennated MT elements).

After each perturbation, a new static optimization problem was solved to compute the corresponding changes in the predicted MT forces. For reasons of symmetry, perturbations were performed only in the right leg. Consistently with our previous study on MS geometry (Carbone et al., 2012), we calculated the sensitivity of the model by using two metrics:

1. Local Sensitivity Index (LSI), to quantify the effect of the perturbation on the force produced by the perturbed MT part:

$$
\mathrm{LSI} = \frac {\sum_ {i = p e r t} \int_ {0} ^ {T} | F _ {n e w , i} ^ {M T} (t) - F _ {o l d , i} ^ {M T} (t) | d t}{\sum_ {i = p e r t} \int_ {0} ^ {T} F _ {o l d , i} ^ {M T} (t) d t} \cdot 100 \%\tag{1}
$$

2. Overall Sensitivity Index (OSI), to quantify the effect of the perturbation on the force produced by all the remaining MT parts:

$$
\mathrm{OSI} = \frac {\sum_ {i \neq p e r t} \int_ {0} ^ {T} | F _ {n e w , i} ^ {M T} (t) - F _ {o l d , i} ^ {M T} (t) | d t}{\sum_ {i \neq p e r t} \int_ {0} ^ {T} F _ {o l d , i} ^ {M T} (t) d t} \cdot 100 \%\tag{2}
$$

where $F _ { o l d , i } ^ { M T } ( t )$ and $F _ { n e w , i } ^ { M T } ( t )$ are the nominal and perturbed values of force, respectively, produced by the perturbed (i pert) and not-perturbed (iapert) MT parts at time step t, and T is the final time of the simulated gait cycle.

A

Finally, Pearson product-moment correlation coefficients r were calculated to assess if the model was most sensitive to potential errors in MT parts presenting the largest $L _ { T } ^ { s } , \overline { { { L _ { f } } } } , \overline { { { F ^ { M } } } }$ and ${ \overline { { \gamma } } } ,$ respectively, to assess if the MT parts that showed the highest LSI values tended to show the highest OSI values too, and to assess if the most sensitive MT parts tended to show the highest sensitivity values for all four MT parameters perturbed. Pearson's r were classified as follows, based on Altman (1990): poor correlation for r o0.20, fair correlation for 0.21o r o0.40, moderate correlation for $0 . 4 1 < | r | < 0 . 6 0 ,$ , good correlation for $0 . 6 1 < | r | < 0 . 8 0 ,$ , very good correlation for $0 . 8 1 < | r | < 1 . 0 0 .$

B  
![](images/1f6925d26abcf8e233e505cbb2973f7e11890225c928a0f7edfffbb533ee9069.jpg)

![](images/c7b1b4865aa44d7a3c9ddec0d7bb863eabd4a31307e455571db8b1b00e30bdf3.jpg)  
Fig, 1. A: TLEM model consisting of 12 body segments: HAT (Head, Arms and Trunk), pelvis, and right and left femur, patella, tibia, talus and foot, The fibula was considerec as one unit in combination with the tibia, The model comprised 11 joints: I5S1 and left and right hip knee, patella/femur, talocrural and subtalar The I5S1 and hip joints were modeled as a ball-and-socket, defined by a rotation center and three orthogonal axes, The knee, talocrural and subtalar joints were defined as a binge, with a fixed rotation center and axis. The patella could rotate with respect to the femur around a rotation axis with a fixed rotation center. The patellar tendon was defined as a non deformable element that connected the patella to the tibia. Thus without introducing an extra Degree of Freedom (DOF). the orientation and position of the patella depended solely on the knee flexion angle. The orientation and position of the center of mass of the pelvis with respect to a 3D global frame, together with the joint rotations of the L5S1, hip, knee, talocrural and subtalar joints, resulted in a model with 21 DOFs. B: Schematic representation of three-elements Hill-type muscle-tendon element (Zajac, 1989). The MT element consists of an active muscle, represented by a contractile element CE in parallel with a passive element $P E ,$ in series with an elastic tendon $T ,$ modeled as a nonlinear spring. $L _ { f }$ is the muscle fiber length $, L _ { T }$ is the tendon length $\cdot \gamma$ is the pennation angle between the muscle fiber and the tendon. $L _ { M T }$ is the total length of the MT element, which is equal to the sum of tendon length L and muscle-fiber length $. L _ { f }$ adjusted by pennation angle $\gamma \left( \mathrm { i . e . } L _ { M T } = L _ { T } + L _ { f } \cos \gamma \right)$ Force in the tendon $F ^ { M \breve { T } }$ is equal to the summed force in the muscle fibers $F ^ { M }$ adjusted by pennation angle gamma $( \mathrm { i } . { \mathsf { e } } . \ \bar { F } ^ { M T } = F ^ { \bar { M } } \cos \gamma )$

## 3. Results

Sensitivity of the MS model predictions to potential errors in Hill MT model parameters presented large variability. Detailed results for all perturbations performed are presented in Table A1 in electronic appendix.

LSI values, representing the sensitivity of the perturbed MT parts, depended strongly on the MT part and the MT parameter perturbed (Fig. 2A and B). $L _ { T } ^ { s }$ was the most sensitive parameter (average LSI 93.83%; max LSI 899.89% for Vastus Lateralis Inferior; min $\mathrm { L S I } { = } 0 . 7 7 \%$ for Pectineus), followed by $\overline { { F ^ { M } } }$ (average LSI 23.08%; max LSI 30.36% for Gemellus Superior; min $\mathrm { L S I } { = } 5 . 9 0 \%$ for Tibialis Anterior), $\overline { { L _ { f } } }$ (average $\mathrm { { L S I } } = 1 3 . 9 3 \% $ max $\mathrm { L S I } { = } 5 3 . 9 3 \%$ for Vastus Medialis Inferior; min $\mathrm { L S I } { = } 0 . 7 6 \%$ for Pectineus) and $\bar { \gamma }$ (average ${ \mathrm { L S I } } = 3 . 3 3 \%$ ; max $\mathrm { L S I } { = } 1 7 . 5 9 \%$ for Gastrocnemius Medialis; min $\mathrm { L S I } { = } 0 . 2 9 \%$ for Tibialis Anterior).

Similarly, OSI values, representing the sensitivity of all the remaining MT parts, depended strongly on the MT part and the MT parameter perturbed (Fig. 2C and D). Again, $L _ { T } ^ { s }$ was the most sensitive parameter (average $0 \mathrm { S I } { = } 3 . 1 6 \% $ max $0 S \mathrm { I } { = } 2 1 . 7 3 \%$ for Gastrocnemius Medialis; min $0 S \mathrm { I } < 0 . 0 1 \%$ for Pectineus), followed by $\overline { { F ^ { M } } }$ (average OSI 0.64%; max OSI 3.32% for Soleus Lateralis; min $0 S \mathrm { I } { = } 0 . 0 3 \% $ for Popliteus), $\overline { { L _ { f } } }$ (average $0 \mathrm { S I } { = } 0 . 4 0 \% $ max

B  
![](images/e766e9bc0072daf4948340c8dfd8019a7c5dabdb7fb5f72d78d96fdb330e8501.jpg)

C  
![](images/cdcefbf2b266f8aba0fae928e22b3d8ff3f1a3f2d25b6efd1f7b0e5bc4e18138.jpg)  
D

OSI 2.95% for Rectus Femoris; min $0 S \mathrm { I } < 0 . 0 1 \%$ for Adductor Magnus Proximal) and $\bar { \gamma }$ (average OSI 0.13%; max OSI 1.86% for Rectus Femoris; min $0 S \mathrm { I } < 0 . 0 1 \%$ for Adductor Brevis Mid).

The highest OSI values were found not only for the MT parts that act as prime movers during gait (Triceps Surae, Quadriceps Femoris, Gluteal muscles, Biceps Femoris Caput Longum), considered amongst the most representative muscles, but also for MT parts that are usually excluded from sensitivity studies, such as Peroneus Longus and Brevis, Obturator Internus, Tibialis Anterior and Piriformis.

We found good correlation between OSI values and nominal values of $\overline { { F ^ { M } } } \ ( r = 0 . 7 2 )$ and $\overline { { \gamma } } \ ( r { = } 0 . 6 7 )$ , hence stronger and more pennated MT parts showed the highest sensitivity, while only poor to fair correlation was found for $L _ { T } ^ { s }$ and $\overline { { L _ { f } } } ( - 0 . 1 7 < r < 0 . 3 4 )$ (Fig. 3A–D). Moreover, we found poor to even negative correlation between LSI and OSI values of the same MT parameter $\left( - 0 . 5 5 < r < 0 . 3 9 \right)$ , hence the MT parts that showed the highest LSI values did not necessarily show the highest OSI values (Fig. 3E– H). Finally, we found moderate to even negative correlation between LSI values of the same MT part for different MT parameters $\left( - 0 . 3 3 < r < 0 . 5 5 \right)$ , but good to very good correlation between OSI values $( 0 . 6 1 < r < 0 . 9 1 )$ , indicating that the most sensitive MT parts tended to show the highest OSI values for all four MT parameters perturbed. Detailed results of correlation analysis are presented in Table 1.

![](images/5dbb3efee96b381789bd438e3a9a3f8b2acad1c8a444ff8e4bb6939a4437a1d9.jpg)

![](images/a6fdbf7122273aa07c816d1b294d57fa95744c3a2cc17a67fedd858c7991d465.jpg)  
Fig. 2. Effect of perturbation of MT parameters on the predicted MT forces during gait. The black solid lines are the nominal MT forces before the perturbation; the blue, red, green and yellow lines are the predicted MT forces after the perturbation of tendon slack length $L _ { T } ^ { s } ,$ , optimal muscle fiber length ${ \overline { { L _ { f } } } } ,$ , maximal isometric muscle force $\overline { { F ^ { M } } }$ , and nominal pennation angle $\overline { { \gamma } } ,$ respectively. A: Force produced by the Rectus Femoris after perturbation of MT parameters of the Rectus Femoris; B: Force produced by the Popliteus after perturbation of MT parameters of the Popliteus: C: Sum of the forces produced by all the remaining MT parts after perturbation of the MT parameters of the Rectus Femoris: D: Sum of the forces produced by all the remaining MT parts after perturbation of MT parameters of the Popliteus. Local Sensitivity Index (LSI) was calculated by integrating the absolute difference between nominal and perturbed MT forces over the simulated gait cycle, and then summing these integrated quantities across all the perturbed MT parts (see Eq. (1)). Overall Sensitivity Index (OSI) was calculated by integrating the absolute difference between nominal and perturbed MT forces over the simulated gait cycle, and then summing these integrated quantities across all the non-perturbed MT parts (see Fa. (2)). Please note that different scales were used for subplots A–B and C–D. (For interpretations of the references to color in this figure legend, the reader is referred to the web version of this article).

![](images/cb9520fa2133719afb4561a62e8b799bdde43754f8dd259ac0251e5b20fb3b54.jpg)

B  
![](images/d6abaa445f4a4a3ce69aec6c2d5ca630d9b96ec15bd1a7d347d34d381c65837d.jpg)

C  
![](images/b7323ae60576af97095adc1db56a3cda33e6535486a2edfafaba4704312f5576.jpg)

D  
![](images/a1ad5381240734aa45542ef2fec9d7cae3f31742fc986e618890005bd20508c4.jpg)

E  
![](images/ab74b2fba981999358ca23ea379b442063c07c515e6a6e6ecd450dcdab064b5d.jpg)

F  
![](images/be567977eeb1e6e13f208d5aaa13a18a6320a65d1b66001960d485c888e985ac.jpg)

G  
![](images/4bc6fc58cc2134e50977619045308defe622ec9d445c5a1ac08709edda37440b.jpg)

H  
![](images/95d5bc9837c0ae6a79878b19d84967a39f59ef7d19f661b71e3eeb829c3f9d2e.jpg)  
Fig. 3. A–D: Overall Sensitivity Index (OSI) values plotted against the nominal value of the corresponding MT parameters perturbed (to assess if the model was most sensitive to potential errors in MT parts presenting the largest $L _ { T } ^ { s } , \overline { { L _ { f } } } , \overline { { F ^ { M } } }$ and γ, respectively). E–H: Overall Sensitivity Index (OSI) values plotted against the corresponding Local Sensitivity Index (LSI) values of the same MT part (to assess if the MT parts that showed the highest LSI values tended to show the highest OSI values too).

## 4. Discussion

The purpose of this extensive sensitivity analysis was to assess the effect of potential errors in Hill MT model parameters on the predictions of a detailed state-of-the art MS model of the lower extremity. In accordance with the results of previous investigations (Hoy et al., 1990; Out et al., 1996; Redl et al., 2007), $L _ { T } ^ { s }$ was found to be the most sensitive parameter, whose perturbation greatly affects the model force predictions, followed by $\overline { { { F } ^ { M } } }$ and ${ \overline { { L _ { f } } } } ,$ while $\bar { \gamma }$ showed very low sensitivity values. However, sensitivity results depended largely also on the specific function of the MT part perturbed (Fig. 2), in accordance with Ackland et al. (2012). OSI values were greatly reduced compared to LSI values, indicating that the complex MS system was able to compensate for perturbations better than the perturbed MT part alone, in agreement with Scovil and Ronsky (2006). Moreover, some MT parts showed very high LSI values but very low OSI values (for example the Popliteus muscle, Fig. 2B–D), indicating that large relative changes in MT parts contributing only little to joint moments did not influence the rest of the model, similarly to De Groote et al. (2010). On the other hand, high OSI values indicated MT parts with an important role during gait, for which potential errors would affect the remaining MT parts (for example the Rectus Femoris muscle, Fig. 2A–C). Also, the most sensitive MT parts tended to show the highest OSI values for all four MT parameters perturbed. For these reasons, OSI values were used as an index to draw up a priority list of the MT parameters and the MT parts that should be estimated most carefully when creating a detailed subject-specific MS model (Table 2).

The highest sensitivity values were found for the prime movers of gait (Triceps Surae, Rectus Femoris, Vastus Lateralis, Biceps Femoris Caput Longum) and hip stabilizers (Gluteal muscles, Obturator Internus, and Piriformis), followed by the Peroneal muscles and Tibialis Anterior. It is important to note that some MT parts, such as Peroneal muscles, Obturator Internus, Tibialis Anterior and Piriformis, which were not included in previous sensitivity studies (Redl et al., 2007; Xiao and Higginson, 2010; Ackland et al., 2012), showed higher sensitivity than other MT parts that are usually considered more representative, such as Iliacus, Adductors, Semitendinosus and

## Table 1

Pearson product-moment correlation coefficients r calculated to measure the linear correlation between the LSI or OSI values and the nominal values of the MT paramete perturbed (to assess if the model was most sensitive to potential errors in MT parts presenting the largest $L _ { T } ^ { s } , \overline { { { L _ { f } } } } , \overline { { { F ^ { M } } } }$ and $\overline { { \gamma } } ,$ respectively), between LSI and OSI values of the same MT part (to assess if the MT parts that showed the highest LSI values tended to show the highest OSI values too), and between the sensitivity of the same MT part for different MT parameters perturbed (to assess if the MT parts that showed the highest OSI values for one MT parameter tended to show the highest OSI values for the other MT parameters as well).

Correlation between LSI values and nominal values of MT parameter perturbed

<table><tr><td> $L_T^s$ </td><td> $\overline{L_f}$ </td><td> $\overline{F^M}$ </td><td> $\overline{\gamma}$ </td></tr><tr><td>0.21</td><td>-0.14</td><td>-0.50</td><td>0.58</td></tr></table>

Correlation between OSI values and nominal values of MT parameter perturbed

<table><tr><td> $L_T^S$ </td><td> $\overline{L_f}$ </td><td> $\overline{F^M}$ </td><td> $\overline{\gamma}$ </td></tr><tr><td>0.34</td><td>-0.17</td><td>0.72</td><td>0.67</td></tr></table>

Correlation between LSI and OSI of the same MT parameter perturbed

<table><tr><td> $L_T^s$ </td><td> $\overline{L_f}$ </td><td> $\overline{F^M}$ </td><td> $\overline{\gamma}$ </td></tr><tr><td>0.15</td><td>0.26</td><td>-0.55</td><td>0.40</td></tr></table>

Correlation between LSI values of different MT parameters perturbed

<table><tr><td>LSI</td><td> $L_T^s$ </td><td> $\overline{L_f}$ </td><td> $\overline{F^M}$ </td><td> $\overline{\gamma}$ </td></tr><tr><td> $L_T^s$ </td><td>1.00</td><td></td><td></td><td></td></tr><tr><td> $\overline{L_f}$ </td><td>0.55</td><td>1.00</td><td></td><td></td></tr><tr><td> $\overline{F^M}$ </td><td>-0.18</td><td>0.05</td><td>1.00</td><td></td></tr><tr><td> $\overline{\gamma}$ </td><td>0.18</td><td>0.07</td><td>-0.33</td><td>1.00</td></tr></table>

Correlation between OSI values of different MT parameters perturbed

<table><tr><td>OSI</td><td> $L_T^s$ </td><td> $\overline{L_f}$ </td><td> $\overline{F^M}$ </td><td> $\overline{\gamma}$ </td></tr><tr><td> $L_T^s$ </td><td>1.00</td><td></td><td></td><td></td></tr><tr><td> $\overline{L_f}$ </td><td>0.79</td><td>1.00</td><td></td><td></td></tr><tr><td> $\overline{F^M}$ </td><td>0.91</td><td>0.77</td><td>1.00</td><td></td></tr><tr><td> $\overline{\gamma}$ </td><td>0.68</td><td>0.61</td><td>0.80</td><td>1.00</td></tr></table>

<table><tr><td>-1.00</td><td>-0.80</td><td>-0.60</td><td>-0.40</td><td>-0.20</td><td>0.00</td><td>0.20</td><td>0.40</td><td>0.60</td><td>0.80</td><td>1.00</td></tr></table>

Priority list of the most sensitive MT parts to potential errors in MT parameters. For each MT part and each MT parameter, mean values of Local Sensitivity Index (LSI) and Overall Sensitivity Index (OSI) over the two perturbations are indicated. Blue cell shading is proportional to LSI values, red cell shading is proportional to OSI values.

<table><tr><td rowspan="2"></td><td colspan="2">Tendon slack length  $L_f^2$ </td><td colspan="2">Optimal muscle fiber length  $L_f$ </td><td colspan="2">Maximal isom. muscle force  $F_M$ </td><td colspan="2">Nominal penn. angle  $\bar{y}$ </td><td rowspan="2">Average OSI</td></tr><tr><td>LSI</td><td>OSI</td><td>LSI</td><td>OSI</td><td>LSI</td><td>OSI</td><td>LSI</td><td>OSI</td></tr><tr><td>Soleus Lateralis</td><td>97.83%</td><td>17.30%</td><td>7.15%</td><td>1.22%</td><td>17.90%</td><td>3.11%</td><td>6.34%</td><td>1.23%</td><td>5.72%</td></tr><tr><td>Gastrocnemius Medialis</td><td>113.09%</td><td>18.43%</td><td>7.50%</td><td>1.08%</td><td>16.63%</td><td>2.41%</td><td>9.11%</td><td>0.12%</td><td>5.51%</td></tr><tr><td>Soleus Medialis</td><td>91.75%</td><td>13.69%</td><td>7.55%</td><td>1.19%</td><td>19.23%</td><td>3.01%</td><td>8.50%</td><td>1.43%</td><td>4.83%</td></tr><tr><td>Rectus Femoris</td><td>68.36%</td><td>11.33%</td><td>12.69%</td><td>2.34%</td><td>14.36%</td><td>2.69%</td><td>9.26%</td><td>1.53%</td><td>4.47%</td></tr><tr><td>Gluteus Medius Anterior</td><td>89.13%</td><td>10.46%</td><td>17.73%</td><td>2.09%</td><td>16.54%</td><td>1.95%</td><td>1.19%</td><td>0.14%</td><td>3.66%</td></tr><tr><td>Peroneus Longus</td><td>95.94%</td><td>8.76%</td><td>7.51%</td><td>0.66%</td><td>14.98%</td><td>1.25%</td><td>1.63%</td><td>0.19%</td><td>2.72%</td></tr><tr><td>Gluteus Minimus Anterior</td><td>162.06%</td><td>8.44%</td><td>37.27%</td><td>1.99%</td><td>19.96%</td><td>1.07%</td><td>1.99%</td><td>0.11%</td><td>2.90%</td></tr><tr><td>Gluteus Medius Posterior</td><td>148.89%</td><td>7.55%</td><td>23.21%</td><td>1.32%</td><td>18.36%</td><td>1.04%</td><td>6.31%</td><td>0.26%</td><td>2.54%</td></tr><tr><td>Vastus Lateralis Superior</td><td>90.17%</td><td>7.27%</td><td>18.76%</td><td>1.41%</td><td>16.15%</td><td>1.16%</td><td>0.53%</td><td>0.05%</td><td>2.47%</td></tr><tr><td>Gastrocnemius Lateralis</td><td>131.12%</td><td>6.43%</td><td>8.10%</td><td>0.39%</td><td>25.12%</td><td>1.11%</td><td>3.81%</td><td>0.24%</td><td>2.04%</td></tr><tr><td>Obturator Internus</td><td>88.09%</td><td>6.01%</td><td>6.87%</td><td>0.48%</td><td>18.18%</td><td>1.27%</td><td>1.09%</td><td>0.08%</td><td>1.96%</td></tr><tr><td>Peroneus Brevis</td><td>96.51%</td><td>5.17%</td><td>7.91%</td><td>0.46%</td><td>18.75%</td><td>0.94%</td><td>3.97%</td><td>0.20%</td><td>1.69%</td></tr><tr><td>Tibialis Anterior</td><td>83.31%</td><td>6.04%</td><td>4.89%</td><td>0.45%</td><td>6.66%</td><td>0.59%</td><td>5.99%</td><td>0.04%</td><td>1.78%</td></tr><tr><td>Biceps Femoris Caput Longum</td><td>74.89%</td><td>2.90%</td><td>13.76%</td><td>0.47%</td><td>16.77%</td><td>0.54%</td><td>7.41%</td><td>0.33%</td><td>1.06%</td></tr><tr><td>Piriformis</td><td>75.94%</td><td>2.73%</td><td>6.79%</td><td>0.25%</td><td>25.19%</td><td>0.90%</td><td>1.59%</td><td>0.06%</td><td>0.98%</td></tr><tr><td>Gluteus Maximus Superior</td><td>76.77%</td><td>2.07%</td><td>22.48%</td><td>0.62%</td><td>24.03%</td><td>0.65%</td><td>1.34%</td><td>0.04%</td><td>0.85%</td></tr><tr><td>Obturator Externus Superior</td><td>76.59%</td><td>2.20%</td><td>13.15%</td><td>0.43%</td><td>22.34%</td><td>0.65%</td><td>0.82%</td><td>0.02%</td><td>0.83%</td></tr><tr><td>Tibialis Posterior Medialis</td><td>90.40%</td><td>2.79%</td><td>10.95%</td><td>0.12%</td><td>23.37%</td><td>0.28%</td><td>8.61%</td><td>0.10%</td><td>0.82%</td></tr><tr><td>Gluteus Minimus Mid</td><td>107.56%</td><td>2.41%</td><td>11.77%</td><td>0.27%</td><td>22.17%</td><td>0.51%</td><td>0.84%</td><td>0.02%</td><td>0.80%</td></tr><tr><td>Flexor Hallucis Longus</td><td>126.61%</td><td>2.61%</td><td>9.54%</td><td>0.14%</td><td>23.63%</td><td>0.33%</td><td>10.78%</td><td>0.13%</td><td>0.80%</td></tr><tr><td>Vastus Intermedius</td><td>169.61%</td><td>2.15%</td><td>42.96%</td><td>0.50%</td><td>25.87%</td><td>0.29%</td><td>5.52%</td><td>0.08%</td><td>0.76%</td></tr><tr><td>Sartorius Distal</td><td>20.03%</td><td>1.13%</td><td>3.98%</td><td>0.22%</td><td>24.66%</td><td>1.33%</td><td>1.55%</td><td>0.09%</td><td>0.69%</td></tr><tr><td>Semitendinosus</td><td>85.42%</td><td>2.09%</td><td>5.13%</td><td>0.11%</td><td>19.87%</td><td>0.47%</td><td>1.21%</td><td>0.03%</td><td>0.67%</td></tr><tr><td>Peroneus Tertius</td><td>76.27%</td><td>1.89%</td><td>8.00%</td><td>0.15%</td><td>21.54%</td><td>0.35%</td><td>6.27%</td><td>0.09%</td><td>0.62%</td></tr><tr><td>Tensor Fasciae Latae</td><td>119.19%</td><td>2.00%</td><td>22.59%</td><td>0.36%</td><td>20.46%</td><td>0.34%</td><td>1.15%</td><td>0.02%</td><td>0.68%</td></tr><tr><td>Sartorius Proximal</td><td>17.48%</td><td>0.81%</td><td>2.65%</td><td>0.13%</td><td>25.56%</td><td>1.14%</td><td>1.72%</td><td>0.08%</td><td>0.54%</td></tr><tr><td>Semimembranosus</td><td>96.75%</td><td>1.58%</td><td>8.85%</td><td>0.14%</td><td>21.64%</td><td>0.34%</td><td>6.34%</td><td>0.10%</td><td>0.54%</td></tr><tr><td>Tibialis Posterior Lateralis</td><td>147.90%</td><td>1.78%</td><td>10.19%</td><td>0.07%</td><td>23.82%</td><td>0.20%</td><td>9.95%</td><td>0.17%</td><td>0.55%</td></tr><tr><td>Vastus Medialis Superior</td><td>163.84%</td><td>1.59%</td><td>39.05%</td><td>0.34%</td><td>26.42%</td><td>0.22%</td><td>2.19%</td><td>0.02%</td><td>0.54%</td></tr><tr><td>Gluteus Minimus Posterior</td><td>76.81%</td><td>1.43%</td><td>5.88%</td><td>0.12%</td><td>24.39%</td><td>0.49%</td><td>0.67%</td><td>0.01%</td><td>0.51%</td></tr><tr><td>Adductor Magnus Distal</td><td>106.47%</td><td>1.43%</td><td>15.99%</td><td>0.22%</td><td>20.98%</td><td>0.28%</td><td>0.73%</td><td>0.01%</td><td>0.49%</td></tr><tr><td>Flexor Digitorum Longus</td><td>100.00%</td><td>1.90%</td><td>16.20%</td><td>0.06%</td><td>20.55%</td><td>0.08%</td><td>6.54%</td><td>0.04%</td><td>0.52%</td></tr><tr><td>Extensor Digitorum Longus</td><td>120.20%</td><td>1.38%</td><td>15.82%</td><td>0.19%</td><td>25.48%</td><td>0.31%</td><td>6.42%</td><td>0.03%</td><td>0.48%</td></tr><tr><td>Vastus Medialis Mid</td><td>170.14%</td><td>1.30%</td><td>53.24%</td><td>0.39%</td><td>26.94%</td><td>0.20%</td><td>5.39%</td><td>0.04%</td><td>0.48%</td></tr><tr><td>Gluteus Maximus Inferior</td><td>26.11%</td><td>0.64%</td><td>14.71%</td><td>0.39%</td><td>25.00%</td><td>0.55%</td><td>1.62%</td><td>0.04%</td><td>0.41%</td></tr><tr><td>Iliacus Mid</td><td>87.53%</td><td>1.11%</td><td>7.95%</td><td>0.08%</td><td>27.64%</td><td>0.36%</td><td>0.60%</td><td>0.01%</td><td>0.39%</td></tr><tr><td>Adductor Longus</td><td>24.97%</td><td>0.57%</td><td>18.50%</td><td>0.41%</td><td>21.89%</td><td>0.46%</td><td>1.72%</td><td>0.04%</td><td>0.37%</td></tr><tr><td>Extensor Hallucis Longus</td><td>112.28%</td><td>0.97%</td><td>29.65%</td><td>0.27%</td><td>26.68%</td><td>0.24%</td><td>2.23%</td><td>0.02%</td><td>0.37%</td></tr><tr><td>Quadratus Femoris</td><td>61.59%</td><td>0.91%</td><td>12.35%</td><td>0.19%</td><td>24.50%</td><td>0.35%</td><td>0.69%</td><td>0.01%</td><td>0.36%</td></tr><tr><td>Gemellus Inferior</td><td>68.40%</td><td>0.92%</td><td>6.30%</td><td>0.09%</td><td>27.47%</td><td>0.37%</td><td>1.74%</td><td>0.02%</td><td>0.35%</td></tr><tr><td>Gracilis</td><td>64.90%</td><td>0.72%</td><td>6.93%</td><td>0.08%</td><td>26.28%</td><td>0.29%</td><td>1.39%</td><td>0.02%</td><td>0.28%</td></tr><tr><td>Vastus Lateralis Inferior</td><td>499.94%</td><td>0.92%</td><td>33.19%</td><td>0.05%</td><td>21.17%</td><td>0.04%</td><td>2.73%</td><td>0.00%</td><td>0.25%</td></tr><tr><td>AdductorMagnusMid</td><td>91.32%</td><td>0.46%</td><td>20.87%</td><td>0.11%</td><td>26.56%</td><td>0.13%</td><td>1.22%</td><td>0.01%</td><td>0.18%</td></tr><tr><td>Plantaris</td><td>242.64%</td><td>0.58%</td><td>13.29%</td><td>0.03%</td><td>24.79%</td><td>0.05%</td><td>1.37%</td><td>0.00%</td><td>0.17%</td></tr><tr><td>Biceps Femoris Caput Breve</td><td>53.10%</td><td>0.34%</td><td>6.83%</td><td>0.04%</td><td>26.88%</td><td>0.21%</td><td>1.69%</td><td>0.01%</td><td>0.15%</td></tr><tr><td>Iliacus Medialis</td><td>52.40%</td><td>0.32%</td><td>4.92%</td><td>0.03%</td><td>28.79%</td><td>0.18%</td><td>1.47%</td><td>0.01%</td><td>0.13%</td></tr><tr><td>Gemellus Superior</td><td>64.56%</td><td>0.31%</td><td>11.32%</td><td>0.06%</td><td>29.11%</td><td>0.14%</td><td>1.62%</td><td>0.01%</td><td>0.13%</td></tr><tr><td>Obturator Externus Inferior</td><td>32.61%</td><td>0.19%</td><td>7.84%</td><td>0.05%</td><td>27.61%</td><td>0.15%</td><td>1.33%</td><td>0.01%</td><td>0.10%</td></tr><tr><td>Vastus Medialis Inferior</td><td>106.59%</td><td>0.23%</td><td>34.63%</td><td>0.07%</td><td>28.12%</td><td>0.06%</td><td>5.89%</td><td>0.01%</td><td>0.09%</td></tr><tr><td>Iliacus Lateralis</td><td>33.99%</td><td>0.15%</td><td>3.93%</td><td>0.02%</td><td>29.08%</td><td>0.14%</td><td>4.51%</td><td>0.06%</td><td>0.09%</td></tr><tr><td>Pectineus</td><td>0.79%</td><td>0.01%</td><td>3.23%</td><td>0.02%</td><td>27.58%</td><td>0.17%</td><td>1.55%</td><td>0.01%</td><td>0.05%</td></tr><tr><td>Adductor Brevis Mid</td><td>8.56%</td><td>0.04%</td><td>7.85%</td><td>0.03%</td><td>27.80%</td><td>0.11%</td><td>0.91%</td><td>0.00%</td><td>0.05%</td></tr><tr><td>Adductor Brevis Proximal</td><td>8.05%</td><td>0.03%</td><td>7.55%</td><td>0.03%</td><td>27.82%</td><td>0.11%</td><td>0.95%</td><td>0.00%</td><td>0.04%</td></tr><tr><td>Popliteus</td><td>138.89%</td><td>0.17%</td><td>12.65%</td><td>0.01%</td><td>24.34%</td><td>0.03%</td><td>1.29%</td><td>0.00%</td><td>0.05%</td></tr><tr><td>Adductor Brevis Distal</td><td>9.90%</td><td>0.04%</td><td>6.80%</td><td>0.03%</td><td>27.79%</td><td>0.10%</td><td>1.02%</td><td>0.00%</td><td>0.04%</td></tr><tr><td>Adductor Magnus Proximal</td><td>10.09%</td><td>0.03%</td><td>2.65%</td><td>0.01%</td><td>27.27%</td><td>0.08%</td><td>1.86%</td><td>0.01%</td><td>0.03%</td></tr><tr><td></td><td colspan="9">min 0.79% 0.01% 2.65% 0.01% 6.66% 0.03% 0.53% 0.00%</td></tr><tr><td></td><td colspan="9">max 499.94% 18.43% 53.24% 2.34% 29.11% 3.11% 10.78% 1.53%</td></tr><tr><td></td><td colspan="9">mean 93.83% 3.16% 13.93% 0.40% 23.08% 0.64% 3.33% 0.13%</td></tr></table>

Semimembranosus, highlighting the importance of an extensive sensitivity analysis when using a detailed MS model.

Sensitivity results showed strong correlation with nominal values of $\overline { { F ^ { M } } }$ and ${ \overline { { \gamma } } } ,$ but poor correlation with $L _ { T } ^ { s }$ and ${ \overline { { L _ { f } } } } ,$ indicating that the order of the proposed priority list was not merely determined by the size and length of the MT parts. Moreover, high sensitivity values were not directly related to high nominal values of muscle activity, but depended deeply on the specific role of the perturbed MT part during gait, and on the capability of the model to compensate to each specific perturbation. Differences in MT parameters had an effect on the capacity of the perturbed MT parts to generate force, hence on the distribution of forces predicted by the muscle recruitment problem. For example, perturbations of the Tibialis Anterior, which represents the biggest contributor to the ankle dorsiflexion, would require a compensation by other minor dorsiflexor MT parts who act also on the subtalar joint, such as Extensor Hallucis Longus, Extensor Digitorum Longus and Peroneus Tertius. This would result in a redistribution of forces generated by other MT parts acting on the ankle/subtalar joint and, through the Gastrocnemius, on the knee joint too, in order to reproduce the nominal net joint moments and minimize the cost function of the static optimization problem. Perturbations of the Peroneal muscles, which represent the biggest contributors to the subtalar eversion, would result in a similar situation. Also, perturbations of the Obturator Internus, which acts only as hip lateral rotator, would require a compensation by other

MT parts acting also as hip abductor/adductors, such as Gluteus Maximus Inferior, Piriformis and Quadratus Femoris, resulting again in a new optimal redistribution of muscle forces to reproduce the nominal net joint moments. Furthermore, sensitivity results could be used as an indication to how much perturbation can be tolerated before normal walking is affected, suggesting that gait is most sensitive to weakness in ankle plantarflexors and hip abductors, and more robust to weakness in other muscles, similarly to what found by van der Krogt et al. (2012).

Since this study was based on the same MS model and metrics used for our previous extensive sensitivity analysis on MS geometry (Carbone et al., 2012), it is possible to directly compare the two priority lists. $L _ { T } ^ { s }$ represented the most sensitive parameter, while perturbations of MT attachment sites showed sensitivity values similar to $\overline { { F ^ { M } } }$ and ${ \overline { { L _ { f } } } } .$ However, it is important to note that MS geometry has a direct effect on the total length of MT elements and, consequently, on their force-generating characteristics. Therefore, we suggest that both MS geometry and MT parameters should be taken into account in order to obtain a reliable and detailed subject-specific model. Moreover, even though the two studies were based on the same model, some MT parts like Iliacus and Adductor Longus presented a relatively high sensitivity to MS geometry but low sensitivity to MT parameters, while other MT parts like Gluteus Maximus showed opposite characteristics. This difference in the ranking is ascribable to the fact that perturbations of MT parameters only affected the force-generating characteristics of the perturbed MT parts, while perturbations of MT attachment sites affected the moment arms and function of the perturbed MT parts, and consequently their torque-generating characteristics. This indicates how extensive sensitivity analyses of different parameters are very important indeed to better understand the complexity of MS models.

It is furthermore important to note that this study was applied only to normal gait, and that sensitivity results are expected to change when analyzing different movements like slow or fast gait and running, as shown before by Scovil and Ronsky (2006). Moreover, this study was based on the gait simulation of a single healthy subject using a generic model, with parameter perturbations based on anatomical variability between healthy subjects. Gait simulations using subject-specific models are expected to show great similarities in the ranking of the most sensitive MT parts, however results could be different when analyzing patients suffering from pathologies affecting their MS system and their movements. Finally, although sensitivity depends on the method used to predict MT forces, the ranking is expected to be similar when using dynamic optimization or forward simulation instead of static optimization, as indicated by De Groote et al. (2010).

In conclusion, this study showed that potential errors in Hill MT model parameters, within the anatomical variability measured in literature, can have a significant impact on MT force predictions, depending on the specific role of the perturbed MT part. Therefore, it is necessary to acquire a good estimate of the most sensitive MT parameters for the most sensitive MT parts, through accurate measurements and detailed optimization, to obtain reliable model outcomes. The proposed priority list provided quantitative information that, combined with our previous extensive sensitivity analysis on MS geometry (Carbone et al., 2012), could be used to define suitable measurement protocols, in order to create detailed subject-specific models that meet the required accuracy of the specific application.

## Conflict of interest statement

The authors do not have any financial or personal relationships with other people or organization that could inappropriately influence their work.

## Acknowledgment

We gratefully acknowledge financial support by the Seventh Framework Programme (FP7) of the European Union (Grant agreement no: 247860) for the TLEMsafe project: http://www.tlemsafe.eu/.

## Appendix A. Supplementary material

Supplementary data associated with this article can be found in the online version at http://dx.doi.org/10.1016/j.jbiomech. 2016.04.008.

## References

Ackland, D.C., Lin, Y.C., Pandy, M.G., 2012. Sensitivity of model predictions of muscle function to changes in moment arms and muscle-tendon properties: a Monte-Carlo analysis. J. Biomech. 45, 1463–1471.

Altman, D.G., 1990. Practical Statistics for Medical Research. Taylor & Francis, London.

Arnold, E.M., Ward, S.R., Lieber, R.L., Delp, S.L., 2010. A model of the lower limb fo analysis of human movement, Ann. Biomed. Eng. 38 269–279

Brand, R.A., Pedersen, D.R., Friederich, J.A., 1986. The sensitivity of muscle force predictions to changes in physiologic cross-sectional area. J. Biomech. 19, 589-596.

Carbone, V., van der Krogt, M.M., Koopman, H.F.J.M., Verdonschot, N., 2012. Sensitivity of subject-specific models to errors in musculo-skeletal geometry. J. Biomech. 45, 2476–2480.

Crowninshield, R.D., Brand, R.A., 1981. A physiologically based criterion of muscle force prediction in locomotion. J. Biomech. 14, 793–801.

Damsgaard, M., Rasmussen, J., Christensen, S.T., Surma, E., de Zee, M., 2006. Analysis of musculoskeletal systems in the AnyBody Modeling System. Simul. Model Pract. Theory 14, 1100–1111

De Groote, F., Van Campen, A., Jonkers, I., De Schutter, J., 2010. Sensitivity of dynamic simulations of gait and dynamometer experiments to hill muscle model parameters of knee flexors and extensors. J. Biomech. 43, 1876–1883.

Friederich, J.A., Brand, R.A., 1990. Muscle fiber architecture in the human lower limb. L Biomech. 23. 91-95

Fukunaga, T., Ichinose, Y., Ito, M., Kawakami, Y., Fukashiro, S., 1997. Determination of fascicle length and pennation in a contracting human muscle in vivo. J. Appl. Physiol. 82, 354–358.

Garner, B.A., Pandy, M.G., 2003. Estimation of musculotendon properties in the human upper limb. Ann. Biomed. Eng. 31, 207–220.

Gerus, P., Sartori, M., Besier, T.F., Fregly, B.J., Delp, S.L., Banks, S.A., Pandy, M.G., D'Lima, D.D., Lloyd, D.G., 2013. Subject-specific knee joint geometry improves predictions of medial tibiofemoral contact forces. J. Biomech. 46, 2778–2786.

Hoy, M.G., Zajac, F.E., Gordon, M.E., 1990. A musculoskeletal model of the human lower extremity: The effect of muscle, tendon, and moment arm on the moment-angle relationship of musculotendon actuators at the hip, knee, and ankle. J. Biomech. 23, 157–169.

Klein Horsman, M.D., Koopman, H.F.J.M., van der Helm, F.C.T., Prosé, L.P., Veeger, H. E.J., 2007. Morphological muscle and joint parameters for musculoskeletal modelling of the lower extremity. Clin. Biomech. 22 239–247

Lenaerts, G., Bartels, W., Gelaude, F., Mulier, M., Spaepen, A., Van der Perre, G., Jonkers, I., 2009. Subject-specific hip geometry and hip joint centre location affects calculated contact forces at the hip during gait. J. Biomech. 42, 1246-1251

Modenese, L., Phillips, A.T.M., Bull, A.M.J., 2011. An open source lower limb model: Hip joint validation. I. Biomech. 44. 2185–2193.

Narici, M.V.. Binzoni, T.. Hiltbrand. E.. Fasel. I.. Terrier. E.. Cerretelli. P.. 1996. In vivo human gastrocnemius architecture with changing joint angle at rest and during graded isometric contraction. I. Physiol, 496 287-297.

Out, L., Vrijkotte, T.G.M., Van Soest, A.J., Bobbert, M.F., 1996. Influence of the parameters of a human triceps surae muscle model on the isometric torqueangle relationship. J. Biomech. Eng. 118, 17–25.

Piazza, S.J., Adamson, R.L., Moran, M.F., Sanders, J.O., Sharkey, N.A., 2003. Effects of tensioning errors in split transfers of tibialis anterior and posterior tendons. J. Bone Joint Surg. Ser. A 85, 858 865.

Redl, C., Gfoehler, M., Pandy, M.G., 2007. Sensitivity of muscle force estimates to variations in muscle-tendon properties. Hum. Mov. Sci. 26, 306–319

Reinbolt, J.A., Fox, M.D., Schwartz, M.H., Delp, S.L., 2009. Predicting outcomes of rectus femoris transfer surgery. Gait Posture 30, 100–105.

Scheys, L., Desloovere, K., Suetens, P., Jonkers, I., 2011. Level of subject-specific detail in musculoskeletal models affects hip moment arm length calculation during gait in pediatric subjects with increased femoral anteversion. J. Biomech. 44, 1346–1353.

Scovil, C.Y., Ronsky, J.L., 2006. Sensitivity of a Hill-based muscle model to perturbations in model parameters. L Biomech 39 2055-2063

Taddei, F., Martelli, S., Valente, G., Leardini, A., Benedetti, M.G., Manfrini, M., Viceconti, M., 2012. Femoral loads during gait in a patient with massive skeletal reconstruction. Clin. Biomech. 27, 273–280.

Valente, G., Pitto, L., Testi, D., Seth, A., Delp, S.L., Stagni, R., Viceconti, M., Taddei, F., 2014. Are Subject-Specific Musculoskeletal Models Robust to the Uncertainties in Parameter Identification? PLoS One 9, e112625

Van Campen, A., Pipeleers, G., De Groote, F., Jonkers, I., De Schutter, J., 2014. A new method for estimating subject-specific muscle-tendon parameters of the knee joint actuators: a simulation study. Int. J. Num. Methods Biomed. Eng. 30, 969–987.

van der Krogt, M.M., Delp, S.L., Schwartz, M.H., 2012. How robust is human gait to muscle weakness? Gait Posture 36, 113–119.

Ward, S.R., Eng, C.M., Smallwood, L.H., Lieber, R.L., 2009. Are current measurements of lower extremity muscle architecture accurate? Clin. Orthop. Related Res. 467, 1074–1082.

Winby, C.R., Lloyd, D.G., Kirk, T.B., 2008. Evaluation of different analytical methods for subject-specific scaling of musculotendon parameters. J. Biomech. 41, 1682–1688.

Xiao, M., Higginson, J., 2010. Sensitivity of estimated muscle force in forward simulation of normal walking. J. Applied Biomech. 26, 142–149.

Zajac, F.E., 1989. Muscle and tendon: properties, models, scaling, and application to biomechanics and motor control. Crit. Rev. Biomed. Eng. 17, 359–411.