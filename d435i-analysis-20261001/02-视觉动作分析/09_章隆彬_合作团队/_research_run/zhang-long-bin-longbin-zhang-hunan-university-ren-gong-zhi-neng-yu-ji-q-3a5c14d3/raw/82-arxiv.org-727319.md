<!-- source: http://arxiv.org/abs/2604.05701v1 -->
<!-- title: Measurement of the CKM angle $γ$ in $B^{\pm} \rightarrow D(\rightarrow K^{0}_{\rm S} h^{\prime+}h^{\prime-})h^{\pm}$ decays with a novel approach -->
<!-- fetched: 2026-09-30T23:47:34Z -->
<!-- via: arxiv_search("章隆彬 Longbin Zhang Hunan University 人工智能与机器人学院 rehabilitation robotics exoskeleton neuromusculoskeletal model surface EMG markerless motion capture brain computer interface muscle tendon estimation joint torque prediction gait balance assessment 实验室团队 合作网络") -->

1Introduction

  
    
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

1 Introduction

2 Analysis strategy

3 Detectors and simulation

4 Event selection and signal extraction

5 Observables from data

6 Fit to observables from data

6.1 Joint fit to determine 𝑪​𝑷\boldsymbol{{C\!P}} observables

6.2 Determination of strong-phase parameters

7 Systematic uncertainties

8 Results of 𝜸\gamma

9 Conclusions

A 𝑫\boldsymbol{D}-candidate mass fit at BESIII

B Alternative fits to strong-phase parameters

C Correlation matrices

References

  
    License: CC BY 4.0
  
  
arXiv:2604.05701v1 [hep-ex] 07 Apr 2026

EUROPEAN ORGANIZATION FOR NUCLEAR RESEARCH (CERN)

INSTITUTE OF HIGH ENERGY PHYSICS (IHEP)

​​​
 

CERN-EP-2026-067

LHCb-PAPER-2025-063

April 7, 2026

Measurement of the CKM angle 𝜸\gamma in B±→D(→K𝐒𝟎h′⁣+h′⁣−)h±{{B}^{\pm}}\!\rightarrow D(\rightarrow{{K}^{0}_{\mathrm{S}}}h^{\prime+}h^{\prime-})h^{\pm} decays with a novel approach

BESIII and LHCb collaborations††
          †
          
          
          
        Authors are listed at the end of this paper.

      
A measurement of the CKM angle γ\gamma and related strong-phase parameters is performed using a novel, model-independent approach in B±→D(→KS0h′⁣+h′⁣−)h±{{B}^{\pm}}\!\rightarrow D(\rightarrow{{K}^{0}_{\mathrm{S}}}h^{\prime+}h^{\prime-})h^{\pm} decays, where h(′)≡π,Kh^{(\prime)}\equiv\pi,K.
The analysis uses a joint data sample of electron-positron collisions collected by the BESIII experiment at the Beijing Electron-Positron Collider II during 2010–2011 and 2021–2022, corresponding to an integrated luminosity of 8 fb−1\text{\,fb}^{-1}, and proton-proton collisions collected by the LHCb experiment at the Large Hadron Collider during 2011–2018,
corresponding to an integrated luminosity of 9 fb−1\text{\,fb}^{-1}.
The two datasets are analyzed simultaneously by applying per-event weights based on the amplitude variation over the DD-decay phase space to enhance the sensitivity to C​PC\!P-violating observables.
The CKM angle γ\gamma is determined to be γ=(71.3±5.0)∘\gamma=(71.3\pm 5.0){{}^{\circ}}, which constitutes the most precise single measurement to date.

    

Submitted to Phys. Rev. D

© 2026 CERN for the benefit of the LHCb collaboration and

IHEP for the benefit of the BESIII collaboration. CC BY 4.0 licence.

 

1 Introduction

In the Standard Model of particle physics (SM), the single irreducible complex phase in the Cabibbo–Kobayashi–Maskawa (CKM) quark-mixing matrix [1, *Kobayashi:1973fv] is responsible for all the C​PC\!P-violating phenomena in the quark sector.
Assuming unitarity of the CKM matrix, the relation Vu​d​Vu​b∗+Vc​d​Vc​b∗+Vt​d​Vt​b∗=0{{V_{{u}{d}}^{\phantom{\ast}}}{V_{{u}{b}}^{\ast}}+{V_{{c}{d}}^{\phantom{\ast}}}{V_{{c}{b}}^{\ast}}+{V_{{t}{d}}^{\phantom{\ast}}}{V_{{t}{b}}^{\ast}}=0} (Vx​yV_{xy} represents the transition amplitude from quark xx to quark yy) forms a closed triangle in the complex plane, known as the Unitarity Triangle (UT),
which has been extensively studied due to the high experimental accessibility of its sides and angles [3, 4, 5].
Among its three internal angles, γ=ϕ3≡arg(−Vu​dVu​b∗/Vc​dVc​b∗){\gamma=\phi_{3}\equiv\arg(-{V_{{u}{d}}^{\phantom{\ast}}}{V_{{u}{b}}^{\ast}}/{V_{{c}{d}}^{\phantom{\ast}}}{V_{{c}{b}}^{\ast}})} can be measured experimentally by exploiting interference between b→c​u¯​sb\!\rightarrow c{\overline{{u}}}s and b→u​c¯​sb\!\rightarrow u{\overline{{c}}}s transitions at leading order within B→D(∗)​K(∗)B\!\rightarrow D^{(*)}K^{(*)} decays.
The extraction of γ\gamma in these decays has negligible theoretical uncertainty, as small as 𝒪⁡(10−7){\mathcal{O}}(10^{-7}) [6].
The angle γ\gamma can also be measured with charmless BB-meson decays [7, 8, 9, 10, 11, 12, 13].
Comparison of the results measured with the two methods can provide insights into potential new-physics contributions entering loop-mediated processes within charmless bb-hadron decays.
In addition to direct measurements from B→D(∗)​K(∗)B\!\rightarrow D^{(*)}K^{(*)} decays,
indirect determinations of γ\gamma are obtained from global fits to other CKM parameters under the assumption of unitarity [3, 4, 5].
Since some of these inputs arise from loop-level processes, discrepancies between direct and indirect determinations would indicate the existence of new physics.

The world-average values of γ\gamma are obtained from global fits to various measurements that constrain the CKM matrix elements.
When direct measurements of γ\gamma are excluded, the results are found to be γ=(66.29−1.86+0.72)∘{\gamma=(66.29^{+0.72}_{-1.86}){{}^{\circ}}} from CKMfitter [3, 4] and γ=(64.9±1.4)∘{\gamma=(64.9\pm 1.4){{}^{\circ}}} from UTfit [5].
A recent combination of direct measurements by the LHCb collaboration yields γ=(62.8±2.6)∘{\gamma=(62.8\pm 2.6){{}^{\circ}}} [14],
while a combination of BaBar measurements yields γ=(69−16+17)∘\gamma=(69^{+17}_{-16}){{}^{\circ}} [15], and a combination of Belle and Belle II measurements yields γ=(75.2±7.6)∘\gamma=(75.2\pm 7.6){{}^{\circ}} [16].
These direct measurements are consistent with the indirect constraints but have larger uncertainties.
Hence, larger data samples and optimized analysis techniques are required to improve the precision on γ\gamma.

The most precise direct measurement of γ\gamma to date is from the LHCb experiment through the decays B±→D(→KS0h′⁣+h′⁣−)h±{{B}^{\pm}}\!\rightarrow D(\rightarrow{{K}^{0}_{\mathrm{S}}}h^{\prime+}h^{\prime-})h^{\pm}, where h(′)h^{(\prime)} is either a π\pi or KK meson, and DD represents a superposition of the D0{{D}^{0}} and D¯0{{\kern 1.79993pt\overline{\kern-1.79993ptD}}{}^{0}} mesons [17].
The analysis employed a binned phase-space approach as proposed in Refs. [18, 19, 20], benefiting from the local asymmetries across the phase space.
The reported result of γ=(68.7−5.1+5.2)∘{\gamma=(68.7^{+5.2}_{-5.1}){{}^{\circ}}} is statistically limited.
Since the binned phase-space approach only exploits about 85% of the sensitivity to γ\gamma [21], further optimization of the analysis method is desirable.
A novel approach that applies per-event weights to data has been proposed as an alternative [22].
This method for the determination of γ\gamma has been found to use the phase-space information more effectively than the binned approach.

The strong-phase differences between the decay amplitudes of D0{{D}^{0}} and D¯0{{\kern 1.79993pt\overline{\kern-1.79993ptD}}{}^{0}} to the KS0​h′⁣+​h′⁣−{{{K}^{0}_{\mathrm{S}}}h^{\prime+}h^{\prime-}} final states are essential inputs for determining γ\gamma.
However, uncertainties due to model assumptions of the decay amplitudes are difficult to quantify.
In practice, the strong-phase differences are measured directly from quantum-correlated charm decays typically produced at the ψ⁡(3770)\psi(3770) resonance. The strong-phase differences for the binned approach have been measured by the CLEO [21] and BESIII experiments [23, 24, 25, 26], and are statistically limited.
These uncertainties introduce a non-negligible systematic uncertainty to γ\gamma.
However, these inputs can only be used by the binned approach.

This paper, which expands upon the accompanying Letter [27], presents a more precise determination of the CKM angle γ\gamma by applying the novel approach to the same LHCb dataset used in Ref. [17].
The required strong-phase parameters for this method are obtained from a joint measurement using quantum-correlated D​D¯{D{{\kern 1.79993pt\overline{\kern-1.79993ptD}}}} decays produced at the BESIII experiment.
The BESIII data were collected during 2010–2011 and 2021–2022 in e+​e−{e^{+}e^{-}} collisions at the ψ⁡(3770){\psi(3770)} threshold, with an integrated luminosity of 8 fb−1\text{\,fb}^{-1}, where the signal DD mesons are reconstructed in the KS/L0​h′⁣+​h′⁣−K^{0}_{\mathrm{S/L}}h^{\prime+}h^{\prime-} final states.
The LHCb data were collected during 2011–2018 in p​ppp collisions at center-of-mass energies of s=7,8{\sqrt{s}=7,8} and 13 TeV, corresponding to an integrated luminosity of 9 fb−1\text{\,fb}^{-1}.
The decay channels are the same as those in Ref. [17], namely B±→D​K±{{B}^{\pm}}\!\rightarrow D{{K}^{\pm}} and B±→D​π±{{B}^{\pm}}\!\rightarrow D{{\pi}^{\pm}},
where the DD meson is reconstructed via the self-conjugate decays D→KS0​h′⁣+​h′⁣−D\!\rightarrow{{K}^{0}_{\mathrm{S}}}h^{\prime+}h^{\prime-}.
In this analysis, both the BESIII and LHCb datasets have also been analyzed using the binned phase-space approach [26, 17] except for the D→KS/L0​K+​K−D\!\rightarrow K^{0}_{\mathrm{S/L}}{{K}^{+}}{{K}^{-}} signal in BESIII data.

The outline of this paper is as follows: the formalism of the novel approach is explained in Sect. 2;
the BESIII and LHCb detectors are introduced in Sect. 3;
selection of candidates from data and extraction of signal yields are described in Sect. 4,
while Sect. 5 presents the method to obtain key observables from data;
the C​PC\!P observables and strong-phase parameters are obtained using the fits described in Sect. 6;
the corresponding systematic uncertainties are analyzed in Sect. 7;
Section 8 presents the results of the measurements of C​PC\!P observables and strong-phase parameters, together with the interpretation of C​PC\!P observables determining the CKM angle γ\gamma; and
finally, Sect. 9 summarizes the measurement.

2 Analysis strategy

The amplitude of the B−→D​h−{{{B}^{-}}}\!\rightarrow Dh^{-} decay is a superposition of the favored B−→D0​h−{{{B}^{-}}}\!\rightarrow{{D}^{0}}h^{-} and suppressed B−→D¯​h−0{{{B}^{-}}}\!\rightarrow{{\kern 1.79993pt\overline{\kern-1.79993ptD}}{}^{0}}h^{-} decays,

AB∝AD+rBD​h​ei⁡(δBD​h−γ)​A¯D,A_{B}\propto A_{D}+r_{B}^{Dh}e^{i(\delta_{B}^{Dh}-\gamma)}\overline{A}_{D},

(1)

where ADA_{D} (A¯D\overline{A}_{D}) is the amplitude of the D0(D¯)0→KS0h′⁣+h′⁣−{{D}^{0}}({{\kern 1.79993pt\overline{\kern-1.79993ptD}}{}^{0}})\!\rightarrow{{K}^{0}_{\mathrm{S}}}h^{\prime+}h^{\prime-} decay.
The amplitude of the B+→D​h+{{{B}^{+}}}\!\rightarrow Dh^{+} decay is obtained by substituting −γ-\gamma with +γ+\gamma and exchanging ADA_{D} and A¯D\overline{A}_{D}.
The hadronic parameters rBD​hr_{B}^{Dh} and δBD​h\delta_{B}^{Dh} are the ratio and strong-phase difference between the suppressed and favored B−{{{B}^{-}}} decay amplitudes, respectively.
As the decay phase space has two degrees of freedom, often referred to as the Dalitz plot [28, 29], the amplitudes are expressed as functions of the squared KS0​h′⁣±{{K}^{0}_{\mathrm{S}}}h^{\prime\pm} masses, m±2m^{2}_{\pm}.
Neglecting C​PC\!P violation and mixing in DD decays, as their impact on the phase-space distributions is expected to be well below current experimental sensitivity [30], the amplitudes are defined as AD≡AD​(𝐳){A_{D}\equiv A_{D}(\mathbf{z})} and A¯D≡A¯D​(𝐳)=AD​(𝐳¯){\overline{A}_{D}\equiv\overline{A}_{D}(\mathbf{z})=A_{D}(\overline{\mathbf{z}})} with 𝐳≡(m+2,m−2)\mathbf{z}\equiv(m^{2}_{+},m^{2}_{-}) and 𝐳¯≡(m−2,m+2)\overline{\mathbf{z}}\equiv(m^{2}_{-},m^{2}_{+}).

The rates of the B−{{B}^{-}} and B+{{B}^{+}} decays, pBp_{B} and p¯B\overline{p}_{B}, can be derived as

pB​(𝐳)\displaystyle p_{B}(\mathbf{z})
∝pD​(𝐳)+[(x−D​h)2+(y−D​h)2]​p¯D​(𝐳)+2​[x−D​h​𝒞​(𝐳)+y−D​h​𝒮​(𝐳)],\displaystyle\propto p_{D}(\mathbf{z})+\left[(x_{-}^{Dh})^{2}+(y_{-}^{Dh})^{2}\right]\overline{p}_{D}(\mathbf{z})+2\left[x_{-}^{Dh}\mathcal{C}(\mathbf{z})+y_{-}^{Dh}\mathcal{S}(\mathbf{z})\right],

(2)

p¯B​(𝐳)\displaystyle\overline{p}_{B}(\mathbf{z})
∝p¯D​(𝐳)+[(x+D​h)2+(y+D​h)2]​pD​(𝐳)+2​[x+D​h​𝒞​(𝐳)−y+D​h​𝒮​(𝐳)],\displaystyle\propto\overline{p}_{D}(\mathbf{z})+\left[(x_{+}^{Dh})^{2}+(y_{+}^{Dh})^{2}\right]p_{D}(\mathbf{z})+2\left[x_{+}^{Dh}\mathcal{C}(\mathbf{z})-y_{+}^{Dh}\mathcal{S}(\mathbf{z})\right],

where 
 
( ) 
[-.7ex] p 
D≡|
 
( ) 
[-.7ex] A 
D|2\shortstack{{(\rule[2.15277pt]{12.50002pt}{0.51212pt})}
\\
 [-.7ex] $p$}_{\hskip-2.8681ptD}\equiv\big|\shortstack{{(\rule[2.15277pt]{12.50002pt}{0.51212pt})}
\\
 [-.7ex] $A$}_{\hskip-2.04861ptD}\big|^{2} is the decay rate of the D0(D¯)0→KS0h′⁣+h′⁣−{{D}^{0}}({{\kern 1.79993pt\overline{\kern-1.79993ptD}}{}^{0}})\!\rightarrow{{K}^{0}_{\mathrm{S}}}h^{\prime+}h^{\prime-} process.
The C​PC\!P observables, x±D​hx_{\pm}^{Dh} and y±D​hy_{\pm}^{Dh}, are expressed as

x±D​h=rBD​h​cos⁡(δBD​h±γ),y±D​h=rBD​h​sin⁡(δBD​h±γ).x_{\pm}^{Dh}=r_{B}^{Dh}\cos(\delta_{B}^{Dh}\pm\gamma),\hskip 10.00002pty_{\pm}^{Dh}=r_{B}^{Dh}\sin(\delta_{B}^{Dh}\pm\gamma).

(3)

The functions 𝒞\mathcal{C} and 𝒮\mathcal{S} are related to the strong-phase difference between D0{{D}^{0}} and D¯0{{\kern 1.79993pt\overline{\kern-1.79993ptD}}{}^{0}} decays,

𝒞⁡(𝐳)\displaystyle\mathcal{C}(\mathbf{z})
≡|AD​(𝐳)||AD​(𝐳¯)|cos⁡[δD​(𝐳)−δD​(𝐳¯)],\displaystyle\equiv|A_{D}(\mathbf{z})||A_{D}(\overline{\mathbf{z}})|\cos\left[\delta_{D}(\mathbf{z})-\delta_{D}(\overline{\mathbf{z}})\right],

(4)

𝒮⁡(𝐳)\displaystyle\mathcal{S}(\mathbf{z})
≡|AD​(𝐳)||AD​(𝐳¯)|sin⁡[δD​(𝐳)−δD​(𝐳¯)],\displaystyle\equiv|A_{D}(\mathbf{z})||A_{D}(\overline{\mathbf{z}})|\sin\left[\delta_{D}(\mathbf{z})-\delta_{D}(\overline{\mathbf{z}})\right],

where δD\delta_{D} is the strong phase of the D0→KS0​h′⁣+​h′⁣−{{D}^{0}}\!\rightarrow{{K}^{0}_{\mathrm{S}}}h^{\prime+}h^{\prime-} decay amplitude.

The C​PC\!P observables x±D​hx_{\pm}^{Dh} and y±D​hy_{\pm}^{Dh} are independent of the phase-space coordinates. Therefore, a phase-space-dependent weighting function can be applied to Eq. 2, while x±D​hx_{\pm}^{Dh} and y±D​hy_{\pm}^{Dh} remain unchanged, and the weighted integrals of the decay rates can then be evaluated experimentally.
The binned phase-space approach [18, 19, 20] is a special case in that regard, using a uniform weighting function within different regions of the Dalitz plot.

A novel approach, referred to as the Fourier split method, has been proposed in Ref. [22].
It applies Fourier expansions of the strong-phase difference ϕ⁡(𝐳)≡δD​(𝐳)−δD​(𝐳¯){\phi(\mathbf{z})\equiv\delta_{D}(\mathbf{z})-\delta_{D}(\overline{\mathbf{z}})} to the functions in Eq. 2, which is equivalent to applying per-event Fourier weights cos⁡(k​ϕ)\cos(k\phi) and sin⁡(k​ϕ)\sin(k\phi).
The distributions of ϕ⁡(𝐳)\phi(\mathbf{z}), shown in Fig. 1, are derived from amplitude models measured by the Belle and BaBar experiments [31, 32, 33].
As in the case of the binned approach, the choice of model affects the sensitivity to γ\gamma but does not bias the results.
This method achieves a lower statistical uncertainty on γ\gamma than the binned approach by making use of the included intra-bin variation of the strong-phase difference across the phase space.

The sensitivity to γ\gamma also depends on the decay rate pDp_{D} and the background level in data, which were not considered in Ref. [22].
To improve the sensitivity to γ\gamma, an additional weighting function is introduced, which is denoted as the optimal weight, woptw^{\rm opt}.
The optimal weights, as shown in Fig. 2, are obtained using amplitude models for the DD decay, which originate from the Belle and BaBar measurements [31, 32, 33].
The weights also take into account the effects of efficiency and background in the LHCb measurement [17].
The structure of these weights across the Dalitz plot reflects the statistical uncertainty of signal events, showing enhancement and suppression around the K∗​(892)±{{K}^{*}}(892)^{\pm} (ϕ⁡(1020)\phi(1020)) resonance in the D→KS0​π+​π−​(KS0​K+​K−)D\!\rightarrow{{K}^{0}_{\mathrm{S}}}{{\pi}^{+}}{{\pi}^{-}}~({{K}^{0}_{\mathrm{S}}}{{K}^{+}}{{K}^{-}}) decay.
The values of these optimal weights across the Dalitz plots are provided in a HEPData record [34].
Analogous to the symmetric splitting of the phase space by m+2>(<)​m−2m^{2}_{+}>(<)m^{2}_{-} in the binned approach,
the diametrically opposed optimal weight, wopt​(𝐳¯){w^{\rm opt}(\overline{\mathbf{z}})}, is also applied to improve the sensitivity.

Figure 1: Distributions of the strong-phase difference ϕ⁡(m+2,m−2)\phi(m^{2}_{+},m^{2}_{-}) across the (left) D→KS0​π+​π−D\!\rightarrow{{K}^{0}_{\mathrm{S}}}{{\pi}^{+}}{{\pi}^{-}} and (right) D→KS0​K+​K−D\!\rightarrow{{K}^{0}_{\mathrm{S}}}{{K}^{+}}{{K}^{-}} Dalitz plots. They are derived from amplitude models measured by the Belle and BaBar [31, 32, 33] experiments.

Figure 2: Distributions of the optimal weight woptw^{\rm opt} across the (left) D→KS0​π+​π−D\!\rightarrow{{K}^{0}_{\mathrm{S}}}{{\pi}^{+}}{{\pi}^{-}} and (right) D→KS0​K+​K−D\!\rightarrow{{K}^{0}_{\mathrm{S}}}{{K}^{+}}{{K}^{-}} Dalitz plots. They are obtained from amplitude models measured by the Belle and BaBar [31, 32, 33] experiments, and also take into account the effects of efficiency and background in the LHCb measurement [17]. The integrals of the weighting functions are normalized to unity.

In summary, the complete set of weighting functions combines the optimal weights woptw^{\rm opt} with the Fourier weights proposed in Ref. [22],

wn​(𝐳)\displaystyle w_{n}(\mathbf{z})
={wopt​(𝐳)​cos⁡[k​ϕ​(𝐳)],n=2​k,wopt​(𝐳)​sin⁡[k​ϕ​(𝐳)],n=2​k−1.\displaystyle=\left\{\begin{array}[]{ll}w^{\rm opt}(\mathbf{z})\cos\left[k\phi(\mathbf{z})\right],&n=2k,\\
w^{\rm opt}(\mathbf{z})\sin\left[k\phi(\mathbf{z})\right],&n=2k-1.\end{array}\right.

(5)

w¯n​(𝐳)\displaystyle\overline{w}_{n}(\mathbf{z})
={wopt​(𝐳¯)​cos⁡[k​ϕ​(𝐳)],n=2​k,wopt​(𝐳¯)​sin⁡[k​ϕ​(𝐳)],n=2​k−1.\displaystyle=\left\{\begin{array}[]{ll}w^{\rm opt}(\overline{\mathbf{z}})\cos\left[k\phi(\mathbf{z})\right],&n=2k,\\
w^{\rm opt}(\overline{\mathbf{z}})\sin\left[k\phi(\mathbf{z})\right],&n=2k-1.\end{array}\right.

With a maximum Fourier order of MM, kk enumerates from 0 (1) to MM for the cosine (sine) weights,
and 2×(2​M+1)2\times(2M+1) weighting functions are defined.
This novel approach is denoted as the optimal Fourier method.

The weighted integrals of the B±{{B}^{\pm}} decay rates yield 4×(2​M+1)4\times(2M+1) parameters,

𝒩n+\displaystyle\mathcal{N}_{n}^{+}
≡hB+​[Pn−+((x+D​h)2+(y+D​h)2)​Pn++2​(−1)n​(x+D​h​Cn−y+D​h​Sn)],\displaystyle\equiv h_{{{{B}^{+}}}}[P_{n}^{-}+\left((x_{+}^{Dh})^{2}+(y_{+}^{Dh})^{2}\right)P_{n}^{+}+2(-1)^{n}(x_{+}^{Dh}C_{n}-y_{+}^{Dh}S_{n})],

(6)

𝒩¯n+\displaystyle\smash{\overline{\mathcal{N}}}_{n}^{+}
≡hB+​[Pn++((x+D​h)2+(y+D​h)2)​Pn−+2​(−1)n​(x+D​h​Cn+y+D​h​Sn)],\displaystyle\equiv h_{{{{B}^{+}}}}[P_{n}^{+}+\left((x_{+}^{Dh})^{2}+(y_{+}^{Dh})^{2}\right)P_{n}^{-}+2(-1)^{n}(x_{+}^{Dh}C_{n}+y_{+}^{Dh}S_{n})],

𝒩n−\displaystyle\mathcal{N}_{n}^{-}
≡hB−​[Pn++((x−D​h)2+(y−D​h)2)​Pn−+2​(−1)n​(x−D​h​Cn+y−D​h​Sn)],\displaystyle\equiv h_{{{{B}^{-}}}}[P_{n}^{+}+\left((x_{-}^{Dh})^{2}+(y_{-}^{Dh})^{2}\right)P_{n}^{-}+2(-1)^{n}(x_{-}^{Dh}C_{n}+y_{-}^{Dh}S_{n})],

𝒩¯n−\displaystyle\smash{\overline{\mathcal{N}}}_{n}^{-}
≡hB−​[Pn−+((x−D​h)2+(y−D​h)2)​Pn++2​(−1)n​(x−D​h​Cn−y−D​h​Sn)],\displaystyle\equiv h_{{{{B}^{-}}}}[P_{n}^{-}+\left((x_{-}^{Dh})^{2}+(y_{-}^{Dh})^{2}\right)P_{n}^{+}+2(-1)^{n}(x_{-}^{Dh}C_{n}-y_{-}^{Dh}S_{n})],

where separate normalization factors hB±h_{{{B}^{\pm}}} for B±{{B}^{\pm}} decays are applied, thereby accounting for any production and detection asymmetry in the B±{{B}^{\pm}} decays.
The parameters Pn+P_{n}^{+}, Pn−P_{n}^{-}, CnC_{n} and SnS_{n} are wnw_{n}-weighted integrals of the pDp_{D}, p¯D\overline{p}_{D}, 𝒞\mathcal{C} and 𝒮\mathcal{S} functions, respectively.
In addition to hB±h_{{{B}^{\pm}}} that are proportional to the signal yields,
a normalization over the amplitude is applied by requiring P0++P0−=1P_{0}^{+}+P_{0}^{-}=1.
The weighted parameters are defined as

Xn+{−}≡∫fX​(𝐳)​wn​{w¯n}​(𝐳)​𝑑𝐳∫[pD​(𝐳)+p¯D​(𝐳)]​wopt​(𝐳)​𝑑𝐳,X_{n}^{+\{-\}}\equiv\frac{\int f_{X}(\mathbf{z})w_{n}\{\overline{w}_{n}\}(\mathbf{z})\mathrm{d}\mathbf{z}}{\int\left[p_{D}(\mathbf{z})+\overline{p}_{D}(\mathbf{z})\right]w^{\rm opt}(\mathbf{z})\mathrm{d}\mathbf{z}},

(7)

where fXf_{X} denotes the functions pDp_{D}, 𝒞\mathcal{C} and 𝒮\mathcal{S} for X=PX=P, CC and SS, respectively.
Implicit definitions

Cn≡Cn+=(−1)n​Cn−,Sn≡Sn+=(−1)n+1​Sn−,C_{n}\equiv C_{n}^{+}=(-1)^{n}C_{n}^{-},\hskip 10.00002ptS_{n}\equiv S_{n}^{+}=(-1)^{n+1}S_{n}^{-},

(8)

are applied, because the 𝒞\mathcal{C} function and cosine weights are symmetric, while the 𝒮\mathcal{S} function and sine weights are antisymmetric across the Dalitz plot.
However, due to the asymmetric nature of woptw^{\rm opt}, weighted 𝒞\mathcal{C} and 𝒮\mathcal{S} functions are no longer symmetric or antisymmetric, respectively, which yields nonzero CnC_{n} and SnS_{n} terms at every sine and cosine order. This relationship is identical to that in the binned method.

By including both B±→D​K±{{B}^{\pm}}\!\rightarrow D{{K}^{\pm}} and B±→D​π±{{B}^{\pm}}\!\rightarrow D{{\pi}^{\pm}} decays, the above equations contain 8×(2​M+1)8\times(2M+1) observables when the Fourier order kk enumerates from 0 to MM.
The number of free parameters is 2×(2​M+1)+122\times(2M+1)+12 when the strong-phase parameters Cn,SnC_{n},S_{n} are constrained by the BESIII dataset.
As a result, under the condition M≥1M\geq 1, these observables are sufficient to determine all the parameters.

As the B±→D​K±{{B}^{\pm}}\!\rightarrow D{{K}^{\pm}} and B±→D​π±{{B}^{\pm}}\!\rightarrow D{{\pi}^{\pm}} processes have the same weak phase difference γ\gamma,
only two additional observables are needed to describe the B±→D​π±{{B}^{\pm}}\!\rightarrow D{{\pi}^{\pm}} decay. They are defined as

xξD​π≡rBD​πrBD​K​cos⁡(δBD​π−δBD​K),yξD​π=rBD​πrBD​K​sin⁡(δBD​π−δBD​K),x_{\xi}^{D\pi}\equiv\frac{r_{B}^{D\pi}}{r_{B}^{DK}}\cos(\delta_{B}^{D\pi}-\delta_{B}^{DK}),\hskip 10.00002pty_{\xi}^{D\pi}=\frac{r_{B}^{D\pi}}{r_{B}^{DK}}\sin(\delta_{B}^{D\pi}-\delta_{B}^{DK}),

(9)

and the C​PC\!P observables of the B±→D​π±{{B}^{\pm}}\!\rightarrow D{{\pi}^{\pm}} channel can be determined using

x±D​π≡xξD​π​x±D​K−yξD​π​y±D​K,y±D​π=xξD​π​y±D​K+yξD​π​x±D​K.x_{\pm}^{D\pi}\equiv x_{\xi}^{D\pi}x_{\pm}^{DK}-y_{\xi}^{D\pi}y_{\pm}^{DK},\hskip 10.00002pty_{\pm}^{D\pi}=x_{\xi}^{D\pi}y_{\pm}^{DK}+y_{\xi}^{D\pi}x_{\pm}^{DK}.

(10)

This reduces the number of free parameters to 2×(2​M+1)+102\times(2M+1)+10.

In the quantum-correlated ψ(3770)→D0D¯0{\psi(3770)}\!\rightarrow{{D}^{0}}{{\kern 1.79993pt\overline{\kern-1.79993ptD}}{}^{0}} system, the decay amplitude can be written as

A=12​(AD(1)​A¯D(2)−A¯D(1)​AD(2)),A=\frac{1}{\sqrt{2}}\left(A_{D}^{(1)}\overline{A}_{D}^{(2)}-\overline{A}_{D}^{(1)}A_{D}^{(2)}\right),

(11)

where the superscripts (1) and (2) uniquely label the two DD mesons,
and AD(i)A_{D}^{(i)} (A¯D(i)\overline{A}_{D}^{(i)}) is the decay amplitude of the corresponding D0{{D}^{0}} (D¯0{{\kern 1.79993pt\overline{\kern-1.79993ptD}}{}^{0}}) decay.
The strong-phase parameters, Cn,SnC_{n},S_{n}, are measured in this system by a double-tag (DT) method [35],
in which one of the DD mesons is reconstructed as the signal decay,
and the other is reconstructed as a tag decay.
In addition to D→KS0​h′⁣+​h′⁣−D\!\rightarrow{{K}^{0}_{\mathrm{S}}}h^{\prime+}h^{\prime-} decays, D→KL0​h′⁣+​h′⁣−D\!\rightarrow{{K}^{0}_{\mathrm{L}}}h^{\prime+}h^{\prime-} decays are also included as signal
to provide further constraints on the strong-phase parameters.
The tag mode can be either a (quasi-)flavor-specific state ff like K∓​π±{{K}^{\mp}}{{\pi}^{\pm}}, a C​PC\!P eigenstate like K+​K−{{K}^{+}}{{K}^{-}}, or another signal decay.
In addition, the single-tag (ST) method, where only one of the DD mesons is reconstructed as the tag decay, is adopted to determine ST yields for normalization as detailed in Sect. 6.

Applying the weighting functions to flavor-tagged decay rates yields 2×(2​M+1)2\times(2M+1) parameters,

𝒩n(K0S/Lh′⁣+h′⁣−|f)∝[Pn++(rDf)2Pn−−2RDfrDfsK(−1)n(CncosδDf−SnsinδDf)],\displaystyle\mathcal{N}_{n}(K^{0}_{\mathrm{S/L}}h^{\prime+}h^{\prime-}|f)\propto[P_{n}^{+}+(r_{D}^{f})^{2}P_{n}^{-}-2R_{D}^{f}r_{D}^{f}s_{K}(-1)^{n}(C_{n}\cos\delta_{D}^{f}-S_{n}\sin\delta_{D}^{f})],

(12)

𝒩¯n(K0S/Lh′⁣+h′⁣−|f)∝[Pn−+(rDf)2Pn+−2RDfrDfsK(−1)n(CncosδDf+SnsinδDf)],\displaystyle\smash{\overline{\mathcal{N}}}_{n}(K^{0}_{\mathrm{S/L}}h^{\prime+}h^{\prime-}|f)\propto[P_{n}^{-}+(r_{D}^{f})^{2}P_{n}^{+}-2R_{D}^{f}r_{D}^{f}s_{K}(-1)^{n}(C_{n}\cos\delta_{D}^{f}+S_{n}\sin\delta_{D}^{f})],

where rDr_{D} and δD\delta_{D} are the amplitude ratio and strong-phase difference, respectively, between the doubly-Cabibbo-suppressed (DCS) and Cabibbo-favored (CF) processes,
and RDR_{D} is the coherence factor of the tag mode.
Depending on the signal decay, sKs_{K} is 11 for D→KS0​h′⁣+​h′⁣−D\!\rightarrow{{K}^{0}_{\mathrm{S}}}h^{\prime+}h^{\prime-} and −1-1 for D→KL0​h′⁣+​h′⁣−D\!\rightarrow{{K}^{0}_{\mathrm{L}}}h^{\prime+}h^{\prime-}, due to the different C​PC\!P content of the neutral kaons.
The parameters 𝒩n\mathcal{N}_{n} (𝒩¯n\smash{\overline{\mathcal{N}}}_{n}) refer to the case where the tag mode is a DCS (CF) decay like D0→K+​π−​(D0→K−​π+)\mbox{${{D}^{0}}\!\rightarrow{{K}^{+}}{{\pi}^{-}}$}(\mbox{${{D}^{0}}\!\rightarrow{{K}^{-}}{{\pi}^{+}}$}).

The 2×(2​M+1)2\times(2M+1) parameters from C​PC\!P tags are

𝒩n​(KS/L0​h′⁣+​h′⁣−|C​P)∝[Pn++Pn−−2​sK​(2​F+−1)​(−1)n​Cn],\displaystyle\mathcal{N}_{n}(K^{0}_{\mathrm{S/L}}h^{\prime+}h^{\prime-}|{C\!P})\propto\left[P_{n}^{+}+P_{n}^{-}-2s_{K}(2F_{+}-1)(-1)^{n}C_{n}\right],

(13)

𝒩¯n​(KS/L0​h′⁣+​h′⁣−|C​P)∝[Pn−+Pn+−2​sK​(2​F+−1)​(−1)n​Cn],\displaystyle\smash{\overline{\mathcal{N}}}_{n}(K^{0}_{\mathrm{S/L}}h^{\prime+}h^{\prime-}|{C\!P})\propto\left[P_{n}^{-}+P_{n}^{+}-2s_{K}(2F_{+}-1)(-1)^{n}C_{n}\right],

where 𝒩n\mathcal{N}_{n} (𝒩¯n\smash{\overline{\mathcal{N}}}_{n}) corresponds to the wnw_{n}(w¯n\overline{w}_{n})-weighted integrals.
The C​PC\!P-even fraction F+F_{+} is 1 for C​PC\!P-even tags, 0 for C​PC\!P-odd tags, and F+=0.9406±0.0032​ (stat)±0.0021​ (syst){F_{+}=0.9406\pm 0.0032\text{\,(stat)}\pm 0.0021\text{\,(syst)}} for the π+​π−​π0{{{\pi}^{+}}{{\pi}^{-}}{{\pi}^{0}}} tag [36].

For the case where both signal and tag DD mesons are reconstructed by the D→KS/L0​h′⁣+​h′⁣−D\!\rightarrow K^{0}_{\mathrm{S/L}}h^{\prime+}h^{\prime-} decays,
four sets of parameters, 𝒩n1​n2±⁣±\mathcal{N}_{n_{1}n_{2}}^{\pm\pm}, can be computed by the four possible weighting-function combinations {wn1,w¯n1}×{wn2,w¯n2}{\{w_{n_{1}},\overline{w}_{n_{1}}\}\times\{w_{n_{2}},\overline{w}_{n_{2}}\}}.
As a result, there are a total of 4×(2​Mh′+1)×(2​Mh′′+1){4\times(2M_{h^{\prime}}+1)\times(2M_{h^{\prime\prime}}+1)} parameters
when the maximum Fourier order of the D→K0S/Lh′(′)+h′(′)−D\!\rightarrow K^{0}_{\mathrm{S/L}}h^{\prime(\prime)+}h^{\prime(\prime)-} decay is set to Mh′(′)M_{h^{\prime(\prime)}}.
Letting s1s_{1} and s2s_{2} denote the signs of the first and second weights with the +(−)+(-) sign itself corresponding to wn​(w¯n)w_{n}(\overline{w}_{n}),
and letting s¯1\overline{s}_{1} and s¯2\overline{s}_{2} denote the opposite signs, the parameters can be expressed as

𝒩n1​n2s1​s2(KS0h′⁣+h′⁣−|KS/L0h′′+h′′−)∝\displaystyle\mathcal{N}_{n_{1}n_{2}}^{s_{1}s_{2}}({{K}^{0}_{\mathrm{S}}}h^{\prime+}h^{\prime-}|K^{0}_{\mathrm{S/L}}h^{\prime\prime+}h^{\prime\prime-})\propto
[Pn1s1Pn2s2+Pn1s¯1Pn2s¯2\displaystyle\left[P_{n_{1}}^{s_{1}}P_{n_{2}}^{{s_{2}}}+P_{n_{1}}^{\overline{s}_{1}}P_{n_{2}}^{\overline{s}_{2}}\right.

−2sK1sK2(−1)n1(−1)n2(Cn1Cn2+s1s2Sn1Sn2)],\displaystyle\left.-2s_{K_{1}}s_{K_{2}}(-1)^{n_{1}}(-1)^{n_{2}}(C_{n_{1}}C_{n_{2}}+s_{1}s_{2}S_{n_{1}}S_{n_{2}})\right],

(14)

where sK1​(2)=1s_{K_{1(2)}}=1 when the first (second) DD meson decays into the KS0​h′⁣+​h′⁣−{{{K}^{0}_{\mathrm{S}}}h^{\prime+}h^{\prime-}} final state,
and sK1​(2)=−1s_{K_{1(2)}}=-1 when the first (second) DD meson decays into the K0Lh′′+h′′−{{{K}^{0}_{\mathrm{L}}}h^{\prime\prime+}h^{\prime\prime-}} final state.

In summary, the optimal Fourier method introduces an extra optimal weight in addition to the Fourier weights proposed in Ref. [22].
Setting the maximum Fourier order to MhM_{h} for D→KS/L0​h′⁣+​h′⁣−D\!\rightarrow K^{0}_{\mathrm{S/L}}h^{\prime+}h^{\prime-} decays, applying the weighting functions to LHCb data yields 2×(2​Mh+1)2\times(2M_{h}+1) parameters for each BB charge, BB decay and DD decay.
The C​PC\!P observables and flavor parameters Pn±P_{n}^{\pm} can be determined from a simultaneous fit to B±→D​K±{{B}^{\pm}}\!\rightarrow D{{K}^{\pm}} and B±→D​π±{{B}^{\pm}}\!\rightarrow D{{\pi}^{\pm}} candidates when the input strong-phase parameters Cn,SnC_{n},S_{n} are known.
From BESIII data, 2×(2​Mh′+1){2\times(2M_{h^{\prime}}+1)} parameters of flavor tags, 2×(2​Mh′+1){2\times(2M_{h^{\prime}}+1)} parameters of C​PC\!P tags,
and 4×(2​Mh′+1)×(2​Mh′′+1){4\times(2M_{h^{\prime}}+1)\times(2M_{h^{\prime\prime}}+1)} parameters of KS0​h′⁣+​h′⁣−{{{K}^{0}_{\mathrm{S}}}h^{\prime+}h^{\prime-}} tags are computed.
The flavor-tag parameters are used to determine Pn±P_{n}^{\pm} in BESIII data, and the strong-phase parameters Cn,SnC_{n},S_{n}, are obtained from fits to C​PC\!P-tag and KS0​h′⁣+​h′⁣−{{{K}^{0}_{\mathrm{S}}}h^{\prime+}h^{\prime-}}-tag parameters.
The CnC_{n} and SnS_{n} parameters need to be shared between BESIII and LHCb data, while separate Pn±P_{n}^{\pm} parameters are obtained independently from flavor tags in BESIII and B±→D​π±{{B}^{\pm}}\!\rightarrow D{{\pi}^{\pm}} decays in LHCb.
Therefore, a joint fit to the BESIII and LHCb parameters allows for a simultaneous determination of the C​PC\!P observables and strong-phase parameters.

Pseudoexperiments are conducted to validate the approach, where it is found that the smallest statistical uncertainty on γ\gamma is achieved with Mπ=2M_{\pi}=2 and MK=1M_{K}=1 when the dataset size is comparable to that analyzed in this paper.
The increasing sensitivity with MhM_{h} is limited by sizes of both BESIII and LHCb datasets.
The average uncertainty of γ\gamma from the optimal Fourier approach is also lower than that from the binned approach when both approaches are applied to the same ensemble.
The study is performed under ideal conditions where the generated amplitude models are the same as those used to compute the weighting functions.
As the underlying model of the data is not perfectly known, imperfect description of the models may not lead to the most optimal results.
Nevertheless, the choice of model does not bias the central values, and the orders Mπ=2M_{\pi}=2 and MK=1M_{K}=1 are chosen to determine the C​PC\!P observables x±D​Kx_{\pm}^{DK}, y±D​Ky_{\pm}^{DK}, xξD​πx_{\xi}^{D\pi} and yξD​πy_{\xi}^{D\pi}.

3 Detectors and simulation

The BESIII detector [37] records symmetric e+​e−e^{+}e^{-} collisions
provided by the BEPCII storage ring [38]
in the center-of-mass energy range from 1.84 to 4.95 GeV with a peak luminosity of 1.1×1033​cm−2​s−11.1\times 10^{33}\;\text{cm}^{-2}\text{s}^{-1}
achieved at s=3.773​ GeV\sqrt{s}=3.773\text{\,Ge\kern-1.00006ptV}.
Large data samples in this energy region have been collected by the BESIII experiment [39, 40, 41].
The cylindrical core of the BESIII detector covers 93%93\% of the full solid angle and consists of a helium-based
multilayer drift chamber (MDC), a plastic scintillator time-of-flight
system (TOF), and a CsI(Tl) electromagnetic calorimeter (EMC),
which are all enclosed in a superconducting solenoidal magnet
providing a 1.0 T magnetic field, which was 0.9 T in 2012.
The solenoid is surrounded by an octagonal flux-return yoke made of steel, interleaved with resistive-plate-counter muon-identification modules.

The charged-particle momentum resolution at 1​ GeV​/​c1\text{\,Ge\kern-1.00006ptV\!/}c is 0.5%0.5\%, and the d​E/d​x{\rm d}E/{\rm d}x resolution is 6%6\% for electrons from Bhabha scattering. The EMC measures photon energies with a resolution of 2.5%2.5\% (5%5\%) at 11 GeV in the barrel (end-cap) region. The time resolution in the TOF barrel region is 68 ps, while that in the end-cap region is 110 ps. The end-cap TOF system was upgraded in 2015 using multigap resistive-plate chamber technology, providing a time resolution of 60 ps [42]. This upgrade benefits 63% of the data used in this analysis. More details can be found in Ref. [37].

The simulation samples produced with a Geant4-based software package [43], which includes the geometric description of the BESIII detector and the detector response, are used to determine detection efficiencies and estimate backgrounds. The simulation models the beam energy spread and initial-state radiation (ISR) in e+​e−e^{+}e^{-} annihilations with the generator kkmc [44, 45].

Inclusive simulation samples are produced including D​D¯D{{\kern 1.79993pt\overline{\kern-1.79993ptD}}} pairs corrected with quantum-correlation effects, non-D​D¯D{{\kern 1.79993pt\overline{\kern-1.79993ptD}}} decays of the ψ⁡(3770)\psi(3770), the ISR production of the J/ψJ/\psi and ψ⁡(3686)\psi(3686) states, and continuum e+​e−→u​u¯,d​d¯,s​s¯e^{+}e^{-}\!\rightarrow u{\overline{{u}}},~d{\overline{{d}}},~s{\overline{{s}}} processes incorporated in kkmc. All particle decays in the inclusive simulation samples are modeled with EvtGen [46, 47] using branching fractions either taken from the Particle Data Group [48], when available, or estimated with Lundcharm [49, 50]. Final-state radiation from charged final-state particles is incorporated using the Photos package [51].
Simulated signal samples are produced for DT KS/L0​h′⁣+​h′⁣−K^{0}_{\mathrm{S/L}}h^{\prime+}h^{\prime-} vs. tag modes, where quantum-correlation effects are implemented. The D→KS0​π+​π−D\!\rightarrow{{K}^{0}_{\mathrm{S}}}\pi^{+}\pi^{-} decay is simulated with the amplitude model measured by the Belle experiment [31, 32] and the D→KL0​π+​π−D\!\rightarrow{{K}^{0}_{\mathrm{L}}}\pi^{+}\pi^{-} decay is simulated with the KS0​π+​π−{{K}^{0}_{\mathrm{S}}}\pi^{+}\pi^{-} model implemented with U-spin breaking parameters [52]. The D→KS/L0​K+​K−D\!\rightarrow K^{0}_{\mathrm{S/L}}{{K}^{+}}{{K}^{-}} decay is simulated with the amplitude model developed by the BESIII experiment. Multibody tag-side decays are simulated with the aforementioned EvtGen according to the most recent models from experimental studies.

The LHCb detector [53, 54] is a single-arm forward
spectrometer covering the pseudorapidity range 2<η<52<\eta<5,
designed for the study of particles containing bb or cc quarks. The detector used to collect the data analyzed in this paper includes a high-precision tracking system
consisting of a silicon-strip vertex detector surrounding the p​ppp
interaction region, a large-area silicon-strip detector located
upstream of a dipole magnet with a bending power of about
4​T​m4{\mathrm{\,T\,m}}, and three stations of silicon-strip detectors and straw
drift tubes are placed downstream of the magnet.
The tracking system provides a measurement of the momentum of charged particles with
a relative uncertainty that varies from 0.5% at low momentum to 1.0% at 200 GeV​/​c\text{\,Ge\kern-1.00006ptV\!/}c.
The minimum distance from a track to a primary p​ppp collision vertex, the impact parameter,
is measured with a resolution of (15+29/pT)​μ​m(15+29/p_{\mathrm{T}})\,\upmu\text{m},
where pTp_{\mathrm{T}} is the component of the momentum transverse to the beam, in  GeV​/​c\text{\,Ge\kern-1.00006ptV\!/}c.
Different types of charged hadrons are distinguished using information
from two ring-imaging Cherenkov detectors.
Photons, electrons and hadrons are identified by a calorimeter system consisting of
scintillating-pad and preshower detectors, an electromagnetic
and a hadronic calorimeter. Muons are identified by a
system composed of alternating layers of iron and multiwire
proportional chambers.
The online event selection is performed by a trigger,
which consists of a hardware stage, based on information from the calorimeter and muon
systems, followed by a software stage, which applies a full event
reconstruction.
Triggered data further undergo a centralized, offline processing step
to deliver physics-analysis-ready data across the entire LHCb physics program [55].

Simulation is required to correct for reconstruction and selection efficiencies in LHCb data.
In the simulation, p​ppp collisions are generated using
Pythia [56, *Sjostrand:2006za]
with a specific LHCb configuration [58].
Decays of unstable particles
are described by EvtGen [46], in which final-state
radiation is generated using Photos [59].
The interaction of the generated particles with the detector and its response,
are implemented using the Geant4 toolkit [60, 43] as described in
Ref. [61].
When generating simulation samples for background processes, the underlying p​ppp interaction is reused multiple times, with an independently generated background decay for each [62].
Some subdominant background processes are generated with RapidSim [63] that mimics the LHCb detector acceptance, reconstruction efficiency and dynamics of the decay.

4 Event selection and signal extraction

Quantum-correlated D​D¯D{{\kern 1.79993pt\overline{\kern-1.79993ptD}}} decays collected by the BESIII detector are reconstructed by requiring one of the DD mesons to decay to a signal channel, KS/L0​h′⁣+​h′⁣−{K^{0}_{\mathrm{S/L}}h^{\prime+}h^{\prime-}}, and the other to decay via a tag mode.
As described in Sect. 2, three categories of tag modes are used in the analysis.
The flavor and C​PC\!P-tag modes are listed in Table 1.
The decays where both DD mesons are reconstructed in a signal channel are selected according to the following self-conjugate DT modes: KS0​π+​π−{{{K}^{0}_{\mathrm{S}}}{{\pi}^{+}}{{\pi}^{-}}} vs. {KS0π+π−,KS0π±[π∓],KS0(→π0[π0])π+π−,[KL0]π+π−,KS0K±[K∓],[KL0]K+K−}\{{{K}^{0}_{\mathrm{S}}}{{\pi}^{+}}{{\pi}^{-}},~{{K}^{0}_{\mathrm{S}}}{{\pi}^{\pm}}[{{\pi}^{\mp}}],~{{K}^{0}_{\mathrm{S}}}(\rightarrow{{\pi}^{0}}[{{\pi}^{0}}]){{\pi}^{+}}{{\pi}^{-}},~[{{K}^{0}_{\mathrm{L}}}]{{\pi}^{+}}{{\pi}^{-}},~{{K}^{0}_{\mathrm{S}}}{{K}^{\pm}}[{{K}^{\mp}}],~[{{K}^{0}_{\mathrm{L}}}]{{K}^{+}}{{K}^{-}}\} and KS0​K+​K−{{{K}^{0}_{\mathrm{S}}}{{K}^{+}}{{K}^{-}}} vs. {KS0​K±​[K∓],[KL0]​K+​K−,[KL0]​π+​π−}\{{{K}^{0}_{\mathrm{S}}}{{K}^{\pm}}[{{K}^{\mp}}],~[{{K}^{0}_{\mathrm{L}}}]{{K}^{+}}{{K}^{-}},~[{{K}^{0}_{\mathrm{L}}}]{{\pi}^{+}}{{\pi}^{-}}\},
where particles in square brackets are not reconstructed.

Table 1: Summary of tag modes selected against D→KS/L0​h′⁣+​h′⁣−D\!\rightarrow K^{0}_{\mathrm{S/L}}h^{\prime+}h^{\prime-}.

Tag
KS0​h′⁣+​h′⁣−{{K}^{0}_{\mathrm{S}}}h^{\prime+}h^{\prime-}
KL0​h′⁣+​h′⁣−{{K}^{0}_{\mathrm{L}}}h^{\prime+}h^{\prime-}

Flavor
K​π​π0K\pi\pi^{0}, K​πK\pi, K​π​π​πK\pi\pi\pi, K​e​νeKe\nu_{e}
K​π​π0K\pi\pi^{0}, K​πK\pi, K​π​π​πK\pi\pi\pi

C​PC\!P-even
K​KKK, π​π\pi\pi, KS0​π0​π0{{K}^{0}_{\mathrm{S}}}\pi^{0}\pi^{0}, KL0​π0{{K}^{0}_{\mathrm{L}}}\pi^{0}, π​π​π0\pi\pi\pi^{0}
K​KKK, π​π\pi\pi, KS0​π0​π0{{K}^{0}_{\mathrm{S}}}\pi^{0}\pi^{0}, π​π​π0\pi\pi\pi^{0}

C​PC\!P-odd
KS0​π0{{K}^{0}_{\mathrm{S}}}\pi^{0}, KS0η(→γγ){{K}^{0}_{\mathrm{S}}}\eta(\rightarrow\gamma\gamma), KS0η(→πππ0){{K}^{0}_{\mathrm{S}}}\eta(\rightarrow\pi\pi{{\pi}^{0}}),
KS0​π0{{K}^{0}_{\mathrm{S}}}\pi^{0}, KS0η(→γγ){{K}^{0}_{\mathrm{S}}}\eta(\rightarrow\gamma\gamma), KS0η(→πππ0){{K}^{0}_{\mathrm{S}}}\eta(\rightarrow\pi\pi{{\pi}^{0}}),

KS0η′(→ππγ){{K}^{0}_{\mathrm{S}}}\eta^{\prime}(\rightarrow\pi\pi\gamma), KS0η′(→ππη){{K}^{0}_{\mathrm{S}}}\eta^{\prime}(\rightarrow\pi\pi\eta),
KS0η′(→ππη){{K}^{0}_{\mathrm{S}}}\eta^{\prime}(\rightarrow\pi\pi\eta), KS0η′(→ππγ){{K}^{0}_{\mathrm{S}}}\eta^{\prime}(\rightarrow\pi\pi\gamma),

KS0ω(→πππ0){{K}^{0}_{\mathrm{S}}}\omega(\rightarrow\pi\pi{{\pi}^{0}}), KL0​π0​π0{{K}^{0}_{\mathrm{L}}}\pi^{0}\pi^{0}
KS0ω(→πππ0){{K}^{0}_{\mathrm{S}}}\omega(\rightarrow\pi\pi{{\pi}^{0}})

Final-state particles of charm-meson decays are reconstructed from candidates of charged tracks and neutral photons using selection criteria identical to those applied in the binned analysis of D→KS0​π+​π−D\!\rightarrow{{K}^{0}_{\mathrm{S}}}\pi^{+}\pi^{-} [26].
In addition, the requirement of no extra π0\pi^{0} candidates is removed for the KL0​K+​K−{{K}^{0}_{\mathrm{L}}}{{K}^{+}}{{K}^{-}} vs. KS0​K+​K−{{K}^{0}_{\mathrm{S}}}{{K}^{+}}{{K}^{-}} and KS0​K+​K−{{K}^{0}_{\mathrm{S}}}{{K}^{+}}{{K}^{-}} vs. KL0​π+​π−{{K}^{0}_{\mathrm{L}}}\pi^{+}\pi^{-} final states, as higher statistics are obtained without a decrease in signal purity.
Meanwhile, to mitigate resolution effects in selected D​D¯D{{\kern 1.79993pt\overline{\kern-1.79993ptD}}} decay samples,
DD, KS0{{K}^{0}_{\mathrm{S}}} and KL0{{K}^{0}_{\mathrm{L}}} candidates are kinematically constrained to their known masses [48].

The charm-meson decays where all final-state particles are reconstructed are classified as fully reconstructed modes.
Following Ref. [26], a mode-specific requirement on the energy difference Δ​E≡ED−s/2\Delta E\equiv E_{D}-\sqrt{s}/2 is applied,
and the beam-constrained mass MBC≡(s/2)2−|𝒑D|2​c2{M_{\rm BC}\equiv\sqrt{({\sqrt{s}/2})^{2}-|\boldsymbol{p}_{D}|^{2}c^{2}}} is utilized as the fit variable.
Here, s/2\sqrt{s}/2 is the beam energy, EDE_{D} is the sum of final-state-particle energies and 𝒑D\boldsymbol{p}_{D} represents the reconstructed momentum of the DD candidate both evaluated in the center-of-mass system.
When there is a final-state particle such as a KL0{{K}^{0}_{\mathrm{L}}} meson or neutrino that cannot be reconstructed within the BESIII detector, a partial reconstruction method is adopted.
The squared missing mass is calculated by Mmiss2=Emiss2−|𝒑miss|2​c2M^{2}_{{\rm miss}}=E^{2}_{{\rm miss}}-|\boldsymbol{p}_{{\rm miss}}|^{2}c^{2}, where Emiss=s/2−EotherE_{{\rm miss}}=\sqrt{s}/2-E_{{\rm other}} and 𝒑miss=−𝒑tag−𝒑other\boldsymbol{p}_{{\rm miss}}=-\boldsymbol{p}_{\rm tag}-\boldsymbol{p}_{{\rm other}}.
Here, 𝒑tag\boldsymbol{p}_{\rm tag} represents the momentum of the fully reconstructed DD candidate, while 𝒑other\boldsymbol{p}_{{\rm other}} and EotherE_{{\rm other}} are the total momentum and energy of the other reconstructed particles in the partially reconstructed modes, all evaluated in the center-of-mass system.
In the D→K​e​νeD\!\rightarrow Ke{{\nu}_{e}} mode, Umiss=Emiss−|𝒑miss|​cU_{\rm miss}=E_{\rm miss}-|\boldsymbol{p}_{\rm miss}|c is employed as the fit variable as signal events form a peak around Umiss=0​ GeVU_{\rm miss}=0\text{\,Ge\kern-1.00006ptV}.

Possible sources of background in DD decays are estimated through inclusive simulation samples.
Details on background studies, extraction of signal yields and efficiency determinations of the D→KS/L0​π+​π−D\!\rightarrow K^{0}_{\mathrm{S/L}}\pi^{+}\pi^{-} signal channel can be found in Ref. [26].
For the channel D→KS/L0​K+​K−D\!\rightarrow K^{0}_{\mathrm{S/L}}{{K}^{+}}{{K}^{-}}, the dominant peaking backgrounds are D→π+​π−​K+​K−D\!\rightarrow\pi^{+}\pi^{-}{{K}^{+}}{{K}^{-}} and D→KS0(→π0π0)K+K−D\!\rightarrow{{K}^{0}_{\mathrm{S}}}(\rightarrow{{\pi}^{0}}{{\pi}^{0}}){{K}^{+}}{{K}^{-}} with a contamination rate of 0.5%0.5\% and 2.5%2.5\%, respectively.
The nonpeaking backgrounds are mainly from other D​D¯D{{\kern 1.79993pt\overline{\kern-1.79993ptD}}} decays and continuum processes. One-dimensional unbinned maximum-likelihood fits are performed to discriminate signal from background.
In the fits, double-sided Crystal Ball functions [64] are used as signal shapes, with all of their parameters fixed from simulated signal samples. The signal models are then convolved with Gaussian functions whose mean values and widths vary freely in order to take into account the different resolution effects between data and simulation.
The background shapes and contributions are directly derived from inclusive simulation samples except for those from the KS0→π0​π0{{K}^{0}_{\mathrm{S}}}\!\rightarrow\pi^{0}\pi^{0} decay in D→KL0​K+​K−D\!\rightarrow{{K}^{0}_{\mathrm{L}}}{{K}^{+}}{{K}^{-}} vs. C​PC\!P-tag modes. For those background components, the shapes are extracted from simulated signal samples and the background yields are estimated from data based on the signal yields of KS0​K+​K−{{K}^{0}_{\mathrm{S}}}{{K}^{+}}{{K}^{-}} vs. C​PC\!P-eigenstate modes.
Example fit results for the D→KS/L0​K+​K−D\!\rightarrow K^{0}_{\mathrm{S/L}}{{K}^{+}}{{K}^{-}} channel are shown in Fig. 3. Details on fits separated by individual tag modes along with their corresponding signal yields are given in Appendix A.

Figure 3: Example fit results for the D→KS/L0​K+​K−D\!\rightarrow K^{0}_{\mathrm{S/L}}{{K}^{+}}{{K}^{-}} channel at BESIII, where (top left) corresponds to the D→KS0​K​KD\!\rightarrow{{K}^{0}_{\mathrm{S}}}KK vs. K​KKK tag; (top right) the D→KS0​K​KD\!\rightarrow{{K}^{0}_{\mathrm{S}}}KK vs. K​e​[νe]Ke[\nu_{e}] tag; (bottom left) the D→KS0​[K]​KD\!\rightarrow{{K}^{0}_{\mathrm{S}}}[K]K vs. D→KS0​π​πD\!\rightarrow{{K}^{0}_{\mathrm{S}}}\pi\pi tag; and (bottom right) the D→[KL0]​K​KD\!\rightarrow[{{K}^{0}_{\mathrm{L}}}]KK vs. K​KKK tag.

In the datasets collected by the LHCb detector, decays of KS0→π+​π−{{K}^{0}_{\mathrm{S}}}\!\rightarrow{{\pi}^{+}}{{\pi}^{-}}  are reconstructed in two different categories.
The first involves KS0{K}^{0}_{\mathrm{S}} mesons that decay early enough for the
pions to be reconstructed in the vertex detector, and the
second involves KS0{K}^{0}_{\mathrm{S}} mesons that decay later such that track segments of the
pions cannot be formed in the vertex detector. These categories are
referred to as long and downstream, respectively. The
long category has better mass, momentum and vertex resolution than the
downstream category, while the downstream category has higher reconstruction efficiency and contributes approximately two thirds of the dataset.
The DD-meson candidates are then formed by combining a KS0{{K}^{0}_{\mathrm{S}}} candidate with two tracks both with either the pion or kaon mass hypothesis.
The BB-meson candidate is subsequently built by combining a DD candidate and another pion or kaon track.
This constitutes eight categories in the B±→D(→KS0h′⁣+h′⁣−)h±{{B}^{\pm}}\!\rightarrow D(\rightarrow{{K}^{0}_{\mathrm{S}}}h^{\prime+}h^{\prime-})h^{\pm} data.

The selection criteria and fit model to the D​h±Dh^{\pm} mass spectra are identical to those described in Ref. [17].
Signal and background yields in data are extracted by a two-stage fit.
The first stage is the same as the global fit in Ref. [17], in which the parametrizations of signal and background components are determined.
The global fit includes: the signal B±→D​h±{{B}^{\pm}}\!\rightarrow Dh^{\pm} decays;
partially reconstructed B±→D∗(→Dγ/π0)h±{{B}^{\pm}}\!\rightarrow{{D}^{*}}(\rightarrow D\gamma/{{\pi}^{0}})h^{\pm}, B0→D∗⁣∓(→Dπ∓)h±{{B}^{0}}\!\rightarrow{{D}^{*\mp}}(\rightarrow D{{\pi}^{\mp}})h^{\pm}, B0→D​h±​π∓{{B}^{0}}\!\rightarrow Dh^{\pm}{{\pi}^{\mp}} and Bs0→D​K−​π+{{B}^{0}_{s}}\!\rightarrow D{{K}^{-}}{{\pi}^{+}} decays, where a pion or photon is not reconstructed; the aforementioned background decays in B±→D​π±{{B}^{\pm}}\!\rightarrow D{{\pi}^{\pm}} with the companion pion misidentified as a kaon; and a combinatorial component in each category.
All shape parameters are subsequently fixed in the second stage, in which another fit is performed in a tightened mass range, from 5150 to 5800 MeV​/​c2\text{\,Me\kern-1.00006ptV\!/}c^{2}, to reject most of the partially reconstructed background events.
The mass distributions with fit results also included are shown in Fig. 4.

Figure 4: Mass distributions and fit results of the (left) B±→D​K±{{B}^{\pm}}\!\rightarrow D{{K}^{\pm}} and (right) B±→D​π±{{B}^{\pm}}\!\rightarrow D{{\pi}^{\pm}} data, with the DD meson decaying to either (top four) KS0​π+​π−{{{K}^{0}_{\mathrm{S}}}{{\pi}^{+}}{{\pi}^{-}}} or (bottom four) KS0​K+​K−{{{K}^{0}_{\mathrm{S}}}{{K}^{+}}{{K}^{-}}} final states. The first and third rows correspond to the long category, while the second and fourth rows correspond to the downstream category.

5 Observables from data

The observables are obtained from weighted sums over data events,
in which background contributions are subtracted and efficiency effects are corrected.
The signal contribution in LHCb data is isolated by the sPlot technique [65]
from which per-event weights, denoted as fif_{i}, are obtained based on the mass fit results.
However, sPlot is not applicable to BESIII data due to the known correlation between the fit variables and Dalitz plot coordinates.
The DD-decay observables are computed from both BESIII data and simulated background samples,
and the background observables are subtracted from the data observables.
Efficiency corrections are applied to both datasets by applying inverse efficiency weights derived from relative efficiency distributions.

Relative efficiency distributions are described as functions of the Dalitz-plot coordinates and obtained from simulated signal samples.
The D→KS0​h′⁣+​h′⁣−D\!\rightarrow{{K}^{0}_{\mathrm{S}}}h^{\prime+}h^{\prime-} decays in both BESIII and LHCb simulation samples are generated with a uniform phase-space distribution; thus, the Dalitz-plot distributions post selection are proportional to the efficiencies across the Dalitz plot,
which include the effects of detector acceptance, reconstruction and selection.
Efficiency distributions are obtained by fitting each of the Dalitz-plot distributions of simulated signal samples with a two-dimensional polynomial function, with terms up to fourth order.
In addition, extra per-event efficiency weights are introduced to correct the difference between data and simulation samples for DD decays in BESIII. The main discrepancies exist in the tracking efficiency, the particle-identification (PID) efficiency of K​πK\pi separation, and the reconstruction efficiency of KS0{{K}^{0}_{\mathrm{S}}} candidates. They are investigated using a control sample of D​D¯D{{\kern 1.79993pt\overline{\kern-1.79993ptD}}} hadronic events [26]. Subsequently, corrections are applied to the signal decay modes: KS0​π+​π−{{K}^{0}_{\mathrm{S}}}\pi^{+}\pi^{-}, KS0​π±​[π∓]{{K}^{0}_{\mathrm{S}}}{{\pi}^{\pm}}[{{\pi}^{\mp}}], KS0(→π0[π0])π+π−{{K}^{0}_{\mathrm{S}}}(\rightarrow{{\pi}^{0}}[{{\pi}^{0}}]){{\pi}^{+}}{{\pi}^{-}}, [KL0]​π+​π−[{{K}^{0}_{\mathrm{L}}}]\pi^{+}\pi^{-}, KS0​K+​K−{{K}^{0}_{\mathrm{S}}}{{K}^{+}}{{K}^{-}}, KS0​K±​[K∓]{{K}^{0}_{\mathrm{S}}}{{K}^{\pm}}[{{K}^{\mp}}], and [KL0]​K+​K−[{{K}^{0}_{\mathrm{L}}}]{{K}^{+}}{{K}^{-}}. For tag modes, these differences largely cancel out in the ratio of ST to DT yields for fully reconstructed tag modes, and are found to be negligible for the partially reconstructed tag modes, such as K​e​νeKe\nu_{e}, KL0​π0{{K}^{0}_{\mathrm{L}}}\pi^{0}, and KL0​π0​π0{{K}^{0}_{\mathrm{L}}}\pi^{0}\pi^{0}.
Figure 5 shows example efficiency distributions from BESIII simulation.
Figure 6 shows the efficiency distributions from B±→D​π±{{B}^{\pm}}\!\rightarrow D{{\pi}^{\pm}} simulation at LHCb, which are also used in the B±→D​K±{{B}^{\pm}}\!\rightarrow D{{K}^{\pm}} decays,
as the Dalitz-plot distributions of simulated samples are found to be consistent between B±→D​π±{{B}^{\pm}}\!\rightarrow D{{\pi}^{\pm}} and B±→D​K±{{B}^{\pm}}\!\rightarrow D{{K}^{\pm}} decays.

Figure 5: Example efficiency distributions from BESIII simulation. For the K​πK\pi tag mode, the plots correspond to the (top left) D→KS0​π+​π−D\!\rightarrow{{K}^{0}_{\mathrm{S}}}{{\pi}^{+}}{{\pi}^{-}}, (top right) D→KL0​π+​π−D\!\rightarrow{{K}^{0}_{\mathrm{L}}}{{\pi}^{+}}{{\pi}^{-}}, (bottom left) D→KS0​K+​K−D\!\rightarrow{{K}^{0}_{\mathrm{S}}}{{K}^{+}}{{K}^{-}} and (bottom right) D→KL0​K+​K−D\!\rightarrow{{K}^{0}_{\mathrm{L}}}{{K}^{+}}{{K}^{-}} signal modes.
The average efficiencies over the Dalitz plot are scaled to unity.

Figure 6: Efficiency distributions of B±→D​π±{{B}^{\pm}}\!\rightarrow D{{\pi}^{\pm}} decays with (top) D→KS0​π+​π−D\!\rightarrow{{K}^{0}_{\mathrm{S}}}{{\pi}^{+}}{{\pi}^{-}} and (bottom) D→KS0​K+​K−D\!\rightarrow{{K}^{0}_{\mathrm{S}}}{{K}^{+}}{{K}^{-}} decays for the (left) long and (right) downstream categories of the KS0{{K}^{0}_{\mathrm{S}}} candidate.
The average efficiencies over the Dalitz plot are scaled to unity.

The weighted signal yields from flavor tags are computed by

𝒩n=∑i∈D0wn,i/ϵi,𝒩¯n=∑i∈D¯0wn,i/ϵi,\mathcal{N}_{n}=\sum_{i\in{{D}^{0}}}w_{n,i}/\epsilon_{i},\hskip 10.00002pt\smash{\overline{\mathcal{N}}}_{\!n}=\sum_{i\in{{\kern 1.47495pt\overline{\kern-1.47495ptD}}{}^{0}}}w_{n,i}/\epsilon_{i}\,,

(15)

where the weights are obtained from the functions wnw_{n} defined in Eq. 5, efficiency distributions ϵ\epsilon and the Dalitz-plot coordinates of candidate ii.
The computation of 𝒩n\mathcal{N}_{n} (𝒩¯n\smash{\overline{\mathcal{N}}}_{\!n}) is performed by summing over D0{{D}^{0}} (D¯0{{\kern 1.79993pt\overline{\kern-1.79993ptD}}{}^{0}})-dominated KS0​h+​h−{{K}^{0}_{\mathrm{S}}}h^{+}h^{-} data, where the tag side decays into DCS (CF) D0{{D}^{0}} final states, e.g. K+​π−{{K}^{+}}{{\pi}^{-}} (K−​π+{{K}^{-}}{{\pi}^{+}}).
These observables from the flavor tags are used to compute the flavored parameters, Pn±P_{n}^{\pm}.
The DCS contributions in these channels are corrected according to Eq. 12,
where the strong-phase parameters Cn,SnC_{n},S_{n} are fixed to model predictions and the hadronic parameters rD,δDr_{D},\delta_{D} and RDR_{D} are fixed to values that are listed in Table 2.
The corrected parameters Pn±P_{n}^{\pm} from different tags, as shown in Figs. 7 and 8 for D→KS0​π+​π−D\!\rightarrow{{K}^{0}_{\mathrm{S}}}{{\pi}^{+}}{{\pi}^{-}} and D→KL0​π+​π−D\!\rightarrow{{K}^{0}_{\mathrm{L}}}{{\pi}^{+}}{{\pi}^{-}} decays, respectively,
are found to be consistent.

Table 2: Parameters of the flavor tags that are used to correct flavor observables, taken from Ref. [66].

Tag
rD(×10−2)r_{D}~(\times 10^{-2})
δD()∘\;\delta_{D}~({{}^{\circ}})
RDR_{D}

K​πK\pi
5.87±0.025.87\pm 0.02
190.2190.2 
+−{}^{+}_{-}  
4.83.6{}^{3.6}_{4.8}

11  

K​π​π0K\pi\pi^{0}
4.41±0.114.41\pm 0.11
196196 
±\pm  
11
0.790.79 
±\pm  
0.040.04

K​π​π​πK\pi\pi\pi
5.50±0.075.50\pm 0.07
161161 
+−{}^{+}_{-}  
1828{}^{28}_{18}
0.440.44 
+−{}^{+}_{-}  
0.090.10{}^{0.10}_{0.09}

Figure 7: Flavored parameters (left) Pn+P_{n}^{+} and (right) Pn−P_{n}^{-} of the D0→KS0​π+​π−{{D}^{0}}\!\rightarrow{{K}^{0}_{\mathrm{S}}}{{\pi}^{+}}{{\pi}^{-}} decay from different tags. The total observables are obtained from combining those of all tags. The observables are normalized to P0++P0−=1{P_{0}^{+}+P_{0}^{-}=1}.

Figure 8: Flavored parameters (left) Pn+P_{n}^{+} and (right) Pn−P_{n}^{-} of the D0→KL0​π+​π−{{D}^{0}}\!\rightarrow{{K}^{0}_{\mathrm{L}}}{{\pi}^{+}}{{\pi}^{-}} decay from different tags. The total observables are obtained from combining those of all tags. The observables are normalized to P0++P0−=1{P_{0}^{+}+P_{0}^{-}=1}.

Similarly, weighted yields from C​PC\!P tags are computed as

𝒩n=∑iwn,i/ϵi,𝒩¯n=∑iw¯n,i/ϵi,\mathcal{N}_{n}=\sum_{i}w_{n,i}/\epsilon_{i},\hskip 10.00002pt\smash{\overline{\mathcal{N}}}_{\!n}=\sum_{i}\overline{w}_{n,i}/\epsilon_{i}\,,

(16)

and four combinations of weighted yields of self-conjugated tags are computed by combining per-event weights from the phase space of both sides

𝒩n1​n2s1​s2=∑i({wn1,i1,w¯n1,i1}/ϵi1)⋅({wn2,i2,w¯n2,i2}/ϵi2),\mathcal{N}_{n_{1}n_{2}}^{s_{1}s_{2}}=\sum_{i}\left.\left(\left\{w_{n_{1},i_{1}},\overline{w}_{n_{1},i_{1}}\right\}/\epsilon_{i_{1}}\right)\cdot\left(\left\{w_{n_{2},i_{2}},\overline{w}_{n_{2},i_{2}}\right\}/\epsilon_{i_{2}}\right)\right.,

(17)

where the first (second) set of weights is based on the Dalitz-plot coordinates of the first (second) DD meson.

By summing over the same data samples, the observables are naturally correlated with each other.
The covariance matrices between two weighted sums, ∑iw1i\sum_{i}w_{1}^{i} and ∑iw2i\sum_{i}w_{2}^{i}, are estimated by Poisson bootstrapping with V=∑iw1i​w2iV=\sum_{i}w_{1}^{i}w_{2}^{i}.
Therefore, the covariance between observables of Eq. 16, e.g. 𝒩n1\mathcal{N}_{n_{1}} and 𝒩n2\mathcal{N}_{n_{2}}, are

cov⁡(𝒩n1,𝒩n2)=∑iwn1,i​wn2,i/(ϵi)2,{\rm cov}(\mathcal{N}_{n_{1}},\mathcal{N}_{n_{2}})=\sum_{i}w_{n_{1},i}w_{n_{2},i}/\left(\epsilon_{i}\right)^{2},

(18)

and the same computation is applied to other combinations.
For the KS/L0​π+​π−K^{0}_{\mathrm{S/L}}\pi^{+}\pi^{-} signal samples, covariances between C​PC\!P-tag observables are computed by Eq. 18.
Meanwhile, the correlations between self-conjugated tag observables are derived from pseudoexperiments with a much larger sample size, which are generated with the amplitude model as the correlations from data are found to bias the results under current statistics.
The covariance estimates in both C​PC\!P and self-conjugate tag observables with KS/L0​K+​K−K^{0}_{\mathrm{S/L}}{{K}^{+}}{{K}^{-}} are also computed from an ensemble due to low statistics.
The uncertainties of those observables are calculated from data as a special case of Eq. 18, in which the two weights are identical, reducing the expression to the sum of squared weights.

The weighted yields from LHCb data are computed by

 
( ) 
[-.7ex] 𝒩 
n±=∑i∈B±→D​h±
 
( ) 
[-.7ex] w 
n,i​fi/ϵi,\smash{\shortstack{{(\rule[2.15277pt]{12.50002pt}{0.51212pt})}
\\
 [-.7ex] $\mathcal{N}$}}_{\!n}^{\pm}=\sum_{i\in{{B}^{\pm}}\rightarrow Dh^{\pm}}\shortstack{{(\rule[2.15277pt]{12.50002pt}{0.51212pt})}
\\
 [-.7ex] $w$}_{\hskip-1.63885ptn,i}f_{i}/\epsilon_{i},

(19)

which includes the extra signal weight, fif_{i}.
The covariance between LHCb observables are analogously computed by Eq. 18, e.g. between 𝒩n1−\mathcal{N}_{n_{1}}^{-} and 𝒩¯n2−\smash{\overline{\mathcal{N}}}_{\!n_{2}}^{-}, as

cov⁡(𝒩n1−,𝒩¯n2−)=∑i∈B−→D​h−wn1,i​w¯n2,i​(fi)2/(ϵi)2.{\rm cov}(\mathcal{N}_{n_{1}}^{-},\smash{\overline{\mathcal{N}}}_{\!n_{2}}^{-})=\sum_{i\in{{{B}^{-}}}\rightarrow Dh^{-}}w_{n_{1},i}\overline{w}_{n_{2},i}\left(f_{i}\right)^{2}/\left(\epsilon_{i}\right)^{2}.

(20)

Figure 9 shows the weighted yields obtained from the B±→D​h±{{B}^{\pm}}\!\rightarrow Dh^{\pm}, D→KS0​π+​π−D\!\rightarrow{{K}^{0}_{\mathrm{S}}}{{\pi}^{+}}{{\pi}^{-}} data in the downstream category. The model predictions are computed from the amplitude model measured in Refs. [31, 32, 33] and physics parameters measured in Ref. [17].
Sizable C​PC\!P-violation effects are seen in the observables from the B±→D​K±{{B}^{\pm}}\!\rightarrow D{{K}^{\pm}} data,
while much smaller C​PC\!P asymmetries are present in the B±→D​π±{{B}^{\pm}}\!\rightarrow D{{\pi}^{\pm}} channel due to the small value of rBD​πr_{B}^{D\pi}.

Figure 9: Observables from the (top) B±→D​K±{{B}^{\pm}}\!\rightarrow D{{K}^{\pm}} and (bottom) B±→D​π±{{B}^{\pm}}\!\rightarrow D{{\pi}^{\pm}} data sample with D→KS0​π+​π−D\!\rightarrow{{K}^{0}_{\mathrm{S}}}{{\pi}^{+}}{{\pi}^{-}} decays in the downstream category.
The C​PC\!P-conjugated observable pairs, (left) 𝒩n+\mathcal{N}_{n}^{+} or 𝒩¯n−\smash{\overline{\mathcal{N}}}_{\!n}^{-} and (right) 𝒩n−\mathcal{N}_{n}^{-} or 𝒩¯n+\smash{\overline{\mathcal{N}}}_{\!n}^{+}, are plotted together to visualize C​PC\!P asymmetries.

6 Fit to observables from data

The parameters of interest are extracted by the least-squares approach,
where a χ2\chi^{2} test statistic is constructed with the observables from data.
This step is referred to as the C​PC\!P fit.
In Sect. 6.1, a joint fit to both BESIII and LHCb observables is performed to determine the C​PC\!P observables,
while in Sect. 6.2, a fit using only BESIII observables is carried out to determine the strong-phase parameters independently.

6.1 Joint fit to determine 𝑪​𝑷\boldsymbol{{C\!P}} observables

For each C​PC\!P tag in BESIII data, the 2×(2​Mh′+1){2\times(2M_{h^{\prime}}+1)} expected observables are computed with Eq. 13 as

⟨𝒩n⟩=hC​P​[Pn++Pn−−2​sK​(2​F+−1)​(−1)n​Cn],\displaystyle\left\langle\mathcal{N}_{n}\right\rangle=h_{{C\!P}}\left[P_{n}^{+}+P_{n}^{-}-2s_{K}(2F_{+}-1)(-1)^{n}C_{n}\right],

(21)

⟨𝒩¯n⟩=hC​P​[Pn−+Pn+−2​sK​(2​F+−1)​(−1)n​Cn],\displaystyle\left\langle\smash{\overline{\mathcal{N}}}_{\!n}\right\rangle=h_{{C\!P}}\left[P_{n}^{-}+P_{n}^{+}-2s_{K}(2F_{+}-1)(-1)^{n}C_{n}\right],

where hC​P=NC​P±ST/2​NfST{h_{{C\!P}}=N_{{C\!P}\pm}^{\rm ST}/2N_{f}^{\rm ST}} is the normalization factor,
with NC​PSTN_{{C\!P}}^{\rm ST} and NfSTN_{f}^{\rm ST} defined as efficiency-corrected ST yields of the C​PC\!P tag and flavor tags, respectively.
The ST yields and efficiencies are fixed to those measured in Ref. [26].
The flavor parameters Pn±P_{n}^{\pm} are fixed to those computed from flavor tags,
and the strong-phase parameters CnC_{n} are allowed to vary.
The χ2\chi^{2} test statistic from C​PC\!P tags is constructed as

χC​P​-tag2=[𝒩n−⟨𝒩n⟩,𝒩¯n−⟨𝒩¯n⟩]T​VC​P−1​[𝒩n−⟨𝒩n⟩,𝒩¯n−⟨𝒩¯n⟩].\chi^{2}_{{C\!P}\text{-tag}}=\left[\mathcal{N}_{n}-\left\langle\mathcal{N}_{n}\right\rangle,\smash{\overline{\mathcal{N}}}_{\!n}-\left\langle\smash{\overline{\mathcal{N}}}_{\!n}\right\rangle\right]^{\rm T}V_{{C\!P}}^{-1}\left[\mathcal{N}_{n}-\left\langle\mathcal{N}_{n}\right\rangle,\smash{\overline{\mathcal{N}}}_{\!n}-\left\langle\smash{\overline{\mathcal{N}}}_{\!n}\right\rangle\right].

(22)

Here, the observed weighted yields 𝒩n\mathcal{N}_{n} are computed from the weighted sums over the relevant data samples, in which the background observables, computed from the weighted sums over simulated background samples, are subtracted.
The covariance matrices VC​PV_{{C\!P}} are computed from the weighted sums as described in Sect. 5.

The 4×(2​Mh′+1)×(2​Mh′′+1){4\times(2M_{h^{\prime}}+1)\times(2M_{h^{\prime\prime}}+1)} expected observables of the self-conjugate tags are computed from Eq. 14 as

⟨𝒩n1​n2s1​s2⟩=hD​D​[Pn1s1​Pn2s2+Pn1s¯1​Pn2s¯2−2​sK1​sK2​(−1)n1​(−1)n2​(Cn1​Cn2+s1​s2​Sn1​Sn2)],\left\langle\mathcal{N}_{n_{1}n_{2}}^{s_{1}s_{2}}\right\rangle=h_{DD}\left[P_{n_{1}}^{s_{1}}P_{n_{2}}^{{s_{2}}}+P_{n_{1}}^{\overline{s}_{1}}P_{n_{2}}^{\overline{s}_{2}}-2s_{K_{1}}s_{K_{2}}(-1)^{n_{1}}(-1)^{n_{2}}(C_{n_{1}}C_{n_{2}}+s_{1}s_{2}S_{n_{1}}S_{n_{2}})\right],

(23)

where hD​D=κ​ND​D/(2​Nf,1ST​Nf,2ST)h_{DD}=\kappa N_{DD}/(2N_{f,1}^{\rm ST}N_{f,2}^{\rm ST}) is the normalization factor.
The factor κ\kappa accounts for the different reconstruction methods between the two DD mesons,
with κ=2ℬ(KS0→π0​π0)/ℬ(KS0→π+​π−){\kappa=\left.2{\mathcal{B}}(\mbox{${{K}^{0}_{\mathrm{S}}}\!\rightarrow{{\pi}^{0}}{{\pi}^{0}}$})\middle/{\mathcal{B}}(\mbox{${{K}^{0}_{\mathrm{S}}}\!\rightarrow{{\pi}^{+}}{{\pi}^{-}}$})\right.} for the KS0(→π0[π0])π+π−{{K}^{0}_{\mathrm{S}}}(\rightarrow{{\pi}^{0}}[{{\pi}^{0}}]){{\pi}^{+}}{{\pi}^{-}} vs. KS0​π+​π−{{K}^{0}_{\mathrm{S}}}{{\pi}^{+}}{{\pi}^{-}} channel,
κ=2{\kappa=2} for the channels where the two DD mesons decay to different final states,
and κ=1{\kappa=1} for the channels where the two DD mesons decay to the same final state.
The branching fractions ℬ⁡(KS0→π0​π0){\mathcal{B}}(\mbox{${{K}^{0}_{\mathrm{S}}}\!\rightarrow{{\pi}^{0}}{{\pi}^{0}}$}) and ℬ⁡(KS0→π+​π−){\mathcal{B}}(\mbox{${{K}^{0}_{\mathrm{S}}}\!\rightarrow{{\pi}^{+}}{{\pi}^{-}}$}) are fixed to their known values [48].
The number of D​D¯D{{\kern 1.79993pt\overline{\kern-1.79993ptD}}} pairs are measured to be ND​D=(28.66±0.25)×106{N_{DD}=(28.66\pm 0.25)\times 10^{6}} [67],
and Nf,1​(2)STN_{f,1(2)}^{\rm ST} correspond to the efficiency-corrected ST yield of flavor tags of the first (second) DD-meson decays. The χ2\chi^{2} test statistic of the self-conjugate tags is constructed in the same way as the C​PC\!P tags,
where 4 sets of observables, 𝒩n1​n2±⁣±\mathcal{N}_{n_{1}n_{2}}^{\pm\pm}, are included.

An additional weak constraint from the amplitude model is included to help determine the strong-phase parameters, following the same approach as in Ref. [26].
Differences in strong-phase parameters between D0→KL0​h′⁣+​h′⁣−{{D}^{0}}\!\rightarrow{{K}^{0}_{\mathrm{L}}}h^{\prime+}h^{\prime-} and D0→KS0​h′⁣+​h′⁣−{{D}^{0}}\!\rightarrow{{K}^{0}_{\mathrm{S}}}h^{\prime+}h^{\prime-} decays, Δ​Cn,Δ​Sn\Delta C_{n},\Delta S_{n}, are constrained to those from the amplitude models.
The amplitude models of the D0→KS0​h′⁣+​h′⁣−{{D}^{0}}\!\rightarrow{{K}^{0}_{\mathrm{S}}}h^{\prime+}h^{\prime-} decays are measured by the Belle and BaBar experiments [31, 32, 33],
while the amplitude models of the D0→KL0​h′⁣+​h′⁣−{{D}^{0}}\!\rightarrow{{K}^{0}_{\mathrm{L}}}h^{\prime+}h^{\prime-} decays are derived from the corresponding KS0​h′⁣+​h′⁣−{{{K}^{0}_{\mathrm{S}}}h^{\prime+}h^{\prime-}} models and the assumption of U-spin symmetry.
Further U-spin breaking effects are taken into account in D0→KL0​π+​π−{{D}^{0}}\!\rightarrow{{K}^{0}_{\mathrm{L}}}{{\pi}^{+}}{{\pi}^{-}} decays from the BESIII measurement [24],
and estimated with the approach described in Ref. [21] for D0→KL0​K+​K−{{D}^{0}}\!\rightarrow{{K}^{0}_{\mathrm{L}}}{{K}^{+}}{{K}^{-}} decays as no measurement is available at the moment.
Uncertainties of Δ​Cn,Δ​Sn\Delta C_{n},\Delta S_{n}, denoted as δ​Δ​Cn,δ​Δ​Sn\delta\Delta C_{n},\delta\Delta S_{n}, are estimated from pseudoexperiments,
in which the amplitude model is altered by sampling input parameters within their measured uncertainties in Refs. [31, 32, 33],
or by using alternative amplitude models that are measured independently [68, 52].
The χ2\chi^{2} test statistic of this constraint, e.g., for Δ​Cn\Delta C_{n}, is constructed as

χΔ​Cn2=∑n=02​Mh+1(Cn′−Cn−⟨Δ​Cn⟩δ​Δ​Cn)2,\chi^{2}_{\Delta C_{n}}=\sum_{n=0}^{2M_{h}+1}\left(\frac{C_{n}^{\prime}-C_{n}-\left\langle\Delta C_{n}\right\rangle}{\delta\Delta C_{n}}\right)^{2},

(24)

where CnC_{n} and Cn′C_{n}^{\prime} are the measured strong-phase parameters of the D0→KS0​h′⁣+​h′⁣−{{D}^{0}}\!\rightarrow{{K}^{0}_{\mathrm{S}}}h^{\prime+}h^{\prime-} and D0→KL0​h′⁣+​h′⁣−{{D}^{0}}\!\rightarrow{{K}^{0}_{\mathrm{L}}}h^{\prime+}h^{\prime-} decays, respectively.
Furthermore, the expected values of ⟨Δ​Cn⟩\left\langle\Delta C_{n}\right\rangle are computed from the amplitude models,
and MhM_{h} denotes the highest Fourier order in the fit with h=π,K{h=\pi,K}.

The χ2\chi^{2} test statistic constructed from BESIII data is

χ2C=χ2KS/L0​h′⁣+​h′⁣−​vs.​C​P​-tag+χ2K0Sh′⁣+h′⁣−vs.K0S/Lh′′+h′′−+χ2Δ​(C,S)n,\chi^{2}_{C}=\chi^{2}_{K^{0}_{\mathrm{S/L}}h^{\prime+}h^{\prime-}~\text{{vs.}}~{C\!P}\text{-tag}}+\chi^{2}_{{{K}^{0}_{\mathrm{S}}}h^{\prime+}h^{\prime-}~\mbox{vs.}~K^{0}_{\mathrm{S/L}}h^{\prime\prime+}h^{\prime\prime-}}+\chi^{2}_{\Delta(C,S)_{n}},

(25)

in which χKS/L0​h′⁣+​h′⁣−​vs.​C​P​-tag2\chi^{2}_{K^{0}_{\mathrm{S/L}}h^{\prime+}h^{\prime-}~\text{{vs.}}~{C\!P}\text{-tag}} contains 12 categories: 4 signal DD decays and 3 tag modes summarized in Table 1. The χ2K0Sh′⁣+h′⁣−vs.K0S/Lh′′+h′′−\chi^{2}_{{{K}^{0}_{\mathrm{S}}}h^{\prime+}h^{\prime-}~\mbox{vs.}~K^{0}_{\mathrm{S/L}}h^{\prime\prime+}h^{\prime\prime-}} term contains 9 self-conjugate tags as explained in Sect. 4.

For LHCb data, the χ2\chi^{2} test statistic can be constructed by the same approach,
using 2×(2​Mh′+1){2\times(2M_{h^{\prime}}+1)} expected signal observables ⟨
 
( ) 
[-.7ex] 𝒩 
±⟩\left\langle\smash{\shortstack{{(\rule[2.15277pt]{12.50002pt}{0.51212pt})}
\\
 [-.7ex] $\mathcal{N}$}}^{\pm}\right\rangle computed from Eq. 6 in each category,
the weighted signal yields 
 
( ) 
[-.7ex] 𝒩 
±\smash{\shortstack{{(\rule[2.15277pt]{12.50002pt}{0.51212pt})}
\\
 [-.7ex] $\mathcal{N}$}}^{\pm} from Eq. 19, and their covariance matrices VB±V_{B^{\pm}} from Eq. 20,

χB2=[𝒩n±−⟨𝒩n±⟩,𝒩¯n±−⟨𝒩¯n±⟩]T​VB±−1​[𝒩n±−⟨𝒩n±⟩,𝒩¯n±−⟨𝒩¯n±⟩].\chi^{2}_{B}=\left[\mathcal{N}_{n}^{\pm}-\left\langle\mathcal{N}_{n}^{\pm}\right\rangle,\smash{\overline{\mathcal{N}}}_{\!n}^{\pm}-\left\langle\smash{\overline{\mathcal{N}}}_{\!n}^{\pm}\right\rangle\right]^{\rm T}V_{{{B}^{\pm}}}^{-1}\left[\mathcal{N}_{n}^{\pm}-\left\langle\mathcal{N}_{n}^{\pm}\right\rangle,\smash{\overline{\mathcal{N}}}_{\!n}^{\pm}-\left\langle\smash{\overline{\mathcal{N}}}_{\!n}^{\pm}\right\rangle\right].

(26)

This construction represents 16 categories of LHCb data.
The normalization factors of each category, hB±h_{{{B}^{\pm}}}, 2×(2​Mh+1){2\times(2M_{h}+1)} flavor parameters of LHCb data in each DD decay and KS0{{K}^{0}_{\mathrm{S}}} reconstruction type, Pn±P_{n}^{\pm}, and C​PC\!P observables of B±→D​K±{{B}^{\pm}}\!\rightarrow D{{K}^{\pm}} decays, x±D​Kx_{\pm}^{DK} and y±D​Ky_{\pm}^{DK}, are free parameters in the fit.
The C​P{C\!P} observables of B±→D​π±{{B}^{\pm}}\!\rightarrow D{{\pi}^{\pm}} decays, as explained in Sect. 2, are described by four B±→D​K±{{B}^{\pm}}\!\rightarrow D{{K}^{\pm}} parameters and two B±→D​π±{{B}^{\pm}}\!\rightarrow D{{\pi}^{\pm}} parameters, which are also free parameters.
Finally, the strong-phase parameters are shared between χC2\chi^{2}_{C} and χB2\chi^{2}_{B}, so that the C​P{C\!P} observables can be extracted from a joint fit to both BESIII and LHCb data. Comparing with those fitted with only the BESIII data at the same orders Mπ=2M_{\pi}=2 and MK=1M_{K}=1, strong-phase parameters from the joint fit have slightly lower uncertainties in SnS_{n}, indicating extra sensitivity provided by LHCb data as presented in Appendix B.
The joint fit is found to have a good quality, with the minimized χ2\chi^{2} corresponding to a pp-value of 1%.

The C​PC\!P observables from the joint fit are shown in Fig. 10,
including the two-dimensional likelihood contours for the pairs of observables (x−D​K,y−D​K){(x_{-}^{DK},y_{-}^{DK})}, (x+D​K,y+D​K){(x_{+}^{DK},y_{+}^{DK})} and (xξD​π,yξD​π){(x_{\xi}^{D\pi},y_{\xi}^{D\pi})}.
The contours are determined from the statistical uncertainties and correlations from the joint fit,
and those from systematic uncertainty studies as detailed in Sect. 7.
The likelihood distributions are assumed to be Gaussian-like and were validated by the binned measurement [17].
The lengths of the two vectors from the origin to (x±D​K,y±D​K)(x_{\pm}^{DK},y_{\pm}^{DK}) are the individual rB±D​Kr_{{{B}^{\pm}}}^{DK} values from B±→D​K±{{B}^{\pm}}\!\rightarrow D{{K}^{\pm}} decays.
The nonzero opening angle between these two vectors, which equals 2​γ2\gamma, shows a clear sign of C​PC\!P violation.

Figure 10: Results of C​PC\!P observables from the joint fit.
The contours correspond to 68% and 95% confidence levels of observable pairs, (left) (x+D​K,y+D​K)(x_{+}^{DK},y_{+}^{DK}) in blue and (x−D​K,y−D​K)(x_{-}^{DK},y_{-}^{DK}) in red, and (right) (xξD​π,yξD​π)(x_{\xi}^{D\pi},y_{\xi}^{D\pi}).

Several cross-checks have been performed to validate the fit procedure.
The fits to data are performed with higher Fourier orders, (Mπ,MK)=(2,2){(M_{\pi},M_{K})=(2,2)}, (3,1)(3,1) and (3,2)(3,2),
and the results of C​PC\!P observables and strong-phase parameters are found to be consistent.
Fits are also performed using only the LHCb data with both the binned and unbinned approaches, in which the unbinned strong-phase parameters are fixed to those measured in Fig. 11,
and the binned strong-phase parameters are fixed to those measured in Refs. [26, 25].
Correlation between the resultant C​PC\!P observables was studied with pseudoexperiments and was found to be around 0.8 for each of the observables.
Using the same pseudoexperiments their consistency in six-dimensional parameter space was found to correspond to a pp-value of 39%.

Searches for intrinsic bias are conducted by generating and fitting an ensemble of pseudoexperiments.
The yields and mass distributions are based on the fits to data in Sect. 4,
and the distributions across the phase space are generated from the baseline amplitude models [31, 32, 33].
Standardized-residual distributions of the C​PC\!P observables are found to be consistent with the standard normal distribution.

6.2 Determination of strong-phase parameters

The strong-phase parameters Cn,SnC_{n},S_{n} serve as crucial inputs for future measurements of γ\gamma with the novel approach.
To avoid potential correlations between future γ\gamma measurements and this measurement, results of these parameters determined by a fit to BESIII data alone with the KS/L0​π+​π−K^{0}_{\mathrm{S/L}}{{\pi}^{+}}{{\pi}^{-}} and KS/L0​K+​K−K^{0}_{\mathrm{S/L}}{{K}^{+}}{{K}^{-}} channels combined are also reported in this analysis.
The Fourier orders are chosen to be one order higher than in the baseline with Mπ=3M_{\pi}=3 and MK=2M_{K}=2, so that the truncation choice does not limit future higher-precision measurements.
The strong-phase parameters are obtained with the χ2\chi^{2} test statistic as written in Eq. 25 and listed in Table 3, where the first uncertainties are statistical and the second systematic. The correlations between each source of uncertainty are included in the Supplemental Material [69].
Figure 11 shows good agreement between the model predictions and fitted results of the strong-phase parameters, where the statistical and systematic uncertainties have been combined.

Figure 11: Strong-phase parameters of the (top) D→KS/L0​π+​π−D\!\rightarrow K^{0}_{\mathrm{S/L}}{{\pi}^{+}}{{\pi}^{-}} and (bottom) D→KS/L0​K+​K−D\!\rightarrow K^{0}_{\mathrm{S/L}}{{K}^{+}}{{K}^{-}} decays from BESIII data.

Table 3: Results of the strong-phase parameters Cn,SnC_{n},S_{n} from BESIII-only data under Fourier orders Mπ=3M_{\pi}=3, MK=2M_{K}=2.

nn
Weight
CnC_{n}
SnS_{n}

KS0​π+​π−{{K}^{0}_{\mathrm{S}}}{{\pi}^{+}}{{\pi}^{-}}

0
woptw^{\rm opt}
0.0684±0.0038±0.00290.0684\pm 0.0038\pm 0.0029
−0.0179±0.0087±0.0037-0.0179\pm 0.0087\pm 0.0037

1
wopt​sin⁡ϕw^{\rm opt}\sin{\phi}
0.0100±0.0023±0.00070.0100\pm 0.0023\pm 0.0007
0.0941±0.0051±0.00210.0941\pm 0.0051\pm 0.0021

2
wopt​cos⁡ϕw^{\rm opt}\cos{\phi}
0.1730±0.0028±0.00140.1730\pm 0.0028\pm 0.0014
0.0089±0.0068±0.00210.0089\pm 0.0068\pm 0.0021

3
wopt​sin⁡2​ϕw^{\rm opt}\sin{2\phi}
0.0067±0.0021±0.00070.0067\pm 0.0021\pm 0.0007
0.0186±0.0064±0.00130.0186\pm 0.0064\pm 0.0013

4
wopt​cos⁡2​ϕw^{\rm opt}\cos{2\phi}
0.0418±0.0031±0.00120.0418\pm 0.0031\pm 0.0012
0.0138±0.0068±0.00210.0138\pm 0.0068\pm 0.0021

5
wopt​sin⁡3​ϕw^{\rm opt}\sin{3\phi}
0.0209±0.0024±0.00070.0209\pm 0.0024\pm 0.0007
−0.0141±0.0067±0.0015-0.0141\pm 0.0067\pm 0.0015

6
wopt​cos⁡3​ϕw^{\rm opt}\cos{3\phi}
0.0740±0.0030±0.00090.0740\pm 0.0030\pm 0.0009
−0.0016±0.0065±0.0015-0.0016\pm 0.0065\pm 0.0015

KL0​π+​π−{{K}^{0}_{\mathrm{L}}}{{\pi}^{+}}{{\pi}^{-}}

0
woptw^{\rm opt}
0.1703±0.0038±0.00150.1703\pm 0.0038\pm 0.0015
−0.0120±0.0100±0.0027-0.0120\pm 0.0100\pm 0.0027

1
wopt​sin⁡ϕw^{\rm opt}\sin{\phi}
−0.0063±0.0015±0.0006-0.0063\pm 0.0015\pm 0.0006
0.0747±0.0062±0.00160.0747\pm 0.0062\pm 0.0016

2
wopt​cos⁡ϕw^{\rm opt}\cos{\phi}
0.2365±0.0034±0.00170.2365\pm 0.0034\pm 0.0017
−0.0074±0.0079±0.0019-0.0074\pm 0.0079\pm 0.0019

3
wopt​sin⁡2​ϕw^{\rm opt}\sin{2\phi}
−0.0105±0.0020±0.0007-0.0105\pm 0.0020\pm 0.0007
0.0491±0.0080±0.00110.0491\pm 0.0080\pm 0.0011

4
wopt​cos⁡2​ϕw^{\rm opt}\cos{2\phi}
0.1238±0.0030±0.00110.1238\pm 0.0030\pm 0.0011
−0.0124±0.0074±0.0015-0.0124\pm 0.0074\pm 0.0015

5
wopt​sin⁡3​ϕw^{\rm opt}\sin{3\phi}
−0.0105±0.0022±0.0007-0.0105\pm 0.0022\pm 0.0007
0.0297±0.0070±0.00140.0297\pm 0.0070\pm 0.0014

6
wopt​cos⁡3​ϕw^{\rm opt}\cos{3\phi}
0.1254±0.0031±0.00130.1254\pm 0.0031\pm 0.0013
−0.0023±0.0077±0.0009-0.0023\pm 0.0077\pm 0.0009

KS0​K+​K−{{K}^{0}_{\mathrm{S}}}{{K}^{+}}{{K}^{-}}

0
woptw^{\rm opt}
0.0401±0.0111±0.00880.0401\pm 0.0111\pm 0.0088
0.0391±0.0251±0.00580.0391\pm 0.0251\pm 0.0058

1
wopt​sin⁡ϕw^{\rm opt}\sin{\phi}
−0.0089±0.0055±0.0017-0.0089\pm 0.0055\pm 0.0017
0.0978±0.0203±0.00730.0978\pm 0.0203\pm 0.0073

2
wopt​cos⁡ϕw^{\rm opt}\cos{\phi}
0.2809±0.0097±0.00590.2809\pm 0.0097\pm 0.0059
0.0227±0.0159±0.00280.0227\pm 0.0159\pm 0.0028

3
wopt​sin⁡2​ϕw^{\rm opt}\sin{2\phi}
−0.0024±0.0063±0.0018-0.0024\pm 0.0063\pm 0.0018
−0.0424±0.0297±0.0060-0.0424\pm 0.0297\pm 0.0060

4
wopt​cos⁡2​ϕw^{\rm opt}\cos{2\phi}
0.0509±0.0080±0.00540.0509\pm 0.0080\pm 0.0054
−0.0275±0.0163±0.0027-0.0275\pm 0.0163\pm 0.0027

KL0​K+​K−{{K}^{0}_{\mathrm{L}}}{{K}^{+}}{{K}^{-}}

0
woptw^{\rm opt}
0.0583±0.0112±0.00820.0583\pm 0.0112\pm 0.0082
0.0213±0.0254±0.00580.0213\pm 0.0254\pm 0.0058

1
wopt​sin⁡ϕw^{\rm opt}\sin{\phi}
−0.0079±0.0055±0.0016-0.0079\pm 0.0055\pm 0.0016
0.0840±0.0242±0.00770.0840\pm 0.0242\pm 0.0077

2
wopt​cos⁡ϕw^{\rm opt}\cos{\phi}
0.2794±0.0110±0.00540.2794\pm 0.0110\pm 0.0054
0.0238±0.0163±0.00280.0238\pm 0.0163\pm 0.0028

3
wopt​sin⁡2​ϕw^{\rm opt}\sin{2\phi}
−0.0128±0.0064±0.0014-0.0128\pm 0.0064\pm 0.0014
−0.0391±0.0276±0.0056-0.0391\pm 0.0276\pm 0.0056

4
wopt​cos⁡2​ϕw^{\rm opt}\cos{2\phi}
0.0701±0.0082±0.00500.0701\pm 0.0082\pm 0.0050
−0.0211±0.0152±0.0024-0.0211\pm 0.0152\pm 0.0024

Based on a pseudoexperiment study described in Sect. 6.1, the average standardized residuals of three strong-phase parameters of D→KS/L0​π+​π−D\!\rightarrow K^{0}_{\mathrm{S/L}}{{\pi}^{+}}{{\pi}^{-}} decays and half of all strong-phase parameters of D→KS/L0​K+​K−D\!\rightarrow K^{0}_{\mathrm{S/L}}{{K}^{+}}{{K}^{-}} decays are found to deviate from zero.
Their biases are less than 30% of the statistical uncertainties and are due to low statistics in data,
so are not used to correct central values of the strong-phase parameters, but assigned as a source of systematic uncertainty as described in Sect. 7.
The statistical uncertainties of five strong-phase parameters of D→KS/L0​K+​K−D\!\rightarrow K^{0}_{\mathrm{S/L}}{{K}^{+}}{{K}^{-}} decays are found to be underestimated under the current sample size,
thus corrections of statistical uncertainties are applied to the fit results.

7 Systematic uncertainties

The systematic uncertainties affecting this measurement are considered from two groups: those associated with the BESIII inputs and those related to LHCb.
The uncertainties of the strong-phase parameters are summarized in Tables 4–7,
while those for the C​PC\!P observables are listed in Table 8.

The dominant sources of systematic uncertainties from BESIII are identified as follows: biases estimated from pseudoexperiments; statistical fluctuations in the flavor-tag observables and ST yields; uncertainties associated with charm mixing and the hadronic parameters used for flavor tags; limited statistics of the uniform phase-space simulation samples for efficiency profiles, and the finite statistics of background samples for modeling the background shapes and yields.

Most of these uncertainties can be addressed through dedicated pseudoexperiments. The sets of values for the flavor and charm-mixing parameters, ST yields, efficiency profiles, signal efficiencies, background models and yields are generated independently according to Gaussian distributions with means and standard deviations set to their baseline values and uncertainties. The C​PC\!P fit is then repeated for each varied set, and the standard deviation of the resulting pull distribution for each strong-phase parameter is assigned as the relative systematic uncertainty.

Since the DCS contributions in flavor tags and some of the correlation matrices depend on the amplitude models, the imperfection of the models can bias the strong-phase parameters. To assess this, alternative amplitude models are employed, and the differences in the extracted strong-phase parameters are assigned as the corresponding systematic uncertainties.

Another source of uncertainty originates from the tracking efficiency of pions from KS0{{K}^{0}_{\mathrm{S}}} decays. This effect is due to the relatively low momentum and displaced topology of these secondary pions, making their tracking efficiencies less well constrained than those of prompt tracks. Although the overall uncertainties related to PID and tracking are found to be negligible after correcting the efficiency with the method described in Sect. 5, residual mismodeling of the pions from KS0{{K}^{0}_{\mathrm{S}}} decays can still affect the signal efficiencies. To quantify this, the signal efficiencies and efficiency profiles are varied according to the tracking-efficiency corrections derived from π±\pi^{\pm} control samples. The resulting shifts in the refitted strong-phase parameters are taken as systematic uncertainties.

Possible mismodeling of the detector resolution between data and simulation is taken into consideration. The efficiency profiles are derived separately using truth-level and reconstruction-level information of reconstructed events in the simulation samples. The differences in the extracted strong-phase parameters with these two kinds of efficiency profiles are treated as related systematic uncertainties.

For each source of systematic uncertainty from BESIII, its impact on the strong-phase parameters is studied by analyzing data or pseudoexperiments at Mπ=3M_{\pi}=3 and MK=2M_{K}=2. All sources of systematic uncertainties can be found in Tables 4–7.

Table 4: Summary of all uncertainties on the strong-phase parameters of D0→KS0​π+​π−{{D}^{0}}\!\rightarrow{{K}^{0}_{\mathrm{S}}}{{\pi}^{+}}{{\pi}^{-}} decays from BESIII data. All uncertainties are quoted in units of 10−410^{-4}.

Source
C0C_{0}
C1C_{1}
C2C_{2}
C3C_{3}
C4C_{4}
C5C_{5}
C6C_{6}
S0S_{0}
S1S_{1}
S2S_{2}
S3S_{3}
S4S_{4}
S5S_{5}
S6S_{6}

Statistical uncertainty
38
23
28
21
31
24
30
87
51
68
64
68
67
65

Fitter bias
1
<<1
4
<<1
2
1
1
<<1
2
9
3
6
1
3

Flavor parameters
1
3
8
5
9
5
8
2
10
1
8
1
7
6

Input parameters
6
1
2
<<1
3
<<1
<<1
20
4
7
1
3
2
1

DCS corrections
6
1
<<1
<<1
4
1
<<1
2
2
2
4
1
<<1
1

Correlation matrices
<<1
1
<<1
1
<<1
<<1
<<1
3
1
3
<<1
2
<<1
2

ST yields
22
4
8
1
2
2
1
12
<<1
5
4
5
6
6

Efficiency profiles
3
3
3
2
3
1
2
3
1
2
2
2
2
2

DT efficiencies
3
1
1
<<1
<<1
<<1
<<1
1
<<1
1
<<1
1
<<1
1

Background shapes
3
1
3
2
3
2
2
11
6
7
7
6
6
6

Background contributions
6
1
3
<<1
1
1
2
1
1
1
1
1
1
1

Tracking of π±{{\pi}^{\pm}} in KS0{{K}^{0}_{\mathrm{S}}}
6
<<1
<<1
1
<<1
1
7
5
<<1
2
1
2
3
3

Resolution differences
3
4
5
2
3
3
2
14
8
6
5
15
10
10

Total systematic uncertainty
29
7
14
7
12
7
9
37
21
21
13
21
15
15

Table 5: Summary of all uncertainties on the strong-phase parameters of D0→KL0​π+​π−{{D}^{0}}\!\rightarrow{{K}^{0}_{\mathrm{L}}}{{\pi}^{+}}{{\pi}^{-}} decays from BESIII data. All uncertainties are quoted in units of 10−410^{-4}.

Source
C0C_{0}
C1C_{1}
C2C_{2}
C3C_{3}
C4C_{4}
C5C_{5}
C6C_{6}
S0S_{0}
S1S_{1}
S2S_{2}
S3S_{3}
S4S_{4}
S5S_{5}
S6S_{6}

Statistical uncertainty
38
15
34
20
30
22
31
100
62
79
80
74
70
77

Fitter bias
4
<<1
1
<<1
2
1
3
3
11
2
4
<<1
2
<<1

Flavor parameters
7
2
7
3
5
3
5
20
7
13
7
10
6
7

Input parameters
3
<<1
1
<<1
2
<<1
<<1
3
<<1
4
1
3
1
2

DCS corrections
4
<<1
2
<<1
4
<<1
2
1
1
1
<<1
<<1
1
<<1

Correlation matrices
1
<<1
<<1
<<1
1
<<1
1
3
1
2
4
2
<<1
1

ST yields
9
1
6
2
4
2
2
9
<<1
4
1
8
5
4

Efficiency profiles
3
1
3
2
2
2
2
4
1
3
1
2
2
2

DT efficiencies
4
<<1
3
<<1
2
<<1
1
1
<<1
1
<<1
1
<<1
1

Background shapes
4
2
6
2
4
2
5
11
7
8
8
7
7
6

Background contributions
8
1
5
1
4
1
2
2
2
1
2
1
1
1

Tracking of π±{{\pi}^{\pm}} in KS0{{K}^{0}_{\mathrm{S}}}
<<1
<<1
<<1
<<1
<<1
<<1
<<1
5
<<1
3
1
1
1
1

Resolution differences
5
5
12
6
4
5
9
8
4
7
3
8
8
4

Total systematic uncertainty
15
6
17
7
11
7
13
27
16
19
11
15
14
9

Table 6: Summary of all uncertainties on the strong-phase parameters of D0→KS0​K+​K−{{D}^{0}}\!\rightarrow{{K}^{0}_{\mathrm{S}}}{{K}^{+}}{{K}^{-}} decays from BESIII data. All uncertainties are quoted in units of 10−410^{-4}.

Source
C0C_{0}
C1C_{1}
C2C_{2}
C3C_{3}
C4C_{4}
S0S_{0}
S1S_{1}
S2S_{2}
S3S_{3}
S4S_{4}

Statistical uncertainty
111
55
97
63
80
251
203
159
297
163

Fitter bias
14
7
1
8
4
18
31
7
33
7

Flavor parameters
76
14
45
14
50
21
37
11
33
11

Input parameters
4
1
4
1
3
8
6
4
8
4

DCS corrections
4
1
8
<<1
3
3
6
<<1
1
<<1

Correlation matrices
<<1
<<1
<<1
<<1
<<1
<<1
1
1
2
<<1

ST yields
11
1
1
1
1
1
1
1
1
1

Efficiency profiles
7
2
8
3
7
16
10
5
11
6

DT efficiencies
5
1
1
1
1
3
1
1
2
1

Background shapes
10
4
8
4
6
28
32
16
30
20

Background contributions
34
4
31
4
16
13
22
11
16
9

Tracking of π±{{\pi}^{\pm}} in KS0{{K}^{0}_{\mathrm{S}}}
3
<<1
1
<<1
<<1
<<1
2
<<1
1
1

Resolution differences
12
1
16
1
7
36
36
14
<<1
7

Total systematic uncertainty
88
17
59
18
54
58
73
28
60
27

Table 7: Summary of all uncertainties on the strong-phase parameters of D0→KL0​K+​K−{{D}^{0}}\!\rightarrow{{K}^{0}_{\mathrm{L}}}{{K}^{+}}{{K}^{-}} decays from BESIII data. All uncertainties are quoted in units of 10−410^{-4}.

Source
C0C_{0}
C1C_{1}
C2C_{2}
C3C_{3}
C4C_{4}
S0S_{0}
S1S_{1}
S2S_{2}
S3S_{3}
S4S_{4}

Statistical uncertainty
112
55
110
64
82
254
242
163
276
152

Fitter bias
21
8
10
7
5
18
34
7
33
5

Flavor parameters
70
13
39
10
46
21
41
11
31
10

Input parameters
3
1
2
2
2
8
7
4
7
4

DCS corrections
11
<<1
1
<<1
5
3
6
<<1
1
<<1

Correlation matrices
1
1
1
1
1
4
4
1
3
2

ST yields
7
1
1
1
2
1
1
1
1
9

Efficiency profiles
11
2
7
3
8
16
10
5
11
5

DT efficiencies
6
1
1
1
1
3
2
1
2
1

Background shapes
9
3
11
4
6
28
31
16
27
17

Background contributions
31
4
32
5
15
13
27
11
15
8

Tracking of π±{{\pi}^{\pm}} in KS0{{K}^{0}_{\mathrm{S}}}
4
<<1
2
1
1
<<1
2
<<1
1
1

Resolution differences
9
1
10
1
2
36
34
14
<<1
7

Total systematic uncertainty
82
16
54
14
50
58
77
28
56
24

In the analysis of LHCb data, the major difference between this measurement and Ref. [17] is the approach to extract observables.
In this measurement, background subtraction is performed with the sPlot technique, which assumes that the BB-candidate mass and Dalitz-plot coordinates are independent variables.
Potential correlations between these variables may introduce bias on the C​PC\!P observables,
and are evaluated by inspecting the Dalitz-plot-dependent BB-candidate mass distributions in simulated samples.
Pseudoexperiments generated with such correlations are fitted with the baseline model.
Mean biases of the C​PC\!P observables are taken as systematic uncertainties.

Efficiency profiles from the simulated signal are needed in this measurement to correct B±→D​h±{{B}^{\pm}}\!\rightarrow Dh^{\pm} observables,
thus imperfect modeling in simulation is considered as a source of systematic uncertainty.
Correction profiles are obtained by comparing Dalitz-plot distributions between simulated signal and B±→D​π±{{B}^{\pm}}\!\rightarrow D{{\pi}^{\pm}} signal from data.
This is possible as the B±→D​π±{{B}^{\pm}}\!\rightarrow D{{\pi}^{\pm}} decays have clean signal peaks in data and minimal C​PC\!P violation.
New efficiency profiles are obtained by multiplying the baseline profiles with the correction profiles, and are used to generate pseudoexperiments.
Weighted signal yields from the ensemble are computed with the baseline efficiency profiles, and mean biases of the C​PC\!P observables are taken as systematic uncertainties.
The sample size of simulated signal can affect the precision of the efficiency profiles, so its impact is assessed by sampling efficiency parameters from the fits to simulated samples and computing a number of new profiles.
The subsequent computation of observables and fits to data indicate that the standard deviations
of the C​PC\!P observables are two orders of magnitude smaller than other systematic uncertainties.
Therefore, no systematic uncertainty due to the simulation sample size is assigned.

The effect of detector resolution is taken into account by smearing the Dalitz-plot distributions in pseudoexperiments.
The smearing is based on the m±2m_{\pm}^{2} resolution as determined in simulation and scaled to account for data-simulation differences.
The baseline fit model is then applied to the ensemble, and mean biases of the C​PC\!P observables are taken as systematic uncertainties.
This source is equivalent to the bin-migration effect described in Ref. [17].

Systematic uncertainties related to the global fit are estimated by including extra effects when generating pseudoexperiments, and fitting the ensemble with the baseline model.
Several background decays are not considered in the baseline fit due to their low yields,
which include the semileptonic B→D​μ​νμB\!\rightarrow D\mu{{\nu}_{\mu}} and B→D(→KS0πlνℓ)hB\!\rightarrow D(\rightarrow{{K}^{0}_{\mathrm{S}}}\pi l{{\nu}_{\ell}})h (l=μ,el=\mu,e) decays,
the baryonic Λb0→D​p​π−{{\mathchar 28931\relax}^{0}_{b}}\!\rightarrow Dp\pi^{-} and Λb0→Λc+(→pKS0π+π−)h−{{\mathchar 28931\relax}^{0}_{b}}\!\rightarrow{{\mathchar 28931\relax}^{+}_{c}}(\rightarrow p{{K}^{0}_{\mathrm{S}}}{{\pi}^{+}}{{\pi}^{-}})h^{-} decays,
the charmless B±→KS0​h+​h−​h′⁣±{{B}^{\pm}}\!\rightarrow{{K}^{0}_{\mathrm{S}}}h^{+}h^{-}h^{\prime\pm} decays,
and the B±→D(→K±π∓)KS0π±{{B}^{\pm}}\!\rightarrow D(\rightarrow{{K}^{\pm}}{{\pi}^{\mp}}){{K}^{0}_{\mathrm{S}}}{{\pi}^{\pm}} decays where the final-state tracks are wrongly assigned.
Their yields relative to signal are estimated from either LHCb simulation or fast simulation [63], as well as the BB-candidate mass and Dalitz-plot distributions.
Alternative parameter sets describing the BB-candidate mass are obtained by resampling data and simulated samples,
and propagated to the fit in the tighter mass region and the C​PC\!P fit.
The relative yields of partially reconstructed background and PID efficiencies for misidentified background are fixed in the global fit.
They are varied within their uncertainties in the mass fits, and standard deviations of the C​PC\!P observables from the C​PC\!P fit are assigned as systematic uncertainties.
The partially reconstructed backgrounds are treated as a whole with the same Dalitz-plot distributions and uniform D​hDh mass distributions across the phase space,
which ignores underlying physics effects like C​PC\!P violation in B→D(∗)​K(∗)B\!\rightarrow D^{(*)}K^{(*)} decays.
These effects are taken into account when generating pseudoexperiments, where the parameters in Ref. [70] are used.
Global fit-related systematic uncertainties are evaluated by adapting the same approach as in Ref. [17].

The amount of C​PC\!P violation in KS0{{K}^{0}_{\mathrm{S}}} decay and interactions between detector materials and neutral kaons are ignored in the measurement.
These effects are included to generate a pseudosignal following the approach described in Ref. [71].
The baseline fit model is used to fit the pseudosignal, and biases on the C​PC\!P observables are assigned as systematic uncertainties.
The impact of ignoring DD mixing in LHCb data is assessed by the same approach,
where the signal model including DD mixing is described in Ref. [30].

From pseudoexperiments, increasing bias on the C​PC\!P observables is found when the Fourier orders increase.
This is due to sizeable correlations present within the covariance matrix determined from weighted sums of data in Eq. 20.
Nevertheless, biases found in pseudoexperiments under baseline Fourier orders are small and assigned as a source of systematic uncertainty.

All uncertainties of C​PC\!P observables are summarized in Table 8,
and the systematic uncertainties are found to be one order of magnitude smaller than the statistical ones.
For the sources that have been studied in both this measurement and Ref. [17],
their impact is found to be at the same level.
The extra sources due to the new method are not dominant,
making the total systematic uncertainties comparable to those in Ref. [17].
The overall systematic uncertainties on the C​PC\!P observables from BESIII are at the same order as those from LHCb sources, which are much smaller than the statistical uncertainties.

Table 8: Summary of all uncertainties on the C​PC\!P observables from BESIII and LHCb data.
All uncertainties are quoted in units of 10−210^{-2}.

Source
σ⁡(x−D​K)\sigma(x_{-}^{DK})
σ⁡(y−D​K)\sigma(y_{-}^{DK})
σ⁡(x+D​K)\sigma(x_{+}^{DK})
σ⁡(y+D​K)\sigma(y_{+}^{DK})
σ⁡(xξD​π)\sigma(x_{\xi}^{D\pi})
σ⁡(yξD​π)\sigma(y_{\xi}^{D\pi})

Statistical uncertainty
0.980.98
1.391.39
1.021.02
1.401.40
2.092.09
2.372.37

Correlation between m⁡(D​h)m(Dh) and m±2m_{\pm}^{2}
0.030.03
0.050.05
<0.01<0.01
0.010.01
0.080.08
0.050.05

Data/simulation disagreement
0.020.02
0.100.10
0.060.06
0.100.10
0.040.04
0.030.03

Resolution
0.050.05
0.110.11
0.150.15
0.100.10
0.100.10
0.110.11

Small backgrounds
0.150.15
0.110.11
0.130.13
0.180.18
0.130.13
0.080.08

Mass-shape parameters
0.070.07
0.070.07
0.050.05
0.090.09
0.150.15
0.100.10

Fixed-yield ratios
0.020.02
0.020.02
0.030.03
0.020.02
0.020.02
0.010.01

PID efficiencies
0.020.02
0.030.03
0.030.03
0.030.03
0.020.02
0.010.01

Low-mass physics effects
0.010.01
0.060.06
0.080.08
0.130.13
0.030.03
0.050.05

C​PC\!P violation in KS0{{K}^{0}_{\mathrm{S}}} decays
0.010.01
0.090.09
0.030.03
0.070.07
0.180.18
0.060.06

DD mixing
0.030.03
0.030.03
0.040.04
0.040.04
0.010.01
0.020.02

Fit bias
0.030.03
0.030.03
0.060.06
0.090.09
0.070.07
0.020.02

Systematic uncertainty from LHCb
0.180.18
0.230.23
0.240.24
0.310.31
0.300.30
0.200.20

Systematic uncertainty from BESIII
0.070.07
0.090.09
0.080.08
0.060.06
0.010.01
0.020.02

Total systematic uncertainty
0.200.20
0.250.25
0.260.26
0.310.31
0.300.30
0.200.20

8 Results of 𝜸\gamma

From the joint analysis of BESIII and LHCb data, the C​PC\!P observables, determined from B±→D(→KS0h′⁣+h′⁣−)h±{{B}^{\pm}}\!\rightarrow D(\rightarrow{{K}^{0}_{\mathrm{S}}}h^{\prime+}h^{\prime-})h^{\pm} decays, are

x−D​K\displaystyle x_{-}^{DK}
=( 5.84±0.98±0.20)×10−2,\displaystyle=(~~\,5.84\pm 0.98\pm 0.20)\times 10^{-2},

y−D​K\displaystyle y_{-}^{DK}
=( 6.74±1.39±0.25)×10−2,\displaystyle=(~~\,6.74\pm 1.39\pm 0.25)\times 10^{-2},

x+D​K\displaystyle x_{+}^{DK}
=(−9.68±1.02±0.26)×10−2,\displaystyle=(-9.68\pm 1.02\pm 0.26)\times 10^{-2},

y+D​K\displaystyle y_{+}^{DK}
=(−2.42±1.40±0.31)×10−2,\displaystyle=(-2.42\pm 1.40\pm 0.31)\times 10^{-2},

xξD​π\displaystyle x_{\xi}^{D\pi}
=(−6.57±2.09±0.30)×10−2,\displaystyle=(-6.57\pm 2.09\pm 0.30)\times 10^{-2},

yξD​π\displaystyle y_{\xi}^{D\pi}
=(−1.12±2.37±0.20)×10−2,\displaystyle=(-1.12\pm 2.37\pm 0.20)\times 10^{-2},

where the first uncertainties are statistical and the second systematic.
Statistical and systematic uncertainties from BESIII and LHCb are also analyzed separately to study their individual impact.
The values and correlations of each source of uncertainty are summarized in Tables 12–15 of Appendix C.

The measured C​PC\!P observables can be reinterpreted by a frequentist treatment as described in Refs. [72, 73], which is denoted as the Plugin method.
The solution in the region 0<γ<180∘{0<\gamma<180{{}^{\circ}}} is chosen, which is

γ\displaystyle\gamma
=(71.3±5.0),∘\displaystyle=(71.3\pm 5.0){{}^{\circ}},

rBD​K\displaystyle r_{B}^{DK}
=0.0949−0.0085+0.0086,\displaystyle=0.0949^{+0.0086}_{-0.0085},

δBD​K\displaystyle\delta_{B}^{DK}
=(121.6−5.9+5.6),∘\displaystyle=(121.6^{+5.6}_{-5.9}){{}^{\circ}},

rBD​π\displaystyle r_{B}^{D\pi}
=0.0064−0.0019+0.0021,\displaystyle=0.0064^{+0.0021}_{-0.0019},

δBD​π\displaystyle\delta_{B}^{D\pi}
=(311−20+17).∘\displaystyle=(311^{+17}_{-20}){{}^{\circ}}.

These are consistent with the world-average values [3, 74] and the previous measurement using the binned phase-space approach [17] when considering statistical correlations estimated by pseudoexperiments.
The uncertainty of γ\gamma is smaller than the previous measurement, marking the most precise single measurement of γ\gamma to date.
A one-dimensional confidence-level scan of γ\gamma is shown in Fig. 12,
in which a profile-likelihood method, denoted as Prob, is also applied.
Results from the Plugin and Prob methods are found to be identical.
Figure 13 shows the two-dimensional confidence levels of the (γ,rB)(\gamma,r_{B}) and (δB,rB)(\delta_{B},r_{B}) pairs.

Figure 12: One-dimensional scan of the confidence level of the CKM angle γ\gamma from the Plugin and Prob methods.

Figure 13: Two-dimensional confidence levels of the indicated parameter pairs from the Prob method. The contours correspond to 68% and 95% confidence levels of the observable pairs.

Additional studies are performed to evaluate uncertainties of γ\gamma from different sources and different approaches.
The statistical and systematic uncertainties from LHCb are found to be 4.9∘4.9{{}^{\circ}} and 0.8∘0.8{{}^{\circ}}, respectively,
which are compatible with the previous binned measurement [17].
The BESIII data has an impact of 0.6∘0.6{{}^{\circ}} on γ\gamma, which is smaller than that in Ref. [17], owing to the increase in the BESIII sample size.
The LHCb data is refitted using the binned approach with updated strong-phase inputs from D→KS0​π+​π−D\!\rightarrow{{K}^{0}_{\mathrm{S}}}{{\pi}^{+}}{{\pi}^{-}} decays [26] and using the unbinned approach with higher orders.
Results are listed in Table 9, where the uncertainties are from LHCb statistics only.
It is shown that the statistical uncertainty of γ\gamma is improved by 5% in the unbinned method.
The precision of γ\gamma is found to be improved when employing higher Fourier orders,
indicating that more information can be extracted from higher-order observables.
This is because the model to compute weights does not perfectly describe the data.
In future unbinned measurement of γ\gamma, better precision can be achieved by incorporating higher-order observables in the fits.

Table 9: Results of the extracted parameters (γ,rBD​K,δBD​K,rBD​π,δBD​π)(\gamma,r_{B}^{DK},\delta_{B}^{DK},r_{B}^{D\pi},\delta_{B}^{D\pi}), where the uncertainties are statistical only, deriving from LHCb data.

Configuration
γ()∘\gamma~({{}^{\circ}})
rBD​K​(10−2)r_{B}^{DK}~(10^{-2})
δBD​K()∘\delta_{B}^{DK}~({{}^{\circ}})
rBD​π​(10−2)r_{B}^{D\pi}~(10^{-2})
δBD​π()∘\delta_{B}^{D\pi}~({{}^{\circ}})

Binned (refitted)
67.767.7 
±\pm  
5.15.1
9.619.61 
+−{}^{+}_{-}  
0.780.77{}_{0.77}^{0.78}
118.6−5.6+5.2118.6^{+5.2}_{-5.6}
0.530.53 
±\pm  
0.190.19
286−28+24286^{+24}_{-28}

Mπ=2M_{\pi}=2, MK=1M_{K}=1 (baseline)
71.371.3 
±\pm  
4.94.9
9.489.48 
±\pm  
0.810.81
121.6−5.5+5.3121.6_{-5.5}^{+5.3}
0.640.64 
±\pm  
0.200.20
311−20+16311^{+16}_{-20}

Mπ=2M_{\pi}=2, MK=2M_{K}=2
71.671.6 
+−{}^{+}_{-}  
4.84.7{}^{4.7}_{4.8}
9.589.58 
+−{}^{+}_{-}  
0.800.81{}^{0.81}_{0.80}
122.5−5.4+5.1122.5^{+5.1}_{-5.4}
0.620.62 
+−{}^{+}_{-}  
0.190.20{}^{0.20}_{0.19}
309−20+17309^{+17}_{-20}

Mπ=3M_{\pi}=3, MK=1M_{K}=1
71.071.0 
±\pm  
4.94.9
9.369.36 
+−{}^{+}_{-}  
0.800.79{}_{0.79}^{0.80}
120.7−5.6+5.2120.7_{-5.6}^{+5.2}
0.640.64 
+−{}^{+}_{-}  
0.190.20{}^{0.20}_{0.19}
311−20+16311^{+16}_{-20}

Mπ=3M_{\pi}=3, MK=2M_{K}=2
71.371.3 
+−{}^{+}_{-}  
4.84.9{}_{4.9}^{4.8}
9.469.46 
+−{}^{+}_{-}  
0.800.79{}_{0.79}^{0.80}
121.5−5.5+5.1121.5^{+5.1}_{-5.5}
0.630.63 
+−{}^{+}_{-}  
0.200.19{}^{0.19}_{0.20}
309−20+17309^{+17}_{-20}

9 Conclusions

In this paper and an accompanying Letter [27], a novel approach developed from Ref. [22] is adopted to determine the C​PC\!P observables that lead to a precise measurement of the CKM angle γ\gamma.
The approach uses per-event weights that account for the strong-phase variation, magnitude of the decay amplitude and background level to optimize the sensitivity to γ\gamma.
The approach is employed in datasets collected by the BESIII and LHCb detectors,
corresponding to integrated luminosities of 8 fb−1\text{\,fb}^{-1} and 9 fb−1\text{\,fb}^{-1}, respectively.
The CKM angle γ\gamma is determined from B±→D​K±{{B}^{\pm}}\!\rightarrow D{{K}^{\pm}} and B±→D​π±{{B}^{\pm}}\!\rightarrow D{{\pi}^{\pm}} decays followed by D→KS0​π+​π−D\!\rightarrow{{K}^{0}_{\mathrm{S}}}{{\pi}^{+}}{{\pi}^{-}} and D→KS0​K+​K−D\!\rightarrow{{K}^{0}_{\mathrm{S}}}{{K}^{+}}{{K}^{-}} decays with input of the strong-phase information
from the doubly-tagged ψ⁡(3770)→D​D¯{\psi(3770)}\!\rightarrow D{{\kern 1.79993pt\overline{\kern-1.79993ptD}}} process.
The C​PC\!P-violating parameters are determined from a joint fit to both datasets,
and an interpretation of the underlying physics parameters yields γ=(71.3±5.0)∘\gamma=(71.3\pm 5.0){{}^{\circ}}.
The statistical uncertainty from LHCb data is improved by 5% with respect to the binned phase-space measurement [17], while the systematic uncertainty from LHCb is at the same level
and the uncertainty introduced by the strong-phase input from BESIII is lower.
The hadronic parameters rBr_{B} and δB\delta_{B} of the B±→D​K±{{B}^{\pm}}\!\rightarrow D{{K}^{\pm}} and B±→D​π±{{B}^{\pm}}\!\rightarrow D{{\pi}^{\pm}} decays are also measured.
Finally, the strong-phase parameters of the novel approach are measured from BESIII data, and will serve as inputs to future γ\gamma measurement.

Acknowledgements

We acknowledge important input from Alex Bondar, which helped to shape the analysis reported here.
The BESIII collaboration thanks the staff of BEPCII (https://cstr.cn/31109.02.BEPC) and the IHEP computing center for their strong support. This work is supported in part by National Key R&D Program of China under Contracts Nos. 2023YFA1606000, 2023YFA1606704, 2022YFA1601901; National Natural Science Foundation of China (NSFC) under Contracts Nos. 11635010, 11935015, 11935016, 11935018, 12025502, 12035009, 12035013, 12061131003, 12192260, 12192261, 12192262, 12192263, 12192264, 12192265, 12221005, 12225509, 12235017, 12342502, 12361141819, 12375087, 12405112; the Chinese Academy of Sciences (CAS) Large-Scale Scientific Facility Program; the Strategic Priority Research Program of Chinese Academy of Sciences under Contract No. XDA0480600; CAS under Contract No. YSBR-101; 100 Talents Program of CAS; The Institute of Nuclear and Particle Physics (INPAC) and Shanghai Key Laboratory for Particle Physics and Cosmology; ERC under Contract No. 758462; German Research Foundation DFG under Contract No. FOR5327; Istituto Nazionale di Fisica Nucleare, Italy; Knut and Alice Wallenberg Foundation under Contracts Nos. 2021.0174, 2021.0299, 2023.0315; Ministry of Development of Turkey under Contract No. DPT2006K-120470; National Research Foundation of Korea under Contract No. NRF-2022R1A2C1092335; National Science and Technology fund of Mongolia; Polish National Science Centre under Contract No. 2024/53/B/ST2/00975; STFC (United Kingdom); Swedish Research Council under Contract No. 2019.04595; U. S. Department of Energy under Contract No. DE-FG02-05ER41374.

The LHCb collaboration expresses gratitude to our colleagues in the CERN
accelerator departments for the excellent performance of the LHC. We
thank the technical and administrative staff at the LHCb
institutes.
We acknowledge support from CERN and from the national agencies:
ARC (Australia);
CAPES, CNPq, FAPERJ and FINEP (Brazil);
MOST and NSFC (China);
CNRS/IN2P3 and CEA (France); BMFTR, DFG and MPG (Germany);
INFN (Italy);
NWO (Netherlands);
MNiSW and NCN (Poland);
MEC/IFA (Romania);
MICIU and AEI (Spain);
SNSF and SER (Switzerland);
NASU (Ukraine);
STFC (United Kingdom);
DOE NP and NSF (USA).
We acknowledge the computing resources that are provided by ARDC (Australia),
CBPF (Brazil),
CERN,
IHEP and LZU (China),
IN2P3 (France),
KIT and DESY (Germany),
INFN (Italy),
SURF (Netherlands),
Polish WLCG (Poland),
IFIN-HH (Romania),
PIC (Spain), CSCS (Switzerland),
GridPP (United Kingdom),
and NSF (USA). We are indebted to the communities behind the multiple open-source
software packages on which we depend.
Individual groups or members have received support from
RTP (Australia), Key Research Program of Frontier Sciences of CAS, CAS PIFI, CAS CCEPP (China);
Minciencias (Colombia);
EPLANET, Marie Skłodowska-Curie Actions, ERC and NextGenerationEU (European Union);
A*MIDEX, ANR, IPhU and Labex P2IO, and Région Auvergne-Rhône-Alpes (France);
Alexander-von-Humboldt Foundation (Germany);
ICSC (Italy);
Severo Ochoa and María de Maeztu Units of Excellence, GVA, XuntaGal, GENCAT, InTalent-Inditex and Prog. Atracción Talento CM (Spain);
the Leverhulme Trust, the Royal Society and UKRI (United Kingdom).

Appendices

Appendix A 𝑫\boldsymbol{D}-candidate mass fit at BESIII

Figures 14–16 show the one-dimensional mass fits for the D→KS/L0​K+​K−D\!\rightarrow K^{0}_{\mathrm{S/L}}{{K}^{+}}{{K}^{-}} channel in BESIII, from which the signal yields and DT efficiencies are extracted. These results are summarized in Table 10, while those for the D→KS/L0​π+​π−D\!\rightarrow K^{0}_{\mathrm{S/L}}{{\pi}^{+}}{{\pi}^{-}} channel can be found in Ref. [26].

Figure 14: One-dimensional mass-fit results for the MBCM_{\rm BC} distributions of D→KS0​K+​K−D\!\rightarrow{{K}^{0}_{\mathrm{S}}}{{K}^{+}}{{K}^{-}} decays with the indicated fully reconstructed tags.

Figure 15: One-dimensional mass-fit results for the Mmiss2M_{\rm miss}^{2} distributions of D→KS0​K+​K−D\!\rightarrow{{K}^{0}_{\mathrm{S}}}{{K}^{+}}{{K}^{-}} decays with partially reconstructed tags and KS0​K±​[K∓]{{K}^{0}_{\mathrm{S}}}{{K}^{\pm}}[K^{\mp}] with fully reconstructed tags. The decay-chain labels indicate the tag decays.

Figure 16: One-dimensional mass-fit results for the Mmiss2M_{\rm miss}^{2} distributions, featuring D→[KL0]​K+​K−D\!\rightarrow[{{K}^{0}_{\mathrm{L}}}]{{K}^{+}}{{K}^{-}} decays against the indicated fully reconstructed tag modes.

Table 10: Signal yields and DT efficiencies of the D→KS/L0​K+​K−D\!\rightarrow K^{0}_{\mathrm{S/L}}{{K}^{+}}{{K}^{-}} channel in BESIII.

Tag mode
D→KS0​K+​K−D\!\rightarrow{{K}^{0}_{\mathrm{S}}}{{K}^{+}}{{K}^{-}}
D→KL0​K+​K−D\!\rightarrow{{K}^{0}_{\mathrm{L}}}{{K}^{+}}{{K}^{-}}

Signal yield
ϵ\epsilon (%)
Signal yield
ϵ\epsilon (%)

K​πK\pi
963±31\phantom{0}963\pm 31
13.22±0.0713.22\pm 0.07
1594±421594\pm 42
15.15±0.0715.15\pm 0.07

K​π​π0K\pi{{\pi}^{0}}
1946±451946\pm 45
7.43±0.05\phantom{0}7.43\pm 0.05
3284±643284\pm 64
8.69±0.06\phantom{0}8.69\pm 0.06

K​π​π​πK\pi\pi\pi
1166±351166\pm 35
7.74±0.05\phantom{0}7.74\pm 0.05
1949±481949\pm 48
8.61±0.06\phantom{0}8.61\pm 0.06

K​e​νeKe{{\nu}_{e}}
684±26\phantom{0}684\pm 26
10.46±0.0410.46\pm 0.04
—
—

π​π​π0\pi\pi{{\pi}^{0}}
160±13\phantom{0}160\pm 13
6.92±0.05\phantom{0}6.92\pm 0.05
381±27\phantom{0}381\pm 27
9.99±0.05\phantom{0}9.99\pm 0.05

π​π\pi\pi
29±5\phantom{00}29\pm\phantom{0}5
11.78±0.0711.78\pm 0.07
71±9\phantom{00}71\pm\phantom{0}9
16.61±0.0716.61\pm 0.07

K​KKK
86±9\phantom{00}86\pm\phantom{0}9
11.51±0.0611.51\pm 0.06
222±15\phantom{0}222\pm 15
16.28±0.0716.28\pm 0.07

KS0​π0​π0{{K}^{0}_{\mathrm{S}}}{{\pi}^{0}}{{\pi}^{0}}
26±5\phantom{00}26\pm\phantom{0}5
2.82±0.03\phantom{0}2.82\pm 0.03
79±10\phantom{00}79\pm 10
4.33±0.04\phantom{0}4.33\pm 0.04

KL0​π0{{K}^{0}_{\mathrm{L}}}{{\pi}^{0}}
94±11\phantom{00}94\pm 11
5.91±0.05\phantom{0}5.91\pm 0.05
—
—

KS0​π0{{K}^{0}_{\mathrm{S}}}{{\pi}^{0}}
128±11\phantom{0}128\pm 11
8.10±0.05\phantom{0}8.10\pm 0.05
147±13\phantom{0}147\pm 13
7.04±0.06\phantom{0}7.04\pm 0.06

KS0η(→γγ){{K}^{0}_{\mathrm{S}}}\eta(\rightarrow\gamma\gamma)
17±4\phantom{00}17\pm\phantom{0}4
6.66±0.05\phantom{0}6.66\pm 0.05
20±5\phantom{00}20\pm\phantom{0}5
5.98±0.05\phantom{0}5.98\pm 0.05

KS0η′(→ππη){{K}^{0}_{\mathrm{S}}}\eta^{\prime}(\rightarrow\pi\pi\eta)
6±2\phantom{000}6\pm\phantom{0}2
2.71±0.03\phantom{0}2.71\pm 0.03
9±3\phantom{000}9\pm\phantom{0}3
2.59±0.03\phantom{0}2.59\pm 0.03

KS0η(→πππ0){{K}^{0}_{\mathrm{S}}}\eta(\rightarrow\pi\pi{{\pi}^{0}})
5±2\phantom{000}5\pm\phantom{0}2
3.67±0.04\phantom{0}3.67\pm 0.04
7±3\phantom{000}7\pm\phantom{0}3
3.57±0.04\phantom{0}3.57\pm 0.04

KS0η′(→ππγ){{K}^{0}_{\mathrm{S}}}\eta^{\prime}(\rightarrow\pi\pi\gamma)
15±4\phantom{00}15\pm\phantom{0}4
4.00±0.04\phantom{0}4.00\pm 0.04
19±5\phantom{00}19\pm\phantom{0}5
3.92±0.04\phantom{0}3.92\pm 0.04

KS0ω(→πππ0){{K}^{0}_{\mathrm{S}}}\omega(\rightarrow\pi\pi{{\pi}^{0}})
62±8\phantom{00}62\pm\phantom{0}8
3.35±0.03\phantom{0}3.35\pm 0.03
55±8\phantom{00}55\pm\phantom{0}8
3.30±0.04\phantom{0}3.30\pm 0.04

KL0​π0​π0{{K}^{0}_{\mathrm{L}}}{{\pi}^{0}}{{\pi}^{0}}
49±7\phantom{00}49\pm\phantom{0}7
3.65±0.03\phantom{0}3.65\pm 0.03
—
—

KS0​K+​K−{{K}^{0}_{\mathrm{S}}}{{K}^{+}}{{K}^{-}}
35±7\phantom{00}35\pm\phantom{0}7
12.72±0.0312.72\pm 0.03
40±8\phantom{00}40\pm\phantom{0}8
4.86±0.02\phantom{0}4.86\pm 0.02

KS0​π+​π−{{K}^{0}_{\mathrm{S}}}{{\pi}^{+}}{{\pi}^{-}}
564±27\phantom{0}564\pm 27
16.62±0.0316.62\pm 0.03
512±24\phantom{0}512\pm 24
8.53±0.02\phantom{0}8.53\pm 0.02

KL0​π+​π−{{K}^{0}_{\mathrm{L}}}{{\pi}^{+}}{{\pi}^{-}}
562±29\phantom{0}562\pm 29
9.61±0.03\phantom{0}9.61\pm 0.03
—
—

Appendix B Alternative fits to strong-phase parameters

Table 11 lists the strong-phase parameters from fits to BESIII and LHCb data or only to BESIII data.
Data from LHCb are found to slightly improve the sensitivity to SnS_{n} parameters.

Table 11: Strong-phase parameters of the D→KS0​π+​π−D\!\rightarrow{{K}^{0}_{\mathrm{S}}}{{\pi}^{+}}{{\pi}^{-}} decay with different setups under Mπ=2M_{\pi}=2 and MK=1M_{K}=1.
The uncertainties are statistical only.

Parameter
Weight
BESIII and LHCb
BESIII only

C0C_{0}
woptw^{\rm opt}
0.0701±0.00380.0701\pm 0.0038
0.0691±0.00380.0691\pm 0.0038

C1C_{1}
wopt​sin⁡ϕw^{\rm opt}\sin{\phi}
0.0102±0.00230.0102\pm 0.0023
0.0102±0.00230.0102\pm 0.0023

C2C_{2}
wopt​cos⁡ϕw^{\rm opt}\cos{\phi}
0.1729±0.00280.1729\pm 0.0028
0.1726±0.00280.1726\pm 0.0028

C3C_{3}
wopt​sin⁡2​ϕw^{\rm opt}\sin{2\phi}
0.0067±0.00210.0067\pm 0.0021
0.0064±0.00220.0064\pm 0.0022

C4C_{4}
wopt​cos⁡2​ϕw^{\rm opt}\cos{2\phi}
0.0437±0.00310.0437\pm 0.0031
0.0439±0.00320.0439\pm 0.0032

S0S_{0}
woptw^{\rm opt}
−0.0213±0.0085-0.0213\pm 0.0085
−0.0185±0.0088-0.0185\pm 0.0088

S1S_{1}
wopt​sin⁡ϕw^{\rm opt}\sin{\phi}
0.0933±0.00520.0933\pm 0.0052
0.0939±0.00520.0939\pm 0.0052

S2S_{2}
wopt​cos⁡ϕw^{\rm opt}\cos{\phi}
0.0073±0.00670.0073\pm 0.0067
0.0071±0.00690.0071\pm 0.0069

S3S_{3}
wopt​sin⁡2​ϕw^{\rm opt}\sin{2\phi}
0.0198±0.00640.0198\pm 0.0064
0.0214±0.00670.0214\pm 0.0067

S4S_{4}
wopt​cos⁡2​ϕw^{\rm opt}\cos{2\phi}
0.0129±0.00690.0129\pm 0.0069
0.0116±0.00720.0116\pm 0.0072

Appendix C Correlation matrices

The statistical uncertainties and correlation matrices of the C​PC\!P observables from LHCb are given in Table 12, while those from BESIII are given in Table 13. The corresponding information pertaining to systematic uncertainties are then given in Tables 14–15.

Table 12: Statistical uncertainties and correlation matrix of the C​PC\!P observables from LHCb.

Uncertainty (10−210^{-2})

x−D​Kx_{-}^{DK}
y−D​Ky_{-}^{DK}
x+D​Kx_{+}^{DK}
y+D​Ky_{+}^{DK}
xξD​πx_{\xi}^{D\pi}
yξD​πy_{\xi}^{D\pi}

σ\sigma
0.960.96
1.341.34
1.011.01
1.361.36
2.062.06
2.302.30

Correlation matrix

x−D​Kx_{-}^{DK}
y−D​Ky_{-}^{DK}
x+D​Kx_{+}^{DK}
y+D​Ky_{+}^{DK}
xξD​πx_{\xi}^{D\pi}
yξD​πy_{\xi}^{D\pi}

x−D​Kx_{-}^{DK}
1.0001.000
−0.083-0.083
−0.028-0.028
0.0210.021
0.1110.111
−0.124-0.124

y−D​Ky_{-}^{DK}

1.0001.000
−0.018-0.018
−0.041-0.041
0.1000.100
0.1920.192

x+D​Kx_{+}^{DK}

1.0001.000
0.2270.227
−0.240-0.240
−0.046-0.046

y+D​Ky_{+}^{DK}

1.0001.000
−0.115-0.115
−0.231-0.231

xξD​πx_{\xi}^{D\pi}

1.0001.000
0.3020.302

yξD​πy_{\xi}^{D\pi}

1.0001.000

Table 13: Statistical uncertainties and correlation matrix of the C​PC\!P observables from BESIII.

Uncertainty (10−210^{-2})

x−D​Kx_{-}^{DK}
y−D​Ky_{-}^{DK}
x+D​Kx_{+}^{DK}
y+D​Ky_{+}^{DK}
xξD​πx_{\xi}^{D\pi}
yξD​πy_{\xi}^{D\pi}

σ\sigma
0.200.20
0.370.37
0.150.15
0.340.34
0.380.38
0.570.57

Correlation matrix

x−D​Kx_{-}^{DK}
y−D​Ky_{-}^{DK}
x+D​Kx_{+}^{DK}
y+D​Ky_{+}^{DK}
xξD​πx_{\xi}^{D\pi}
yξD​πy_{\xi}^{D\pi}

x−D​Kx_{-}^{DK}
1.0001.000
0.1470.147
−0.572-0.572
0.5300.530
−0.258-0.258
−0.155-0.155

y−D​Ky_{-}^{DK}

1.0001.000
−0.022-0.022
−0.234-0.234
−0.213-0.213
−0.043-0.043

x+D​Kx_{+}^{DK}

1.0001.000
−0.013-0.013
−0.096-0.096
−0.186-0.186

y+D​Ky_{+}^{DK}

1.0001.000
−0.307-0.307
−0.395-0.395

xξD​πx_{\xi}^{D\pi}

1.0001.000
0.9540.954

yξD​πy_{\xi}^{D\pi}

1.0001.000

Table 14: Systematic uncertainties and correlation matrix of the C​PC\!P observables from LHCb.

Uncertainty (10−210^{-2})

x−D​Kx_{-}^{DK}
y−D​Ky_{-}^{DK}
x+D​Kx_{+}^{DK}
y+D​Ky_{+}^{DK}
xξD​πx_{\xi}^{D\pi}
yξD​πy_{\xi}^{D\pi}

σ\sigma
0.180.18
0.230.23
0.240.24
0.310.31
0.300.30
0.200.20

Correlation matrix

x−D​Kx_{-}^{DK}
y−D​Ky_{-}^{DK}
x+D​Kx_{+}^{DK}
y+D​Ky_{+}^{DK}
xξD​πx_{\xi}^{D\pi}
yξD​πy_{\xi}^{D\pi}

x−D​Kx_{-}^{DK}
1.0001.000
0.3220.322
0.2000.200
0.1670.167
0.0000.000
0.2770.277

y−D​Ky_{-}^{DK}

1.0001.000
−0.361-0.361
−0.358-0.358
−0.320-0.320
0.2830.283

x+D​Kx_{+}^{DK}

1.0001.000
0.4230.423
0.1630.163
−0.415-0.415

y+D​Ky_{+}^{DK}

1.0001.000
0.2230.223
0.0080.008

xξD​πx_{\xi}^{D\pi}

1.0001.000
0.2980.298

yξD​πy_{\xi}^{D\pi}

1.0001.000

Table 15: Systematic uncertainties and correlation matrix of the C​PC\!P observables from BESIII.

Uncertainty (10−210^{-2})

x−D​Kx_{-}^{DK}
y−D​Ky_{-}^{DK}
x+D​Kx_{+}^{DK}
y+D​Ky_{+}^{DK}
xξD​πx_{\xi}^{D\pi}
yξD​πy_{\xi}^{D\pi}

σ\sigma
0.070.07
0.090.09
0.080.08
0.060.06
0.010.01
0.020.02

Correlation matrix

x−D​Kx_{-}^{DK}
y−D​Ky_{-}^{DK}
x+D​Kx_{+}^{DK}
y+D​Ky_{+}^{DK}
xξD​πx_{\xi}^{D\pi}
yξD​πy_{\xi}^{D\pi}

x−D​Kx_{-}^{DK}
1.0001.000
−0.008-0.008
−0.701-0.701
−0.288-0.288
0.0310.031
−0.563-0.563

y−D​Ky_{-}^{DK}

1.0001.000
0.1910.191
−0.520-0.520
0.5280.528
−0.235-0.235

x+D​Kx_{+}^{DK}

1.0001.000
0.3210.321
0.2980.298
0.4270.427

y+D​Ky_{+}^{DK}

1.0001.000
0.0580.058
0.7210.721

xξD​πx_{\xi}^{D\pi}

1.0001.000
0.0730.073

yξD​πy_{\xi}^{D\pi}

1.0001.000

References

      
[1]

N. Cabibbo, Unitary symmetry and
leptonic decays,
Phys. Rev. Lett. 10 (1963) 531

      
[2]

M. Kobayashi and T. Maskawa,
C​PC\!P-violation in the
renormalizable theory of weak interaction,
Prog. Theor. Phys. 49 (1973) 652

      
[3]

CKMfitter group, J. Charles et al.,
C​PC\!P violation and the CKM matrix:
Assessing the impact of the asymmetric BB factories,
Eur. Phys. J. C41 (2005) 1,
arXiv:hep-ph/0406184, updated results and plots available at
http://ckmfitter.in2p3.fr/

      
[4]

CKMfitter group, J. Charles et al.,
Current status of the standard
model CKM fit and constraints on Δ​F=2\Delta F=2 new physics,
Phys. Rev. D91 (2015) 073007,
arXiv:1501.05013, updated results and plots available at
http://ckmfitter.in2p3.fr/

      
[5]

UTfit collaboration, M. Bona et al.,
The unitarity triangle fit in the
standard model and hadronic parameters from lattice QCD: A reappraisal after
the measurements of Δ​ms\Delta m_{s} and B​R​(B→τ​ντ)BR(B\rightarrow\tau\nu_{\tau}),
JHEP 10
(2006) 081, arXiv:hep-ph/0606167, updated results and plots available at
http://www.utfit.org/

      
[6]

J. Brod and J. Zupan, The ultimate
theoretical error on γ\gamma from B→D​KB\rightarrow DK decays,
JHEP 01 (2014)
051, arXiv:1308.5663

      
[7]

R. Fleischer, New strategies to
extract β\beta and γ\gamma from Bd→π+​π−B_{d}\rightarrow\pi^{+}\pi^{-} and Bs→K+​K−B_{s}\rightarrow K^{+}K^{-}, Phys. Lett. B459 (1999) 306,
arXiv:hep-ph/9903456

      
[8]

R. Fleischer, Bs,d→π​π,π​K,K​KB_{s,d}\rightarrow\pi\pi,\pi K,KK: Status and prospects,
Eur. Phys. J. C52 (2007) 267,
arXiv:0705.1121

      
[9]

R. Fleischer and R. Knegjens, In
pursuit of new physics with Bs0→K+​K−B^{0}_{s}\rightarrow K^{+}K^{-},
Eur. Phys. J. C71 (2011) 1532,
arXiv:1011.1096

      
[10]

R. Fleischer, R. Jaarsma, and K. K. Vos,
Zooming into CP violation in
B(s)→ hh decays,
JHEP 02 (2023)
081, arXiv:2211.08346

      
[11]

N. Rey-Le Lorier and D. London,
Measuring γ\gamma with B→K​π​πB\rightarrow K\pi\pi and B→K​K​K¯B\rightarrow KK{\kern 1.79993pt\overline{\kern-1.79993ptK}} decays,
Phys. Rev. D85 (2012) 016010,
arXiv:1109.0881

      
[12]

B. Bhattacharya et al.,
Charmless B→P​P​PB\rightarrow PPP decays: The
fully-symmetric final state,
Phys. Rev. D89 (2014) 074043,
arXiv:1402.2909

      
[13]

B. Bhattacharya et al.,
Charmless B→P​P​PB\rightarrow PPP decays: The
fully antisymmetric final state,
Phys. Rev. D109 (2024) 013001,
arXiv:2308.16240

      
[14]

LHCb collaboration, Simultaneous
determination of the CKM angle γ\gamma and parameters related to mixing and
C​PC\!P violation in the charm sector, 
LHCb-CONF-2025-003, 2025

      
[15]

BaBar collaboration, J. P. Lees et al.,
Observation of direct CP
violation in the measurement of the Cabibbo–Kobayashi–Maskawa angle
γ\gamma with B±→D(∗)​K(∗)±B^{\pm}\rightarrow D^{(*)}K^{(*)\pm} decays,
Phys. Rev. D87 (2013) 052015,
arXiv:1301.1029

      
[16]

Belle and Belle-II collaborations, I. Adachi et al.,
Determination of the CKM angle
ϕ\phi3 from a combination of Belle and Belle II results,
JHEP 10 (2024)
143, arXiv:2404.12817

      
[17]

LHCb collaboration, R. Aaij et al.,
Measurement of the CKM angle
γ\gamma in B±→D​K±B^{\pm}\rightarrow DK^{\pm} and B±→D​π±B^{\pm}\rightarrow D\pi^{\pm}
decays with D→KS0​h+​h−D\rightarrow{{K}^{0}_{\mathrm{S}}}h^{+}h^{-},
JHEP 02 (2021)
0169, arXiv:2010.08483

      
[18]

A. Giri, Y. Grossman, A. Soffer, and J. Zupan,
Determining γ\gamma
using
B±→D​K±{B}^{\pm{}}\rightarrow{DK}^{\pm{}}
with multibody D decays,
Phys. Rev. D68 (2003) 054018,
arXiv:hep-ph/0303187

      
[19]

A. Bondar, Proceedings of BINP
special analysis meeting on Dalitz analysis,  unpublished, 2002

      
[20]

A. Bondar and A. Poluektov, The use
of quantum-correlated D0D^{0} decays for ϕ3\phi_{3} measurement,
Eur. Phys. J. C55 (2008) 51,
arXiv:0801.0840

      
[21]

CLEO collaboration, J. Libby et al.,
Model-independent determination
of the strong-phase difference between D0{{D}^{0}} and D¯→0KS,L0h+h−(h=π,K){{\kern 1.79993pt\overline{\kern-1.79993ptD}}{}^{0}}\rightarrow K^{0}_{\rm S,L}h^{+}h^{-}(h=\pi,K) and its impact on the measurement of the CKM angle
γ/ϕ3\gamma/\phi_{3},
Phys. Rev. D82 (2010) 112006,
arXiv:1010.2817

      
[22]

A. Poluektov, Unbinned
model-independent measurements with coherent admixtures of multibody neutral
DD meson decays,
Eur. Phys. J. C78 (2018) 121,
arXiv:1712.08326

      
[23]

BESIII collaboration, M. Ablikim et al.,
Determination of strong-phase
parameters in D→KS,L0​π+​π−D\rightarrow K^{0}_{\rm S,L}\pi^{+}\pi^{-},
Phys. Rev. Lett. 124 (2020) 241802,
arXiv:2002.12791

      
[24]

BESIII collaboration, M. Ablikim et al.,
Model-independent determination
of the relative strong-phase difference between D0{{D}^{0}} and D¯0→K0S,Lπ+π−{{\kern 1.79993pt\overline{\kern-1.79993ptD}}{}^{0}}\rightarrow K^{0}_{\rm S,L}\pi^{+}\pi^{-} and its impact on the measurement of the CKM angle
γ/ϕ3\gamma/\phi_{3},
Phys. Rev. D101 (2020) 112002,
arXiv:2003.00091

      
[25]

BESIII collaboration, M. Ablikim et al.,
Improved model-independent
determination of the strong-phase difference between D0{{D}^{0}} and D¯0→K0S,LK+K−{{\kern 1.79993pt\overline{\kern-1.79993ptD}}{}^{0}}\rightarrow K^{0}_{\mathrm{S,L}}K^{+}K^{-} decays,
Phys. Rev. D102 (2020) 052008,
arXiv:2007.07959

      
[26]

BESIII collaboration, M. Ablikim et al.,
Updated model-independent
measurement of the strong-phase differences between D0{{D}^{0}} and D¯0→K0S/Lπ+π−{{\kern 1.79993pt\overline{\kern-1.79993ptD}}{}^{0}}\rightarrow K^{0}_{\rm S/L}\pi^{+}\pi^{-} decays,
JHEP 06 (2025)
086, arXiv:2503.22126

      
[27]

BESIII and LHCb collaborations, M. Ablikim et al.,
Precise measurement of the CKM
angle γ\gamma with a novel approach,  LHCb-PAPER-2025-064,
CERN-EP-2026-068, in preparation, to be submitted to Phys. Rev.
Lett.

      
[28]

R. H. Dalitz, On the analysis of
τ\tau-meson data and the nature of the τ\tau-meson,
Phil. Mag. Ser. 7
44 (1953) 1068

      
[29]

E. Fabri, A study of τ\tau-meson
decay, Nuovo Cim. 11 (1954) 479

      
[30]

A. Bondar, A. Poluektov, and V. Vorobiev,
Charm mixing in a
model-independent analysis of correlated D0D¯0{{D}^{0}}{{\kern 1.79993pt\overline{\kern-1.79993ptD}}{}^{0}} decays,
Phys. Rev. D82 (2010) 034033,
arXiv:1004.2350

      
[31]

BaBar and Belle collaborations, I. Adachi et al.,
First Evidence for c​o​s​2​β>0cos2\beta>0
and resolution of the Cabibbo-Kobayashi-Maskawa quark-mixing unitarity
triangle ambiguity,
Phys. Rev. Lett. 121 (2018) 261801,
arXiv:1804.06152

      
[32]

BaBar and Belle collaborations, I. Adachi et al.,
Measurement of cos⁡2​β\cos{2\beta} in
B0→D(∗)​h0B^{0}\rightarrow D^{(*)}h^{0} with D→KS0​π+​π−D\rightarrow K_{\rm S}^{0}\pi^{+}\pi^{-} decays
by a combined time-dependent Dalitz plot analysis of BaBar and Belle data,
Phys. Rev. D98 (2018) 112012,
arXiv:1804.06153

      
[33]

BaBar collaboration, P. del Amo Sanchez et al.,
Measurement of D0−D¯0{{D}^{0}}-{{\kern 1.79993pt\overline{\kern-1.79993ptD}}{}^{0}} mixing
parameters using D0→KS0​π+​π−{{D}^{0}}\rightarrow{{K}^{0}_{\mathrm{S}}}\pi^{+}\pi^{-} and D0→KS0​K+​K−{{D}^{0}}\rightarrow{{K}^{0}_{\mathrm{S}}}K^{+}K^{-}
decays, Phys. Rev. Lett.  105 (2010) 081803,
arXiv:1004.5053

      
[34]

LHCb collaboration, HEPData for
this analysis,  2026.

doi: XXXX/hepdata.XXX

      
[35]

R. M. Baltrusaitis et al.,
Direct measurements of
charmed-DD-meson hadronic branching fractions,
Phys. Rev. Lett. 56 (1986) 2140

      
[36]

BESIII collaboration, M. Ablikim et al.,
Measurements of the CP-even
fractions of D0→π+​π−​π0{{D}^{0}}\rightarrow\pi^{+}\pi^{-}\pi^{0} and D0→K+​K−​π0{{D}^{0}}\rightarrow K^{+}K^{-}\pi^{0}
at BESIII, Phys. Rev.  D111 (2025) 012007,
arXiv:2409.07197

      
[37]

BESIII collaboration, M. Ablikim et al.,
Design and construction of the
BESIII detector,
Nucl. Instrum. Meth. A614 (2010) 345,
arXiv:0911.4960

      
[38]

C. Yu et al., BEPCII
performance and beam dynamics studies on luminosity in the proceeding of the
7th International Particle Accelerator Conference,  8-13 May.
2016

      
[39]

BESIII collaboration, M. Ablikim et al.,
Future physics programme of
BESIII, Chin. Phys.  C44 (2020) 040001,
arXiv:1912.05983

      
[40]

J. Lu et al., Online
monitoring of the center-of-mass energy from real data at BESIII,
Radiat. Detect. Technol. Methods 4 (2020) 337

      
[41]

J. Zhang et al., Suppression
of top-up injection backgrounds with offline event filter in the BESIII
experiment, Radiat. Detect. Technol. Methods 6 (2022) 289

      
[42]

Y.-X. Guo et al., The study
of time calibration for upgraded end cap TOF of BESIII,
Radiat. Detect. Technol. Methods 1 (2017) 15

      
[43]

Geant4 collaboration, S. Agostinelli et al.,
Geant4: A simulation toolkit,
Nucl. Instrum. Meth.  A506 (2003) 250

      
[44]

S. Jadach, B. F. L. Ward, and Z. Was,
Coherent exclusive exponentiation
for precision Monte Carlo calculations,
Phys. Rev. D63 (2001) 113009,
arXiv:hep-ph/0006359

      
[45]

S. Jadach, B. F. L. Ward, and Z. Was,
The precision Monte Carlo event
generator KK for two-fermion final states in e+​e−e^{+}e^{-} collisions,
Comput. Phys. Commun.  130 (2000) 260,
arXiv:hep-ph/9912214

      
[46]

D. J. Lange, The EvtGen particle
decay simulation package,
Nucl. Instrum. Meth.  A462 (2001) 152

      
[47]

R.-G. Ping, Event generators at
BESIII, Chin. Phys. C32 (2008) 599

      
[48]

Particle Data Group, S. Navas et al.,
Review
of particle physics,
Phys. Rev. D110 (2024) 030001

      
[49]

J. C. Chen et al., Event
generator for J/ψJ/\psi and
ψ⁡(2​S)\psi(2S) decay,
Phys. Rev. D62 (2000) 034003

      
[50]

R.-L. Yang, R.-G. Ping, and H. Chen,
Tuning and validation of the
Lundcharm model with J/ψJ/\psi decays,
Chin. Phys. Lett. 31 (2014) 061301

      
[51]

E. Barberio, B. van Eijk, and Z. Wa̧s,
Photos - a universal Monte Carlo
for QED radiative corrections in decays,
Comput. Phys. Commun.  66 (1991) 115

      
[52]

BESIII collaboration, M. Ablikim et al.,
Determination of U-spin breaking
parameters with an amplitude analysis of the decay D0→KL0​π+​π−D^{0}\rightarrow K_{\rm L}^{0}\pi^{+}\pi^{-},
arXiv:2212.09048

      
[53]

LHCb collaboration, A. A. Alves Jr. et al.,
The LHCb detector at the LHC,
JINST 3
(2008) S08005

      
[54]

LHCb collaboration, R. Aaij et al.,
LHCb detector performance,
Int. J. Mod. Phys. A30 (2015) 1530022,
arXiv:1412.6352

      
[55]

N. Grieser et al., The LHCb
stripping project: Sustainable legacy data processing for high-energy
physics, Comput. Softw. Big. Sci.  9 (2025) 21,
arXiv:2509.05294

      
[56]

T. Sjöstrand, S. Mrenna, and P. Skands,
A brief introduction to PYTHIA
8.1, Comput. Phys. Commun.  178 (2008) 852,
arXiv:0710.3820

      
[57]

T. Sjöstrand, S. Mrenna, and P. Skands,
PYTHIA 6.4 physics and manual,
JHEP 05
(2006) 026, arXiv:hep-ph/0603175

      
[58]

I. Belyaev et al., Handling
of the generation of primary events in Gauss, the LHCb simulation
framework, J. Phys. Conf. Ser.  331 (2011) 032047

      
[59]

N. Davidson, T. Przedzinski, and Z. Was,
PHOTOS interface in C++:
Technical and physics documentation,
Comput. Phys. Commun.  199 (2016) 86,
arXiv:1011.0937

      
[60]

Geant4 collaboration, J. Allison et al.,
Geant4 developments and
applications, IEEE
Trans. Nucl. Sci.  53 (2006) 270

      
[61]

M. Clemencic et al., The
LHCb simulation application, Gauss: Design, evolution and experience,
J. Phys. Conf. Ser.  331 (2011) 032023

      
[62]

D. Müller, M. Clemencic, G. Corti, and M. Gersabeck,
ReDecay: A novel approach to
speed up the simulation at LHCb,
Eur. Phys. J. C78 (2018) 1009,
arXiv:1810.10362

      
[63]

G. A. Cowan, D. C. Craik, and M. D. Needham,
RapidSim: an application for the
fast simulation of heavy-quark hadron decays,
Comput. Phys. Commun. 214 (2017) 239,
arXiv:1612.07489

      
[64]

T. Skwarnicki, A study of the radiative cascade transitions between the
Upsilon-prime and Upsilon resonances, PhD thesis, Institute of Nuclear
Physics, Krakow, 1986,
DESY-F31-86-02

      
[65]

M. Pivk and F. R. Le Diberder,
sPlot: A statistical tool to
unfold data distributions,
Nucl. Instrum. Meth. A555 (2005) 356,
arXiv:physics/0402083

      
[66]

BESIII collaboration, M. Ablikim et al.,
Measurement of the D→K−​π+​π+​π−D\rightarrow K^{-}\pi^{+}\pi^{+}\pi^{-} and D→K−​π+​π0D\rightarrow K^{-}\pi^{+}\pi^{0} coherence factors
and average strong-phase differences in quantum-correlated D​D¯D\overline{D}
decays, JHEP 05
(2021) 164

      
[67]

BESIII collaboration, M. Ablikim et al.,
Measurement of integrated
luminosity of data collected at 3.773 GeV by BESIII from 2021 to 2024,
Chin. Phys. C48 (2024) 123001,
arXiv:2406.05827

      
[68]

Belle collaboration, A. Poluektov et al.,
Evidence for direct CP violation
in the decay B±→D(∗)​K±B^{\pm}\rightarrow D^{(*)}K^{\pm}, D→KS0​π+​π−D\rightarrow{{K}^{0}_{\mathrm{S}}}\pi^{+}\pi^{-} and
measurement of the CKM phase ϕ3\phi_{3},
Phys. Rev. D81 (2010) 112002,
arXiv:1003.3360

      
[69]

See Supplemental Material at [link] for complete results files including
optimal weights and correlation matrices of the total uncertainties.

      
[70]

LHCb collaboration, Update of the
LHCb combination of the CKM angle γ\gamma using B→D​KB\rightarrow DK decays, 
LHCb-CONF-2018-002, 2018

      
[71]

M. Bjørn and S. Malde, CP
violation and material interaction of neutral kaons in measurements of the
CKM angle γ\gamma using B±→D​K±B^{\pm}\rightarrow DK^{\pm} decays where D→KS0​π+​π−D\rightarrow K_{\text{\rm S}}^{0}\pi^{+}\pi^{-}, JHEP
07 (2019) 106,
arXiv:1904.01129

      
[72]

LHCb collaboration, R. Aaij et al.,
Measurement of the CKM angle
γ\gamma from a combination of LHCb results,
JHEP 12 (2016)
087, arXiv:1611.03076

      
[73]

LHCb collaboration, R. Aaij et al.,
Simultaneous determination of
CKM angle γ\gamma and charm mixing parameters,
JHEP 12 (2021)
141, arXiv:2110.02350

      
[74]

Heavy Flavor Averaging Group, S. Banerjee et al.,
Averages of bb-hadron,
cc-hadron, and τ\tau-lepton properties as of 2023,
Phys. Rev.  D113
(2026) 012008, arXiv:2411.18639, updated results and plots available at
https://hflav.web.cern.ch

    

BESIII collaboration

M. Ablikim1 ,
M. N. Achasov4,d ,
P. Adlarson83 ,
X. C. Ai89 ,
C. S. Akondi31A,31B ,
R. Aliberti39 ,
A. Amoroso82A,82C ,
Q. An79,65,†,
Y. H. An89 ,
Y. Bai63 ,
O. Bakina40 ,
H. R. Bao71 ,
X. L. Bao50 ,
M. Barbagiovanni82C ,
V. Batozskaya1,49 ,
K. Begzsuren35,
N. Berger39 ,
M. Berlowski49 ,
M. B. Bertani30A ,
D. Bettoni31A ,
F. Bianchi82A,82C ,
E. Bianco82A,82C,
A. Bortone82A,82C ,
I. Boyko40 ,
R. A. Briere5 ,
A. Brueggemann76 ,
D. Cabiati82A,82C ,
H. Cai84 ,
M. H. Cai42,l,m ,
X. Cai1,65 ,
A. Calcaterra30A ,
G. F. Cao1,71 ,
N. Cao1,71 ,
S. A. Cetin69A ,
X. Y. Chai51,i ,
J. F. Chang1,65 ,
T. T. Chang48 ,
G. R. Che48 ,
Y. Z. Che1,65,71 ,
C. H. Chen10 ,
Chao Chen1 ,
G. Chen1 ,
H. S. Chen1,71 ,
H. Y. Chen20 ,
M. L. Chen1,65,71 ,
S. J. Chen47 ,
S. M. Chen68 ,
T. Chen1,71 ,
W. Chen50 ,
X. R. Chen34,71 ,
X. T. Chen1,71 ,
X. Y. Chen12,h ,
Y. B. Chen1,65 ,
Y. Q. Chen16 ,
Z. K. Chen66 ,
J. Cheng50 ,
L. N. Cheng48 ,
S. K. Choi11 ,
X. Chu12,h ,
G. Cibinetto31A ,
F. Cossio82C ,
J. Cottee-Meldrum70 ,
H. L. Dai1,65 ,
J. P. Dai87 ,
X. C. Dai68 ,
A. Dbeyssi19,
R. E. de Boer3 ,
D. Dedovich40 ,
C. Q. Deng80 ,
Z. Y. Deng1 ,
A. Denig39 ,
I. Denisenko40 ,
M. Destefanis82A,82C ,
F. De Mori82A,82C ,
E. Di Fiore31A,31B ,
X. X. Ding51,i ,
Y. Ding44 ,
Y. X. Ding32 ,
Yi. Ding38 ,
J. Dong1,65 ,
L. Y. Dong1,71 ,
M. Y. Dong1,65,71 ,
X. Dong84 ,
Z. J. Dong66 ,
M. C. Du1 ,
S. X. Du89 ,
Shaoxu Du12,h ,
X. L. Du12,h ,
Y. Q. Du84 ,
Y. Y. Duan61 ,
Z. H. Duan47 ,
P. Egorov40,b ,
G. F. Fan47 ,
J. J. Fan20 ,
Y. H. Fan50 ,
J. Fang1,65 ,
Jin Fang66 ,
S. S. Fang1,71 ,
W. X. Fang1 ,
Y. Q. Fang1,65,† ,
L. Fava82B,82C ,
F. Feldbauer3 ,
G. Felici30A ,
C. Q. Feng79,65 ,
J. H. Feng16 ,
L. Feng42,l,m ,
Q. X. Feng42,l,m ,
Y. T. Feng79,65 ,
M. Fritsch3 ,
C. D. Fu1 ,
J. L. Fu71 ,
Y. W. Fu1,71 ,
H. Gao71 ,
Xu Gao38 ,
Y. Gao79,65 ,
Y. N. Gao51,i ,
Y. Y. Gao32 ,
Yunong Gao20 ,
Z. Gao48 ,
S. Garbolino82C ,
I. Garzia31A,31B ,
L. Ge63 ,
P. T. Ge20 ,
Z. W. Ge47 ,
C. Geng66 ,
E. M. Gersabeck75 ,
A. Gilman77 ,
K. Goetzen13 ,
J. Gollub3 ,
J. B. Gong1,71 ,
J. D. Gong38 ,
L. Gong44 ,
W. X. Gong1,65 ,
W. Gradl39 ,
S. Gramigna31A,31B ,
M. Greco82A,82C ,
M. D. Gu56 ,
M. H. Gu1,65 ,
C. Y. Guan1,71 ,
A. Q. Guo34 ,
H. Guo55 ,
J. N. Guo12,h ,
L. B. Guo46 ,
M. J. Guo55 ,
R. P. Guo54 ,
X. Guo55 ,
Y. P. Guo12,h ,
Z. Guo79,65 ,
A. Guskov40,b ,
J. Gutierrez29 ,
J. Y. Han79,65 ,
T. T. Han1 ,
X. Han79,65 ,
F. Hanisch3 ,
K. D. Hao79,65 ,
X. Q. Hao20 ,
F. A. Harris72 ,
C. Z. He51,i ,
K. K. He17,47 ,
K. L. He1,71 ,
F. H. Heinsius3 ,
C. H. Heinz39 ,
Y. K. Heng1,65,71 ,
C. Herold67 ,
P. C. Hong38 ,
G. Y. Hou1,71 ,
X. T. Hou1,71 ,
Y. R. Hou71 ,
Z. L. Hou1 ,
H. M. Hu1,71 ,
J. F. Hu62,k ,
Q. P. Hu79,65 ,
S. L. Hu12,h ,
T. Hu1,65,71 ,
Y. Hu1 ,
Y. X. Hu84 ,
Z. M. Hu66 ,
G. S. Huang79,65 ,
K. X. Huang66 ,
L. Q. Huang34,71 ,
P. Huang47 ,
X. T. Huang55 ,
Y. P. Huang1 ,
Y. S. Huang66 ,
T. Hussain81 ,
N. Hüsken39 ,
N. in der Wiesche76 ,
J. Jackson29 ,
Q. Ji1 ,
Q. P. Ji20 ,
W. Ji1,71 ,
X. B. Ji1,71 ,
X. L. Ji1,65 ,
Y. Y. Ji1 ,
L. K. Jia71 ,
X. Q. Jia55 ,
D. Jiang1,71 ,
H. B. Jiang84 ,
S. J. Jiang10 ,
X. S. Jiang1,65,71 ,
Y. Jiang71 ,
J. B. Jiao55 ,
J. K. Jiao38 ,
Z. Jiao25 ,
L. C. L. Jin1 ,
S. Jin47 ,
Y. Jin73 ,
M. Q. Jing56 ,
X. M. Jing71 ,
T. Johansson83 ,
S. Kabana36 ,
X. L. Kang10 ,
X. S. Kang44 ,
B. C. Ke89 ,
V. Khachatryan29 ,
A. Khoukaz76 ,
O. B. Kolcu69A ,
B. Kopf3 ,
L. Kröger76 ,
L. Krümmel3,
Y. Y. Kuang80 ,
M. Kuessner3 ,
X. Kui1,71 ,
N. Kumar28 ,
A. Kupsc49,83 ,
W. Kühn41 ,
Q. Lan80 ,
W. N. Lan20 ,
T. T. Lei79,65 ,
M. Lellmann39 ,
T. Lenz39 ,
C. Li52 ,
C. H. Li46 ,
C. K. Li48 ,
Chunkai Li21 ,
Cong Li48 ,
D. M. Li89 ,
F. Li1,65 ,
G. Li1 ,
H. B. Li1,71 ,
H. J. Li20 ,
H. L. Li89 ,
H. N. Li62,k ,
H. P. Li48 ,
Hui Li48 ,
J. N. Li32 ,
J. S. Li66 ,
J. W. Li55 ,
K. Li1 ,
K. L. Li42,l,m ,
L. J. Li1,71 ,
Lei Li53 ,
M. H. Li48 ,
M. R. Li1,71 ,
M. T. Li55 ,
P. L. Li71 ,
P. R. Li42,l,m ,
Q. M. Li1,71 ,
Q. X. Li55 ,
R. Li18,34 ,
S. Li89 ,
S. X. Li89 ,
S. Y. Li89 ,
Shanshan Li27,j ,
T. Li55 ,
T. Y. Li48 ,
W. D. Li1,71 ,
W. G. Li1,† ,
X. Li1,71 ,
X. H. Li79,65 ,
X. K. Li51,i ,
X. L. Li55 ,
X. Y. Li1,9 ,
X. Z. Li66 ,
Y. Li20 ,
Y. B. Li85 ,
Y. C. Li66 ,
Y. G. Li71 ,
Y. P. Li38 ,
Z. H. Li42 ,
Z. J. Li66 ,
Z. L. Li89 ,
Z. X. Li48 ,
Z. Y. Li87 ,
C. Liang47 ,
H. Liang79,65 ,
Y. F. Liang60 ,
Y. T. Liang34,71 ,
Z. Z. Liang66 ,
G. R. Liao14 ,
L. B. Liao66 ,
M. H. Liao66 ,
Y. P. Liao1,71 ,
J. Libby28 ,
A. Limphirat67 ,
C. C. Lin61 ,
C. X. Lin34 ,
D. X. Lin34,71 ,
T. Lin1 ,
B. J. Liu1 ,
B. X. Liu84 ,
C. Liu38 ,
C. X. Liu1 ,
F. Liu1 ,
F. H. Liu59 ,
Feng Liu6 ,
G. M. Liu62,k ,
H. Liu42,l,m ,
H. B. Liu15 ,
H. M. Liu1,71 ,
Huihui Liu22 ,
J. B. Liu79,65 ,
J. J. Liu21 ,
K. Liu42,l,m ,
K. Y. Liu44 ,
Ke Liu23 ,
Kun Liu80 ,
L. Liu42 ,
L. C. Liu48 ,
Lu Liu48 ,
M. H. Liu38 ,
P. L. Liu55 ,
Q. Liu71 ,
S. B. Liu79,65 ,
T. Liu1 ,
W. M. Liu79,65 ,
W. T. Liu43 ,
X. Liu42,l,m ,
X. K. Liu42,l,m ,
X. L. Liu12,h ,
X. P. Liu12,h ,
X. Y. Liu84 ,
Y. Liu42,l,m ,
Y. B. Liu48 ,
Yi Liu89 ,
Z. A. Liu1,65,71 ,
Z. D. Liu85 ,
Z. L. Liu80 ,
Z. Q. Liu55 ,
Z. X. Liu1 ,
Z. Y. Liu42 ,
X. C. Lou1,65,71 ,
H. J. Lu25 ,
J. G. Lu1,65 ,
X. L. Lu16 ,
Y. Lu7 ,
Y. H. Lu1,71 ,
Y. P. Lu1,65 ,
Z. H. Lu1,71 ,
C. L. Luo46 ,
J. R. Luo66 ,
J. S. Luo1,71 ,
M. X. Luo88,
T. Luo12,h ,
X. L. Luo1,65 ,
Z. Y. Lv23 ,
X. R. Lyu71,p ,
Y. F. Lyu48 ,
Y. H. Lyu89 ,
F. C. Ma44 ,
H. L. Ma1 ,
Heng Ma27,j ,
J. L. Ma1,71 ,
L. L. Ma55 ,
L. R. Ma73 ,
Q. M. Ma1 ,
R. Q. Ma1,71 ,
R. Y. Ma20 ,
T. Ma79,65 ,
X. T. Ma1,71 ,
X. Y. Ma1,65 ,
Y. M. Ma34 ,
F. E. Maas19 ,
I. MacKay77 ,
M. Maggiora82A,82C ,
S. Maity34 ,
S. Malde77 ,
Q. A. Malik81 ,
H. X. Mao42,l,m ,
Y. J. Mao51,i ,
Z. P. Mao1 ,
S. Marcello82A,82C ,
A. Marshall70 ,
F. M. Melendi31A,31B ,
Y. H. Meng71 ,
Z. X. Meng73 ,
G. Mezzadri31A ,
H. Miao1,71 ,
T. J. Min47 ,
R. E. Mitchell29 ,
X. H. Mo1,65,71 ,
B. Moses29 ,
N. Yu. Muchnoi4,d ,
J. Muskalla39 ,
Y. Nefedov40 ,
F. Nerling19,f ,
H. Neuwirth76 ,
Z. Ning1,65 ,
S. Nisar33,a,
Q. L. Niu42,l,m ,
W. D. Niu12,h ,
Y. Niu55 ,
C. Normand70 ,
S. L. Olsen11,71 ,
Q. Ouyang1,65,71 ,
I. V. Ovtin4 ,
S. Pacetti30B,30C ,
Y. Pan63 ,
A. Pathak11 ,
Y. P. Pei79,65 ,
M. Pelizaeus3 ,
G. L. Peng79,65 ,
H. P. Peng79,65 ,
X. J. Peng42,l,m ,
Y. Y. Peng42,l,m ,
K. Peters13,f ,
K. Petridis70 ,
J. L. Ping46 ,
R. G. Ping1,71 ,
S. Plura39 ,
V. Prasad38 ,
L. Pöpping3 ,
F. Z. Qi1 ,
H. R. Qi68 ,
M. Qi47 ,
S. Qian1,65 ,
W. B. Qian71 ,
C. F. Qiao71 ,
J. H. Qiao20 ,
J. J. Qin80 ,
J. L. Qin61 ,
L. Q. Qin14 ,
L. Y. Qin79,65 ,
P. B. Qin80 ,
X. P. Qin43 ,
X. S. Qin55 ,
Z. H. Qin1,65 ,
J. F. Qiu1 ,
Z. H. Qu80 ,
J. Rademacker70 ,
K. Ravindran74 ,
C. F. Redmer39 ,
A. Rivetti82C ,
M. Rolo82C ,
G. Rong1,71 ,
S. S. Rong1,71 ,
F. Rosini30B,30C ,
Ch. Rosner19 ,
M. Q. Ruan1,65 ,
N. Salone49,r ,
A. Sarantsev40,e ,
Y. Schelhaas39 ,
M. Schernau36 ,
K. Schoenning83 ,
M. Scodeggio31A ,
W. Shan26 ,
X. Y. Shan79,65 ,
Z. J. Shang42,l,m ,
J. F. Shangguan17 ,
L. G. Shao1,71 ,
M. Shao79,65 ,
C. P. Shen12,h ,
H. F. Shen1,9 ,
W. H. Shen71 ,
X. Y. Shen1,71 ,
B. A. Shi71 ,
Ch. Y. Shi87,c ,
H. Shi79,65 ,
J. L. Shi8,q ,
J. Y. Shi1 ,
M. H. Shi89 ,
S. Y. Shi80 ,
X. Shi1,65 ,
H. L. Song79,65 ,
J. J. Song20 ,
M. H. Song42 ,
T. Z. Song66 ,
W. M. Song38 ,
Y. X. Song51,i,n ,
Zirong Song27,j ,
S. Sosio82A,82C ,
S. Spataro82A,82C ,
S. Stansilaus77 ,
F. Stieler39 ,
M. Stolte3 ,
S. S Su44 ,
G. B. Sun84 ,
G. X. Sun1 ,
H. Sun71 ,
H. K. Sun1 ,
J. F. Sun20 ,
K. Sun68 ,
L. Sun84 ,
R. Sun79 ,
S. S. Sun1,71 ,
T. Sun57,g ,
W. Y. Sun56 ,
Y. C. Sun84 ,
Y. H. Sun32 ,
Y. J. Sun79,65 ,
Y. Z. Sun1 ,
Z. Q. Sun1,71 ,
Z. T. Sun55 ,
H. Tabaharizato1 ,
C. J. Tang60,
G. Y. Tang1 ,
J. Tang66 ,
J. J. Tang79,65 ,
L. F. Tang43 ,
Y. A. Tang84 ,
Z. H. Tang1,71 ,
L. Y. Tao80 ,
M. Tat77 ,
J. X. Teng79,65 ,
J. Y. Tian79,65 ,
W. H. Tian66 ,
Y. Tian34 ,
Z. F. Tian84 ,
K. Yu. Todyshev4 ,
I. Uman69B ,
E. van der Smagt3 ,
B. Wang66 ,
Bin Wang1 ,
Bo Wang79,65 ,
C. Wang42,l,m ,
Chao Wang20 ,
Cong Wang23 ,
D. Y. Wang51,i ,
F. K. Wang66 ,
H. J. Wang42,l,m ,
H. R. Wang86 ,
J. Wang10 ,
J. J. Wang84 ,
J. P. Wang37 ,
K. Wang1,65 ,
L. L. Wang1 ,
L. W. Wang38 ,
M. Wang55 ,
Mi Wang79,65 ,
N. Y. Wang71 ,
S. Wang42,l,m ,
Shun Wang64 ,
T. Wang12,h ,
W. Wang66 ,
W. P. Wang39 ,
X. F. Wang42,l,m ,
X. L. Wang12,h ,
X. N. Wang1,71 ,
Xin Wang27,j ,
Y. Wang1 ,
Y. D. Wang50 ,
Y. F. Wang1,9,71 ,
Y. H. Wang42,l,m ,
Y. J. Wang79,65 ,
Y. L. Wang20 ,
Y. N. Wang50 ,
Yanning Wang84 ,
Yaqian Wang18 ,
Yi Wang68 ,
Yuan Wang18,34 ,
Z. Wang1,65 ,
Z. L. Wang2 ,
Z. Q. Wang12,h ,
Z. Y. Wang1,71 ,
Zhi Wang48 ,
Ziyi Wang71 ,
D. Wei48 ,
D. H. Wei14 ,
D. J. Wei73 ,
H. R. Wei48 ,
F. Weidner76 ,
H. R. Wen34 ,
S. P. Wen1 ,
U. Wiedner3 ,
G. Wilkinson77 ,
M. Wolke83,
J. F. Wu1,9 ,
L. H. Wu1 ,
L. J. Wu20 ,
Lianjie Wu20 ,
S. G. Wu1,71 ,
S. M. Wu71 ,
X. W. Wu80 ,
Z. Wu1,65 ,
H. L. Xia79,65 ,
L. Xia79,65 ,
B. H. Xiang1,71 ,
D. Xiao42,l,m ,
G. Y. Xiao47 ,
H. Xiao80 ,
Y. L. Xiao12,h ,
Z. J. Xiao46 ,
C. Xie47 ,
K. J. Xie1,71 ,
Y. Xie55 ,
Y. G. Xie1,65 ,
Y. H. Xie6 ,
Z. P. Xie79,65 ,
T. Y. Xing1,71 ,
D. B. Xiong1 ,
G. F. Xu1 ,
H. Y. Xu2 ,
Q. J. Xu17 ,
Q. N. Xu32 ,
T. D. Xu80 ,
X. P. Xu61 ,
Y. Xu12,h ,
Y. C. Xu86 ,
Z. S. Xu71 ,
F. Yan24 ,
L. Yan12,h ,
W. B. Yan79,65 ,
W. C. Yan89 ,
W. H. Yan6 ,
W. P. Yan20 ,
X. Q. Yan12,h ,
Y. Y. Yan67 ,
H. J. Yang57,g ,
H. L. Yang38 ,
H. X. Yang1 ,
J. H. Yang47 ,
R. J. Yang20 ,
X. Y. Yang73 ,
Y. Yang12,h ,
Y. G. Yang56 ,
Y. H. Yang48 ,
Y. M. Yang89 ,
Y. Q. Yang10 ,
Y. Z. Yang20 ,
Youhua Yang47 ,
Z. Y. Yang80 ,
W. J. Yao6 ,
Z. P. Yao55 ,
M. Ye1,65 ,
M. H. Ye9,† ,
Z. J. Ye62,k ,
K. Yi46 ,
Junhao Yin48 ,
Z. Y. You66 ,
B. X. Yu1,65,71 ,
C. X. Yu48 ,
G. Yu13 ,
J. S. Yu27,j ,
L. W. Yu12,h ,
T. Yu80 ,
X. D. Yu51,i ,
Y. C. Yu89 ,
Yongchao Yu42 ,
C. Z. Yuan1,71 ,
H. Yuan1,71 ,
J. Yuan38 ,
Jie Yuan50 ,
L. Yuan2 ,
M. K. Yuan12,h ,
S. H. Yuan80 ,
Y. Yuan1,71 ,
C. X. Yue43 ,
Ying Yue20 ,
A. A. Zafar81 ,
F. R. Zeng55 ,
S. H. Zeng70 ,
X. Zeng12,h ,
Y. J. Zeng1,71 ,
Yujie Zeng66 ,
Y. C. Zhai55 ,
Y. H. Zhan66 ,
B. L. Zhang1,71 ,
B. X. Zhang1,† ,
D. H. Zhang48 ,
G. Y. Zhang20 ,
Gengyuan Zhang1,71 ,
H. Zhang79,65 ,
H. C. Zhang1,65,71 ,
H. H. Zhang66 ,
H. Q. Zhang1,65,71 ,
H. R. Zhang79,65 ,
H. Y. Zhang1,65 ,
Han Zhang89 ,
J. Zhang66 ,
J. J. Zhang58 ,
J. L. Zhang21 ,
J. Q. Zhang46 ,
J. S. Zhang12,h ,
J. W. Zhang1,65,71 ,
J. X. Zhang42,l,m ,
J. Y. Zhang1 ,
J. Z. Zhang1,71 ,
Jianyu Zhang71 ,
Jin Zhang53 ,
Jiyuan Zhang12,h ,
L. M. Zhang68 ,
Lei Zhang47 ,
N. Zhang38 ,
P. Zhang1,9 ,
Q. Zhang20 ,
Q. Y. Zhang38 ,
Q. Z. Zhang71 ,
R. Y. Zhang42,l,m ,
S. H. Zhang1,71 ,
S. N. Zhang77 ,
Shulei Zhang27,j ,
X. M. Zhang1 ,
X. Y. Zhang55 ,
Y. T. Zhang89 ,
Y. H. Zhang1,65 ,
Y. P. Zhang79,65 ,
Yao Zhang1 ,
Yu Zhang80 ,
Yu Zhang66 ,
Z. Zhang34 ,
Z. D. Zhang1 ,
Z. H. Zhang1 ,
Z. L. Zhang38 ,
Z. X. Zhang20 ,
Z. Y. Zhang84 ,
Zh. Zh. Zhang20 ,
Zhilong Zhang61 ,
Ziyang Zhang50 ,
Ziyu Zhang48 ,
G. Zhao1 ,
J.-P. Zhao71 ,
J. Y. Zhao1,71 ,
J. Z. Zhao1,65 ,
L. Zhao1 ,
Lei Zhao79,65 ,
M. G. Zhao48 ,
R. P. Zhao71 ,
S. J. Zhao89 ,
Y. B. Zhao1,65 ,
Y. L. Zhao61 ,
Y. P. Zhao50 ,
Y. X. Zhao34,71 ,
Z. G. Zhao79,65 ,
A. Zhemchugov40,b ,
B. Zheng80 ,
B. M. Zheng38 ,
J. P. Zheng1,65 ,
W. J. Zheng1,71 ,
W. Q. Zheng10 ,
X. R. Zheng20 ,
Y. H. Zheng71,p ,
B. Zhong46 ,
C. Zhong20 ,
X. Zhong45 ,
H. Zhou39,55,o ,
J. Q. Zhou38 ,
S. Zhou6 ,
X. Zhou84 ,
X. K. Zhou6 ,
X. R. Zhou79,65 ,
X. Y. Zhou43 ,
Y. X. Zhou86 ,
Y. Z. Zhou20 ,
A. N. Zhu71 ,
J. Zhu48 ,
K. Zhu1 ,
K. J. Zhu1,65,71 ,
K. S. Zhu12,h ,
L. X. Zhu71 ,
Lin Zhu20 ,
S. H. Zhu78 ,
T. J. Zhu12,h ,
W. D. Zhu12,h ,
W. J. Zhu1 ,
W. Z. Zhu20 ,
Y. C. Zhu79,65 ,
Z. A. Zhu1,71 ,
X. Y. Zhuang48 ,
M. Zhuge55 ,
J. H. Zou1 ,
J. Zu34 .

1 Institute of High Energy Physics, Beijing 100049, People’s Republic of China
2 Beihang University, Beijing 100191, People’s Republic of China
3 Bochum Ruhr-University, D-44780 Bochum, Germany
4 Budker Institute of Nuclear Physics SB RAS (BINP), Novosibirsk 630090, Russia
5 Carnegie Mellon University, Pittsburgh, Pennsylvania 15213, USA
6 Central China Normal University, Wuhan 430079, People’s Republic of China
7 Central South University, Changsha 410083, People’s Republic of China
8 Chengdu University of Technology, Chengdu 610059, People’s Republic of China
9 China Center of Advanced Science and Technology, Beijing 100190, People’s Republic of China
10 China University of Geosciences, Wuhan 430074, People’s Republic of China
11 Chung-Ang University, Seoul, 06974, Republic of Korea
12 Fudan University, Shanghai 200433, People’s Republic of China
13 GSI Helmholtzcentre for Heavy Ion Research GmbH, D-64291 Darmstadt, Germany
14 Guangxi Normal University, Guilin 541004, People’s Republic of China
15 Guangxi University, Nanning 530004, People’s Republic of China
16 Guangxi University of Science and Technology, Liuzhou 545006, People’s Republic of China
17 Hangzhou Normal University, Hangzhou 310036, People’s Republic of China
18 Hebei University, Baoding 071002, People’s Republic of China
19 Helmholtz Institute Mainz, Staudinger Weg 18, D-55099 Mainz, Germany
20 Henan Normal University, Xinxiang 453007, People’s Republic of China
21 Henan University, Kaifeng 475004, People’s Republic of China
22 Henan University of Science and Technology, Luoyang 471003, People’s Republic of China
23 Henan University of Technology, Zhengzhou 450001, People’s Republic of China
24 Hengyang Normal University, Hengyang 421001, People’s Republic of China
25 Huangshan College, Huangshan 245000, People’s Republic of China
26 Hunan Normal University, Changsha 410081, People’s Republic of China
27 Hunan University, Changsha 410082, People’s Republic of China
28 Indian Institute of Technology Madras, Chennai 600036, India
29 Indiana University, Bloomington, Indiana 47405, USA
30 INFN Laboratori Nazionali di Frascati, (A)INFN Laboratori Nazionali di Frascati, I-00044, Frascati, Italy; (B)INFN Sezione di Perugia, I-06100, Perugia, Italy; (C)University of Perugia, I-06100, Perugia, Italy
31 INFN Sezione di Ferrara, (A)INFN Sezione di Ferrara, I-44122, Ferrara, Italy; (B)University of Ferrara, I-44122, Ferrara, Italy
32 Inner Mongolia University, Hohhot 010021, People’s Republic of China
33 Institute of Business Administration, Karachi,
34 Institute of Modern Physics, Lanzhou 730000, People’s Republic of China
35 Institute of Physics and Technology, Mongolian Academy of Sciences, Peace Avenue 54B, Ulaanbaatar 13330, Mongolia
36 Instituto de Alta Investigación, Universidad de Tarapacá, Casilla 7D, Arica 1000000, Chile
37 Jiangsu Ocean University, Lianyungang 222000, People’s Republic of China
38 Jilin University, Changchun 130012, People’s Republic of China
39 Johannes Gutenberg University of Mainz, Johann-Joachim-Becher-Weg 45, D-55099 Mainz, Germany
40 Joint Institute for Nuclear Research, 141980 Dubna, Moscow region, Russia
41 Justus-Liebig-Universitaet Giessen, II. Physikalisches Institut, Heinrich-Buff-Ring 16, D-35392 Giessen, Germany
42 Lanzhou University, Lanzhou 730000, People’s Republic of China
43 Liaoning Normal University, Dalian 116029, People’s Republic of China
44 Liaoning University, Shenyang 110036, People’s Republic of China
45 Longyan University, Longyan 364000, People’s Republic of China
46 Nanjing Normal University, Nanjing 210023, People’s Republic of China
47 Nanjing University, Nanjing 210093, People’s Republic of China
48 Nankai University, Tianjin 300071, People’s Republic of China
49 National Centre for Nuclear Research, Warsaw 02-093, Poland
50 North China Electric Power University, Beijing 102206, People’s Republic of China
51 Peking University, Beijing 100871, People’s Republic of China
52 Qufu Normal University, Qufu 273165, People’s Republic of China
53 Renmin University of China, Beijing 100872, People’s Republic of China
54 Shandong Normal University, Jinan 250014, People’s Republic of China
55 Shandong University, Jinan 250100, People’s Republic of China
56 Shandong University of Technology, Zibo 255000, People’s Republic of China
57 Shanghai Jiao Tong University, Shanghai 200240, People’s Republic of China
58 Shanxi Normal University, Linfen 041004, People’s Republic of China
59 Shanxi University, Taiyuan 030006, People’s Republic of China
60 Sichuan University, Chengdu 610064, People’s Republic of China
61 Soochow University, Suzhou 215006, People’s Republic of China
62 South China Normal University, Guangzhou 510006, People’s Republic of China
63 Southeast University, Nanjing 211100, People’s Republic of China
64 Southwest University of Science and Technology, Mianyang 621010, People’s Republic of China
65 State Key Laboratory of Particle Detection and Electronics, Beijing 100049, Hefei 230026, People’s Republic of China
66 Sun Yat-Sen University, Guangzhou 510275, People’s Republic of China
67 Suranaree University of Technology, University Avenue 111, Nakhon Ratchasima 30000, Thailand
68 Tsinghua University, Beijing 100084, People’s Republic of China
69 Turkish Accelerator Center Particle Factory Group, (A)Istinye University, 34010, Istanbul, Turkey; (B)Near East University, Nicosia, North Cyprus, 99138, Mersin 10, Turkey
70 University of Bristol, H H Wills Physics Laboratory, Tyndall Avenue, Bristol, BS8 1TL, UK
71 University of Chinese Academy of Sciences, Beijing 100049, People’s Republic of China
72 University of Hawaii, Honolulu, Hawaii 96822, USA
73 University of Jinan, Jinan 250022, People’s Republic of China
74 University of La Serena, Av. Raúl Bitrán 1305, La Serena, Chile
75 University of Manchester, Oxford Road, Manchester, M13 9PL, United Kingdom
76 University of Muenster, Wilhelm-Klemm-Strasse 9, 48149 Muenster, Germany
77 University of Oxford, Keble Road, Oxford OX13RH, United Kingdom
78 University of Science and Technology Liaoning, Anshan 114051, People’s Republic of China
79 University of Science and Technology of China, Hefei 230026, People’s Republic of China
80 University of South China, Hengyang 421001, People’s Republic of China
81 University of the Punjab, Lahore-54590, Pakistan
82 University of Turin and INFN, (A)University of Turin, I-10125, Turin, Italy; (B)University of Eastern Piedmont, I-15121, Alessandria, Italy; (C)INFN, I-10125, Turin, Italy
83 Uppsala University, Box 516, SE-75120 Uppsala, Sweden
84 Wuhan University, Wuhan 430072, People’s Republic of China
85 Xi’an Jiaotong University, No.28 Xianning West Road, Xi’an, Shaanxi 710049, P.R. China
86 Yantai University, Yantai 264005, People’s Republic of China
87 Yunnan University, Kunming 650500, People’s Republic of China
88 Zhejiang University, Hangzhou 310027, People’s Republic of China
89 Zhengzhou University, Zhengzhou 450001, People’s Republic of China

† Deceased
a Also at Bogazici University, 34342 Istanbul, Turkey
b Also at the Moscow Institute of Physics and Technology, Moscow 141700, Russia
c Also at the Functional Electronics Laboratory, Tomsk State University, Tomsk, 634050, Russia
d Also at the Novosibirsk State University, Novosibirsk, 630090, Russia
e Also at the NRC ”Kurchatov Institute”, PNPI, 188300, Gatchina, Russia
f Also at Goethe University Frankfurt, 60323 Frankfurt am Main, Germany
g Also at Key Laboratory for Particle Physics, Astrophysics and Cosmology, Ministry of Education; Shanghai Key Laboratory for Particle Physics and Cosmology; Institute of Nuclear and Particle Physics, Shanghai 200240, People’s Republic of China
h Also at Key Laboratory of Nuclear Physics and Ion-beam Application (MOE) and Institute of Modern Physics, Fudan University, Shanghai 200443, People’s Republic of China
i Also at State Key Laboratory of Nuclear Physics and Technology, Peking University, Beijing 100871, People’s Republic of China
j Also at School of Physics and Electronics, Hunan University, Changsha 410082, China
k Also at Guangdong Provincial Key Laboratory of Nuclear Science, Institute of Quantum Matter, South China Normal University, Guangzhou 510006, China
l Also at MOE Frontiers Science Center for Rare Isotopes, Lanzhou University, Lanzhou 730000, People’s Republic of China
m Also at Lanzhou Center for Theoretical Physics, Lanzhou University, Lanzhou 730000, People’s Republic of China
n Also at Ecole Polytechnique Federale de Lausanne (EPFL), CH-1015 Lausanne, Switzerland
o Also at Helmholtz Institute Mainz, Staudinger Weg 18, D-55099 Mainz, Germany
p Also at Hangzhou Institute for Advanced Study, University of Chinese Academy of Sciences, Hangzhou 310024, China
q Also at Applied Nuclear Technology in Geosciences Key Laboratory of Sichuan Province, Chengdu University of Technology, Chengdu 610059, People’s Republic of China
r Currently at University of Silesia in Katowice, Institute of Physics, 75 Pulku Piechoty 1, 41-500 Chorzow, Poland

LHCb collaboration

R. Aaij38 ,
A.S.W. Abdelmotteleb58 ,
C. Abellan Beteta52 ,
F. Abudinén60 ,
T. Ackernley62 ,
A. A.  Adefisoye70 ,
B. Adeva48 ,
M. Adinolfi56 ,
P. Adlarson88 ,
C. Agapopoulou14 ,
C.A. Aidala90 ,
Z. Ajaltouni11,
S. Akar11 ,
K. Akiba38 ,
P. Albicocco28 ,
J. Albrecht19,f ,
R.  Aleksiejunas82 ,
F. Alessio50 ,
P. Alvarez Cartelle57,48 ,
R. Amalric16 ,
S. Amato3 ,
J.L. Amey56 ,
Y. Amhis14 ,
L. An6 ,
L. Anderlini27 ,
M. Andersson52 ,
P. Andreola52 ,
M. Andreotti26 ,
S.  Andres Estrada45 ,
A. Anelli31,o ,
D. Ao7 ,
C. Arata12 ,
F. Archilli37 ,
Z. Areg70 ,
M. Argenton26 ,
S. Arguedas Cuendis9,50 ,
L.  Arnone31,o ,
A. Artamonov44 ,
M. Artuso70 ,
E. Aslanides13 ,
R. Ataíde Da Silva51 ,
M. Atzeni66 ,
B. Audurier12 ,
J. A.  Authier15 ,
D. Bacher65 ,
I. Bachiller Perea51 ,
S. Bachmann22 ,
M. Bachmayer51 ,
J.J. Back58 ,
Z. B.  Bai8 ,
P. Baladron Rodriguez48 ,
V. Balagura15 ,
A.  Balboni26 ,
W. Baldini26 ,
Z. Baldwin80 ,
L. Balzani19 ,
H.  Bao7 ,
J. Baptista de Souza Leite2 ,
C. Barbero Pretel48,12 ,
M. Barbetti27 ,
I. R. Barbosa71 ,
R.J. Barlow64,† ,
M. Barnyakov25 ,
S. Barsuk14 ,
W. Barter60 ,
J. Bartz70 ,
S. Bashir40 ,
B. Batsukh83 ,
P. B.  Battista14 ,
A.  Bavarchee81 ,
A. Bay51 ,
A. Beck66 ,
M. Becker19 ,
F. Bedeschi35 ,
I.B. Bediaga2 ,
N. A.  Behling19 ,
S. Belin48 ,
A.  Bellavista25 ,
K. Belous44 ,
I. Belov29 ,
I. Belyaev36 ,
G. Benane13 ,
G. Bencivenni28 ,
E. Ben-Haim16 ,
A. Berezhnoy44 ,
R. Bernet52 ,
A. Bertolin33 ,
F. Betti60 ,
J.  Bex57 ,
O. Bezshyyko89 ,
S.  Bhattacharya81 ,
M.S. Bieker18 ,
N.V. Biesuz26 ,
A. Biolchini38 ,
M. Birch63 ,
F.C.R. Bishop10 ,
A. Bitadze64 ,
A. Bizzeti27,p ,
T. Blake58,b ,
F. Blanc51 ,
J.E. Blank19 ,
S. Blusk70 ,
V. Bocharnikov44 ,
J.A. Boelhauve19 ,
O. Boente Garcia50 ,
T. Boettcher91 ,
A.  Bohare60 ,
A. Boldyrev44 ,
C. Bolognani85 ,
R. Bolzonella26,l ,
R. B.  Bonacci1 ,
N. Bondar44,50 ,
A. Bordelius50 ,
F. Borgato33,50 ,
S. Borghi64 ,
M. Borsato31,o ,
J.T. Borsuk87 ,
E.  Bottalico62 ,
S.A. Bouchiba51 ,
M.  Bovill65 ,
T.J.V. Bowcock62 ,
A. Boyer50 ,
C. Bozzi26 ,
J. D. Brandenburg92 ,
A. Brea Rodriguez51 ,
N. Breer19 ,
J. Brodzicka41 ,
J. Brown62 ,
D. Brundu32 ,
E. Buchanan60 ,
M.  Burgos Marcos85 ,
C. Burr50 ,
C.  Buti27 ,
J.S. Butter57 ,
J. Buytaert50 ,
W. Byczynski50 ,
S. Cadeddu32 ,
H. Cai76 ,
Y.  Cai5 ,
A. Caillet16 ,
R. Calabrese26,l ,
L. Calefice46 ,
M. Calvi31,o ,
M. Calvo Gomez47 ,
P. Camargo Magalhaes2,a ,
J. I. Cambon Bouzas48 ,
P. Campana28 ,
A. C. Campos3 ,
A.F. Campoverde Quezada7 ,
Y.  Cao6,
S. Capelli31 ,
M.  Caporale25 ,
L. Capriotti26 ,
R. Caravaca-Mora9 ,
A. Carbone25,j ,
L. Carcedo Salgado48 ,
R. Cardinale29,m ,
A. Cardini32 ,
P. Carniti31 ,
L. Carus22 ,
A. Casais Vidal66 ,
R. Caspary22 ,
G. Casse62 ,
M. Cattaneo50 ,
G. Cavallero26 ,
V. Cavallini26,l ,
S. Celani50 ,
I.  Celestino35,s ,
S.  Cesare50,n ,
A.J. Chadwick62 ,
I. Chahrour90 ,
H.  Chang4,c ,
M. Charles16 ,
Ph. Charpentier50 ,
E.  Chatzianagnostou38 ,
R.  Cheaib81 ,
M. Chefdeville10 ,
C. Chen58 ,
J.  Chen51 ,
S. Chen5 ,
Z. Chen7 ,
A.  Chen Hu63 ,
M.  Cherif12 ,
A. Chernov41 ,
S. Chernyshenko54 ,
X.  Chiotopoulos85 ,
G.  Chizhik1 ,
V. Chobanova45 ,
M. Chrzaszcz41 ,
A. Chubykin44 ,
V. Chulikov28,50,36 ,
P. Ciambrone28 ,
X. Cid Vidal48 ,
G. Ciezarek50 ,
P. Cifra50 ,
P.E.L. Clarke60 ,
M. Clemencic50 ,
H.V. Cliff57 ,
J. Closier50 ,
C. Cocha Toapaxi22 ,
V. Coco50 ,
J. Cogan13 ,
E. Cogneras11 ,
L. Cojocariu43 ,
S.  Collaviti51 ,
P. Collins50 ,
T. Colombo50 ,
M. Colonna19 ,
A. Comerma-Montells46 ,
L. Congedo24 ,
J.  Connaughton58 ,
A. Contu32 ,
N. Cooke61 ,
G. Cordova35,s ,
C.  Coronel67 ,
I. Corredoira 12 ,
A. Correia16 ,
G. Corti50 ,
J. Cottee Meldrum56 ,
B. Couturier50 ,
D.C. Craik52 ,
M. Cruz Torres2,g ,
M.  Cubero Campos9 ,
E. Curras Rivera51 ,
R. Currie60 ,
C.L. Da Silva69 ,
X. Dai4 ,
E. Dall’Occo50 ,
J. Dalseno45 ,
C. D’Ambrosio63 ,
J. Daniel11 ,
G. Darze3 ,
A.  Davidson58 ,
J.E. Davies64 ,
O. De Aguiar Francisco64 ,
C. De Angelis32,k ,
F. De Benedetti50 ,
J. de Boer38 ,
K. De Bruyn84 ,
S. De Capua64 ,
M. De Cian64 ,
U. De Freitas Carneiro Da Graca2 ,
E. De Lucia28 ,
J.M. De Miranda2 ,
L. De Paula3 ,
M. De Serio24,h ,
P. De Simone28 ,
F. De Vellis19 ,
J.A. de Vries85 ,
F. Debernardis24 ,
D. Decamp10 ,
S.  Dekkers1 ,
L. Del Buono16 ,
B. Delaney66 ,
J. Deng8 ,
V. Denysenko52 ,
O. Deschamps11 ,
F. Dettori32,k ,
B. Dey81 ,
P. Di Nezza28 ,
I. Diachkov44 ,
S. Ding70 ,
Y.  Ding51 ,
L. Dittmann22 ,
A. D.  Docheva61 ,
A.  Doheny58 ,
C. Dongc,4 ,
F. Dordei32 ,
A.C. dos Reis2 ,
A. D.  Dowling70 ,
L. Dreyfus13 ,
W. Duan74 ,
P. Duda87 ,
L. Dufour51 ,
V. Duk34 ,
P. Durante50 ,
M. M. Duras87 ,
J.M. Durham69 ,
O. D.  Durmus81 ,
A. Dziurda41 ,
A. Dzyuba44 ,
S. Easo59 ,
E. Eckstein18 ,
U. Egede1 ,
A. Egorychev44 ,
V. Egorychev44 ,
S. Eisenhardt60 ,
E. Ejopu62 ,
L. Eklund88 ,
M. Elashri67 ,
D.  Elizondo Blanco9 ,
J. Ellbracht19 ,
S. Ely63 ,
A. Ene43 ,
J. Eschle70 ,
T. Evans38 ,
F. Fabiano14 ,
S.  Faghih67 ,
L.N. Falcao31,o ,
B. Fang7 ,
R. Fantechi35 ,
L. Fantini34,r ,
M. Faria51 ,
K.  Farmer60 ,
F.  Fassin84,38 ,
D. Fazzini31,o ,
L. Felkowski87 ,
C.  Feng6,
M. Feng5,7 ,
A. Fernandez Casani49 ,
M. Fernandez Gomez48 ,
A.D. Fernez68 ,
F. Ferrari25,j ,
F. Ferreira Rodrigues3 ,
M. Ferrillo52 ,
M. Ferro-Luzzi50 ,
S. Filippov44 ,
R.A. Fini24 ,
M. Fiorini26,l ,
M. Firlej40 ,
K.L. Fischer65 ,
D.S. Fitzgerald90 ,
C. Fitzpatrick64 ,
T. Fiutowski40 ,
F. Fleuret15 ,
A.  Fomin53 ,
M. Fontana25,50 ,
L. A.  Foreman64 ,
R. Forty50 ,
D. Foulds-Holt60 ,
V. Franco Lima3 ,
M. Franco Sevilla68 ,
M. Frank50 ,
E. Franzoso26,l ,
G. Frau64 ,
C. Frei50 ,
D.A. Friday64,50 ,
J. Fu7 ,
Q. Führing19,57,f ,
T. Fulghesu13 ,
G. Galati24,h ,
M.D. Galati38 ,
A. Gallas Torreira48 ,
D. Galli25,j ,
S. Gambetta60 ,
M. Gandelman3 ,
P. Gandini30 ,
B.  Ganie64 ,
H. Gao7 ,
R. Gao65 ,
T.Q. Gao57 ,
Y. Gao8 ,
Y. Gao6 ,
Y. Gao8 ,
L.M. Garcia Martin51 ,
P. Garcia Moreno46 ,
J. García Pardiñas66 ,
P.  Gardner68 ,
L. Garrido46 ,
C. Gaspar50 ,
A.  Gavrikov33 ,
L.L. Gerken19 ,
E. Gersabeck20 ,
M. Gersabeck20 ,
T. Gershon58 ,
S. Ghizzo29,m ,
Z. Ghorbanimoghaddam56 ,
F. I. Giasemis16,e ,
V. Gibson57 ,
H.K. Giemza42 ,
A.L. Gilman67 ,
M. Giovannetti28 ,
A. Gioventù48 ,
L. Girardey64,59 ,
M.A. Giza41 ,
F.C. Glaser22,14 ,
V.V. Gligorov16 ,
C. Göbel71 ,
L.  Golinka-Bezshyyko89 ,
E. Golobardes47 ,
D. Golubkov44 ,
A. Golutvin63,50 ,
S. Gomez Fernandez46 ,
W.  Gomulka40 ,
F. Goncalves Abrantes65 ,
I. Gonçales Vaz50 ,
M. Goncerz41 ,
G. Gong4,c ,
J. A. Gooding19 ,
I.V. Gorelov44 ,
C. Gotti31 ,
E. Govorkova66 ,
J.P. Grabowski30 ,
L.A. Granado Cardoso50 ,
E. Graugés46 ,
E. Graverini35,t,51 ,
L. Grazette58 ,
G. Graziani27 ,
A. T. Grecu43 ,
N.A. Grieser67 ,
L. Grillo61 ,
C.  Gu15 ,
M. Guarise26 ,
L.  Guerry11 ,
A.-K. Guseinov51 ,
E. Gushchin44 ,
Y. Guz6 ,
T. Gys50 ,
K. Habermann18 ,
T. Hadavizadeh1 ,
C. Hadjivasiliou68 ,
G. Haefeli51 ,
C. Haen50 ,
S.  Haken57 ,
G.  Hallett58 ,
P.M. Hamilton68 ,
J. Hammerich62 ,
Q. Han33 ,
X. Han22,50 ,
S. Hansmann-Menzemer22 ,
L. Hao7 ,
N. Harnew65 ,
T. J.  Harris1 ,
M. Hartmann14 ,
S. Hashmi40 ,
J. He7,d ,
N.  Heatley14 ,
A.  Hedes64 ,
F. Hemmer50 ,
C. Henderson67 ,
R. Henderson14 ,
R.D.L. Henderson1 ,
A.M. Hennequin50 ,
K. Hennessy62 ,
J. Herd63 ,
P. Herrero Gascon22 ,
J. Heuel17 ,
A.  Heyn13 ,
A. Hicheur3 ,
G. Hijano Mendizabal52 ,
J. Horswill64 ,
R. Hou8 ,
Y. Hou11 ,
D.C. Houston61 ,
N. Howarth62 ,
W. Hu7,d ,
X. Hu4 ,
W. Hulsbergen38 ,
R.J. Hunter58 ,
M. Hushchyn44 ,
D. Hutchcroft62 ,
M. Idzik40 ,
D. Ilin44 ,
P. Ilten67 ,
A.  Iohner10 ,
A. Ishteev44 ,
H. Jage17 ,
S.J. Jaimes Elles78,49,50 ,
S. Jakobsen50 ,
T. Jakoubek79 ,
E. Jans38 ,
B.K. Jashal49 ,
A. Jawahery68 ,
C.  Jayaweera55 ,
A.  Jelavic1 ,
V. Jevtic19 ,
Z.  Jia16 ,
E. Jiang68 ,
X. Jiang5,7 ,
Y. Jiang7 ,
Y. J.  Jiang6 ,
E. Jimenez Moya9 ,
N.  Jindal92 ,
M. John65 ,
A.  John Rubesh Rajan23 ,
D. Johnson55 ,
C.R. Jones57 ,
S. Joshi42 ,
B. Jost50 ,
J.  Juan Castella57 ,
N. Jurik50 ,
I. Juszczak41 ,
K.  Kalecinska40,
D. Kaminaris51 ,
S. Kandybei53 ,
M.  Kane60 ,
Y. Kang4,c ,
C. Kar11 ,
M. Karacson50 ,
A. Kauniskangas51 ,
J.W. Kautz67 ,
M.K. Kazanecki41 ,
F. Keizer50 ,
M. Kenzie57 ,
T. Ketel38 ,
B. Khanji70 ,
S. Kholodenko63,50 ,
G. Khreich14 ,
F.  Kiraz14,
T. Kirn17 ,
V.S. Kirsebom31,o ,
S. Klaver39 ,
N. Kleijne35,s ,
A. Kleimenova51 ,
D. K.  Klekots89 ,
K. Klimaszewski42 ,
M.R. Kmiec42 ,
T.  Knospe19 ,
R.  Kolb22 ,
S. Koliiev54 ,
L. Kolk19 ,
A. Konoplyannikov6 ,
P. Kopciewicz50 ,
P. Koppenburg38 ,
A.  Korchin53 ,
I. Kostiuk38 ,
O. Kot54 ,
S. Kotriakhova ,
E.  Kowalczyk68 ,
A. Kozachuk44 ,
P. Kravchenko44 ,
L. Kravchuk44 ,
O.  Kravcov82 ,
M. Kreps58 ,
P. Krokovny44 ,
W. Krupa70 ,
W. Krzemien42 ,
O. Kshyvanskyi54 ,
S. Kubis87 ,
M. Kucharczyk41 ,
V. Kudryavtsev44 ,
E. Kulikova44 ,
A. Kupsc88 ,
V. Kushnir53 ,
B. Kutsenko13 ,
J. Kvapil69 ,
I.  Kyryllin53 ,
D. Lacarrere50 ,
P.  Laguarta Gonzalez46 ,
A. Lai32 ,
A. Lampis32 ,
D. Lancierini63 ,
C. Landesa Gomez48 ,
J.J. Lane1 ,
G. Lanfranchi28 ,
C. Langenbruch22 ,
J. Langer19 ,
T. Latham58 ,
F. Lazzari35,t ,
C. Lazzeroni55 ,
R. Le Gac13 ,
H.  Lee62 ,
R. Lefèvre11 ,
A. Leflat44 ,
M. Lehuraux58 ,
E. Lemos Cid50 ,
O. Leroy13 ,
T. Lesiak41 ,
E. D. Lesser50 ,
B. Leverington22 ,
A. Li4,c ,
C.  Li4 ,
C.  Li13 ,
H. Li74 ,
J. Li8 ,
K. Li77 ,
L. Li64 ,
P. Li7 ,
P.-R. Li75 ,
Q.  Li5,7 ,
T. Li73 ,
T. Li74 ,
Y. Li8 ,
Y. Li5 ,
Y.  Li4 ,
Z. Lian4,c ,
Q.  Liang8,
X. Liang70 ,
Z.  Liang32 ,
S. Libralon49 ,
A.  Lightbody12 ,
C. Lin7 ,
T. Lin59 ,
R. Lindner50 ,
H.  Linton63 ,
R. Litvinov32 ,
D. Liu8 ,
F. L.  Liu1 ,
G. Liu74 ,
K. Liu75 ,
S. Liu5 ,
W.  Liu8 ,
Y. Liu60 ,
Y. Liu75 ,
Y. L.  Liu63 ,
G. Loachamin Ordonez71 ,
I.  Lobo1 ,
A. Lobo Salvia10 ,
A. Loi32 ,
T. Long57 ,
F. C. L. Lopes2,a ,
J.H. Lopes3 ,
A. Lopez Huertas46 ,
C.  Lopez Iribarnegaray48 ,
S. López Soliño48 ,
Q. Lu15 ,
C. Lucarelli50 ,
D. Lucchesi33,q ,
M. Lucio Martinez49 ,
Y. Luo6 ,
A. Lupato33,i ,
E. Luppi26,l ,
K. Lynch23 ,
S.  Lyu6,
X.-R. Lyu7 ,
G. M.  Ma4,c ,
H.  Ma73 ,
S. Maccolini50 ,
F. Machefert14 ,
F. Maciuc43 ,
B.  Mack70 ,
I. Mackay65 ,
L. M.  Mackey70 ,
L.R. Madhan Mohan57 ,
M. J.  Madurai55 ,
D. Magdalinski38 ,
D. Maisuzenko44 ,
J.J. Malczewski41 ,
S. Malde65 ,
L. Malentacca50 ,
A. Malinin44 ,
T. Maltsev44 ,
G. Manca32,k ,
G. Mancinelli13 ,
C. Mancuso14 ,
R. Manera Escalero46 ,
F. M.  Manganella37 ,
D. Manuzzi25 ,
D. Marangotto30,n ,
J.F. Marchand10 ,
R. Marchevski51 ,
U. Marconi25 ,
E. Mariani16 ,
S. Mariani50 ,
C. Marin Benito46 ,
J. Marks22 ,
A.M. Marshall56 ,
L.  Martel65 ,
G. Martelli34 ,
G. Martellotti36 ,
L. Martinazzoli50 ,
M. Martinelli31,o ,
D.  Martinez Gomez84 ,
D. Martinez Santos45 ,
F. Martinez Vidal49 ,
A.  Martorell i Granollers47 ,
A. Massafferri2 ,
R. Matev50 ,
A. Mathad50 ,
V. Matiunin44 ,
C. Matteuzzi70 ,
K.R. Mattioli15 ,
A. Mauri63 ,
E. Maurice15 ,
J. Mauricio46 ,
P. Mayencourt51 ,
J. Mazorra de Cos49 ,
M. Mazurek42 ,
D.  Mazzanti Tarancon46 ,
M. McCann63 ,
N.T. McHugh61 ,
A. McNab64 ,
R. McNulty23 ,
B. Meadows67 ,
D. Melnychuk42 ,
D. Mendoza Granada16 ,
P.  Menendez Valdes Perez48 ,
F. M.  Meng4,c ,
M. Merk38,85 ,
A. Merli51,30 ,
L. Meyer Garcia68 ,
D. Miao5,7 ,
H. Miao7 ,
M. Mikhasenko80 ,
D.A. Milanes86 ,
A. Minotti31,o ,
E. Minucci28 ,
T. Miralles11 ,
B. Mitreska64 ,
D.S. Mitzel19 ,
R.  Mocanu43 ,
A. Modak59 ,
L. Moeser19 ,
R.D. Moise17 ,
E. F. Molina Cardenas90 ,
T. Mombächer67 ,
M. Monk57 ,
T. Monnard51 ,
S. Monteil11 ,
A. Morcillo Gomez48 ,
G. Morello28 ,
M.J. Morello35,s ,
M.P. Morgenthaler22 ,
A.  Moro31,o ,
J. Moron40 ,
W.  Morren38 ,
A.B. Morris82,50 ,
A.G. Morris13 ,
R. Mountain70 ,
Z. Mu6 ,
E. Muhammad58 ,
F. Muheim60 ,
M. Mulder19 ,
K. Müller52 ,
F. Muñoz-Rojas9 ,
V.  Mytrochenko53 ,
P. Naik62 ,
T. Nakada51 ,
R. Nandakumar59 ,
G.  Napoletano51 ,
I. Nasteva3 ,
M. Needham60 ,
E.  Nekrasova44 ,
N. Neri30,n ,
S. Neubert18 ,
N. Neufeld50 ,
P. Neustroev44,
J. Nicolini50 ,
D. Nicotra85 ,
E.M. Niel15 ,
N. Nikitin44 ,
L.  Nisi19 ,
Q. Niu75 ,
B. K. Njoki50 ,
P. Nogarolli3 ,
P. Nogga18 ,
C. Normand48 ,
J. Novoa Fernandez48 ,
G. Nowak67 ,
C. Nunez90 ,
H. N.  Nur61 ,
A. Oblakowska-Mucha40 ,
V. Obraztsov44 ,
T. Oeser17 ,
A. Okhotnikov44,
O. Okhrimenko54 ,
R. Oldeman32,k ,
F. Oliva60,50 ,
E.  Olivart Pino46 ,
M. Olocco19 ,
R.H. O’Neil50 ,
J.S. Ordonez Soto11 ,
D. Osthues19 ,
J.M. Otalora Goicochea3 ,
P. Owen52 ,
A. Oyanguren49 ,
O. Ozcelik50 ,
F. Paciolla35,u ,
A.  Padee42 ,
K.O. Padeken18 ,
B. Pagare48 ,
T. Pajero50 ,
A. Palano24 ,
L.  Palini30 ,
M. Palutan28 ,
C.  Pan76 ,
X.  Pan4,c ,
S. Panebianco12 ,
S. Paniskaki50,33 ,
L. Paolucci64 ,
A. Papanestis59 ,
M. Pappagallo24,h ,
L.L. Pappalardo26 ,
C. Pappenheimer67 ,
C. Parkes64 ,
D.  Parmar80 ,
G. Passaleva27 ,
D. Passaro35,s ,
A. Pastore24 ,
M. Patel63 ,
J. Patoc65 ,
C. Patrignani25,j ,
A.  Paul70 ,
C.J. Pawley85 ,
A. Pellegrino38 ,
J.  Peng5,7 ,
X.  Peng75,
M. Pepe Altarelli28 ,
S. Perazzini25 ,
D. Pereima44 ,
H.  Pereira Da Costa69 ,
M.  Pereira Martinez48 ,
A. Pereiro Castro48 ,
C.  Perez47 ,
P. Perret11 ,
A.  Perrevoort84 ,
A. Perro50 ,
M.J. Peters67 ,
K. Petridis56 ,
A. Petrolini29,m ,
S.  Pezzulo29,m ,
J. P.  Pfaller67 ,
H. Pham70 ,
L. Pica35,s ,
M. Piccini34 ,
L.  Piccolo32 ,
B. Pietrzyk10 ,
R. N. Pilato62 ,
D. Pinci36 ,
F. Pisani50 ,
M. Pizzichemi31,o,50 ,
V. M. Placinta43 ,
M. Plo Casasus48 ,
T. Poeschl50 ,
F. Polci16 ,
M. Poli Lener28 ,
A. Poluektov13 ,
N. Polukhina44 ,
I. Polyakov64 ,
E. Polycarpo3 ,
S. Ponce50 ,
D. Popov7,50 ,
K. Popp19 ,
S. Poslavskii44 ,
K. Prasanth60 ,
C. Prouve45 ,
D. Provenzano32,k,50 ,
V. Pugatch54 ,
A.  Puicercus Gomez50 ,
G. Punzi35,t ,
J.R. Pybus69 ,
Q. Qian6 ,
W. Qian7 ,
N. Qin4,c ,
R. Quagliani50 ,
R.I. Rabadan Trejo58 ,
R.  Racz82 ,
J.H. Rademacker56 ,
M. Rama35 ,
M.  Ramírez García90 ,
V. Ramos De Oliveira71 ,
M. Ramos Pernas50 ,
M.S. Rangel3 ,
F. Ratnikov44 ,
G. Raven39 ,
M. Rebollo De Miguel49 ,
F. Redi30,i ,
J. Reich56 ,
F. Reiss20 ,
Z. Ren7 ,
P.K. Resmi65 ,
M.  Ribalda Galvez46 ,
R. Ribatti51 ,
G. Ricart12 ,
D. Riccardi35,s ,
S. Ricciardi59 ,
K. Richardson66 ,
M. Richardson-Slipper57 ,
F.  Riehn19 ,
K. Rinnert62 ,
P. Robbe14,50 ,
G. Robertson61 ,
E. Rodrigues62 ,
A. Rodriguez Alvarez46 ,
E. Rodriguez Fernandez48 ,
J.A. Rodriguez Lopez78 ,
E. Rodriguez Rodriguez50 ,
J. Roensch19 ,
A. Rogachev44 ,
A. Rogovskiy59 ,
D.L. Rolf19 ,
P. Roloff50 ,
V. Romanovskiy67 ,
A. Romero Vidal48 ,
G. Romolini26,50 ,
F. Ronchetti51 ,
T. Rong6 ,
M. Rotondo28 ,
M.S. Rudolph70 ,
M. Ruiz Diaz22 ,
R.A. Ruiz Fernandez48 ,
J. Ruiz Vidal85 ,
J. J. Saavedra-Arias9 ,
J.J. Saborido Silva48 ,
S. E. R. Sacha Emile R.50 ,
N. Sagidova44 ,
D. Sahoo81 ,
N. Sahoo55 ,
B. Saitta32 ,
M. Salomoni31,50,o ,
I. Sanderswood49 ,
R. Santacesaria36 ,
C. Santamarina Rios48 ,
M. Santimaria28 ,
L. Santoro 2 ,
E. Santovetti37 ,
A. Saputi26,50 ,
D. Saranin44 ,
A. Sarnatskiy84 ,
G. Sarpis50 ,
M. Sarpis82 ,
C. Satriano36 ,
A. Satta37 ,
M. Saur75 ,
D. Savrina44 ,
H. Sazak17 ,
F. Sborzacchi50,28 ,
A. Scarabotto19 ,
S. Schael17 ,
S. Scherl62 ,
M. Schiller22 ,
H. Schindler50 ,
M. Schmelling21 ,
B. Schmidt50 ,
N. Schmidt69 ,
S. Schmitt66 ,
H. Schmitz18,
O. Schneider51 ,
A. Schopper63 ,
N. Schulte19 ,
M.H. Schune14 ,
G. Schwering17 ,
B. Sciascia28 ,
A. Sciuccati50 ,
G.  Scriven85 ,
I. Segal80 ,
S. Sellam48 ,
A. Semennikov44 ,
T. Senger52 ,
M. Senghi Soares39 ,
A. Sergi29,m ,
N. Serra52 ,
L. Sestini27 ,
B.  Sevilla Sanjuan47 ,
Y. Shang6 ,
D.M. Shangase90 ,
M. Shapkin44 ,
R. S.  Sharma70 ,
L. Shchutska51 ,
T. Shears62 ,
J.  Shen6,
Z. Shen38 ,
S. Sheng51 ,
V. Shevchenko44 ,
B. Shi7 ,
J.  Shi57 ,
Q. Shi7 ,
W. S.  Shi74 ,
E. Shmanin25 ,
R. Shorkin44 ,
R. Silva Coutinho2 ,
G. Simi33,q ,
S. Simone24,h ,
M.  Singha81 ,
I. Siral51 ,
N. Skidmore58 ,
T. Skwarnicki70 ,
M.W. Slater55 ,
E. Smith66 ,
M. Smith63 ,
L. Soares Lavra60 ,
M.D. Sokoloff67 ,
F.J.P. Soler61 ,
A. Solomin56 ,
A. Solovev44 ,
K.  Solovieva20 ,
N. S.  Sommerfeld18 ,
R. Song1 ,
Y. Song51 ,
Y. Song4,c ,
Y. S.  Song6 ,
F.L. Souza De Almeida46 ,
B. Souza De Paula3 ,
K.M. Sowa40 ,
E. Spadaro Norella29,m ,
E. Spedicato25 ,
J.G. Speer19 ,
P. Spradlin61 ,
F. Stagni50 ,
M. Stahl80 ,
S. Stahl50 ,
S. Stanislaus65 ,
M.  Stefaniak92 ,
O. Steinkamp52 ,
D. Strekalina44 ,
Y. Su7 ,
F. Suljik65 ,
J. Sun32 ,
J.  Sun64 ,
L. Sun76 ,
D. Sundfeld2 ,
W. Sutcliffe52 ,
P. Svihra79 ,
V. Svintozelskyi49 ,
K. Swientek40 ,
F. Swystun57 ,
A. Szabelski42 ,
T. Szumlak40 ,
Y. Tan4 ,
Y. Tang76 ,
Y. T.  Tang7 ,
M.D. Tat22 ,
J. A. Teijeiro Jimenez48 ,
A. Terentev44 ,
F. Terzuoli35,u ,
F. Teubert50 ,
E. Thomas50 ,
D.J.D. Thompson55 ,
A. R.  Thomson-Strong60 ,
H. Tilquin63 ,
V. Tisserand11 ,
S. T’Jampens10 ,
M. Tobin5,50 ,
T. T.  Todorov20 ,
L. Tomassetti26,l ,
G. Tonani30 ,
X. Tong6 ,
T. Tork30 ,
L. Toscano19 ,
D.Y. Tou4,c ,
C. Trippl47 ,
G. Tuci22 ,
N. Tuning38 ,
L.H. Uecker22 ,
A. Ukleja40 ,
D.J. Unverzagt22 ,
A.  Upadhyay50 ,
B.  Urbach60 ,
A. Usachov38 ,
A. Ustyuzhanin44 ,
U. Uwer22 ,
V. Vagnoni25,50 ,
A.  Vaitkevicius82 ,
V.  Valcarce Cadenas48 ,
G. Valenti25 ,
N. Valls Canudas50 ,
J. van Eldik50 ,
H. Van Hecke69 ,
E. van Herwijnen63 ,
C.B. Van Hulse48,w ,
R. Van Laak51 ,
M. van Veghel85 ,
G. Vasquez52 ,
R. Vazquez Gomez46 ,
P. Vazquez Regueiro48 ,
C. Vázquez Sierra45 ,
S. Vecchi26 ,
J.  Velilla Serna49 ,
J.J. Velthuis56 ,
M. Veltri27,v ,
A. Venkateswaran51 ,
M. Verdoglia32 ,
M. Vesterinen58 ,
W. Vetens70 ,
D.  Vico Benet65 ,
P.  Vidrier Villalba46 ,
M. Vieites Diaz48 ,
X. Vilasis-Cardona47 ,
E. Vilella Figueras62 ,
A. Villa25 ,
P. Vincent16 ,
B. Vivacqua3 ,
F.C. Volle55 ,
D. vom Bruch13 ,
N. Voropaev44 ,
K. Vos85 ,
C. Vrahas60 ,
J. Wagner19 ,
J. Walsh35 ,
E.J. Walton1 ,
G. Wan6 ,
A.  Wang7 ,
B.  Wang5 ,
C. Wang22 ,
G. Wang8 ,
H. Wang75 ,
J. Wang7 ,
J. Wang5 ,
J. Wang4,c ,
J. Wang76 ,
M. Wang50 ,
N. W.  Wang7 ,
R. Wang56 ,
X. Wang8 ,
X. Wang74 ,
X. W.  Wang63 ,
Y. Wang77 ,
Y. Wang6 ,
Y. H.  Wang75 ,
Z. Wang14 ,
Z. Wang30 ,
J.A. Ward58,1 ,
M. Waterlaat50 ,
N.K. Watson55 ,
D. Websdale63 ,
Y. Wei6 ,
Z.  Weida7 ,
J. Wendel45 ,
B.D.C. Westhenry56 ,
C. White57 ,
M. Whitehead61 ,
E. Whiter55 ,
A.R. Wiederhold64 ,
D. Wiedner19 ,
M. A. Wiegertjes38 ,
C.  Wild65 ,
G. Wilkinson65,50 ,
M.K. Wilkinson67 ,
M. Williams66 ,
M. J. Williams50 ,
M.R.J. Williams60 ,
R. Williams57 ,
S.  Williams56 ,
Z.  Williams56 ,
F.F. Wilson59 ,
M. Winn12 ,
W. Wislicki42 ,
M. Witek41 ,
L. Witola19 ,
T. Wolf22 ,
E.  Wood57 ,
G. Wormser14 ,
S.A. Wotton57 ,
H. Wu70 ,
J. Wu8 ,
X. Wu76 ,
Y. Wu6,57 ,
Z. Wu7 ,
K. Wyllie50 ,
S. Xian74 ,
Z. Xiang5 ,
Y. Xie8 ,
T. X.  Xing30 ,
A. Xu35,s ,
L. Xu4,c ,
M. Xu50 ,
R.  Xu90,
Z. Xu50 ,
Z. Xu7 ,
Z. Xu5 ,
S.  Yadav26 ,
K.  Yang63 ,
X. Yang6 ,
Y. Yang7 ,
Y.  Yang81 ,
Z. Yang6 ,
Z.  Yang4 ,
H. Yeung64 ,
H. Yin8 ,
X.  Yin7 ,
C. Y.  Yu6 ,
J. Yu73 ,
X. Yuan5 ,
Y Yuan5,7 ,
J. A. Zamora Saa72 ,
M. Zavertyaev21 ,
M. Zdybal41 ,
F. Zenesini25 ,
C.  Zeng5,7 ,
M. Zeng4,c ,
S.H Zeng56 ,
C. Zhang6 ,
D. Zhang8 ,
J. Zhang7 ,
L. Zhang4,c ,
R. Zhang8 ,
S. Zhang65 ,
S. L.  Zhang73 ,
Y. Zhang6 ,
Y. Z.  Zhang4,c ,
Z. Zhang4,c ,
Y. Zhao22 ,
A. Zhelezov22 ,
S. Z.  Zheng6 ,
X. Z.  Zheng4,c ,
Y. Zheng7 ,
T. Zhou6 ,
X. Zhou8 ,
V. Zhovkovska58 ,
L. Z.  Zhu60 ,
X. Zhu4,c ,
X. Zhu8 ,
Y.  Zhu17 ,
V. Zhukov17 ,
J. Zhuo49 ,
D. Zuliani33,q ,
G. Zunica28 .

1School of Physics and Astronomy, Monash University, Melbourne, Australia
2Centro Brasileiro de Pesquisas Físicas (CBPF), Rio de Janeiro, Brazil
3Universidade Federal do Rio de Janeiro (UFRJ), Rio de Janeiro, Brazil
4Department of Engineering Physics, Tsinghua University, Beijing, China
5Institute Of High Energy Physics (IHEP), Beijing, China
6School of Physics State Key Laboratory of Nuclear Physics and Technology, Peking University, Beijing, China
7University of Chinese Academy of Sciences, Beijing, China
8Institute of Particle Physics, Central China Normal University, Wuhan, Hubei, China
9Consejo Nacional de Rectores (CONARE), San Jose, Costa Rica
10Université Savoie Mont Blanc, CNRS, IN2P3-LAPP, Annecy, France
11Université Clermont Auvergne, CNRS/IN2P3, LPC, Clermont-Ferrand, France
12Université Paris-Saclay, Centre d’Etudes de Saclay (CEA), IRFU, Gif-Sur-Yvette, France
13Aix Marseille Univ, CNRS/IN2P3, CPPM, Marseille, France
14Université Paris-Saclay, CNRS/IN2P3, IJCLab, Orsay, France
15Laboratoire Leprince-Ringuet, CNRS/IN2P3, Ecole Polytechnique, Institut Polytechnique de Paris, Palaiseau, France
16Laboratoire de Physique Nucléaire et de Hautes Énergies (LPNHE), Sorbonne Université, CNRS/IN2P3, Paris, France
17I. Physikalisches Institut, RWTH Aachen University, Aachen, Germany
18Universität Bonn - Helmholtz-Institut für Strahlen und Kernphysik, Bonn, Germany
19Fakultät Physik, Technische Universität Dortmund, Dortmund, Germany
20Physikalisches Institut, Albert-Ludwigs-Universität Freiburg, Freiburg, Germany
21Max-Planck-Institut für Kernphysik (MPIK), Heidelberg, Germany
22Physikalisches Institut, Ruprecht-Karls-Universität Heidelberg, Heidelberg, Germany
23School of Physics, University College Dublin, Dublin, Ireland
24INFN Sezione di Bari, Bari, Italy
25INFN Sezione di Bologna, Bologna, Italy
26INFN Sezione di Ferrara, Ferrara, Italy
27INFN Sezione di Firenze, Firenze, Italy
28INFN Laboratori Nazionali di Frascati, Frascati, Italy
29INFN Sezione di Genova, Genova, Italy
30INFN Sezione di Milano, Milano, Italy
31INFN Sezione di Milano-Bicocca, Milano, Italy
32INFN Sezione di Cagliari, Monserrato, Italy
33INFN Sezione di Padova, Padova, Italy
34INFN Sezione di Perugia, Perugia, Italy
35INFN Sezione di Pisa, Pisa, Italy
36INFN Sezione di Roma La Sapienza, Roma, Italy
37INFN Sezione di Roma Tor Vergata, Roma, Italy
38Nikhef National Institute for Subatomic Physics, Amsterdam, Netherlands
39Nikhef National Institute for Subatomic Physics and VU University Amsterdam, Amsterdam, Netherlands
40AGH - University of Krakow, Faculty of Physics and Applied Computer Science, Kraków, Poland
41Henryk Niewodniczanski Institute of Nuclear Physics Polish Academy of Sciences, Kraków, Poland
42National Center for Nuclear Research (NCBJ), Warsaw, Poland
43Horia Hulubei National Institute of Physics and Nuclear Engineering, Bucharest-Magurele, Romania
44Authors affiliated with an institute formerly covered by a cooperation agreement with CERN.
45Universidade da Coruña, A Coruña, Spain
46ICCUB, Universitat de Barcelona, Barcelona, Spain
47La Salle, Universitat Ramon Llull, Barcelona, Spain
48Instituto Galego de Física de Altas Enerxías (IGFAE), Universidade de Santiago de Compostela, Santiago de Compostela, Spain
49Instituto de Fisica Corpuscular, Centro Mixto Universidad de Valencia - CSIC, Valencia, Spain
50European Organization for Nuclear Research (CERN), Geneva, Switzerland
51Institute of Physics, Ecole Polytechnique Fédérale de Lausanne (EPFL), Lausanne, Switzerland
52Physik-Institut, Universität Zürich, Zürich, Switzerland
53NSC Kharkiv Institute of Physics and Technology (NSC KIPT), Kharkiv, Ukraine
54Institute for Nuclear Research of the National Academy of Sciences (KINR), Kyiv, Ukraine
55School of Physics and Astronomy, University of Birmingham, Birmingham, United Kingdom
56H.H. Wills Physics Laboratory, University of Bristol, Bristol, United Kingdom
57Cavendish Laboratory, University of Cambridge, Cambridge, United Kingdom
58Department of Physics, University of Warwick, Coventry, United Kingdom
59STFC Rutherford Appleton Laboratory, Didcot, United Kingdom
60School of Physics and Astronomy, University of Edinburgh, Edinburgh, United Kingdom
61School of Physics and Astronomy, University of Glasgow, Glasgow, United Kingdom
62Oliver Lodge Laboratory, University of Liverpool, Liverpool, United Kingdom
63Imperial College London, London, United Kingdom
64Department of Physics and Astronomy, University of Manchester, Manchester, United Kingdom
65Department of Physics, University of Oxford, Oxford, United Kingdom
66Massachusetts Institute of Technology, Cambridge, MA, United States
67University of Cincinnati, Cincinnati, OH, United States
68University of Maryland, College Park, MD, United States
69Los Alamos National Laboratory (LANL), Los Alamos, NM, United States
70Syracuse University, Syracuse, NY, United States
71Pontifícia Universidade Católica do Rio de Janeiro (PUC-Rio), Rio de Janeiro, Brazil, associated to 3
72Universidad Andres Bello, Santiago, Chile, associated to 52
73School of Physics and Electronics, Hunan University, Changsha City, China, associated to 8
74State Key Laboratory of Nuclear Physics and Technology, South China Normal University, Guangzhou, China, associated to 4
75Lanzhou University, Lanzhou, China, associated to 5
76School of Physics and Technology, Wuhan University, Wuhan, China, associated to 4
77Henan Normal University, Xinxiang, China, associated to 8
78Departamento de Fisica , Universidad Nacional de Colombia, Bogota, Colombia, associated to 16
79Institute of Physics of the Czech Academy of Sciences, Prague, Czech Republic, associated to 64
80Ruhr Universitaet Bochum, Fakultaet f. Physik und Astronomie, Bochum, Germany, associated to 19
81Eotvos Lorand University, Budapest, Hungary, associated to 50
82Faculty of Physics, Vilnius University, Vilnius, Lithuania, associated to 20
83Institute of Physics and Technology, Ulan Bator, Mongolia, associated to 5
84Van Swinderen Institute, University of Groningen, Groningen, Netherlands, associated to 38
85Universiteit Maastricht, Maastricht, Netherlands, associated to 38
86Universidad de Ingeniería y Tecnología (UTEC), Lima, Peru, associated to 66
87Tadeusz Kosciuszko Cracow University of Technology, Cracow, Poland, associated to 41
88Department of Physics and Astronomy, Uppsala University, Uppsala, Sweden, associated to 61
89Taras Schevchenko University of Kyiv, Faculty of Physics, Kyiv, Ukraine, associated to 14
90University of Michigan, Ann Arbor, MI, United States, associated to 70
91Indiana University, Bloomington, United States, associated to 69
92Ohio State University, Columbus, United States, associated to 69

aUniversidade Estadual de Campinas (UNICAMP), Campinas, Brazil
bDepartment of Physics and Astronomy, University of Victoria, Victoria, Canada
cCenter for High Energy Physics, Tsinghua University, Beijing, China
dHangzhou Institute for Advanced Study, UCAS, Hangzhou, China
eLIP6, Sorbonne Université, Paris, France
fLamarr Institute for Machine Learning and Artificial Intelligence, Dortmund, Germany
gUniversidad Nacional Autónoma de Honduras, Tegucigalpa, Honduras
hUniversità di Bari, Bari, Italy
iUniversità di Bergamo, Bergamo, Italy
jUniversità di Bologna, Bologna, Italy
kUniversità di Cagliari, Cagliari, Italy
lUniversità di Ferrara, Ferrara, Italy
mUniversità di Genova, Genova, Italy
nUniversità degli Studi di Milano, Milano, Italy
oUniversità degli Studi di Milano-Bicocca, Milano, Italy
pUniversità di Modena e Reggio Emilia, Modena, Italy
qUniversità di Padova, Padova, Italy
rUniversità di Perugia, Perugia, Italy
sScuola Normale Superiore, Pisa, Italy
tUniversità di Pisa, Pisa, Italy
uUniversità di Siena, Siena, Italy
vUniversità di Urbino, Urbino, Italy
wUniversidad de Alcalá, Alcalá de Henares , Spain

†Deceased

  
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