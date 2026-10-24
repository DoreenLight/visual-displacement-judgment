<!-- source: https://doi.org/10.1109/icorr66766.2025.11063171 -->
<!-- title: Design and Evaluation of a Single-Sided Mobility Assistive Exoskeleton (SMAEXO) for Hemiplegia -->
<!-- fetched: 2026-10-01T03:49:39Z -->
<!-- via: local_paper_fulltext(pdftotext -layout) -->
<!-- note: 用户于 2026-10-01 从学校数据库下载的原文 PDF：papers/Design and Evaluation of a Single-Sided Mobility Assistive Exoskeleton                                                           for Hemiplegia .pdf -->

May 12-16, 2025. Chicago, IL, USA

                                                                                                                                                    Design and Evaluation of a Single-Sided Mobility Assistive Exoskeleton
                                                                                                                                                                         (SMAEXO) for Hemiplegia
                                                                                                                                                                 Xiaoyue Yan1, 2 , Wei Yun Yau3 , Isaac Okumura Tan2 , Longbin Zhang2, 4 , Youlong Wang1, 2 ,
                                                                                                                                                                                    Oliver Roberts2 , Seng Kwee Wee5 , Wei Tech Ang1, 2

                                                                                                                                                       Abstract— This study presents the Single-Sided Mobility                         reliance on position control make them unsuitable for stroke
                                                                                                                                                    Assistive Exoskeleton (SMAEXO), a novel, single-sided, multi-                      survivors, who need responsive, tailored support [8].
                                                                                                                                                    joint device designed to enhance mobility and rehabilitation                          For hemiplegic patients, unilateral exoskeletons offer a
                                                                                                                                                    for individuals with hemiplegia. SMAEXO provides active
                                                                                                                                                    assistance at the hip and knee, combined with a passive,                           simpler, lighter alternative. Unlike bilateral devices, which
                                                                                                                                                                                                                                       restrict natural movement and disrupt gait symmetry [9], uni-

                                                                                                                                                    flexible ankle joint to prevent foot drop, ensuring comprehensive
                                                                                                                                                    support throughout the gait cycle. Its innovative design opti-                     lateral exoskeletons assist only the affected limb, enhancing
                                                                                                                                                    mizes reaction force support and weight distribution, promoting                    mobility with less burden on the wearer.
                                                                                                                                                    balance and reducing strain on the affected side. Additionally,                       However, the unilateral approach introduces its own chal-
                                                                                                                                                    its modular, customizable design accommodates diverse body
                                                                                                                                                    dimensions and biomechanical needs, while safety features such                     lenges. The asymmetric structure of these devices can make
                                                                                                                                                    as self-locking mechanisms, cushions, bumpers, and elastic                         it difficult to maintain balance, and the limited reaction
                                                                                                                                                    components enhance user safety and comfort. The SMAEXO’s                           force support hampers the generation of sufficient torque to
                                                                                                                                                    usability, comfort, safety, and assistance were assessed through                   adequately stabilize and propel the affected limb.
                                                                                                                                                    trials with eight able-bodied participants, including physiother-                     Furthermore, many unilateral exoskeletons assist only one
                                                                                                                                                    apists simulating stroke gait patterns. Participants reported
                                                                                                                                                    the device as comfortable, safe, and effective, highlighting its                   joint, limiting their effectiveness for stroke-related multi-joint
                                                                                                                                                    benefits during level and slope walking and obstacle negotiation.                  impairments. Knee devices like the University of Michigan’s
                                                                                                                                                    Notable feedback included the need for enhanced ventilation,                       passive exoskeleton [10], the University of New York’s active
                                                                                                                                                    reduced device weight, and improved ease of donning and                            exoskeleton [11], and the REFLEX system [12] aid the knee
                                                                                                                                                    doffing. The results demonstrate SMAEXO’s potential to de-                         for stance support and swing clearance. Similarly, ankle
                                                                                                                                                    liver comprehensive multi-joint assistance for individuals with
                                                                                                                                                    hemiplegia. This work contributes to advancing rehabilitation                      exoskeletons such as the ReStore Exo-Suit [13] and Northern
                                                                                                                                                    and assistive technologies for post-stroke individuals, enabling                   Arizona University’s device [14] improve plantarflexion and
                                                                                                                                                    more effective and personalized mobility support.                                  dorsiflexion but provide only partial support, neglecting
                                                                                                                                                                                                                                       impairments in other joints. Multi-joint unilateral devices
                                                                                                                                                                             I. I NTRODUCTION                                          such as HAL [15], AGoRA [16] and LitteStepper [17]
                                                                                                                                                                                                                                       attempt to assist multiple joints simultaneously but tend to be
                                                                                                                                                       Stroke, a leading cause of disability, often results in
                                                                                                                                                                                                                                       heavier, less user-friendly, and more challenging to integrate
                                                                                                                                                    hemiparesis, affecting 65% of survivors with impairing gait
                                                                                                                                                                                                                                       seamlessly with the user’s intent.
                                                                                                                                                    [1]. Common issues include asymmetry, poor balance, and
                                                                                                                                                                                                                                          In practice, existing unilateral exoskeletons still fall short
                                                                                                                                                    foot drop [2], requiring intensive rehabilitation to restore
                                                                                                                                                                                                                                       of delivering seamless, multi-joint assistance while maintain-
                                                                                                                                                    mobility. Traditional therapy is labor-intensive, requiring
                                                                                                                                                                                                                                       ing usability, comfort, and safety. These limitations under-
                                                                                                                                                    two physiotherapists per session, limiting scalability [3].
                                                                                                                                                                                                                                       score the need for a new generation of unilateral exoskeletons
                                                                                                                                                    Exoskeletons offer a more efficient solution, providing con-
                                                                                                                                                                                                                                       that can provide comprehensive, user-centered support for
                                                                                                                                                    sistent, automated support to enhance rehabilitation while
                                                                                                                                                                                                                                       hemiparetic individuals.
                                                                                                                                                    reducing therapist workload [4].
                                                                                                                                                                                                                                          To address these gaps, this work introduces the Single-
                                                                                                                                                       Most lower-limb exoskeletons are bilateral, providing full                      Sided Mobility Assistive Exoskeleton (SMAEXO), a novel
                                                                                                                                                    mobility support by targeting the hip, knee, and ankle.                            multi-joint device specifically engineered for individuals
                                                                                                                                                    Devices like Ekso Indego [5], ReWalk [6], and Rex [7]                              with hemiparesis. SMAEXO incorporates innovative features
                                                                                                                                                    assist users with severe motor loss, such as spinal cord                           designed to optimize reaction force management and weight
                                                                                                                                                    injury patients. However, their bulky design, high cost, and                       distribution, while ensuring that assistance at the hip, knee,
                                                                                                                                                                                                                                       and ankle joints is both effective and intuitive. The key
                                                                                                                                                       This research is supported by ASTAR under the National Robotics
                                                                                                                                                    Programme (NRP) BAU grant, Project - Assistive Robotics Programme                  contributions of this work include:
                                                                                                                                                    (Award No: M22NBK0074).                                                               1) Development of the SMAEXO enables integrated assis-
                                                                                                                                                       1 School of Mechanical and Aerospace Engineering, Nanyang Techno-
                                                                                                                                                                                                                                       tance across the hip, knee, and ankle joints, providing com-
                                                                                                                                                    logical University, Singapore.
                                                                                                                                                       2 Rehabilitation Research Institute of Singapore, Nanyang Technological         prehensive support to the affected limb. Optimized weight
                                                                                                                                                    University, Singapore.                                                             distribution and reaction force management strategies ensure
                                                                                                                                                       3 Institute for Infocomm Research, A*STAR, Singapore.
                                                                                                                                                       4 School of Robotics, Hunan University, Changsha, China.
                                                                                                                                                                                                                                       that the device can effectively stabilize and support the user
                                                                                                                                                       5 Tan Tock Seng Hospital, Singapore.                                            without compromising comfort, safety or ease of movement.
                                                                                                                                                       Corresponding authors’ e-mails: {xiaoyue.yan, wtang}@ntu.edu.sg                    2) The performance of SMAEXO is validated through

                                                                                                                                                    979-8-3503-8068-2/25/$31.00 ©2025 IEEE                                       597
                                                                                                                                                      Authorized licensed use limited to: South-Central Minzu University. Downloaded on October 01,2026 at 03:37:54 UTC from IEEE Xplore. Restrictions apply.

                                                                                 the user’s biomechanical capabilities. The soft limiters, im-
                                                                                 plemented as joint buffers, absorb impacts when the joint
                                               Battery& Control                  approaches the hard limiters (Fig. 2 (b)). The joint buffers
                  Vest Harness                     Components
                                                                                 can be easily replaced to accommodate users with limited
                                          Active Hip                             ROM. Additionally, all the leg bars are equipped with cuffs
                 Back Module
                                                                                 and cushions, as shown in Fig. 2 (d), these cuffs can be easily
                 Unaffected-side                                                 opened for quick donning and doffing and the major semi-
                                                     Passive
                    Module                                                       axis, minor semi-axis of the cuffs can be easily adjusted to
                                                Active Hip
                                                 Knee                            accommodate users with varying leg circumferences.
                    Affected-side                                                  To further enhance adaptability and alignment, the
                       Module
                                                Passive                          SMAEXO features the corresponding adjustable mechanisms
                                                 Ankle
                                                                                 to allow tool-free adjustments for waist width, upper and
                                                                                 lower leg bar lengths, and foot size, accommodating users
                                                                                 of various body dimensions (Fig. 2 and Fig. 3 (a)). All
                                                                                 adjustable mechanisms are equipped with reset springs to
Fig. 1. Architecture of the single-sided mobility assistive exoskeleton
                                                                                 ensure the locked status for safety.
(SMAEXO). The red arrows indicate joint movements.                                 The main specifications of the SMAEXO system are
                                                                                 summarized in Table I.
                                                                                                                TABLE I
a user study involving eight participants. This evaluation                                         S PECIFICATIONS OF THE SMAEXO.
focuses on usability, comfort, safety, and effectiveness, of-
fering valuable insights into SMAEXO’s practicality and its                        Items                 Specifications
potential to enhance the rehabilitation process for stroke                         Motion Range (°)      Hip: (-20, 125); Knee: (0, 135); Ankle: (-40, 24)
                                                                                                         Upper Limb: 400–500 (Long), 300–400 (Short);
survivors.                                                                         Length Range (mm)     Lower Limb: 380–480 (Long), 280–380 (Short);
   By offering a thoughtfully engineered solution that meets                                             Hip Width: 335–435; Feet: 250–305
                                                                                   Hip and               Rated Torque: 12Nm;
the unique needs of hemiparetic individuals, the SMAEXO                            Knee Motors           Max Torque: 37Nm; Ratio: 1:6; Weight: 0.759kg
has the potential to transform stroke rehabilitation.                              Battery               Lithium Battery Pack, 24V 6Ah, Weight: 0.82kg
                                                                                                         Total: 7.5;
                     II. S YSTEM OVERVIEW                                          Weight (kg)           Affected-side Module: 3.3;
                                                                                                         Unaffected-side Module: 0.3
A. Architecture of SMAEXO                                                                                Back Module: 2.1
                                                                                                         Vest Harness: 1.8
   SMAEXO was developed through close collaboration
with physiotherapists specializing in stroke rehabilitation and
lower-limb exoskeletons. This consultative process ensured                       B. Efficient Reaction Force Management and Weight Distri-
that the system aligns with the unique needs of individuals                      bution
with hemiplegia, emphasizing usability, safety, comfort and                         Effective reaction force management is key to minimizing
comprehensive assistance. The device features a modular                          discomfort and enhancing usability. Double-sided exoskele-
design comprising four primary components (Fig. 1):                              tons like Ekso Indego [5], ReWalk [6], and Rex [7] achieve
   • Affected-side module: Incorporates active hip and knee                      this through heavier self-weight and extensive user contact,
     joints powered by quasi-direct drive (QDD) motors,                          enabling higher assistive torque but increasing bulk and
     which combine high-torque motors with a low gear ratio                      reducing mobility. In contrast, lightweight exoskeletons face
     transmission (1:6) to deliver high bandwidth and back-                      torque limitations due to weaker reaction force support. For
     drivability. Additionally, a passive ankle joint paired                     example, Honda’s SMA device provides 4 Nm of hip torque
     with a footplate is involved to prevent foot drop.                          [18], while Samsung’s GEMS, with a hip brace for additional
   • Unaffected-side module: Features a passive hip                              support, reaches 12 Nm [19]. Unilateral devices like AGoRA
     joint that facilitates hip flexion/extension and abduc-                     improve torque and balance with bilateral configurations but
     tion/adduction, ensuring natural movement on the un-                        increase mass to 17 kg [16].
     affected side.                                                                 The SMAEXO addresses these challenges by integrating
   • Back module: Connects all modules and houses the                            innovative features for efficient reaction force management
     control components and battery, strategically placing                       and weight distribution. Through a combination of an er-
     the battery on the unaffected side to optimize weight                       gonomic vest harness and a flexible ankle joint coupled with
     distribution and balance.                                                   a footplate, SMAEXO ensures effective stabilization and
   • Vest harness: Distributes device weight and reaction                        comprehensive assistance while maintaining a lightweight
     force evenly across the user’s core, enhancing comfort                      and user-friendly profile.
     and stability.                                                                 • Ergonomic Vest Harness
   For safety and comfort, all joints incorporate both hard                         As depicted in Fig. 3 (a), the SMAEXO vest harness is
and soft limiters to restrict the range of motion (ROM) within                   designed to conform closely to the wearer’s body. It consists

979-8-3503-8068-2/25/$31.00 ©2025 IEEE                                     598
  Authorized licensed use limited to: South-Central Minzu University. Downloaded on October 01,2026 at 03:37:54 UTC from IEEE Xplore. Restrictions apply.

                                                   Waist Bar Ⅰ                                                                              Reset
                                                                                                                                            Spring
                                                                                                                                                                  Leg Bar
                                                                                                                                                                    Ⅰ
                                                                                                                                                Latch
                                                                                              Actuator
                                                                                                                                                                  Leg Bar
                                                                                              Bracket
                              Upper Leg                                                                                                                             Ⅱ
                               Bar Ⅰ                              Cuff-Adjustable
                                                                   Mechanism                                                          Soft Limits         (c)
                                              Upper Leg
                                               BarⅡ                                                                                  Hard Limits
                                                                                           Joint                                                        Cushion
                               Length-Adjustable                                          Buffers                                    Joint Output
                                  Mechanism                                                                                              Rod              o          Cuff
                                                                       Ankle Cuff
                                      Lower Leg Bar Ⅱ                        Footdrop
                                                                         Prevention Strap
                                      Angle-Adjustable
                                           Mechanism                                                                      ROM for Hard Limits
                                              Footplate                                                     ROM for Soft Limits
                                                                (a)                                               (b)                                      (d)

Fig. 2. The affected-side module of the SMAEXO. (a) Module with length adjustments (red arrows). (b) Joint with limited ROM, featuring replaceable
buffers to absorb impact. (c) Length-adjustable mechanism with a reset spring for secure locking. (d) Cuff-adjustable mechanism with linear (a, b) and
rotational adjustments for leg circumference and easy donning (red arrows).

                                                                                                                    of a vest integrated with a harness and belt, featuring an
                                                                                                                    elastic spine made from flexible materials such as spring
  Harness
                                                                                                                    steel, which naturally adapts to the wearer’s spinal curvature.
Elastic                                                                 Airbag
Spine                                                                  Bumper             Passive Hip               To enhance fit and contact, an adjustable airbag bump is
     Vest                                                                                  Encoder
Waist Bar Ⅰ
                                                                                                                    included, allowing compatibility with diverse body shapes
                                                                      Waist Bar Ⅰ         Limb Bar                  and improving overall user comfort. This configuration effi-
Waist Bar
Connector                                                                                                           ciently utilizes the wearer’s waist, chest, abdominal, and back
    Torso IMU                                                                                                       muscles to stabilize reaction forces, reducing discomfort and
                                      Belt                                                   Cuff
            Waist Bar Ⅱ                                                                                             enhancing mobility.
                                                    Passive Hip
                                      (a)                                                     (b)                      • Flexible Ankle Joint

                                                                                                                       Unlike most exoskeletons, such as Knee-Ankle-Foot Or-
                                                                                                                    thoses, ReWalk™ [6], and HAL [15], which use fixed ankle
Fig. 3. The SMAEXO vest harness, back, and unaffected-side modules.
(a) The back module, paired with the vest harness, includes an IMU for                                              joints and rigid footplates to prevent foot drop and transfer
torso angle monitoring. (b) The unaffected-side module features a passive                                           weight to the ground, SMAEXO features a versatile ankle
hip with an encoder to measure sagittal hip angles. Red arrows indicate                                             joint design that prioritizes both comfort and a natural
linear adjustments for waist width and hip movement.
                                                                                                                    gait. This innovative design adapts to individual user needs,
                                                                                                                    facilitating improved mobility.
                                                                                                                       As depicted in Fig. 4 and Fig. 5, the SMAEXO ankle joint
                                                                                                                    incorporates the following features: (1) An angle-adjustable
        Lower Leg Bar Ⅱ                                                                                             mechanism, which can either be allowed to rotate within a
                                                    Ankle Joint                     Front Sole Ⅰ
                                                    Output Rod                                                      controlled range or fixed at a specific angle. (2) A flexible
              Ankle Cuff                                                                     Cable Channels
                                 Footdrop                                                                           footplate with force sensors at the heel and forefoot to
                              Prevention Strap          Straps
                                                                                                                    detect gait events. (3) Effective weight transfer to the ground
                                            Sole
                                                                                                                    during stance without impeding force sensor functionality.
    Ankle Pivot                                        Heel
                                                       Sole Ⅰ
                                                                                                                    (4) Integration with an elastic strap and ankle cuff to prevent
     Angle-                                                                                  Front Sole Ⅱ
 adjustable                                                                                                         foot drop during swing. (5) Adjustable footplate length
Mechanism                                              Force
                                                      Sensors         Heel Sole Ⅱ                                   to accommodate various foot sizes. (6) Construction using
 Ankle Joint
 Output Rod                  Straps                                                                                 elastic and durable materials such as polyurethane (PU),
                       (a)                                                          (b)                             rubber, and ethylene-vinyl acetate (EVA) for natural ground
                                                                                                                    contact and durability. This configuration facilitates a natural
                                                                                                                    gait while offering personalized adjustments to align with
Fig. 4. The ankle joint and footplate of the SMAEXO. (a) The ankle
joint connects to the footplate via an output rod and an elastic foot-drop                                          individual walking patterns and assistance needs, enhancing
prevention strap, which acts like a spring with adjustable tension for user                                         both comfort and mobility.
adaptability. (b) The footplate features adjustable length via a sliding front                                         With its advancements in reaction force management and
sole I over heel sole I, accommodating various foot sizes. Embedded force
sensors in front sole I and heel sole I are protected by an additional layer                                        weight support, the SMAEXO achieves a maximum assistive
(front sole II and heel sole II). Integrated cable channels facilitate wiring.                                      torque of 25 Nm. This capability addresses the high torque
                                                                                                                    demands of hemiplegia rehabilitation while ensuring user

979-8-3503-8068-2/25/$31.00 ©2025 IEEE                                                                        599
   Authorized licensed use limited to: South-Central Minzu University. Downloaded on October 01,2026 at 03:37:54 UTC from IEEE Xplore. Restrictions apply.

                                                   Lower Leg
                                                     Bar Ⅱ                                          Latch                  Slewing                       Lower Leg
                                                                                                   Housing                  Block                          Bar Ⅱ
                        Latch Cover                Ankle Pivot                          Sliding Guide                                                       Ankle Joint
                                                                                           Groove Ⅰ                         Sliding Guide                   Output Rod
                                                     Latch Housing                                                            Groove Ⅱ
                                                                                                                                                             Slewing Block
                                                                                                 Limit
                                                                                                Blocks                                                               Latch
                                                           Reset Springs                                                                                           Housing
                         Latch                                                              Spring-loaded                    Reset Springs                          Limit Blocks
                         Holes                          Spring-loaded                           Latch
                                                            Latch                                                                                                          Pivot
                                                        Limit Blocks                                 Slide                    Slide
                                                                                                    Limit Ⅱ                  Limit Ⅰ
                      Ankle Joint                                                                                                                 Spring-loaded             Latch
                      Output Rod                                                                                                                      Latch                 Cover
                                                    Ankle ROM

                                       (a)                                                                           (b)                                          (c)

Fig. 5. The angle-adjustable mechanism of ankle joint. (a) Locked with limited movement. (b) Fixed at a specific angle. (c) Exploded view. The mechanism
enables controlled ankle rotation or fixed positioning using a latch housing, sliding guide grooves, a slewing block, spring-loaded limit blocks, a pivot,
and a latch cover. The limit blocks rotate along guide groove II and lock when the spring-loaded latch engages a latch hole. Unlocking releases the latch,
restoring movement. The slewing block rotates within guide groove I and locks the ankle at a set angle when contacting both limit blocks.

   Plantar Pressure         Gait        Gait                Filter
                           Event
  Force Sensors           Detection    phase        •    EWMA
                                                    Exoskeleton
   Joint Kinematics                             Transparency Control
                                                     Algorithm       𝜏𝜏exo_𝑔𝑔𝑔𝑔
  Encoders, IMUs                                                                            +
                                               • Exo GC                                          SMAEXO
   Exo Parameters                                                                                 Joints
                                                                                            +
                                                   User Assistance
                                                   Control Algorithm        𝜏𝜏user_𝑎𝑎𝑎𝑎𝑎𝑎
  User Parameters
                                               •    Partial User Mobility
                                                    Assistance

                                                                                                                                             Fig. 7.   Experiment setup.

              Fig. 6.       Control block diagram of the SMAEXO.

                                                                                                                    simulated stroke gait patterns—including foot drop, stiff
comfort, providing an effective and user-friendly solution for                                                      knee, circumduction, and weak knee—based on their ex-
mobility assistance.                                                                                                perience to evaluate the exoskeleton’s effectiveness from
                                                                                                                    a patient’s perspective. Demographic and anthropometric
C. Control Scheme                                                                                                   details of the participants are provided in Table II. All
   The SMAEXO utilizes a multi-level control strategy with                                                          participants gave written informed consent after being briefed
a reliable and cost-effective sensor system to enhance its                                                          on the study’s objectives and potential risks. The study
functionality. This system incorporates a torso IMU and                                                             protocol received approval and oversight from the Nanyang
encoders embedded in the hip and knee joints to provide                                                             Technological University Institutional Review Board.
accurate joint kinematic measurements. Foot pressure mea-                                                              During the experiment, participants could take rest breaks,
surements from force sensors, combined with kinematic data,                                                         stop, or withdraw from testing at any time. The assessment
are employed to identify gait phases during various daily                                                           consisted of two sessions (with and without the exoskeleton),
activities, including sit-to-stand transitions and walking on                                                       each comprising three tasks. Participants repeated each task
flat surfaces, ramps, or stairs. The control system lever-                                                          at least three times:
ages this gait phase information to implement exoskeleton
transparency(e.g., Gravity Compensation) and partial user                                                             • 10-meter level walking;
mobility assistance for optimized support and a better user                                                           • 4-meter slope with 8 degrees inclination up and down;
experience. Additionally, exoskeleton and user parameters                                                             • Stepping over a 10 cm high, 12.5 cm wide obstacle.
are tailored to meet individual assistance needs, while filters
like the Exponentially Weighted Moving Average (EWMA)                                                                  In the second session, participants were familiarized with
ensure a smooth torque profile throughout the entire gait                                                           the exoskeleton for 30 minutes before testing. Investigators
cycle (Fig. 6).                                                                                                     guided them on exoskeleton use and ensured acclimatization.
                                                                                                                    Participants were instructed to report any discomfort or pain,
                                    III. E XPERIMENTS                                                               which was documented and addressed.
   To evaluate the usability, comfort, safety, and assistance                                                          At the study’s conclusion, participants completed a ques-
provided by the SMAEXO, eight able-bodied participants                                                              tionnaire assessing the exoskeleton’s usability, comfort,
were recruited, including two physiotherapists specializing                                                         safety and assistance. All questions are rated 1 to 5, with
in lower-limb rehabilitation. During testing, physiotherapists                                                      1 as “strongly disagree” and 5 as “strongly agree”.

979-8-3503-8068-2/25/$31.00 ©2025 IEEE                                                                        600
  Authorized licensed use limited to: South-Central Minzu University. Downloaded on October 01,2026 at 03:37:54 UTC from IEEE Xplore. Restrictions apply.

                                                 Strongly Disagree                     Disagree                  Neutral         Agree      Strongly Agree

                                       It is easy to wear the device.

                                         I find the device not heavy
                                                     when wearing it.
                           I did not experience any discomfort,
                              such as pain, itching, irritation, or
                        unusual warmth when using the device.
                                        I felt safe using the device,
                                   with no perceived risk of falling.

                              I felt less effort during level walking
                                              when using the device.

                             I felt less effort during slope walking
                                             when using the device.

                        I felt less effort during obstacle crossing
                                             when using the device.

                                      I felt the overall assistance
                              provided by the device is adequate.

                                                                            0                      20              40             60        80               100
                                                                                                                  Percentage of Responses

                   Fig. 8.      User questionnaire responses to the SMAEXO during level and slope walking, as well as obstacle crossing tasks.

                         TABLE II
                                                                                                              ferred weight to the ground, reducing the burden on the knee
  D EMOGRAPHIC AND ANTHROPOMETRIC DETAILS OF PARTICIPANTS .
                                                                                                              joint. However, physiotherapists raised concerns regarding
             Participants       Age        Sex      Height (cm)            Mass (kg)                          hemiplegic users, as the unaffected leg bears the device’s
             P1                31-40        F           155                  54.5                             weight during the swing phase. This could potentially in-
             P2                21-30        M           171                   70
             P3                31-40        F           163                   55                              crease stress on the unaffected knee, particularly concerning
             P4                31-40        M           183                   76                              for elderly post-stroke individuals who are more susceptible
             P5                21-30        M           183                   73                              to knee issues, such as osteoarthritis. Further reducing the
             P6                21-30        M           181                   85
             P7                21-30        M           176                   85                              exoskeleton’s weight is recommended to alleviate this strain.
             P8                41-50        M           169                   71                                 The device was reported to be comfortable, with no
                                                                                                              instances of pain, itching, or irritation. The adjustable mecha-
                                          TABLE III
                                                                                                              nisms enabled proper adaptation to various body dimensions,
                        S PECIFICATIONS OF THE SMAEXO.
                                                                                                              ensuring a snug fit that effectively engaged the user’s core
 Devices       HAL (Cyberdyne)                 LiteStepper (Angelexo)        SMAEXO                           muscles for optimal reaction force management and weight
 Control
 Strategy
               Model-based control
               (amplify muscle activities)
                                               Replicate gait trajectory
                                               of intact limb & Crutch
                                                                             Hybrid Model-based
                                                                             control
                                                                                                              distribution. However, two participants experienced slight
               or Predefined gait trajectory   handle to initiate walk                                        perspiration on their backs after wearing the device for
 Actuators     2 Brushless DC motors           2 Brushless DC motors         2 QDD motors
 Sensors       Kinematic Sensors               Kinematic Sensors             Kinematic Sensors (4)            more than an hour, likely due to the vest and airbag bump.
 Used          Sole Pressure Sensors           Sole Pressure Sensors         Sole Pressure Sensors
               EMGs                            Cane Pressure Sensor                                           Incorporating breathable materials into these components
 Footplates
 Type
               Customized shoes with
               rigid shoes insole
                                               Mechanical footplates         Flexible footplates              could enhance comfort during prolonged use.
 Weight                                        Vest (core)                   Vest Harness (core)
 Support       Rigid insole(ground)            Footplates (ground)           Footplates (ground)
                                                                                                                 Regarding safety, participants indicated no concerns about
 Weight        9 kg                            16.9 kg                       7.5 kg                           falling or other safety issues during the trials, indicating
                                                                                                              that the SMAEXO’s design effectively solved the weight
                                                                                                              distribution and balance challenges. However, one participant
                      IV. R ESULTS AND D ISCUSSION                                                            highlighted the potential risk of the footplate contacting
   As shown in Fig. 8, participants appreciated SMAEXO’s                                                      obstacles or stairs if the straps were not tightly secured. This
modular design and tool-free adjustments for easy don-                                                        emphasizes the importance of ensuring that all straps are
ning and doffing. However, its unbalanced structure made                                                      properly adjusted and secured to maintain safety during use.
maintaining a neutral position challenging, especially when                                                      Participants reported the SMAEXO provided adequate
seated. Physiotherapists noted that post-stroke users often re-                                               assistance during level and slope walking, as well as obstacle
quire two assistants for stabilization. Incorporating a support                                               negotiation, reducing their effort. This shows the SMAEXO
frame could address this issue by providing stability during                                                  offered comprehensive joint support throughout the gait cycle
the donning process.                                                                                          by combining active hip and knee assistance with a passive,
   Participants considered the device’s weight acceptable.                                                    flexible ankle joint.
During the stance phase, the SMAEXO effectively trans-                                                           Compared to double-sided exoskeletons, the SMAEXO

979-8-3503-8068-2/25/$31.00 ©2025 IEEE                                                                  601
  Authorized licensed use limited to: South-Central Minzu University. Downloaded on October 01,2026 at 03:37:54 UTC from IEEE Xplore. Restrictions apply.

demonstrated advantages in weight reduction and freedom of                                                     R EFERENCES
movement, while offering more comprehensive support than
single-joint devices. Table III compares the main features of                     [1] V. L. Feigin, M. Brainin, B. Norrving, S. Martins, R. L. Sacco,
                                                                                      W. Hacke, M. Fisher, J. Pandian, and P. Lindsay, “World stroke orga-
our device with commercial unilateral exoskeletons, HAL                               nization (wso): global stroke fact sheet 2022,” International Journal
and LiteStepper. SMAEXO offers advantages in compact                                  of Stroke, vol. 17, no. 1, pp. 18–29, 2022.
design, lighter weight, an intuitive and cost-effective sensor                    [2] Y. Li, T. Fan, Q. Qi, J. Wang, H. Qiu, L. Zhang, X. Wu, J. Ye, G. Chen,
system, improved force management, and more flexible                                  J. Long et al., “Efficacy of a novel exoskeletal robot for locomotor
                                                                                      rehabilitation in stroke patients: A multi-center, non-inferiority, ran-
footplates for natural gait. However, challenges remain in                            domized controlled trial,” Frontiers in aging neuroscience, vol. 13, p.
minimizing the overall weight and improving the ease of                               706569, 2021.
donning and doffing, particularly due to the lack of a stable                     [3] G. Chen, C. K. Chan, Z. Guo, and H. Yu, “A review of lower
                                                                                      extremity assistive robotic exoskeletons in rehabilitation therapy,”
sitting pose without external support. Future iterations should                       Critical Reviews™ in Biomedical Engineering, vol. 41, no. 4-5, 2013.
focus on incorporating a dedicated support mechanism to                           [4] E. Read, C. Woolsey, C. A. McGibbon, and C. O’Connell, “Physio-
address this limitation and further enhance usability.                                therapists’ experiences using the ekso bionic exoskeleton with patients
                                                                                      in a neurological rehabilitation hospital: A qualitative study,” Rehabil-
   Compared to double-sided exoskeletons, SMAEXO offers                               itation research and practice, vol. 2020, no. 1, p. 2939573, 2020.
reduced weight and greater freedom of movement while pro-                         [5] “Ekso indego,” Ekso Bionics. [Online]. Available: https://eksobionics.
viding more comprehensive support than single-joint devices.                          com/ekso-indego-therapy/
Table III compares its key features with commercial unilat-                       [6] “Rewalk personal exoskeleton,” Lifeward. [Online]. Available:
                                                                                      https://golifeward.com/products/rewalkpersonal-exoskeleton/
eral exoskeletons, HAL [20] and LiteStepper [17]. SMAEXO                          [7] “Rex,” Rex Bionics. [Online]. Available: https://www.rexbionics.com/
excels in compact design, lighter weight, an intuitive and                            product-information/
cost-effective sensor system, improved force management,                          [8] J. S. Lora-Millan, J. C. Moreno, and E. Rocon, “Coordination between
                                                                                      partial robotic exoskeletons and human gait: A comprehensive review
and flexible footplates for natural gait. However, challenges                         on control strategies,” Frontiers in Bioengineering and Biotechnology,
remain in further reducing weight and improving donning                               vol. 10, p. 842294, 2022.
and doffing, particularly due to the lack of stable sitting                       [9] J. M. Hidler and A. E. Wall, “Alterations in muscle activation patterns
                                                                                      during robotic-assisted walking,” Clinical Biomechanics, vol. 20, no. 2,
posture without external support. Future iterations should in-                        pp. 184–193, 2005.
tegrate a dedicated support mechanism to enhance usability.                      [10] E. P. Washabaugh, T. E. Augenstein, A. M. Ebenhoeh, J. Qiu, K. A.
                                                                                      Ford, and C. Krishnan, “Design and preliminary assessment of a
                       V. C ONCLUSION                                                 passive elastic leg exoskeleton for resistive gait rehabilitation,” IEEE
   In this study, we developed the Single-Sided Mobility                              Transactions on Biomedical Engineering, vol. 68, no. 6, pp. 1941–
                                                                                      1950, 2020.
Assistive Exoskeleton (SMAEXO), a single-sided, multi-                           [11] J. Wang, X. Li, T.-H. Huang, S. Yu, Y. Li, T. Chen, A. Carriero,
joint lower limb exoskeleton designed to enhance mobility                             M. Oh-Park, and H. Su, “Comfort-centered design of a lightweight
in individuals with hemiplegia. SMAEXO actively supports                              and backdrivable knee exoskeleton,” IEEE Robotics and Automation
                                                                                      Letters, vol. 3, no. 4, pp. 4265–4272, 2018.
the hip and knee while incorporating a passive, flexible
                                                                                 [12] J. S. Lora-Millan, F. J. Sanchez-Cuesta, J. P. Romero, J. C. Moreno,
ankle joint to prevent foot drop, ensuring dynamic assistance                         and E. Rocon, “A unilateral robotic knee exoskeleton to assess the
throughout the gait cycle. Its tool-free adjustments, optimized                       role of natural gait assistance in hemiparetic patients,” Journal of
weight distribution, and reaction force management enhance                            NeuroEngineering and Rehabilitation, vol. 19, no. 1, p. 109, 2022.
                                                                                 [13] “Restore exo-suit,” Lifeward. [Online]. Available: https://golifeward.
usability, balance, comfort, and safety while minimizing                              com/products/restore-exo-suit/
strain on the affected side.                                                     [14] G. Orekhov, Y. Fang, C. F. Cuddeback, and Z. F. Lerner, “Usability
   We evaluated usability, safety, comfort, and assistance with                       and performance validation of an ultra-lightweight and versatile un-
                                                                                      tethered robotic ankle exoskeleton,” Journal of neuroengineering and
eight able-bodied participants during level walking, slope                            rehabilitation, vol. 18, pp. 1–16, 2021.
walking, and obstacle crossing. Participants reported the                        [15] H. Kawamoto, T. Hayashi, T. Sakurai, K. Eguchi, and Y. Sankai,
device as safe, comfortable, and effective, with no adverse                           “Development of single leg version of hal for hemiplegia,” in 2009
                                                                                      Annual international conference of the IEEE engineering in medicine
effects such as itching or irritation, and found the assistance                       and biology society. IEEE, 2009, pp. 5038–5043.
beneficial. However, areas for improvement were noted,                           [16] L. J. A. Mayag, M. Múnera, and C. A. Cifuentes, “Human-in-the-
including better cooling, reduced weight, and simplified                              loop control for agora unilateral lower-limb exoskeleton,” Journal of
donning.                                                                              Intelligent & Robotic Systems, vol. 104, no. 1, p. 3, 2022.
                                                                                 [17] Y. Jin, B. Xiong, L. Chen, W. Zhao, Z. Li, C. Zhang, and X. Xu,
   This study is limited by a small and unbalanced participant                        “Efficacy and safety of using a unilateral lower limb exoskeleton
sample, which may impact the generalizability of the find-                            combined with conventional treatment in post-stroke rehabilitation: a
ings. Future work will address this by conducting extensive                           randomized controlled trial,” Frontiers in Bioengineering and Biotech-
                                                                                      nology, vol. 12, p. 1441986, 2024.
testing with a larger, more diverse participant pool for robust                  [18] R. Kitatani, K. Ohata, H. Takahashi, S. Shibuta, Y. Hashiguchi,
validation.                                                                           and N. Yamakami, “Reduction in energy expenditure during walking
   Building on the these findings, we will advance to clinical                        using an automated stride assistance device in healthy young adults,”
                                                                                      Archives of physical medicine and rehabilitation, vol. 95, no. 11, pp.
trials to evaluate SMAEXO’s effectiveness in post-stroke                              2128–2133, 2014.
individuals. Comprehensive data analysis will inform design                      [19] K. Seo, J. Lee, Y. Lee, T. Ha, and Y. Shim, “Fully autonomous
refinements to enhance functionality, comfort, and user ex-                           hip exoskeleton saves metabolic cost of walking,” in 2016 IEEE
perience. With continued development, SMAEXO holds the                                International Conference on Robotics and Automation (ICRA). IEEE,
                                                                                      2016, pp. 4628–4635.
potential to significantly improve mobility and quality of life                  [20] “Hal for well-being (lower limb type),” Cyborg. [Online]. Available:
for individuals with hemiplegia.                                                      https://www.cyberdyne.jp/english/products/fl05.html

979-8-3503-8068-2/25/$31.00 ©2025 IEEE                                     602
  Authorized licensed use limited to: South-Central Minzu University. Downloaded on October 01,2026 at 03:37:54 UTC from IEEE Xplore. Restrictions apply.