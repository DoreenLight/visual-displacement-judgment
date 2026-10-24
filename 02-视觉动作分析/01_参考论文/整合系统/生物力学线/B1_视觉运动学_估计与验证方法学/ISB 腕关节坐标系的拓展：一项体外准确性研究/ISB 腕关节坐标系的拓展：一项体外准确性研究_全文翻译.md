# ISB 腕关节坐标系的拓展：一项体外准确性研究

M. Lenhoff<sup>1</sup>, T.I. Carter<sup>1</sup>, B. Pansy<sup>1</sup>, A. Wolff<sup>1</sup>, R.Y. Cheng<sup>1</sup>, S.I. Backus<sup>1</sup>, J.J. Crisco<sup>2</sup>, H.J. Hillstrom<sup>1,3</sup>, S.W. Wolfe<sup>1</sup> <sup>1</sup>Hospital for Special Surgery, New York, NY 10021; <sup>2</sup>Brown University, Providence, RI 02912 <sup>3</sup>通讯作者（Corresponding Author）; email: HillstromH@HSS.EDU

## 引言

研究人员使用了各种标记点集和方法来描述上肢运动学。然而，如果缺乏用于三维运动分析（3DMA）的标准解剖和关节坐标系定义，不同研究人员之间的数据集便无法进行比较。为了统一上肢运动的描述，Wu 及其同事 [1] 发表了一套关节坐标系推荐标准。本研究所提出的运动学方案是对该标准的拓展。本研究的目的在于开发一种基于解剖结构的腕关节坐标系（WJCS）及相应的追踪坐标架。为了测试这种新的 WJCS，本研究对三维运动分析（3DMA）实现方式、手动测角法（MG）和电子测角法（EG）进行了对比。我们假设 3DMA 将比其他方法提供更高的准确性。

## 方法

所提出的 WJCS 基于国际生物力学学会（International Society of Biomechanics, ISB）[1] 的推荐标准。追踪坐标架由额外的前臂和第三掌骨三标记点组构成。为了实现该 WJCS，需要若干物理标记点（9mm 无源逆反射标记点）、由数字指示棒确定的三维位置以及虚拟标记点（Figure 1）。

为了建立腕关节坐标系（WJCS）原点，使用三维数字化指示器对桡侧鼻烟窝（P ）和尺侧鼻烟窝 $\left( \mathrm { P _ { U S B } } \right)$ 进行配准。虚拟腕关节原点（V ）被定义为连接 P 与 $\mathrm { P _ { U S B } . }$ 的线段中点。请注意，WJCS 原点穿过头状骨。这与 ISB（国际生物力学学会）的定义有所不同，在 ISB 定义中，WJCS 的原点是桡骨茎突与尺骨茎突（P 和 $\mathrm { P _ { U S } ) }$ 之间的中点。Neu 等人以及 Youm 等人先前的研究已确定头状骨为运动原点所在的腕部结构 [2,3]。WJCS 定义为：$\mathrm { V _ { W J O } } { = } ~ ( \mathrm { P _ { R S B } } { - } \mathrm { P _ { U S B } } ) / 2$，其中 V 为腕关节坐标系原点；${ \bf e } _ { 1 } = { \bf Z } _ { \bf R }$，其中 $\mathbf { Z _ { R } }$ 为固定于 $\mathrm { V } _ { \mathrm { W J O } ; }$ 的桡骨坐标系内-外侧轴；$\mathbf { e } _ { 3 } = \mathbf { Y } _ { \mathbf { M C } }$，其中 $\mathbf { Y _ { M C } }$ 为掌骨坐标系的纵轴；且 $\mathbf { e } _ { 2 } = \mathbf { e } _ { 1 } \mathrm { \mathrm { X } } \ \mathbf { e } _ { 3 }$，其中 $\scriptstyle { e _ { 2 } }$ 垂直于 e 和 e 。

![](images/d7eaa583c9fea0adc1254e9199d534aeed725ca4ae9c231a3a5dc5fdccb2f952.jpg)

注：$\theta _ { \uparrow }$ 为 $\mathsf { Y } _ { \mathsf { M C } } ； \mathsf { e } _ { 3 }$ 为固定于 $v _ { w s i o }$ 的 $Z _ { \scriptscriptstyle { \mathrm { R } } } ，$ $\mathtt { e } _ { 2 }$ 垂直于 $\ominus _ { \gamma }$ 和 $\mathtt { e } _ { 3 }$

运动学求解技术基于 Grood 和 Suntay 浮动轴方法 [4]，该方法能够对关节运动进行六自由度表征。

通过插入空心髓内棒制备了 10 具尸体腕关节，用于透视确定（Hologic Insight 小型 C 臂机）实际腕关节姿态。使用能够同时测量屈-伸和桡尺偏的双轴电子量角器（Biometrics Ltd. (Cwmfelinfach, Gwent, UK)，型号 SG65）评估腕关节位置。WJCS 的三维运动学数据使用 Motion Analysis Corporation (Santa Rosa, CA) 的 3DMA 系统采集，该系统配备 8 台 Eagle（1.3 百万像素）摄像机，以 200Hz 的频率捕获数据。所有背/掌侧量角器测量均使用分辨率为 1 度的手动量角器。将腕关节分别放置在约 10、15、30 和 45 度的屈曲角度以及最大屈曲活动范围。通过背/掌侧 MG、EG 和 3DMA 记录旋转值。对其余三种主要腕部运动（伸展、桡偏、尺偏）以及中立位重复该过程。计算了透视检查（金标准）与各技术测量值之间的均方根误差（RMSE）值。

## 结果与讨论

如 Table 1 的 RMSE 计算所示，基于视频动作捕捉实现的新型 WJCS（腕关节坐标系）最为准确（3DMA：4.4°；MG：6.6°；EG：7.2°）。EG 在最大运动范围时准确性有所下降（RMSE 9.17°），而 3DMA 则未表现出这一特征（RMSE 4.25°）。

Figure 1: 腕关节坐标系（Wrist Joint Coordinate System）。背侧观——右腕。  
Table 1: 以荧光透视为参考的准确性

<table><tr><td></td><td>MG</td><td>EG</td><td>3DMA</td></tr><tr><td>RMSE</td><td>6.6°</td><td>7.2°</td><td>4.4°</td></tr></table>

我们验证了背侧/掌侧 MG、EG 以及 ISB WJCS 扩展方法的准确性，并证明了 3DMA 比 MG 更准确（均方误差双尾配对 t 检验，p<0.0001），且比在动态运动中测量腕关节位置的传统技术 EG 更准确（p<0.005）。针对新型 WJCS 体内（in vivo）准确性与可靠性的后续研究目前正在进行中。

## 参考文献

1. Wu G, et al. J Biomech 38(5), 981-92, 2005.

2. Neu CP, et al. J Biomech 34(11), 1429-38, 2001.

3. Youm Y, et al. J Bone Joint Surg Am 60(4), 423-31, 1978.

4. Grood ES, et al. J Biomech Eng 105(2), 136-44, 1983.