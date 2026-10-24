# 基于单相机与 CNN-Transformer 混合网络的图像化步态时空参数估计

**Ankhzaya Jamsrandorj¹,⁴, Quynh Hoang Ngan Nguyen²,⁴, Dawoon Jung⁴, Min Seok Baek³, Kyung-Ryoul Mun⁴, Jinwook Kim⁴**

¹ KIST School，韩国科学技术联合大学院大学（UST）人机接口与机器人工程系，首尔，韩国
² KIST School，UST 人工智能机器人系，首尔，韩国
³ 延世大学原州医学院原州 Severance 基督医院神经内科，原州，韩国
⁴ **韩国科学技术研究院（KIST）人工智能中心**，首尔，韩国（通讯：`jwkim@imrc.kist.re.kr`）

*2023 45th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC)*
5 页 · 来源：Arinex 会议微站 `full-paper_472.pdf`（**开放获取**）

---

> ### ✅ 证据说明
>
> **本文是原论文 PDF 的完整中文翻译**（非替代包）。
> 原文 PDF 已落盘于 `01_参考论文/RGB_直接竞品_2026_Chao/Jamsrandorj2023：单相机 CNN-Transformer 的步态时空参数估计/Jamsrandorj2023：单相机 CNN-Transformer 的步态时空参数估计.pdf`
> （475,279 字节，5 页，来源为会议官方微站，**合法免费**）。
>
> **🔴 为什么这篇被收进本库**：它是 **Chao 2026 Measurement（我方直接竞品）的方法前身**，
> 同一 KIST 团队、同一架构族（CNN + Transformer）。**它是目前可获得的对 Chao 2026 方法细节的最强旁证。**
> 详见文末"§ 对 Chao 2026 的关键情报"。
>
> 译者注：凡标 `[译者注]` 处为译者补充，非原文内容。

---

## 摘要

**摘要——** 基于视觉的步态分析可在**老年人健康状况的远程与连续监测**中发挥重要作用。然而，大多数基于视觉的方法**使用人体姿态信息**计算步态时空参数，且只提供**平均值**参数。本研究旨在提出一种**直接的（straightforward）方法**，实现**逐步伐（stride-by-stride）**的步态时空参数估计。本研究共纳入 **160 名老年受试者**。数据使用 **GAITRite 系统**与**一部移动相机**同步采集。**三个深度学习网络**以**少量 RGB 帧**为输入，以**包含空间与时间步态参数的一维连续信号**为输出进行训练。所训练的网络**以 0.938 及以上的相关系数估计步长**，并以 **0.914 及以上的 F1 分数检测步态事件**。

**临床相关性——** 所提方法在步态时空参数分析上与 GAITRite 系统表现出**高度一致性**。我们的方法可应用于基于步态参数监测老年人健康状况，用于疾病早期诊断、恰当治疗与及时干预。

**关键词**—— 步态分析、时空参数、单相机、CNN-Transformer、无标记

---

## I. 引言

步态可因损伤、虚弱与衰老而发生改变，导致异常行走，进而引发更严重的健康问题，如肌肉骨骼、心血管与心理健康问题。因此，对老年人进行步态分析，可作为**评估其身体与心理状况、早期发现医疗隐患**的有效临床指标。

当前的专用系统——包括**光学动作捕捉系统、测力台与可穿戴设备**——**昂贵、不可移动、日常不可及**。由于需要高度配合，它们还可能给老年人带来**不适**。因此，**基于视觉的步态分析系统**对于远程评估与监测老年人步态**至关重要**。

大多数基于视觉的方法聚焦于**2D 或 3D 人体姿态估计**（使用特定相机系统或外部姿态机），然后基于骨骼跟踪数据计算步态参数 [1]–[9]。例如，Ismail 等人应用 **DeepPose** 框架获取 2D 人体关节位置，并用回归模型估计选定关节的运动进行步态分析 [2]；Stenum 等人使用**侧视视频**，通过 **OpenPose** 定位并跟踪关节以测量步态时空参数 [6]。

**然而**，从人体姿态检测器提取的 **2D 关节位置可能并不位于关节中心**，因而存在误差。此外，**2D 姿态估计结果会因视角、部分遮挡、光照变化与生物特征差异而改变**，从而影响步态参数计算。因此，一些研究工作改用 **3D 关节位置**来规避这些问题并提升步态分析的可靠性。Azhand 等人提出了基于从 2D 人体姿态提升得到的 3D 人体姿态及其跟踪的步态分析系统 [7]；Zhu 等人提出一种**级联网络**，使用红外热成像视频与 OpenPose 提取的 2D 姿态回归 3D 人体关节位置，再通过跟踪估计的关节来计算步态参数 [8]。

**然而，这些方法仍依赖 2D 姿态，且使用复杂模型来估计 3D 人体关节坐标。此外，这些方法是在健康年轻与年长受试者上验证的。**

本研究**主要聚焦老年人**，包括**健康对照（HC）**与**神经退行性疾病患者**，特别是**帕金森病（PD）、阿尔茨海默病（AD）与轻度认知障碍（MCI）**——考虑到它们在老年人中的高患病率，且步态可用于评估神经退行性疾病 [10]–[12]。此外，对老年人的**连续逐步伐步态分析**可实现对**损伤或跌倒风险**的基本临床评估 [9]、**术后康复效果**评估，以及**运动障碍的诊断与评估**。

因此，本研究旨在提出一种**基于视觉的方法，它不需要人体关节位置，且能轻易应用于日常生活**，用于**逐步伐的步态时空参数估计**。

---

## II. 材料与方法

### A. 受试者

本研究基于**韩国科学技术研究院（KIST）机构审查委员会（IRB）**支持的工作开展。

共 **160 名老年受试者**参与本实验。所有参与者在参与前均获得**书面知情同意**。所有受试者——包括 HC 以及被诊断为 PD、AD、MCI 及其他疾病的患者——**均可无需他人帮助或辅助设备独立行走**。

**表 I：各研究组的人口统计学与人体测量学特征**

| 项目 | AD | MCI | PD | HC | 其他 |
|---|---|---|---|---|---|
| 受试者数（女性） | **37 (18)** | **28 (15)** | **39 (18)** | **26 (14)** | **30 (12)** |
| 年龄（岁） | 73.5 ± 7.6 | 74.4 ± 6.0 | 70.9 ± 10.6 | 70.8 ± 9.7 | 70.7 ± 11.8 |
| 体重（kg） | 56.7 ± 10.4 | 55.5 ± 9.7 | 61.2 ± 10.4 | 59.8 ± 12.9 | 61.8 ± 9.9 |
| 身高（cm） | 157.7 ± 8.8 | 153.4 ± 6.8 | 160.9 ± 9.1 | 162.1 ± 7.1 | 161.7 ± 8.8 |
| **平均行走速度（cm/s）** | 76.2 ± 16.7 | 84.7 ± 23.1 | 91.7 ± 24.2 | **102.9 ± 20.6** | 80.1 ± 25.0 |

数据以 mean ± SD 表示。

> 🔴 **【本库重点标注】**：**HC 组的平均步速（102.9 cm/s）显著高于全部病理组**（AD 76.2 / PD 91.7 / MCI 84.7）。
> 这与 **Zhang 2025 JBHI** 的临床发现（高跌倒风险 → 速度更低）**互为独立验证**。
> 我方 `04_实验设计/评价指标.md` 把 **velocity error / ROM violation** 设为一级指标，有了**两组独立临床数据的背书**。

### B. 实验协议

数据在室内环境中使用 **GAITRite 步道系统**与**一部移动相机**同步采集（图 1）。要求参与者以**自选速度**在压力垫上行走，**重复行走动作 10 次**。

相机以**侧视视角（lateral view）**记录行走动作，分辨率为 **HD**，帧率 **30 Hz**。

步态时空参数的**真值**——包括**步长（stride length, SL）**、**足触地（foot contact, FC）时刻**与**足离地（foot-off, FO）时刻**——由 **GAITRite 步道系统**测量。

视频与 GAITRite 系统数据由一位专家依据**首个 FC 事件**进行**人工同步**，再由另一位专家**交叉校验**同步结果。

> 🔴 **【本库重点标注】**：**相机的"高度与位置并不固定"**（原文 §II.C.2 与 §II.D.2 均强调），
> 这是刻意的**抗视角变化设计**。→ 与前文的 view-independent（2022 EMBC）和 Chao 2026 的"跨视角"
> 构成**同一团队一贯的鲁棒性主张**。

### C. 步态时空参数估计器的开发

#### 1) 面向视觉估计的目标构建

GAITRite 系统提供两组时空步态参数：**测试区域内的平均参数**与**详细区域内的单次落地（individual-footfall）参数**。

我们提取了 FC 与 FO 事件发生时刻的时间信息，以及每位参与者的**个体化步长（SL）**；**每个 SL 参数都使用该参与者的腿长进行了归一化**。

如图 2 所示，这些时空参数被用于构建**逐步伐的一维信号**，供视觉估计使用。

**步长定义为同一只脚相邻两次 FC 时刻之间的距离。** 但我们的做法是**构建同一只脚 FO 与 FC 事件之间的一维信号**。假设参与者在行走过程中速度一致，**信号被线性递增以达到从 GAITRite 系统提取的 SL 真值**。因此，**每个构建的信号同时包含每一步伐的空间与时间步态参数**。

#### 2) 网络架构

所提出的 **CNN-Transformer** 网络由三部分组成：**空间特征提取器、时间特征提取器、全连接层**。

| 组件 | 实现 |
|---|---|
| **空间特征提取器** | **ResNet18**（在 ImageNet 上预训练） |
| **时间特征提取器** | **Transformer 编码器**，含 **2 层编码器层**；每层有 **4 个多头自注意力（multi-head self-attention）子层** + 1 个前馈层 |
| **最终模块** | **2 个全连接层**，中间以 **ReLU** 激活函数与 **Dropout** 相连 |

输入特征序列**被注入位置信息（positional information）**后送入 Transformer 编码器。

**输入：27 个连续视频帧**。**输出：双腿构建的步长目标值。**

为作性能对比，我们另外使用两个深度学习网络作为时间特征提取器：**CNN-LSTM** 与 **CNN-BiLSTM**。

### D. 实现细节

#### 1) 数据处理

网络输入为 **27 个连续帧**，附带**真值人体边界框**与**所有帧构建的步长值**。原始帧的预处理为：**裁剪出画面中的人**，然后**左右填充（pad left and right）成正方形**，再**缩放至 448 × 448**。

**数据集按受试者划分为三部分**：

| 划分 | 受试者数 | AD | MCI | PD | HC | 其他 |
|---|---|---|---|---|---|---|
| **训练集** | **96** | 20 | 17 | 21 | 15 | 23 |
| **验证集** | **32** | 7 | 5 | 8 | 5 | 7 |
| **测试集** | **32** | 10 | 6 | 10 | 6 | 0 |

> 🔴 **【本库重点标注】**：这是**按受试者划分（cross-subject split）**，即**跨被试协议**。
> 与 Kumar 2022 EMBC 的 7/3/3 按受试者划分一致，**再次证明 KIST 团队在 2022–2023 年就已采用跨被试划分**。

#### 2) 数据增强

尽管本研究仅用**单相机**采集步态数据，但**相机的高度与位置并不固定**，以避免受限于某一特定相机视角——它们对每位参与者可能略有不同。

此外，对输入帧应用了**旋转增强**：在 **x 轴上分别旋转 +30° 与 −30°**，使所提方法对相机角度变化**更鲁棒**。同时使用了**水平翻转**。

#### 3) 训练方案

| 项 | 值 |
|---|---|
| 框架 | **PyTorch** |
| 权重初始化 | 在 **ImageNet** 上预训练 |
| 损失函数 | **平均绝对误差（MAE）** |
| 优化器 | **Adam** |
| 初始学习率 | **0.0001** |
| 批大小 | **16** |
| 学习率调度 | **每 3 个 epoch 降低两次**（reduced two times after every three epochs） |

### E. 评价指标

步态时空参数估计的**目标被设计为简单的一维信号**。

| 指标 | 用途 | 细节 |
|---|---|---|
| **MAE（cm）** | 步长估计 | 定义步长的**峰值**先被逐个信号提取 |
| **ICC（组内相关系数）** | 步长估计 | 使用 **(2,k) 型**，评估 GAITRite 系统与所提方法之间的一致性 |
| **F1 分数** | 步态事件检测 | 见下 |

**F1 的判定规则**：
1. 先找到**峰值所在的帧**，以精确定位含 FC 事件的帧号；
2. 用 **2 cm 阈值**找最近点，确定含 FO 事件的帧号；
3. 计算目标帧号与预测帧号的**帧差**；
4. **帧差小于 2 帧阈值 → 判为真正例（true positive）**；
5. 其余步态事件预测计为**假正例（false positive）**；
6. 所有未被检测到的目标视为**假负例（false negative）**。

---

## III. 结果

估计结果见图 4：**三个网络都准确地估计了步长并检测出步态事件所在帧**。

### 表 II：各网络在测试集上的空间参数（步长）估计性能

| 模型 | RSL MAE (cm) | LSL MAE (cm) | RSL ICC | LSL ICC |
|---|---|---|---|---|
| CNN-LSTM | 9.236 ± 4.677 | 9.983 ± 4.523 | 0.905 ± 0.041 | 0.885 ± 0.047 |
| CNN-BiLSTM | 7.211 ± 3.648 | 8.643 ± 4.742 | 0.953 ± 0.043 | 0.918 ± 0.043 |
| **CNN-Transformer** | **6.909 ± 3.657** | **7.420 ± 3.153** | **0.962 ± 0.047** | **0.957 ± 0.045** |

RSL = 右步长，LSL = 左步长。数据以 mean ± SD 表示。

**CNN-Transformer 取得最小 MAE（6.909 / 7.420 cm）与最高 ICC（0.962 / 0.957）。**

### 表 III：各网络在测试集上的步态事件检测性能（F1）

| 模型 | RFC | RFO | LFC | LFO |
|---|---|---|---|---|
| CNN-LSTM | 0.908 ± 0.132 | 0.941 ± 0.127 | 0.914 ± 0.136 | 0.949 ± 0.112 |
| **CNN-BiLSTM** | **0.907 ± 0.125** | 0.941 ± 0.119 | 0.906 ± 0.128 | **0.965 ± 0.090** |
| **CNN-Transformer** | 0.901 ± 0.146 | **0.951 ± 0.114** | **0.928 ± 0.141** | **0.967 ± 0.086** |

RFC/RFO/LFC/LFO = 右足触地 / 右足离地 / 左足触地 / 左足离地。数据以 mean ± SD 表示。

**关键发现**：
- **对所有网络，FO 事件的检测性能都高于 FC 事件。**
- **CNN-Transformer 取得最高 F1**：右 FO **0.951**（precision 0.928 / recall 0.985）、左 FC **0.928**（precision 0.909 / recall 0.969）、左 FO **0.967**（precision 0.953 / recall 0.988）。
- CNN-BiLSTM 在**右 FC** 上最高（**0.908**，precision 0.857 / recall 0.993）。

### 表 IV / V：相机角度变化时的性能

| 模型 | 旋转 30° | 旋转 0° | 旋转 −30° |
|---|---|---|---|
| **MAE (cm)** | | | |
| CNN-LSTM | 9.635 ± 5.141 | 9.610 ± 4.600 | 9.487 ± 5.162 |
| CNN-BiLSTM | 7.930 ± 4.821 | 7.927 ± 4.195 | 7.685 ± 4.773 |
| **CNN-Transformer** | **7.171 ± 3.971** | **7.165 ± 3.405** | **7.061 ± 3.854** |
| **ICC** | | | |
| CNN-LSTM | 0.882 ± 0.045 | 0.895 ± 0.063 | 0.896 ± 0.078 |
| CNN-BiLSTM | 0.937 ± 0.049 | 0.936 ± 0.047 | 0.940 ± 0.045 |
| **CNN-Transformer** | **0.961 ± 0.042** | **0.960 ± 0.042** | **0.963 ± 0.041** |
| **F1（表 V）** | | | |
| CNN-LSTM | 0.928 ± 0.093 | 0.928 ± 0.079 | 0.936 ± 0.091 |
| CNN-BiLSTM | 0.931 ± 0.074 | 0.930 ± 0.069 | 0.933 ± 0.079 |
| **CNN-Transformer** | **0.939 ± 0.061** | **0.937 ± 0.086** | **0.943 ± 0.055** |

**三个网络在输入帧被水平轴旋转时，都表现出对时空步态参数相当鲁棒的估计能力。**

**所有网络在输入旋转 −30° 时性能优于旋转 +30°。** 性能提升可能是因为**下肢在负角度旋转的帧中显得更展开**。

---

## IV. 讨论

本研究提出了一种**新颖的基于视觉的步态时空参数估计方法**。我们构建了**同时包含空间与时间步态参数的简单一维信号**，用于**连续的逐步伐步态分析**。以少量 RGB 帧训练的深度学习网络，通过所构建的信号在估计时空步态参数上取得了**有前景的性能**。

值得注意的是，**我们的方法不需要任何专用传感器、标记点或人体姿态估计器**即可完成老年人步态分析。

所提方法与 **GAITRite 步道系统**在分析时空步态参数上表现出**高度一致性**。

**CNN-Transformer 网络取得最佳估计性能的原因**，**可能源于它借助多头自注意力机制同时访问长程序列数据的能力**：

> **不同于 CNN-LSTM，CNN-Transformer 网络能够同时考虑未来与过去的信息**，这也带来了 CNN-BiLSTM 的收益，**但计算成本只有后者的一半。**

> 🔴 **【本库重点标注】**：这是对 **"为什么用 Transformer 而不是 BiLSTM"** 的**作者亲口给出的理由**：
> ① 同时利用过去与未来信息；② **计算成本仅为 CNN-BiLSTM 的一半**。
> → 这两条理由是**现成的、可引用的"Transformer 优于 BiLSTM"论证**，我方 paper 的 method 章节可直接沿用并致谢来源。

使用 CNN-Transformer 网络，我们分析了**不同受试者组别**（AD、MCI、PD、HC）上的估计性能。

### 表 VI：CNN-Transformer 在不同受试者组别上的估计性能对比

| 组别 | RSL ICC | LSL ICC | FC F1 | FO F1 |
|---|---|---|---|---|
| AD | 0.957 ± 0.139 | 0.933 ± 0.131 | 0.895 ± 0.210 | 0.933 ± 0.178 |
| MCI | 0.966 ± 0.039 | 0.929 ± 0.026 | 0.922 ± 0.096 | 0.970 ± 0.069 |
| PD | 0.939 ± 0.063 | 0.937 ± 0.074 | 0.895 ± 0.175 | 0.949 ± 0.102 |
| HC | **0.981 ± 0.029** | **0.983 ± 0.033** | **0.941 ± 0.051** | **0.982 ± 0.049** |

**CNN-Transformer 网络在不论受试者组别的情况下，都以 0.929 及以上的 ICC 估计步长、以 0.933 及以上的 F1 检测 FO 事件。**
**但 AD 与 PD 患者的 FC 事件检测效果不如 HC 与 MCI。**

基于该估计性能，所提方法可用于**正常步态与病理步态**的远程监测与分析。

如引言所述，大多数基于视觉的方法使用**人体姿态估计机**定位人体关节并跟踪以计算时空参数。然而，一些研究**仅使用 RGB 视频**进行步态分析。Yu 与 Jeong 等人开发并使用同一深度学习网络分析步态，他们用 **3D CNN** 网络从 RGB 视频帧提取的**预准备特征**估计平均时空参数 [13], [14]。

**与这些研究相比，我们的方法在没有任何预准备特征的情况下估计逐步伐步态参数，使其更适合日常生活场景中的连续与远程步态分析。** 此外，由于健康状况，老年人与患者在不同行走阶段（如起始、中段与结束阶段）可能具有**可区分的步态特征**。**逐步伐的时空步态参数能够实现对这些行走阶段步态变异性的分析。**

### 局限（原文自述）

本研究可能被视为局限的几点：

1. **本研究仅旨在检测步态事件与估计步长。** 时间步态参数可用两个主要步态事件计算，但**更多空间参数（如步幅长度 step length、宽度 width 与足部廓清 foot clearance）将在未来工作中考虑**。
2. 我们的方法在单视角视频下取得了有前景的性能，但**扩展到多视角视频、改进网络架构以提升性能，以及检验不同形状的目标信号，均在未来工作计划中**。

---

## V. 结论

本研究提出了一种**基于视觉的逐步伐步态时空参数估计方法**。我们的结果支持所提方法评估**各类步态**的能力。以**少量视频帧与连续信号为目标**训练的深度学习网络，在检测步态事件与估计步长上表现出**有竞争力的性能**。

我们的方法可应用于**在家庭环境中监测老年人健康状况**，并有助于**疾病早期诊断、恰当治疗与及时干预**。

---

## 致谢

本工作由"**面向残障人士的足部诊断与鞋类制造技术开发项目（No. 1711192863）**"与"**运动员训练/比赛数据管理与基于 AI 的性能提升解决方案技术开发项目（No. 1375027432）**"支持。

---

## 参考文献

1. I. González et al., "Comparison between passive vision-based system and a wearable inertial-based system for estimating temporal gait parameters related to the gaitrite electronic walkway," *Journal of Biomedical Informatics*, vol. 62, pp. 210–223, 2016.
2. H. Ismail, I. Radwan, H. Suominen, and R. Goecke, "Gait estimation and analysis from noisy observations," in *2019 41st EMBC*, IEEE, 2019, pp. 2707–2712.
3. Ł. Kidziński, B. Yang, J. L. Hicks, A. Rajagopal, S. L. Delp, and M. H. Schwartz, "Deep neural networks enable quantitative movement analysis using single-camera videos," *Nature Communications*, vol. 11, no. 1, pp. 1–10, 2020.
4. S. Hazra, A. A. Pratap, D. Tripathy, and A. Nandy, "Novel data fusion strategy for human gait analysis using multiple kinect sensors," *Biomedical Signal Processing and Control*, vol. 67, p. 102512, 2021.
5. Y. Ma, K. Mithraratne, N. Wilson, Y. Zhang, and X. Wang, "Kinect v2-based gait analysis for children with cerebral palsy…," *Sensors*, vol. 21, no. 6, p. 2104, 2021.
6. J. Stenum, C. Rossi, and R. T. Roemmich, "Two-dimensional video-based analysis of human gait using pose estimation," *PLoS Computational Biology*, vol. 17, no. 4, p. e1008935, 2021.
7. A. Azhand, S. Rabe, S. Müller, I. Sattler, and A. Heimann-Steinert, "Algorithm based on one monocular video delivers highly valid and reliable gait parameters," *Scientific Reports*, vol. 11, no. 1, pp. 1–10, 2021.
8. Y. Zhu, W. Lu, R. Zhang, R. Wang, and D. Robbins, "Dual-channel cascade pose estimation network trained on infrared thermal image and groundtruth annotation for real-time gait measurement," *Medical Image Analysis*, vol. 79, p. 102435, 2022.
9. E. E. Stone and M. Skubic, "Passive in-home measurement of stride-to-stride gait variability comparing vision and kinect sensing," in *2011 Annual International Conference of the IEEE EMBC*, IEEE, 2011, pp. 6491–6494.
10. D. Jung, J. Kim, M. Kim, C. W. Won, and K.-R. Mun, "Classifying the risk of cognitive impairment using sequential gait characteristics and long short-term memory networks," *IEEE JBHI*, vol. 25, no. 10, pp. 4029–4040, 2021.
11. B. Vidya and P. Sasikumar, "Gait based parkinson's disease diagnosis and severity rating using multi-class support vector machine," *Applied Soft Computing*, vol. 113, p. 107939, 2021.
12. B. Ghoraani et al., "Detection of mild cognitive impairment and alzheimer's disease using dual-task gait assessments and machine learning," *Biomedical Signal Processing and Control*, vol. 64, p. 102249, 2021.
13. H. Yu, K. Kang, S. Jeong, and J. Park, "Deep vision system for clinical gait analysis in and out of hospital," in *ICONIP*, Springer, 2019, pp. 633–642.
14. S. Jeong, H. Yu, J. Park, and K. Kang, "Quantitative gait analysis of idiopathic normal pressure hydrocephalus using deep learning algorithms on monocular videos," *Scientific Reports*, vol. 11, no. 1, pp. 1–10, 2021.

---

## 🔴 对 Chao 2026 的关键情报（译者补充）

> 本节为**译者补充**，非原文内容。这是本文件被收进本库的**核心理由**。

### 1. 架构对照：本文 vs Chao 2026

| 组件 | **本文（EMBC 2023，已验证真实全文）** | **Chao 2026（替代包，摘要级）** | 是否吻合 |
|---|---|---|---|
| 空间特征 | **ResNet18**（ImageNet 预训练） | "a convolutional neural network for spatial feature extraction" | ✅ 高度吻合 |
| 时序特征 | **Transformer 编码器（2 层，每层 4 头自注意力）** | "a transformer module to capture temporal gait dynamics" | ✅ 吻合 |
| 输入 | **27 连续视频帧**，裁人 → 补方 → **448×448** | `[未验证]` | ⬜ 待核 |
| 输出 | 步长一维信号（时空参数） | **髋/膝/踝关节角度** | ❌ 任务不同 |
| 训练损失 | MAE + Adam，lr 1e-4，batch 16 | **L_angle + λ·L_angular_velocity** | ⚠️ Chao 新增了角速度项 |
| 划分 | **按受试者**（96/32/32） | `[未验证]` | ⬜ 待核 |
| 鲁棒性检验 | **相机旋转 ±30°**、跨疾病组 | **跨视角 + 跨速度 + 跨步态异常** | ✅ 同源思路 |
| 团队 | KIST（Mun, Kim, Jamsrandorj, Jung） | KIST（Chao, Park, **Jamsrandorj**, **Jung**, Kim, Mun） | ✅ **共享 3 位作者** |

### 2. 三条可安全推进的推论（仍标推断，但置信度显著提高）

1. **Chao 2026 的 CNN + Transformer 极可能就是本架构的延续**：同一个 ResNet18 空间提取器 + 同一个 Transformer 编码器的"两步走"设计。
   → Chao 2026 的新增量**更可能集中在损失函数**（联合惩罚角度与角速度）与**任务切换**（时空参数 → 关节角度）上，**而不是架构创新**。
   → **这对我是重大利好**：说明"CNN+Transformer"在 KIST 团队内部是**已定型的基础设施**，Chao 2026 的贡献面比我方原先担心的更窄。

2. **"Transformer 优于 BiLSTM"的论证可直接复用**：原文给出了两条**作者亲口**的理由——
   ① 同时利用过去与未来信息；② **计算成本仅为 CNN-BiLSTM 的一半**。

3. **KIST 的鲁棒性方法论已完全清晰**：**相机位置不固定 + ±30° 旋转增强 + 跨疾病组验证**。
   → 我方若声称"跨视角鲁棒"，必须能对标这**三重扰动**，否则不如 2023 年的同类工作。

### 3. 我方（ViKiT）与本文的结构性差异

| 维度 | Jamsrandorj 2023 EMBC | ViKiT（我方） |
|---|---|---|
| 输出 | **步态时空参数**（步长等标量信号） | **关节角序列**（高维运动学） |
| 任务定义 | 回归 | **Seq2Seq 翻译** |
| 表征 | 无显式 token（ResNet 特征直连 Transformer） | **Anatomy-aware tokens** |
| 物理约束 | 无（仅 MAE） | **6 项 grammar loss** |
| 关节/病灶 | 下肢步态、神经退行性疾病 | **上肢代偿**（腕/肘/肩） |
| 视角 | 单相机侧视（+ 旋转增强） | 跨视角 |
| 输出粒度 | **逐步伐**（stride-by-stride） | **逐帧**（frame-by-frame）高分辨率 |

> **可引用的软肋**：本文输出的是**逐步伐的标量信号**（步长、事件时刻），**粒度远低于逐帧关节角**。
> 我方"逐帧关节角序列翻译"在**时间分辨率**上是它的严格超集——**这是一条不需要实验结果就能成立的结构性差异**。
