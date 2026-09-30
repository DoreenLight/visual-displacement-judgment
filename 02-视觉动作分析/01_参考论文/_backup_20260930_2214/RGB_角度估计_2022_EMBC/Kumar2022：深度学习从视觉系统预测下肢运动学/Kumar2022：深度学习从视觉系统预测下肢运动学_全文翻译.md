# 使用深度学习方法从视觉系统预测下肢运动学

**Konki Sravan Kumar¹, Ankhzaya Jamsrandorj², Jinwook Kim¹, Kyung-Ryoul Mun¹**

¹ 韩国科学技术研究院（KIST）人工智能中心，首尔，韩国（e-mail: krmoon02@kist.re.kr）
² 韩国科学技术联合大学院大学（UST）人机交互机器人工程系，大田，韩国

*2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC)*
Scottish Event Campus, Glasgow, UK, July 11–15, 2022 · pp. 177–181 · DOI `10.1109/EMBC48229.2022.9871577`

---

> ### ✅ 证据说明
>
> **本文是原论文 PDF 的完整中文翻译**（非替代包）。
> 原文 PDF 已获取并落盘于同目录：`Kumar2022：深度学习从视觉系统预测下肢运动学/Kumar2022：深度学习从视觉系统预测下肢运动学.pdf`（1,237,638 字节，5 页，`file` 报合法 PDF）。
>
> **DOI 更正记录**：本库 `LIST.md` 与 `STATUS.md` 此前登记为 `10.1109/EMBC48229.2022.9871780`，该 DOI **实际指向一篇心脏 MRI 论文**（Hasan & Linte，"Joint Segmentation and Uncertainty Estimation of Ventricular Structures from Cardiac MRI"），**系误填**。
> 正确 DOI 为 **`10.1109/EMBC48229.2022.9871577`**（Crossref + OpenAlex + 论文页眉 + IEEE 页脚四源一致）。
>
> 译者注：凡本文件中方括号 `[译者注]` 标注处为译者补充说明，非原文内容。

---

## 摘要

**摘要——** 日常锻炼中的关节角速度是损伤风险指数、康复进展监测以及运动员表现评估的重要临床结局指标。近年来，可穿戴传感器已被广泛用于监测下肢运动学。然而，这些传感器在日常生活中难以使用且不便。为缓解这些局限，本研究提出一种**基于视觉的系统**，使用**深度卷积神经网络结合双向长短期记忆网络（bi-directional LSTM）与门控循环单元网络（GRU）**来估计下肢关节运动学。我们计算了由**光学动作捕捉系统**获得的真值与所提模型估计的关节角速度之间的**归一化相关系数**与**平均绝对误差（MAE）**。估计结果显示，**深蹲动作中最高相关系数为 0.93，跑步机行走中为 0.92**。此外，我们还对**髋、膝、踝**三个关节的角速度**分别建立了独立模型**并进行了分析与比较。在三个关节角速度中，**膝关节具有最佳估计精度（深蹲 0.96，跑步机行走 0.96）**。所提模型在**侧视图与正视图**下均表现出更高的估计精度，且**不受相机位置与角度影响**。本研究证明了使用**无传感器视觉系统**在家庭锻炼场景中监测下肢运动学、服务于医疗健康与康复的可行性。

**关键词**——视觉系统、下肢运动学、关节角速度、深度学习、CNN、Bi-LSTM、GRU

> [译者注] 原文未标注关键词章节，此处依据摘要与全文主题归纳，便于检索。

---

## I. 引言

人类步态是肢体运动的模式，步态参数可以反映人体的功能能力与健康状况 [1]。在众多步态参数中，**日常锻炼时的下肢关节角速度**是一个重要的临床结局指标，可用于：评估损伤风险指数、判断患者康复进展、早期诊断人体关节运动障碍，以及提升运动员表现 [2, 3]。此外，近期研究已确认 [4, 5] **关节角速度是监测下肢关节功率变化的强候选指标**——因为关节功率是**关节角速度与关节力矩的乘积**，而关节功率是步态病理临床诊断与下肢假肢控制的重要参数 [6, 7]。

**惯性传感器系统**被普遍用于采集下肢运动学。基于惯性测量单元（IMU）的系统被设计用于测量人体肢体动态运动过程中的膝关节角度 [8]。由加速度计、地磁传感器与陀螺仪构成的基于 IMU 的三维动作捕捉系统，是记录下肢关节（膝、髋、脊柱）日常动作运动的有效测量工具 [9]。过去几年间，基于 IMU 传感器的系统已引起研究者关注，并在关节运动学测量中展现出显著精度 [10, 11]。**然而，一个 IMU 传感器无法用于追踪一个以上身体节段的运动。** 因此，仅仅为了追踪单侧下肢运动（如骨盆、踝、膝、髋），通常就需要**七个 IMU 传感器**（左右各一套）。随着 IMU 传感器数量的增加，出现了若干缺点：**复杂的预处理流程**（传感器-节段标定与受试者标定）、**对日常生活的不便**，以及为监测准确的下肢运动学而必须考虑的**人体运动链问题** [12]。此外，使用 IMU 传感器等运动追踪系统，会因**为固定传感器而绑紧的束带**而妨碍受试者的自然运动；并且这类方案仍然**昂贵、有功耗消耗、设备操作复杂**，需要**受试者配合**、依赖**受控实验室环境**、需要**粘贴标记点**，属于劳动密集型工作。

为克服可穿戴传感器采集系统的局限，研究者已开发出**基于相机的系统**来监测人体下肢运动。**深度学习框架的涌现与相机技术的发展**，为从视频帧信息中预测下肢角速度创造了新的机会。近期研究 [13, 14] 提出了使用 **OpenPose 库**的基于相机的系统，能以良好精度记录下肢关节角度。**然而，基于 OpenPose 库的系统的估计精度依赖于真值 2D 关键点数据集；关键点是人工标注的，且并不位于关节中心位置。因此，用它来估计下肢角速度可能并不可靠。** 此外，这类方法被开发用于**仅在侧视图下**监测受试者的关节角度，**无法在受试者的正面与背面视角下追踪关节角度**。

在**我们的前期研究**中，已提出一种深度卷积神经网络（CNN）架构来测量髋、膝、踝关节的下肢角速度，该架构在**不同相机视角**下都能提供良好的估计精度 [15]。**然而，这些预训练的深度 CNN 框架仍因估计结果的时间延迟而表现出较高误差。**

在本研究中，我们提出一种基于视觉的系统，使用**深度卷积神经网络结合双向 LSTM / GRU 网络（DeepConvLSTM / DeepConvGRU）**来估计下肢关节运动学。所提方法基于 **RGB 相机系统**，在**深蹲**与**跑步机行走**动作中检测髋、膝、踝关节（左右两侧）的角速度。我们量化了 **DeepConvGRU** 模型的估计性能，并与动作捕捉系统采集的真值关节角速度进行了比较。

---

## II. 方法

### A. 实验设备与协议

为构建视频数据集，我们招募了 **33 名健康成年人（25 名男性，8 名女性）**。实验在一间同时安装了**光学动作捕捉系统**与**小米米家相机 4K 数字相机系统**的实验室内进行。**图 1** 展示了实验期间相机的布置方式，用于获取视频数据集。

作为记录人体关节角度的金标准，我们使用了一套**六相机光学动作捕捉系统**，分辨率为 **1280×1024**（约 130 万像素），频率为 **120 Hz**。依照 **Plug-in-gait 标记点集（marker set）**，在人体肢体上粘贴了**十五个反光标记点**，部位包括：骨盆、髋、膝、大腿、小腿、脚趾、踝与足跟。光学动作捕捉系统追踪粘贴于人体的反光标记点，并给出所有关节准确的三维角度与三维位置。关节角速度通过对关节角度**求导**计算得到。**捕捉区域为地面 2.000 m × 3.685 m。**

数字相机系统使用了**六台相机**，分辨率为 **1920×1080**，帧率为 **60 Hz**。六台相机被放置在受试者的**前方、后方与侧面**，以在 **30 秒内**采集深蹲与跑步机行走动作的视频。我们设计了一个**数字电路来同步光学动作捕捉相机与数字相机系统**。由于光学动作捕捉相机系统的采样率为 **120 帧/秒**（是数字相机帧率的两倍），因此**标注数据被从 120 Hz 降采样至 60 Hz**。

> **图 1**：实验装置。光学动作捕捉系统追踪标记点、记录关节角速度；数字相机采集受试者在深蹲与跑步机行走动作中的视频信息。

### B. 用于估计下肢关节角速度的深度学习框架

我们提出一种基于 **CNN + LSTM** 神经网络的深度学习模型（**DeepConvLSTM**），用于估计下肢关节（左右两侧的髋、膝、踝）的角速度。**卷积神经网络用于从动作视频中提取（空间域）特征，而循环神经网络用于提供时间序列预测（时间域）。** 以监督方式，DeepConvLSTM 架构学习 **RGB 图像信息作为输入**与**关节角速度作为输出**之间的时间依赖关系。

具体流程如下：

1. 首先，应用**基于深度学习的特征选择技术（YOLOv4）**来检测人体并移除动作视频中的**非信息性背景** [16]。
2. 然后，**基于得到的人体边界框对视频帧进行裁剪，并缩放至 448 × 448**。
3. 接着，将 **27 帧视频信息**作为所提 DeepConvLSTM 网络的输入。
4. 随后，训练网络以输出对应的**六个关节角速度**。

本工作中，我们采用**卷积神经网络（ResNet18）结合 Bi-LSTM / GRU** 来预测关节角速度估计。**在每一个时间步，CNN 层提取的空间特征被连接到 LSTM/GRU 网络。** 最后，**全连接（FC）层**在每个时间步给出**六个关节角速度的预测值**（左/右髋屈伸、膝屈伸、踝背屈/跖屈），如**图 2** 所示。

> **图 2**：DeepConvLSTM 架构。深度 CNN 从动作视频中提取（空间域）特征，LSTM 用于提供时间序列（时间域）的关节角速度预测。

### C. DeepConvLSTM 模型实现

DeepConvLSTM 使用 Python 的 **PyTorch** 库实现。模型使用**自适应矩估计（Adam）**作为优化器，使用**平均绝对误差（MAE）**作为损失函数。计算平台配置为 **CUDA 10.2** 库、**GTX 2080Ti** 显卡，操作系统为 **Ubuntu 18.04.5 LTS**。

超参数设置：

| 超参数 | 值 |
|---|---|
| 学习率（learning rate） | **0.0001** |
| 批大小（batch size） | **8** |
| 输入序列长度 | **27 张图像** |
| 训练轮数（epochs） | **20** |

**数据划分**：为训练所提模型，使用 **7 名受试者**在深蹲与跑步机行走动作中的动作视频用于**训练**，**3 名受试者**的动作视频用于**验证**，**3 名受试者**用于**测试**。

> [译者注] 33 名受试者中实际仅使用了 7 + 3 + 3 = 13 名的数据，原文未说明其余受试者的用途。

---

## III. 实验结果

本节展示所提 DeepConvLSTM / DeepConvGRU 网络在预测深蹲与跑步机行走动作中下肢关节角速度的结果。由光学动作捕捉系统获得的角速度与由 DeepConvLSTM、DeepConvGRU 网络估计的角速度之间的差异，通过**归一化相关系数（r）**与**平均绝对误差（MAE）**进行评估。

为说明所提模型的估计性能，**图 3** 展示了深蹲动作中下肢关节的估计角速度与真值角速度。与真值角速度相比，**使用 DeepConvGRU 估计的角速度优于 ResNet18 与 DeepConvLSTM 的估计结果**。**图 4** 描绘了跑步机行走期间的预测角速度。

> **图 3**：深蹲动作中受试者髋、膝、踝关节的估计角速度示例。真值（红线）由光学动作捕捉系统采集；ResNet18（黑线）、GRU（橙线）与 Bi-LSTM（青线）为估计的关节角速度。
>
> **图 4**：跑步机行走动作中受试者髋、膝、踝关节的估计角速度图。真值（红线）由光学动作捕捉系统采集；ResNet18（黑线）、GRU（橙线）与 Bi-LSTM（青线）为估计的关节角速度。

**表 I** 详细报告了 GRU、Bi-LSTM 与 ResNet18 三个网络在深蹲与跑步机行走中三个关节的**平均 r**。**GRU 网络的平均 r 在三个模型中表现出强相关性（深蹲 0.93，跑步机行走 0.92）。**

**表 I：所提模型在深蹲与跑步机行走动作中预测关节角速度的平均 r**

| 模型 | 深蹲（Squat） | 跑步机行走（Walking on treadmill） |
|---|---|---|
| ResNet18 | 0.87 | 0.90 |
| Bi-LSTM | 0.92 | 0.91 |
| **GRU** | **0.93** | **0.92** |

**表 II** 给出了三个模型在**深蹲**动作中预测**逐关节**（髋、膝、踝）角速度的**平均 r 与 MAE**。在髋、膝、踝三个关节中，**使用 GRU 模型估计的膝关节角速度在平均 r 上表现出更好的精度**。**GRU 网络在深蹲动作的所有关节估计中都给出了比 Bi-LSTM 与 ResNet18 网络更小的 MAE 误差。**

**表 II：GRU、Bi-LSTM 与 ResNet18 模型在深蹲动作中髋、膝、踝关节角速度的平均 r（MAE）**

| 模型 | 髋关节（Hip） | 膝关节（Knee） | 踝关节（Ankle） |
|---|---|---|---|
| ResNet18 | 0.87（44.18） | 0.94（30.92） | 0.90（6.09） |
| Bi-LSTM | 0.91（18.48） | 0.96（21.56） | 0.89（6.23） |
| **GRU** | **0.94（15.13）** | **0.96（18.09）** | **0.92（5.83）** |

**表 III：GRU、Bi-LSTM 与 ResNet18 模型在跑步机行走动作中髋、膝、踝关节角速度的平均 r（MAE）**

| 模型 | 髋关节（Hip） | 膝关节（Knee） | 踝关节（Ankle） |
|---|---|---|---|
| ResNet18 | 0.80（27.20） | 0.94（37.48） | 0.84（6.05） |
| Bi-LSTM | 0.89（21.05） | 0.95（24.80） | 0.85（5.70） |
| **GRU** | **0.91（15.97）** | **0.96（20.47）** | **0.89（4.15）** |

**表 III** 展示了跑步机行走动作中髋、膝、踝关节角速度的平均 r 与 MAE。**GRU 网络的平均 MAE 在深蹲与跑步机行走中都给出了比 Bi-LSTM 与 ResNet18 网络更小的误差。**

**表 IV** 展示了三名受试者在**侧视图（lateral）**与**正视图（frontal）**下角速度的平均 r。

**表 IV：三名受试者在侧视图与正视图相机视角下角速度的平均 r**

| 受试者 | 侧视图（Side view） | 正视图（Front view） |
|---|---|---|
| S18 | 0.92 | 0.92 |
| S30 | 0.91 | 0.87 |
| S31 | 0.93 | 0.92 |
| **平均** | **0.92** | **0.89** |

**侧视图与正视图的平均 r 分别为 0.92 与 0.89。** 结果表明，**所提模型在两种视角下都具有更高的估计精度，且不受相机位置与角度的影响**。

---

## IV. 讨论

本研究提出了一种基于视觉的系统，使用 **DeepConvLSTM / DeepConvGRU 网络**来估计深蹲与跑步机行走动作期间的下肢关节角速度。我们使用**光学动作捕捉系统**记录角速度、使用**小米米家相机 4K 数字相机系统**采集动作视频来开展实验研究。所提架构以**动作视频信息为输入**、以**对应的关节角速度为输出**进行训练。为训练所提模型，使用 **7 名受试者**在深蹲与跑步机行走中的动作视频用于训练，**3 名**用于验证，**3 名**用于测试。由光学动作捕捉系统获得的角速度与由 DeepConvLSTM、DeepConvGRU 网络估计的角速度之间的差异，用 **r 与 MAE** 进行评估。**r 衡量估计角速度与真值角速度之间的相关性。** 使用 DeepConvLSTM / DeepConvGRU 网络预测的关节角速度，与使用光学动作捕捉系统获得的角速度在深蹲与跑步机行走动作中**表现出相当良好的一致性**。

如**表 I** 所报告，**DeepConvGRU 模型在三个深度学习架构中提供了最高的精度——深蹲 0.93，跑步机行走 0.92**。我们针对深蹲与跑步机行走动作中的**髋、膝、踝逐关节角速度**，评估了独立的 DeepConvLSTM / DeepConvGRU 模型的估计性能。**DeepConvGRU 网络在两种动作中都给出了更高的 r。** 在三个关节角速度中，**估计的膝关节表现出更高的精度（深蹲 0.96，跑步机行走 0.96）**，高于估计的髋关节与踝关节角速度。

作者给出的解释是：

- **膝关节在矢状面内的运动在受试者之间更为稳定**；
- **踝关节具有更高的自由度**；
- **髋关节的估计精度可能受到受试者体格测量差异的影响**，也可能受到**髋关节处标记点波动所引起的运动伪影**的影响。

三名受试者在**侧视图与正视图**相机视角下角速度的平均 r 如**表 IV** 所示。**侧视图与正视图的平均 r 均大于 0.90**，表明 DeepConvGRU 网络在两种相机视角下都提供了相对更高的预测精度，**不受相机位置与角度影响**。

关于 MAE：深蹲与跑步机行走动作中髋、膝、踝关节角速度的平均 MAE，**GRU 网络的平均 MAE 比 Bi-LSTM 与 ResNet18 网络更小**，详见表 II 与表 III。

**然而，所提模型仍表现出较高的 MAE 误差**，原因是：① 由动作捕捉系统采集的**真值角速度本身缺乏精度**；② **标记点在人体上的位置不精确**会产生**偏移值误差**；③ **受试者体格测量值的差异**。

**所提 DeepConvGRU 模型表现出 15 至 20 rad/s 的角速度误差。** 尽管存在这一较高的 MAE 误差局限，但**通过应用序列网络，该局限已在某种程度上被克服**。

### 未来工作（原文自述）

> 在未来的工作中，实验应在**更大的受试者群体、不同年龄组、更多不同速度的日常动作，以及患有步态病理的患者**中开展。所提 DeepConvGRU 模型将被扩展为使用 **transformer 与多头注意力（multi-head attention）网络**，以实现更高精度的角速度估计。确认使用其他精确的动作捕捉系统进行真值标注是必要的。我们的工作将扩展到估计**其他步态参数（步频 cadence、速度 speed、步长 stride length 与步宽 step width）**。

> 🔴 **【本库重点标注】**：这段"未来工作"是**本知识库最重要的单条证据**——
> KIST 团队在 **2022 年**就已经明确宣布将把方法扩展为 **transformer + 多头注意力**。
> 四年后 **Chao 2026 Measurement**（同团队，CNN + Transformer，单目 RGB → 髋/膝/踝角度）**正是这条路线图的兑现**。
> 这直接证明：**Chao 2026 不是偶然的竞品，而是该团队 2022 年就公开预告的既定路线。**

---

## V. 结论

本文提出了一种基于视觉的系统，使用 **Bi-LSTM 与 GRU 循环神经网络**估计**深蹲**与**跑步机行走**动作中的**下肢关节角速度**。所提架构以动作视频信息为输入、以对应的关节角速度为输出进行训练。

**使用 DeepConvGRU 网络估计的关节角速度，与使用动作捕捉系统测量的关节角速度强相关**，并在**不同相机视角**下提供了更高的精度。这些估计结果证明了**在不使用可穿戴传感器的情况下、仅用视频信息监测关节角速度的潜力**。

**然而，所提模型仍表现出 15 至 20 rad/s 的角速度误差**，原因是动作捕捉系统采集的真值角速度缺乏精度。

为了实现**更高精度的角速度估计**，我们的工作将扩展为使用 **transformer 与基于多头注意力的网络**。所提基于视觉的方法也将适用于估计**其他步态参数（步频、速度、步长与步宽）**。

---

## 致谢

本研究由**韩国科学技术研究院机构计划（项目号 2E31051）**支持，并由**韩国政府（科学技术信息通信部、产业通商资源部、保健福祉部、食品药品安全部）资助的韩国医疗器械开发基金**支持（项目编号：1711139131）。

---

## 参考文献

1. P. Messier, C. Legault, C. Schoenlank, J. Newman, D. Martin, and P. Devita, "Risk factors and mechanisms of knee injury in runners," *Medicine Science in Sports Exercise*, vol. 40, no. 11, pp. 1873–1879, Nov. 2008.
2. R. Ibrahim, I. Kingma, V. de Boode, G. S. Faber, and J. H. van Dieën, "Angular velocity, moment, and power analysis of the ankle, knee, and hip joints in the goalkeeper's diving save in football," *Frontiers in Sports and Active Living*, vol. 2, Feb 2020.
3. J. Dicharry, "Kinematics and kinetics of gait: From lab to clinic," *Clinics in Sports Medicine*, vol. 29, no. 3, pp. 347–364, Jul 2010.
4. M. G. Browne and J. R. Franz, "Ankle power biofeedback attenuates the distal-to-proximal redistribution in older adults," *Gait Posture*, vol. 71, pp. 44–49, Jun 2019.
5. N. L. Pieper, M. D. Lewek, and J. R. Franz, "Can shank acceleration provide a clinically feasible surrogate for individual limb propulsion during walking," *Journal of Biomechanics*, vol. 98, p. 109449, Jan 2020.
6. Jiang, Gholami, Khoshnam, Eng, and Menon, "Estimation of ankle joint power during walking using two inertial sensors," *Sensors*, vol. 19, no. 12, p. 2796, Jan 2019.
7. J. F. Hafer and R. F. Zernicke, "Propulsive joint powers track with sensor-derived angular velocity: A potential tool for lab-less gait retraining," *Journal of Biomechanics*, vol. 106, p. 109821, Jun 2020.
8. M. Ajdaroski, R. Tadakala, L. Nichols, and A. Esquivel, "Validation of a device to measure knee joint angles for a dynamic movement," *Sensors*, vol. 20, no. 6, p. 1747, Mar 2020.
9. I. Tak, W. Wiertz, M. Barendrecht, and R. Langhout, "Validity of a new 3-d motion analysis tool for the assessment of knee, hip and spine joint angles during the single leg squat," *Sensors*, vol. 20, no. 16, p. 4539, Aug 2020.
10. V. Hernandez, D. Dadkhah, V. Babakeshizadeh, and D. Kulić, "Lower body kinematics estimation from wearable sensors for walking and running: A deep learning approach," *Gait Posture*, vol. 83, Jan 2021.
11. J. S. et al, "Prediction of lower extremity multi-joint angles during overground walking by using a single imu with a low frequency based on an lstm recurrent neural network," *Sensors*, vol. 22, no. 1, p. 53, Dec 2021.
12. Lim, Kim, and Park, "Prediction of lower limb kinetics and kinematics during walking by a single imu on the lower back using machine learning," *Sensors*, vol. 20, no. 1, p. 130, Dec 2019.
13. A. Schmitz, M. Ye, R. Shapiro, R. Yang, and B. Noehren, "Accuracy and repeatability of joint angles measured using a single camera markerless motion capture system," *Journal of Biomechanics*, vol. 47, no. 2, pp. 587–591, Nov 2014.
14. E. D'Antonio, J. Taborri, E. Palermo, S. Rossi, and F. Patanè, "A markerless system for gait analysis based on openpose library," in *2020 IEEE International Instrumentation and Measurement Technology Conference (I2MTC)*, 2020, pp. 1–6.
15. **K. S. Kumar, A. Jamsarndorj, D. Jung, D. Lee, J. Kim, and K. R. Mun, "Vision-based human joint angular velocity estimation during squat and walking on a treadmill actions," in *2021 43rd IEEE Engineering in Medicine Biology Society (EMBC)*, 2021, pp. 2186–2190.**
16. A. Bochkovskiy, C. Wang, and H. Liao, "Yolov4: Optimal speed and accuracy of object detection," *ArXiv*, vol. abs/2004.10934, 2020.

---

## 附录：本论文对 ViKiT（我方）的可复用要点

> 本节为**译者补充**，非原文内容。

### 1. 可直接引用的具体数字（真实、已验证）

| 项 | 值 | 我方用途 |
|---|---|---|
| 受试者 | 33 名健康成年人（25 男 / 8 女） | 样本量对照 |
| 光学动捕 | 6 相机，1280×1024，**120 Hz**，Plug-in-gait，15 反射标记 | 真值规格 |
| RGB 相机 | 6 台，1920×1080，**60 Hz**，小米米家 4K | 相机规格 |
| 输入表征 | **27 帧** → 裁剪至 **448×448** | 短窗 baseline 配置 |
| 人体检测 | **YOLOv4** | 预处理对照 |
| Backbone | **ResNet18** | 轻量 backbone 对照 |
| 时序模块 | **Bi-LSTM / GRU** | 时序 baseline |
| 输出 | **6 个关节角速度**（左右髋/膝/踝） | 输出维度对照 |
| 训练 | Adam，lr 1e-4，batch 8，**20 epochs**，MAE loss | 训练配置 |
| 划分 | **7 训练 / 3 验证 / 3 测试**（按受试者） | **跨被试划分**已有先例 |
| 最佳结果 | r = **0.93**（深蹲）/ **0.92**（跑步机）；膝最优 **0.96/0.96** | 精度基线 |
| 跨视角 | 侧视 0.92 / 正视 0.89 | 跨视角基线 |
| 残余误差 | **15–20 rad/s** | 方法天花板 |

### 2. 三条关键情报

1. **跨被试划分（7/3/3）在 2022 年已实现** → 我方 `04_实验设计/数据集与协议.md` 的跨被试协议**必须**采用，否则达不到同类工作的严格度。
2. **方法天花板是 15–20 rad/s 的角速度误差** → 这给"Transformer 是否值得换"提供了可量化的对照基线。
3. 🔴 **"未来工作将扩展为 transformer + 多头注意力"是 Chao 2026 的公开预告**（见 §IV 未来工作）——**KIST 团队 2022 年就公开了这条路线**。这使 Chao 2026 从"意外竞品"变为"既定路线的兑现"，**我方必须在 Related Work 中正面把这条 4 年时间线写出来**。

### 3. 与我方的结构性差异

| 维度 | Kumar 2022 EMBC | ViKiT（我方） |
|---|---|---|
| 输出物理量 | **角速度**（angular velocity） | **角度**（+ 速度 + 加速度） |
| 任务定义 | 回归 | **Seq2Seq 翻译** |
| 输入窗口 | **27 帧**（≈0.45 s @60fps，短上下文） | **长序列** |
| 时序模块 | Bi-LSTM / GRU | Spatio-Temporal **Transformer** + cross-attention |
| 物理约束 | 无（仅 MAE loss） | **6 项 grammar loss** |
| 输出表征 | 连续向量 | **Kinematic tokens** |
| 关节 | 下肢 髋/膝/踝 | **上肢 腕/肘/肩** |
| 提前预测 | ❌ 无 | ✅ 100/200/300 ms |
