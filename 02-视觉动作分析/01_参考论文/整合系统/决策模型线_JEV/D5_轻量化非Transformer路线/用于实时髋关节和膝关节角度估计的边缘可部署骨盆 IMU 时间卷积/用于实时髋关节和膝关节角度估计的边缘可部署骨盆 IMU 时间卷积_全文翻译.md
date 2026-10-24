# 用于实时髋关节和膝关节角度估计的边缘可部署骨盆 IMU 时间卷积

![](images/bd168c690f162478c95588b17ee657104e593b4031252801e43d21a3eb40e440.jpg)

Shu Xiaolong , Luo Shengli, Wang Xu , Qin Yiyao, Liu Qiwen, Yu Hongliu <sup>∗</sup>

Institute of Intelligent Rehabilitation Engineering, University of Shanghai for Science and Technology, Shanghai, 200093, China (上海理工大学智能康复工程研究所，中国上海，200093)

Shanghai Engineering Research Center of Assistive Devices, Shanghai, 200093, China (上海康复器具工程技术研究中心，中国上海，200093)

School of Health Science and Engineering, University of Shanghai for Science and Technology, Shanghai, 200093, China (上海理工大学健康科学与工程学院，中国上海，200093)

## 文章信息

关键词：骨盆运动学 惯性测量单元 时间卷积网络 关节角度估计 边缘计算

## 摘 要

下肢关节运动学的精确和低延迟估计对于动力假肢控制至关重要，特别是对于在行走过程中严重依赖代偿性骨盆运动的髋关节离断（hip disarticulation, HD）截肢者而言。本研究提出了一种仅使用单个骨盆惯性测量单元（inertial measurement unit, IMU）来估计连续双侧髋关节和膝关节屈曲角度的边缘可部署框架。该任务被构建为从六通道骨盆运动学到四个下肢关节角度输出的时间回归。在统一的训练、验证和 ONNX 部署协议下对八种模型族进行了基准测试，以评估精度与延迟之间的权衡。Transformer 取得了最低的误差（RMSE = 1.698°，MAE = 1.290°），而时间卷积网络（temporal convolutional network, TCN）在 Raspberry Pi 5 上提供了最佳的边缘部署平衡（RMSE = 1.921°，MAE = 1.484°，p95 延迟 = 0.229 ms）。泛化性分析表明，髋关节角度估计相对稳定，而膝关节角度估计对速度依赖性和受试者依赖性误差的贡献更大。这些发现表明，骨盆 IMU 驱动的时间卷积可作为未来动力 HD 假肢研究中实用的初步感知模块，但仍需在 HD 截肢者和真实世界行走条件下进行进一步验证。

## 1. 引言

髋关节离断（Hip disarticulation, HD）是下肢截肢中位置最高（最靠近近端）的截肢水平，对HD患者而言，恢复自然步态仍然是最具挑战性的任务之一 [1]。对于被动式HD假肢，由于假肢髋关节缺乏主动力矩，使用者往往不得不通过大幅度的骨盆摆动来带动假肢完成摆动和迈步，这造成了极高的体力消耗；对于需要充足耐力以维持稳定行走的的老年使用者而言，这一点尤为困难 [2,3]。此外，长期通过过度的骨盆倾斜和旋转来实现身体前移，可能会增加下背部负荷，并进一步加剧脊柱或软组织损伤的风险 [4]。为了减轻HD患者的身体负担并提高其行动能力，在假肢髋关节处引入动力驱动与智能控制是一个潜在的发展方向 [5–7]。然而，动力髋关节控制必须能够推断使用者的运动意图，而对于HD患者而言，完整髋关节及远端下肢节段的缺失极大地限制了可用的生理信号和运动线索，使得可靠的用户输入识别成为一项关键挑战 [8,9]。

近年来，三维步态生物力学以及执行器/传感器技术的进步催生了一系列行之有效的下肢假肢控制系统 [10,11]。例如，在楼梯和斜坡等复杂地形中，动力膝假肢能够主动辅助推进并减少代偿动作，从而降低代谢消耗和关节负荷 [12,13]。相比之下，针对HD的动力髋假肢研究仍处于起步阶段，关于控制策略和可用输入信号的证据仍然有限 [14,15]。Brannen et al. 开发了一种前置式微处理器控制的动力髋假肢样机，该样机已完成机械测试和功能验证，但要在临床常规应用中推广仍需进一步优化 [16]。类似地，Botros et al. 研究了一种四连杆动力髋假肢样机，并指出了其在多样化场景下面临的自适应性和稳定性挑战 [17]。

现有用于为动力HD系统生成关节轨迹（角度/角速度）或参考力矩的方法大致可分为三种范式：(1) 健侧镜像/回声参考；(2) 步态事件或相位驱动、具有有限状态参考曲线的分层控制；以及 (3) 基于可控人体信息（特别是骨盆运动学）的回归/学习型参考。

首先，Ueyama et al. 构建了一个具有动力髋-膝关节的原型，并采用了镜像/回声控制（mirroring/echo control）策略：以健侧运动作为提示线索，利用虚拟弹性储能-释能机制驱动假肢摆动，在稳态条件下产生了相对自然的步态外观 [18,19]。然而，此类方法高度依赖于步态周期性。在非稳态日常活动中（例如变速、避障、转弯和不对称任务），通常需要额外的基于规则的补偿来维持可用性。此外，这些方法对从健侧感知到响应生成的端到端链路非常敏感：延迟和相位误差可能会降低用户的实时调节能力，并可能损害闭环助力中的稳定性。

其次，为了提高适应性并减少对健侧镜像的依赖，人们越来越多地采用“高-中-低”分层架构：步态事件检测或相位分割驱动有限状态机（或子相位机）；在每个子相位内，定义目标髋关节状态（例如期望角度/角速度），并通过阻抗或力矩控制进行跟踪。Golshan et al. 为 HD 提出了一种“Power Hip”控制框架，利用内部传感器（IMU、负载传感器、编码器等）识别步态子相位，并切换目标髋关节状态和阻抗参数，从而实现了一定程度的速度自适应 [20]。同样，对于集成了动力髋关节与智能微处理器膝关节的系统，在意图识别尚不够可靠时，原型机通常采用预编程的髋关节角度/角速度曲线进行验证；例如，Bader et al. 在原型测试期间将髋关节设置为预定义的角速度曲线，为平地行走模拟生成了固定运动范围的髋关节轨迹 [21,22]。这种“相位机 + 参考曲线/参数”范式的一个常见局限在于其对鲁棒事件检测和反复参数调优的依赖；面对不同的用户、行走速度和任务时，往往需要重新调参或进行规则切换。此外，当从稳态场景扩展到非稳态场景（变速、转弯和复杂地形）时，有限状态转换和固定的参考曲线可能会引起相位漂移、切换抖颤或轨迹不连续，进而对可用性和安全性产生不利影响。

第三种范式旨在利用更可控、更稳定的人体输入来更直接地计算或推断关节参考轨迹。一些研究仍将步态相位/意图作为中间变量。例如，Chen 等人提出了一种基于 TCN 的步态相位解码器，该解码器通过有限的传感估计相位，为随后的参考生成和控制提供时间基线 [23]。Li 等人采用分层控制进行摆动辅助，并利用二次 Bézier 曲线构建了理想的摆动期辅助力矩参考，从而能够根据速度和步态特征进行调整 [24]。其他工作则侧重于阻抗/刚度调制，通过可变关节刚度或半主动策略改变参考响应特性，以适应不同的步态需求 [25]。总体而言，“相位解码 + 曲线参数化”仍然遵循“先相位，后轨迹”的链式过程：一旦相位估计在非稳态场景中受到扰动，参考轨迹也可能会随之漂移。此外，许多方法仅输出单关节（通常是髋关节）的参考，未能充分解决髋-膝协调性或面向控制估计所需的连续双侧关节角度轨迹问题。

此外，Golshan 等人提出了一种基于骨盆运动学计算髋关节角度的方法：利用相关性和线性回归，从骨盆倾斜/旋转和支撑时间特征中逐步估计关键髋关节角度和关键时间点，在某些指标上实现了高相关性（高达约 0.94）且与动作捕捉结果相比差异较小 [26]。这项工作支持了骨盆运动学可以作为髋关节运动有效观测的这一前提；然而，它仍然主要基于手工特征和逐步回归，侧重于关键事件/点，而不是提供适用于实时边缘部署的连续双侧髋/膝轨迹。受这些研究空白的启发，本研究提出了一种针对边缘部署的端到端深度时序回归框架。与基于云端的推理相比，边缘执行减少了往返延迟，降低了对网络连接的依赖，并支持保护隐私的辅助控制。该模型仅使用 6 通道骨盆 IMU 序列，直接学习从骨盆运动学到连续双侧髋/膝屈曲/伸展角度轨迹的映射，避免了显式步态周期分割、复杂的相位状态机规则以及健侧镜像，从而为动力假肢控制中的参考生成和跟踪提供了更直接的信号接口。

![](images/622e81a57140a6e9785d33fce859db7d709a1c35affd9a58373018e030ade47e.jpg)  
Fig. 1. 实验设置及骨盆、大腿和小腿部位 IMU 布置示意图。

在本研究中，健康受试者的步态被用作学习这种骨盆-髋/膝时间耦合的初始规范参考，这与将假肢辅助行走恢复至更符合生理特性的下肢运动学的目标是一致的。本文的主要贡献如下：

(1) 我们提出了一个因果时序回归框架，仅通过单组六通道骨盆 IMU 序列即可预测连续的双侧髋/膝屈曲/伸展角度，为未来的 HD 假肢（HD prosthesis）研究提供了一个最小传感器配置的感知模块。

(2) 在统一的训练、评估和 ONNX 部署协议下，我们对八种代表性模型族进行了基准测试，并使用平均延迟和尾部延迟指标表征了精度与延迟之间的权衡。

(3) 我们实现了 PyTorch→ONNX→ONNX Runtime 流程，并在 Raspberry Pi 5 上进行了部署基准测试。我们进一步使用特定关节误差、速度与受试者维度的泛化性、LOSO 验证以及轨迹级波形指标对所选的 TCN 模型进行了分析，以阐明其可部署性和泛化边界。

## 2. 数据采集与处理

数据按受试者和行走速度进行组织。共招募了 8 名健康受试者（4 名男性，4 名女性），并在 8 种速度条件（速度标签：1.5–5.0 km/h）下进行了记录。受试者年龄为 24.75 ± 1.04 岁，身高为 1.685 ± 0.048 m，体重为 54.06 ± 6.81 kg。所有受试者均签署了知情同意书，且该研究方案已获得上海健康医学院伦理委员会（Ethics Committee of Shanghai University of Medicine & Health Sciences，编号：2019ZYXM104）的批准。在每种速度条件下，受试者行走两分钟，同时 NORAXON IMU 系统以 100 Hz 的频率记录来自骨盆、大腿和小腿单元的运动学信号。在本研究中，骨盆 IMU 数据流作为模型输入，而由 NORAXON 系统导出的双侧髋/膝屈曲/伸展角度则用作基于 IMU 的参考关节角度。实验装置和传感器放置如图 Fig. 1 所示。

输入特征包括三轴骨盆旋转（Pelvis Rot X/Y/Z）和三轴骨盆加速度（Pelvis Accel Sensor X/Y/Z）。输出目标包括双侧髋/膝屈曲/伸展角度（Hip Flex ion LT/RT，Knee Flexion LT/RT），单位为度。导出后，对所有选定的输入和输出通道进行了缺失值或非有限值的检查。在构建时间窗口之前，剔除了在 6 个骨盆 IMU 输入通道或 4 个目标关节角度通道中任意一个存在缺失值或非有限值的样本。除 NORAXON 系统内置的信号处理和方向估计算法外，导出后未施加额外的低通滤波。在数据划分或模型训练过程中，未采用基于步态周期的手动伪影剔除。

![](images/865f4f0609799840fb8328e851e95863c523019eb408be039cc77611a8555b4c.jpg)  
Fig. 2. 每个受试者及步行速度记录内的按时间顺序（时间连续）划分：序列按时间先后顺序划分为训练集、验证集和测试集片段。

为了反映无法获取未来步态周期且显式分割可能不可靠的实际部署场景，我们未对模型训练和评估进行步态周期分割。相反，在每个连续的“受试者—速度”序列中，我们执行了按时间顺序的划分：前 70% 用于训练，接下来的 10% 用于验证，剩余的 20% 用于测试，如图 Fig. 2 所示。这种划分保留了时间连续性，并避免了时间序列数据中随机打乱可能导致的信息泄露。

## 3. 模型训练与选择

## 3.1. 候选模型与训练设置

为了在估计精度、实时性能和部署复杂度之间实现可解释的权衡，我们对比了八种具有代表性的模型：ridge\_linear（线性基线）、mlp（非线性回归基线）、cnn1d（局部时间模式）、tcn（用于长程依赖的膨胀卷积）、gru（轻量级门控循环神经网络，RNN）、lstm\_uni（经典 RNN 基线）、bilstm（用于离线拟合的双向上下文）以及 transformer\_base（基于注意力的基线）。卷积模型（CNN/TCN）具有高度可并行性，通常更利于边缘推理；RNN 变体通过循环计算显式建模时间依赖；引入 Transformer 是为了评估在相同输入/输出设置下注意机制的潜在收益。在单骨盆 IMU 输入约束下，ridge\_linear 模型作为相同输入的经典回归基线被纳入对比。Table 1 总结了架构亮点与关键超参数。

## 3.2. 窗口化训练与 ONNX 基准测试协议

所有模型均采用固定长度的滑动窗口进行训练。窗口长度为 WINDOW\_SIZE = 50（0.5 s），步长为 1，其中每个输入窗口为一个 $5 0 \times 6$ 的骨盆 IMU 序列，输出为该窗口最后一个时间步的 4D 关节角度向量。该设计具有因果性且兼容在线使用，因为仅需要窗口内的历史样本。标准化仅使用训练统计数据：在展平训练 IMU 序列后计算通道维度的均值和标准差，并在前向传播过程中将输入归一化为 $( x - \mu ) / \sigma$。

训练在 PyTorch 中实现，DataLoader 设置与代码保持一致：训练批次大小（batch size）为 256，验证/测试为 512，num\_workers = 4。当使用 CUDA 时，启用 pin\_memory；否则禁用。我们采用了 Adam 优化器（学习率 $1 0 ^ { - 3 }$，权重衰减 $1 0 ^ { - 5 } )$ 与 MSE 损失函数。学习率调度采用 ReduceLROnPlateau（衰减因子为 0.5，耐心值 patience 为 10），并在验证集损失上采用早停机制（patience 为 15）来选取用于评估和部署的最佳检查点。

Table 1  
统一滑动窗口回归设置下的候选模型族及关键超参数。

<table><tr><td>Model</td><td>Architecture</td><td>Key hyperparameters</td></tr><tr><td>ridge_linear</td><td>线性回归（linear regression）</td><td>L2 正则化</td></tr><tr><td>mlp</td><td>MLP 回归</td><td>2 层 MLP：隐藏层大小为 128 和 64</td></tr><tr><td>cnn1d</td><td>1D CNN</td><td>3 个 Conv1D 层，卷积核 kernel = 3，通道数 channels: 32→64→128，自适应池化</td></tr><tr><td>tcn</td><td>时间卷积网络（Temporal convolution network）</td><td>4 层，膨胀率 dilations [1,2,4,8]，通道数 channels = 64，卷积核 kernel = 3，dropout = 0.1</td></tr><tr><td>gru</td><td>GRU</td><td>hidden = 64, layers = 2</td></tr><tr><td>lstm_uni</td><td>LSTM</td><td>hidden = 64, layers = 2</td></tr><tr><td>bilstm</td><td>BiLSTM</td><td>hidden = 64, layers = 2</td></tr><tr><td>transformer_base</td><td>Transformer</td><td>2 个编码器层，d_model = 64，nhead = 4，FF = 128</td></tr></table>

对于桌面端 ONNX 基准测试，训练好的 PyTorch 模型被导出为 ONNX 并在相同的输入特征和推理环境下进行评估。各模型通过四个目标输出上的宏平均（macro-average）RMSE 和 MAE 进行比较，同时记录包括均值（mean）、$p 5 0 , p 9 5 ,$ 以及 $p 9 9$ 在内的推理延迟统计指标。桌面端 ONNX 基准测试用于模型族对比和模型选择，而 Raspberry Pi 5 基准测试则用于评估端侧设备上的部署性能。

泛化性统计分析。对于泛化性分析，在排除前 50 个样本以减少预热偏差后，在每个“受试者 × 速度”条件下计算 RMSE。在报告髋关节/膝关节的汇总结果时，对左/右侧输出取平均值。速度效应采用以行走速度作为受试者内因素的跨受试者重复测量方差分析（repeated-measures ANOVA）进行评估，随后进行 Holm 校正的事后检验。在去除按速度划分的平均 RMSE 后，使用残差组内相关系数（ICC）量化速度之外的受试者间变异性。对于面向部署的报告，我们进一步汇总了所有受试者 × 速度条件下的均值、95 和最大 RMSE。

跨受试者与波形级评估。为了进一步评估跨受试者的泛化能力，我们使用所选的 TCN 模型进行了留一受试者法（leave-one-subject-out, LOSO）验证。在每个折叠中，一名参与者的所有记录均被保留作为测试集，而其余参与者则用于模型开发。归一化统计量仅根据每个折叠内的训练参与者数据计算得出。

在预测之后还额外计算了波形级指标，以评估峰值角度误差、峰值时间误差、基于互相关的时间对齐、归一化频谱误差以及波形相关性。这些分析仅在预测后使用步态周期检测，不影响训练、模型选择或数据划分。

## 4. 结果

## 4.1. 桌面端 ONNX 模型对比与 TCN 选择

在相同的输入特征和 ONNX 推理环境下，使用四个目标输出的宏平均 RMSE 和 MAE，以及包括均值、p50、p95 和 p99 在内的推理延迟统计量对八个候选模型进行了比较。具有相同输入的经典基线 ridge\_linear 速度极快但误差较大，可作为单骨盆 IMU 输入设置下的下界参考。MLP 保持了极低的延迟，同时大幅降低了误差，而 CNN1D 在此设置下未能提供良好的权衡。

Table 2  
测试集上的桌面端 ONNX 基准测试结果。

<table><tr><td>模型</td><td>RMSE</td><td>MAE</td><td> $R^2$ </td><td>均值 (ms)</td><td>p50</td><td>p95</td><td>p99</td><td>备注</td></tr><tr><td>ridge_linear</td><td>9.01</td><td>6.98</td><td>0.70</td><td>0.006</td><td>0.005</td><td>0.006</td><td>0.009</td><td>同输入经典基准</td></tr><tr><td>mlp</td><td>2.25</td><td>1.63</td><td>0.98</td><td>0.013</td><td>0.012</td><td>0.015</td><td>0.021</td><td>快速非线性基准</td></tr><tr><td>cnn1d</td><td>2.56</td><td>1.89</td><td>0.97</td><td>0.056</td><td>0.054</td><td>0.061</td><td>0.083</td><td>仅局部模式</td></tr><tr><td>tcn</td><td>2.03</td><td>1.48</td><td>0.98</td><td>0.113</td><td>0.109</td><td>0.116</td><td>0.132</td><td>尾部延迟稳定</td></tr><tr><td>gru</td><td>1.92</td><td>1.37</td><td>0.99</td><td>0.266</td><td>0.258</td><td>0.277</td><td>0.336</td><td>开销较高</td></tr><tr><td>lstm_uni</td><td>1.94</td><td>1.36</td><td>0.99</td><td>0.328</td><td>0.315</td><td>0.347</td><td>0.462</td><td>开销较高</td></tr><tr><td>bilstm</td><td>1.88</td><td>1.33</td><td>0.99</td><td>0.624</td><td>0.601</td><td>0.721</td><td>1.523</td><td>长尾延迟</td></tr><tr><td>transformer_base</td><td>1.81</td><td>1.29</td><td>0.99</td><td>1.053</td><td>1.019</td><td>1.165</td><td>1.602</td><td>精度最高，速度最慢</td></tr></table>

![](images/a70c1149cc428d5f446d13ff980ea042f26eb926002f9183386b5b3cbca13d18.jpg)  
表 3  
图 3. ONNX 部署下的准确率与延迟权衡。x 轴表示推理延迟（ms，对数尺度）；水平误差棒表示从 $p 5 0$ 到 $p 9 9 ,$ 的延迟范围，垂直刻度线表示 $p 9 5 .$。左侧 ${ \bf y } -$ 轴报告宏观 RMSE（圆圈），右侧 y 轴报告宏观 MAE（方块）。

在深度时序模型中，Transformer 取得了最低的误差，但在测试的模型中具有最高的延迟。循环模型也取得了较低的误差，但相比卷积模型或前馈模型产生了更高的推理延迟。相比之下，TCN 提供了良好的准确率与延迟之间的平衡，在亚毫秒级延迟和稳定的尾部表现下实现了低误差（表 2 和图 3）。考虑到需要为实时控制组件（包括感知、安全监控和反馈控制回路）预留计算预算，我们选择 TCN 作为后续以 TCN 为重点的分析和部署实验的默认模型。该选择优先考虑了预测准确率、推理延迟和边缘部署可扩展性之间的实际平衡，而不仅仅是最小化预测误差。

## 4.2. TCN 消融实验

为了分析 TCN 超参数如何影响预测性能和部署开销，我们使用基线配置（隐藏通道数 64、4 个 TCN 层、卷积核大小 3、Dropout 0.1）作为均衡设置，并沿着三个维度进行消融实验：模型容量、时间感受野和正则化强度。表 3 列出了消融配置。为了评估模型容量，smaller\_channels（32）和 larger\_channels（128）改变通道宽度，其中较少的通道减少了容量和计算量，而较多的通道可能会提高拟合能力，但代价是更高的延迟和潜在的过拟合。为了评估时序建模和感受野，我们调整了深度和卷积核大小：fewer\_levels（3 层）和 more\_levels（5 层）缩小或扩大了感受野，而 larger\_kernel（kernel $= 5 )$ 在固定深度下扩大了局部上下文。为了评估正则化，no\_dropout（0）和 higher\_dropout（0.2）移除或增强了 Dropout，以检验其对泛化性和稳定性的影响。所有其他超参数与基线保持一致，以便观察到的差异可以归因于目标因素。

相同 ONNX 部署设置下的 TCN 消融配置。

<table><tr><td>配置</td><td>关键设置</td></tr><tr><td>baseline</td><td>channels = 64, levels = 4, kernel = 3, dropout = 0.1</td></tr><tr><td>smaller_channels</td><td>channels = 32</td></tr><tr><td>larger_channels</td><td>channels = 128</td></tr><tr><td>fewer_levels</td><td>levels = 3</td></tr><tr><td>more_levels</td><td>levels = 5</td></tr><tr><td>larger_kernel</td><td>kernel = 5</td></tr><tr><td>no_dropout</td><td>dropout = 0</td></tr><tr><td>higher_dropout</td><td>dropout = 0.2</td></tr></table>

TCN 消融实验：精度-延迟权衡 (ONNX)  
![](images/8b329152b131505ccad36cb8de1e9f55b6ce1ed1fafef77903e9f00a8d2fce9e.jpg)  
Fig. 4. TCN 消融实验的精度-延迟权衡 (ONNX)。x 轴表示平均推理延迟；水平误差棒表示从 $p 5 0$ 到 $p 9 9$ 的延迟范围，并标注了 $p 9 5$。y 轴报告宏观 RMSE（越低越好）。基线配置以星号标出。

Fig. 4 和 Table 4 总结了宏观 RMSE/MAE 以及 ONNX 推理延迟（均值和分位数）。总体而言，消融实验结果展现出明显的精度-延迟权衡。在模型容量方面，smaller\_channels 增加了误差（RMSE 2.78→2.95），而延迟几乎不变（均值 ≈0.22 ms）；而 larger\_channels 降低了误差（RMSE 2.66，MAE 1.70），但增加了计算开销（均值 0.27 ms，p99 0.28 ms）。在感受野方面，fewer\_levels 降低了延迟（均值 0.18 ms，p99 0.19 ms），但将 RMSE 增加到 $2 . 8 4 ;$ 扩大感受野（more\_levels 和 larger\_kernel）进一步降低了误差（RMSE 分别为 2.63 和 $2 . 6 0 ,$；larger\_kernel 实现了最低的 RMSE），代价是更高的延迟（均值 ≈ 0.25–0.27 ms，p99 ≈ 0.27–0.28 ms）。在正则化方面，no\_dropout 略微降低了延迟（均值 0.20 ms），但增大了误差（RMSE 2.83）并降低了某些关节的拟合优度（例如，髋关节屈曲 Hip Flexion RT $R ^ { 2 } = 0 . 9 4 )$，这表明在该设置下 dropout 正则化是有益的；higher\_dropout 带来的 RMSE 改善有限（2.75），但 MAE 更大（1.87）。

综合考虑整体误差、推理延迟和长尾稳定性 $( p 9 5 / p 9 9 )$，基线配置为 TCN 消融分析提供了一个均衡的参考设置。

Table 4  
TCN 消融实验结果：宏观误差与 ONNX 延迟。

<table><tr><td>配置</td><td>宏平均 RMSE (°)</td><td>宏平均 MAE (°)</td><td>宏平均  $R^2$ </td><td>均值 (ms)</td><td>p50 (ms)</td><td>p95 (ms)</td><td>p99 (ms)</td></tr><tr><td>baseline</td><td>2.780</td><td>1.833</td><td>0.969</td><td>0.222</td><td>0.220</td><td>0.226</td><td>0.237</td></tr><tr><td>smaller_channels</td><td>2.948</td><td>1.957</td><td>0.965</td><td>0.217</td><td>0.215</td><td>0.220</td><td>0.230</td></tr><tr><td>larger_channels</td><td>2.656</td><td>1.704</td><td>0.971</td><td>0.271</td><td>0.269</td><td>0.276</td><td>0.283</td></tr><tr><td>fewer_levels</td><td>2.835</td><td>1.887</td><td>0.966</td><td>0.177</td><td>0.175</td><td>0.181</td><td>0.189</td></tr><tr><td>more_levels</td><td>2.633</td><td>1.710</td><td>0.972</td><td>0.247</td><td>0.245</td><td>0.252</td><td>0.272</td></tr><tr><td>larger_kernel</td><td>2.597</td><td>1.696</td><td>0.973</td><td>0.269</td><td>0.268</td><td>0.275</td><td>0.282</td></tr><tr><td>no_dropout</td><td>2.834</td><td>1.857</td><td>0.965</td><td>0.200</td><td>0.198</td><td>0.205</td><td>0.212</td></tr><tr><td>higher_dropout</td><td>2.752</td><td>1.870</td><td>0.968</td><td>0.222</td><td>0.220</td><td>0.227</td><td>0.239</td></tr></table>

Table 5  
Raspberry Pi 5 设备端部署结果。

<table><tr><td>模型</td><td>宏平均 RMSE (°)</td><td>宏平均 MAE (°)</td><td>宏平均  $R^2$ </td><td>均值 (ms)</td><td>p50 (ms)</td><td>p95 (ms)</td><td>p99 (ms)</td></tr><tr><td>ridge_linear</td><td>8.441</td><td>6.989</td><td>0.6602</td><td>0.019</td><td>0.018</td><td>0.020</td><td>0.029</td></tr><tr><td>mlp</td><td>2.081</td><td>1.634</td><td>0.9741</td><td>0.035</td><td>0.034</td><td>0.042</td><td>0.047</td></tr><tr><td>cnn1d</td><td>2.490</td><td>1.939</td><td>0.9661</td><td>0.116</td><td>0.111</td><td>0.125</td><td>0.158</td></tr><tr><td>tcn</td><td>1.921</td><td>1.484</td><td>0.9779</td><td>0.225</td><td>0.215</td><td>0.229</td><td>0.722</td></tr><tr><td>gru</td><td>1.801</td><td>1.381</td><td>0.9805</td><td>0.846</td><td>0.841</td><td>0.862</td><td>0.959</td></tr><tr><td>lstm_uni</td><td>1.804</td><td>1.351</td><td>0.9799</td><td>0.771</td><td>0.752</td><td>0.784</td><td>1.187</td></tr><tr><td>bilstm</td><td>1.736</td><td>1.317</td><td>0.9819</td><td>1.836</td><td>1.816</td><td>1.941</td><td>2.223</td></tr><tr><td>transformer_base</td><td>1.698</td><td>1.290</td><td>0.9833</td><td>5.299</td><td>5.150</td><td>5.783</td><td>6.018</td></tr></table>

## 4.3. Raspberry Pi 5 设备端部署

我们使用 PyTorch→ONNX→ONNX Runtime 流程将这八个模型部署在 Raspberry Pi 5 上，并对离线推理进行基准测试，以对比边缘端精度和推理延迟（Table 5 和 Fig. 5）。为了与泛化性分析保持一致，边缘端指标是在排除每个受试者 × 速度条件下的前 50 个测试窗口后计算的，以避免预热效应和运行时稳定偏差；每个受试者 × 速度条件对宏平均的贡献均等。总体而言，ridge\_linear 速度极快 $( \mathtt { p 9 5 } = 0 . 0 2 0$ ms)，但误差较大（宏 $\mathrm { R M S E } = 8 . 4 4 1 ^ { \circ } )$，仅作为下界基线。MLP 保持了极低的延迟 $( \mathsf { p 9 5 } = 0 . 0 4 2$ ms)，但精度不如深度时序模型 $( \mathrm { R M S E } = 2 . 0 8 1 ^ { \circ } )$。CNN1D 在精度和延迟方面均不具备竞争力（RMSE = 2.490<sup>◦</sup>, p95 = 0.125 ms）。在时序模型中，transformer\_base 取得了最佳精度（RMSE = 1.698<sup>◦</sup>, MAE = 1.290<sup>◦</sup>），但带来了最大的延迟 $( \mathrm { p 9 5 } = 5 . 7 8 3$ ms)；bilstm 在精度上排名第二 $( \mathrm { R M S E } = 1 . 7 3 6 ^ { \circ } )$，但也大幅增加了推理时间（p95 = 1.941 ms）。相比之下，TCN 在保持相对较低误差的同时，实现了更小且更稳定的尾部延迟 $( \mathrm { R M S E } = 1 . 9 2 1 ^ { \circ }$ , p95 = 0.229 ms)，与 GRU/LSTM 相比（分别为 $\mathrm { p } 9 5 = 0 . 8 6 2 / 0 . 7 8 4$ ms），显著降低了设备端推理开销。需要注意的是，TCN 出现了罕见的尾部延迟尖峰 $( \mathbf { p } 9 9  0 . 7 2 2 ~ \mathrm { m } s )$，但其整体尾部表现仍显著低于 GRU/LSTM 和 Transformer。

## 4.4. 波形定性分析

为了直观地检查所选的 TCN 模型，Fig. 6 展示了将 TCN 预测结果与基于 IMU 的参考髋/膝关节屈曲轨迹进行比较的代表性片段。TCN 在整个步态周期内紧密跟踪角度动态变化，在峰值时序和相位上具有良好的对齐性。误差主要集中在快速转换区域（例如，摆动相开始和结束附近），而整体偏差保持较小，波形形状保持稳定。这种形状一致且相位稳定的特性对于面向控制的应用（包括参考跟踪、相位感知调制和安全监控）尤为重要。为了进一步表征逐点 RMSE、MAE 和 $R ^ { 2 }$ 之外的波形保真度，我们量化了所选 TCN 模型的步态周期级峰值角度误差、峰值时序误差、基于互相关的时间对齐、归一化频谱误差以及波形相关性（Table 6，Panel B）。在四个关节角度输出中，峰值角度误差约为 1–2<sup>◦</sup>，互相关滞后接近于零（平均通常低于一个采样点），且波形相关性保持在较高水平（>0.99），这表明所选 TCN 保留了基于 IMU 的参考轨迹的幅度和时间结构。

![](images/89bd4e167182cc66cc23e528c29df7bf65bf07c90f984655b73ec75bbef177a5.jpg)  
(a) 精度（RMSE/MAE）。

![](images/4e9496a72fdf1f6ad1095bc7b227ad98dac869f8b0cfed844828a7a5f62030fb.jpg)  
Fig. 5. Raspberry Pi 5 端侧基准测试（ONNX Runtime）：(a) 精度和 (b) 延迟。模型按宏平均 RMSE 排序。

## 4.5. 跨速度和关节输出的泛化性

除综合指标外，我们还分析了在“受试者 × 速度”全部条件网格（8 位受试者 × 8 种速度）下的泛化性。对于每个网格单元，在排除前 50 个样本（以消除预热瞬态）后计算 RMSE，然后在时间上进行聚合；进一步对左/右输出取平均，以分别获得髋关节和膝关节的简明汇总。Fig. 7 直观展示了所得的误差模式。总体而言，髋关节估计在各条件下更加稳定，而膝关节估计表现出更大的变异性，在一部分受试者和极端速度下误差增加更为显著。

为了在考虑受试者内部相关性的同时量化速度效应，我们以步行速度作为受试者内因素，对单元级 RMSE 进行了重复测量方差分析（repeated-measures ANOVA）（Fig. 8(a)）。对于髋关节 RMSE，速度效应并不显著 $( F ( 7 , 4 9 ) \ = \ 1 . 0 8 , \ p \ = \ 0 . 3 9 4 ) _ { \mathrm { \scriptsize ~ \cdot ~ } }$ ，这表明在该数据集中髋关节角度估计对速度变化相对不敏感。相比之下，膝关节 RMSE 表现出显著的速度主效应 $( F ( 7 , 4 9 ) = 6 . 3 2 , p = 2 . 6 4 \times 1 0 ^ { - 5 } , \eta _ { \upsilon } ^ { 2 } \approx 0 . 4 7 )$ 。平均趋势表明，在较低速度下膝关节 RMSE 较高，而在中等至较快速度下误差较低。

Table 6  
所选 TCN 模型的特定关节逐点及波形级性能。  
Panel A. 逐关节逐点性能

<table><tr><td>关节角度</td><td>RMSE (°)</td><td>MAE (°)</td><td> $R^2$ </td><td>备注</td></tr><tr><td>髋屈曲 LT</td><td>1.494</td><td>1.071</td><td>0.9838</td><td>速度级宏平均</td></tr><tr><td>髋屈曲 RT</td><td>1.657</td><td>1.121</td><td>0.9797</td><td>速度级宏平均</td></tr><tr><td>膝屈曲 LT</td><td>1.921</td><td>1.411</td><td>0.9901</td><td>速度级宏平均</td></tr><tr><td>膝屈曲 RT</td><td>1.975</td><td>1.391</td><td>0.9892</td><td>速度级宏平均</td></tr></table>

Panel B. 步态周期级波形指标

<table><tr><td>关节角度</td><td>峰值角度误差 (°)</td><td>时序误差 (% cycle)</td><td>XCorr 滞后 (samples)</td><td>频谱误差</td><td>波形相关性</td></tr><tr><td>髋屈曲 LT</td><td>1.31 ± 1.17</td><td>1.77 ± 2.87</td><td>0.00 ± 0.00</td><td>0.069 ± 0.046</td><td>0.994 ± 0.014</td></tr><tr><td>髋屈曲 RT</td><td>1.23 ± 1.10</td><td>3.33 ± 11.26</td><td>0.07 ± 0.32</td><td>0.069 ± 0.047</td><td>0.994 ± 0.010</td></tr><tr><td>膝屈曲 LT</td><td>1.48 ± 1.30</td><td>1.02 ± 3.71</td><td>0.26 ± 0.55</td><td>0.068 ± 0.073</td><td>0.993 ± 0.026</td></tr><tr><td>膝屈曲 RT</td><td>1.73 ± 1.41</td><td>1.08 ± 4.28</td><td>0.13 ± 0.48</td><td>0.068 ± 0.071</td><td>0.993 ± 0.029</td></tr></table>

注：Panel A 报告了在八个步行速度标签上进行宏平均的逐点特定关节指标；这些数值采用速度级聚合，与模型级 Raspberry Pi 5 汇总的聚合方式不同。Panel B 报告的值为检测到的步态周期内的平均值 $\pm \nobreakspace \mathrm { ~ \ } S \mathrm { D } \nobreakspace$。峰值角度误差是预测峰值角度与基于 IMU 的参考峰值角度之间的绝对差值。时序误差是每个步态周期内的绝对峰值相位差。XCorr 滞后表示基于互相关的绝对时间滞后（以样本数计）；在 100-Hz 采样率下，一个样本对应 10 ms。频谱误差表示归一化频域差异。

![](images/307ae0733df63a5b3669148b3927aeda7989a8bfa74d845427f7cf5043477e06.jpg)  
Fig. 6. 代表性 TCN 推理：双侧髋关节和膝关节屈曲/伸展预测角度与基于 IMU 的参考角度对比（四路输出）。

我们通过计算去除按速度计算的平均 RMSE 后的残差组内相关系数（ICC）来进一步表征受试者间的变异性（Fig. 8(b)）。髋关节 $( \mathrm { I C C } \approx 0 . 3 9 )$ 和膝关节 $( \mathrm { I C C } \approx 0 . 4 0 )$ 的残差 ICC 均处于中等水平，表明除速度之外还存在不可忽视的受试者依赖成分。从实际部署的角度来看，我们通过报告均值、95 和最大 RMSE 来总结所有“受试者 × 速度”条件下的风险边界（Fig. 8(c)）。上尾误差达到髋关节：$p 9 5 = 2 . 8 5 ^ { \circ }$ 且 $\mathrm { { m a x } = 4 . 0 7 ^ { \circ } ; }$ ；膝关节：$p 9 5 \ = \ 3 . 3 7 ^ { \circ }$ 且 max $= \ 4 . 6 6 ^ { \circ }$ ，凸显了极端条件下的尾部风险。

![](images/a87e1de8d1f214fa3ccf982c780d5740ca23de466d1d2d830b46e016861240ce.jpg)  
(a) 髋关节 RMSE（热图）。

![](images/221b73622a9defc576fd2da42aeb7b9bae3bbc2f72a5940ffb7d225c26ac586a.jpg)  
(b) 膝关节 RMSE（热图）。  
Fig. 7. 完整“受试者 × 速度”网格（8 名受试者 × 8 种速度）上的泛化误差全景图。髋关节 RMSE 在不同条件下更为稳定，而膝关节 RMSE 表现出更大的变异性，并在极端速度下针对部分受试者有所增加。

为了检验所选 TCN 模型的关节特异性表现，我们进一步汇总了八种步行速度标签下的各关节误差。在四个目标输出中，速度层面的宏平均（macro-average）RMSE 范围为 $1 . 4 9 4 ^ { \circ }$ 至 $\boldsymbol { 1 . 9 7 5 ^ { \circ } }$，其中膝关节屈曲输出的误差略大于髋关节屈曲输出（Table 6，Panel A）。这种关节特异性模式与受试者–速度泛化分析一致，其中膝关节估计对上尾部误差和速度相关误差的贡献更大。

Table 7  
所选 TCN 模型的留一受试者交叉验证（LOSO）。

<table><tr><td>验证设置</td><td>宏 RMSE (°)</td><td>宏 MAE (°)</td><td>宏  $R^{2}$ </td><td>备注</td></tr><tr><td>LOSO，全部 8 折</td><td> $5.521 \pm 1.701$ </td><td> $3.871 \pm 1.198$ </td><td> $0.9474 \pm 0.1145$ </td><td>留出受试者间的均值 ± 标准差</td></tr></table>

![](images/bb366252d229f0d969e7a179c19a1d0093c05bea2e0e5b5fbb0cada065c9fdb3.jpg)  
(a) 速度效应（CI）。

![](images/4738671f51a2b01e3e626c76eb32a58705a40deada403f52432173fd0363e63a.jpg)  
(b) 残差 ICC。

![](images/577218f606fa62b97d54fa32ce7c1c5e5f2e9277cbbb2eaffcdd4ca416a5c862.jpg)  
(c) 风险边界。  
Fig. 8. TCN 在受试者和步行速度维度的泛化统计量：(a) 各速度下的均值 RMSE 及 95% CI，(b) 去除按速度计算的均值后的残差 ICC，以及 (c) 所有受试者 × 速度单元格中的均值/95/最大 RMSE。

## 4.6. 跨受试者验证

为了进一步评估跨受试者的泛化能力（subject-independent generalization），我们使用选定的 TCN 模型进行了留一受试者交叉验证（leave-one-subject-out, LOSO）。在所有八个留出折（held-out folds）中，LOSO 的宏平均 RMSE 为 $5 . 5 2 1 \pm$ $1 . 7 0 1 ^ { \circ }$，宏平均 MAE 为 $3 . 8 7 1 \pm 1 . 1 9 8 ^ { \circ }$，宏平均 $R ^ { 2 }$ 为 $0 . 9 4 7 4 \pm 0 . 1 1 4 5$（Table 7）。与受试者内按时间顺序测试相比，LOSO 验证的误差更高，这表明跨受试者估计仍然更具挑战性。尽管如此，相对较高的宏平均 $R ^ { 2 }$ 表明模型在未见过的参与者中仍保留了骨盆与关节角度之间的大部分映射关系。这些结果支持了基于单骨盆 IMU 进行跨受试者估计的可行性，同时也为未来的个性化、迁移学习和领域自适应（domain adaptation）策略提供了研究动机。

## 5. 讨论

本研究提出了一种仅使用单骨盆 IMU 数据流、可边缘部署的连续双侧髋/膝关节角度估计框架。主要发现是，在受控跑步机行走条件下，骨盆运动学包含了用于估计下肢关节角度轨迹的有价值时序信息，这与先前用于主动式髋关节离断（HD）假肢控制的基于骨盆运动学估计的研究一致 [26]。尽管 Transformer 实现了最低的预测误差，但 TCN 在估计精度和部署效率之间取得了更具优势的平衡，在 Raspberry Pi 5 上具有显著更低的推理延迟和更稳定的尾部延迟（tail latency）。因此，所选的 TCN 更适合被理解为一个可边缘部署的感知模块，而非一个完整的假肢控制器。

对健康参与者数据的使用应予以谨慎解读。健康步态无法复现 HD 患者的代偿性骨盆运动、步态不对称性、接受腔-假肢交互作用以及改变的动力学特征，这些在先前的综述以及截肢与假肢部件的多模态步态表征研究中已被重点强调 [8,9]。然而，标准健康步态仍然是一个重要的参考基准，因为智能动力 HD 假肢的目标之一正是恢复假肢辅助行走，使其趋向于更符合生理学的下肢运动学。因此，本研究结果表明骨盆运动包含了用于估计标准双侧髋/膝轨迹的有价值时序信息，但并不能直接代表其在 HD 使用者中的实际表现。

从边缘部署的角度来看，尾部延迟与平均延迟同等重要。极少数的慢帧可能会降低跟踪性能、安全监控能力和闭环响应速度。在桌面端 ONNX 和 Raspberry Pi 5 的基准测试中，与循环模型和基于注意力的模型相比，TCN 在保持具有竞争力的准确率的同时具有更小的延迟开销。这使得 TCN 非常适合集成到诸如参考轨迹生成、相位感知安全检查、阻抗调制以及自适应辅助等下游模块中，这些都是动力下肢假肢控制框架中的常见组件 [12, 13,20]，前提是在未来的工作中对完整的假肢控制闭环进行验证。

泛化性分析揭示了在不同行走速度和受试者之间存在的结构化变异性。髋关节角度估计相对稳定，而膝关节角度估计对速度依赖性误差和上尾部误差的贡献更大。LOSO 验证进一步表明，跨受试者独立估计是可行的，但其准确性低于受试者内按时间顺序进行的测试。尽管该模型在留出受试者上保持了相对较高的宏 $R ^ { 2 }$，但 RMSE 和 MAE 的增加表明仍存在受试者间的残留变异性。这一发现支持了在假肢级部署之前，采用针对特定用户的校准、迁移学习、不确定性感知滤波以及传感器融合策略的必要性。在未来的 HD 应用中，骨盆 IMU 信息可能应与假肢端编码器、力传感器、接受腔或残肢 IMU、足底压力信息或 EMG 进行融合，以提高非对称和非稳态步态条件下的鲁棒性。这与假肢步态分析和控制中所采用的多模态传感策略相一致 [9,20]。

有几项局限性需要予以考虑。首先，本研究是在健康受试者进行两分钟受控跑步机记录期间开展的，目前的结果不能直接推广至 HD（髋离断）假肢使用者、地面行走、地形转换、转弯或长期使用场景。其次，参考关节角度来源于商用多 IMU 系统，而非同步光学运动捕捉。尽管 IMU 传感更接近动力假肢系统中采用的可部署传感模态，但残余漂移、传感器与肢体段标定误差以及软组织伪影仍可能影响导出的参考角度。第三，边缘端基准测试侧重于 ONNX Runtime 下的模型推理延迟，未测量完整的闭环延迟、执行器响应、热稳定性或系统级功耗。最后，尽管 ridge\_linear 作为同输入经典基线被纳入对比，但完整的逆向运动学并未作为直接比较基准，因为传统的逆向运动学流程通常需要光学标记点轨迹、多环节方向，或至少需要大腿、小腿及足部的运动学观测数据。这些信息在单骨盆 IMU 输入约束下无法获取，且会导致传感器配置不对等的比较。总体而言，当前的结果应被视为对基于骨盆 IMU 的、可边缘部署感知模块的工程可行性评估。未来的工作应在 HD 使用者、多地形行走、长期记录以及闭环动力假肢实验中验证该框架，同时联合研究传感器融合、迁移学习和针对特定用户的标定。未来的工作还应利用同步光学运动捕捉或其他多模态生物力学参考测量，进一步量化参考标签的不确定性。将输出空间从矢状面屈曲/伸展扩展到包括髋关节和膝关节的外展/内收以及轴向旋转，是未来面向控制的研究的另一个重要方向。

## 6. 结论

我们提出了一个可边缘部署的框架，用于根据单个骨盆 IMU 数据流实时估计连续的双侧髋关节和膝关节屈曲/伸展角度。跨八个模型家族的系统基准测试表明，Transformer 实现了最高的准确率，而 TCN 则在准确率与延迟之间取得了更有利的平衡，并具有适用于边缘部署的稳定尾部延迟。TCN 消融实验进一步阐明了模型容量、时间感受野和正则化如何影响预测性能及推理开销。泛化性分析表明，髋关节角度估计相对稳定，而膝关节角度估计受行走速度和受试者间变异性的影响更大。LOSO 验证进一步支持了独立于受试者进行估计的可行性，而剩余的性能差距则为未来的个性化和领域自适应策略提供了研究动力。总体而言，所提出的方法为可边缘部署的步态感知提供了一个实用的初步关节角度估计模块，并为未来动力 HD（髋关节离断）假肢的研究提供了可行的数据接口。在进行假肢级部署之前，该框架应在 HD 用户、多地形行走、长期记录以及闭环动力假肢实验中进行进一步验证。

## CRediT 编写者贡献声明

Shu Xiaolong：撰写 – 原始初稿，软件，方法学，概念化。Luo Shengli：撰写 – 审阅与编辑，验证，数据管理。Wang Xu：验证，数据管理。Qin Yiyao：撰写 – 审阅与编辑，可视化，形式分析。Liu Qiwen：调查，数据管理。Yu Hongliu：撰写 – 审阅与编辑，监督，项目管理，方法学，概念化。

在稿件准备过程中使用生成式人工智能和人工智能辅助技术的声明

在准备本研究期间，作者使用了 ChatGPT (OpenAI) 以提高稿件的语言质量、可读性和整体组织结构。使用该工具/服务后，作者根据需要审阅并编辑了内容，并对已发表文章的内容承担全部责任。

## 利益冲突声明

作者声明，他们不存在任何可能影响本文所报道工作的已知竞争性经济利益或人际关系。

## 致谢

本研究得到了国家自然科学基金（National Natural Science Foundation of China，项目编号：62473263）的资助。

## 数据可用性

所使用的数据为保密数据。

## 参考文献

[1] A. Fernández, J. Formigo, Are Canadian prostheses used? A long-term experience, Prosthet. Orthot. Int. 29 (2005) 177–181, http://dx.doi.org/10.1080 03093640500217208.

[2] T. Chin, H. Oyabu, Y. Maeda, I. Takase, K. Machida, 老年髋关节离断截肢者使用假肢行走和轮椅移动期间的能量消耗, Am. J. Phys. Med. Rehabil. 88 (2009) 399–403, http://dx.doi.org 10.1097/PHM.0b013e3181a0dbe2

[3] X. Li, Y. Xie, H. Chen, B. Huang, H. Yu, 一种用于髋关节离断假肢的步态模拟与评估系统, IEEE Trans. Autom. Sci. Eng. 18 (2021) 448–457, http://dx.doi.org/10.1109/TASE.2020.3035438.

[4] E. Ludwigs, M. Bellmann, T. Schmalz, S. Blumentritt, 平地行走期间两种体外髋关节假肢系统的生物力学差异, Prosthet. Orthot. Int. 34 (2010) 449–460, http://dx.doi.org/10.3109/03093646.2010. 499551.

[5] L. Devillez, B. Herman, R. Ronsse, 具有类人运动范围与扭矩的紧凑型主动髋关节假肢设计, in: Proc. 2024 10th IEEE RAS/EMBS Int. Conf. Biomed. Robot. Biomechatron., BioRob, 2024, pp. 160–166, http: //dx.doi.org/10.1109/BioRob60516.2024.10719919.

[6] M. Song, Z. Li, J. Jiang, W. Chen, S. Guo, H. Zheng, et al., 具有多电机功能的髋关节假肢机构的设计、仿真与运动学验证, J. Bionic Eng. 21 (2024) 1321–1333, http://dx.doi.org/10.1007/s42235-024- 00490-x.

[7] S. Mroz, 用于髋关节离断假肢的侧装式动力髋关节设计与样机验证 (M.Sc. thesis), University of Ottawa (渥太华大学), 2023, http://hdl.handle.net/10393/45011 (accessed 25 September 2025).

[8] H. Gholizadeh, N. Baddour, M. Botros, K. Brannen, F. Golshan, E.D. Lemaire, 髋关节离断与半骨盆切除假肢：文献综述, Prosthet. Orthot. Int. 45 (2021) 434–439, http://dx.doi.org/10.1097/PXR. 0000000000000029.

[9] C. De Marchis, S. Ranaldi, T. Varrecchia, M. Serrao, S.F. Castiglia, A. Tatarelli, et al., 通过多模态测量表征不同截肢类型及假肢部件患者的步态：方法学视角, Front. Rehabil. Sci. 3 (2022) 804746, http://dx.doi.org/10.3389/ fresc.2022.804746

[10] D. Kim, R. Triolo, H. Charkhkar, 足底本体感觉恢复增强步态、速度感知与运动适应, Sci. Robot. 8 (2023) eadf8997, h d d b df

[11] M. Tran, L. Gabert, S. Hood, T. Lenzi, 一种复现膝、踝和足趾关节生物力学的轻量化机器人假肢腿, Sci. Robot. 7 (2022) eabo3996, http://dx.doi.org/10.1126/scirobotics.abo3996.

[12] T.K. Best, C.G. Welker, E.J. Rouse, R.D. Gregg, 用于自适应速度和斜坡行走的动力膝-踝假肢数据驱动可变阻抗控制, IEEE Trans. Robot. 39 (2023) 2151–2169, http://dx.doi.org/10.1109 TRO.2022.3226887.

[13] S. Cheng, C.A. Laubscher, T.K. Best, R.D. Gregg, 动力膝-踝假肢的双侧活动识别与连续自适应, IEEE Trans. Robot. 41 (2025) 2251–2267, http://dx.doi.org/10.1109/TRO.2025.3539206.

[14] S. Luo, X. Shu, J. Du, H. Li, H. Yu, Enhancing gait symmetry via intact limb kinematic mapping control of a hip disarticulation prosthesis, Biomimetics 10 (2025).714. http://dx doi org/10.3390/biomimetics10100714

[15] S. Luo, X. Shu, H. Zhu, H. Yu, Early prediction of lower limb prostheses locomotion mode transition based on terrain recognition, IEEE Sens. J. 23 (2023) 27941–27948, http://dx.doi.org/10.1109/JSEN.2023.3320274.

[16] K. Brannen, N. Baddour, L. Cho, D. Langlois, P. Dumond, E.D. Lemaire, Development and evaluation of an anteriorly mounted microprocessor-controlled powered hip joint prosthesis, Can. Prosthet. Orthot. J. 7 (2024) 44494, http: //dx.doi.org/10.33137/cpoj.v7i2.44494.

[17] M. Botros, H. Gholizadeh, F. Golshan, D. Langlois, N. Baddour, E.D. Lemaire, Development of a powered four-bar prosthetic hip joint prototype, Prosthesis 7 (2025) 105. http://dx.doi.org/10.3390/prosthesis7050105.

[18] Y. Uevama, T. Kubo, M. Shibata, Robotic hip-disarticulation prosthesis: evaluation of prosthetic gaits in a non-amputee individual, Adv. Robot. 34 (2020) 37–44. http://dx.doi.org/10.1080/01691864.2019.1705908.

[19] N. Hata, T. Kubo, T. Inoue, Development and experimental evaluation of a swing phase controller for a powered hip disarticulation prosthesis (日文), Trans. Jpn. Soc. Mech. Eng. Ser. C 75 (2009) 404–412, http://dx.doi.org/10.1299 kikaic 75 404

[20] F. Golshan, E. Lemaire, H. Gholizadeh, D. Nielen, N. Baddour, Design and preliminary evaluation of a gait control strategy for hip–knee–ankle–foot prostheses with motorized hip joint, IEEE Trans. Neural Syst. Rehabil. Eng. 33 (2025) 3432–3442. http://dx doi org/10.1109/TNSRE 2025.3602715

[21] Y. Bader, D. Langlois, N. Baddour, E.D. Lemaire, Development of an integrated powered hip and microprocessor-controlled knee for a hip– knee–ankle–foot prosthesis, Bioengineering 10 (2023) 614, http://dx.doi.org/10. 3390/bioengineering10050614.

[22] Y. Bader, Design and Prototyping of an Integrated Powered Hip and Microprocessor-Controlled Knee Unit for Hip-Knee-Ankle-Foot Prostheses (硕士学位论文), University of Ottawa, 2023, http://hdl.handle.net/10393/45116 (2025年9月25日访问).

[23] Y. Chen, X. Li, H. Su, D. Zhang, H. Yu, Design of a bio-inspired gait phase decoder based on temporal convolution network architecture with contralateral surface electromyography toward hip prosthesis control, Front. Neurorobot. 16 (2022) 791169, http://dx.doi.org/10.3389/fnbot.2022.791169.

[24] L. Li, H. Luo, S. Luo, X. Wang, X. Shu, Q. Meng, H. Yu, Design and swing control of a power-assisted hip disarticulation prosthesis, IEEE/ASME Trans. Mechatronics (2025) 1–11, http://dx.doi.org/10.1109/TMECH.2025.3586300

[25] M. Fan, Y. Chen, B. He, Q. Meng, H. Yu, Study on adaptive adjustment of variable joint stiffness for a semi-active hip prosthesis, in: H. Liu, Z. Yin, L. Liu, L. Jiang, G. Gu, X. Wu, et al. (Eds.), Intelligent Robotics and Applications, Springer, Cham, 2022, pp. 13–23, http://dx.doi.org/10.1007/978-3-031-13822-5\_2.

[26] F. Golshan, E.D. Lemaire, H. Gholizadeh, D. Nielen, N. Baddour, A pelvic kinematic approach for calculating hip angles for active hip disarticulation prosthesis control, J. Neuroeng. Rehabil. 20 (2023) 152, http://dx.doi.org/10. 1186/s12984-023-01273-x.