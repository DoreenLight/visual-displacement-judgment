# D5 轻量化非Transformer路线

> 7 篇 ｜ ✅ 已入库 7 ｜ 需人工取件 0
> 本文件由 `build_jev.py` 生成。数字口径见 `../index.json`。

| # | 年 | 论文题名 | 载体 | 链接 | 证据等级 | 编者结论 | 本地 |
|---|---|---|---|---|---|---|---|
| 1 | 2021 | sEMG-based Regression of Hand Kinematics with Temporal Convolutional Networks on a Low-Power Edge Microcontroller | IEEE（EMBC 附近） | [DOI](https://ieeexplore.ieee.org/document/9524188) | 第三方实测 | 可行主路线 | ✅ |
| 2 | 2023 | SE-TCN network for continuous estimation of upper limb joint angles | Mathematical Biosciences and Engineering 20(2): 3237-3260 | [DOI](https://pubmed.ncbi.nlm.nih.gov/36899579/) | 第三方实测 | 条件可行 | ✅ |
| 3 | 2024 | Personalization of Wearable Sensor-Based Joint Kinematic Estimation Using Computer Vision for Hip Exoskeleton Applications | arXiv 预印本（后有会议版） | [2411.15366](https://arxiv.org/abs/2411.15366) | 第三方实测 | 可行主路线 | ✅ |
| 4 | 2025 | Multijoint Continuous Motion Estimation for Human Lower Limb Based on Surface Electromyography | Sensors (MDPI) 25(3): 719（Gold OA + PMC 开放全文） | [DOI](https://pmc.ncbi.nlm.nih.gov/articles/PMC11820049/) | 第三方实测 | 条件可行 | ✅ |
| 5 | 2025 | Continuous Motion Prediction for Upper Limb Rehabilitation Robotics Based on Surface Electromyography: A Spatio-Temporal Attention and Lightweight TCN Approach | IEEE ICMA 2025 | [DOI](https://doi.org/10.1109/icma65362.2025.11120720) | 第三方实测 | 条件可行 | ✅ |
| 6 | 2026 | Edge-deployable pelvic IMU temporal convolution for real time hip and knee joint angle estimation | Biomedical Signal Processing and Control 126: 110933 | [DOI](https://doi.org/10.1016/j.bspc.2026.110933) | 第三方实测 | 条件可行 | ✅ |
| 7 | 2026 | Kinematic Tokenization: Optimization-Based Continuous-Time Tokens for Learnable Decision Policies in Noisy Time Series | arXiv 预印本 | [2601.09949](https://arxiv.org/abs/2601.09949) | 官方一手 | 命名占用标记 | ✅ |

---

### sEMG-based Regression of Hand Kinematics with Temporal Convolutional Networks on a Low-Power Edge Microcontroller（2021，IEEE（EMBC 附近））

- **作者**：Grenzholt 等（TEMPONet）
- **来源**：<https://ieeexplore.ieee.org/document/9524188>
- **关键数字（A）**：NinaPro DB8，12 名被试 9 个对侧动作，5 个连续自由度：MAE 6.89°（比 SoA 好 0.15°）；int8 量化后仅 **70.9 kB**；GAP8 上 4.76 ms 延迟、0.243 mJ/次推理
- **证据等级**：第三方实测
- **编者结论**：可行主路线
- **本地工件**：PDF 全文已在库：`01_参考论文/整合系统/决策模型线_JEV/D5_轻量化非Transformer路线/sEMG-based Regression of Hand Kinematics with Temporal Convo.pdf`
- **备注**：**手部 sEMG → 腕/手指运动学**，与本项目腕部任务最接近的轻量化先例。
70.9 kB 与任何十亿级参数模型不在一个量级 —— **这说明「非自回归」不是轻量化的答案，「小模型才是」。**
注意：它做的是回归（连续角度），不是介入决策，恰好印证 D2 的分工判断。

### SE-TCN network for continuous estimation of upper limb joint angles（2023，Mathematical Biosciences and Engineering 20(2): 3237-3260）

- **作者**：Liu X、Wang J、Liang T、Lou C
- **来源**：<https://pubmed.ncbi.nlm.nih.gov/36899579/>
- **关键数字（A）**：10 名被试 7 种上肢动作，测肘角 EA / 肩垂直角 SVA / 肩水平角 SHA：较 BP 网络 RMSE 分别降低 25.0% / 45.6% / 38.6%，较 LSTM 降低 36.8% / 49.5% / 43.6%
- **证据等级**：第三方实测
- **编者结论**：条件可行
- **本地工件**：PDF 全文已在库：`01_参考论文/整合系统/决策模型线_JEV/D5_轻量化非Transformer路线/SE-TCN network for continuous estimation of upper limb joint.pdf`
- **备注**：**肩肘腕三关节角度 + sEMG + TCN**，与本项目任务结构几乎同构。开源获取。
⚠️ 但需与 D2 的 TCN 反例并读：Wei 2026 在同为 sEMG 的任务上，TCN 却是八个基线里最差的。
**差异可能来自电极数与动作自由度**（SE-TCN 7 个动作 vs Wei 8×4 电极多通道），需在方法学上区分。

### Personalization of Wearable Sensor-Based Joint Kinematic Estimation Using Computer Vision for Hip Exoskeleton Applications（2024，arXiv 预印本（后有会议版））

- **作者**：Song, Changseob、Ivanyuk-Skulskyi, Bogdan、Krieger, Adrian、Luo, Kaitao
- **来源**：<https://arxiv.org/abs/2411.15366>
- **关键数字（A）**：仅需 1–2 个步态周期的数据做迁移学习，RMSE 较纯 TCN 降低 9.7% / 19.9%
- **证据等级**：第三方实测
- **编者结论**：可行主路线
- **本地工件**：PDF 全文已在库：`01_参考论文/整合系统/决策模型线_JEV/D5_轻量化非Transformer路线/Personalization of Wearable Sensor-Based Joint Kinematic Est.pdf`
- **备注**：**直击本项目已知难点**（跨被试泛化只有 50–70%）。用计算机视觉姿态估计产生的标签 + TCN 迁移，是绕开「每个新被试都要重新标定」的可行路径。
**且它同时命中了 AAN 决策层的一个前置问题**：决策阈值如果因被试而异，就必须先解决个体化，否则阈值不可迁移。

### Multijoint Continuous Motion Estimation for Human Lower Limb Based on Surface Electromyography（2025，Sensors (MDPI) 25(3): 719（Gold OA + PMC 开放全文））

- **作者**：Han, Yonglin、Tao, Qing、Zhang, Xiaodong
- **来源**：<https://pmc.ncbi.nlm.nih.gov/articles/PMC11820049/>
- **关键数字（A）**：CB-TCN（TCN + 卷积块注意力），8 名被试 4 种下肢动作：膝角 R² 最高 0.9718、RMSE 最低 1.2648°、NRMSE 最低 0.05234
- **证据等级**：第三方实测
- **编者结论**：条件可行
- **本地工件**：PDF 全文已在库：`01_参考论文/整合系统/决策模型线_JEV/D5_轻量化非Transformer路线/Multijoint Continuous Motion Estimation for Human Lower Limb.pdf`
- **备注**：下肢任务，方法可迁移。MDPI Gold OA + PMC 开放全文，取件容易。
**关键点：它加了卷积块注意力** —— 纯 TCN 不足，注意力有增益。这与 Wei 2026「TF 优于 BiLSTM」的结论方向一致。

### Continuous Motion Prediction for Upper Limb Rehabilitation Robotics Based on Surface Electromyography: A Spatio-Temporal Attention and Lightweight TCN Approach（2025，IEEE ICMA 2025）

- **作者**：Chen, Weiyu、Li, Pengcheng、Guo, Shuxiang、Li, Chunying
- **来源**：<https://doi.org/10.1109/icma65362.2025.11120720>
- **关键数字（B）**：STA-TCN，6 名健康成人；表I：RMSE 14.77° vs 基线TCN 21.27°，R² 0.91 vs 0.89；表II中Li[10]为15.25°/0.93，不是同一基线。摘要报告单次预测延迟1.5 ms。
- **证据等级**：第三方实测
- **编者结论**：条件可行
- **本地工件**：PDF 全文已在库：`01_参考论文/整合系统/决策模型线_JEV/D5_轻量化非Transformer路线/Continuous Motion Prediction for Upper Limb Rehabilitation R.pdf`
- **备注**：已用作者实验室公开的6页IEEE ICMA 2025正式PDF核对题名、作者、DOI、摘要及表I/II。原Zenodo 17850061错配，已撤下。比较数字须区分本实验TCN基线与表II文献比较；6名健康成人的离线结果不能直接当作卒中患者闭环疗效证据。1.5 ms为摘要报告的单次预测延迟，不等于整套系统闭环时延。

### Edge-deployable pelvic IMU temporal convolution for real time hip and knee joint angle estimation（2026，Biomedical Signal Processing and Control 126: 110933）

- **作者**：Shu, Xiaolong、Luo, Shengli、Wang, Xu、Qin, Yiyao
- **来源**：<https://doi.org/10.1016/j.bspc.2026.110933>
- **关键数字（A）**：统一协议下基准 8 个模型族：Transformer 误差最低 RMSE 1.698° MAE 1.290°；TCN 在树莓派 5 上最佳精度-延迟平衡（RMSE 1.921°，p95 0.229 ms）。留一被试交叉验证显示跨被试性能下降
- **证据等级**：第三方实测
- **编者结论**：条件可行
- **本地工件**：PDF 全文已在库：`01_参考论文/整合系统/决策模型线_JEV/D5_轻量化非Transformer路线/Edge-deployable pelvic IMU temporal convolution for real tim.pdf`
- **备注**：⚠️ **读这篇时必须配 D2 的 TCN 反例一起读。**
本篇的信号是**髋部 IMU 加速度**（低维强周期），此时 TCN 够用、Transformer 精度最高。
但同为 sEMG 信号的 Wei 2026 给出相反排序（TCN 垫底）。
**结论：路线选择按信号类型分组，不能跨信号搬。** 上海理工大学智能康复工程研究所。

### Kinematic Tokenization: Optimization-Based Continuous-Time Tokens for Learnable Decision Policies in Noisy Time Series（2026，arXiv 预印本）

- **作者**：（见语料三索引）
- **来源**：<https://arxiv.org/abs/2601.09949>
- **关键数字（B）**：无性能数字（该文目的不是刷指标，而是占用方法名）
- **证据等级**：官方一手
- **编者结论**：命名占用标记
- **本地工件**：PDF 全文已在库：`01_参考论文/整合系统/生物力学线/B4_离散运动学token_与表征量化/Kinematic Tokenization_ Optimization-Based Continuous-Time T.pdf`（跨库引用，不重复下载）
- **备注**：⚠️ **标题与本项目高度相似，实为金融时间序列论文**（资产价格、样条系数、交易决策）。
列入只为标记：**「Kinematic Tokenization」这个方法名已被占用**，本项目论文中不可直接用作方法名，否则检索会与此文混淆。领域不撞。
PDF 已在语料三 B4 入库。
