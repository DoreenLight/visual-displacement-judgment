# RGB 直接角度估计（2022 EMBC + 同团队相关 OA）

证明「RGB → joint angle」这条路在 2022 年就已经不是创新点了。我们做的是「升级到结构化翻译」。

> **✅ 状态更新（2026-09-30）**
> 两篇 EMBC 2022 原文**已穷尽渠道重新检索**，结果：
> - **Kumar et al. 2022 EMBC —— ✅ 原文 PDF 已获取**（`Kumar2022：深度学习从视觉系统预测下肢运动学/Kumar2022：深度学习从视觉系统预测下肢运动学.pdf`，1,237,638 B，5 页）+ **完整中文翻译**（`Kumar2022：深度学习从视觉系统预测下肢运动学/Kumar2022：深度学习从视觉系统预测下肢运动学_全文翻译.md`）
> - **Jamsrandorj et al. 2022 EMBC —— ✅ 原文 PDF 已获取**（`Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计/Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计.pdf`，1,630,820 B，5 页，IEEE 官方版）+ **全译级中文精读**。原替代包已归档为取件记录。
> - **另**：Chao 2026 Measurement 的原文亦已获取（见 `../RGB_直接竞品_2026_Chao/`）。
> - **DOI 误填已更正**：Kumar 的 DOI 此前登记为 `…9871780`，该 DOI **实际指向一篇心脏 MRI 论文**；正确值为 `…9871577`（四源一致）。
>
> 🔴 **新增关键情报**：Kumar 2022 的原文 §IV「未来工作」明确写道 *"will be extended to use transformers and multi-head attention networks"*。
> **即 KIST 团队在 2022 年就公开预告了 Transformer 路线 —— 四年后的 Chao 2026 Measurement 正是这条路线图的兑现。**
> 详见 `Kumar2022：深度学习从视觉系统预测下肢运动学/Kumar2022：深度学习从视觉系统预测下肢运动学_全文翻译.md` §IV 与附录。

---

## 1. Jamsrandorj et al. — EMBC 2022 ⭐⭐⭐⭐ ✅ 原文已获取

| 字段 | 值 |
|---|---|
| 标题 | Deep Learning Networks for View-independent Knee and Elbow Joint Angle Estimation |
| 会议 | 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC) |
| 年份 / 页码 | 2022 · **pp. 2703–2707** |
| DOI | `10.1109/EMBC48229.2022.9871106` |
| IEEE Xplore | <https://ieeexplore.ieee.org/document/9871106> |
| PubMed | <https://pubmed.ncbi.nlm.nih.gov/36085943/> |
| KIST 机构仓库 | <https://pubs.kist.re.kr/handle/201004/77156>（元数据 + 完整摘要，**无全文**） |
| **PDF 状态** | ✅ **已获取原文**（`Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计/Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计.pdf`，1,630,820 B / 5 页，IEEE 官方版） |
| **归档留证** | 📦 `Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计/Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计_reconstruction.md`（原深度替代包，现转为取件记录 + 竞品对照） |
| 中文精读 | ✅ `Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计/Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计_全文翻译.md` |
| 优先级 | ⭐⭐⭐⭐ 必读 |
| 与你的方向 | **历史基线**：9 帧 RGB → CNN/BiLSTM → 膝/肘关节角度 |

**关键证据**（来自 KIST 机构仓库官方摘要，已验证）：
- 输入 **9 个连续帧**，输出**每帧**膝/肘屈伸角度
- **4 种网络对比**，**BiLSTM 最优**
- 膝关节相关系数 **0.955**，肘关节 **0.917**，**与相机视角无关（view-independent）**
- 真值：**光学动作捕捉系统**；被试：**15 名健康受试者**；活动：**4 种日常活动**

**对我们的意义**：
- 进一步证明「RGB → angle」不是 2026 才出现的新点子
- **9 帧短窗 + BiLSTM** 是我方长序列 + Transformer 的直接对照基线
- **"view-independent" 在 2022 年就已是卖点** → 跨视角鲁棒性是我的**必备门槛**，不是创新点

---

## 2. Jamsrandorj et al. — arXiv 2203.00503 (2022, **已下载为替代**) ⭐⭐⭐

| 字段 | 值 |
|---|---|
| 文件 | `Jamsrandorj2022-Wearable：腰戴混合 CNN-RNN 的步态事件预测/Jamsrandorj2022-Wearable：腰戴混合 CNN-RNN 的步态事件预测.pdf` (1.6 MB) |
| 标题 | Gait Events Prediction using Hybrid CNN-RNN-based Deep Learning models through a Single Waist-worn Wearable Sensor |
| 作者 | Muhammad Zeeshan Arshad, **Ankhzaya Jamsrandorj**, Jinwook Kim, Kyung-Ryoul Mun |
| 机构 | Korea Institute of Science and Technology (KIST) |
| 年份 | 2022 |
| arXiv | `2203.00503` |
| **PDF 状态** | ✅ 已下载 + 已翻译 (`Jamsrandorj2022-Wearable：腰戴混合 CNN-RNN 的步态事件预测/Jamsrandorj2022-Wearable：腰戴混合 CNN-RNN 的步态事件预测_全文翻译.md`) |
| 与 EMBC 原文关系 | **同团队、同主题的不同任务**：原文做 joint angle 估计，这篇做 gait events prediction。 |
| 与你的方向 | 同一团队的姊妹工作，方法（CNN-RNN hybrid）证实他们的研究路线 |

**与 EMBC 2022 原文的差异**：
| 维度 | EMBC 2022 原文 | arXiv 2203.00503 (替代) |
|---|---|---|
| 输入 | 9 帧 RGB | 单个腰部可穿戴传感器（IMU）|
| 输出 | 膝/肘关节角度 | gait events (heel-strike / toe-off) |
| 方法 | CNN / BiLSTM | CNN-RNN hybrid |
| 任务 | view-independent joint angle | gait events prediction |
| 团队 | 同一 KIST 团队 | 同 |

→ 仍作为补充材料，证明 KIST 团队 2022 年就在做 vision/sensor → kinematic 这条路。

---

## 3. Kumar et al. — EMBC 2022 ⭐⭐⭐ ✅ **原文已获取 + 已翻译**

| 字段 | 值 |
|---|---|
| 标题 | Prediction of lower limb kinematics from vision-based system using deep learning approaches |
| 会议 | EMBC 2022 · **pp. 177–181** |
| 年份 | 2022（出版 2022-07-11） |
| **DOI** | **`10.1109/EMBC48229.2022.9871577`** ✅ **已校验** |
| IEEE Xplore | <https://ieeexplore.ieee.org/document/9871577> |
| PubMed | <https://pubmed.ncbi.nlm.nih.gov/36086538/> |
| KIST 机构仓库 | <https://pubs.kist.re.kr/handle/201004/77158> |
| **PDF 状态** | ✅ **已获取** —— `Kumar2022：深度学习从视觉系统预测下肢运动学/Kumar2022：深度学习从视觉系统预测下肢运动学.pdf`（1,237,638 B，5 页，`file` 报合法 PDF） |
| **中文翻译** | ✅ `Kumar2022：深度学习从视觉系统预测下肢运动学/Kumar2022：深度学习从视觉系统预测下肢运动学_全文翻译.md`（完整翻译，非摘要） |
| 与你的方向 | **历史基线**：RGB video → CNN + BiLSTM/GRU → 下肢角速度 |

### ⚠️ DOI 误填更正记录（重要）

| 项 | 值 |
|---|---|
| 此前登记（**错误**） | `10.1109/EMBC48229.2022.9871780` |
| 该错误 DOI 实际指向 | **Hasan & Linte, "Joint Segmentation and Uncertainty Estimation of Ventricular Structures from Cardiac MRI using a Bayesian Conditional…"** —— 一篇**心脏 MRI 论文**，与下肢运动学毫无关系 |
| **正确 DOI** | **`10.1109/EMBC48229.2022.9871577`** |
| 校验依据（四源一致） | ① **Crossref** `api.crossref.org/works/10.1109/EMBC48229.2022.9871577` → 标题/页码 pp.177-181/作者全对；② **OpenAlex** 同 DOI → 标题/作者一致；③ **论文 PDF 页眉**逐字：`DOI: 10.1109/EMBC48229.2022.9871577`；④ **IEEE 页脚**授权声明行 |

### 关键证据（全部来自已获取的原文，已验证）

| 项 | 值 |
|---|---|
| 受试者 | **33 名健康成年人**（25 男 / 8 女） |
| 光学动捕（真值） | 6 相机，1280×1024，**120 Hz**，Plug-in-gait，**15 个反光标记点**；捕捉区域 2.000 m × 3.685 m |
| RGB 相机 | **6 台小米米家相机 4K**，1920×1080，**60 Hz**，布置在前/后/侧面 |
| 动作 | **深蹲（squat）** + **跑步机行走**，各 30 秒 |
| 人体检测 | **YOLOv4** → 裁剪并缩放至 **448×448** |
| 输入 | **27 帧** RGB |
| Backbone | **ResNet18** |
| 时序模块 | **Bi-LSTM / GRU**（DeepConvLSTM / DeepConvGRU） |
| 输出 | **6 个关节角速度**（左右髋屈伸/膝屈伸/踝背屈跖屈） |
| 训练 | PyTorch，Adam，lr **1e-4**，batch **8**，**20 epochs**，MAE loss，GTX 2080Ti |
| 划分 | **7 训练 / 3 验证 / 3 测试**（**按受试者**，即跨被试） |
| 最佳结果 | **GRU：深蹲 r = 0.93，跑步机 r = 0.92**；膝关节最优 **0.96 / 0.96** |
| 跨视角 | 侧视图 **0.92** / 正视图 **0.89** |
| 残余误差 | **15–20 rad/s** |

### 🔴 原文最重磅的一条（本库此前完全未掌握）

> 原文 §IV 未来工作原文照录：
> *"The proposed DeepConvGRU model will be extended to use **transformer and multi-head attention networks** to achieve higher accuracy in angular velocity estimation."*
> *"our work will be extended to estimate other gait parameters (**cadence, speed, stride length, and step width**)."*

**含义**：KIST 团队 **2022 年**就公开了「Transformer + 多头注意力」的路线图。
四年后 **Chao 2026 Measurement**（同团队、CNN + Transformer、单目 RGB → 髋/膝/踝角度）**正是这条路线图的兑现**。
→ **Chao 2026 不是偶然竞品，是该团队公开预告的既定路线。** 我方 Related Work 必须正面写出这条 4 年时间线。

**对我们的意义**：
- 「vision → kinematic temporal signal」这条路 2022 已成熟
- 我们必须把创新点聚焦在 **token + grammar** 上，而不是 signal type
- **跨被试划分（7/3/3）在 2022 年已实现** → 我方跨被试协议必须采用，否则严格度不如 4 年前的同类工作

---

## 4. 同团队后续路线（情报汇总）

| 年 | 论文 | 方法 | 状态 |
|---|---|---|---|
| 2021 | Kumar et al., EMBC 2021, pp. 2186–2190 | 深度 CNN，角速度 | 未收 |
| 2022 | Jamsrandorj et al., EMBC 2022, pp. 2703–2707 | **BiLSTM**，膝/肘角度 | ✅ **原文已获取** |
| 2022 | Kumar et al., EMBC 2022, pp. 177–181 | **ResNet18 + BiLSTM/GRU**，下肢角速度 | ✅ **原文已获取** |
| **2023** | Jamsrandorj et al., EMBC 2023（paperId 472） | **CNN-Transformer 混合**，单相机步态时空参数 | ✅ **已获取** → 见 `../RGB_直接竞品_2026_Chao/` |
| 2026 | **Chao et al., Measurement 270:120772** | **CNN + Transformer，端到端** | ✅ **原文已获取** → 见 `../RGB_直接竞品_2026_Chao/` |

→ 这条 5 年演化线是本库最重要的竞品情报，**必须写进 Related Work**。
