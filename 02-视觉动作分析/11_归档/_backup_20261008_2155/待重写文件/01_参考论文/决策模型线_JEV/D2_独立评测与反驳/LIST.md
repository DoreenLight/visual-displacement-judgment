# D2 独立评测与反驳

> 7 篇 ｜ ✅ 已入库 1 ｜ 需人工取件 6
> 本文件由 `build_jev.py` 生成。数字口径见 `../index.json`。

| # | 年 | 论文题名 | 载体 | 链接 | 证据等级 | 编者结论 | 本地 |
|---|---|---|---|---|---|---|---|
| 1 | 2026 | More Features Are Not More Evidence: Limits of Training-Free Human Activity Recognition with Jev | arXiv 预印本 | [2609.36154](https://arxiv.org/abs/2609.36154) | 第三方实测 | 不可行但需引证 | ✅ |
| 2 | 2026 | Archer Hume 对 JEV 的黑盒探测记录 | 第三方博客（探测记录） | [DOI](https://archerhume.com/posts/jevs-architecture-unmasked/) | 黑盒探测 | 不可行但需引证 | 需取件 |
| 3 | 2026 | JEV 社区实测汇总：选项顺序敏感、国际象棋纯随机、校准无公开数据 | 第三方技术博客 | [DOI](https://regolo.ai/jev-and-system-one-models-benchmarks-open-source-alternatives-and-when-to-use-them) | 社区报告 | 不可行但需引证 | 需取件 |
| 4 | 2026 | TypeSafe OpenAPI 3.1.0 规格（接口权威契约） | 厂商文档 | [DOI](https://api.typesafe.ai/openapi.json) | 官方一手 | 不可行但需引证 | 需取件 |
| 5 | 2026 | akhilaaa3/Jev-Omni（HuggingFace 社区模型核查） | HuggingFace 模型库（社区上传） | [DOI](https://huggingface.co/akhilaaa3/Jev-Omni) | 官方一手 | 不可行但需引证 | 需取件 |
| 6 | 2026 | JEV 输出类型 vs 3DCNN-TF 连续回归的不可替换性（对照分析） | 编者分析，对照 10.1109/JBHI.2025.3589889 与 arXiv 2609.36154 | [DOI](https://doi.org/10.1109/JBHI.2025.3589889) | 编者判断 | 不可行但需引证 | 需取件 |
| 7 | 2026 | Wei 2026 JBHI 八基线中的 TCN 反例（与 BSPC 2026 结论方向相反） | 编者分析，对照 10.1109/JBHI.2025.3589889 与 10.1016/j.bspc.2026.110933 | [DOI](https://doi.org/10.1109/JBHI.2025.3589889) | 编者判断 | 不可行但需引证 | 需取件 |

---

### More Features Are Not More Evidence: Limits of Training-Free Human Activity Recognition with Jev（2026，arXiv 预印本）

- **作者**：Konak, Orhan
- **来源**：<https://arxiv.org/abs/2609.36154>
- **关键数字（A）**：WISDM / UCI341 / PAMAP2 三数据集，1800 个类别均衡加速度计窗口，5400 次 JEV 决策：macro-F1 仅 0.038 / 0.118 / 0.089，监督模型 0.686–0.907。特征从 20 个扩到 32 个反而更差（预测塌向单一类）。论文明确结论：JEV 的概率「not reliably calibrated for recognition」
- **证据等级**：第三方实测
- **编者结论**：不可行但需引证
- **本地工件**：PDF 全文已在库：`01_参考论文/决策模型线_JEV/D2_独立评测与反驳/More Features Are Not More Evidence_ Limits of Training-Free.pdf`
- **备注**：★★ **本语料最重要的一篇**。Hasso Plattner Institute（波茨坦大学）独立完成，方法公开、数据集公开。
**与本项目直接相关**：ViKiT 往模型里加更多模态 token，走的正是这条「特征越多越好」的曲线，而这篇论文证明该假设在 JEV 上是反的。

### Archer Hume 对 JEV 的黑盒探测记录（2026，第三方博客（探测记录））

- **作者**：Hume, Archer
- **来源**：<https://archerhume.com/posts/jevs-architecture-unmasked/>
- **关键数字（C）**：探测得 MMLU-Pro 84.6%、ECE ≈0.031（1200 题样本）；推断为 causal-decoder Transformer、可能稀疏 MoE 约 10B 激活参数；自承无法从外部区分 causal decoder 与双向编码器
- **证据等级**：黑盒探测
- **编者结论**：不可行但需引证
- **本地工件**：**全文未入库，需补件或核查网页证据**
- **备注**：**属推断而非披露**。其价值恰恰在于：JEV 官方从未公布架构，外部只能测不能知。「JEV 是 Transformer」这一点目前只有黑盒探测这一个来源支撑。

### JEV 社区实测汇总：选项顺序敏感、国际象棋纯随机、校准无公开数据（2026，第三方技术博客）

- **作者**：多方社区（regolo.ai / vandatateam / robotworld.top 等）
- **来源**：<https://regolo.ai/jev-and-system-one-models-benchmarks-open-source-alternatives-and-when-to-use-them>
- **关键数字（C）**：国际象棋：喂原始棋盘位置 ≈ 随机；提取结构特征后 65%；给出战术事实后 78%。倒转选项顺序使相同输入的概率从 0.84–0.89 变为 0.93–0.96。至今无 ECE 数值或可靠性图
- **证据等级**：社区报告
- **编者结论**：不可行但需引证
- **本地工件**：**全文未入库，需补件或核查网页证据**
- **备注**：多个独立站点测出同一现象，可信度中等偏高。**「无法自行从原始物理量推理」这一点对介入决策同样致命** —— 那正是康复输入的本质。

### TypeSafe OpenAPI 3.1.0 规格（接口权威契约）（2026，厂商文档）

- **作者**：TypeSafe AI
- **来源**：<https://api.typesafe.ai/openapi.json>
- **关键数字（B）**：自承弱点：数学与计数、日期比较、数字邻近性、双重否定、长上下文「rot」（过滤须在客户端做）、对抗输入、**CJK 语言准确率明显偏低**
- **证据等级**：官方一手
- **编者结论**：不可行但需引证
- **本地工件**：**全文未入库，需补件或核查网页证据**
- **备注**：⚠️ **原条目名有误**：我此前把「厂商承认的能力短板」与「API 规格」混为一条，URL 又指向 404 页面。
现更正为**权威接口契约**存档：`D2_独立评测与反驳/..._网页存档.md`（14 KB 完整 OpenAPI 文档）。

**「能力短板清单」仍在厂商文档内，但需人工阅读存档核对** —— 存档文件头部已记录来源、HTTP 状态、字节数与 SHA256，可追溯。

### akhilaaa3/Jev-Omni（HuggingFace 社区模型核查）（2026，HuggingFace 模型库（社区上传））

- **作者**：akhilaaa3（个人账号，非 TypeSafe）
- **来源**：<https://huggingface.co/akhilaaa3/Jev-Omni>
- **关键数字（A）**：HF API 元数据：base_model=google/gemma-4-12B-it；tags 含 base_model:finetune:google/gemma-4-12B-it 与 merged；架构 gemma4_unified；参数量 11,959,730,224（≈12B）；权重 23.9 GB BF16；pipeline_tag=text-classification；region=us only；创建于 2026-09-20（JEV 发布后 5 天）
- **证据等级**：官方一手
- **编者结论**：不可行但需引证
- **本地工件**：**全文未入库，需补件或核查网页证据**
- **备注**：⚠️ **这是一个被误认为 JEV 官方多模态版的社区模型，实为 Gemma-4-12B 的 LoRA 合并微调。**
三重证伪：① 元数据底座是 google/gemma-4-12B-it；② chat template 是 Gemma 家族的 `<|turn>`/`<|channel>thought` 格式，非 JEV 的 Choice/Score/Null 接口；③ TypeSafe 从未开源 JEV 权重（闭源、仅 API），社区不可能有 JEV 微调。
**且 12B / 23.9 GB 与「轻量化」目标正好相反。**
这是本项目第三次「预期来源失真」：T2M 题名错 → TATS 论文不存在 → Jev-Omni 张冠李戴。

### JEV 输出类型 vs 3DCNN-TF 连续回归的不可替换性（对照分析）（2026，编者分析，对照 10.1109/JBHI.2025.3589889 与 arXiv 2609.36154）

- **作者**：（本项目编者对照分析）
- **来源**：<https://doi.org/10.1109/JBHI.2025.3589889>
- **关键数字（C）**：Wei 2026 任务为连续回归（IRASS 腕关节 nRMSE 6.2%、R² 95.5%；EPIC 膝关节 5.5%/96.2%）；JEV 输出为 Choice（≤255 离散选项）或 Score，量化误差地板将高于当前 6.2% 的总误差。且 Wei 的核心贡献是因果掩码 + 自回归逐步生成，与 JEV 砍掉自回归解码的设计直接冲突
- **证据等级**：编者判断
- **编者结论**：不可行但需引证
- **本地工件**：**全文未入库，需补件或核查网页证据**
- **备注**：**用户曾提出「JEV 支持 JSON，可替代该论文的 3DCNN-TF」——经核对不成立**，四条硬约束：
① 输出类型：连续回归 vs 离散选项，量化误差先于模型误差出现；
② 输入类型：JEV 只吃文本/JSON，sEMG 张量 (6×250×5) 需序列化，而 2609.36154 证明 JEV 读原始物理量约等于随机；
③ 目标冲突：Wei 依赖自回归，JEV 消灭自回归；
④ 效率反转：Wei 卖点是单层 TFE+TFD 训练 <2 min，换大模型该卖点归零。
**JEV 式方法论的正确位置是叠在 3DCNN-TF 之上的决策层，不是替换它。**

### Wei 2026 JBHI 八基线中的 TCN 反例（与 BSPC 2026 结论方向相反）（2026，编者分析，对照 10.1109/JBHI.2025.3589889 与 10.1016/j.bspc.2026.110933）

- **作者**：（本项目编者对照分析）
- **来源**：<https://doi.org/10.1109/JBHI.2025.3589889>
- **关键数字（A）**：Wei 2026：TCN 在八个基线中**最差**，IRASS 腕关节 nRMSE 12.5%/R² 82.6%，EPIC 下肢 10.6%/86.2%，双双垫底（3DCNN-TF 为 6.2%/5.5%）。BSPC 2026 髋部 IMU：Transformer 误差最低 RMSE 1.698°，TCN 仅因树莓派 5 p95 0.229 ms 而边缘最优
- **证据等级**：编者判断
- **编者结论**：不可行但需引证
- **本地工件**：**全文未入库，需补件或核查网页证据**
- **备注**：★ **两篇方向相反的结论，说明「TCN 更轻更好」根本不是普适规律，取决于信号。**
IMU 加速度：低维、强周期，TCN 够用；sEMG：高维、强个体差异、需肌肉解剖先验，3DCNN 的空间结构建模不可替代。
**这修正了本项目 2026-10-01 早先的判断**（当时仅凭 BSPC 2026 得出「D5 该走 TCN」）。
方法论教训：路线选择必须按**信号类型**分组论证，不能跨信号搬结论。
