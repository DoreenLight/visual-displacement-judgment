# D3 开源复现与替代

> 4 篇 ｜ ✅ 已入库 2 ｜ 需人工取件 2
> 本文件由 `build_jev.py` 生成。数字口径见 `../index.json`。

| # | 年 | 论文题名 | 载体 | 链接 | 证据等级 | 编者结论 | 本地 |
|---|---|---|---|---|---|---|---|
| 1 | 2025 | SalesRLAgent: A Reinforcement Learning Approach for Real-Time Sales Conversion Prediction | arXiv 预印本 | [2503.23303](https://arxiv.org/abs/2503.23303) | 第三方实测 | 仅作背景 | ✅ |
| 2 | 2025 | Confidence-Aware Routing for Large Language Model Reliability Enhancement: A Multi-Tier Framework | arXiv 预印本 | [2510.01237](https://arxiv.org/abs/2510.01237) | 第三方实测 | 条件可行 | ✅ |
| 3 | 2026 | Laya：开源非自回归 System 1 决策模型（ConvAI Innovations） | 开源模型（无同行评审论文） | [DOI](https://huggingface.co/convaiinnovations/laya/raw/main/README.md) | 官方一手 | 条件可行 | 需取件 |
| 4 | 2026 | System One 开源生态：Nimble / Kev / OpenThai / OpenJev / fast-browser-use | 开源项目集合（无 PDF） | [DOI](https://blog.invidelabs.com/system-one-models-jev-laya-open-alternatives/) | 社区报告 | 仅作背景 | 需取件 |

---

### SalesRLAgent: A Reinforcement Learning Approach for Real-Time Sales Conversion Prediction（2025，arXiv 预印本）

- **作者**：M, Nandakishor
- **来源**：<https://arxiv.org/abs/2503.23303>
- **关键数字（A）**：作者主张：2025 年 3 月即用 PPO 对序列表示做非自回归决策，权重与数据集均开源，而 JEV 无论文无权重无公开数据集
- **证据等级**：第三方实测
- **编者结论**：仅作背景
- **本地工件**：PDF 全文已在库：`01_参考论文/决策模型线_JEV/D3_开源复现与替代/SalesRLAgent_ A Reinforcement Learning Approach for Real-Tim.pdf`
- **备注**：**优先权争议的直接证据**，但需注意：作者同时是该开源模型的作者，立场不完全中立。其价值在于证明「非自回归决策」不是 2026 年才出现的新概念。

### Confidence-Aware Routing for Large Language Model Reliability Enhancement: A Multi-Tier Framework（2025，arXiv 预印本）

- **作者**：M, Nandakishor
- **来源**：<https://arxiv.org/abs/2510.01237>
- **关键数字（A）**：提出按置信度分层路由：高置信自动执行、中置信人工复核、低置信回退大模型
- **证据等级**：第三方实测
- **编者结论**：条件可行
- **本地工件**：PDF 全文已在库：`01_参考论文/决策模型线_JEV/D3_开源复现与替代/Confidence-Aware Routing for Large Language Model Reliabilit.pdf`
- **备注**：★★ **本语料最值得迁移到本项目的一篇**。它不依赖 JEV 任何东西，讲的是「用校准概率做控制流分叉」这一方法论 —— 正好对上 ViKiT「代偿识别不确定时如何处置」与 AAN「什么时候介入」的实际需求。
**D4 的 arXiv 2601.04982 是这一思路在辅助机器人上的更完整实现，优先引那篇。**

### Laya：开源非自回归 System 1 决策模型（ConvAI Innovations）（2026，开源模型（无同行评审论文））

- **作者**：ConvAI Innovations
- **来源**：<https://huggingface.co/convaiinnovations/laya/raw/main/README.md>
- **关键数字（B）**：基座 ModernBERT-large 395M 全量微调 + 2 层决策头 = 约 421M 参数；单题 P50 ≈38.4 ms（GPU）；10 题并批 156 ms；自测任务宏观准确率 83.8% vs JEV 67.8%；路由 ECE 0.009、内容安全 ECE 0.061；输入上限 512 token/题；RLCD 训练，7,313 步 1 epoch 约 1.96 小时
- **证据等级**：官方一手
- **编者结论**：条件可行
- **本地工件**：**全文未入库，需补件或核查网页证据**
- **备注**：**若要试 System One 路线，应该试这个而不是 JEV**：权重开放、可本地部署、数据不出域、延迟比 JEV 低一个量级。
但注意 512 token 输入上限对 sEMG 序列是硬约束 —— 说明 System One 范式天生面向**短问题**，不是长时序。
**这对 AAN 决策层反而是优点**：介入决策本来就是短问题。
其自报 ECE 0.009（路由）/ 0.061（内容安全）若能被独立复现，将比 JEV 的校准可信得多。

### System One 开源生态：Nimble / Kev / OpenThai / OpenJev / fast-browser-use（2026，开源项目集合（无 PDF））

- **作者**：Bespoke Labs / Jared Palmer / OpenThai / APUS 等
- **来源**：<https://blog.invidelabs.com/system-one-models-jev-laya-open-alternatives/>
- **关键数字（C）**：JEV 发布后一周内出现多个开源同构实现；APUS 发布 MIT 协议的 fast-browser-use 本地复现
- **证据等级**：社区报告
- **编者结论**：仅作背景
- **本地工件**：**全文未入库，需补件或核查网页证据**
- **备注**：记录生态格局用。**无 PDF、无法逐条核验**，仅作背景。
