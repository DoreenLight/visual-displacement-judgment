# D1 JEV本体与SystemOne范式

> 4 篇 ｜ ✅ 已入库 2 ｜ 需人工取件 2
> 本文件由 `build_jev.py` 生成。数字口径见 `../index.json`。

| # | 年 | 论文题名 | 载体 | 链接 | 证据等级 | 编者结论 | 本地 |
|---|---|---|---|---|---|---|---|
| 1 | 2023 | UDE: A Unified Driving Engine for Human Motion Generation | CVPR 2023 (IEEE/CVF), pp. 5632-5641 | [DOI](https://openaccess.thecvf.com/content/CVPR2023/html/Zhou_UDE_A_Unified_Driving_Engine_for_Human_Motion_Generation_CVPR_2023_paper.html) | 官方一手 | 仅作背景 | ✅ |
| 2 | 2026 | Calibrated Decision Models for Autonomous Penetration-Testing Harnesses: JEV and Laya as System One Decision Layers for LLM-Driven Pentest Agents | arXiv 预印本 | [2609.28940](https://arxiv.org/abs/2609.28940) | 第三方实测 | 条件可行 | ✅ |
| 3 | 2026 | Jev API 官方文档（typesafe.ai） | 厂商文档（无同行评审） | [DOI](https://api.typesafe.ai/docs) | 官方一手 | 不可行但需引证 | 需取件 |
| 4 | 2026 | Jev 发布博文与 Workflow Evals（TypeSafe AI，2026-09-15） | 厂商博文（无同行评审） | [DOI](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | 厂商自述 | 不可行但需引证 | 需取件 |

---

### UDE: A Unified Driving Engine for Human Motion Generation（2023，CVPR 2023 (IEEE/CVF), pp. 5632-5641）

- **作者**：Zhou Zixiang、Wang Baoyuan
- **来源**：<https://openaccess.thecvf.com/content/CVPR2023/html/Zhou_UDE_A_Unified_Driving_Engine_for_Human_Motion_Generation_CVPR_2023_paper.html>
- **关键数字（A）**：非自回归 token 打分思路的早期实现：VQ-VAE 量化 + GPT 式自回归
- **证据等级**：官方一手
- **编者结论**：仅作背景
- **本地工件**：PDF 全文已在库：`01_参考论文/Transformer_发展线/T5_动作生成与离散token/2023_Zhou_UDE_统一人体动作生成驱动引擎.pdf`（跨库引用，不重复下载）
- **备注**：⚠️ **标题极易误认**：这不是 JEV 模型。它是语料二已收录的 UDE 动作生成论文。列入 D1 是因为它的「离散码 + 序列打分」结构是 JEV 所宣称范式的动作领域先例。语料二已有同一篇 PDF，此处只作交叉索引，不重复入库。；本次订正题名 UDE（非 JEV），复用语料二原件。

### Calibrated Decision Models for Autonomous Penetration-Testing Harnesses: JEV and Laya as System One Decision Layers for LLM-Driven Pentest Agents（2026，arXiv 预印本）

- **作者**：Barbosa, Joas Antonio dos Santos
- **来源**：<https://arxiv.org/abs/2609.28940>
- **关键数字（A）**：JEV 单次请求 P50 延迟 236–276 ms；Laya 在 T4 GPU 上 32.8–39.5 ms
- **证据等级**：第三方实测
- **编者结论**：条件可行
- **本地工件**：PDF 全文已在库：`01_参考论文/决策模型线_JEV/D1_JEV本体与SystemOne范式/Calibrated Decision Models for Autonomous Penetration-Testin.pdf`
- **备注**：本语料**唯一一篇给出 JEV 独立延迟实测的 arXiv 论文**。应用域是渗透测试（安全），不是康复，但延迟数字与架构分析可迁移。
**对 AAN 决策层的意义**：JEV 的 236–276 ms 对康复闭环仍太慢，Laya 的 32.8 ms 在「决策层」这个量级上是可用的。

### Jev API 官方文档（typesafe.ai）（2026，厂商文档（无同行评审））

- **作者**：TypeSafe AI
- **来源**：<https://api.typesafe.ai/docs>
- **关键数字（B）**：**权威契约（OpenAPI 3.1.0, version 0.2.0）**：POST /v1/systemone 请求体 = {state, model, questions}；questions 是**具名映射**，answer 按同名返回。state 类型为 string | object | array（**接受结构化 JSON，不只纯文本**）。模型名示例 jev-latest，release_date 2026-09-15（佐证发布日）
- **证据等级**：官方一手
- **编者结论**：不可行但需引证
- **本地工件**：**全文未入库，需补件或核查网页证据**
- **备注**：**网页型证据，无 PDF。** 已抓取存档：`D1_JEV本体与SystemOne范式/..._网页存档.md`

⚠️ **2026-10-01 抓取时更正了三处此前记错的事实：**
1. 第三原语官方拼写是 **`noul`**（yes/no 概率），**不是 `null`**；
2. `state` 接受 **object | array**，即结构化 JSON 也能进 —— 「JEV 只吃文本」的表述不准确，准确说法是「**只吃 JSON，不吃图像/音频/视频**」；
3. `choice` 与 `score` 的 answer **自带 `confidence` 字段**，官方描述是「use lower values to flag uncertain selections for review」——
   **这正是 R1 决策层要的东西，且是厂商自己设计的**，不是我的推断。

另：`choice` 返回完整 `probabilities` 分布而非只给 top-1；`score` 返回概率加权均值，官方注明「may fall between integer levels」（连续量）。

### Jev 发布博文与 Workflow Evals（TypeSafe AI，2026-09-15）（2026，厂商博文（无同行评审））

- **作者**：TypeSafe AI
- **来源**：<https://typesafe.ai/blog/introducing-system-one-models-and-jev>
- **关键数字（C）**：自称 193.6× 更快、444.6× 更便宜；自有四工作流评测准确率 67.8%，低于自家 Sol 74.1% 与 Opus 5 73.1%；参考答案是 GPT-6 Astra 与 Fable 5.1 的平均（即模型互评，非人工标注）
- **证据等级**：厂商自述
- **编者结论**：不可行但需引证
- **本地工件**：**全文未入库，需补件或核查网页证据**
- **备注**：**这是全部营销数字的源头，也是最需要打折的一手材料**。TypeSafe 自承评测工作流由自家模型团队构造，「0% 幻觉」按其文档自述「not empirical」。
