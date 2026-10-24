# 2026-10-01 · Transformer 发展线语料库 + 上肢康复迁移推演

> **触发**：用户要求「先搜集 Transformer 领域的所有论文，整理发展线，推算康复领域后续可用方向与论文创新点」。
> **三项边界决策**（用户确认）：① 收集深度 = 索引 + 全部 PDF 入库；② 组织方式 = 双轴（时间阶段 × 技术线程）；③ 康复聚焦 = 上肢腕–肘–肩代偿。
> **本次只做文献层与推演层，未启动任何人体实验、硬件采集或模型实现。**

---

## 1. 交付物清单

| # | 交付物 | 路径 | 性质 |
|---|---|---|---|
| 1 | Transformer 语料库（7 线程 / 85 篇 PDF / 85 张摘要级定位卡） | `01_参考论文/Transformer_发展线/` | 参考资料（非精读、非全文翻译） |
| 2 | 可检索机器索引 | `01_参考论文/Transformer_发展线/index.json` | 机器生成，勿手改 |
| 3 | 需人工取件清单 | `01_参考论文/Transformer_发展线/需人工取件清单.md` | 需经学校数据库取件 |
| 4 | 发展线与康复迁移推演 | `00_研究背景/Transformer_发展线与康复迁移推演.md` | 文献支撑论证 |
| 5 | 追加创新点矩阵 | `02_创新点矩阵/Transformer前沿到上肢康复创新点矩阵.md` | 创新假设（**未验证**） |
| 6 | 可视化图谱页 | `08_三维动作分析展示台/Transformer_发展线图谱.html` | 本地自包含 HTML |
| 7 | 语料构建与自校验脚本（6 个） | `06_附录/Transformer语料工具/` | 可复现 |

## 2. 四个口径（严禁混写）

| 口径 | 数值 | 含义 |
|---|---|---|
| 理论搜索空间 | **109** | 人工编制的候选条目（含尚未核验的） |
| 元数据已核验 | **85** | arXiv 官方 API 确认存在、标题相似度达标 |
| PDF 实际入库 | **85** | 磁盘实测，含 `%PDF` magic + `file` 双重校验 |
| 未能核验 | **25** | arXiv 多次检索未命中，**不得作为可引证据** |

覆盖率只能表述为 **85/85 = 100%**（PDF ÷ 已核验），**不得**写成「已收录 109 篇」。

分布：阶段 S0=5 / S1=22 / S2=24 / S3=23 / S4=8 / S5=3；线程 T1=11 / T2=18 / T3=2 / T4=23 / T5=9 / T6=12 / T7=10；
迁移价值 高=15 / 中=34 / 低=36。

## 3. 核验方法与它抓到的错误

不硬编码 arXiv ID（避免臆造），改为 **标题检索 → 相似度比对**。该机制在本次工作中**实际拦截了 17 处错误**：

| 类型 | 实例 |
|---|---|
| 标题记忆错误 | 写成 "Scaling Laws for Vision Transformers"（实为 "Scaling Vision Transformers"） |
| 标题记忆错误 | "PhysPT: Modeling Physical Dynamics From Video…" 实为 "PhysPT: Physics-aware Pretrained Transformer…" |
| ID 指向无关论文 | `2204.02605` 实为一篇**天文学**光学导管论文（非 T2M） |
| ID 指向无关论文 | `2303.06553` 实为**数学**论文（Nakayama 代数） |
| ID 指向无关论文 | `2306.06079` 实为**天气预报**深度学习论文（非 MotionGPT） |
| ID 指向无关论文 | `2106.13230` 实为 Video Swin（非 TimeSformer） |
| ID 指向无关论文 | `2412.04045` 实为建筑能效 AI（非 Byte Latent Transformer） |
| ID 不存在 | `2305.21046`（Mamba-2 实为 `2405.21060`） |
| 误匹配 | "Sparse Transformer" 被配到 Adaptively Sparse Transformers（实为 `1904.10509` Generating Long Sequences with Sparse Transformers） |
| 检索变体缺陷 | arXiv `ti:"..."` 遇冒号整条落空（PoseFormer/METRO/Performer 等均受影响）→ 已加 5 级回退检索 |
| PDF 抽检 | 定位卡统一自述「不是全文翻译」；下载全部通过 PDF 合法性校验 |

其中 5 处错误 ID 出现在**我起草的推演文档**里，是 `verify_citations.py` 逐条反查 arXiv 官方 API 后发现并修正的。
若未做该步，会把天文学、数学、气象学论文当作「领域已有结论」的证据。

**仍未能核验、已进人工清单**：T2M、TATS、STFormer、GPT-2 技术报告、Sora 报告，以及 5 篇期刊论文
（TNSRE sBERT 手部运动学 / MDPI iTransformer-DTL / MDPI Sensors 上肢代偿 / Front Neurol 代偿泛化 / IEEE CAC 肘关节 sEMG）。

## 4. 对自检脚本的改造（v2/v3 → v4，20 个用例）

| 改动 | 动机 | 结果 |
|---|---|---|
| 备份目录 `_backup*` 排除出计数 | 昨晚新增的 `_backup_20260930_2214/` 被当现役条目，zh 计数 35→64 虚增 | **修掉既有 FAIL** |
| `_research_run` 排除出计数 | 网页抓取暂存层含 33 个 `javascript:` 伪链接残页 | **修掉 605 条 Case 1 误报** |
| Case 7 拆分语料二 | 语料二只有摘要级定位卡，若沿用「必须全文翻译」规则就得靠占位文件凑数 | 定位卡成为**声明式合法孪生** |
| Case 18 新增 | index.json 标为已入库但磁盘无文件 / 字节数不符 / 定位卡未自述非全文翻译 | 新增机器校验 |
| Case 19 新增 | 取件清单与 index.json 状态漂移 | 新增机器校验 |
| 计数块新增 `tf_pdf` / `tf_cards` | 语料二**独立口径**，不污染主语料 35 篇的既有数字 | 主语料数字不变 |

**收尾结果：20 个用例 PASS 20 / FAIL 0**（进入本轮前为 PASS 13 / FAIL 5）。

## 5. 推演结论摘要（详见两份推演文档）

**发展线主线**：Transformer 这十年的单一主线是「如何在不牺牲表达力前提下降低注意力代价」。
康复领域恰好卡在三个**中间态**上：离散 token 化（T5）、长时序线性化（T7）、物理约束注入（T6）。

**追加创新点**：P0-1 约束投影解码（软惩罚→硬保证）、P0-2 时间感知 token 粒度（替代固定 256 桶）、
P1-3 变量维注意力、P1-4 LoRA 个体化、P2-5 多模态融合、P2-6 约束生成。建议第一篇只做 P0 两项 + 既有 4 项。

**三条对现有设计的硬约束**（本轮新发现，比创新点本身更重要）：
1. **代偿识别不是天然的 Transformer 战场**：ETH Zurich 用 2 个 IMU + XGBoost 即达 macro-F1 0.80；另有工作端侧笔记本 22–30 fps。
2. **跨被试泛化已知会失败**：个体内 >90%，跨被试仅 50–70%。
3. **iTransformer-DTL（MDPI 2025）已在 sEMG/IMU 上用可学习 query + 非自回归做轨迹预测**，威胁既有创新点 ①④ 的独占性表述。

## 6. 边界声明

- 本轮**未**做全文翻译；语料二 85 张卡均为**摘要级定位卡**，不得表述为「已精读/已全文翻译」。
- P0–P2 创新点全部是**推断**，附证伪判据，未经本项目实验验证。
- 竞品指标（macro-F1、fps、跨被试比例等）均来自**论文摘要原文自述，本项目未复现**。
- 「领域尚未吃透」类判断基于本语料的覆盖度，**不等于**证明无人做过；只能写「在本语料范围内未见」。
- 未启动人体采集、患者实验或机器人控制；D435i 四问任务包状态不变（仍为 proposed）。

## 7. 复现

```bash
cd 06_附录/Transformer语料工具
python3 resolve_corpus.py     # 标题 → arXiv 规范元数据
python3 adjudicate.py         # ID 假设逐条反查证实/证伪
python3 download_pdfs.py      # 下载 OA PDF（%PDF + file 双重校验）
python3 gen_cards.py          # 摘要级定位卡
python3 build_index.py        # index.json / README / 各线程 LIST.md / 取件清单
python3 build_page.py         # 可视化页面
python3 verify_citations.py   # 推演文档内 arXiv 引用逐条反查
bash ../自检脚本/verify_kb.sh # 20 用例
```
