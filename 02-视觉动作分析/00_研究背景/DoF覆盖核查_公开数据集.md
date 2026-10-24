# 公开数据集上肢/腕部 DoF 覆盖核查（2026-10-01）

> **为什么必须先做这件事**：研究决定 1 锁定「上肢腕-肘-肩」，决定 2 要建**离散运动学词表**。
> 词表的规模与合法性完全由「公开数据集到底提供哪些腕与前臂 DoF」决定。
> **若数据集只给到 wrist 且不含前臂旋转，那么「腕关节」在词表里就是一个末端点，
> 而旋前/旋后根本没有监督信号** —— 整个设计要重做，不是调参能解决的。
>
> **本文严格区分三类信息**：
> ✅ **已核实**：本轮从官方来源直读，附来源与可复算的证据。
> ⚠️ **待核实**：记忆或二手信息，本轮未能直读权威来源，**不得写入论文结论**。
> ❌ **已证伪**：本轮实测发现与记忆不符。
>
> 核查时点：2026-10-01

---

## 1. 一句话结论

> **三个拟用公开数据集（Human3.6M / AMASS(SMPL) / CMU MoCap）里，没有一个提供前臂
> 旋前/旋后（pronation/supination）的标注。** 本轮已从三者的官方来源逐条直读核实。
>
> 根因是**同一个**：前臂旋转需要骨架树中存在「前臂」这个中间关节，而
> **SMPL 的 24 关节树里 wrist（索引 20/21）之后没有任何关节**；AMASS 的主参数化正是 SMPL；
> Human3.6M 与 CMU 更是连手指都没采集。
>
> 因此：词表**不能**把「旋前旋后」当成公开数据可监督的维度。这不是采样问题，是**骨架定义问题**，
> 调参解决不了 —— 只能改走自采数据、换用含手部标记的数据集，或放弃该维度。

---

## 2. 已核实（✅）—— 本轮全部完成

### 2.1 权威记录（Crossref / 官方包直读，ID 逐条反查）

| 记录 | 权威标识 | 备注 |
|---|---|---|
| **SMPL** | ACM TOG 2015，**DOI `10.1145/2816795.2818013`**（被引 3624） | Crossref 正式题名仅 `SMPL`；重印本 DOI `10.1145/3596711.3596800` |
| **SMPL-X**（"Expressive Body Capture"，加手加脸） | CVPR 2019，**DOI `10.1109/cvpr.2019.01123`**（被引 1728） | 标题检索相似度 1.000 |
| **AMASS** | ICCV 2019，**DOI `10.1109/iccv.2019.00554`**（被引 1240） | 官方页作者：Mahmood, Ghorbani, Troje, Pons-Moll, Black；与 arXiv `1904.03278` 同源 |
| **MoSh** | ACM TOG 2015，**DOI `10.1145/2661229.2661273`** | Crossref 标题检索命中 |
| **Human3.6M** | CVPR 2018，**DOI `10.1109/cvpr.2018.00551`** | Crossref 正式题名为 "3D **Human Pose** Estimation in the Wild…"，与简称有出入 |
| **ISB Part II recommendation**（**腕角定义的唯一权威**） | **DOI `10.1016/j.jbiomech.2004.05.042`**，**Wu G** 等，J Biomech **2005;38(5):981–992** | 题名：*…—Part II: shoulder, elbow, wrist and hand*；本轮由**全文参考文献反查**得到 |
| **ISB Part I recommendation** | **DOI `10.1016/S0021-9290(01)00222-6`**，**Wu G** 等，J Biomech 2002;35:543–548 | 题名：*…—part I: ankle, hip, and spine*，**只管下肢** |
| **Joint coordinate systems of axes for coherence in reporting kinematic and kinetic data** | **DOI `10.1016/s0268-0033(00)00082-6`**，Clinical Biomechanics 2001 | 关节坐标系轴向一致性的方法学奠基；本轮已补入语料三 B1 |
| **MANO** | ❌ **本轮未取到** | Crossref 无记录；候选 DOI `10.1145/3272127.3273004` / `…3273013` 均 404；Semantic Scholar 429；`manopth.com` 不可达。**不得凭记忆写 DOI** |

### 2.2 关节树（来源：官方包 `vchoutas/smplx` 的 `smplx/joint_names.py`，本轮直读）

| 常量 | 关节总数 | `left_wrist` / `right_wrist`（0-based） | wrist 之后 |
|---|---|---|---|
| `SMPL_JOINT_NAMES` | **24** | **20 / 21** | **无任何关节 → 末端关节** |
| `SMPLH_JOINT_NAMES` | **73** | 20 / 21 | 22 起为 `left_index1` … 指关节 |
| `JOINT_NAMES`（SMPL-X） | 144 | 20 / 21 | 22 起为手指 |

> ⚠️ 注意：SMPL-H 与 SMPL-X 的**关节表里 wrist 之后直接是指关节**，
> 但这**不能据此断定腕部没有额外旋转参数** —— 手的相对姿势由 MANO 的
> body-pose 混合形状承载，不体现为独立关节。**该点仍未核实（见 V2/V3）。**

### 2.3 各数据集的腕/手可用性（逐个官方来源直读）

**Human3.6M**（官方站 `vision.imar.ro/human3.6m/` 原文）
> "…we use the same skeleton of **32 joints** … we reduce the number of joints to the relevant ones
> e.g. leaving only **one joint for each hand** and each foot."

→ 手/腕是**末端**，无手指 DoF，无前臂旋转。

**AMASS**（官方页 `amass.is.tue.mpg.de` 原文）
> "Here we use **SMPL** [26] … The method works for arbitrary marker sets, while recovering
> soft-tissue dynamics and **realistic hand motion**."

→ **主参数化是 SMPL**（wrist 末端）；所谓「hand motion」是**表面/软组织层面的**，
> **不是额外的腕部自由度**。

**CMU MoCap**（官方站 `mocap.cs.cmu.edu` 原文）
> "Note: The 'toe' and **'hand' joints in our motions tend to be noisy**, and may require some smoothing.
> The **'finger' and 'thumb' joints are added to the skeleton for editing convenience — we do not
> actually capture these joints' motions and any such data should be ignored**."

→ CMU 只有一个噪声的 `hand` 关节；**手指/拇指根本没采集**。无前臂旋转。


## 3. 已证伪（❌）—— 记忆中的 arXiv ID 不可信

本轮为了引用上述模型，从记忆里取了 arXiv ID 并直接核对，**三个全错**：

| 我记忆中的 ID | 我以为是什么 | **实际是什么** |
|---|---|---|
| `1603.08130` | SMPL | ❌ **等离子体物理**：EUV 诱导的氩/氢等离子体演化数值模拟 |
| `1808.06336` | MANO | ❌ **量子网络**：multipartite nonlocality 近似，journal_ref 竟为 `Phys. Rev. A 101, 062317` |
| `1506.02010` | MoSh | ❌ **神经动力学**：FitzHugh-Nagumo 模型的分岔与现象学响应 |
| `1904.03278` | AMASS | ✅ 正确 |

**这与本项目语料二 README 记录的教训完全一致**：
> 「记忆中的 ID 会错，且错了极难自查。实测 17 处错误中有多处 ID 指向天文学/数学/气象学等完全无关论文。」

**因此本项目的硬规则继续成立**：**所有 arXiv ID / DOI 必须经机器反查后才能进文档**。
本轮新增两条具体教训：

1. **标题带冒号会让 arXiv `ti:` 检索整条落空**。MANO 与 MoSh 的标题都含冒号，
   `search_query=ti:"Modeling and Capturing Hands and Bodies Together"` 返回 0 条。
   → 必须准备**去冒号的尾部检索**与**Crossref/DOI 回退**两条旁路。
2. **Crossref 的正式题名可能与常用简称不同**。SMPL 的 Crossref 题名就是 `SMPL`（4 个字母），
   论文全称只出现在 SIGGRAPH Seminal Papers 重印本（DOI `10.1145/3596711.3596800`）。
   → 按全称检索可能只命中重印本，**必须按 DOI 回取原始记录**。

---

## 4. 待核实（⚠️ 不得写入论文结论）

| # | 未核实项 | 为什么重要 | 状态 |
|---|---|---|---|
| **V2** | SMPL-H 腕部**是否携带额外旋转参数** | 关节表里 wrist 后接指关节，但手姿由 MANO 混合形状承载；两者不等价 | [ ] 需 MANO/SMPL-H 官方模型说明 |
| **V3** | MANO 权威标识与腕部 DoF 数 | 直接决定 V2 | [ ] Crossref 无记录，DOI 猜测全部 404 |
| **V4** | AMASS 各子集 marker 规格（哪些带手部） | 决定能否换用含手部数据（路径 B） | [ ] GitHub 出网受限，逐子集 README 未取到 |
| **V5** | CMU MoCap 31 关节的确切数值编号 | 已核实「手指未采集」，但编号表未取到（`cmuconvert` 文档出网不可达） | [ ] 官方站仅定性说明 |
| **V6** | 各库坐标系与单位归一方案 | 不同数据集腕角不可直接比较 | [ ] 依赖 ISB 腕坐标系全文（语料三 B1） |
| **V7** | **D435i 单目能否可靠估计前臂旋前/旋后** | **公开数据全部给不了旋前旋后，此项决定腕关节贡献能否成立** | [ ] 工程验证，Phase 3 前必须清零 |
| **V8** | 词表 DoF 清单定稿 | A/B/C 三路径择一 | [ ] 由研究者决定 |

## 5. 对研究设计的三条直接后果

### 后果 1：词表不能在公开数据上放「前臂旋前/旋后」

Human3.6M 与 SMPL 的 wrist 都是末端关节。**把旋前旋后放进词表 → 公开数据上无监督信号。**

三个可选路径（须用户裁定）：
- **A** 词表只放**肩/肘/腕屈伸/肩外展/肘外展**等公开数据确有的 DoF，旋前旋后留给自采 D435i
  （但 D435i 单目是否可靠测旋前旋后本身也需验证）；
- **B** 换用含手部标记的数据集做腕部监督（如 MPI-INF-3DHP 经 AMASS 的 smplh 子集），代价是数据量骤减；
- **C** 承认公开数据只能支撑「肩肘」层级，把腕关节的贡献完全放在自采数据上。

### 后果 2：不同数据集的腕角不可直接比较

6 个数据集各有自己的骨架与坐标系。词表若跨库，**必须先做坐标归一**，
否则同一个 token 在不同库里指的不是同一个量 —— 词表的「解剖含义」主张会直接崩掉。

**本轮对归一依据做了实测，结论比预想更受限：**

| 文件 | 首作者 / 出处 | 覆盖部位 | 状态 |
|---|---|---|---|
| **ISB Part II** | **Wu G** 等，J Biomech **2005;38(5):981–992**<br/>DOI `10.1016/j.jbiomech.2004.05.042` | **shoulder, elbow, wrist and hand** | ✅ **已入语料（本轮补入）**，付费闭取 |
| ISB Part I | **Wu G** 等，J Biomech 2002;35:543–548<br/>DOI `10.1016/S0021-9290(01)00222-6` | ankle, hip, spine | 只管下肢 |
| Clinical Biomechanics 2001 JCS | DOI `10.1016/s0268-0033(00)00082-6` | 方法学层面 | 已入语料，闭取 |
| ~~ISB 2007「extension」~~ | **M. Lenhoff** 等，J Biomech 2007;40 | 体外精度研究 | ⚠️ **不是标准本身**，此前被误当作依据 |

> **V6 结论**：腕关节坐标定义的**唯一权威是 Wu G 等 2005 年的 ISB Part II**，
> 本轮已把它补进语料三 B1（`10.1016/j.jbiomech.2004.05.042`）。
> 但该文为 Elsevier **付费闭取**（Unpaywall `oa_status=closed`，Europe PMC 无收录），
> **仍须经机构订阅取件后才能真正定稿腕角定义**。
>
> 在此之前，词表**只能按「未归一」处理**：所有库的腕角各自独立定义，
> 跨库比较一律标注坐标系来源，不做数值等同。

### 后果 3：自采 D435i 的价值上升，但风险也上升

既然公开数据给不了旋前旋后，**自采数据从「补充」变成「腕关节贡献的唯一来源」**。
这提高了 D435i 四问任务包的优先级，同时把「D435i 能否可靠估计前臂旋转」变成一个
**必须在 Phase 3 前验证的工程问题**（见下）。

---

---

## 6. 缺口登记位置

本核查产生的 **11 项缺口（J1–J3 已闭环留证 + V1–V8 未闭环）集中登记在
[`../06_附录/缺口台账.md`](../06_附录/缺口台账.md) 的 J 类**，
正文不重复登记 —— 遵循本项目「缺口登记集中在一处、不散落正文」的约定。

其中**必须在 Phase 3 前清零的两项**：
- **V7**　D435i 单目能否可靠估计前臂旋前/旋后（工程验证）
- **V8**　词表 DoF 清单定稿：三条路径择一 —— 只放公开数据确有的 DoF /
  换用含手部标记的数据集 / 承认公开数据只支撑肩肘层级、腕关节贡献全部来自自采数据
