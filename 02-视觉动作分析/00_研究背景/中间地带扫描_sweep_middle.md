# 中间地带扫描（sweep_middle）— 补上从未查过的第三面，以及一次分类器翻车

> **这份文档解决什么**：前几轮我的检索面一直只有**两面** ——
> 康复/临床侧（`sweep_clinical.py`）和 CV/3D 姿态侧（`discover.py` / `sweep_vocab_claim.py`）。
> **中间那块（「多媒体/语义运动表示」）从来没人查过。**
>
> 而 F15（ICMCS 2012，*An Ontology for video human movement representation based on Benesh notation*）
> 恰恰就掉在中间 —— **既不是临床研究，也不是 3D 姿态估计**。
> 这是 V9e 那条老教训（「式子检索会漏」）的**第三次复现**，
> 而且这次漏的是**我压根没想过要去查的那一块**。
>
> **最后更新**：2026-10-01
> **执行工具**：`06_附录/生物力学语料工具/sweep_middle.py` → `middle_sweep.json`

---

## 0. 一句话结论

> ✅ **这个面确实有货，但它是 1990s–2010s 的「语义网 / 多媒体」老领地** ——
> **arXiv 上几乎不存在**（4/4 查询零命中，且已用正例验证查询语法无问题）。
>
> 🚨 **但更要紧的是方法论**：我的自动分带分类器**精度只有 2/36 ≈ 6%**。
> 36 条「中间带」命中里，真货只有 2 条，其余全是 **K2 同形异义污染** ——
> 判据词「interchange format」「data standard」「ontology」「description language」
> **全是通用计算机词汇**，于是捞回来 UUCP 邮件格式、食品数据交换格式、
> 核光谱直方图交换格式、航天器数据描述语言……**跟人体运动一条线都没有。**

---

## 1. 覆盖与口径

| 项 | 值 |
|---|---|
| 检索式 | **17** 条（M1–M17）|
| 来源 | Crossref 13 条 + arXiv 4 条 |
| 唯一条目 | **148** |
| 分类器判为「中间带」 | **36** → **人工复核后真货 2**（精度 ≈ 6%）|
| 其他分布 | 其他 85 · CV 12 · 艺术 14 · 临床 1 |
| arXiv 零命中 | **4 条**（全部 arXiv 查询均零）|

> ⚠️ **注意「临床 1」这个数字**：
> 17 条检索式覆盖整个中间地带，**只捞到 1 条临床相关**。
> 这反过来印证了上一轮的结论 —— **记谱/语义表示这条线与临床确实是断开的。**

---

## 2. arXiv 零命中（已做查询语法自检）

| 检索式 | 命中 |
|---|---|
| `all:"human movement description"` | **0** |
| `all:"human movement description" AND all:motion` | **0** |
| `all:"ontology" AND all:"human movement"` | **0** |
| `all:"motion capture" AND all:"interchange format"` | **0** |
| `all:"digital human" AND all:"motion representation" AND all:"standard"` | **0** |

> ✅ **零命中是真的，不是查询写坏了。** 自检过程如下
> （**这一条应该变成所有零命中结论的强制流程**）：
>
> | 探针 | 命中 | 用途 |
> |---|---|---|
> | `all:"human movement description"` | 0 | 待验证的零命中 |
> | `all:electron` | **3** | **正例** —— 证明查询语法能返回结果 |
> | `all:helicopter` | **3** | **正例** |
> | `ti:"human movement description"` | 0 | 题名限定，仍 0 |
> | `all:"movement description" AND all:human` | 3 | 放宽后有结果 |
>
> 📌 **所以「human movement description」（HMD）这个领域的旗舰术语，在 arXiv 上完全没有存在感。**
> 合理解释：这是一个**早于 arXiv 的领域**（语义网 / 多媒体，1990s–2010s），
> 工作多发表在 IEEE ICMMS、Neurocomputing、SemWeb 会议与期刊上。

---

## 3. 人工复核后的 2 条真货

### F16（候选）— 最接近的一条现代工作

| 字段 | 值（Crossref 反查）|
|---|---|
| 题名 | *Ontology-based human behavior indexing with multimodal video data* |
| 作者 | **Vizcarra Julio, Nishimura Satoshi, Fukuda Ken**（3 人，JAIST）|
| 出处 | **IEEE ICSC 2021**（15th International Conference on Semantic Computing），**pp. 262–267** |
| 标识 | `10.1109/icsc50631.2021.00052` |
| 开放获取 | ❌ `oa_status: closed`；无 arXiv → **仅标题级** |

- 📌 **与我们的关系**：**本体 + 人类行为 + 多模态视频**，是这个中间地带里**最接近我们方向**的现代条目。
- ⚠️ **但从题名看它的目标是「索引/检索」（indexing）**，不是「学习一个离散符号表」——
  ontology 通常是**人工构建的形式化知识**，与**学出来的 codebook** 性质不同。
  **这个区分恰好是 F7（VQ-HPS）那条差异论证的同一形状**，可以复用。
- 📌 **须订阅取正文才能定论**（登记 **F16a**）。

### 非撞车，但值得知道

**Favorskaya M, Nosov A（2014）**，*Hand-gesture Description Based on Skeleton Representation and
Hu Moments*，Frontiers in AI and Applications（book-chapter，`10.3233/978-1-61499-405-3-431`），
**Crossref 有完整摘要**，机器读取要点：

> "all stages for visual-based static **hand-gestures representation** were improved…
> The proposed robust hand-gesture **description** includes two main estimations:
> the **topological (approximated skeleton representation)** and geometric (**Hu moments**)…
> recognition accuracy … achieved **92–96 %**."

→ 应用是 **HCI**（CAD、人机交互、游戏、遥控、智能家居）。
→ **不撞**：手部静态手势识别、无关节角、无语法、无临床、**「description」在此是特征表示之意**
（**又是 K2 的词义变义**）。
→ 但它证明一件事：**这个中间地带 2010 年代的形态是「HCI 导向的手势特征工程」，不是符号语言。**

---

## 4. ✅ 这个面对定位的实际含义

| 面 | 状态 | 证据 |
|---|---|---|
| 临床侧符号记谱 | 1955–1997，**人工**，1998–2011 中断，2012 后转入舞蹈 | F14 + 37 条年表 |
| **中间带：语义/本体/描述语言** | ✅ **存在，但全是 1990s–2010s 的语义网老工作，且与临床、与 3D 姿态都不搭界** | 本轮 + F15 + F16 |
| CV 侧视觉自动符号记谱 | 2024→，面向艺术记录与机械手 | F11 / F13 / F5 |
| **三者交汇** | ❌ **空** | 三代引文网络：临床引用者 = 0 |

> 📌 **对论文的正面作用**：
> 「把运动离散符号化」这条线**在语义网时代被认真试过一次（ontology / description language）**，
> **但从未与 3D 姿态估计或康复结合，也从未被做成可学习模型。**
> 这恰好说明我们面对的**不是「没人想到」，而是「三代人各做了一半、互不相通」** ——
> 而 F11 已经证明**这个交叉点连 CV 侧都刚被重新发明**（2024）。

> ⚠️ **但也要如实说**：这个面的文献**大量是我读不到正文的**（closed），
> 所以**不能声称「这个领域我查透了」**。只能说「按 17 条式子扫，中间带无临床、无 3D 姿态交叉」。

---

## 5. 🚨 方法论翻车（这一条比结论更重要）

### 5.1 分类器精度 2/36 ≈ 6%

我的分带判据里，「中间带」的锚点是：

```
human movement description | description language | ontology |
semantic representation | interchange format | data standard | …
```

**问题在于：这些词本身几乎全是通用计算机词汇。**
于是 36 条「中间带」命中里混进来的是：

- `10.17487/rfc976` — **UUCP mail interchange format standard**
- `10.3403/30217587` — **Food data. Structure and interchange format**
- `10.1109/ieeestd.1993.115130` — **IEEE Standard Multichannel Analyzer (MCA) Histogram Data
  Interchange Format for Nuclear Spectroscopy**
- `10.3403/02080951` — **Space data and information transfer systems. Data description language. EAST specification**
- `10.1109/ieeestd.2017.7837032` — **IEEE Standard for Digital Test Interchange Format (DTIF)**
- 基因本体 / 网络访问控制本体 / 眼睛移动分类 / 拉��言数学文本本体 / 视觉概念本体检索 ……

> 📌 **规律**：一旦判据词是**领域通用词**而不是**领域特征词**，
> 分类器就会把整个 计算机标准与语义网 的语料吸进来。
> **「ontology」「interchange format」「description language」这三个词，
> 在人体运动领域没有任何区分度。**

### 5.2 ⇒ 由此形成一条纪律（登记 K10）

> 🚨 **规则式分类器只允许用于「召回」，绝不允许直接用于「结论」。**
> 每一类都必须**人工过一遍标题**才能进台账；
> 分类器输出的数字**不得**写进任何对外文档当证据。
>
> 这与 K2 是同一条纪律的两次执行：
> K2 说的是「Crossref 零命中不可信」，
> K10 说的是「**机器分类结果不可信**」——
> **机器可以帮你把 148 条缩到 36 条，但不能替你从 36 条里挑出 2 条。**

### 5.3 ✅ 另一条可复用的做法：**零命中必须带正例自检**

本轮所有 arXiv 零命中，我都先跑了 `all:electron` / `all:helicopter` 作为**正例探针**，
确认查询语法能返回结果，才敢把「0 命中」写成结论。

> 📌 **一个零命中结论，如果没证明「查询本身能工作」，就没有价值。**
> 这条应该成为本项目所有空白声明的**强制前置检查** ——
> 主张 C、§4 的临床侧空白、以及本轮的中间带空白，**都应该补做一次正例自检**。

---

---

## 7. ✅ 逐条人工判定（2026-10-02 补做，V9h 闭环）

> **为什么补这一节**：K10 当初定的纪律是「机器分类只用于把 148 条缩到 36 条，
> **从 36 条里挑出真货必须人工过标题**」—— 但当时**只凭印象报了个数**（2 条 / 6%），
> **没有逐条列全**。本轮逐条重过 36 条，结论被推翻（见下）。

### 7.1 更正：真货是 5 条，不是 2 条

| 判定 | 条数 | 占比 |
|---|---|---|
| ✅ 真货（2 已登记 + **1 新撞车 F21** + 2 背景） | **5** | **13.9%** |
| ⚪ 背景（不构成撞车） | 2 | 5.6% |
| ❌ 无关（标准/语义网/邻域领域噪声） | 31 | 86.1% |

> 🚨 **K10 当时报的是「2 条 / ≈6%」，是错的** —— 漏掉的里面有一条是真撞车。
> 📌 **教训追加**：**复核若只过一遍印象深的标题，等于没复核。**
> 逐条列全、逐条判，比凭印象报数更慢，但它是唯一可靠的。

### 7.2 逐条判定表（36 条，按判定排序）

| # | 年 | 题名（DOI） | 人工判定 | 理由 |
|---|---|---|---|---|
| 1 | 2012 | A comprehensive representation scheme for video semantic ontology and <br>`10.1016/j.neucom.2011.05.044` | ⚪ 背景（F21b） | 通用视频语义本体，非人体运动专属；Crossref 无摘要 |
| 2 | 2008 | Recognizing human actions based on silhouette energy image and global <br>`10.1109/afgr.2008.4813435` | ⚪ 背景（F21b） | 剪影能量图 + 全局运动描述，是**视觉特征描述**非符号化 |
| 3 | 2021 | Ontology-based human behavior indexing with multimodal video data<br>`10.1109/icsc50631.2021.00052` | ✅ **真货 → F16** | 本体 + RDF 知识图谱 + 老年人照护，已定案不撞车 |
| 4 | 2014 | Hand-gesture Description Based on Skeleton Representation and Hu Momen<br>`10.3233/978-1-61499-405-3-431` | ✅ **真撞车 → F21** | 2014 骨架表示 + Hu 矩做**静态手势描述**。输出是特征描述子非符号词表；只做手部；HCI 场景；无 Transformer |
| 5 | 2012 | An Ontology for video human movement representation based on Benesh no<br>`10.1109/icmcs.2012.6320129` | ✅ **真货 → F15** | Benesh 记谱法视频本体，2012，已定案不撞车 |
| 6 | 2026 | Eye Movement Classification via Global Semantic Representation with Fi<br>`10.1109/tmm.2026.3689446` | ❌ 无关 | 眼动分类，非身体运动 |
| 7 | 2026 | Methods of semantic annotation and ontological modeling of mathematica<br>`10.18127/j20729472-202601-18` | ❌ 无关 | LaTeX 数学文本本体 |
| 8 | 2024 | Visualizing OWL and RDF: Advancing Ontology Representation for Enhance<br>`10.5220/0012855900003756` | ❌ 无关 | OWL/RDF 可视化，语义网工具 |
| 9 | 2021 | Efficient Semantic Representation of Network Access Control Configurat<br>`10.5220/0010285305500557` | ❌ 无关 | 网络访问控制本体 |
| 10 | 2016 | Ontology development for semantic representation of GeneReviews<br>`10.7490/f1000research.1112740.1` | ❌ 无关 | GeneReviews 医学本体 |
| 11 | 2015 | Ontology and the Semantic Web<br>`10.1201/b18639-10` | ❌ 无关 | 《Ontology and the Semantic Web》书章（Multimedia Ontology） |
| 12 | 2015 | Ontology and the Semantic Web<br>`10.1201/b18639-3` | ❌ 无关 | 同上，另一卷 |
| 13 | 2010 | Research of video semantic retrieval system based on ontology<br>`10.3724/sp.j.1087.2010.00835` | ❌ 无关 | 视频语义检索系统，非人体运动 |
| 14 | 2010 | 3.1 Ontology Engineering<br>`10.1515/9783598441585.3.242` | ❌ 无关 | 社会语义网本体工程书章 |
| 15 | 2010 | 3.2 Community-based Ontology Engineering<br>`10.1515/9783598441585.3.272` | ❌ 无关 | 社区式本体工程书章 |
| 16 | 2002 | Die Spezifikationssprachen MSC und SDL / Teil 2: Specification and Des<br>`10.1524/auto.2002.50.2.a1` | ❌ 无关 | SDL 规约语言（德语，通信领域） |
| 17 | 1995 | A Service Request Description Language<br>`10.1007/978-0-387-34878-0_3` | ❌ 无关 | 服务请求描述语言 |
| 18 | 1993 | Standard Simulator Data Base (SSDB) Interchange Format (SIF) Design St<br>`10.21236/ada281516` | ❌ 无关 | 美军 SSDB 仿真器数据交换格式 |
| 19 | 1986 | UUCP mail interchange format standard<br>`10.17487/rfc976` | ❌ 无关 | UUCP 邮件交换格式（RFC 976） |
| 20 | — | Coding data for: Knowledge representation using Ontology-Based Social <br>`10.22452/rd/3vewuh` | ❌ 无关 | 马来文手稿数字图书馆本体 |
| 21 | — | Semantic representation learning for natural language understanding<br>`10.32657/10356/73345` | ❌ 无关 | 自然语言语义表示学习（NLP） |
| 22 | — | Information technology � Data Format Description Language (DFDL) v1.0 <br>`10.3403/30477175` | ❌ 无关 | DFDL 数据格式描述语言 v1.0 |
| 23 | — | Information technology � Data Format Description Language (DFDL) v1.0 <br>`10.3403/30477175u` | ❌ 无关 | 同上 |
| 24 | — | Space data and information transfer systems. Data description language<br>`10.3403/02080951u` | ❌ 无关 | 航天 EAST 数据描述语言 |
| 25 | — | Space data and information transfer systems. Data description language<br>`10.3403/02080951` | ❌ 无关 | 同上 |
| 26 | — | Space data and information transfer systems. Data description language<br>`10.3403/02826006` | ❌ 无关 | 同上，另一版 |
| 27 | — | COLLADA Audio : A Formal Representation of Sound in Virtual Cities by <br>`10.70675/c8af4187z2f2bz483fza40dz4072d79784d7` | ❌ 无关 | COLLADA Audio 场景描述语言 |
| 28 | — | IEEE Standard Multichannel Analyzer (MCA) Histogram Data Interchange F<br>`10.1109/ieeestd.1993.115130` | ❌ 无关 | 核光谱 MCA 直方图数据交换格式 |
| 29 | — | IEEE Standard for Digital Test Interchange Format (DTIF)<br>`10.1109/ieeestd.2017.7837032` | ❌ 无关 | DTIF 数字测试交换格式 |
| 30 | — | IEEE Standard for Digital Test Interchange Format (DTIF)<br>`10.1109/ieeestd.1998.89205` | ❌ 无关 | DTIF 上一版 |
| 31 | — | Hand Data Interchange Format, Standardization<br>`10.1007/springerreference_71022` | ❌ 无关 | Hand 数据交换格式（**注意：不是手势**） |
| 32 | — | Food data. Structure and interchange format<br>`10.3403/30217587u` | ❌ 无关 | 食品数据结构与交换格式 |
| 33 | — | Biometric Data Interchange Format, Standardization<br>`10.1007/springerreference_70976` | ❌ 无关 | 生物特征数据交换格式 |
| 34 | — | Finger Data Interchange Format, Standardization<br>`10.1007/springerreference_70977` | ❌ 无关 | Finger 数据交换格式（**注意：不是手势**） |
| 35 | — | Food data. Structure and interchange format<br>`10.3403/30217587` | ❌ 无关 | 同上 |
| 36 | — | Data interchange format for simulated and measured data (ISMD)<br>`10.3403/01426795` | ❌ 无关 | ISMD 仿真/实测数据交换格式 |

### 7.3 本节最重要的产出：F21

> **Favorskaya M, Nosov A（2014），*Hand-gesture Description Based on Skeleton Representation
> and Hu Moments*，IOS Press，`10.3233/978-1-61499-405-3-431`（Crossref `type=book-chapter`，已反查）**

摘要原句：

> "The proposed robust hand-gesture description includes two main estimations: the
> **topological (approximated skeleton representation)** and geometric (Hu moments)
> descriptions of hand-gesture."（识别精度 92–96%）

🚨 **「用骨架表示来描述手部运动」这个组合 2014 年就存在。**

⭐ **它为什么前三轮全漏**：摘要自述应用为**计算机辅助设计、人机交互、虚拟游戏、智能家居控制** ——
它落在 **HCI / CAD 语料**里，**既不在康复检索面，也不在 3D 姿态估计检索面**。
**V9h 登记时写的「没人查过中间那块」是对的，而且那块里确实藏着漏掉的工作。**

✅ **不构成机制级撞车的五条**（已写入 `论文定位陈述_收窄版.md` §1 红线表）：
① 输出是**特征描述子**（近似骨架 + Hu 矩），**不是离散符号词表**；② **静态手势**，无时间维度；
③ 无 Transformer（2014）；④ HCI 场景非临床；⑤ **只做手部**，不涉及腕/肘/肩。

⚠️ **不得写「首次用骨架描述人体运动」**；可写「首次在**离散符号词表**框架下描述上肢运动学量」。

## 6. 我没做的事（诚实边界）

- ❌ **只扫了 17 条式子**，虽然比前两轮宽，但**仍是定向扫，不是 discover 式铺开**
- ❌ **F16（F16a）与 F15（F15a）都只有标题级** —— 都 closed，无 arXiv
- ❌ **没有读 Semantic Web 会议的老正文**（ICSC / ICMMS / IEEE WIC），
  这块很可能还有更多本体类工作
- ❌ **没查非英语文献** —— 这个中间地带**德语区贡献很大**
  （年表里已见法语的 Benesh 民族舞蹈学、Circus 论文）
- ❌ **没追 F16 的引文网络**（Vizcarra 2021 是 2021 年的，理论上可追）
- ❌ **零命中全部只在 arXiv 上验证**，Crossref 侧零命中按 K2 不作结论
