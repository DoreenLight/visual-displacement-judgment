# 生物力学语料工具（可复现构建链）

> **现行位置**：`06_附录/生物力学语料工具/`。由已确认清单迁入 40 文件（22 脚本、18 JSON），不搬动原文、历史备份或线程目录。
> **职责**：BM 检索、人工筛选、元数据证实、取件、入库、索引及证据扫描；与 TF 流程分开维护。
> **当前口径**：111 条已证实（79 A / 32 B）；100 份全文（83 PDF + 17 XML）；缺件 11。动态状态以[语料索引](../../01_参考论文/生物力学线/README.md)及[取件总清单](../待下载论文总清单.md)为准，不把阶段报告当现行待办。

## 本地构建与验收（项目根目录）

```bash
python3 06_附录/生物力学语料工具/build_biomech.py
python3 06_附录/gen_acquisition_list.py
bash 06_附录/自检脚本/verify_kb.sh
```

取件阶段：`download_oa.py`、`recover_oa.py`、`bm3_fetch.py`；浏览器取得文件后用 `bm3_ingest.py` 校验并入库，`bm3_probe.py` 生成阶段缺件报告。联网、取件和人工裁定不属于本地构建验收，不应为了验收重跑。

补充材料工具 `fetch_epmc_supp.py` 必须从项目根目录运行，其默认路径是根相对路径；补充材料仍位于 `06_附录/补充材料/`。arXiv 专用下载器仍由 TF 维护，按下面明确的跨目录路径调用，不复制第二份。

## 方法与留证说明

以下内容从原混合工具 README 的 BM 部分迁入。标有日期的实测数据保留当时口径；旧备份、原执行日志和历史扫描产物不因目录整理改写。迁移及回滚清单见[已确认提案](../../05_进展日志/2026-10-02_论文收尾与目录整理/目录迁移提案.md)。


> **为什么需要第二套工具**：语料二只查 arXiv，因为 Transformer 论文几乎都在 arXiv。
> 但肌肉骨骼动力学 / 上肢康复 / 临床生物力学的关键论文发表在 J Biomech、Clinical Biomech、
> TNSRE、JBHI、Frontiers、Springer 等，**关键词实测 arXiv 命中 0 篇**。
> 继续用 `resolve_corpus.py` 会把它们全部判成「arXiv 未收录」——**那是工具盲区，不是事实。**
> 故本部分的解析器改为 **Crossref（标题+DOI 权威记录）+ Unpaywall（按 DOI 的 OA 定位）+ arXiv**。

## 6. 语料三执行顺序

| 顺序 | 脚本 | 作用 | 产物 | 联网 |
|---|---|---|---|---|
| 1 | `discover.py` | 5 条线程 × 90 条检索式，多源打捞候选池 | `biomech_pool.json` | 是 |
| 2 | `curate.py` | 上肢加权 / 碎片过滤 / venue 配额 / 模糊去重 → 短名单 | `biomech_shortlist.json` | 否 |
| 3 | `select_candidates.py` | **人工裁定**（PICKS 表）→ 候选清单 | `biomech_candidates.json` | 否 |
| 4 | `resolve_multisource.py` | Crossref+Unpaywall 交叉证实，抓 DOI 冲突 | `biomech_resolved.json` | 是 |
| 5 | `patch_arxiv_abs.py` | arXiv API 被限流时的第三条通路（abs 页面 citation_* 标签） | 回填 `biomech_resolved.json` | 是 |
| 6 | `build_biomech.py` | 可检索索引 + 五份线程 LIST + 定位卡 + **取件入口指针** | `01_参考论文/生物力学线/**` | 否 |
| 7 | `../Transformer语料工具/fetch_arxiv.py` | 按 arXiv ID 优先取件（三重校验） | 线程目录 `*.pdf` | 是 |
| 8 | `download_oa.py` | 按 Unpaywall 官方 OA 位取件（三重校验） | 线程目录 `*.pdf` | 是 |
| 8b | `recover_oa.py` | **替代 OA 通路**：Europe PMC JATS 全文 XML + arXiv 预印本 | 线程目录 `*.xml` | 是 |
| 9 | `build_page_biomech.py` | 双轴矩阵 + 可检索清单展示页 | `08_三维动作分析展示台/生物力学语料图谱.html` | 否 |

## 7. 一键复现（语料三）

```bash
cd 06_附录/生物力学语料工具
python3 discover.py --per-query 25                 # 全线程；OpenAlex 限流时改用 --sources crossref,pubmed
python3 curate.py --all --top 90 --venue-quota 6
python3 select_candidates.py                       # 人工裁定在 PICKS 表里，脚本只执行
python3 resolve_multisource.py
python3 patch_arxiv_abs.py                         # 补 arXiv API 限流漏掉的条目
python3 build_biomech.py
python3 ../Transformer语料工具/fetch_arxiv.py --dest ../../01_参考论文/生物力学线/_待并入 --ids <id,...>
# 取件后须按 index.json 的规范文件名并入对应线程目录，再删除暂存目录（现行文件只留一个规范路径）
python3 download_oa.py --retry-failed
python3 build_page_biomech.py
bash ../自检脚本/verify_kb.sh
```

## 8. 语料三的关键设计决策

| 决策 | 理由 |
|---|---|
| **不用 OpenAlex 做证实与 OA 定位** | 实测该源在本机 IP 触发**硬 429**（带不带 `mailto` 都一样）。改用 Unpaywall：它以 **DOI 为键**——DOI 是权威标识，比标题检索更不易错配，且直接返回 `url_for_pdf` |
| **OpenAlex 仍保留在 `discover.py` 发现阶段** | 它的召回广，池子已建好；B5 因限流只由 Crossref 生成，该口径已在 `index.json` 与页面限制栏登记 |
| **PubMed 必须加 `[Title]` 限定** | 实测不加限定时 `muscle redundancy problem motor control` 会返回「uterine mechanics」这类完全无关条目，那是假信号，会污染交叉证实计数 |
| **`select_candidates.py` 允许回落到候选池** | 短名单受 venue 配额与模糊去重影响，会把同刊物但同样重要的条目挤掉。短名单是裁定辅助，**不是硬门槛** |
| **新增 Europe PMC JATS 全文 XML 通路**（`recover_oa.py`） | Unpaywall 判「无 PDF 直链」的条目里，**有一部分 PMC 实际有开放全文**。
  实测三种端点：`?pdf=render` **403**、REST `fullTextPdf` **404**、REST **`fullTextXML` 正常**（返回完整 JATS 正文）。
  且 **XML 比 PDF 对本项目更有用** —— 能取到结构化正文、表格与具体数值定义（如 ISB 坐标轴、ROM 限值），
  正是 `00_研究背景/DoF覆盖核查_公开数据集.md` 的 V6 所需要的东西。校验要求含 `<body>`，防止把错误页当全文 |
| **arXiv 预印本作为替代来源，但必须打标记** | 工程/计算机类论文常有预印本。**同一工作**，预印本可合法取用，
  但**必须打 `preprint_of` 标记**，注明「须与正式版核对后方可引用数值」，否则会把预印本数值当正式版写进论文 |
| **下载器补齐完整浏览器请求头** | 只带 UA 时 Wiley 与 MDPI 一律 403、Springer/BMC 返回 3 KB JS 挑战页。补 `Referer`/`Sec-Fetch-*` 后这些**金色 OA** 刊物可正常取件——这不是绕过付费墙，是让请求看起来像正常浏览器访问开放 PDF |
| **反爬验证页一律不绕过** | MDPI 返回 `bm-verify` 反爬令牌页。程序化绕过属**规避访问控制**，不做。这类条目单列为「OA 但被出版商拦截」，附精确 OA 链接交人工浏览器下载 |
| **证据等级 A/B/C/D 分级** | A = ≥2 独立来源确认标题；B = 单来源 ≥0.90；C = 0.74–0.90 或 **DOI 冲突**；D = 未达阈值。**只有 A/B 进可引用索引**，C/D 单列 `pending_adjudication` |
| **DOI 冲突独立成类** | 这类是最危险的错误引用（语料二曾出现登记的 EMBC DOI 实际指向一篇心脏 MRI 论文）。即使标题相似度 1.0 也一律降级待裁 |
| **五组口径严格分列** | 理论搜索空间 / 实际命中 / 人工裁定 / 元数据证实 / 实际入库 —— 任何两者的差额都要能解释，不允许混写成「N 篇」 |

## 9. 语料三的已知限制（如实声明）

- **B5 线程候选池仅由 Crossref 生成**（OpenAlex 限流），召回面窄于其他线程。**其「未命中」不等于「不存在」。**
- **反爬拦截未绕过**：MDPI / Springer / BMC / Wiley 部分条目取件失败，原因与补救已逐条记入
  [`../待下载论文总清单.md`](../待下载论文总清单.md) §5（**全项目唯一取件入口**，2026-10-03 起五份清单合并为一份）。
- **检索为英文单语**，未覆盖中文与非英语文献。
- **定位卡是摘要级**，不是全文翻译。任何数值、方法细节、结论都必须回原文核对。
- 相关度、证据范式、线程归属、收录理由**全部是人工判断**，不是来源官方字段。
- 跨源重复：同一工作可能同时出现在 B1 与 B5（如标记无关动捕的验证工作），这是**有意的交叉引用**，不是重复收录。

---

## 附：预印本标记如何落盘（2026-10-01 补，含两个实测坑）

### 为什么需要这条

`recover_oa.py` 会在报告里写「arXiv 预印本 2312.08291v4 —— **需与正式版核对后方可引用数值**」，
但**预印本 PDF 与正式版 PDF 在磁盘上文件名完全一样**，肉眼无法区分。
若不落到索引与定位卡，后续任何人（或 `mine_fulltext.py` 自动挖掘）都可能把预印本当正式版引用数值。

### 现在怎么落的

1. `build_biomech.py` 新读 `biomech_recovery_report.json`（此前**只读下载报告**，这是缺陷），
   用正则 `^arXiv 预印本\s+(\S+?)，` 抽出预印本 ID；
2. **且仅当盘上是 PDF 时才打标记** —— 若在盘全文是 Europe PMC JATS XML，那是**正式版**，不能标；
3. 写入 `index.json` 的 `preprint_of` 字段并追加到 `note`；
4. 4 张受影响的定位卡已手工插入警示横幅（`build` 对已有卡 `if card.exists(): continue`，有意不覆写）。

### 实测踩到的两个坑（勿重犯）

| 坑 | 现象 | 正解 |
|---|---|---|
| **子串误判** | 报告里「arXiv **无相似**预印本」也含「预印本」三字 → 用 `'预印本' in note` 会把 64 条里 **59 条误标** | 必须**行首**匹配 `^arXiv 预印本 <id>，` |
| **正式版被误标** | Dasgupta 2023 命中过预印本 `2209.14456`，但盘上是 Europe PMC XML = **正式版** | 标记条件必须**叠加「盘上是 PDF」** |

**当前实际结果：4 条被标记**（VQ-HPS / Learning 3D Human Pose / Optimal-state Dynamics / PoseAug），
第 5 条（Dasgupta 2023）因在盘全文是正式版 XML 而**正确地未标记**。

---

## 附二：`sweep_vocab_claim.py` —— 按**主张**定向撞红（2026-10-01 新建）

### 为什么它和 discover.py 不重复

| | `discover.py` | `sweep_vocab_claim.py` |
|---|---|---|
| 驱动方式 | 按**线程**铺开 | 按**主张**定向 |
| 目标 | 找全（广度）| **证伪**（深度）|
| 用法 | 建语料三的候选池 | 只回答「这条新颖性主张会不会撞」|

**为什么必须有它**：「离散运动学词表仍开放」是整个选题的立身之本，
但此前**只靠 B4 线程 6 篇支撑**，而那 6 篇没有一篇在做临床关节角 token 化。
第三轮意外撞到 `2607.02796` 是**顺手挖 PDF 碰上的** —— 说明这一刀从未系统砍过。

### 本轮结果（`vocab_claim_sweep.json`，13 条式子）

- 🚨 **撞到一条真先行**：arXiv `2608.03127` *DigitCode: Symbolic Tokenization of Hand Motion by
  Anatomical Units*（2026-08-04）—— 已用 Hand Labanotation 做**离散符号化关节表征**。
  **该主张必须收窄**，已登记 F5。
- **两条零命中**：`all:"kinematic vocabulary"`、`all:"angle" AND all:"vector quantization" AND all:"pose"`
  在 arXiv 上**各 0** —— 这是"开放"一侧的正面证据。

### ⚠️ 踩到的坑：**Crossref 的零命中不可信**（重要）

本轮 Crossref 侧的式子大面积**被同形异义的文献污染**：

| 式子 | 实际返回 |
|---|---|
| `visual to joint angle translation model` | 返回 **"Visual Angle"（视觉角度，感知科学教科书章节）** |
| `biomechanical language grammar joint` | 返回**纯语言学专著**（German grammar、Language Surveys）|

→ **所以 Crossref 这些「没找到真正相关文献」的结果，不能当作"该方向空白"的证据。**
→ 只有 **arXiv 全字段检索**（`all:"..."` AND `all:"..."`）的零命中才可引用。
→ 记录这一点是为了**防止以后自己拿被污染的零命中去论证空白**。

### null 结果的地位

脚本明写：**null 结果也是结论**（指"这些式子在公开索引里取不到"），
但**不等于"不存在"**。入库仍须走 `curate → select → resolve → build` 可复现链 + 多源交叉证实。

---

## 附三：`sweep_intersection.py` —— 第二阶段，**证伪「交集真空」这句话本身**

### 为什么它是必需的

第一阶段撞到 4 篇先行工作（`2402.17192` / TMRB 2026 / `2607.02796` / `2608.03127`），
由此归纳出「它们都撞在生物力学侧，没有一篇撞在离散符号化机制本身，交集看起来是空的」。

**但那 4 篇全是"顺手碰上的"** —— `2607.02796` 是第三轮挖 PDF 时偶然遇到的。
**用 4 个偶遇样本断言一个交集为空，方法上不成立。** 这个工具就是补这一刀。

### 检索设计

三段式交集：`{上肢/腕/肘/肩/前臂}` × `{单目/单相机/RGB-D/无标记/视频}` × `{离散/符号化/词表/codebook/VQ}`，
共 12 条 arXiv 全字段式子。结果写在 `intersection_sweep.json`。

### ✅ 四条零命中（空白一侧的正面证据）

```
all:wrist AND all:monocular AND all:discrete                      → 0
all:"joint angle" AND all:monocular AND all:token                  → 0
all:vocabulary AND all:"joint angle"                               → 0
all:"clinical" AND all:"joint angle" AND all:token                 → 0
```

### 有命中的式子，逐条判过都不是撞车（摘要级）

| arXiv | 实际是什么 |
|---|---|
| `2601.09949` **Kinematic Tokenization** | ⚠️ **金融时间序列**（资产价格、样条系数、交易决策）。标题撞，领域不撞 |
| `2609.15012` Atomic Motion Coordinate | 机器人 VLA 操作，**vision withheld** |
| `2608.24334` SeMoCo | 文本生成动作的 codec（动画域）|
| `2602.23738` Continuous sEMG → Discrete Muscle State Tokens | **输入 sEMG 非视觉**；思路相邻 |

### ⚠️ 由此得到一条命名告警

**「Kinematic Tokenization」这个方法名已被 `2601.09949` 占用（金融领域）。**
论文里不要直接拿它当方法名，检索时会被这篇混淆。

### 三条已登记的局限（别把"检索未命中"说成"证明不存在"）

1. 英文单语，**未覆盖中文与非英语文献**；
2. arXiv **不含期刊论文**；Crossref 因异义污染**不可用作空白证据**（见附二）；
3. **未做引文反向追溯（snowballing）** —— 最可能捞出式子搜不到的邻近工作，**Phase 2 优先补**（已登记 V9e）。

---

## 附四：**哪些数据源在本机是硬失效的**（2026-10-01 实测，别再试）

查 V3（MANO 权威标识）时把公开检索路径走了一遍，全败。把结果记下来是为了
**下一个会话不要把这 7 条原样再走一遍**。

| 源 | 状态 | 实测证据 |
|---|---|---|
| **OpenAlex** | 🔴 **本机 IP 硬 429** | 多轮实测均 `HTTP Error 429: Too Many Requests` |
| **Semantic Scholar** | 🔴 **本机 IP 硬 429** | 本轮连试 3 次（2402.17192 / 2607.02796 / 2608.03127）全 `HTTP Error 429` |
| **DBLP API** | 🔴 **反爬挑战页** | 返回 HTML `Making sure you're not a bot!`（Anubis），**不绕过** |
| **Crossref 题名检索** | 🟡 **可用但会漏** | 检索 `MANO: Modeling and Capturing Hands and Bodies Together` 返回的是「Hands Together」「Put your hands together」等**同形异义无关文献**；按作者+Romeo+SIGGRAPH Asia 也捞不到 |
| **arXiv API** | 🟢 可用 | 但 **MANO 是 SIGGRAPH Asia 2018 论文，不在 arXiv**，检索无果属正常 |
| **OpenCitations COCI** | 🟢 **可用（新发现）** | `api/v1/citations/<DOI>` 可换「谁引了它」，**是前向引文追溯的可用通路** |
| **Unpaywall** | 🟢 可用 | 但命中的 OA 位置可能受反爬保护（如 Salford 机构库 Cloudflare） |
| **Europe PMC** | 🟢 可用（限生物医学） | 对 CS 预印本无效；但**对已正式发表的生物医学期刊论文极强**（本轮靠它取到 JMIR 2026 发表版全文 XML 与 Pohl 等焦点小组研究）|

### 🟢🟢 **附四之二：新发现一条合法免费 OA 通路 —— 会议自建开放库**

> **2026-10-01 晚间实测发现。这条通道解决了「ACM/IEEE 闭取但同组有同系列会议论文」的一类困境。**

| 库 | 覆盖 | 取件方法 | 实测 |
|---|---|---|---|
| **`ojs.aaai.org`** | **AAAI 官方开放 proceedings（全部年份）** | `GET /index.php/AAAI/article/view/<id>` 取 `citation_pdf_url`，再 `GET <该 URL>` | ✅ **200，PDF 直出** |

**取件要点（踩过的坑记下来）**：
1. `curl` 必须加 **`--compressed`**，否则拿到的是 **gzip 二进制**（本轮首次尝试时 4783 字节解出乱码）
2. DOI 后缀的数字**就是 article id**：`10.1609/aaai.v32i1.12270` → `article/view/12270`
3. PDF 直链形如 `https://ojs.aaai.org/index.php/AAAI/article/download/<id>/<galleyId>`

**本轮用它取到的两篇**：
- `10.1609/aaai.v32i1.12270` = **F6 pose grammar 官方 camera-ready**（2.26 MB）
  → **本项目最吃重的那条撞车，此前语料里只有摘要卡、现在有官方全文**
- `10.1609/aaai.v39i5.32511` = **F17**（F11 同组第三篇，5.72 MB）

> 📌 **可复用判断**：**遇到 IEEE/ACM 闭取的论文，先查同一课题组是否有同系列会议论文已进开放库**，
> 哪怕不是同一篇，也能拿到**作者自述**（作者在正文里常会描述自己另一篇工作的定位与规模），
> 这往往比第三方摘要站可靠得多（第三方站本轮实测会出提示注入，见 K12）。
>
> ⚠️ **但注意**：**这只解决「取到全文」，不解决「那篇闭取论文本身的细节」**。
> F11（ACM MM 2024）**仍然 closed**，F11a 仍需订阅。

### 一条方法论提醒

「从语料全文的参考文献表反查 DOI」这招**对 ISB Part II 有效，对 MANO 无效**
（41 份全文里 MANO 引文**零命中**）。**不能默认某一招通用**，每个条目要单独验证这条路走不走得通。

### 仍然可行的 V3 取件途径

- SIGGRAPH Asia 2018 proceedings 官方页面（ACM DL）
- 作者主页 / `manopth.com`（此前实测不可达，可再试）


---

## 附五：🟢🟢 **附四之三：新发现第二条合法免费 OA 通路 —— Europe PMC 补充材料接口**（2026-10-02）

> 这条是**一次误判逼出来的**，误判过程本身比通路更值得记。

### 通路

```bash
curl -sS -L -o epmc_supp.zip \
  "https://www.ebi.ac.uk/europepmc/webservices/rest/<PMCID>/supplementaryFiles"
# → HTTP 200 / application-zip    无反爬、无需订阅、无 cookie
```

工具：`fetch_epmc_supp.py`（本轮新建）。落盘与说明见 `../补充材料/README.md`。

**同期另一条官方通道，结论不同 —— 两条都要记**：

```bash
curl -sS "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=<PMCID>"
# 实测 13/13 全部 HTTP 404（**不是反爬**）
```

| 通道 | 语料三 13 个 PMCID 实测 |
|---|---|
| Europe PMC `supplementaryFiles` | **13/13 返回 200 + zip** |
| NCBI `oa.fcgi` | **13/13 返回 404** |

### 🚨 三个坑，勿重犯

1. **「某条通路走不通」≠「这条路不存在」。**
   V9g 被判「须人工取件」挂了整整一轮，理由是「JATS 正文不含补充材料」+「JMIR 直连反爬」
   —— **两条都是对的，但都不是结论**，真正的接口压根没试。
   → **一个接口失败时，至少再试同一机构的其他公开接口再下结论。**

2. **HTTP 200 ≠ 有补充材料。**
   接口名叫 `supplementaryFiles`，但它**把正文配图和公式图一并打包**，
   对**每一个** PMCID 都返回 200。语料三实测：**12 个 PMCID 全部 200，真补充件只有 5 个。**
   → **必须按文件名分类**（`-s00N.` / `MOESM` / `_ESM.` / `Data_Sheet` / `Table_*.DOCX`
   才是补充件；`-g00N.` / `_FigN_HTML.` / `_Article_[IE]EqN.` 是正文插图）。
   工具的 `classify()` 实现了这个判定，**汇总会同时打印「HTTP 200 数」与「真补充件数」，看后者。**

3. **文件名编号 ≠ 表号编号。**
   Pohl 那篇正文引用的「表 S2/S3」**全在 `s002.pdf` 一份里**，打开表号却从 S1 开始。
   → **取件后必须逐份打开核对表号**，不要按文件名推断。
   工具落盘时自动生成**分好类的 `_清单.txt`**（含每个 PDF 的首行文字），
   **让人不逐个打开也能确认拿到的是不是要的材料**。

### 纪律：补充材料**不进语料三**

补充材料是**已入库论文的附件**、不是独立论文条目。塞进去会同时踩：
**K8**（手工造定位卡）、**自检 Case 7**（每个 PDF 须有 `_Abstract_Card.md` 孪生）、
**自检 Case 6.5**（`bm_pdf` 计数块与磁盘复算对账）。
→ **原始件存 `../补充材料/`，知识另行提炼进 `00_研究背景/`，语料三条目口径一个字不动。**

---

## 附六：`research-tool` 作为**召回层**（2026-10-02 实测并整合）

> 📌 **研究者裁定**（台账 L5）：把本地 `research-tool` 接进检索管线，
> **定位是「召回层」，不是可复现链的替代品。**
>
> | | 职责 | 工具 |
> |---|---|---|
> | **召回层** | **广度** —— 一次扫多个引擎，找候选 | `research-tool` |
> | **可复现链** | **窄而可核** —— DOI 机器反查、逐条裁定、落盘可复查 | Crossref / Unpaywall / Europe PMC / arXiv 直查（本库既有工具）|
>
> 🚨 **为什么不能互相替代**：
> research-tool 的召回结果**不能直接进 `index.json`** —— 本库的五组口径
> （理论搜索空间 / 实际命中 / 人工裁定 / 元数据证实 / 实际入库）
> 要求每条都经**DOI 机器反查 + 人工裁定**；
> 而 `research-tool` 的 `--llm-expand` / 规则分类结果**按 K10 只能用于召回，不得用于结论**。
> **正确用法：它给你候选标题与线索，可复现链给你可引用的标识。**

### 能力（实测，非读文档推断）

- 仓库 `/Users/xbpd/Projects/research-tool`，CLI 入口 `.venv/bin/research`，Python 包 `research_tool/`
- 命令：`collect` / `ingest-pdf` / `clean` / `extract` / `organize` / `report` / `run` / `wiki-stage` / `publish-wiki` / `status` / `ui` / `config` / `setup`
- **17 个引擎**：`web` `opencli` `arxiv` `tavily` `scholar` `semantic_scholar` `wikipedia` `github` `pubmed` `google_news` `openalex` `crossref` `cvpr` `bilibili` `youtube` `x` `twitter`
- `collect` 支持 `--max-results` / `--depth` / `--rounds(1-3)` / `--llm-expand` / `--from-year` / `--to-year` / `--deep-search` / `--dry-run`

### 🚨 两条实测限制，勿重犯（本轮踩出来的）

1. **多引擎一次可以跑 —— 写法是重复 `-s`，不是逗号串。**
   - ✅ 正确：`-s arxiv -s crossref`（**实测有效**：一次返回 arxiv 3 条 + crossref 2 条，dry-run）
   - ❌ 错误：`-s arxiv,crossref` → **pydantic literal 校验失败**（不是「静默只取第一个」）
   - 走 Python SDK 亦可：`ResearchPipeline(search_engines=["tavily", "openalex"])`
   🚨 **⚠️ 本条曾被写错过一次**（初版记「一次只能跑一个引擎」）。
   **是工具自己的 README 第 263/274/284 行一直写着 `-s openalex -s arxiv` 才把它挡下来。**
   → 教训已登记为 **K18**：**「某工具不支持 X」是比「支持 X」更难证明的断言，落盘前必须先跑反例。**

2. **`openalex` 引擎裸跑会返回 `HTTP 429 Too Many Requests`，召回 0 条。**
   实测：`per-page=3` 即被限流，日志里是 `[WARNING] ... 429` + `共 0 条结果`。
   🚨 **按 K15 / K11：0 召回 ≠ 0 命中。**
   **限流是「没取到」，不是「没有」** —— 不得据此登记「该引擎无结果」。
   → 处置：配 OpenAlex 凭证（polite pool / API key）后重试，或改走 Crossref / Europe PMC 直查。

### 建议用法（本库语境）

```bash
cd /Users/xbpd/Projects/research-tool

# ① 广度召回：多引擎可一次跑（重复 -s，不要用逗号串）；按引擎分次记账
.venv/bin/research collect "<英文查询式>" -s crossref -s arxiv -n 20 --dry-run \
    -o research-output/<语料>/<日期>_<引擎组>

# 若需按引擎分别记账，就分次跑（-s 引擎名），一次只指定一个引擎

# ② 拿到候选后，逐条交给可复现链做 DOI 机器反查（禁止凭记忆写标识）
#    语料三用的是本库既有工具：Unpaywall / Europe PMC / Crossref 三源

# ③ 召回结果**不直接进 index.json**
```

📌 **查询式一律英文**（本库检索为英文单语，中文与非英语未覆盖 —— 诚实清单 §6 已登记）。
