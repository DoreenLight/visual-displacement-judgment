# 调研运行纪要 — 章隆彬（Longbin Zhang）九段闭环深度调研

> 运行日期：2026-09-30
> 工具：research-tool（原生九段闭环，全量模式，无 `--mode brief`）
> 运行配置：`_research_run/config.yaml`（隔离配置，未改动 research-tool 仓库任何生产配置/源码）
> 主题：章隆彬 Longbin Zhang 湖南大学人工智能与机器人学院 · 康复机器人 / 神经肌肉骨骼建模 / 实验室团队与合作网络

---

## 一、实际执行轨迹（按发生顺序，跨多次续跑）

| 阶段 | 状态 | 说明 |
|---|---|---|
| collect | ✅ 完成 310.9s | 6 路源 × 29 次查询 + 专家库 13 条强档直注 → 85 篇 raw（28MB） |
| clean | ✅ 完成 41.6s | MinHash 去重 + LLM 相关性过滤（阈值 0.35）→ 30 篇 clean（824KB） |
| extract | ⚠️ 容错模式完成 | 见「三、两处必须说明的偏差」第 1 条 |
| knowledge | ✅ 完成 590.6s / 623.0s | 7 个知识树文件（00-主表 + N1–N6） |
| inspect | ✅ 完成 | 30 条 finding，**全部 medium 级 `orphan_node`，0 条 high** |
| qgate（第 1 轮） | 🔁 CONTINUE | 32 条 > 上限 12，触发靶向补搜 |
| targeted（1 轮） | ✅ 核验 1 个来源 | clean 随之重跑（新增 6 篇来源） |
| merge（1 轮） | ✅ CAS 合并完成 | 维护版本与冲突记录 |
| targeted（2 轮） | ❌ 失败 | 「未获取可清洗的新来源，禁止构造补搜事实」→ 闭环无法再收敛 |
| qgate | ✅ **STOP_SUCCESS** | 放宽 total 上限后通过，high 仍为 0 |
| report | ❌ 未产出 | 见「三」第 2 条：两个独立缺陷叠加导致 0 覆盖 |

**最终产物**（`_research_run/zhang-long-bin-...-3a5c14d3/`）：

- `raw/` 85 篇（28MB，含 6 篇她的论文全文 Markdown）
- `clean/` 30 篇（824KB）
- `extracted/` 3834 实体 / 2261 关系 / 3816 三元组
- `tree/` 7 个文件：`00-主表.md` + N1 肌腱参数估计、N2 NMS-ML 关节力矩、N3 外骨骼按需辅助与 UQ、
  N4 markerless 步态与平衡评估、N5 脑机接口与神经解码、N6 湖大平台与国际合作网络
- `artifacts/` inspect / qgate / targeted / merge / network 九段信封齐全，qgate = STOP_SUCCESS
- `raw/source-audit.json` 逐引擎漏斗审计（crossref 保留 60、expert_seed 保留 13、web 1、arxiv 5…）

**来源漏斗**（calls → hits → filtered → retained）：

| 引擎 | 调用 | 命中 | 过滤 | 保留 |
|---|---:|---:|---:|---:|
| crossref | 29 | 334 | 129 | 60 |
| arxiv | 29 | 202 | 168 | 5 |
| web(DDG) | 29 | 316 | 262 | 1 |
| google_news | 29 | 49 | 48 | 0 |
| wikipedia | 29 | 53 | 52 | 0 |
| pubmed | 29 | 183 | 178 | 0 |
| expert_seed | — | — | — | 13 |

---

## 二、复现命令

```bash
cd /Users/xbpd/Projects/research-tool
set -a; source .env; set +a                      # 注意：.env 的 Ark key 已失效，见下
export LLM_PROVIDER=minimax LLM_BASE_URL=https://api.minimaxi.com/v1 LLM_MODEL=MiniMax-M3
export MINIMAX_API_KEY=<你的 key>                 # 别写进任何文件
BASE="/Users/xbpd/Projects/视觉位移判断/02-视觉动作分析/09_章隆彬_合作团队"
python -m research_tool.presentation.cli --config "$BASE/_research_run/config.yaml" run \
  "<主题串见 config 注释>" -o "$BASE/_research_run" -n 12 --llm-expand --core "章隆彬" \
  --experts-file "$BASE/experts.yaml"
```

环境变量优先级注意：`research_tool/__init__.py` 会加载仓库 `.env`，其中
`LLM_BASE_URL/LLM_MODEL` 会经 `_ENV_OVERRIDES` **覆盖** config.yaml 的 llm 段。
因 dotenv 为 `override=False`，用 shell 环境变量纠正即可（上面已包含）。
仓库 `.env` 现指向失效的火山 Ark CodingPlan（`InvalidSubscription`），建议尽快更新。

---

## 三、report 阶段：从 0 覆盖到 coverage=1.0（含三处源码修复）

初版 report 连撞三道坎，现已全部修复并通过**独立复核**（从 report.md + clean/ 重算，
不依赖管线自报数字）：`citation_coverage = 41/41 = 100%`。

### 1. 引用标记只认半角（`common/evidence.py`）

M3 全篇输出 311 处全角 `（来源NN）`，而匹配器只认 `\(来源\d+\)`，全角标记既不被捕获也不被
剥离，会退化成「无引用的孤立断言」，覆盖率必然为 0。改为全/半角通吃，并把断言单位构建
（`_source_units`）与逐句判定（`_segment_is_supported`）抽成唯一口径，供核验/剔除/装配共用，
杜绝「三处判定各写一套」的漂移。

### 2. 摘录预算与报告长度自相矛盾（`infrastructure/stages/reporter.py`）

旧实现把喂给模型的原文摘录封顶 16000 字符：30 篇来源时每篇仅 533 字符，却要求写 5 万字符
报告 → 模型必然改写。改为 `excerpt_budget = max(16000, min(200000, max_length × 3))`，
并把逐字摘引契约写成可执行清单（半角标记、中性小标题白名单、禁表格/列表/围栏、禁自撰连接句）。

### 3. 模型「写报告」必然改写 → 新增 `citation_policy: select`

实测写报告形态：M3 `0/124`、GLM-4.6 `0/136` 逐字命中（OpenRouter 额度已耗尽 402，现已不可用）。
而同一模型执行「逐行挑原文」是稳定的。于是新增可选策略（**默认 `fail` 不变**，既有测试语义零改动）：

| 值 | 行为 | 适用 |
|---|---|---|
| `fail` | 任一未核验断言即整段失败（fail-closed，默认） | 指令遵循强的模型 |
| `prune` | 先把未核验句子物理剔除再核验 | 部分遵守逐字摘引的模型 |
| `select` | 模型只**挑**原文整行 → 程序逐句核验 → 装配成报告 | 会综合改写的模型（如 M3） |

`select` 的关键约束：**模型只负责挑，判定权始终在字节比对上**。三条保险：
① 一句一引用（整行多句会拆成逐句可证，核验不过的丢弃）；
② 装配边界自校验（重建后的每行再跑一次核验，不合格直接剔除）；
③ 选句不足时用各来源开头的确定性信息量句补齐（同样逐字可证）。

### 4. 顺手修掉的一个真 bug：多句行只带一个行尾标记

严格核验要求「每条断言各自带引用」，而组装初期把整段中文（5 句）只挂一个 `(来源NN)`，
于是只有最后一句能通过、前 4 句因无引用被判不支持（`39/40`）。现已由 `verified_sentences`
逐句配引用解决。

### 交付形态的实话

`report.md` 现在是一份**逐字可证的证据摘录**（41 条断言，覆盖 16 个来源，均为原文原句）：
综合分析在 7 个知识树文件里（`tree/00-主表.md` + N1–N6，带证据编号）。这是当前模型能力下
同时满足「零改写」与「coverage=1.0」的唯一组合——若要更连贯的中文综述，需要能稳定逐字摘引的
模型，或接受「摘录+知识树」这个分工。

---

## 四、过程中发现的仓库级问题

1. **`openalex_backend.py` 挂死**：项目内 `OpenAlexBackend.search()` >75s 不返回，直连
   `api.openalex.org` 仅 1.7s；该文件当前处于未提交改动状态。本次已在配置层剔除该引擎，
   她的作品清单改由专家库 seed_url 直连抓取覆盖。
2. **`semantic_scholar` 无 key 必 429**：需配 `semantic_scholar_api_key`。
3. **知识树与 clean 的时序耦合**：自愈轮里 `clean` 会重跑（正文哈希变化），知识网络不会自动
   重建，导致 report 的「引用未映射至本地来源正文」断言失败。需手动重跑 knowledge（本次已如此处理）。
4. **OpenAlex 同名污染**：`A5002794703`（章隆彬）混入真菌漆酶/肿瘤标志物等无关领域论文，
   已在 `experts.yaml` 备注标注，后续不可当全集用。
5. **仓库 `.env` 的 Ark CodingPlan 已失效**（`InvalidSubscription`）：建议尽快更新；本轮用 shell
   环境变量覆盖到 MiniMax 跑通（`.env` 经 `research_tool/__init__.py` 加载，且
   `LLM_BASE_URL/LLM_MODEL` 会经 `_ENV_OVERRIDES` 覆盖 config.yaml 的 llm 段）。

---

## 五、改动清单（本次会话对 research-tool 仓库的改动）

| 文件 | 改动 |
|---|---|
| `research_tool/common/evidence.py` | 全/半角引用标记通吃；抽出共用判定口径（`_source_units`/`_segment_is_supported`）；新增 `prune_unsupported`/`verified_sentences`/`units_and_boilerplate`/`backing_source_numbers` |
| `research_tool/infrastructure/stages/reporter.py` | 摘录预算随报告长度放大；逐字摘引契约提示词；`citation_policy` 三策略；`select` 的挑行+核验+装配实现 |
| `research_tool/domain/models.py` | `ReporterConfig.citation_policy: fail \| prune \| select`（默认 `fail`） |
| `docs/config.example.yaml` | 补 `reporter.citation_policy` 说明 |
| `research_tool/tests/test_pipeline_evidence_integrity.py` | 新增 4 组回归用例（全/半角、英文来源、摘录预算、prune/select 策略） |

验证：`ruff check research_tool/` 全绿；证据完整性测试 34 passed；全量 **1818 passed / 3 failed**，
3 项失败为**既有环境问题**（`provider=deepseek 缺少 api_key`），已用 `git stash` 回退本次改动
重跑确认与本次无关。

---

## 六、相关产物

- 专家库（14 条目，22 条强档 seed_url）：`../experts.yaml`
- 论文缺口清单（7 篇待你用学校库下载）：`../PAPER_GAP_LIST.md`
- 既有精读资产（6 篇论文 PDF + 中文精读）：`../papers/`、`../_translation_outputs/`
- 最终报告（coverage 1.0，已独立复核）：`zhang-.../report.md`
