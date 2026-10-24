# SalesRLAgent：一种用于实时销售转化预测与优化的强化学习方法

Nandakishor M

Deepmost Innovations

nandakishor@deepmostai.com

摘要—当前的销售对话分析与转化预测方法通常依赖于大型语言模型（LLMs）与基础检索增强生成（RAG）的结合。这些系统虽然能够回答问题，但无法在实时场景中准确预测转化概率或提供战略指导。在本文中，我们提出了 SalesRLAgent，这是一种利用专用强化学习在整个销售对话过程中预测转化概率的新颖框架。与 Kapa.ai、Mendable、Inkeep 等主要使用现成 LLM 进行内容生成的系统不同，我们的方法将转化预测视为一个序列决策问题，通过在利用 GPT-4O 生成的合成数据上进行训练，开发出专用的概率估计模型。我们的系统整合了 Azure OpenAI 嵌入（embeddings）、逐轮状态跟踪以及用于理解自身知识边界的元学习（meta-learning）能力。评估表明，SalesRLAgent 在转化预测方面达到了 96.7% 的准确率，比仅使用 LLM 的方法高出 34.7%，同时提供了显著更快的推理速度。此外，与现有销售平台的集成表明，当销售代表利用我们系统的实时指导时，转化率提升了 43.2%。SalesRLAgent 代表了从内容生成到战略销售智能的根本性转变，能够提供即时的转化概率估计，并为销售专业人员提供切实可行的洞察。

索引术语—强化学习，销售转化预测，对话分析，元学习，序列决策，实时指导，嵌入，销售智能

## I. 引言

销售行业已越来越多地采用人工智能工具来提升业绩，目前众多平台都提供了聊天机器人、知识助手和对话分析功能。尽管取得了这些进展，现有的解决方案主要侧重于检索信息或生成回复，而非提供战略销售智能。来自 Kapa.ai、Mendable 和 Inkeep 等提供商的大多数商业系统，其本质上不过是高级的 RAG（检索增强生成）系统，仅将大型语言模型（LLMs）连接到企业知识库，并未真正理解销售动态 [1]。

这些方法在销售场景中面临若干根本性限制：

• 无法在实时场景中准确预测转化概率

• 缺乏对对话动态如何影响成交可能性的逐轮跟踪

• 提供通用的指导，而非把握战略时机的指导

• 针对查询被动响应，而非主动指导销售策略

• 缺乏元学习能力以了解自身的知识边界

在本文中，我提出了 SalesRLAgent，这是一个新颖的框架，将销售 AI 重新构想为一个专注于转化预测与优化的专用强化学习系统。SalesRLAgent 没有将销售对话视为简单信息检索问题，而是将其建模为序贯决策过程，其中每次交互都会影响转化概率。

本工作的主要贡献包括：

• 一种专门为销售对话分析和转化预测设计的强化学习架构

• 一个利用 GPT-4O 创建多样化且逼真的销售对话的合成数据生成流水线

• 结合销售特定特征并使用 Azure OpenAI 嵌入（3072 维）的新型状态表征技术

• 一种元学习方法，使系统能够根据对话与训练数据的相似度表达对其预测的置信度

• 在现有销售平台内提供实时指导的集成机制

• 广泛的对比评估，证明了相比于基于 LLM 的方法具有显著的性能提升

我们的研究结果表明，在销售转化任务中，专用强化学习模型显著优于最先进的基于 LLM 的系统；SalesRLAgent 实现了 96.7% 的预测准确率，而最先进的 LLM 解决方案仅为 62%。更重要的是，在实际销售环境中部署时，该系统使实际转化率提升了 43.2%。

## II. 相关工作

## A. 基于 LLM 的销售助手

若干商业平台已经开发了基于 LLM 的销售助手，包括 Kapa.ai [2]、Mendable [3] 和 Inkeep [4]。这些系统主要利用来自 OpenAI、Anthropic 和 Google 等提供商的 API 将其模型与公司知识库相连接。尽管对于问答任务有效，但它们的架构从根本上限制了其对销售对话复杂、序贯特性进行建模的能力 [5]。

Gong 的收入智能平台 [6] 提供了对话分析功能，但主要依赖模式识别而非预测建模。类似地，Chorus.ai [7] 提供会后分析，但缺乏实时能力。

## B. 对话式 AI 中的强化学习

强化学习（Reinforcement Learning, RL）在对话任务中展现出了良好的前景。Li 等人 [8] 的工作展示了 RL 如何优化对话策略，而 Asghar 等人 [9] 则探索了利用 RL 进行对话生成。然而，这些应用主要关注通用对话质量，而非销售转化的特定动态。

在销售领域，Takanobu 等人 [10] 提出了一个用于任务导向场景中对话策略的 RL 框架，但并未涉及转化预测或实时指导。Henderson 等人 [16] 强调了在强化学习应用中采用恰当实验方法学的重要性，这指导了我们严谨的评估方法。

## C. 转化预测模型

正如 Sakar et al. [11] 所综述的，传统的转化预测模型通常采用基于从客户行为数据中提取特征的分类方法。与我们的工作最接近的是 Yang et al. [14]，他们应用深度学习来优化电子商务中的转化漏斗。然而，他们的方法侧重于网络导航，而非复杂的对话动态，并且缺乏我们为不确定性估计所开发的元学习能力。

据我所知，现有系统均未成功将强化学习、Azure OpenAI 嵌入（embeddings）以及元学习相结合用于实时销售转化预测与指导，这使得 SalesRLAgent 对学术研究和实际销售技术均作出了新颖的贡献。

## III. 数据生成与处理

## A. 数据集构建

创建高质量的销售对话数据集面临着巨大挑战。与学术对话数据集不同，真实的销售对话包含敏感信息，极少公开可用。为了解决这一挑战，我开发了一种合成数据生成方法：

• 使用 GPT-4O 的高级提示工程进行合成数据生成

• 针对跨 15 个行业的不同销售场景精心设计模板

• 对对话参数（长度、风格、异议类型）进行程序化变异

• 使用多个大语言模型（LLM）智能体对对话进行受控模拟

我们的最终数据集包含超过 120 万场合成对话。我发现 GPT-4O 的输出质量足以训练出有效的模型，而无需实际的销售数据。每场对话均包括：

• 完整的对话实录

• 说话人信息（客户对比销售代表）

• 转化结果（二元）

• 每轮带有时间戳的转化概率

• 模拟的客户参与度指标

• 产品/服务类别和行业

对话轮数从 3 轮到 27 轮不等，中位数为 8 轮。数据集在正向（已转化）和负向（未转化）结果之间的分布大致均衡，略微偏向负向结果（56%），这反映了现实世界的销售动态。

## B. 数据处理与嵌入

原始对话数据需要进行大量的预处理，以便为我们的强化学习方法构建有用的训练样本。我们的流程包括：

1) 文本清洗与归一化

2) 生成对话中的实体标准化

3) 对话分轮切分

4) 每个对话轮次的特征提取

5) 使用 Azure OpenAI 的嵌入模型生成嵌入

对于嵌入（embeddings），我们使用了 Azure OpenAI 的嵌入模型，该模型提供 3072 维嵌入，能够有效捕获语义关系。坦白地说，我最初打算构建一个自定义嵌入模型，但在对 Azure OpenAI 模型进行实验后，我发现其性能在我们的用例中出奇地好，因此我们沿用了该模型。

我们同时处理了：

• 语义内容（说了什么）

• 对话动态（如何说的）

该方法使用了未经定制的标准 3072 维嵌入，但在其之上叠加了特定领域的特征工程。

## C. 状态表征设计

我们方法中的一项关键创新是用于强化学习（Reinforcement Learning, RL）的状态表征。我们没有将每个对话轮次独立对待，而是设计了一种能够捕获对话演变动态的状态表征：

1) 对话历史嵌入（按时效性和重要性加权）

2) 轮次特定特征（发言时长、提问密度、情感）

3) 客户互动信号（响应时间、消息长度、提问行为）

4) 销售技巧识别（SPIN 销售法、价值销售等）

5) 异议与兴趣检测

完整的状态向量将这些元素融合成一个综合表征，使我们的 RL 智能体能够将销售过程精准地建模为一个序贯决策问题。

## IV. 强化学习架构

## A. 建模方法

我们将销售转化预测任务形式化为一个序贯决策问题，其中：

• 状态代表每一轮的对话

• 动作对应转化概率估计

• 奖励衡量这些预测的准确性

这种形式化使我们能够利用强化学习来训练模型，不仅能预测最终转化结果，还能在整个对话过程中跟踪转化概率。

我个人发现，与传统的分类模型相比，这种序贯方法能够更有效地捕获销售动态。销售对话往往在特定的交流节点发生转折，而我们的模型学会了识别这些关键时刻。

## B. 模型架构

SalesRLAgent 的核心是一个强化学习架构，包含：

• 一个处理 Azure OpenAI 嵌入和特征的状态编码器网络

• 一个根据当前状态估计转化概率的策略网络

• 一个估计期望累积奖励的价值网络

• 一个评估预测置信度的元学习模块

我们尝试了几种 RL 算法，最终发现具有以下特性的专用强化学习算法表现最佳：

1) 强策略正则化以防止过拟合

2) 保守的策略更新以保持稳定性

3) 基于预测难度的自适应学习率

4) 处理不确定性的分布感知学习

该模型架构包含若干专用组件：

## C. 用于不确定性估计的元学习

SalesRLAgent 的一项关键创新在于其元学习（Meta-Learning）能力 [15]。与典型的黑盒模型不同，我们的系统能够基于以下维度评估自身预测的置信度：

• 与训练数据中对话的相似度

• 模型集成之间预测的一致性

• 对话结构的模式识别

• 对训练中未出现的新颖元素的识别

这一元学习组件使 SalesRLAgent 具备了“自知无知”的能力——这对生产系统而言是一项至关重要的特性。当系统遇到包含陌生模式的对话时，它可以明确传达出较低的置信度，而不是做出可能产生误导的预测。

```txt
Algorithm 1 Conversion Probability Estimation
1: function ESTIMATECONVERSION(conversation, turn)
2: history ← ExtractHistory(conversation, turn)
3: embeddings ← GenerateAzureEmbeddings(history)
4: features ← ExtractFeatures(history)
5: state ← CombineState(embeddings, features)
6: policy ← PolicyNetwork(state)
7: value ← ValueNetwork(state)
8: confidence ← ConfidenceEstimation(state, policy)
9: return {probability : policy, confidence : confidence}
10: end function
```

我度过了无数个深夜来微调这种元学习方法，这一动力源于早期部署中的发现：模型在不寻常的对话模式上有时会做出自信却错误的预测。事实证明，由此产生的不确定性估计在生产系统中具有无可替代的价值。

## D. 训练方法

训练 SalesRLAgent 需要精心设计的方法，以应对合成对话数据多样化且有时伴有噪声的特性。我们的训练流程包括：

1) 初始监督学习阶段，以建立合理的策略初始化

2) 基于精心构建的奖励函数的强化学习

3) 课程学习（Curriculum Learning），从较为简单的对话开始

4) 引入具有挑战性的反例进行对抗训练

5) 采用集成技术以提升鲁棒性

6) 专用的批次构建，以平衡对话类型

我们在标准的 CPU 基础设施上进行了训练，坦白说，其性能表现之佳令我十分惊喜。整个训练过程大约耗时 6 小时，考虑到模型的复杂度和数据集的规模，这一速度远快于我最初的预期。

## V. 系统集成与部署

## A. 实时推理流水线

将训练好的模型转化为生产系统需要精细的工程设计。完整的推理流水线包括：

1) 对话捕获与预处理

2) 使用 Azure OpenAI 生成增量嵌入

3) 状态跟踪与更新

4) 概率预测与置信度估计

5) 策略指导生成

6) CRM 与通信工具集成

为了满足严苛的延迟要求（100ms 以内），我们实施了多项优化措施：

• 模型量化（精度损失极小的 8-bit 精度）

• 针对重复对话元素的嵌入缓存

• 增量状态更新而非全量重新计算

• 针对多用户环境的 CPU 优化

• 异步指导生成

## B. 高级编排层

我们系统的一个关键组件是先进的编排层，用于处理多节点上下文检索和处理。这种有向图架构通过专用处理节点路由对话，以实现上下文相关的信息检索。

我们的实现集成了：

• 用于相似度匹配的高性能向量搜索

• 用于优化检索的缓存层

• 针对复杂对话模式的基于状态的工作流编排

• 基于转化概率的动态提示工程（Dynamic prompt engineering）

该编排架构使系统能够就检索何种信息以及如何处理信息做出精密决策，显著优于竞争系统所采用的简单 RAG 方法。

## C. CRM 与通信平台集成

SalesRLAgent 旨在与现有销售工具集成，而无需全面更换基础设施。我们开发了适用于以下平台的连接器：

• 主流 CRM 平台（Salesforce、HubSpot 等）

• 通信工具（Zoom、Teams、Google Meet）

• 电子邮件平台（Outlook、Gmail）

• 聊天系统（Slack、Intercom、Drift）

集成选项包括：

• 面向销售代表的实时悬浮界面（Real-time overlay）

• 通话后总结与分析

• 用于团队绩效的管理仪表板

• 用于自定义集成的 API 访问

## VI. 结果与评估

## A. 转化预测准确率

我们将 SalesRLAgent 与几种基准方法进行了评估对比：

• 传统机器学习分类器（Random Forest、XGBoost）

• 基于 LLM 的预测（采用专用提示工程的 GPT-4）

• 商业系统（因许可限制已匿名化）

• 结合 LLM 与传统机器学习的混合方法

表 I 展示了这些方法的预测准确率：

SalesRLAgent 达到了 96.7% 的准确率，比最佳商业替代方案高出 23.7 个百分点，比最佳 LLM 方法高出 34.7 个百分点。

TABLE I  
转化预测准确率

<table><tr><td>方法</td><td>准确率</td><td>AUC-ROC</td></tr><tr><td>Random Forest</td><td>0.67</td><td>0.71</td></tr><tr><td>XGBoost</td><td>0.69</td><td>0.74</td></tr><tr><td>GPT-4 (zero-shot)</td><td>0.59</td><td>0.63</td></tr><tr><td>GPT-4 (few-shot)</td><td>0.62</td><td>0.68</td></tr><tr><td>商业系统 A</td><td>0.71</td><td>0.76</td></tr><tr><td>商业系统 B</td><td>0.73</td><td>0.78</td></tr><tr><td>混合 LLM+ML</td><td>0.75</td><td>0.81</td></tr><tr><td>SalesRLAgent (base)</td><td>0.92</td><td>0.94</td></tr><tr><td>SalesRLAgent (full)</td><td>0.967</td><td>0.98</td></tr></table>

TABLE II  
推理性能对比

<table><tr><td>方法</td><td>延迟 (ms)</td><td>吞吐量 (req/s)</td></tr><tr><td>GPT-4 (API)</td><td>3450</td><td>0.3</td></tr><tr><td>GPT-3.5 (API)</td><td>980</td><td>1.0</td></tr><tr><td>Claude (API)</td><td>2750</td><td>0.4</td></tr><tr><td>Local LLM</td><td>1650</td><td>0.6</td></tr><tr><td>SalesRLAgent (CPU)</td><td>85</td><td>12</td></tr></table>

更重要的是，SalesRLAgent 在整个对话过程中追踪转化概率方面展现出了卓越的性能，而不仅仅是预测最终结果。

## B. 对销售业绩的实际影响

除了技术指标外，我们还通过 A/B 测试在真实的销售环境中对 SalesRLAgent 进行了评估。销售代表被随机分配使用以下方案之一：

• 仅使用传统销售工具（对照组）

• 传统工具 + SalesRLAgent 指导（测试组）

在对 217 名代表和 12,433 场对话进行了为期 90 天的跟踪后，我们观察到：

• 测试组的转化率提升了 43.2%

• 销售周期长度缩短了 22%

• 平均交易规模提升了 14%

• 客户满意度评分提高了 9%

来自销售代表的定性反馈强调了几个关键优势：

1) 及早识别有潜力的线索，从而实现更好的时间分配

2) 实时感知对话转折点

3) 针对处理异议的具体指导

4) 增强对销售预测的信心

5) 使销售代表对对话质量的认知与实际情况更好地保持一致

## C. 推理性能

对于实用的销售工具而言，推理速度至关重要。我们将 SalesRLAgent 与基于 LLM 的方法进行了基准测试：

SalesRLAgent 实现了显著更快的推理时间——仅需 85ms，而 LLM 方法则需要数秒——这使其非常适合在不干扰对话流程的情况下提供实时指导。我们目前仅提供 CPU 部署选项，事实证明这已完全能够满足我们的吞吐量需求。

TABLE III  
消融实验结果（准确率）

<table><tr><td>配置</td><td>准确率</td></tr><tr><td>完整 SalesRLAgent</td><td>0.967</td></tr><tr><td>- Azure OpenAI embeddings</td><td>0.89 (-0.077)</td></tr><tr><td>- 序列建模 (Sequential modeling)</td><td>0.86 (-0.107)</td></tr><tr><td>- 元学习 (Meta-learning)</td><td>0.94 (-0.027)</td></tr><tr><td>- 高级编排 (Advanced orchestration)</td><td>0.92 (-0.047)</td></tr><tr><td>- 缓存优化 (Caching optimization)</td><td>0.95 (-0.017)</td></tr><tr><td>- 所有专用模块 (基础机器学习)</td><td>0.68 (-0.287)</td></tr></table>

## D. 消融实验

为了理解各组件的贡献，我们进行了消融实验，移除了 SalesRLAgent 的关键要素：

这些结果突出了序列建模和 Azure OpenAI embeddings 的显著贡献，它们共同构成了 SalesRLAgent 相比传统方法的大部分性能优势。高级编排和缓存优化也对系统的整体性能做出了实质性贡献。

## VII. 讨论与见解

## A. 与现有解决方案的对比分析

通过大量测试，我发现了我们的强化学习方法相比于来自 Kapa.ai、Mendable 和 Inkeep 等基于 LLM（大语言模型）的系统所具有的几个关键优势：

1) 专业化 vs. 通用化：虽然 LLM 擅长通用文本生成，但它们缺乏我们的 RL（强化学习）方法所提供的针对销售动态的专门训练。这种专业化带来了显著更高的预测准确率和更相关的指导。

2) 时序性 vs. 静态性：基于 LLM 的系统通常独立处理每一次交互，忽略了销售对话中至关重要的时序动态。我们的 RL 框架显式地建模了对话随时间的演变过程。

3) 计算高效 vs. 资源密集：LLM 推理需要大量的计算资源并会产生高延迟。SalesRLAgent 可以在毫秒级而非秒级内给出预测，从而实现真正的实时指导。

4) 定量 vs. 定性：LLM 系统通常提供定性指导，缺乏具体的概率估计。SalesRLAgent 提供带有置信区间的精确转化概率，从而支持更由数据驱动的销售策略。

5) 元学习 vs. 黑盒：我们系统的元学习能力使其能够在适当时表达不确定性，而不像 LLM 系统那样经常将推测当作事实呈现。

这些差异反映了销售 AI 的根本不同方法。基于 LLM 的系统主要充当信息检索和文本生成工具，而 SalesRLAgent 的运作方式更像是一个销售领域的“国际象棋引擎”——分析对话局势、评估转化概率并推荐最佳行动。

## B. 局限性与未来工作

尽管表现强劲，SalesRLAgent 仍存在一些局限性，这也代表了未来工作的研究方向：

• 多语言支持：当前模型主要支持英语，在其他语言上的能力有限。扩展到真正的多语言模型在数据生成和建模方面都带来了独特的挑战。

• 多模态分析：该系统目前仅分析文本，忽略了视频通话中来自语音语调、面部表情或手势的重要信号。纳入这些信号是一个很有前景的改进方向。

• 个性化：虽然该模型能够适应对话动态，但尚未针对各个销售代表的风格和优势进行个性化定制。增加这一层个性化可能会进一步提升有效性。

• 因果性 vs. 相关性：当前模型能够识别对话模式与结果之间的相关性，但确定因果关系的能力有限。加强因果推理代表了一个重要的前沿方向。

• 长期关系建模（Long-term Relationship Modeling）：系统目前侧重于单次对话，而非对跨多次交互的客户关系进行建模。将其拓展至关系层面的预测带来了颇具价值的挑战。

我对多模态分析的潜力尤为感到期待，因为我的初步实验表明，引入音频特征可以在我们当前 96.7% 的基准之上将准确率再提升 2-3%。

## VIII. 结论

本文提出了 SalesRLAgent，这是一种用于销售转化预测与优化的新型强化学习方法。与依赖通用文本生成能力的现有基于 LLM（大语言模型）的系统不同，SalesRLAgent 将销售对话视为序贯决策过程，从而能够实现精准的转化概率跟踪和策略指导。

我们方法的主要创新点包括：

• 专为销售转化动态设计的强化学习框架

• 利用 GPT-4O 进行合成数据生成，以构建多样化的训练场景

• 用于对话表示的 Azure OpenAI 3072 维嵌入

• 用于复杂对话流程管理的高级编排机制

• 带有缓存的优化向量相似度搜索，实现毫秒级检索

• 用于置信度估计和不确定性量化的元学习能力

• 与现有销售工具及平台的实时集成

我们的评估表明，SalesRLAgent 在转化预测方面达到了 96.7% 的准确率，显著优于传统机器学习方法和基于 LLM 的替代方案。更重要的是，在实际销售环境中部署时，该系统带来了 43.2% 的转化率提升以及 22% 的销售周期缩短。

这些结果表明，在销售等特定领域应用中，专用强化学习方法相比通用 LLM 解决方案具有显著优势。尽管大语言模型在通用文本生成方面表现出色，但销售对话的复杂、序贯特性更能从强化学习所提供的针对性优化中获益。

展望未来，我相信这项工作为销售领域的 AI 开辟了新的可能性——超越简单的自动化和信息检索，迈向增强人类能力而非取代人类的战略合作伙伴关系。销售 AI 的未来不在于规模越来越大的语言模型，而在于能够深刻理解有效销售动态的、日益专业化的智能体系。

## 致谢

我对 Deepmost Innovations 团队表示由衷的感谢，他们忍受了无数次模型设计的迭代，以及我在白板前没完没了讨论强化学习架构的时光。特别感谢匿名审稿人，他们建设性的反馈显著提升了本文的质量，当然，文中所残留的任何错误或疏漏完全由我个人承担。我还要感谢 OpenAI 提供的 GPT-4O 模型，使得我们的合成数据生成成为可能；以及 Microsoft 提供的 Azure OpenAI 嵌入模型，它为我们系统的对话理解奠定了基础。最后，我必须感谢那数千杯支撑着深夜调试的咖啡，以及耐心支持这项工作的家人——即便在餐桌旁，他们并不太能理解我关于转化概率分布的兴奋高谈阔论。

## 参考文献

[1] L. Gao, X. Ma, J. Lin, and J. Callan, “Precise zero-shot dense retrieval without relevance labels,” in Proc. 46th Int. ACM SIGIR Conf. Res. Develop. Inf. Retr., 2023, pp. 2196–2206.

[2] Kapa.ai, “Enterprise knowledge assistant platform,” Kapa.ai, Tech. Rep., 2023. [Online]. Available: https://kapa.ai

[3] Mendable, “AI search for technical documentation,” Mendable, Tech. Rep., 2023. [Online]. Available: https://mendable.ai

[4] Inkeep, “Conversational AI for customer support,” Inkeep, Tech. Rep., 2023. [Online]. Available: https://inkeep.com

[5] T. Zhang et al., “Benchmarking large language models for news summarization,” arXiv, 2023, arXiv:2301.13848.

[6] Gong, “Revenue intelligence platform methodology,” Gong.io, Tech. Rep., 2022. [Online]. Available: https://gong.io

[7] Chorus.ai, “Conversation intelligence for sales teams,” Chorus.ai, Tech. Rep., 2021. [Online]. Available: https://chorus.ai

[8] Z. Li, J. Kiseleva, and M. de Rijke, “Guided dialog policy learning: Reward estimation for multi-domain task-oriented dialog,” in Proc. 2020 Conf. Empir. Methods Natural Lang. Process., 2020, pp. 100–120.

[9] N. Asghar et al., “Affective neural response generation,” in Proc. Eur. Conf. Inf. Retr., 2017, pp. 154–166.

[10] R. Takanobu, H. Zhu, and M. Huang, “Guided dialog policy learning: Reward estimation for multi-domain task-oriented dialog,” in Proc. 2019 Conf. Empir. Methods Natural Lang. Process., 2019, pp. 2308–2319.

[11] C. O. Sakar et al., “Real-time prediction of online shoppers’ purchasing intention using multilayer perceptron and LSTM recurrent neural networks,” Neural Comput. Appl., vol. 31, no. 10, pp. 6893–6908, 2019.

[12] J. Devlin et al., “BERT: Pre-training of deep bidirectional transformers for language understanding,” in Proc. NAACL-HLT, 2019, pp. 4171– 4186.

[13] N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings using Siamese BERT-networks,” in Proc. 2019 Conf. Empir. Methods Natural Lang. Process., 2019, pp. 3980–3990.

[14] Y. Yang, Y. Li, and H. Wang, “基于用户行为在线转化预测的序列深度学习,” Inf. Process. Manage., vol. 59, no. 1, p. 102751, 2022.

[15] C. Finn, P. Abbeel, and S. Levine, “用于深度网络快速适应的模型无关元学习,” in Proc. 34th Int. Conf. Mach. Learn., 2017, pp. 1126–1135.

[16] P. Henderson et al., “切实有效的深度强化学习,” in Proc. AAAI Conf. Artif. Intell., vol. 32, no. 1, 2018.