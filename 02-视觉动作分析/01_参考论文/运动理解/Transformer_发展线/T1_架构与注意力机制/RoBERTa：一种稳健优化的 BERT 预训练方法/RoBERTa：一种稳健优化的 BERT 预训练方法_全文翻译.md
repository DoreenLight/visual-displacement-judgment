# RoBERTa：一种稳健优化的 BERT 预训练方法

Yinhan Liu<sup>∗§</sup> Myle Ott<sup>∗§</sup> Naman Goyal<sup>∗§</sup> Jingfei Du<sup>∗§</sup> Mandar Joshi Danqi Chen<sup>§</sup> Omer Levy<sup>§</sup> Mike Lewis<sup>§</sup> Luke Zettlemoyer<sup>†§</sup> Veselin Stoyanov<sup>§</sup>

<sup>†</sup> Paul G. Allen School of Computer Science & Engineering, University of Washington, Seattle, WA {mandar90,lsz}@cs.washington.edu

<sup>§</sup> Facebook AI

{yinhanliu,myleott,naman,jingfeidu, danqi,omerlevy,mikelewis,lsz,ves}@fb.com

## 摘要

语言模型预训练带来了显著的性能提升，但对不同方法之间进行审慎的比较仍然具有挑战性。训练在计算上非常昂贵，通常在不同规模的私有数据集上进行，并且正如我们将展示的，超参数的选择对最终结果有着显著的影响。我们提出了一项关于 BERT 预训练（Devlin et al., 2019）的复现研究，仔细衡量了许多关键超参数和训练数据规模的影响。我们发现 BERT 明显欠训练（undertrained），并且能够达到或超越其后发布的每个模型的性能。我们的最佳模型在 GLUE、RACE 和 SQuAD 上取得了最先进（state-of-the-art）的结果。这些结果凸显了以往被忽视的设计选择的重要性，并对近期报道的改进来源提出了疑问。我们发布了我们的模型和代码。<sup>1</sup>

## 1 引言

诸如 ELMo（Peters et al., 2018）、GPT（Radford et al., 2018）、BERT（Devlin et al., 2019）、XLM（Lample and Conneau, 2019）和 XLNet（Yang et al., 2019）等自训练（self-training）方法带来了显著的性能提升，但要确定这些方法的哪些方面贡献最大却十分困难。训练在计算上成本高昂，限制了可进行的调优数量，且通常使用规模各异的私有训练数据，这限制了我们衡量建模技术进步所产生效果的能力。

我们提出了一项关于 BERT 预训练（Devlin et al., 2019）的复现研究，其中包括对超参数调优和训练集规模影响的仔细评估。我们发现 BERT 严重欠训练，并提出了一种改进的 BERT 模型训练方案，我们称之为 RoBERTa，它能够达到或超越所有在 BERT 之后提出的方法的性能。我们的改进很简单，包括：(1) 在更多的数据上以更大的批次（batch）训练模型更长时间；(2) 移除下一句预测（next sentence prediction）目标；(3) 在更长的序列上训练；以及 (4) 动态改变应用于训练数据的掩码模式（masking pattern）。我们还收集了一个规模与其他私有数据集相当的大型新数据集（CC-NEWS），以更好地控制训练集大小带来的影响。

在控制训练数据的情况下，我们改进的训练流程在 GLUE 和 SQuAD 上均超越了已发表的 BERT 结果。当在更多数据上进行更长时间的训练时，我们的模型在公开的 GLUE 排行榜上获得了 88.5 分，与 Yang 等人（2019）报道的 88.4 分相当。我们的模型在 9 个 GLUE 任务中的 4 个上创下了新的最先进水平（state-of-the-art）：MNLI、QNLI、RTE 和 STS-B。我们还在 SQuAD 和 RACE 上达到了最先进水平的结果。总体而言，我们重新确立了 BERT 的掩码语言模型训练目标与近期提出的其他训练目标（例如扰动自回归语言建模（Yang et al., 2019））相比具有竞争力。<sup>2</sup>

综上所述，本文的贡献在于：(1) 我们提出了一系列重要的 BERT 设计选择和训练策略，并引入了能带来更优下游任务表现的替代方案；(2) 我们使用了一个新颖的数据集 CC-NEWS，并证实使用更多数据进行预训练能进一步提升下游任务的表现；(3) 我们的训练改进表明，在正确的设计选择下，掩码语言模型预训练与近期发表的所有其他方法相比都具有竞争力。我们开源了基于 PyTorch（Paszke et al., 2017）实现并发布的模型、预训练和微调代码。

## 2 背景

在本节中，我们简要概述 BERT（Devlin et al., 2019）预训练方法以及我们将在下一节中通过实验考察的一些训练选择。

## 2.1 设置

BERT 将两个片段（词元序列）$x _ { 1 } , \ldots , x _ { N }$ 和 $y _ { 1 } , \dots , y _ { M }$ 的拼接作为输入。片段通常由不止一个自然句子组成。这两个片段作为一个单一的输入序列呈现给 BERT，并使用特殊词元进行分隔：$[ C L S ] , x _ { 1 } , \ldots , x _ { N } , [ S E P ] , y _ { 1 } , \ldots , y _ { M } , [ E O S ]$。M 和 N 受到约束以满足 $M + N < T$，其中 T 是控制训练期间最大序列长度的参数。

该模型首先在大型无标注文本语料库上进行预训练，随后使用最终任务的标注数据进行微调。

## 2.2 架构

BERT 采用了如今无处不在的 Transformer 架构（Vaswani et al., 2017），在此我们不再详细回顾。我们使用包含 L 层的 Transformer 架构。每个模块使用 A 个自注意力头和隐藏维度 H。

## 2.3 训练目标

在预训练期间，BERT 使用两个目标：掩码语言建模（masked language modeling）和下一句预测（next sentence prediction）。

掩码语言模型（Masked Language Model, MLM） 随机抽取输入序列中的一部分词元，并将其替换为特殊词元 [MASK]。MLM 目标是在预测被掩码词元时的交叉熵损失。BERT 均匀采样 15% 的输入词元作为可能被替换的候选。在选中的词元中，80% 被替换为 [MASK]，10% 保持不变，10% 被替换为随机选取的词表词元。

在最初的实现中，随机掩码和替换在开始时仅执行一次，并在整个训练期间保存，尽管在实践中，数据会被复制，因此掩码对于每个训练句子并不总是相同的（参见第 4.1 节）。

下一句预测（Next Sentence Prediction, NSP） NSP 是一种二分类损失，用于预测两个片段在原文中是否先后相连。正例是通过从文本语料库中提取连续句子来创建的。负例是通过将来自不同文档的片段进行配对来创建的。正例和负例以相同的概率进行采样。

NSP 目标旨在提高下游任务的性能，例如自然语言推理（Natural Language Inference, Bowman et al., 2015），这类任务需要对句对之间的关系进行推理。

## 2.4 优化

BERT 使用 Adam（Kingma and Ba, 2015）进行优化，采用以下参数：$\beta _ { 1 } = 0 . 9 ,$ $\beta _ { 2 } ~ = ~ 0 . 9 9 9 , ~ \epsilon ~ = ~ 1 \mathrm { e } { - 6 }$ 以及 0.01 的 L<sub>2</sub> 权重衰减。学习率在前 10,000 步内预热（warmup）至峰值 1e-4，然后线性衰减。BERT 在所有层和注意力权重上使用 0.1 的 dropout 进行训练，并采用 GELU 激活函数（Hendrycks and Gimpel, 2016）。模型预训练 S = 1,000,000 次更新，每个小批量包含 B = 256 条最大长度为 $T = 5 1 2$ 个词元（token）的序列。

## 2.5 数据

BERT 是在 BOOKCORPUS（Zhu et al., 2015）加上英文 WIKIPEDIA 的组合数据集上训练的，未压缩文本总量为 16GB。<sup>3</sup>

## 3 实验设置

在本节中，我们介绍了 BERT 复现研究的实验设置。

## 3.1 实现

我们在 FAIRSEQ（Ott et al., 2019）中重新实现了 BERT。我们主要遵循第 2 节中给出的原始 BERT 优化超参数，但峰值学习率和预热步数除外，这两者针对每种设置单独进行了调整。此外，我们发现训练对 Adam 的 epsilon 项非常敏感，在某些情况下，调整该项后我们获得了更好的性能或更高的稳定性。类似地，我们发现在大批量训练时，将 $\beta _ { 2 } = 0 . 9 8$ 可以提高稳定性。

我们使用最多 T = 512 个词元的序列进行预训练。与 Devlin et al. (2019) 不同，我们没有随机注入短序列，也没有在前 90% 的更新中使用缩短的序列长度进行训练。我们仅使用完整长度的序列进行训练。

我们在 DGX-1 机器上使用混合精度浮点运算进行训练，每台机器配备 8 块通过 Infiniband 互连的 32GB Nvidia V100 GPU（Micikevicius et al., 2018）。

## 3.2 数据

BERT 风格的预训练关键依赖于大量的文本。Baevski et al. (2019) 表明，增加数据规模可以带来下游任务性能的提升。已有多项工作在比原始 BERT 更大且更多样化的数据集上进行了训练（Radford et al., 2019；Yang et al., 2019；Zellers et al., 2019）。遗憾的是，并非所有额外的数据集都能公开释出。在我们的研究中，我们专注于收集尽可能多的数据用于实验，以便在每次对比中能够视情况匹配相应的数据整体质量和数量。

我们考虑了五个不同规模和领域的英文语料库，未压缩文本总量超过 160GB。我们使用了以下文本语料库：

• BOOKCORPUS (Zhu et al., 2015) 加上英文 WIKIPEDIA。这是用于训练原始 BERT 的数据。（16GB）。

• CC-NEWS，这是我们从 CommonCrawl News 数据集（Nagel, 2016）的英文部分收集而来的。该数据包含 2016 年 9 月至 2019 年 2 月期间爬取的 6300 万篇英文新闻文章。（过滤后为 76GB）。<sup>4</sup>

• OPENWEBTEXT (Gokaslan and Cohen, 2019)，这是对 Radford et al. (2019) 中描述的 WebText 语料库的开源复刻版本。该文本是从 Reddit 上获得至少三个赞同票（upvotes）的 URL 中提取的网页内容。（38GB）。<sup>5</sup>

• STORIES，这是由 Trinh and Le (2018) 引入的数据集，包含 CommonCrawl 数据的子集，经过过滤以匹配 Winograd schemas 的故事风格。（31GB）。

## 3.3 评估

沿用先前的研究工作，我们在下游任务上使用以下三个基准来评估我们的预训练模型。

GLUE 通用语言理解评估（General Language Understanding Evaluation，GLUE）基准（Wang et al., 2019b）是用于评估自然语言理解系统的 9 个数据集的集合。<sup>6</sup> 任务被构建为单句分类或句对分类任务。GLUE 组织者提供了训练集和开发集的划分，以及一个提交服务器和排行榜，允许参与者在私有的保留测试数据上评估和比较其系统。

对于第 4 节中的复现研究，我们在相应的单任务训练数据上对预训练模型进行微调（即不进行多任务训练或模型集成）后，报告开发集上的结果。我们的微调过程遵循原始 BERT 论文（Devlin et al., 2019）。

在第 5 节中，我们还报告了从公开排行榜获得的测试集结果。这些结果依赖于若干特定于任务的修改，我们将在第 5.1 节中对此进行描述。

SQuAD 斯坦福问答数据集（Stanford Question Answering Dataset，SQuAD）提供了一段上下文和一个问题。该任务是通过从上下文中提取相关跨度来回答问题。我们在两个版本的 SQuAD 上进行评估：V1.1 和 V2.0（Rajpurkar et al., 2016, 2018）。在 V1.1 中，上下文始终包含答案，而在

V2.0 中有些问题在提供的上下文中没有答案，这使得该任务更具挑战性。

对于 SQuAD V1.1，我们采用与 BERT（Devlin et al., 2019）相同的跨度预测（span prediction）方法。对于 SQuAD V2.0，我们添加了一个额外的二元分类器来预测问题是否可回答，并通过将分类损失和跨度损失项相加来进行联合训练。在评估期间，我们仅对被分类为可回答的问答对预测跨度索引。

RACE 来自考试的阅读理解（ReAding Comprehension from Examinations, RACE）（Lai et al., 2017）任务是一个大规模阅读理解数据集，包含超过 28,000 篇文章和近 100,000 个问题。该数据集收集自中国面向初中和高中学生的英语考试。在 RACE 中，每篇文章都关联有多个问题。对于每个问题，任务是从四个选项中选择一个正确答案。RACE 的上下文明显长于其他流行的阅读理解数据集，并且需要推理的问题比例非常大。

## 4 Training Procedure Analysis

本节探讨并量化了哪些选择对于成功预训练 BERT 模型至关重要。我们保持模型架构固定不变。<sup>7</sup> 具体而言，我们首先使用与 $\mathtt { B E R T _ { B A S E } } \left( L \ = \right.$ 12, H = 768, A = 12, 110M params) 相同的配置来训练 BERT 模型。

## 4.1 Static vs. Dynamic Masking

正如第 2 节所讨论的，BERT 依赖于随机掩码和预测 token。原始 BERT 实现是在数据预处理期间执行一次掩码，从而产生单个静态掩码。为了避免在每个 epoch 中对每个训练实例使用相同的掩码，训练数据被复制了 10 次，以便在 40 个训练 epoch 中，每个序列以 10 种不同的方式进行掩码。因此，在训练期间，每个训练序列以相同的掩码被看到四次。

我们将此策略与动态掩码进行比较，在动态掩码中，我们每次向模型输入序列时都会生成掩码模式。当使用更多步数或更大规模的数据集进行预训练时，这一点变得至关重要。

<table><tr><td>掩码</td><td>SQuAD 2.0</td><td>MNLI-m</td><td>SST-2</td></tr><tr><td>参考值</td><td>76.3</td><td>84.3</td><td>92.8</td></tr><tr><td colspan="4">我们的重新实现：</td></tr><tr><td>静态</td><td>78.3</td><td>84.3</td><td>92.5</td></tr><tr><td>动态</td><td>78.7</td><td>84.0</td><td>92.9</td></tr></table>

Table 1: $\mathbf { B E R T _ { B A S E } }$ 在静态掩码与动态掩码之间的比较。我们报告了 SQuAD 的 F1 值以及 MNLI-m 和 SST-2 的准确率。所报告的结果为 5 次随机初始化（种子）的中位数。参考结果来自 Yang et al. (2019)。

结果 表 1 比较了 Devlin 等人 (2019) 已发表的 $\mathbf { B E R T _ { B A S E } }$ 结果与我们采用静态或动态掩蔽的复现结果。我们发现，采用静态掩蔽的复现模型表现与原始 BERT 模型相似，而动态掩蔽的表现与静态掩蔽相当或略好。

鉴于这些结果以及动态掩蔽带来的额外效率优势，我们在后续实验中均采用动态掩蔽。

## 4.2 模型输入格式与下一句预测

在原始 BERT 预训练过程中，模型接收两个拼接的文档片段作为输入，这两个片段要么连续采样自同一文档（概率 $p = 0 . 5 )$），要么采样自不同的文档。除了掩蔽语言模型目标之外，模型还通过辅助的下一句预测（Next Sentence Prediction, NSP）损失函数进行训练，以预测所输入的文档片段是来自同一文档还是不同文档。

NSP 损失曾被假定为训练原始 BERT 模型的重要因素。Devlin 等人 (2019) 观察到，去除 NSP 会损害性能，在 QNLI、MNLI 和 SQuAD 1.1 上性能显著下降。然而，近期的一些研究对 NSP 损失的必要性提出了质疑 (Lample and Conneau, 2019; Yang et al., 2019; Joshi et al., 2019)。

为了更好地理解这一分歧，我们对比了几种替代的训练格式：

• SEGMENT-PAIR+NSP：遵循 BERT (Devlin et al., 2019) 中使用的原始输入格式，并包含 NSP 损失。每个输入包含一对片段，每个片段可以包含多个自然句子，但两者的总组合长度必须小于 512 个标记（tokens）。

<table><tr><td>模型</td><td>SQuAD 1.1/2.0</td><td>MNLI-m</td><td>SST-2</td><td>RACE</td></tr><tr><td colspan="5">我们的复现（包含 NSP 损失）：</td></tr><tr><td>SEGMENT-PAIR</td><td>90.4/78.7</td><td>84.0</td><td>92.9</td><td>64.2</td></tr><tr><td>SENTENCE-PAIR</td><td>88.7/76.2</td><td>82.9</td><td>92.1</td><td>63.0</td></tr><tr><td colspan="5">我们的复现（不包含 NSP 损失）：</td></tr><tr><td>FULL-SENTENCES</td><td>90.4/79.1</td><td>84.7</td><td>92.5</td><td>64.8</td></tr><tr><td>DOC-SENTENCES</td><td>90.6/79.7</td><td>84.7</td><td>92.7</td><td>65.6</td></tr><tr><td>BERTBASE</td><td>88.5/76.3</td><td>84.3</td><td>92.8</td><td>64.3</td></tr><tr><td>XLNetBASE(K = 7)</td><td>-/81.3</td><td>85.8</td><td>92.7</td><td>66.1</td></tr><tr><td>XLNetBASE(K = 6)</td><td>-/81.0</td><td>85.6</td><td>93.4</td><td>66.7</td></tr></table>

Table 2: 在 BOOKCORPUS 和 WIKIPEDIA 上预训练的 Base 模型在开发集上的结果。所有模型均以 256 个序列的批次大小（batch size）训练 1M 步。我们报告 SQuAD 的 F1 分数以及 MNLI-m、SST-2 和 RACE 的准确率。所报告的结果为 5 次随机初始化（随机种子）的中位数。$\mathbf { B E R T _ { B A S E } }$ 和 $\mathbf { X L N e t } _ { \mathrm { B A S E } }$ 的结果来自 Yang et al. (2019)。

• SENTENCE-PAIR+NSP：每个输入包含一对自然句子，要么采样自单个文档的连续部分，要么采样自不同的文档。由于这些输入显著短于 512 个 token，我们增大了批次大小（batch size），以使总 token 数与 SEGMENT-PAIR+NSP 保持相近。我们保留了 NSP 损失。

• FULL-SENTENCES：每个输入都由从一个或多个文档中连续采样的完整句子填充，使得总长度至多为 512 个 token。输入可以跨越文档边界。当我们到达一个文档的末尾时，开始从下一个文档采样句子，并在文档之间添加一个额外的分隔符 token。我们移除了 NSP 损失。

• DOC-SENTENCES：输入的构建方式与 FULL-SENTENCES 类似，不同之处在于它们不能跨越文档边界。在文档末尾附近采样的输入可能会短于 512 个 token，因此在这些情况下我们动态增加批次大小，以达到与 FULL-SENTENCES 相似的总 token 数量。我们移除了 NSP 损失。

结果 Table 2 展示了四种不同设置下的结果。我们首先将 Devlin et al. (2019) 中的原始 SEGMENT-PAIR 输入格式与 SENTENCE-PAIR 格式进行比较；两种格式都保留了 NSP 损失，但后者使用的是单个句子。我们发现使用单个句子会损害下游任务的性能，我们推测这是因为模型无法学习长距离依赖关系。

接下来，我们比较了不使用 NSP 损失进行训练与使用来自单个文档的文本块（DOC-SENTENCES）进行训练。我们发现该设置的表现优于最初发表的 $\mathbf { B E R T _ { B A S E } }$ 结果，并且与 Devlin et al. (2019) 的结论相反，移除 NSP 损失在下游任务性能上达到相当或略有提升的效果。这可能是因为原版 BERT 实现可能只是移除了损失项，而仍然保留了 SEGMENT-PAIR 输入格式。

最后，我们发现限制序列仅来自单个文档（DOC-SENTENCES）的表现略优于打包来自多个文档的序列（FULL-SENTENCES）。然而，由于 DOC-SENTENCES 格式会导致可变的批次大小，为了更便于与相关工作进行比较，我们在后续实验中均使用 FULL-SENTENCES。

## 4.3 大批次训练

过去在神经机器翻译（Neural Machine Translation）领域的研究表明，在适当增大学习率的情况下，使用超大 mini-batch 进行训练既可以提高优化速度，也可以提升最终任务性能 (Ott et al., 2018)。最近的研究表明，BERT 也同样适用于大批次训练 (You et al., 2019)。

Devlin et al. (2019) 最初训练 $\mathbf { B E R T _ { B A S E } }$ 时采用了 256 个序列的批次大小，训练了 1M 步。通过梯度累积，这在计算成本上等同于使用 2K 个序列的批次大小训练 125K 步，或使用 8K 的批次大小训练 31K 步。

在表 3 中，我们在控制遍历训练数据次数相同的前提下，比较了随着批大小（batch size）增加，$\mathbf { B E R T _ { B A S E } }$ 的困惑度（perplexity）和下游任务性能。我们观察到，使用大批次进行训练可以改善掩码语言模型（masked language modeling）目标的困惑度，并提高下游任务的准确率。大批次也更容易通过分布式数据并行训练进行并行化<sup>8</sup>，在随后的实验中，我们使用包含 8K 个序列的批次进行训练。

<table><tr><td>bsz</td><td>steps</td><td>lr</td><td>ppl</td><td>MNLI-m</td><td>SST-2</td></tr><tr><td>256</td><td>1M</td><td>1e-4</td><td>3.99</td><td>84.7</td><td>92.7</td></tr><tr><td>2K</td><td>125K</td><td>7e-4</td><td>3.68</td><td>85.2</td><td>92.9</td></tr><tr><td>8K</td><td>31K</td><td>1e-3</td><td>3.77</td><td>84.6</td><td>92.8</td></tr></table>

Table 3: 在 BOOKCORPUS 和 WIKIPEDIA 上以不同批大小（bsz）训练的基础模型在保留训练数据上的困惑度（ppl）和开发集准确率。我们针对每种设置调整了学习率（lr）。模型遍历数据的次数（epochs）相同，且具有相同的计算成本。

值得注意的是，You et al. (2019) 使用了更大的批大小训练 BERT，最高达到 32K 个序列。我们将对大批次训练极限的进一步探索留待未来的工作。

## 4.4 文本编码

字节对编码（Byte-Pair Encoding，简称 BPE）（Sennrich et al., 2016）是字符级和词级表示之间的一种混合方法，能够处理自然语言语料库中常见的大型词表。BPE 不依赖完整的单词，而是依赖于通过对训练语料库进行统计分析提取出的子词单元。

BPE 词表大小通常在 10K 到 100K 个子词单元之间。然而，在对大规模且多样化的语料库（例如本工作中所考虑的语料库）进行建模时，Unicode 字符可能会占用该词表的相当大一部分。Radford et al. (2019) 引入了一种巧妙的 BPE 实现，该实现使用字节而不是 Unicode 字符作为基础子词单元。使用字节使得学习一个规模适中（50K 个单元）的子词词表成为可能，该词表仍然可以编码任何输入文本，而无需引入任何“未知”（unknown）标记。

原版 BERT 实现（Devlin et al., 2019）使用了大小为 30K 的字符级 BPE 词表，该词表是在使用启发式分词规则预处理输入后学习得到的。遵循 Radford et al. (2019) 的做法，我们转而考虑使用包含 50K 个子词单元的更大字节级 BPE 词表来训练 BERT，且不对输入进行任何额外的预处理或分词。这分别为 $\mathbf { B E R T _ { B A S E } }$ 和 $\mathbf { B E R T _ { L A R G E } }$ 增加了约 15M 和 20M 的额外参数。

早期实验表明这些编码之间仅存在细微差异，其中 Radford et al. (2019) 的 BPE 在某些任务上的最终任务性能略差。尽管如此，我们认为通用编码方案的优势超过了性能上的轻微下降，并在后续实验中使用了这种编码。对这些编码更详细的比较留待未来工作中进行。

## 5 RoBERTa

在上一节中，我们提出了对 BERT 预训练流程的修改，从而提高了最终任务性能。我们现在整合这些改进并评估它们的综合影响。我们将这一配置称为 RoBERTa（鲁棒优化版 BERT 方法，Robustly optimized BERT approach）。具体而言，RoBERTa 采用了动态掩码（第 4.1 节）、无 NSP 损失的 FULL-SENTENCES（第 4.2 节）、大 mini-batch（第 4.3 节）以及更大的字节级 BPE（第 4.4 节）进行训练。

此外，我们还研究了以往工作中未被充分重视的另外两个重要因素：(1) 用于预训练的数据，以及 (2) 遍历数据的训练轮次。例如，最近提出的 XLNet 架构（Yang et al., 2019）所使用的预训练数据量几乎是原始 BERT（Devlin et al., 2019）的 10 倍。它还采用了 8 倍大的批大小，优化步数减半，因此在预训练中看到的序列数量是 BERT 的 4 倍。

为了将这些因素的重要性与其他建模选择（例如预训练目标）区分开来，我们首先按照 $\mathbf { B E R T _ { L A R G E } }$ 架构（$L = 2 4 ,$ H = 1024, A = 16, 355M 参数）训练 RoBERTa。我们在与之前相当的 BOOK-CORPUS 加 WIKIPEDIA 数据集上预训练了 100K 步，如

<table><tr><td>模型</td><td>数据</td><td>批大小</td><td>步数</td><td>SQuAD (v1.1/2.0)</td><td>MNLI-m</td><td>SST-2</td></tr><tr><td colspan="7">RoBERTa</td></tr><tr><td>使用 BOOKS + WIKI</td><td>16GB</td><td>8K</td><td>100K</td><td>93.6/87.3</td><td>89.0</td><td>95.3</td></tr><tr><td>+ 额外数据 (§3.2)</td><td>160GB</td><td>8K</td><td>100K</td><td>94.0/87.7</td><td>89.3</td><td>95.6</td></tr><tr><td>+ 延长预训练时间</td><td>160GB</td><td>8K</td><td>300K</td><td>94.4/88.7</td><td>90.0</td><td>96.1</td></tr><tr><td>+ 进一步延长预训练时间</td><td>160GB</td><td>8K</td><td>500K</td><td>94.6/89.4</td><td>90.2</td><td>96.4</td></tr><tr><td colspan="7"> $BERT_{LARGE}$ </td></tr><tr><td>使用 BOOKS + WIKI</td><td>13GB</td><td>256</td><td>1M</td><td>90.9/81.8</td><td>86.6</td><td>93.7</td></tr><tr><td colspan="7"> $XLNet_{LARGE}$ </td></tr><tr><td>使用 BOOKS + WIKI</td><td>13GB</td><td>256</td><td>1M</td><td>94.0/87.8</td><td>88.4</td><td>94.4</td></tr><tr><td>+ 额外数据</td><td>126GB</td><td>2K</td><td>500K</td><td>94.5/88.8</td><td>89.8</td><td>95.6</td></tr></table>

Table 4：RoBERTa 在使用更多数据进行预训练（16GB → 160GB 文本）以及更长时间预训练（$( 1 0 0 \mathrm { K }  3 0 0 \mathrm { K }  5 0 0 \mathrm { K }$ steps)）时的开发集结果。每一行都累加了其上方各行的改进。RoBERTa 的架构和训练目标与 $\mathbf { B E R T _ { L A R G E } }$ 保持一致。$\mathbf { B E R T _ { L A R G E } }$ 和 $\mathrm { X L N e t } _ { \mathrm { L A R G E } }$ 的结果分别来自 Devlin et al. (2019) 和 Yang et al. (2019)。所有 GLUE 任务的完整结果见附录。

Devlin et al. (2019)。我们使用 1024 块 V100 GPU 预训练模型大约一天。

结果 我们在 Table 4 中展示了我们的结果。在控制训练数据相同的情况下，我们观察到 RoBERTa 相比最初报告的 $\mathbf { B E R T _ { L A R G E } }$ 结果有了大幅提升，这再次证实了我们在第 4 节中所探索的设计选择的重要性。

接下来，我们将这些数据与第 3.2 节中描述的三个额外数据集合并。我们在合并后的数据上以与之前相同的训练步数（100K）训练 RoBERTa。总计在 160GB 的文本上进行了预训练。我们观察到所有下游任务的性能都有了进一步的提升，这验证了预训练中数据规模和多样性的重要性。<sup>9</sup>

最后，我们对 RoBERTa 进行了显著更长时间的预训练，将预训练步数从 100K 增加到 300K，进而进一步增加到 500K。我们再次观察到下游任务性能的显著提升，且 300K 和 500K 步的模型在大多数任务上均优于 XLNet<sub>LARGE</sub>。我们注意到，即使是我们训练时间最长的模型，似乎也没有对数据产生过拟合，并且可能还会从进一步的训练中受益。

在本文的其余部分，我们在三个不同的基准测试上评估我们表现最好的 RoBERTa 模型：GLUE、SQuaD 和 RACE。具体来说，我们考虑在第 3.2 节介绍的所有五个数据集上训练了 500K 步的 RoBERTa。

## 5.1 GLUE 结果

对于 GLUE，我们考虑两种微调设置。在第一种设置（单任务，开发集）中，我们仅使用对应任务的训练数据分别对每个 GLUE 任务微调 RoBERTa。我们对每个任务进行了有限的超参数搜索，批大小 $\in \{ 1 6 , 3 2 \}$，学习率 $\in \{ 1 e - 5 , 2 e - 5 , 3 e - 5 \}$，在前 6% 的步数中进行线性预热，随后线性衰减至 0。我们微调 10 个周期，并根据各任务在开发集上的评估指标进行早停。其余超参数与预训练期间保持一致。在这种设置下，我们在不进行模型集成的情况下，报告了五次随机初始化在每个任务开发集上的中位数结果。

在第二种设置（集成，测试集）中，我们通过 GLUE 排行榜在测试集上将 RoBERTa 与其他方法进行了比较。虽然提交到 GLUE 排行榜的许多方法都依赖于多任务微调，但我们的提交仅依赖于单任务微调。对于 RTE、STS 和 MRPC，我们发现从 MNLI 单任务模型开始微调比从基线预训练 RoBERTa 开始更有帮助。我们探索了稍宽的超参数空间（在附录中说明），并在每个任务上集成了 5 到 7 个模型。

<table><tr><td></td><td>MNLI</td><td>QNLI</td><td>QQP</td><td>RTE</td><td>SST</td><td>MRPC</td><td>CoLA</td><td>STS</td><td>WNLI</td><td>Avg</td></tr><tr><td colspan="11">开发集上的单任务单模型</td></tr><tr><td> $BERT_{LARGE}$ </td><td>86.6/-</td><td>92.3</td><td>91.3</td><td>70.4</td><td>93.2</td><td>88.0</td><td>60.6</td><td>90.0</td><td>-</td><td>-</td></tr><tr><td> $XLNet_{LARGE}$ </td><td>89.8/-</td><td>93.9</td><td>91.8</td><td>83.8</td><td>95.6</td><td>89.2</td><td>63.6</td><td>91.8</td><td>-</td><td>-</td></tr><tr><td>RoBERTa</td><td>90.2/90.2</td><td>94.7</td><td>92.2</td><td>86.6</td><td>96.4</td><td>90.9</td><td>68.0</td><td>92.4</td><td>91.3</td><td>-</td></tr><tr><td colspan="11">测试集上的集成模型（截至2019年7月25日的排行榜）</td></tr><tr><td>ALICE</td><td>88.2/87.9</td><td>95.7</td><td>90.7</td><td>83.5</td><td>95.2</td><td>92.6</td><td>68.6</td><td>91.1</td><td>80.8</td><td>86.3</td></tr><tr><td>MT-DNN</td><td>87.9/87.4</td><td>96.0</td><td>89.9</td><td>86.3</td><td>96.5</td><td>92.7</td><td>68.4</td><td>91.1</td><td>89.0</td><td>87.6</td></tr><tr><td>XLNet</td><td>90.2/89.8</td><td>98.6</td><td>90.3</td><td>86.3</td><td>96.8</td><td>93.0</td><td>67.8</td><td>91.6</td><td>90.4</td><td>88.4</td></tr><tr><td>RoBERTa</td><td>90.8/90.2</td><td>98.9</td><td>90.2</td><td>88.2</td><td>96.7</td><td>92.3</td><td>67.8</td><td>92.2</td><td>89.0</td><td>88.5</td></tr></table>

Table 5: GLUE 上的结果。所有结果均基于 24 层架构。$\mathbf { B E R T _ { L A R G E } }$ 和 $\mathrm { X L N e t } _ { \mathrm { L A R G E } }$ 的结果分别来自 Devlin et al. (2019) 和 Yang et al. (2019)。RoBERTa 在开发集上的结果是五次运行的中位数。RoBERTa 在测试集上的结果是单任务模型的集成。对于 RTE、STS 和 MRPC，我们从 MNLI 模型开始微调，而不是基线预训练模型。平均值来自 GLUE 排行榜。

特定于任务的修改 GLUE 中的两个任务需要特定于任务的微调方法才能在排行榜上取得具有竞争力的结果。

QNLI：GLUE 排行榜上最近的提交针对 QNLI 任务采用了成对排序（pairwise ranking）的形式，其中从训练集中挖掘候选答案并进行相互比较，且单个（问题，候选）对被分类为正例 (Liu et al., 2019b,a; Yang et al., 2019)。这种形式显著简化了该任务，但无法直接与 BERT (Devlin et al., 2019) 进行比较。遵循近期的研究工作，我们在测试集提交中采用了排序方法，但为了与 BERT 进行直接比较，我们在开发集上报告了基于纯分类方法的结果。

WNLI：我们发现所提供的 NLI 格式数据处理起来颇具挑战。取而代之的是，我们使用了来自 Super-GLUE (Wang et al., 2019a) 的重新格式化的 WNLI 数据，其中指明了查询代词和指称词的跨度。我们使用 Kocijan et al. (2019) 中的边际排序损失（margin ranking loss）来微调 RoBERTa。对于给定的输入句子，我们使用 spaCy (Honnibal and Montani, 2017) 从句子中提取额外的候选名词短语，并对我们的模型进行微调，使其为正指称短语赋予的分数高于任何生成的负候选短语。这种形式的一个遗憾后果是，我们只能利用正向训练样本，这排除了超过一半的所提供训练样本。<sup>10</sup>

结果 我们在 Table 5 中展示了我们的结果。在第一种设定（单任务，开发集）下，RoBERTa 在全部 9 个 GLUE 任务开发集上均取得了最先进的（state-of-the-art）结果。至关重要的是，RoBERTa 采用了与 $\mathbf { B E R T _ { L A R G E } } ,$ 相同的掩码语言模型预训练目标和架构，但在性能上始终优于 $\mathbf { B E R T _ { L A R G E } }$ 和 $\mathrm { X L N e t } _ { \mathrm { L A R G E } }$ 。与我们在本研究中探索的数据集规模和训练时间等更为常规的细节相比，这也引发了关于模型架构与预训练目标相对重要性的思考。

在第二种设定（集成模型，测试集）下，我们将 RoBERTa 提交至 GLUE 排行榜，并在 9 个任务中的 4 个上取得了最先进的结果，同时获得了迄今为止最高的平均分。这尤其令人振奋，因为与大多数其他排名靠前的提交不同，RoBERTa 并不依赖多任务微调。我们期望未来的工作可以通过引入更复杂的多任务微调流程来进一步提升这些结果。

## 5.2 SQuAD 结果

与以往的研究相比，我们对 SQuAD 采用了更为简便的方法。具体而言，尽管 BERT (Devlin et al., 2019) 和 XL-Net (Yang et al., 2019) 都通过额外的 QA 数据集扩充了其训练数据，但我们仅使用所提供的 SQuAD 训练数据来微调 RoBERTa。Yang et al. (2019) 还采用了一种定制的分层学习率调度来微调

<table><tr><td rowspan="2">模型</td><td colspan="2">SQuAD 1.1</td><td colspan="2">SQuAD 2.0</td></tr><tr><td>EM</td><td>F1</td><td>EM</td><td>F1</td></tr><tr><td colspan="5">开发集上的单模型，无数据增强</td></tr><tr><td>BERT $_{LARGE}$ </td><td>84.1</td><td>90.9</td><td>79.0</td><td>81.8</td></tr><tr><td>XLNet $_{LARGE}$ </td><td>89.0</td><td>94.5</td><td>86.1</td><td>88.8</td></tr><tr><td>RoBERTa</td><td>88.9</td><td>94.6</td><td>86.5</td><td>89.4</td></tr><tr><td colspan="5">测试集上的单模型（截至 2019 年 7 月 25 日）</td></tr><tr><td>XLNet $_{LARGE}$ </td><td></td><td></td><td>86.3 $^{\dagger}$ </td><td>89.1 $^{\dagger}$ </td></tr><tr><td>RoBERTa</td><td></td><td></td><td>86.8</td><td>89.8</td></tr><tr><td>XLNet + SG-Net Verifier</td><td></td><td></td><td>87.0 $^{\dagger}$ </td><td>89.9 $^{\dagger}$ </td></tr></table>

Table 6: SQuAD 上的结果。† 表示依赖额外外部训练数据的结果。RoBERTa 在开发集和测试集设置中仅使用提供的 SQuAD 数据。$\mathbf { B E R T _ { L A R G E } }$ 和 $\mathrm { X L N e t } _ { \mathrm { L A R G E } }$ 的结果分别来自 Devlin et al. (2019) 和 Yang et al. (2019)。

XLNet，而我们对所有层使用相同的学习率。

对于 SQuAD v1.1，我们遵循与 Devlin et al. (2019) 相同的微调流程。对于 SQuAD v2.0，我们额外对给定问题是否可回答进行分类；我们通过将分类损失项和跨度损失项相加，来联合训练该分类器与跨度预测器。

结果 我们在 Table 6 中展示了我们的结果。在 SQuAD v1.1 开发集上，RoBERTa 达到了与 XLNet 相当的最优性能。在 SQuAD v2.0 开发集上，RoBERTa 创下了新的最优表现，相较于 XLNet 提升了 0.4 个点（EM）和 0.6 个点（F1）。

我们还将 RoBERTa 提交到了公开的 SQuAD 2.0 排行榜，并评估了其相对于其他系统的性能。大多数顶尖系统均基于 BERT (Devlin et al., 2019) 或 XLNet (Yang et al., 2019) 构建，这两者都依赖于额外的外部训练数据。相比之下，我们的提交未使用任何额外数据。

我们的 RoBERTa 单模型表现优于除一个单模型提交之外的所有系统，并且在不依赖数据增强的系统中得分最高。

## 5.3 RACE 结果

在 RACE 中，系统会获得一段文本、一个相关问题以及四个候选答案。系统需要分类判断四个候选答案中哪一个是正确的。

我们针对该任务对 RoBERTa 进行了调整，将每个候选答案与对应的问题及段落进行拼接。然后我们对这四个序列分别进行编码，并将得到的 [CLS] 表示传入全连接层，用于预测正确答案。我们将长于 128 个 token 的问答对进行截断，并在必要时对段落进行截断，以使总长度至多为 512 个 token。

<table><tr><td>模型</td><td>准确率</td><td>初中</td><td>高中</td></tr><tr><td colspan="4">测试集上的单模型（截至 2019 年 7 月 25 日）</td></tr><tr><td>BERTLARGE</td><td>72.0</td><td>76.6</td><td>70.1</td></tr><tr><td>XLNetLARGE</td><td>81.7</td><td>85.4</td><td>80.2</td></tr><tr><td>RoBERTa</td><td>83.2</td><td>86.5</td><td>81.3</td></tr></table>

Table 7: RACE 测试集上的结果。$\mathbf { B E R T _ { L A R G E } }$ 和 $\mathrm { X L N e t } _ { \mathrm { L A R G E } }$ 的结果来自 Yang et al. (2019)。

RACE 测试集的结果如 Table 7 所示。RoBERTa 在初中和高中设置下均取得了最先进（state-of-the-art）的结果。

## 6 相关工作

预训练方法设计了不同的训练目标，包括语言建模（Dai and Le, 2015; Peters et al., 2018; Howard and Ruder, 2018）、机器翻译（McCann et al., 2017）以及掩码语言建模（Devlin et al., 2019; Lample and Conneau, 2019）。近期许多论文都采用了一种基本范式：针对每个终端任务对模型进行微调（Howard and Ruder, 2018; Radford et al., 2018），并使用掩码语言模型目标的某种变体进行预训练。然而，较新的方法通过多任务微调（Dong et al., 2019）、结合实体嵌入（Sun et al., 2019）、跨度预测（Joshi et al., 2019）以及自回归预训练的多种变体（Song et al., 2019; Chan et al., 2019; Yang et al., 2019）提升了性能。此外，在更多数据上训练更大的模型通常也能提高性能（Devlin et al., 2019; Baevski et al., 2019; Yang et al., 2019; Radford et al., 2019）。我们的目标是复现、简化并更好地调优 BERT 的训练，以此作为一个参考基准，从而更好地理解所有这些方法的相对性能。

## 7 结论

我们对预训练 BERT 模型时的若干设计决策进行了仔细评估。我们发现，通过在更多数据上以更大的批次训练模型更长时间；移除下一句预测（next sentence prediction）目标；在更长的序列上进行训练；以及动态改变应用于训练数据的掩码模式，性能可以得到显著提升。我们改进后的预训练流程（称之为 RoBERTa）在 GLUE、RACE 和 SQuAD 上均取得了最先进的结果，且在 GLUE 上未使用多任务微调，在 SQuAD 上也未使用额外数据。这些结果说明了这些先前被忽视的设计决策的重要性，并表明 BERT 的预训练目标与最近提出的替代方案相比仍然具有竞争力。

此外，我们使用了一个新的数据集 CC-NEWS，并在以下地址开源了用于预训练和微调的模型及代码：https://github.com/pytorch/fairseq。

## 参考文献

Eneko Agirre, Llu’is M‘arquez, and Richard Wicentowski, editors. 2007. Proceedings of the Fourth International Workshop on Semantic Evaluations (SemEval-2007).

Alexei Baevski, Sergey Edunov, Yinhan Liu, Luke Zettlemoyer, and Michael Auli. 2019. 自注意力网络的完形驱动预训练 (Clozedriven pretraining of self-attention networks). arXiv preprint arXiv:1903.07785.

Roy Bar-Haim, Ido Dagan, Bill Dolan, Lisa Ferro, Danilo Giampiccolo, Bernardo Magnini, and Idan Szpektor. 2006. 第二届 PASCAL 文本蕴涵识别挑战赛 (The second PASCAL recognising textual entailment challenge). In Proceedings of the second PASCAL challenges workshop on recognising textual entailment.

Luisa Bentivogli, Ido Dagan, Hoa Trang Dang, Danilo Giampiccolo, and Bernardo Magnini. 2009. 第五届 PASCAL 文本蕴涵识别挑战赛 (The fifth PASCAL recognizing textual entailment challenge).

Samuel R Bowman, Gabor Angeli, Christopher Potts, and Christopher D Manning. 2015. 用于学习自然语言推理的大规模标注语料库 (A large annotated corpus for learning natural language inference). In Empirical Methods in Natural Language Processing (EMNLP).

William Chan, Nikita Kitaev, Kelvin Guu, Mitchell Stern, and Jakob Uszkoreit. 2019. KERMIT: 基于生成的序列插入式建模 (KERMIT: Generative insertion-based modeling for sequences). arXiv preprint arXiv:1906.01604.

Ido Dagan, Oren Glickman, and Bernardo Magnini. 2006. PASCAL 文本蕴涵识别挑战赛 (The PASCAL recognising textual entailment challenge). In Machine learning challenges. evaluating predictive uncertainty, visual object classification, and recognising tectual entailment.

Andrew M Dai and Quoc V Le. 2015. 半监督序列学习 (Semi-supervised sequence learning). In Advances in Neural Information Processing Systems (NIPS).

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: 用于语言理解的深度双向 Transformers 预训练 (BERT: Pre-training of deep bidirectional transformers for language understanding). In North American Association for Computational Linguistics (NAACL).

William B Dolan and Chris Brockett. 2005. 自动构建句子释义语料库 (Automatically constructing a corpus of sentential paraphrases). In Proceedings of the International Workshop on Paraphrasing.

Li Dong, Nan Yang, Wenhui Wang, Furu Wei, Xiaodong Liu, Yu Wang, Jianfeng Gao, Ming Zhou, and Hsiao-Wuen Hon. 2019. 面向自然语言理解与生成的统一语言模型预训练 (Unified language model pre-training for natural language understanding and generation). arXiv preprint arXiv:1905.03197.

Danilo Giampiccolo, Bernardo Magnini, Ido Dagan, and Bill Dolan. 2007. 第三届 PASCAL 文本蕴涵识别挑战赛 (The third PASCAL recognizing textual entailment challenge). In Proceedings of the ACL-PASCAL workshop on textual entailment and paraphrasing.

Aaron Gokaslan and Vanya Cohen. 2019. OpenWebText 语料库 (Openwebtext corpus). http://web.archive.org/ save/http://Skylion007.github.io/ OpenWebTextCorpus.

Felix Hamborg, Norman Meuschke, Corinna Breitinger, and Bela Gipp. 2017. news-please: 通用新闻抓取与提取工具 (news-please: A generic news crawler and extractor). In Proceedings ofthe 15th International Symposium ofInformation Science.

Dan Hendrycks and Kevin Gimpel. 2016. 高斯误差线性单元 (Gaussian error linear units (gelus)). arXiv preprint arXiv:1606.08415.

Matthew Honnibal and Ines Montani. 2017. spaCy 2: 基于 Bloom 嵌入、卷积神经网络与增量依存分析的自然语言理解 (spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing). 待发表 (To appear).

Jeremy Howard and Sebastian Ruder. 2018. 用于文本分类的通用语言模型微调 (Universal language model fine-tuning for text classification). arXiv preprint arXiv:1801.06146.

Shankar Iyer, Nikhil Dandekar, and Kornl Csernai. 2016. First quora dataset release: Question pairs. https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs.

Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S. Weld, Luke Zettlemoyer, and Omer Levy. 2019. SpanBERT: Improving pre-training by representing and predicting spans. arXiv preprint arXiv:1907.10529.

Diederik Kingma and Jimmy Ba. 2015. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR).

Vid Kocijan, Ana-Maria Cretu, Oana-Maria Camburu, Yordan Yordanov, and Thomas Lukasiewicz. 2019. A surprisingly robust trick for winograd schema challenge. arXiv preprint arXiv:1905.06290.

Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. 2017. Race: Large-scale reading comprehension dataset from examinations. arXiv preprint arXiv:1704.04683.

Guillaume Lample and Alexis Conneau. 2019. Crosslingual language model pretraining. arXiv preprint arXiv:1901.07291.

Hector J Levesque, Ernest Davis, and Leora Morgenstern. 2011. The Winograd schema challenge. In AAAI Spring Symposium: Logical Formalizations of Commonsense Reasoning.

Xiaodong Liu, Pengcheng He, Weizhu Chen, and Jianfeng Gao. 2019a. Improving multi-task deep neural networks via knowledge distillation for natural language understanding. arXiv preprint arXiv:1904.09482.

Xiaodong Liu, Pengcheng He, Weizhu Chen, and Jianfeng Gao. 2019b. Multi-task deep neural networks for natural language understanding. arXiv preprint arXiv:1901.11504.

Bryan McCann, James Bradbury, Caiming Xiong, and Richard Socher. 2017. Learned in translation: Contextualized word vectors. In Advances in Neural Information Processing Systems (NIPS), pages 6297– 6308.

Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, and Hao Wu. 2018. Mixed precision training. In International Conference on Learning Representations.

Sebastian Nagel. 2016. Cc-news. http: //web.archive.org/save/http: //commoncrawl.org/2016/10/newsdataset-available.

Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. 2019. FAIRSEQ: A fast, extensible toolkit for sequence modeling. In North American Association for Computational Linguistics (NAACL): System Demonstrations.

Myle Ott, Sergey Edunov, David Grangier, and Michael Auli. 2018. Scaling neural machine translation. In Proceedings of the Third Conference on Machine Translation (WMT).

Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. 2017. Automatic differentiation in PyTorch. In NIPS AutodiffWorkshop.

Matthew Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. 2018. Deep contextualized word representations. In North American Association for Computational Linguistics (NAACL).

Alec Radford, Karthik Narasimhan, Time Salimans, and Ilya Sutskever. 2018. Improving language understanding with unsupervised learning. Technical report, OpenAI.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language models are unsupervised multitask learners. Technical report, OpenAI.

Pranav Rajpurkar, Robin Jia, and Percy Liang. 2018. Know what you don’t know: Unanswerable questions for squad. In Association for Computational Linguistics (ACL).

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. SQuAD: 100,000+ questions for machine comprehension of text. In Empirical Methods in Natural Language Processing (EMNLP).

Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. Neural machine translation of rare words with subword units. In Association for Computational Linguistics (ACL), pages 1715–1725.

Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In Empirical Methods in Natural Language Processing (EMNLP).

Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, and Tie-Yan Liu. 2019. MASS: Masked sequence to sequence pre-training for language generation. In International Conference on Machine Learning (ICML).

Yu Stephanie Sun, Shuohuan Wang, Yukun Li, Shikun Feng, Xuyi Chen, Han Zhang, Xinlun Tian, Danxiang Zhu, Hao Tian, and Hua Wu. 2019. ERNIE: Enhanced representation through knowledge integration. arXiv preprint arXiv:1904.09223.

Trieu H Trinh and Quoc V Le. 2018. A simple method for commonsense reasoning. arXiv preprint arXiv:1806.02847.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in neural information processing systems.

Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. 2019a. SuperGLUE: A stickier benchmark for general-purpose language understanding systems. arXiv preprint 1905.00537.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. 2019b. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In International Conference on Learning Representations (ICLR).

Alex Warstadt, Amanpreet Singh, and Samuel R. Bowman. 2018. Neural network acceptability judgments. arXiv preprint 1805.12471.

Adina Williams, Nikita Nangia, and Samuel Bowman. 2018. A broad-coverage challenge corpus for sentence understanding through inference. In North American Association for Computational Linguistics (NAACL).

Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, and Quoc V Le. 2019. Xlnet: Generalized autoregressive pretraining for language understanding. arXiv preprint arXiv:1906.08237.

Yang You, Jing Li, Jonathan Hseu, Xiaodan Song, James Demmel, and Cho-Jui Hsieh. 2019. Reducing bert pre-training time from 3 days to 76 minutes. arXiv preprint arXiv:1904.00962.

Rowan Zellers, Ari Holtzman, Hannah Rashkin, Yonatan Bisk, Ali Farhadi, Franziska Roesner, and Yejin Choi. 2019. Defending against neural fake news. arXiv preprint arXiv:1905.12616.

Yukun Zhu, Ryan Kiros, Richard Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. 2015. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. In arXiv preprint arXiv:1506.06724.

## “RoBERTa: A Robustly Optimized BERT Pretraining Approach”附录

## A GLUE 上的完整结果

在 Table 8 中，我们给出了 RoBERTa 在开发集上的完整结果。我们展示了遵循 BERT<sub>LARGE</sub> 的 LARGE 配置结果，以及遵循 $\mathbf { B E R T _ { B A S E } } .$ 的 BASE 配置结果。

## B 预训练超参数

Table 9 描述了 $\mathrm { R o B E R T a } _ { \mathrm { L A R G E } }$ 和 $\mathrm { R o B E R T a _ { B A S E } }$ 的预训练超参数。

## C 微调超参数

Table 10 给出了 RACE、SQuAD 和 GLUE 的微调超参数。我们根据每个任务在 5 个随机种子下的中位数来选择最佳超参数值。

<table><tr><td></td><td>MNLI</td><td>QNLI</td><td>QQP</td><td>RTE</td><td>SST</td><td>MRPC</td><td>CoLA</td><td>STS</td></tr><tr><td> $RoBERTa_{BASE}$ + 全部数据 + 500k 步</td><td>87.6</td><td>92.8</td><td>91.9</td><td>78.7</td><td>94.8</td><td>90.2</td><td>63.6</td><td>91.2</td></tr><tr><td> $RoBERTa_{LARGE}$ 使用 BOOKS + WIKI</td><td>89.0</td><td>93.9</td><td>91.9</td><td>84.5</td><td>95.3</td><td>90.2</td><td>66.3</td><td>91.6</td></tr><tr><td>+ 额外数据 (§3.2)</td><td>89.3</td><td>94.0</td><td>92.0</td><td>82.7</td><td>95.6</td><td>91.4</td><td>66.1</td><td>92.2</td></tr><tr><td>+ 更长预训练 300k</td><td>90.0</td><td>94.5</td><td>92.2</td><td>83.3</td><td>96.1</td><td>91.1</td><td>67.4</td><td>92.3</td></tr><tr><td>+ 更长预训练 500k</td><td>90.2</td><td>94.7</td><td>92.2</td><td>86.6</td><td>96.4</td><td>90.9</td><td>68.0</td><td>92.4</td></tr></table>

Table 8: RoBERTa 在不同配置下于 GLUE 任务开发集上的结果。

<table><tr><td>超参数</td><td>RoBERTaLARGE</td><td>RoBERTaBASE</td></tr><tr><td>层数</td><td>24</td><td>12</td></tr><tr><td>隐藏层大小</td><td>1024</td><td>768</td></tr><tr><td>FFN 内部隐藏层大小</td><td>4096</td><td>3072</td></tr><tr><td>注意力头数</td><td>16</td><td>12</td></tr><tr><td>注意力头大小</td><td>64</td><td>64</td></tr><tr><td>Dropout</td><td>0.1</td><td>0.1</td></tr><tr><td>注意力 Dropout</td><td>0.1</td><td>0.1</td></tr><tr><td>预热步数</td><td>30k</td><td>24k</td></tr><tr><td>峰值学习率</td><td>4e-4</td><td>6e-4</td></tr><tr><td>批大小</td><td>8k</td><td>8k</td></tr><tr><td>权重衰减</td><td>0.01</td><td>0.01</td></tr><tr><td>最大步数</td><td>500k</td><td>500k</td></tr><tr><td>学习率衰减</td><td>线性</td><td>线性</td></tr><tr><td>Adam ε</td><td>1e-6</td><td>1e-6</td></tr><tr><td>Adam β1</td><td>0.9</td><td>0.9</td></tr><tr><td>Adam β2</td><td>0.98</td><td>0.98</td></tr><tr><td>梯度裁剪</td><td>0.0</td><td>0.0</td></tr></table>

Table 9: 预训练 $\mathrm { R o B E R T a _ { L A R G E } }$ 和 $\mathrm { R o B E R T a _ { B A S E } }$ 的超参数

<table><tr><td>超参数</td><td>RACE</td><td>SQuAD</td><td>GLUE</td></tr><tr><td>学习率</td><td>1e-5</td><td>1.5e-5</td><td>{1e-5, 2e-5, 3e-5}</td></tr><tr><td>批大小</td><td>16</td><td>48</td><td>{16, 32}</td></tr><tr><td>权重衰减</td><td>0.1</td><td>0.01</td><td>0.1</td></tr><tr><td>最大轮数</td><td>4</td><td>2</td><td>10</td></tr><tr><td>学习率衰减</td><td>线性</td><td>线性</td><td>线性</td></tr><tr><td>预热比例</td><td>0.06</td><td>0.06</td><td>0.06</td></tr></table>

Table 10: 在 RACE、SQuAD 和 GLUE 上微调 $\mathrm { R o B E R T a _ { L A R G E } }$ 的超参数。