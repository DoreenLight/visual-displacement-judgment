# GLU 变体提升 Transformer 性能

Noam Shazeer Google noam@google.com

2020年2月14日

## 摘要

门控线性单元（Gated Linear Units, GLU）[Dauphin et al., 2016] 由两个线性投影的逐元素乘积组成，其中一个投影首先通过 Sigmoid 函数。我们可以对 GLU 进行变体设计，使用不同的非线性（甚至线性）函数来代替 Sigmoid。我们在 Transformer [Vaswani et al., 2017] 序列到序列模型的前馈子层中测试了这些变体，并发现其中一些变体相比于通常使用的 ReLU 或 GELU 激活函数带来了质量上的提升。

## 1 引言

Transformer [Vaswani et al., 2017] 序列到序列模型在多头注意力机制和所谓的“逐位置前馈网络”（position-wise feed-forward networks, FFN）之间交替进行。FFN 接收一个向量 x（序列中特定位置的隐藏表示），并将其通过两个学习到的线性变换（由矩阵 $W _ { 1 }$ 和 $W _ { 2 }$ 以及偏置向量 $b _ { 1 }$ 和 $b _ { 2 }$ 表示）。在这两个线性变换之间应用了线性整流（ReLU）[Glorot et al., 2011] 激活函数。

$$
\operatorname{FFN} (x, W _ {1}, W _ {2}, b _ {1}, b _ {2}) = \max (0, x W _ {1} + b _ {1}) W _ {2} + b _ {2}\tag{1}
$$

遵循 T5 代码库 [Rafel et al., 2019] <sup>1</sup>，我们使用无偏置的版本：

$$
\mathrm{FFN} _ {\mathrm{ReLU}} (x, W _ {1}, W _ {2}) = \max (x W _ {1}, 0) W _ {2}\tag{2}
$$

后续的研究提出用其他非线性激活函数替代 ReLU，例如高斯误差线性单元（Gaussian Error Linear Units）$\mathrm { { G E L U } } ( x ) = x \Phi ( x )$ [Hendrycks and Gimpel, 2016] 以及 $\mathrm { S w i s h } _ { \beta } ( x ) = x \sigma ( \beta x )$ [Ramachandran et al., 2017]。

$$
\begin{array}{c} \mathrm{FFN} _ {\mathrm{GELU}} (x, W _ {1}, W _ {2}) = \mathrm{GELU} (x W _ {1}) W _ {2} \\ \mathrm{FFN} _ {\mathrm{Swish}} (x, W _ {1}, W _ {2}) = \mathrm{Swish} _ {1} (x W _ {1}) W _ {2} \end{array}\tag{3}
$$

## 2 门控线性单元（GLU）及其变体

[Dauphin et al., 2016] 提出了门控线性单元（GLU），这是一种神经网络层，定义为输入的两个线性变换的逐元素乘积，其中一个经过 Sigmoid 激活。他们还建议省略激活函数，并将其称为“双线性”（bilinear）层，归功于 [Mnih and Hinton, 2007]。

$$
\begin{array}{c} \operatorname{GLU} (x, W, V, b, c) = \sigma (x W + b) \otimes (x V + c) \\ \operatorname{Bilinear} (x, W, V, b, c) = (x W + b) \otimes (x V + c) \end{array}\tag{4}
$$

我们也可以使用其他激活函数来定义 GLU 变体：

$$
\begin{array}{c} \operatorname{ReGLU} (x, W, V, b, c) = \max (0, x W + b) \otimes (x V + c) \\ \operatorname{GEGLU} (x, W, V, b, c) = \operatorname{GELU} (x W + b) \otimes (x V + c) \\ \operatorname{SwiGLU} (x, W, V, b, c, \beta) = \operatorname{Swish} _ {\beta} (x W + b) \otimes (x V + c) \end{array}\tag{5}
$$

在本文中，我们提出了 Transformer FFN 层的其他变体，这些变体使用 GLU 或其变体之一来替代第一个线性变换和激活函数。同样，我们省略了偏置项。

$$
\begin{array}{c} \mathrm{FFN} _ {\mathrm{GLU}} (x, W, V, W _ {2}) = (\sigma (x W) \otimes x V) W _ {2} \\ \mathrm{FFN} _ {\mathrm{Bilinear}} (x, W, V, W _ {2}) = (x W \otimes x V) W _ {2} \\ \mathrm{FFN} _ {\mathrm{ReGLU}} (x, W, V, W _ {2}) = (\max (0, x W) \otimes x V) W _ {2} \\ \mathrm{FFN} _ {\mathrm{GEGLU}} (x, W, V, W _ {2}) = (\mathrm{GELU} (x W) \otimes x V) W _ {2} \\ \mathrm{FFN} _ {\mathrm{SwiGLU}} (x, W, V, W _ {2}) = (\mathrm{Swish} _ {1} (x W) \otimes x V) W _ {2} \end{array}\tag{6}
$$

所有这些层都具有三个权重矩阵，而原始 FFN 只有两个。为了保持参数量和计算量不变，当将这些层与原始的双矩阵版本进行比较时，我们将隐藏单元数 $d _ { f f }$（W 和 V 的第二维度以及 $W _ { 2 } )$ 的第一维度）缩减为原来的 $\textstyle { \frac { 2 } { 3 } }$。

## 3 Text-to-Text Transfer Transformer (T5) 上的实验

我们在 [Rafel et al., 2019] 的迁移学习设置上测试了所描述的 FFN 变体。一个编码器-解码器 Transformer 模型 [Vaswani et al., 2017] 在预测缺失文本片段的去噪目标上进行训练，随后在各种语言理解任务上进行微调。

## 3.1 模型架构

我们使用了与 [Rafel et al., 2019] 中的基础模型相同的代码库、模型架构和训练任务。编码器和解码器各由 12 层组成，其中 $d _ { m o d e l } = 7 6 8$。对于注意力层，$h = 1 2$ 且 $d _ { k } = d _ { v } = 6 4$。FFN 层的隐藏层大小为 $d _ { f f } = 3 0 7 2$。正如我们在上文中所述，对于基于 GLU 变体的 FFN 层（具有三个权重矩阵而非两个），我们将隐藏层大小缩减为 $d _ { f f } = 2 0 4 8$，以保持与基础模型相同的参数量和计算量。

Table 1: Transformer 模型在 [Rafel et al., 2019] 的片段填充任务中在保留集上的对数困惑度（log-perplexity）。所有模型的参数量和计算量均保持一致。

<table><tr><td>训练步数</td><td>65,536</td><td>524,288</td></tr><tr><td> $FFN_{ReLU(baseline)}$ </td><td>1.997 (0.005)</td><td>1.677</td></tr><tr><td> $FFN_{GELU}$ </td><td>1.983 (0.005)</td><td>1.679</td></tr><tr><td> $FFN_{Swish}$ </td><td>1.994 (0.003)</td><td>1.683</td></tr><tr><td> $FFN_{GLU}$ </td><td>1.982 (0.006)</td><td>1.663</td></tr><tr><td> $FFN_{Bilinear}$ </td><td>1.960 (0.005)</td><td>1.648</td></tr><tr><td> $FFN_{GEGLU}$ </td><td>1.942 (0.004)</td><td>1.633</td></tr><tr><td> $FFN_{SwiGLU}$ </td><td>1.944 (0.010)</td><td>1.636</td></tr><tr><td> $FFN_{ReGLU}$ </td><td>1.953 (0.003)</td><td>1.645</td></tr></table>

## 3.2 预训练与困惑度结果

与 [Rafel et al., 2019] 完全一致，我们在 C4 数据集上针对跨度填充（span-filling）目标预训练了 524,288 步。每个训练批次由 128 个样本组成，每个样本具有 512 个 token 的输入和 114 个 token 的输出，输出包含从输入中被删除的多个 token 跨度<sup>2</sup>。与 [Rafel et al., 2019] 类似，我们使用 Adafactor 优化器 [Shazeer and Stern, 2018] 和反平方根学习率调度策略。我们还在最后 10% 的训练步数中对学习率进行线性衰减。与 [Rafel et al., 2019] 的主要不同之处在于，我们在预训练期间没有使用 dropout。我们发现这样做能产生更优的结果。我们在 C4 的保留分片（heldout shard）上计算训练目标上的对数困惑度（log-perplexity），我们认为这是衡量模型质量的良好指标。对于每种模型架构，我们还训练了四个模型，训练时间较短（65,536 步），以测量运行间的可变性。结果列于 Table 1 中。GEGLU 和 SwiGLU 变体取得了最佳的困惑度。

## 3.3 微调

随后，我们在按样本比例混合的斯坦福问答数据集（SQuAD）[Rajpurkar et al., 2016] 以及 GLUE [Wang et al., 2018] 和 SuperGlue [Wang et al., 2019] 基准中的所有语言理解任务上，对每个充分训练的模型进行了一次微调。<sup>3</sup>微调包含 131,072 步，学习率为 10<sup>−3</sup>。与训练中一样，每一步的输入序列总长度约为 65,536 个 token。遵循 [Rafel et al., 2019]，我们在层输出、前馈隐藏层和注意力权重上使用了 0.1 的 dropout 率。嵌入矩阵在微调期间保持固定。

Table 2、3 和 4 展示了在开发集上的结果。对于每个任务，我们报告了微调期间记录的所有检查点中的最佳得分。尽管结果存在噪声波动，但新的 GLU 变体在大多数任务上表现最好。为了进行对比，我们在每个表格的底部列出了来自 [Rafel et al., 2019] 的结果。该模型与我们的 $\mathrm { F F N } _ { \mathrm { R e L U } }$ 模型完全相同。他们的结果明显更差，我们认为这是由于他们在预训练期间使用了 dropout 导致的。表中还列出了由 [Rafel et al., 2019] 测得的运行间标准差。

Table 2：GLUE 语言理解基准 [Wang et al., 2018]（开发集）。

<table><tr><td></td><td>平均得分 (Score Average)</td><td>CoLA MCC</td><td>SST-2 Acc</td><td>MRPC F1</td><td>MRPC Acc</td><td>STSB PCC</td><td>STSB SCC</td><td>QQP F1</td><td>QQP Acc</td><td>MNLIIm Acc</td><td>MNLIImm Acc</td><td>QNLI Acc</td><td>RTE Acc</td></tr><tr><td> $FFN_{ReLU}$ </td><td>83.80</td><td>51.32</td><td>94.04</td><td>93.08</td><td>90.20</td><td>89.64</td><td>89.42</td><td>89.01</td><td>91.75</td><td>85.83</td><td>86.42</td><td>92.81</td><td>80.14</td></tr><tr><td> $FFN_{GELU}$ </td><td>83.86</td><td>53.48</td><td>94.04</td><td>92.81</td><td>90.20</td><td>89.69</td><td>89.49</td><td>88.63</td><td>91.62</td><td>85.89</td><td>86.13</td><td>92.39</td><td>80.51</td></tr><tr><td> $FFN_{Swish}$ </td><td>83.60</td><td>49.79</td><td>93.69</td><td>92.31</td><td>89.46</td><td>89.20</td><td>88.98</td><td>88.84</td><td>91.67</td><td>85.22</td><td>85.02</td><td>92.33</td><td>81.23</td></tr><tr><td> $FFN_{GLU}$ </td><td>84.20</td><td>49.16</td><td>94.27</td><td>92.39</td><td>89.46</td><td>89.46</td><td>89.35</td><td>88.79</td><td>91.62</td><td>86.36</td><td>86.18</td><td>92.92</td><td>84.12</td></tr><tr><td> $FFN_{GEGLU}$ </td><td>84.12</td><td>53.65</td><td>93.92</td><td>92.68</td><td>89.71</td><td>90.26</td><td>90.13</td><td>89.11</td><td>91.85</td><td>86.15</td><td>86.17</td><td>92.81</td><td>79.42</td></tr><tr><td> $FFN_{Bilinear}$ </td><td>83.79</td><td>51.02</td><td>94.38</td><td>92.28</td><td>89.46</td><td>90.06</td><td>89.84</td><td>88.95</td><td>91.69</td><td>86.90</td><td>87.08</td><td>92.92</td><td>81.95</td></tr><tr><td> $FFN_{SwiGLU}$ </td><td>84.36</td><td>51.59</td><td>93.92</td><td>92.23</td><td>88.97</td><td>90.32</td><td>90.13</td><td>89.14</td><td>91.87</td><td>86.45</td><td>86.47</td><td>92.93</td><td>83.39</td></tr><tr><td> $FFN_{ReGLU}$ </td><td>84.67</td><td>56.16</td><td>94.38</td><td>92.06</td><td>89.22</td><td>89.97</td><td>89.85</td><td>88.86</td><td>91.72</td><td>86.20</td><td>86.40</td><td>92.68</td><td>81.59</td></tr><tr><td>[Raffel et al., 2019]</td><td>83.28</td><td>53.84</td><td>92.68</td><td>92.07</td><td>88.92</td><td>88.02</td><td>87.94</td><td>88.67</td><td>91.56</td><td>84.24</td><td>84.57</td><td>90.48</td><td>76.28</td></tr><tr><td>同上标准差 (ibid. stddev.)</td><td>0.235</td><td>1.111</td><td>0.569</td><td>0.729</td><td>1.019</td><td>0.374</td><td>0.418</td><td>0.108</td><td>0.070</td><td>0.291</td><td>0.231</td><td>0.361</td><td>1.393</td></tr></table>

## 4 结论

我们对 GLU（门控线性单元）层族进行了扩展，并提议将其应用于 Transformer。在迁移学习设定下，新变体在预训练所使用的去噪目标上似乎能产生更优的困惑度，同时在诸多下游语言理解任务中也取得了更好的结果。这些架构实现简便，且没有明显的计算缺陷。我们不对这些架构为何有效给出任何解释；我们将它们的成功，一如其他一切事物，归因于神圣的恩惠。

Table 3: SuperGLUE 语言理解基准 [Wang et al., 2019]（开发集）。

<table><tr><td></td><td>平均得分</td><td>BoolQ Acc</td><td>CB F1</td><td>CB Acc</td><td>CoPA Acc</td><td>MultiRC F1</td><td>MultiRC EM</td><td>ReCoRD F1</td><td>ReCoRD EM</td><td>RTE Acc</td><td>WiC Acc</td><td>WSC Acc</td></tr><tr><td> $FFN_{ReLU}$ </td><td>72.76</td><td>80.15</td><td>83.37</td><td>89.29</td><td>70.00</td><td>76.93</td><td>39.14</td><td>73.73</td><td>72.91</td><td>83.39</td><td>67.71</td><td>77.88</td></tr><tr><td> $FFN_{GELU}$ </td><td>72.98</td><td>80.64</td><td>86.24</td><td>91.07</td><td>74.00</td><td>75.93</td><td>38.61</td><td>72.96</td><td>72.03</td><td>81.59</td><td>68.34</td><td>75.96</td></tr><tr><td> $FFN_{Swish}$ </td><td>72.40</td><td>80.43</td><td>77.75</td><td>83.93</td><td>67.00</td><td>76.34</td><td>39.14</td><td>73.34</td><td>72.36</td><td>81.95</td><td>68.18</td><td>81.73</td></tr><tr><td> $FFN_{GLU}$ </td><td>73.95</td><td>80.95</td><td>77.26</td><td>83.93</td><td>73.00</td><td>76.07</td><td>39.03</td><td>74.22</td><td>73.50</td><td>84.12</td><td>67.71</td><td>87.50</td></tr><tr><td> $FFN_{GEGLU}$ </td><td>73.96</td><td>81.19</td><td>82.09</td><td>87.50</td><td>72.00</td><td>77.43</td><td>41.03</td><td>75.28</td><td>74.60</td><td>83.39</td><td>67.08</td><td>83.65</td></tr><tr><td> $FFN_{Bilinear}$ </td><td>73.81</td><td>81.53</td><td>82.49</td><td>89.29</td><td>76.00</td><td>76.04</td><td>40.92</td><td>74.97</td><td>74.10</td><td>82.67</td><td>69.28</td><td>78.85</td></tr><tr><td> $FFN_{SwiGLU}$ </td><td>74.56</td><td>81.19</td><td>82.39</td><td>89.29</td><td>73.00</td><td>75.56</td><td>38.72</td><td>75.35</td><td>74.55</td><td>85.20</td><td>67.24</td><td>86.54</td></tr><tr><td> $FFN_{ReGLU}$ </td><td>73.66</td><td>80.89</td><td>86.37</td><td>91.07</td><td>67.00</td><td>75.32</td><td>40.50</td><td>75.07</td><td>74.18</td><td>84.48</td><td>67.40</td><td>79.81</td></tr><tr><td>[Raffel et al., 2019]</td><td>71.36</td><td>76.62</td><td>91.22</td><td>91.96</td><td>66.20</td><td>66.13</td><td>25.78</td><td>69.05</td><td>68.16</td><td>75.34</td><td>68.04</td><td>78.56</td></tr><tr><td>同上标准差</td><td>0.416</td><td>0.365</td><td>3.237</td><td>2.560</td><td>2.741</td><td>0.716</td><td>1.011</td><td>0.370</td><td>0.379</td><td>1.228</td><td>0.850</td><td>2.029</td></tr></table>

Table 4: SQuAD [Rajpurkar et al., 2016] v1.1（开发集）。

<table><tr><td></td><td>EM</td><td>F1</td></tr><tr><td> $FFN_{ReLU}$ </td><td>83.18</td><td>90.87</td></tr><tr><td> $FFN_{GELU}$ </td><td>83.09</td><td>90.79</td></tr><tr><td> $FFN_{Swish}$ </td><td>83.25</td><td>90.76</td></tr><tr><td> $FFN_{GLU}$ </td><td>82.88</td><td>90.69</td></tr><tr><td> $FFN_{GEGLU}$ </td><td>83.55</td><td>91.12</td></tr><tr><td> $FFN_{Bilinear}$ </td><td>83.82</td><td>91.06</td></tr><tr><td> $FFN_{SwiGLU}$ </td><td>83.42</td><td>91.03</td></tr><tr><td> $FFN_{ReLU}$ </td><td>83.53</td><td>91.18</td></tr><tr><td>[Raffel et al., 2019]</td><td>80.88</td><td>88.81</td></tr><tr><td>同上标准差</td><td>0.343</td><td>0.226</td></tr></table>

## 参考文献

Yann N. Dauphin, Angela Fan, Michael Auli, and David Grangier. Language modeling with gated convolutional networks. CoRR, abs/1612.08083, 2016. URL http://arxiv.org/abs/1612.08083.

Xavier Glorot, Antoine Bordes, and Yoshua Bengio. Deep sparse rectifier neural networks. In Proceedings of the fourteenth international conference on artificial intelligence and statistics, pages 315–323, 2011.

Dan Hendrycks and Kevin Gimpel. Bridging nonlinearities and stochastic regularizers with gaussian error linear units. CoRR, abs/1606.08415, 2016. URL http://arxiv.org/abs/1606.08415.

Andriy Mnih and Geofrey Hinton. Three new graphical models for statistical language modelling. In Proceedings of the 24th international conference on Machine learning, pages 641–648, 2007.

Colin Rafel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv e-prints, 2019.

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions for machine comprehension of text. arXiv preprint arXiv:1606.05250, 2016.

Prajit Ramachandran, Barret Zoph, and Quoc V Le. Searching for activation functions. arXiv preprint arXiv:1710.05941, 2017.

Noam Shazeer and Mitchell Stern. Adafactor: Adaptive learning rates with sublinear memory cost. arXiv preprint arXiv:1804.04235, 2018.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NIPS, 2017.

Alex Wang, Amapreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. GLUE: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint arXiv:1804.07461, 2018.

Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. Superglue: A stickier benchmark for general-purpose language understanding systems. arXiv preprint arXiv:1905.00537, 2019.