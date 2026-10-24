# Kinematic Tokenization: Optimization-Based Continuous-Time Tokens for Learnable Decision Policies in Noisy Time Series

Grifin M. Kearney, Ph.D.

Syracuse, New York, USA gkearneyengineering@gmail.com

January 21, 2026

## Abstract

Transformers are designed for discrete tokens, yet many real-world signals are continuous processes observed through noisy sampling. Discrete tokenizations (raw values, patches, finite diferences) can be brittle in low signal-to-noise regimes, especially when downstream objectives impose asymmetric penalties that rationally encourage abstention. We introduce Kinematic Tokenization, an optimization-based continuous-time representation that reconstructs an explicit spline from noisy measurements and tokenizes local spline coeficients (position, velocity, acceleration, jerk). This is applied to financial time series data in the form of asset prices in conjunction with trading volume profiles. Across a multi-asset daily-equity testbed, we use a risk-averse asymmetric classification objective as a stress test for learnability. Under this objective, several discrete baselines collapse to an absorbing cash policy (the Liquidation Equilibrium), whereas the continuous spline tokens sustain calibrated, non-trivial action distributions and stable policies. These results suggest that explicit continuous-time tokens can improve the learnability and calibration of selective decision policies in noisy time series under abstention-inducing losses.

## 1 Introduction

The adaptation of Foundation Models, which were originally designed for discrete natural lan guage tokens, to continuous time series data presents a fundamental representation challenge. While text is inherently discrete, physical and economic signals are continuous systems viewed through the lens of a discrete sampling clock. Current approaches often rely on patching or direct quantization of raw values. These methods fail to account for measurement uncertainty or the underlying dynamic laws (i.e. the “physics”) governing system evolution between sam ples. While financial markets inherently lack the rigorous first-principle governance found in natural physical systems, in our experiments we impose Newton-like equations of motion on the evolution of trading parameters, utilizing kinematic consistency as a robust inductive bias for representation learning.

## 1.1 Physics-Informed Tokenization Framework

In this paper, we bridge this gap between discrete measurement samples and the continuous underlying processes by situating the time series problem within a framework of Physics-Informed AI. Rather than treating data points as static numbers, we treat them as noisy observations of a latent dynamical system subject to stochastic forcing. We introduce a method ology based on Optimization-Based Data Enrichment [1]. As illustrated in Figure 1, our framework fundamentally incorporates physics through two stochastic channels and two coupled manifolds:

1. Dynamical Systems Model: A description of how the system state (e.g. Price) and system flux (e.g. Volume) evolve continuously over time, subject to random process noise.

2. Measurement Model: A description of how we observe these states, distinguishing between snapshot measurements (e.g. Price) and aggregate measurements (e.g. Volume).

![](images/fae459e887633005f5a73e9c41042114423712f24dfac57afc5e30089be836c5.jpg)  
Figure 1: Conceptual Framework for Physics-Informed Continuous Tokenization. The ap proach integrates models of system dynamics and measurement protocols, both subject to stochastic uncertainty. Unlike traditional filtering (e.g. Kalman) which yields discrete estimates via implicit repre sentations, our optimization-based approach produces an explicit continuous spline representation. This continuum is necessary to extract high-order kinematic tokens for the foundation model.

## 1.2 Explicit vs. Implicit Representations

A critical distinction of our approach is the nature of the resulting representation. Traditional methods like Kalman filtering are implicit: they produce best estimates at pre-specified dis crete filtering times enabling discrete data correction and higher fidelity (but still discrete) sampling. Continuous limits of the Kalman approach do exist [2], but these treatments fail to produce compact discrete representations (e.g. spline coeficients) which are natural for use in Transformer architectures; these methods are typically implicit and recursive. In contrast, our optimization framework solves for an explicit continuous spline that defines the state x(t) and all its derivatives for all times t in the interval via a direct compact representation.

This explicit nature is paramount for tokenization. It allows us to analytically compute high-order derivatives, namely velocity, acceleration, jerk, and volume flow rates, which are often absent in discrete methods but contain vital information about the underlying forces driving the system. These kinematic parameters form the basis of our continuous-time tokens, providing the Transformer with a physically grounded representation of the signal’s coupled evolution.

## 1.3 Finance as a Stochastic Testbed

While the experimental validation in this work focuses on daily data in financial markets, the proposed framework is domain-agnostic. Financial time series represent a particularly challeng ing class of stochastic dynamic systems, characterized by low signal-to-noise ratios and time varying volatility. However, the underlying optimal spline formulation is generalizable. By recasting the dynamics, process noise and measurement models, this strategy can be adapted to distinct physical domains, such as inertial navigation, robotic control, or other diverse signal processing applications, where preserving the continuous derivatives of the signal is essential.

## 2 Related Work

## 2.1 Foundation Models for Time Series

The success of Large Language Models has spurred the development of Large Observation Models (LOMs). Current state-of-the-art architectures like PatchTST [3] and iTransformer [4] utilize patching to capture local semantic information. More recent eforts in 2025, such as Google’s TimesFM-ICF [5], have introduced in-context learning to adapt to non-stationary data. However, these models remain physically agnostic; they treat time series patches as statistical vectors, ignoring the underlying kinematic continuity of the signal.

## 2.2 Emergent Continuous Architectures (SSMs & LNNs)

To address the computational ineficiency of Transformers on long sequences, State Space Models (SSMs) like Mamba have gained prominence. Recent applications such as MambaStock [6] and TimeMachine [7] leverage Mamba’s selective scan mechanism to learn discretized latent dynamics with linear complexity. Similarly, Liquid Neural Networks [8] and Physics-Informed Koopman Networks [9] explicitly model neuronal activations or latent states via diferential equations to handle irregularity.

While efective, these methods typically operate on raw, noisy discrete inputs and must learn the system dynamics (matrix A or ODE parameters) from scratch. SplineGPT difer entiates itself by decoupling the physical extraction from the pattern recognition. We do not ask the model to extract the physics; we explicitly calculate a continuous state (velocity, acceleration, etc) via spline optimization ex-ante, providing the Transformer with pre-computed physically-grounded discrete representations of the continuous physics while retaining the training eficiency of standard attention mechanisms.

## 2.3 Physics-Informed Finance

The application of Physics-Informed Neural Networks (PINNs) [10] to finance has largely focused on solving known Partial Diferential Equations (PDEs). Recent work by Socaciu et al. (2025) [11] demonstrates the utility of PINNs for solving Black-Scholes and Heston equations for option pricing. Deep learning approaches in limit order books typically rely on discrete spatial or temporal structures [12, 13]. However, these approaches are largely deductive—enforcing known laws onto data. Our approach is inductive: we use optimization-based enrichment [1] to derive a latent kinematic state from the asset price data stream, enabling alpha generation in trading scenarios where a general governing law is unknown.

It should be noted that in the present approach local “laws” are prescribed in the absence of fundamental physics (e.g. stochastically forced second order motion mirroring Newton’s second law) which are used for the financial data tokenization. We emphasize that the tokenization scheme that is described here is adaptable to other data domains where rich physical frame works governing system behavior are proven by modifying the governing laws and measurement models. The same general treatment is applicable leveraging rigorous physics when available to adjust the equivalent data enrichment formulation as we demonstrate in (1) and (2).

## 3 Methodology

Our proposed pipeline transforms raw financial data into a rich latent representation suitable for high-dimensional embedding. The process consists of three distinct stages: (1) Signal Preprocessing, (2) Optimization-Based Spline Enrichment, and (3) Stationarity & Normalization. We represent the proposed ML system architecture in Figure 2.

![](images/3f7637cf11aff1bebfdee79c8556f47d6c105ff93d83e7ab958ae3ec232b3d87.jpg)  
Figure 2: Continuous Tokenization Pipeline. The raw discrete data is transformed and then enriched using optimization-based stochastic spline generation [1]. The coeficients representing Position, Velocity, Acceleration, and Jerk are extracted, anchored to the window start, and fed into a Causa Transformer.

## 3.1 Pre-tokenization Transforms

Financial time series, specifically asset prices $P ( t )$ , are defined on the non-negative real line $\mathbb { R } ^ { + }$ . Direct analysis of raw prices often introduces scaling instabilities. To map the state space to a domain better suited to standard stochastic processes, we apply a logarithmic transformation $y _ { k } = \ln ( P ( t _ { k } ) )$ . This aligns the data with symmetric stochastic processes (e.g., Gaussian process) used in the enrichment phase.

The logarithmic transform is also applied to the daily trading volume data $V ( t )$ yielding $\tilde { y } _ { k } = \ln ( V ( t _ { k } ) )$ . As before this aligns the data with symmetric processes and promotes numerical stability by compressing dynamic range; mega-caps trade with daily volumes in the hundreds of millions whereas small-caps trade with daily volumes in the tens of thousands.

## 3.2 Optimization-Based Spline Enrichment

The core innovation relies on our previous work [1]. We view the transformed sequence $\left\{ y _ { k } \right\}$ as noisy measurements of a latent continuous state x(t) governed by a stochastic diferential equation (SDE). We form splines which maximize likelihood over the interval $[ t _ { 0 } , t _ { N } ]$

$$
\begin{array}{r l} \min _ {v, w, x} & \int_ {t _ {0}} ^ {t _ {N}} \frac {1}{2} v ^ {2} d t + \frac {\alpha^ {2}}{2} \sum_ {k} w _ {k} ^ {2} \\ \text {s.t.} & \ddot {x} (t) = v (t) \\ & y _ {k} = x (t _ {k}) + w _ {k}, k = 0, 1,..., N \end{array}\tag{1}
$$

Using the calculus of variations, the optimal solution $x ^ { * } ( t )$ is a continuous spline defined by the ratio of process noise to measurement noise $\begin{array} { r } { ( \alpha ^ { 2 } = \frac { \sigma _ { p } ^ { 2 } } { \sigma _ { m } ^ { 2 } } ) } \end{array}$ . The parameter $\alpha ^ { 2 }$ acts as a filtering parameter, separating the underlying market trend from high-frequency microstructure noise (e.g., bid-ask bounce), ensuring the tokens represent signal rather than jitter. In prototype models described in the present work we use a value of $\alpha = 5$ for proof of concept, but future detailed statistical studies can be used to optimize this stochastic filter parameter for improved performance.

Problem (1) provides a rigorous kinematic basis for the tokenization of asset prices. For trading volume, we adopt a diferent philosophy. Since volume is an aggregate measure (shares traded over an interval) rather than a continuous state variable, we employ a heuristic vol ume smoother designed to stabilize the input for the Transformer. We formulate the volume extraction as:

$$
\begin{array}{r l} \min _ {\tilde {v}, \tilde {w}, \tilde {x}} & \int_ {t _ {0}} ^ {t _ {N}} \frac {1}{2} \tilde {v} ^ {2} d t + \frac {\alpha^ {2}}{2} \sum_ {k} \tilde {w} _ {k} ^ {2} \\ \mathrm{s.t.} & \ddot {\tilde {x}} (t) = \tilde {v} (t) \\ \mathrm{s.t.} & \tilde {y} _ {k} = \int_ {t _ {k - 1}} ^ {t _ {k}} \tilde {x} (t) d t + \tilde {w} _ {k}, k = 1, \dots , N. \end{array}\tag{2}
$$

We explicitly distinguish the interpretation of (2) from (1). While (1) represents a physical equation of motion, (2) acts as a phenomenological filter [14]. The measurement equation uses log-transformed aggregate volume $( \tilde { y } _ { k } = \ln ( V _ { k } ) )$ Mathematically, treating the integral of the state ${ \tilde { x } } ( t )$ as consistent with the log of the sum is a known approximation. However, in the context of deep learning inputs, this heuristic serves a critical engineering purpose: it compresses the dynamic range of volume spikes (which span orders of magnitude) while enforcing temporal continuity via the spline constraints. We find this approximation empirically robust for stabilizing the tokenization of heavy-tailed volume data, preventing outlier gradients from destabilizing the Transformer backbone.

Solving (1) generates a piecewise optimal (maximum likelihood) estimate of the log-price given by

$$
x (t) = c _ {0} (t _ {k}) + c _ {1} (t _ {k}) (t - t _ {k}) + c _ {2} (t _ {k}) \frac {(t - t _ {k}) ^ {2}}{2} + c _ {3} (t _ {k}) \frac {(t - t _ {k}) ^ {3}}{6},\tag{3}
$$

where $t \in [ t _ { k } , t _ { k + 1 } ]$ and the constants corresponding to this interval are provided by the optimal solution. The coeficients $c _ { j } ( t _ { k } )$ represents the j-th kinematic term (e.g. position, velocity, etc) on the interval starting at $t _ { k }$ . Similarly, solving (2) generates a piecewise optimal estimate of the Volume Intensity Function ${ \tilde { x } } ( t )$ defined by

$$
\tilde {x} (t) = \tilde {c} _ {0} (t _ {k}) + \tilde {c} _ {1} (t _ {k}) (t - t _ {k}) + \tilde {c} _ {2} (t _ {k}) \frac {(t - t _ {k}) ^ {2}}{2} + \tilde {c} _ {3} (t _ {k}) \frac {(t - t _ {k}) ^ {3}}{6} + \tilde {c} _ {4} (t _ {k}) \frac {(t - t _ {k}) ^ {4}}{2 4},\tag{4}
$$

for $t \in [ t _ { k } , t _ { k + 1 } ]$ . Here, the coeficients ˜c do not represent strict physical derivatives, but rather shape parameters of the local log-volume rate profile that allow the model to detect demand surges (velocity $\tilde { c } _ { 1 } )$ and exhaustion patterns (acceleration $\tilde { c } _ { 2 } )$ .

## 3.3 Continuous-Time Tokens

The solution to (1) is a cubic spline, and therefore on each interval the spline is fully defined by 4 coeficients. Similarly, the solution to (2) is a quartic spline and therefore on each interval is fully defined by 5 coeficients. From the optimal spline $x ^ { * } ( t )$ , we extract the coeficients defining the signal segment for each interval for use as data tokens. Unlike discrete points, our tokens $\mathbf { t } _ { k }$ and $\tilde { \mathbf { t } } _ { k }$ are vectors representing the kinematic state of the market on the interval:

$$
\mathbf {t} _ {k} = [ c _ {0}, c _ {1}, c _ {2}, c _ {3} ] ^ {T} \in \mathbb {R} ^ {4},\tag{5}
$$

and

$$
\tilde {\mathbf {t}} _ {k} = [ \tilde {c} _ {0}, \tilde {c} _ {1}, \tilde {c} _ {2}, \tilde {c} _ {3}, \tilde {c} _ {4} ] ^ {T} \in \mathbb {R} ^ {5}.\tag{6}
$$

In (5) $c _ { 0 }$ is “position” (log-price), $c _ { 1 }$ is “velocity” (returns), $c _ { 2 }$ is “acceleration” (convexity), and $c _ { 3 }$ is $\mathrm { \bar { \Psi } j e r k } ^ { \prime \prime }$ (trend instability). This interpretation is modified in (6) where trading volume permits interpretation as an “energy” equivalent; the more shares traded, the more energetic the underlying asset. Under this analogy $\tilde { c } _ { 0 }$ is “power” (log-volume rate), $\tilde { c } _ { 1 }$ is the first timederivative of “power”, and so on. This results in an architecture with a net 9-dimensional token on each time interval when $\mathbf { t } _ { k }$ and $\tilde { \mathbf { t } } _ { k }$ are concatenated for processing. The concatenated token vector contains both price and volume information, enabling the Transformer to attend to coupled dynamics between the two quantities.

## 3.4 Stationarity & Normalization

A critical challenge in training foundation models on financial data is stationarity. Raw price levels $\left( c _ { 0 } \right)$ and volume intensities $\left( \tilde { c } _ { 0 } \right)$ are non-stationary, causing “mean reversion bias” where the model incorrectly predicts a return to the dataset mean rather than following the local trend.

To resolve this, we implement Dual Window Anchoring. For every context window W, we subtract the initial price position $c _ { 0 } ( t _ { 0 } )$ and initial volume intensity $\tilde { c } _ { 0 } ( t _ { 0 } )$ from all tokens in the sequence:

$$
c _ {0} ^ {\prime} (t _ {k}) = c _ {0} (t _ {k}) - c _ {0} (t _ {0})\tag{7}
$$

$$
\tilde {c} _ {0} ^ {\prime} (t _ {k}) = \tilde {c} _ {0} (t _ {k}) - \tilde {c} _ {0} (t _ {0})\tag{8}
$$

This forces every training sequence to begin at a relative origin of (0.0, 0.0), allowing the model to better learn relative dynamics for both price and volume.

Furthermore, the higher-order derivatives exhibit vast scale disparities (e.g., price velocity $c _ { 1 } \approx 1 0 ^ { - 2 }$ vs. volume jounce $\tilde { c } _ { 4 } \approx 5 . 7 )$ . We compute global statistics $( \sigma \in \mathbb { R } ^ { 7 } )$ for the non-anchored coeficients over the training corpus and apply Z-score normalization [15] to the derivatives $( c _ { 1 } , c _ { 2 } , c _ { 3 } )$ and volume shape parameters $( \tilde { c } _ { 1 } , \tilde { c } _ { 2 } , \tilde { c } _ { 3 } , \tilde { c } _ { 4 } )$

## 4 Model Architecture and Training

## 4.1 SplineGPT Architecture Specification

We employ a lightweight, decoder-only Causal Transformer designed for high-throughput in ference and rapid adaptation. The architecture is configured as a “Short $\&$ Wide” model, prioritizing high-bandwidth feature extraction (width) over deep logical reasoning (depth) to capture the complex correlations between the 9 coupled physical state variables. The specific hyperparameters are detailed in Table 1.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Layers (L)</td><td>4</td></tr><tr><td>Attention Heads (H)</td><td>8</td></tr><tr><td>Embedding Dimension ( $d_{model}$ )</td><td>512</td></tr><tr><td>Feed-Forward Dimension ( $d_{ff}$ )</td><td>2048</td></tr><tr><td>Context Window (T)</td><td>64</td></tr><tr><td>Input Channels</td><td>9 (4 Price + 5 Volume)</td></tr><tr><td>Dropout</td><td>0.1</td></tr><tr><td>Activation</td><td>GELU</td></tr><tr><td>Positional Encoding</td><td>RoPE</td></tr></table>

Table 1: SplineGPT Hyperparameters

The total trainable parameter count is approximately 12.6 million. While larger than typical “Nano” models, this capacity is beneficial for resolving the 9-dimensional kinematic state space. Despite the larger width, the shallow depth (4 layers) supports a model that is small enough for fast CPU inference $/$ low latency deployments. Prototype models were trained using a personal desktop with 64GB of RAM, an NVIDIA GeForce RTX 2080 SUPER GPU with 8GB of onboard memory, and an Intel i9-9900 CPU.

## 4.2 Training and Testing Data Strategy

The model was pre-trained on a large corpus of US Equities data. Following dataset consolidation and quality assurance filtering – deficient ticker removal based on low liquidity, missing data, etc – the final training corpus comprises 22.5 million continuous spline windows. With a context length of 64 intervals per window, this equates to approximately 1.44 Billion continuous-time tokens.

To ensure rigorous scientific validation and prevent look-ahead bias (data leakage), we enforce a strict temporal cutof:

• Training Set (Foundation & Adapter): Data from Jan 1, 2000 to Dec 31, 2022.

• Test Set (Evaluation Only): Data from Jan 1, 2023 to Present.

The Foundation Model is entirely blind to the test period, ensuring that any performance on recent market regimes is a result of generalized kinematic learning rather than memorization.

## 4.3 Training Stability

Training on high-variance derivative tokens $\left( \mathrm { e . g . } c _ { 3 } , \tilde { c } _ { 4 } \right)$ presents stability challenges. To ensure convergence, we employ a Cosine Annealing Learning Rate Scheduler, decaying the learning rate from $6 e ^ { - 4 }$ to 0 over 5 epochs. Additionally, we implement Gradient Clipping (norm $\leq 1 . 0 )$ to prevent exploding gradients caused by outliers in the volatility surface.

## 4.4 Fine-Tuning via Low-Rank Adaptation (LoRA)

Following the general pre-training phase, we employ Low-Rank Adaptation (LoRA) to specialize the model. LoRA freezes the pre-trained weights W and introduces trainable rank decomposi tion matrices A and B, such that $W ^ { \prime } = W + B A$ . We implement a strategic injection of LoRA adapters targeting the input and output processing stages of our 4-layer architecture:

1. Perception Adaptation (Layers 1 & 2): Fine-tunes low-level feature extraction $( \mathrm { e . g . }$ , detecting volume-price divergence).

2. Reasoning Adaptation (Layer 4): Fine-tunes the final decision boundary for strategy formulation.

The fine-tuning head is trained for a classification objective: Buy, Sell, or Hold.

## 4.5 Training Objectives & Labeling

To align the model’s kinematic reasoning with actionable trading signals, we employ a ternary classification objective (Buy, Sell, Hold) based on a “Momentum Logic” threshold. We adopt the notation $g _ { j }$ for the discrete categorical label to distinguish it from the measurement observations $y _ { k }$ used in the enrichment phase.

For a given context window ending at time $t _ { j }$ , the label $g _ { j }$ is determined by the state of the target token $c _ { 0 } ( t _ { j + 1 } )$ relative to the last input token $c _ { 0 } ( t _ { j } )$ ):

$$
g _ {j} = \left\{ \begin{array}{l l} 0 (\text {Buy}) & \text {if r_{j} >\tau} \\ 1 (\text {Sell}) & \text {if r_{j} <  - \tau} \\ 2 (\text {Hold}) & \text {otherwise} \end{array} \right.\tag{9}
$$

where $r _ { j } = c _ { 0 } ( t _ { j + 1 } ) - c _ { 0 } ( t _ { j } )$ is the log-return over the next interval. In this work, we set the volatility threshold $\tau = 0 . 0 1 \left( 1 \% \right)$ . This filters out minor noise, forcing the model to only predict “actionable” moves when the kinematic signal suggests a significant price excursion.

To induce the “Active Risk Management” behavior, we train using an asymmetric Weighted Cross-Entropy Loss:

$$
\mathcal {L} = - \sum_ {i \in \{0, 1, 2 \}} w _ {i} \mathbb {I} (g = i) \log (\hat {g} _ {i})\tag{10}
$$

Here, $\hat { g } _ { i }$ represents the predicted probability assigned by the model to class i, and $\mathbb { I } ( g = i )$ is the indicator function for the true label $g .$ . We assign weights $\mathbf { w } = [ 2 . 0 , 1 0 . 0 , 1 . 0 ]$ . The high penalty on the Sell class $( w _ { s e l l } = 1 0 . 0 )$ conditions the model to be hyper-sensitive to downside kinematic signatures (e.g., negative jerk), prioritizing capital preservation over accuracy.

## 4.6 Tokenization Ablations and External Baseline

To isolate the efect of continuous kinematic tokenization, we perform two complementary comparisons. (i) Controlled internal ablations: we hold the Transformer backbone, parameter count, training objective, and adaptation method fixed, varying only the input representation (Spline vs. discrete Raw vs. discrete finite diferences). (ii) External architecture baseline: we additionally compare against PatchTST [3] as a representative patching-based time-series foundation model. This second comparison matches parameter scale but is not strictly ce teris paribus, and is therefore interpreted as a behavioral robustness reference rather than a controlled ablation.

<table><tr><td>Model</td><td>Input Representation</td><td>Physics?</td><td>Backbone</td><td>Params</td><td>Adaptation</td></tr><tr><td>SplineGPT</td><td>Continuous Spline Coeffs</td><td>Explicit (Exact)</td><td>Transf. (4L, 8H)</td><td>12.6M</td><td>LoRA</td></tr><tr><td>RawGPT-FD</td><td>Discrete Finite Differences</td><td>Explicit (Noisy)</td><td>Transf. (4L, 8H)</td><td>12.6M</td><td>LoRA</td></tr><tr><td>RawGPT</td><td>Raw Discrete Values</td><td>None (Ablated)</td><td>Transf. (4L, 8H)</td><td>12.6M</td><td>LoRA</td></tr><tr><td>PatchTST [3]</td><td>Discrete Patches (P=16,S=8)</td><td>Implicit (Latent)</td><td>Transf. (4L, 4H)</td><td>≈12M</td><td>Linear Probe</td></tr></table>

Table 2: Model Specifications. Controlled ablations (SplineGPT, RawGPT-FD, RawGPT) share the same Transformer backbone and training objective; diferences are isolated to input representation. PatchTST is included as an external patching-based reference model with comparable parameter scale, but difers architecturally and in adaptation method.

## 4.6.1 Internal Ablations: The Value of Smoothness

We define two internal variants that utilize the exact same transformer backbone as SplineGPT but lack the spline enrichment step.

RawGPT-FD (The Discrete Physics Baseline): This model tests whether the smoothness and accuracy of the spline is superior to standard discrete diferentiators. It is provided with the full kinematic state vector (Position, Velocity, Acceleration), but these features are derived via explicit finite diferences:

$$
c _ {n, k} \approx \Delta^ {n} y _ {k} = \sum_ {i = 0} ^ {n} (- 1) ^ {n - i} {\binom {n} {i}} y _ {k - (n - i)}.\tag{11}
$$

While this theoretically provides the same information as the spline $( c _ { 2 } , c _ { 3 } )$ , discrete diferentiation amplifies measurement noise $( \mathrm { V a r } ( \Delta y ) \approx 2 \mathrm { V a r } ( y ) )$ . If SplineGPT outperforms RawGPT-FD, it confirms the value of optimization-based denoising over arithmetic approximation.

RawGPT (The Physics-Ablated Baseline): This model represents an information-poor condition. It tests whether high-order kinematics are necessary at all. We map discrete inputs directly to the channel dimension, but we mask the higher-order derivatives $( c _ { 2 } , c _ { 3 } )$ and volume shape parameters $( \tilde { c } _ { 2 } - \tilde { c } _ { 4 } )$ with zeros. This forces the model to rely solely on raw directional moves and scalar volatility, efectively removing the “physics” from the input.

## 4.6.2 External Benchmark: PatchTST

To validate that our results are not artifacts of our specific backbone, we benchmark against PatchTST [3], a state-of-the-art foundation model for multivariate time series.

• Matched Capacity: We configure the PatchTST backbone (Channel-Independent) with $d _ { m o d e l } = 2 5 6$ and 4 heads. This results in a total parameter count (≈ 12M) comparable to SplineGPT, preventing the benchmark from being under-parameterized.

• Implicit Smoothing: PatchTST relies on patch-based tokenization $( P = 1 6 , S = 8 )$ to capture local semantics. This serves as a test of Implicit vs. Explicit smoothing: does the model learn better when it aggregates noisy points itself (Patching), or when we mathematically pre-process the points into smooth curves (Splines)?

Accordingly, PatchTST results are interpreted as an external reference point rather than a strictly controlled ablation.

## 5 Experimental Validation

We conducted a backtesting campaign on a test set comprised of 6 major US equities (INTC, NVDA, JPM, XOM, PFE, TSLA) from Jan 1, 2023, to Present.

## 5.1 Experimental Setup & Protocol

A critical requirement for valid time-series backtesting is the strict prevention of look-ahead bias. To ensure this, we implement a Rolling Spline Tokenization protocol. We do not fit a single global spline to the entire dataset. Instead, for every inference step t:

1. Window Extraction: We retrieve the raw discrete price/volume history strictly for the window $[ t - T , t ]$

2. Local Enrichment: We solve the optimization problems (1) and (2) on this isolated window to generate the continuous function $x ^ { * } ( t )$

3. Tokenization: We extract the coeficients $\mathbf { t } _ { t - T : t }$ from this local spline.

4. Inference: The Transformer processes these tokens to predict the state at $t + 1$

The parameter T denotes the context window length. This ensures that the “kinematic state” at time t is derived solely from information available up to time t.

## 5.1.1 Backtest Constraints & Limitations

The simulation assumes an initial capital of \$10,000. We enforce a tax penalty by applying a 32% short-term capital gains tax on realized profits at the end of every 252-day trading year.

• Execution: Trades are simulated at the daily close price at time t, coincident with the inference step. We assume a “Market-On-Close” (MOC) execution regime where the model inference is completed using near-close data, and orders are filled at the closing auction. Given the high liquidity of the selected large-cap universe and the experimental focus on representation learning rather than high-turnover arbitrage, we treat execution latency as negligible.

• Limitations: This study isolates the predictive power of the representation; therefore, it does not currently model variable transaction costs (slippage, commissions) or market impact. While the notably profitable performance of the model provides a significant bufer against these costs, future work will incorporate full order book dynamics.

• Baselines: The primary financial benchmark is the Buy & Hold strategy, representing the standard opportunity cost for an investor.

## 5.1.2 Trading Policy & State Machine

To translate the ternary model outputs $( g _ { t } ~ \in ~ \{ \mathrm { B u y } , \mathrm { S e l l } , \mathrm { H o l d } \} )$ into portfolio states, we implement a strict finite state machine (FSM). The portfolio state $S _ { t }$ at time t is binary: $S _ { t } \in \{ 0 \ ( \mathrm { C a s h } ) , 1 \ ( \mathrm { L o n g } ) \}$ . Short selling and leverage are explicitly disabled to isolate the signal quality from leverage efects.

The state transitions are governed by the following logic:

• Entry (Cash → Long): If $S _ { t } = 0$ and the model predicts $g _ { t } = \mathrm { B u y }$ , a buy order is executed at the closing price $P _ { t }$ . The entire available capital (minus transaction costs) is converted to shares.

• Exit $( \mathbf { L o n g } \to \mathbf { C a s h } )$ : If $S _ { t } = 1$ and the model predicts $g _ { t } = \mathrm { S e l l }$ , a sell order is executed at $P _ { t }$ . All held shares are liquidated to cash.

• Maintenance: If $g _ { t } = \mathrm { H o l d }$ , or if the model signals an invalid transition $( \mathrm { e . g . }$ , signaling $^ { 6 6 } \mathrm { B u y } ^ { 5 9 }$ when already Long), the state remains unchanged $( S _ { t + 1 } = S _ { t } )$

Definition (Liquidation Equilibrium). We distinguish action-space outputs from portfoliostate outcomes. A model may collapse to a degenerate action policy $( \mathrm { e . g . }$ , predicting a single label). Under our FSM, any degenerate policy that fails to generate valid entries (Buy while in Cash) yields an absorbing cash state $S _ { t } = 0$ with zero exposure. We term this portfolio-level absorbing behavior the Liquidation Equilibrium. In our experiments, the observed Liquida tion Equilibrium for discrete baselines is realized specifically through a monotonic Always-Sel prediction $( g _ { t } = \mathrm { S e l l } )$ , which immediately (or persistently) drives and maintains $S _ { t } = 0$ . This definition clarifies that the “Liquidation Equilibrium” observed in the baselines (Section 6.1) corresponds to the absorbing state $S _ { t } = 0$ , reached when a model monotonically predicts $g _ { t } = \mathrm { S e l l }$ regardless of input.

## 5.1.3 Evaluation Metrics

To rigorously quantify the risk-adjusted performance of the learned representation, we report the following metrics alongside total return:

• Sharpe Ratio: Measures excess return per unit of total volatility. Defined as

$$
S = \frac {E [ R _ {p} - R _ {f} ]}{\eta},
$$

where $R _ { p }$ is the annualized portfolio return, $R _ { f }$ is the risk-free rate (assumed $4 \%$ for the 2023–2025 period), and η is the annualized standard deviation of daily returns.

• Sortino Ratio: A modification of the Sharpe ratio that penalizes only downside volatility. Defined as

$$
S _ {o r t} = \frac {E [ R _ {p} - R _ {f} ]}{\sigma_ {d}},
$$

where $\sigma _ { d }$ is the standard deviation of negative asset returns (downside deviation). This metric is particularly relevant for our “Active Risk Management” objective, which seeks to minimize capital destruction rather than variance per se.

• Maximum Drawdown (MaxDD): The maximum observed loss from a peak to a trough of the portfolio equity curve over the test period:

$$
\mathrm{MaxDD} = \min _ {t} \left(\frac {C _ {t} - \max _ {s \leq t} C _ {s}}{\max _ {s \leq t} C _ {s}}\right),
$$

where $C _ { t }$ is the value of the portfolio at time t. This quantifies the worst-case scenario for an investor holding the strategy.

## 5.2 Results

We present our findings in three stages: (1) Behavioral Diagnostics, establishing that the model is actually learning a policy rather than collapsing; (2) Financial Performance, quantifying the value of that policy; and (3) Robustness, verifying stability across thresholds and costs.

## 5.2.1 Behavioral Diagnostics: Escaping the Liquidation Equilibrium

Before evaluating returns, we strictly evaluate the learnability of the task. Given the asymmetric loss $( w _ { s e l l } = 1 0 . 0 )$ , a model with low signal confidence will rationally converge to a “Liquidation Equilibrium” (always Sell/Cash) to minimize risk.

Figure 3 visualizes this divergence for the NVDA trading study. While the discrete baseline (PatchTST) collapses to a monotonic distribution (Action Rate → 100% Sell), SplineGPT maintains a balanced ternary distribution. This confirms that the continuous spline representation provides suficient signal-to-noise ratio to discern actionable information and justify entry risk.

![](images/c971830cfcd770fbb3845a9148d6a4e982fec479abfe73bbe402732b81f82daf.jpg)

![](images/958e5db985aa35781a0d46a8844f8dba84af4e20d7b1cb695878b18abbbcf545.jpg)

![](images/22e5c91743a67df6b8648428d26f0e7dc71ffe31b7ed23708c945f1cce45d222.jpg)  
Figure 3: Behavioral Divergence and Calibration (NVDA). (A) Action distribution shows the discrete baseline (PatchTST, Red) collapsing to a narrow density peak (Liquidation Equilibrium), whereas SplineGPT (Blue) maintains a broad, active policy distribution. (B) Confusion matrix confirms SplineGPT achieves meaningful separation between Hold (2) and Action $( 0 / 1 )$ states, rather than defaulting to a single class. (C) Calibration plot indicates the model’s probability estimates for “Buy” signals track the empirical win rate, staying close to the diagonal (perfect calibration).

## 5.2.2 Financial Performance

Having established behavioral validity, we report the financial outcomes in Tables 3 and 4, with cumulative equity curves visualized in Figure 6. SplineGPT reliably outperforms the market benchmark on downside risk (Max Drawdown) while capturing significant upside in momentum regimes (NVDA, TSLA). Notably, in the PFE crash scenario, the model’s ability to recognize negative kinematic signatures (Jerk) allowed it to exit early, resulting in a flat equity curve while the asset declined ≈ 40%.

<table><tr><td></td><td colspan="5">Total Return (Cumulative)</td><td colspan="5">Max Drawdown (Risk)</td></tr><tr><td>Ticker</td><td>Spline</td><td>Patch</td><td>FD</td><td>Raw</td><td>Mkt</td><td>Spline</td><td>Patch</td><td>FD</td><td>Raw</td><td>Mkt</td></tr><tr><td>INTC</td><td>+240.4%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>+71.6%</td><td>-34.7%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>-48.2%</td></tr><tr><td>NVDA</td><td>+579.2%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>+1185.7%</td><td>-32.1%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>-56.4%</td></tr><tr><td>JPM</td><td>+82.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>+159.9%</td><td>-12.5%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>-22.1%</td></tr><tr><td>XOM</td><td>+48.4%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>+28.5%</td><td>-13.5%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>-26.3%</td></tr><tr><td>PFE</td><td>+3.6%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>-41.2%</td><td>-18.8%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>-44.5%</td></tr><tr><td>TSLA</td><td>+699.8%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>+306.2%</td><td>-33.4%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>-63.5%</td></tr></table>

Table 3: Absolute Performance Metrics. All non-spline baselines collapsed to a Liquidation Equilibrium (0.0% return/risk). SplineGPT generates active policies with superior capital preserva tion (MaxDD).

<table><tr><td></td><td colspan="5">Sharpe Ratio</td><td colspan="5">Sortino Ratio</td></tr><tr><td>Ticker</td><td>Spline</td><td>Patch</td><td>FD</td><td>Raw</td><td>Mkt</td><td>Spline</td><td>Patch</td><td>FD</td><td>Raw</td><td>Mkt</td></tr><tr><td>INTC</td><td>1.24</td><td>N/A</td><td>N/A</td><td>N/A</td><td>0.85</td><td>1.13</td><td>N/A</td><td>N/A</td><td>N/A</td><td>1.12</td></tr><tr><td>NVDA</td><td>2.31</td><td>N/A</td><td>N/A</td><td>N/A</td><td>2.80</td><td>2.16</td><td>N/A</td><td>N/A</td><td>N/A</td><td>3.95</td></tr><tr><td>JPM</td><td>0.98</td><td>N/A</td><td>N/A</td><td>N/A</td><td>1.60</td><td>0.88</td><td>N/A</td><td>N/A</td><td>N/A</td><td>2.25</td></tr><tr><td>XOM</td><td>0.61</td><td>N/A</td><td>N/A</td><td>N/A</td><td>0.55</td><td>0.61</td><td>N/A</td><td>N/A</td><td>N/A</td><td>0.78</td></tr><tr><td>PFE</td><td>-0.19</td><td>N/A</td><td>N/A</td><td>N/A</td><td>-0.80</td><td>-0.18</td><td>N/A</td><td>N/A</td><td>N/A</td><td>-1.05</td></tr><tr><td>TSLA</td><td>2.20</td><td>N/A</td><td>N/A</td><td>N/A</td><td>1.15</td><td>2.23</td><td>N/A</td><td>N/A</td><td>N/A</td><td>1.65</td></tr></table>

Table 4: Risk-Adjusted Eficiency. Risk-adjusted metrics are undefined for strategies in the Liquida tion Equilibrium. Under the asymmetric objective, PatchTST, FD, and Raw converge to this equilibrium (realized via monotonic Always-Sell predictions), yielding N/A Sharpe/Sortino. SplineGPT yields non trivial action and measurable eficiency.

## 5.2.3 Robustness & Sensitivity

To further validate that these results are not artifacts of a specific threshold (τ = 1%) or ide alized execution, we perform comprehensive parameter sweeps using NVDA as a case study. Figure 4 demonstrates that SplineGPT maintains positive expectancy across a wide range of transaction costs (up to 30bps) and volatility thresholds. Conversely, the discrete baselines fai to activate even when the threshold τ is lowered to 0.25%, indicating a fundamental representational failure rather than a tuning mismatch.

## 5.2.4 Kinematic Responsiveness & Regime Adaptation

Finally, we examine the mechanism responsible for the superior downside protection observed in Table 3. Figure 5 visualizes a high-frequency ”whipsaw” scenario in XOM.

Crucially, while the model triggers a momentum entry (Green Triangle) on a breakout attempt, it does not blindly hold when the move fails. As visualized in Panels B and C, the rapid decay in the Acceleration token $\left( c _ { 2 } \right)$ acts as a leading indicator of trend exhaustion. The model detects this kinematic breakdown and flips to a ”Sell” state (Red Triangles) before the primary price collapse occurs. This confirms that the higher-order derivative tokens allow the policy to distinguish between healthy momentum and stalling price action faster than price lagged indicators.

![](images/f028e15f3dfb9db7d56612920d052def754ed6a96f30c3f039570db2457465cf.jpg)

![](images/505804b4d8cc079344604b3286514130996286b2b6204740da278cce12737324.jpg)  
Figure 4: Parameter Sensitivity Analysis (NVDA). (A) SplineGPT exhibits Rational Selectivity: as the volatility threshold (τ) increases (making the task harder), the model monotonically reduces its action rate, avoiding false positives. (B) Performance remains robust (Sharpe > 2.0) even under high-friction regimes (up to 30bps), validating that the alpha generated by the kinematic tokens is not merely microstructure noise arbitrage.

## 6 Discussion

The results provide evidence supporting the Physics-Informed Tokenization hypothesis in this daily-equity testbed. By benchmarking SplineGPT against a hierarchy of baselines (physics ablated (RawGPT), discrete finite-diference approximations (RawGPT-FD), and patch-based aggregation (PatchTST)), we observe a consistent qualitative separation: under an asymmet ric action objective, the baselines converge to an abstention (“Hold”) equilibrium, whereas SplineGPT learns a stable selective trading policy. This separation is central to our claim: the contribution is not only predictive accuracy, but the learnability and stability of non-trivial behavior in a regime where abstention is rational.

## 6.1 Signal Deficits and Liquidation Equilibrium

A critical finding is the behavioral collapse of the discrete baselines. While the absolute return metrics (Table 3) showed 0.0% for PatchTST, the diagnostic analysis (Figure 3) reveals this was due to a convergence to the Liquidation Equilibrium (100% Sell/Cash).

This phenomenon is a rational response to the asymmetric loss function $( w _ { s e l l } = 1 0 . 0 )$ To trigger a “Buy” action, the model’s posterior probability of a price rise must exceed the risk-adjusted threshold implied by the penalty. The collapse indicates that standard discrete tokenization imposes a signal confidence barrier: the discrete input features do not provide suficient information gain to overcome the prior probability of loss, forcing the optimizer to de fault to the global risk-minimizing state (Cash). In contrast, SplineGPT maintains a balanced active policy. This implies that the continuous kinematic tokens (particularly velocity and acceleration) successfully extract suficient actionable signal to justify risk exposure, efectively resolving the latent structure that remains opaque to discrete representations.

(B) Kinematic Token: Velocity

![](images/bb5ce181141a0829e5d5b408bf0bcecb7c1e1c9bff33cde28f00821b06ec97b3.jpg)

![](images/60820779f4455331dd75eaaf8f54c2c8f1e40a4c583e1a1b2d8e3ba36b069212.jpg)

![](images/0e86d5c56df7258384ed7378eec71bf1ec73271cf402fc960f8e4b438710c664.jpg)  
Figure 5: Kinematic Responsiveness (XOM). Rather than a failure, this false breakout illustrates the model’s rapid adaptation. Although high velocity $\left( c _ { 1 } \right)$ triggered an initial entry, the immediate rollover in acceleration $( c _ { 2 } ,$ Panel C) alerted the model to the failed breakout. The policy flipped to ”Sell” (Red Triangles) within days, protecting capital from the subsequent ≈ 10% decline. This rapid reaction speed explains the low Max Drawdown metrics relative to the market.

Cross-Sector Performance Validation (2023 - Present)  
![](images/47a6fb9e9ef6f2b8fbfa939e3177ab2e7af7eb7eef6b0d67e80708c63c12fafc.jpg)  
Figure 6: Cross-Sector Performance Validation (2023 – Present). Left Panel: Equity curves generated by the SplineGPT strategy using Active Risk Management weights. Right Panel: The Buy & Hold benchmark for the same assets. Note the strategy’s ability to smooth out volatility (TSLA) and strictly preserve capital during the PFE crash (red line), yielding a flat return profile while the asset declined 40%. The ablation baselines are omitted because they converged to the Liquidation Equilibrium, realized here by a monotonic Always-Sell action policy; see Tables $^ { 3 , }$ 4 and 5.

<table><tr><td rowspan="2">Model</td><td colspan="3">Action Distribution</td><td colspan="3">Performance</td></tr><tr><td>Buy</td><td>Sell</td><td>Hold</td><td>Action Rate</td><td>Return</td><td>Sharpe</td></tr><tr><td>PatchTST</td><td>0</td><td>762</td><td>0</td><td>100.0%</td><td>0.0%</td><td>N/A*</td></tr><tr><td>SplineGPT</td><td>294</td><td>228</td><td>240</td><td>68.5%</td><td>+579.2%</td><td>2.31</td></tr></table>

Table 5: The Liquidation Equilibrium of Discrete Tokens. We diagnosed the specific action distribution of the baseline (PatchTST) versus SplineGPT at the standard volatility threshold (τ = 1.0%) for NVDA. While SplineGPT maintains a balanced trading policy, the discrete baseline collapses to a monotonic “Always Sell” policy. This indicates that the discrete representation fails to generate suficient signal confidence to overcome the asymmetric risk penalty $( w _ { s e l l } = 1 0 . 0 )$ , forcing the model into the safest possible state (100% Cash).  
<sup>∗</sup>Sharpe ratio is undefined for static cash positions (zero volatility).

## 6.1.1 Sensitivity Sweep Analysis

To verify that the baseline failure was not merely a result of an overly conservative threshold, we performed a sensitivity sweep, lowering the volatility threshold to $\tau = 0 . 5 \%$ to encourage model activity. As shown in Table 6, the discrete baseline failed to recover even in this aggressive regime, remaining trapped in the Liquidation Equilibrium. Conversely, SplineGPT adapted successfully, increasing its action rate to 90.8% while improving its Sharpe ratio to 2.48 and better capturing the extreme bullish nature of the underlying NVDA equity by being actively invested more often. This demonstrates that the performance gap is representationa where discrete measurement tokens (versus discrete spline tokens) lack the signal-to-noise ratio required to support non-trivial decision boundaries under this standard PatchTST architecture, regardless of the threshold.

<table><tr><td rowspan="2">Condition(τ Threshold)</td><td colspan="2">PatchTST</td><td colspan="2">SplineGPT</td><td colspan="2">Divergence</td></tr><tr><td>Action Rate</td><td>Sharpe</td><td>Action Rate</td><td>Sharpe</td><td>Behavior</td><td>Outcome</td></tr><tr><td>Very Aggressive(τ = 0.25%)</td><td rowspan="4">100.0% (Sell)</td><td rowspan="4">N/A</td><td>100.0%</td><td>2.73</td><td>Active</td><td>High-Yield</td></tr><tr><td>Aggressive(τ = 0.50%)</td><td>90.8%</td><td>2.48</td><td>Active</td><td>High-Yield</td></tr><tr><td>Conservative(τ = 1.00%)</td><td>67.1%</td><td>2.20</td><td>Selective</td><td>Robust</td></tr><tr><td>Very Conservative(τ = 2.00%)</td><td>38.3%</td><td>1.74</td><td>Precision</td><td>Defensive</td></tr></table>

Table 6: Behavioral Divergence Under Sensitivity Sweep (NVDA). We varied the volatility threshold (τ) from 0.25% to 2.0%. While SplineGPT scaled its activity rationally—from 100% activity in aggressive regimes to 38% precision in conservative ones—the discrete baseline (PatchTST) remained trapped in the Liquidation Equilibrium (100% Sell) across all thresholds. In the most aggressive regime SplineGPT also exhibits a 100% action rate, but this includes a mix of both buy and sell decisions. This confirms that the baseline’s failure is a fundamental signal deficit, not a tuning artifact.

## 6.2 Explicit vs. Implicit Smoothing

PatchTST is primarily optimized for continuous forecasting under an MSE objective, whereas our downstream task is a discrete, asymmetric action classification problem; thus this comparison should be interpreted as a behavioral robustness test rather than a pure forecasting benchmark. One plausible driver is Channel Independence (CI): PatchTST processes Price and Volume as isolated univariate series, while SplineGPT couples them into a unified 9- dimensional kinematic token.

## 6.3 Objective-Induced Conservatism

In parabolic regimes (e.g., NVDA, JPM), SplineGPT yields positive returns but trails the Buy & Hold benchmark. This divergence is not a failure of prediction, but a structural consequence of the asymmetric risk objective (10). Standard trend-following optimizes for total return (R), implicitly accepting infinite drawdown risk. In contrast, our objective functions as a constrained optimization problem: maximizing R subject to strict downside penalties. Consequently, the model exhibits convexity shedding—exiting positions during intermediate mean-reversion events to reset risk exposure. This behavior characterizes the model as a “regime-switching” agent rather than a pure momentum factor, trading absolute ceiling for a higher risk-adjusted floor (Sharpe/Sortino).

## 6.4 Sensitivity to Execution Costs & Turnover

To address the practical realizability of the generated signals, we conducted a sensitivity analysis across varying transaction cost regimes ranging from 0 to 20 basis points (bps) per trade. Table 7 summarizes the risk-adjusted returns (Sharpe Ratio) and annualized turnover for each asset.

We observe high annualized turnover (36x–89x), indicating that the model has learned a short-horizon active trading policy rather than a passive holding strategy. Despite this highturnover behavior, the strategy demonstrates significant resilience to friction:

• High-Volatility Assets (NVDA, TSLA): Retain strong risk-adjusted performance (Sharpe > 0.49) even under the aggressive 20bps cost assumption, confirming that the kinematic signal captures price moves large enough to overcome substantial friction.

• Lower-Volatility Assets (XOM, JPM): Exhibit higher sensitivity. While JPM remains robust, XOM’s edge decays near the 10bps threshold. This aligns with the expectation that spline-based derivatives provide the most value in regimes where ”physical” forces (momentum, acceleration) are pronounced.

<table><tr><td>Ticker</td><td>Sharpe (0 bps)</td><td>Sharpe (5 bps)</td><td>Sharpe (10 bps)</td><td>Sharpe (20 bps)</td><td>Ann. Turnover</td></tr><tr><td>INTC</td><td>1.31</td><td>1.19</td><td>1.06</td><td>0.83</td><td>77.7x</td></tr><tr><td>NVDA</td><td>2.39</td><td>2.26</td><td>2.12</td><td>1.87</td><td>70.3x</td></tr><tr><td>JPM</td><td>1.15</td><td>1.07</td><td>0.99</td><td>0.82</td><td>36.0x</td></tr><tr><td>XOM</td><td>0.73</td><td>0.62</td><td>0.50</td><td>0.27</td><td>47.6x</td></tr><tr><td>PFE</td><td>-0.31</td><td>-0.43</td><td>-0.56</td><td>-0.83</td><td>44.6x</td></tr><tr><td>TSLA</td><td>2.05</td><td>1.90</td><td>1.76</td><td>1.48</td><td>89.7x</td></tr></table>

Table 7: Sensitivity to Transaction Costs. We report Sharpe Ratios under varying execution cost assumptions (0–20 basis points per trade). Despite high turnover (> 35x), momentum assets (NVDA, TSLA, INTC) retain strong risk-adjusted returns even at 20bps, confirming that the kinematic signal captures substantial price moves that exceed typical friction costs.

## 6.5 Mechanisms of Alpha Generation

Given the observed separation between SplineGPT and the baselines exhibiting Liquidation Equilibrium, we highlight two recurring regimes in which the learned policy appears to benefit from smooth coupled derivatives:

1. Momentum Convexity (TSLA, INTC): In high-beta assets, price action often exhibits significant convexity (acceleration) during breakouts. SplineGPT successfully captured the positive ”Jerk” (c ) associated with the onset of these rallies. By acting on the smooth derivative rather than the noisy price diference, the model entered positions earlier and held through minor volatility that may have induced selling in less sophisticated strategies.

2. Crash Avoidance (PFE): The most critical validation of the ”Active Risk Management” objective is observed in the Pfizer (PFE) case. While the asset shed over 40% of its value, the model preserved capital (MaxDD -18.8%). As visualized in Figure 7, the kinematic signatures during a crash are complex, characterized by volatility clustering in the higherorder derivatives. The model’s ability to navigate this crash demonstrates its capacity to decouple meaningful structural breaks from transient noise within the 9-dimensional manifold.

## 7 Conclusion

This work presented SplineGPT, a foundation model architecture built upon a novel continuous tokenization pipeline. By treating financial time series as physical observables of a continuous stochastic process, we were able to extract high-order kinematic states (velocity, acceleration, jerk) that are invisible to standard discrete tokenizers.

We validated the model through a rigorous, tax-aware backtest across diverse market sectors. The results demonstrate that the model has learned transferable kinematic regularities that are useful for downstream decision rules, enabling it to:

1. Kinematic Momentum Capture: Exploiting positive convexity $( c _ { 2 } > 0 )$ to identify high-beta trend initiations (e.g., TSLA).

2. Structural Break Identification: Utilizing higher-order derivative decay to detect trend exhaustion prior to price capitulation (e.g., PFE).

3. Cross-Domain Generalization: Learning robust features that transfer across uncorrelated sectors (Energy, Tech, Pharma) without asset-specific parameter tuning.

Future work will focus on scaling the model parameter count and integrating alternative data modalities, such as macroeconomic indicators, into the continuous token stream. Addi tionally, while the “Active Risk Management” strategy served as a proof of concept, alternative loss weighting schemes could be explored to train adapters for distinct risk profiles. For in stance, aggressive “risk-on” tunings may be better suited to capture the full upside of parabolic regimes, such as that exhibited by NVDA. Furthermore, the ternary Buy/Sell/Hold classifica tion employed here represents only one potential downstream application. SplineGPT can be fine-tuned for diverse objectives, including regression heads for volatility forecasting or ranking heads for cross-sectional sector rotation. Future studies will utilize action-rate and calibration diagnostics (e.g., confusion matrices and τ-sweeps) to further disentangle objective-induced abstention from representational insuficiency.

Ultimately, the implications of this work extend beyond finance. We have established a gen eralized framework for Physics-Informed Tokenization, applicable to any domain where observations are derived from continuous latent states. Unlike finance, when physical first principles exist (e.g. mechanics, electromagnetism, fluid dynamics, etc) the underlying laws are incorporated into the SDE thus modifying the spline structure adaptively for each domain. By formally encoding the governing dynamics into the tokenization layer itself, rather than relying on the model to learn them from scratch, this approach resolves a fundamental dissonance between discrete AI architectures and the continuous physical world. This paradigm shift holds the potential to significantly accelerate the development of robust, physically grounded AI systems.

![](images/2ef2e0461f8584c6622eb626bf0fffdb575d7d416e37251ad6842c546752dfcb.jpg)

![](images/143fdfe5664f1302ce5c9b3fb6c26b25cb37284913954678e3486c7bf4931a01.jpg)

![](images/0add466378d4781e9232db6b09b590cf5b9840a1afff26cabaddc05b0401df31.jpg)

![](images/c581b01f2c988f3b0813ceedb4e0f18b3d352a9988954a13334842c30119bef0.jpg)

![](images/f2975e6daf44717af7c4b417809dc06d93fa92392fdccb5ca43a0eacf33a7408.jpg)

![](images/fdcff001956dd6230441bc1e5ac77c3a94a1f90cac05781a720b701ec2c61d17.jpg)

![](images/ce14a65c84cca931d8858acbcd0ff01afa7ca42311add85bb7a825b354f8c1e6.jpg)

![](images/9048641ba9d08d598a93824fd4247308609dd53168388f269207bfcb59e9f9be.jpg)

![](images/5f57c14d611002df9099eb96cbe2a009d8502b21e777276e0d6db7c6d5fdea9c.jpg)

![](images/5fcc6786aa5d2d338212002a9033b544fd4686c97767e82e0a250f7d06dd98df.jpg)  
Figure 7: The 9-Dimensional Kinematic Manifold vs. Ground Truth. Top Panel: Standard OHLC Candle and Volume chart for Pfizer (PFE) capturing the capitulation event (Dec 13) and subse quent recovery. Lower Panels: The explicit kinematic tokens extracted by SplineGPT. Left Column: Price derivatives $\left( c _ { 0 } – c _ { 3 } \right)$ . Right Column: Volume derivatives $\left( \tilde { c } _ { 0 } – \tilde { c } _ { 4 } \right)$ . While the ”Ground Truth” (Top) shows a noisy, ambiguous bottoming process, the lower panels reveal the rich, high-frequency signal struc ture available to the model. The distinct volatility clustering in the higher-order terms (Velocity $c _ { 1 }$ and Volume Flow Acceleration $\tilde { c } _ { 2 } )$ highlights the complexity of the signal.

## References

[1] Grifin M. Kearney and Makan Fardad. “Optimization based data enrichment using stochastic dynamical system models”. In: PLOS ONE 19.9 (Sept. 2024), e0310504. doi: 10.1371/journal.pone.0310504. url: https://doi.org/10.1371/journal.pone. 0310504.

[2] Simo S¨arkk¨a and Lennart Svensson. Bayesian Filtering and Smoothing. 2nd ed. Cambridge University Press, 2023.

[3] Yuqi Nie et al. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. 2023. arXiv: 2211.14730 [cs.LG]. url: https://arxiv.org/abs/2211.14730.

[4] Yong Liu et al. iTransformer: Inverted Transformers Are Efective for Time Series Forecasting. 2024. arXiv: 2310.06625 [cs.LG]. url: https://arxiv.org/abs/2310.06625.

[5] Matthew Faw et al. “In-Context Fine-Tuning for Time-Series Foundation Models”. In: Proceedings of the 42nd International Conference on Machine Learning (ICML). 2025. url: https://icml.cc/virtual/2025/poster/43707.

[6] Zhuangwei Shi. MambaStock: Selective state space model for stock prediction. 2024. arXiv: 2402.18959 [cs.CE]. url: https://arxiv.org/abs/2402.18959.

[7] Md Atik Ahamed and Qiang Cheng. TimeMachine: A Time Series is Worth 4 Mambas for Long-term Forecasting. 2024. arXiv: 2403.09898 [cs.LG]. url: https://arxiv.org/ abs/2403.09898.

[8] Ramin Hasani et al. Liquid Time-constant Networks. 2020. arXiv: 2006.04439 [cs.LG]. url: https://arxiv.org/abs/2006.04439.

[9] Hang Liu et al. “Physics-Informed Koopman Networks for Industrial Process Time-Series Prediction”. In: Industrial & Engineering Chemistry Research 64.37 (2025), pp. 18328– 18346. doi: 10.1021/acs.iecr.5c02217.

[10] M. Raissi, P. Perdikaris, and G.E. Karniadakis. “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations”. In: Journal of Computational Physics 378 (2019), pp. 686– 707. issn: 0021-9991. doi: https://doi.org/10.1016/j.jcp.2018.10.045. url: https://www.sciencedirect.com/science/article/pii/S0021999118307125.

[11] Tibor Socaciu and Paul Pa¸scu. “Physics-Informed Neural Networks in Pricing Financial Options”. In: Broad Research in Artificial Intelligence and Neuroscience 16.2 (2025), p. 474.

[12] Justin A. Sirignano. “Deep learning for limit order books”. In: Quantitative Finance 19.4 (2019), pp. 549–570. doi: 10.1080/14697688.2018.1546053.

[13] Zihao Zhang, Stefan Zohren, and Stephen Roberts. “DeepLOB: Deep Convolutional Neural Networks for Limit Order Books”. In: IEEE Transactions on Signal Processing 67.11 (June 2019), pp. 3001–3012. issn: 1941-0476. doi: 10.1109/tsp.2019.2907260. url: http://dx.doi.org/10.1109/TSP.2019.2907260.

[14] J. O. Ramsay and B. W. Silverman. Functional Data Analysis. 2nd. New York: Springer, 2005. doi: 10.1007/b98888.

[15] Yann A. LeCun et al. “Eficient BackProp”. In: Neural Networks: Tricks of the Trade. Ed. by Gr´egoire Montavon, Genevieve B. Orr, and Klaus-Robert M¨uller. Vol. 7700. Lecture Notes in Computer Science. Springer, 2012, pp. 9–48. doi: 10.1007/978-3-642-35289- 8\_3.