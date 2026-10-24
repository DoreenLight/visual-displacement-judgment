# FLASHATTENTION-2: Faster Attention with Better Parallelism and Work Partitioning

Tri Dao $^{1,2}$

$^{1}$ Department of Computer Science, Princeton University $^{2}$ Department of Computer Science, Stanford University trid@cs.stanford.edu

July 18, 2023

## Abstract

Scaling Transformers to longer sequence lengths has been a major problem in the last several years, promising to improve performance in language modeling and high-resolution image understanding, as well as to unlock new applications in code, audio, and video generation. The attention layer is the main bottleneck in scaling to longer sequences, as its runtime and memory increase quadratically in the sequence length. FLASHATTENTION [5] exploits the asymmetric GPU memory hierarchy to bring significant memory saving (linear instead of quadratic) and runtime speedup (2-4× compared to optimized baselines), with no approximation. However, FLASHATTENTION is still not nearly as fast as optimized matrix-multiply (GEMM) operations, reaching only 25-40% of the theoretical maximum FLOPs/s. We observe that the inefficiency is due to suboptimal work partitioning between different thread blocks and warps on the GPU, causing either low-occupancy or unnecessary shared memory reads/writes. We propose FLASHATTENTION-2, with better work partitioning to address these issues. In particular, we (1) tweak the algorithm to reduce the number of non-matmul FLOPs (2) parallelize the attention computation, even for a single head, across different thread blocks to increase occupancy, and (3) within each thread block, distribute the work between warps to reduce communication through shared memory. These yield around 2× speedup compared to FLASHATTENTION, reaching 50-73% of the theoretical maximum FLOPs/s on A100 and getting close to the efficiency of GEMM operations. We empirically validate that when used end-to-end to train GPT-style models, FLASHATTENTION-2 reaches training speed of up to 225 TFLOPs/s per A100 GPU (72% model FLOPs utilization). $^{1}$

## 1 Introduction

Scaling up the context length of Transformers $[18]$ is a challenge, since the attention layer at their heart has runtime and memory requirements quadratic in the input sequence length. Ideally, we would like to go beyond the standard 2k sequence length limit to train models to understand books, high resolution images, and long-form videos. Just within the last year, there have been several language models with much longer context than before: GPT-4 $[12]$ with context length 32k, MosaicML's MPT with context length 65k, and Anthropic's Claude with context length 100k. Emerging use cases such as long document querying and story writing have demonstrated a need for models with such long context.

To reduce the computational requirement of attention on such long context, there have been numerous methods proposed to approximate attention $[2, 3, 4, 8, 9, 14, 19, 20]$ . Though these methods have seen some use cases, as far as we know, most large-scale training runs still use standard attention. Motivated by this, Dao et al. $[5]$ proposed to reorder the attention computation and leverages classical techniques (tiling, recomputation) to significantly speed it up and reduce memory usage from quadratic to linear in sequence length. This yields $2-4\times$ wall-clock time speedup over optimized baselines, up to $10-20\times$ memory saving, with no approximation, and as a result FLASHATTENTION has seen wide adoption in large-scale training and inference of Transformers.

However, context length increases even more, FLASHATTENTION is still not nearly as efficient as other primitives such as matrix-multiply (GEMM). In particular, while FLASHATTENTION is already 2-4× faster than a standard attention implementation, the forward pass only reaches 30-50% of the theoretical maximum FLOPs/s of the device (Fig. 5), while the backward pass is even more challenging, reaching only 25-35% of maximum throughput on A100 GPU (Fig. 6). In contrast, optimized GEMM can reach up to 80-90% of the theoretical maximum device throughput. Through careful profiling, we observe that FLASHATTENTION still has suboptimal work partitioning between different thread blocks and warps on the GPU, causing either low-occupancy or unnecessary shared memory reads/writes.

Building on FLASHATTENTION, we propose FLASHATTENTION-2 with better parallelism and work partitioning to address these challenges.

1. In Section 3.1, we tweak the algorithms to reduce the number of non-matmul FLOPs while not changing the output. While the non-matmul FLOPs only account for a small fraction of the total FLOPs, they take longer to perform as GPUs have specialized units for matrix multiply, and as a result the matmul throughput can be up to $16\times$ higher than non-matmul throughput. It is thus important to reduce non-matmul FLOPs and spend as much time as possible doing matmul FLOPs.

2. We propose to parallelize both the forward pass and backward pass along the sequence length dimension, in addition to the batch and number of heads dimension. This increases occupancy (utilization of GPU resources) in the case where the sequences are long (and hence batch size is often small).

3. Even within one block of attention computation, we partition the work between different warps of a thread block to reduce communication and shared memory reads/writes.

In Section 4, we empirically validate that FLASHATTENTION-2 yields significant speedup compared to even FLASHATTENTION. Benchmarks on different settings (with or without causal mask, different head dimensions) show that FLASHATTENTION-2 achieves around $2\times$ speedup over FLASHATTENTION, reaching up to 73% of the theoretical max throughput in the forward pass, and up to 63% of the theoretical max throughput in the backward pass. When used end-to-end to train GPT-style models, we reach training speed of up to 225 TFLOPs/s per A100 GPU.

## 2 Background

We provide some background on the performance characteristics and execution model of GPUs. We also describe the standard implementation of attention, as well as FLASHATTENTION.

## 2.1 Hardware characteristics

GPU performance characteristics. The GPU consists of compute elements (e.g., floating point arithmetic units) and a memory hierarchy. Most modern GPUs contain specialized units to accelerate matrix multiply in low-precision (e.g., Tensor Cores on Nvidia GPUs for FP16/BF16 matrix multiply). The memory hierarchy comprise of high bandwidth memory (HBM), and on-chip SRAM (aka shared memory). As an example, the A100 GPU has 40-80GB of high bandwidth memory (HBM) with bandwidth 1.5-2.0TB/s and 192KB of on-chip SRAM per each of 108 streaming multiprocessors with bandwidth estimated around 19TB/s [6, 7]. As the L2 cache is not directly controllable by the programmer, we focus on the HBM and SRAM for the purpose of this discussion.

Execution Model. GPUs have a massive number of threads to execute an operation (called a kernel). Threads are organized into thread blocks, which are scheduled to run on streaming multiprocessors (SMs). Within each thread blocks, threads are grouped into warps (a group of 32 threads). Threads within a warp can communicate by fast shuffle instructions or cooperate to perform matrix multiply. Warps within a thread block can communicate by reading from / writing to shared memory. Each kernel loads inputs from HBM to registers and SRAM, computes, then writes outputs to HBM.

## 2.2 Standard Attention Implementation

Given input sequences $Q, K, V \in R^{N \times d}$ where N is the sequence length and d is the head dimension, we want to compute the attention output $O \in R^{N \times d}$ :

$$
\mathbf {S} = \mathbf {Q} \mathbf {K} ^ {\top} \in \mathbb {R} ^ {N \times N}, \quad \mathbf {P} = \mathrm{softmax} (\mathbf {S}) \in \mathbb {R} ^ {N \times N}, \quad \mathbf {O} = \mathbf {P} \mathbf {V} \in \mathbb {R} ^ {N \times d},
$$

where softmax is applied row-wise. $^{2}$ For multi-head attention (MHA), this same computation is performed in parallel across many heads, and parallel over the batch dimension (number of input sequences in a batch).

The backward pass of attention proceeds as follows. Let $\mathbf{dO} \in \mathbb{R}^{N \times d}$ be the gradient of $\mathbf{O}$ with respect to some loss function. Then by the chain rule (aka backpropagation):

$$
\begin{array}{r l} & {\mathbf {d V} = \mathbf {P} ^ {\top} \mathbf {d O} \in \mathbb {R} ^ {N \times d}} \\ & {\mathbf {d P} = \mathbf {d O V} ^ {\top} \in \mathbb {R} ^ {N \times N}} \\ & {\mathbf {d S} = \mathrm{dsoftmax} (\mathbf {d P}) \in \mathbb {R} ^ {N \times N}} \\ & {\mathbf {d Q} = \mathbf {d S K} \in \mathbb {R} ^ {N \times d}} \\ & {\mathbf {d K} = \mathbf {Q d S} ^ {\top} \in \mathbb {R} ^ {N \times d},} \end{array}
$$

where dsoftmax is the gradient (backward pass) of softmax applied row-wise. One can work out that if $p = \text{softmax}(s)$ for some vector $s$ and $p$ , then with output gradient $dp$ , the input gradient $ds = (\text{diag}(p) - pp^{\top})dp$ .

Standard attention implementations materialize the matrices S and P to HBM, which takes $O(N^{2})$ memory. Often $N \gg d$ (typically N is on the order of 1k–8k and d is around 64–128). The standard attention implementation (1) calls the matrix multiply (GEMM) subroutine to multiply $S = QK^{\top}$ , writes the result to HBM, then (2) loads § from HBM to compute softmax and write the result P to HBM, and finally (3) calls GEMM to get O = PV. As most of the operations are bounded by memory bandwidth, the large number of memory accesses translates to slow wall-clock time. Moreover, the required memory is $O(N^{2})$ due to having to materialize S and P. Moreover, one has to save $P \in R^{N \times N}$ for the backward pass to compute the gradients.

## 2.3 FLASHATTENTION

To speed up attention on hardware accelerators such as GPU, $[5]$ proposes an algorithm to reduce the memory reads/writes while maintaining the same output (without approximation).

## 2.3.1 Forward pass

FLASHATTENTION applies the classical technique of tiling to reduce memory IOs, by (1) loading blocks of inputs from HBM to SRAM, (2) computing attention with respect to that block, and then (3) updating the output without writing the large intermediate matrices S and P to HBM. As the softmax couples entire rows or blocks of row, online softmax [11, 13] can split the attention computation into blocks, and rescale the output of each block to finally get the right result (with no approximation). By significantly reducing the amount of memory reads/writes, FLASHATTENTION yields 2-4× wall-clock speedup over optimized baseline attention implementations.

We describe the online softmax technique [11] and how it is used in attention [13]. For simplicity, consider just one row block of the attention matrix $\mathbf{S}$ , of the form $\left[\mathbf{S}^{(1)} - \mathbf{S}^{(2)}\right]$ for some matrices $\mathbf{S}^{(1)}, \mathbf{S}^{(2)} \in \mathbb{R}^{B_r \times B_c}$ , where $B_r$ and $B_c$ are the row and column block sizes. We want to compute softmax of this row block and multiply with the value, of the form $\left[\begin{array}{c}\mathbf{V}^{(1)}\\ \mathbf{V}^{(2)}\end{array}\right]$ for some matrices $\mathbf{V}^{(1)}, \mathbf{V}^{(2)} \in \mathbb{R}^{B_c \times d}$ . Standard softmax would

compute:

$$
\begin{array}{l} m = \max (\operatorname{rowmax} (\mathbf {S} ^ {(1)}), \operatorname{rowmax} (\mathbf {S} ^ {(2)})) \in \mathbb {R} ^ {B _ {r}} \\ \ell = \operatorname{rowsum} (e ^ {\mathbf {S} ^ {(1)} - m}) + \operatorname{rowsum} (e ^ {\mathbf {S} ^ {(2)} - m}) \in \mathbb {R} ^ {B _ {r}} \\ \mathbf {P} = \left[ \begin{array}{c c} \mathbf {P} ^ {(1)} & \mathbf {P} ^ {(2)} \end{array} \right] = \operatorname{diag} (\ell) ^ {- 1} \left[ \begin{array}{c c} e ^ {\mathbf {S} ^ {(1)} - m} & e ^ {\mathbf {S} ^ {(2)} - m} \end{array} \right] \in \mathbb {R} ^ {B _ {r} \times 2 B _ {c}} \\ \mathbf {O} = \left[ \begin{array}{c c} \mathbf {P} ^ {(1)} & \mathbf {P} ^ {(2)} \end{array} \right] \left[ \begin{array}{c} \mathbf {V} ^ {(1)} \\ \mathbf {V} ^ {(2)} \end{array} \right] = \operatorname{diag} (\ell) ^ {- 1} e ^ {\mathbf {S} ^ {(1)} - m} \mathbf {V} ^ {(1)} + e ^ {\mathbf {S} ^ {(2)} - m} \mathbf {V} ^ {(2)} \in \mathbb {R} ^ {B _ {r} \times d}. \end{array}
$$

Online softmax instead computes “local” softmax with respect to each block and rescale to get the right output at the end:

$$
\begin{array}{r l} & m ^ {(1)} = \mathrm{rowmax} (\mathbf {S} ^ {(1)}) \in \mathbb {R} ^ {B _ {r}} \\ & \ell^ {(1)} = \mathrm{rowsum} (e ^ {\mathbf {S} ^ {(1)} - m ^ {(1)}}) \in \mathbb {R} ^ {B _ {r}} \\ & \tilde {\mathbf {P}} ^ {(1)} = \mathrm{diag} (\ell^ {(1)}) ^ {- 1} e ^ {\mathbf {S} ^ {(1)} - m ^ {(1)}} \in \mathbb {R} ^ {B _ {r} \times B _ {c}} \\ & \mathbf {O} ^ {(1)} = \tilde {\mathbf {P}} ^ {(1)} \mathbf {V} ^ {(1)} = \mathrm{diag} (\ell^ {(1)}) ^ {- 1} e ^ {\mathbf {S} ^ {(1)} - m ^ {(1)}} \mathbf {V} ^ {(1)} \in \mathbb {R} ^ {B _ {r} \times d} \\ & m ^ {(2)} = \max (m ^ {(1)}, \mathrm{rowmax} (\mathbf {S} ^ {(2)})) = m \\ & \ell^ {(2)} = e ^ {m ^ {(1)} - m ^ {(2)}} \ell^ {(1)} + \mathrm{rowsum} (e ^ {\mathbf {S} ^ {(2)} - m ^ {(2)}}) = \mathrm{rowsum} (e ^ {\mathbf {S} ^ {(1)} - m}) + \mathrm{rowsum} (e ^ {\mathbf {S} ^ {(2)} - m}) = \ell \\ & \tilde {\mathbf {P}} ^ {(2)} = \mathrm{diag} (\ell^ {(2)}) ^ {- 1} e ^ {\mathbf {S} ^ {(2)} - m ^ {(2)}} \\ & \mathbf {O} ^ {(2)} = \mathrm{diag} (\ell^ {(1)} / \ell^ {(2)}) ^ {- 1} \mathbf {O} ^ {(1)} + \tilde {\mathbf {P}} ^ {(2)} \mathbf {V} ^ {(2)} = \mathrm{diag} (\ell^ {(2)}) ^ {- 1} e ^ {s ^ {(1)} - m} \mathbf {V} ^ {(1)} + \mathrm{diag} (\ell^ {(2)}) ^ {- 1} e ^ {s ^ {(2)} - m} \mathbf {V} ^ {(2)} = \mathbf {0}. \end{array}
$$

We show how FLASHATTENTION uses online softmax to enable tiling (Fig. 1) to reduce memory reads/writes.

![](images/c7f324304008b78f82e5eb1741ba291d5afe0334d99f22265cfe9b1c8111b467.jpg)  
Figure 1: Diagram of how FLASHATTENTION forward pass is performed, when the key K is partitioned into two blocks and the value V is also partitioned into two blocks. By computing attention with respect to each block and rescaling the output, we get the right answer at the end, while avoiding expensive memory reads/writes of the intermediate matrices S and P. We simplify the diagram, omitting the step in softmax that subtracts each element by the row-wise max.

## 2.3.2 Backward pass

In the backward pass, by re-computing the values of the attention matrices S and P once blocks of inputs Q, K, V are already loaded to SRAM, FLASHATTENTION avoids having to store large intermediate values. By not having to save the large matrices S and P of size $N \times N$ , FLASHATTENTION yields 10-20× memory saving depending on sequence length (memory required in linear in sequence length N instead of quadratic). The backward pass also achieves 2-4× wall-clock speedup due to reduce memory reads/writes.

The backward pass applies tiling to the equations in Section 2.2. Though the backward pass is simpler than the forward pass conceptually (there is no softmax rescaling), the implementation is significantly more involved. This is because there are more values to be kept in SRAM to perform 5 matrix multiples in the backward pass, compared to just 2 matrix multiples in the forward pass.

## 3 FLASHATTENTION-2: Algorithm, Parallelism, and Work Partitioning

We describe the FLASHATTENTION-2 algorithm, which includes several tweaks to FLASHATTENTION to reduce the number of non-matmul FLOPs. We then describe how to parallelize the computation on different thread blocks to make full use the GPU resources. Finally we describe we partition the work between different warps within one thread block to reduce the amount of shared memory access. These improvements lead to 2-3× speedup as validated in Section 4.

## 3.1 Algorithm

We tweak the algorithm from FLASHATTENTION to reduce the number of non-matmul FLOPs. This is because modern GPUs have specialized compute units (e.g., Tensor Cores on Nvidia GPUs) that makes matmul much faster. As an example, the A100 GPU has a max theoretical throughput of 312 TFLOPs/s of FP16/BF16 matmul, but only 19.5 TFLOPs/s of non-matmul FP32. Another way to think about this is that each non-matmul FLOP is $16\times$ more expensive than a matmul FLOP. To maintain high throughput (e.g., more than 50% of the maximum theoretical TFLOPs/s), we want to spend as much time on matmul FLOPs as possible.

## 3.1.1 Forward pass

We revisit the online softmax trick as shown in Section 2.3 and make two minor tweaks to reduce non-matmul FLOPs:

1. We do not have to rescale both terms of the output update by $\mathrm{diag}(\ell^{(2)})^{-1}$ :

$$
\mathbf {O} ^ {(2)} = \mathrm{diag} (\ell^ {(1)} / \ell^ {(2)}) ^ {- 1} \mathbf {O} ^ {(1)} + \mathrm{diag} (\ell^ {(2)}) ^ {- 1} e ^ {\mathbf {S} ^ {(2)} - m ^ {(2)}} \mathbf {V} ^ {(2)}.
$$

We can instead maintain an “un-scaled” version of $\mathbf{O}^{(2)}$ and keep around the statistics $\ell^{(2)}$ :

$$
\tilde {\mathbf {O}} ^ {(2)} = \mathrm{diag} (\ell^ {(1)}) ^ {- 1} \mathbf {O} ^ {(1)} + e ^ {\mathbf {S} ^ {(2)} - m ^ {(2)}} \mathbf {V} ^ {(2)}.
$$

Only at the every end of the loop do we scale the final $\tilde{\mathbf{O}}^{(\mathrm{last})}$ by $\mathrm{diag}(\ell^{(\mathrm{last})})^{-1}$ to get the right output.

2. We do not have to save both the max $m^{(j)}$ and the sum of exponentials $\ell^{(j)}$ for the backward pass. We only need to store the logsumexp $L^{(j)} = m^{(j)} + \log (\ell^{(j)})$ .

In the simple case of 2 blocks in Section 2.3, the online softmax trick now becomes:

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
$\begin{array}{rl} &amp; m^{(1)} = \mathrm{rowmax}(\mathbf{S}^{(1)})\in \mathbb{R}^{B_r}\\ &amp; \ell^{(1)} = \mathrm{rowsum}(e^{\mathbf{S}^{(1)} - m^{(1)}})\in \mathbb{R}^{B_r}\\ &amp; \tilde{\mathbf{O}}^{(1)} = e^{\mathbf{S}^{(1)} - m^{(1)}}\mathbf{V}^{(1)}\in \mathbb{R}^{B_r\times d}\\ &amp; m^{(2)} = \max (m^{(1)},\mathrm{rowmax}(\mathbf{S}^{(2)})) = m\\ &amp; \ell^{(2)} = e^{m^{(1)} - m^{(2)}}\ell^{(1)} + \mathrm{rowsum}(e^{\mathbf{S}^{(2)} - m^{(2)}}) = \mathrm{rowsum}(e^{\mathbf{S}^{(1)} - m}) + \mathrm{rowsum}(e^{\mathbf{S}^{(2)} - m}) = \ell \\ &amp; \tilde{\mathbf{P}}^{(2)} = \mathrm{diag}(\ell^{(2)})^{-1}e^{\mathbf{S}^{(2)} - m^{(2)}}\\ &amp; \tilde{\mathbf{O}}^{(2)} = \mathrm{diag}(e^{m^{(1)} - m^{(2)}})^{-1}\tilde{\mathbf{O}}^{(1)} + e^{\mathbf{S}^{(2)} - m^{(2)}}\mathbf{V}^{(2)} = e^{s^{(1)} - m}\mathbf{V}^{(1)} + e^{s^{(2)} - m}\mathbf{V}^{(2)}\\ &amp; \mathbf{O}^{(2)} = \mathrm{diag}(\ell^{(2)})^{-1}\tilde{\mathbf{O}}^{(2)} = \mathbf{O}. \end{array}$
</div>

We describe the full FLASHATTENTION-2 forward pass in Algorithm 1.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 FLASHATTENTION-2 forward pass

Require: Matrices Q, K, V ∈ R^{N×d} in HBM, block sizes B_c, B_r.

1: Divide Q into T_r = $\left[\frac{N}{B_r}\right]$ blocks Q1, ..., Q_Tr of size Br × d each, and divide K, V in to T_c = $\left[\frac{N}{B_c}\right]$ blocks K1, ..., K_Tc and V1, ..., V_Tc, of size Br × d each.

2: Divide the output O ∈ R^{N×d} into Tr blocks O_i, ..., O_Tr of size Br × d each, and divide the logsumexp L into Tr blocks Li, ..., LT_r of size Br each.

3: for 1 ≤ i ≤ Tr do

4: Load Qi from HBM to on-chip SRAM.

5: On chip, initialize O_i^{(0)} = (0)_{Br×d} ∈ R^{Br×d}, \ell_i^{(0)} = (0)_{Br} ∈ R^{Br}, m_i^{(0)} = (-∞)_{Br} ∈ R^{Br}.

6: for 1 ≤ j ≤ T_c do

7: Load K_j, V_j from HBM to on-chip SRAM.

8: On chip, compute S_i^{(j)} = Q_i K_j^T ∈ R^{Br×B_c}.

9: On chip, compute m_i^{(j)} = max(m_i^{(j-1)}, rowmax(S_i^{(j)})) ∈ R^{Br}, \tilde{P}_i^{(j)} = exp(S_i^{(j)} - m_i^{(j)}) ∈ R^{Br×B_c} (pointwise), \ell_i^{(j)} = e^{m_i^{j-1}-m_i^{(j)}} \ell_i^{(j-1)} + rowsum(\tilde{P}_i^{(j)}) ∈ R^{Br}.

10: On chip, compute O_i^{(j)} = diag(e^{m_i^{(j-1)}-m_i^{(j)}})^{-1}O_i^{(j-1)} + \tilde{P}_i^{(j)}V_j.

11: end for

12: On chip, compute O_i = diag(\ell_i^{(T_c)})^{-1}O_i^{(T_c)}.

13: On chip, compute Li = m_i^{(T_c)} + log(\ell_i^{(T_c)}).

14: Write O_i to HBM as the i-th block of O.

15: Write Li to HBM as the i-th block of L.

16: end for

17: Return the output O and the logsumexp L.
</div>

Causal masking. One common use case of attention is in auto-regressive language modeling, where we need to apply a causal mask to the attention matrix S (i.e., any entry $S_{ij}$ with j > i is set to $-\infty$ ).

1. As FLASHATTENTION and FLASHATTENTION-2 already operate by blocks, for any blocks where all the column indices are more than the row indices (approximately half of the blocks for large sequence length), we can skip the computation of that block. This leads to around $1.7-1.8\times$ speedup compared to attention without the causal mask.

2. We do not need to apply the causal mask for blocks whose row indices are guaranteed to be strictly less than the column indices. This means that for each row, we only need apply causal mask to 1 block (assuming square block).

Correctness, runtime, and memory requirement. As with FLASHATTENTION, Algorithm 1 returns the correct output $\mathbf{O} = \mathrm{softmax}(\mathbf{Q}\mathbf{K}^{\top})\mathbf{V}$ (with no approximation), using $O(N^{2}d)$ FLOPs and requires $O(N)$ additional memory beyond inputs and output (to store the logsumexp $L$ ). The proof is almost the same as the proof of Dao et al. [5, Theorem 1], so we omit it here.

## 3.1.2 Backward pass

The backward pass of FLASHATTENTION-2 is almost the same as that of FLASHATTENTION. We make a minor tweak to only use the row-wise logsumexp L instead of both the row-wise max and row-wise sum of exponentials in the softmax. We include the backward pass description in Algorithm 2 for completeness.

```txt
Algorithm 2 FLASHATTENTION-2 Backward Pass

Require: Matrices Q, K, V, O, dO ∈ R^{N×d} in HBM, vector L ∈ R^N in HBM, block sizes B_c, B_r.

1: Divide Q into T_r = [N/B_r] blocks Q_1, ..., Q_{T_r} of size B_r × d each, and divide K, V in to T_c = [N/B_c] blocks K_1, ..., K_{T_c} and V_1, ..., V_{T_c}, of size B_c × d each.

2: Divide O into T_r blocks O_i, ..., O_{T_r} of size B_r × d each, divide dO into T_r blocks dO_i, ..., dO_{T_r} of size B_r × d each, and divide L into T_r blocks L_i, ..., L_{T_r} of size B_r each.

3: Initialize dQ = (0)_{N×d} in HBM and divide it into T_r blocks dQ_1, ..., dQ_{T_r} of size B_r × d each. Divide dK, dV ∈ R^{N×d} in to T_c blocks dK_1, ..., dK_{T_c} and dV_1, ..., dV_{T_c}, of size B_c × d each.

4: Compute D = rowsum(dO ∘ O) ∈ R^d (pointwise multiply), write D to HBM and divide it into T_r blocks D_1, ..., D_{T_r} of size B_r each.

5: for 1 ≤ j ≤ T_c do

6: Load K_j, V_j from HBM to on-chip SRAM.

7: Initialize dK_j = (0)_{B_c×d}, dV_j = (0)_{B_c×d} on SRAM.

8: for 1 ≤ i ≤ T_r do

9: Load Q_i, O_i, dO_i, dQ_i, L_i, D_i from HBM to on-chip SRAM.

10: On chip, compute S_i^{(j)} = Q_i K_j^T ∈ R^{B_r × B_c}

11: On chip, compute P_i^{(j)} = exp(S_{ij} - L_i) ∈ R^{B_r × B_c}

12: On chip, compute dV_j ← dV_j + (P_i^{(j)})^T dO_i ∈ R^{B_c × d}

13: On chip, compute dP_i^{(j)} = dO_i V_j^T ∈ R^{B_r × B_c}

14: On chip, compute dS_i^{(j)} = P_i^{(j)} ∘ (dP_i^{(j)} - D_i) ∈ R^{B_r × B_c}

15: Load dQ_i from HBM to SRAM, then on chip, update dQ_i ← dQ_i + dS_i^{(j)} K_j ∈ R^{B_r × d}, and write back to HBM.

16: On chip, compute dK_j ← dK_j + dS_i^{(j)ᵀ} Q_i ∈ R^{B_c × d}

17: end for

18: Write dK_j, dV_j to HBM.

19: end for

20: Return dQ, dK, dV.
```

Multi-query attention and grouped-query attention. Multi-query attention (MQA) [15] and grouped-query attention (GQA) [1] are variants of attention where multiple heads of query attend to the same head of key and value, in order to reduce the size of KV cache during inference. Instead of having to duplicate the key and value heads for the computation, we implicitly manipulate the indices into the head to perform the same computation. In the backward pass, we need to sum the gradients dK and dV across different heads that were implicitly duplicated.

## 3.2 Parallelism

The first version of FLASHATTENTION parallelizes over batch size and number of heads. We use 1 thread block to process one attention head, and there are overall batch size · number of heads thread blocks. Each thread block is scheduled to run on a streaming multiprocessor (SM), and there are 108 of these SMs on an A100 GPU for example. This scheduling is efficient when this number is large (say $\geq$ 80), since we can effectively use almost all of the compute resources on the GPU.

In the case of long sequences (which usually means small batch sizes or small number of heads), to make better use of the multiprocessors on the GPU, we now additionally parallelize over the sequence length dimension. This results in significant speedup for this regime.

Forward pass. We see that the outer loop (over sequence length) is embarrassingly parallel, and we schedule them on different thread blocks that do not need to communicate with each other. We also parallelize over the batch dimension and number of heads dimension, as done in FLASHATTENTION. The increased parallelism over sequence length helps improve occupancy (fraction of GPU resources being used) when the batch size and number of heads are small, leading to speedup in this case.

These ideas of swapping the order of the loop (outer loop over row blocks and inner loop over column blocks, instead of the other way round in the original FLASHATTENTION paper), as well as parallelizing over the sequence length dimension were first suggested and implemented by Phil Tillet in the Triton [17] implementation. $^{3}$

Backward pass. Notice that the only shared computation between different column blocks is in update dQ in Algorithm 2, where we need to load $dQ_{i}$ from HBM to SRAM, then on chip, update $dQ_{i} \leftarrow dQ_{i} + dS_{i}^{(j)}K_{j}$ , and write back to HBM. We thus parallelize over the sequence length dimension as well, and schedule 1 thread block for each column block of the backward pass. We use atomic adds to communicate between different thread blocks to update dQ.

We describe the parallelization scheme in Fig. 2.

Forward pass  
![](images/3fe6cc62665b75eef6105277fc30a4563709adddd2cfd2693542a7be1c0765f7.jpg)

Backward pass  
![](images/f8db8f8c965446c10f9b92b80ea1dfca2b10c074cae86a31a3ccccbb26931029.jpg)  
Figure 2: In the forward pass (left), we parallelize the workers (thread blocks) where each worker takes care of a block of rows of the attention matrix. In the backward pass (right), each worker takes care of a block of columns of the attention matrix.

## 3.3 Work Partitioning Between Warps

As Section 3.2 describe how we schedule thread blocks, even within each thread block, we also have to decide how to partition the work between different warps. We typically use 4 or 8 warps per thread block, and the partitioning is described in Fig. 3.

Forward pass. For each block, FLASHATTENTION splits K and V across 4 warps while keeping Q accessible by all warps. Each warp multiplies to get a slice of $QK^{\top}$ , then they need to multiply with a slice of V and communicate to add up the result. This is referred to as the “split-K” scheme. However, this is inefficient since all warps need to write their intermediate results out to shared memory, synchronize, then add up the intermediate results. These shared memory reads/writes slow down the forward pass in FLASHATTENTION.

In FLASHATTENTION-2, we instead split Q across 4 warps while keeping K and V accessible by all warps. After each warp performs matrix multiply to get a slice of $QK^{\top}$ , they just need to multiply with their shared slice of V to get their corresponding slice of the output. There is no need for communication between warps. The reduction in shared memory reads/writes yields speedup (Section 4).

![](images/d60b4c71053b794873cb410e01466b3f1507f63f0267e4148c1784c9ef6579d8.jpg)  
Figure 3: Work partitioning between different warps in the forward pass

Backward pass. Similarly for the backward pass, we choose to partition the warps to avoid the “split-K” scheme. However, it still requires some synchronization due to the more complicated dependency between all the different inputs and gradients Q, K, V, O, dO, dQ, dK, dV. Nevertheless, avoiding “split-K” reduces shared memory reads/writes and again yields speedup (Section 4).

Tuning block sizes Increasing block sizes generally reduces shared memory loads/stores, but increases the number of registers required and the total amount of shared memory. Past a certain block size, register spilling causes significant slowdown, or the amount of shared memory required is larger than what the GPU has available, and the kernel cannot run at all. Typically we choose blocks of size $\{64,128\}\times\{64,128\}$ , depending on the head dimension d and the device shared memory size.

We manually tune for each head dimensions since there are essentially only 4 choices for block sizes, but this could benefit from auto-tuning to avoid this manual labor. We leave this to future work.

## 4 Empirical Validation

We evaluate the impact of using FLASHATTENTION-2 to train Transformer models.

\- Benchmarking attention. We measure the runtime of FLASHATTENTION-2 across different sequence lengths and compare it to a standard implementation in PyTorch, FLASHATTENTION, and FLASHATTENTION in Triton. We confirm that FLASHATTENTION-2 is 1.7-3.0× faster than FLASHATTENTION, 1.3-2.5× faster than FLASHATTENTION in Triton, and 3-10× faster than a standard attention implementation.

FLASHATTENTION-2 reaches up to 230 TFLOPs/s, 73% of the theoretical maximum TFLOPs/s on A100 GPUs.

\- End-to-end training speed When used end-to-end to train GPT-style models of size 1.3B and 2.7B on sequence lengths either 2k or 8k, FLASHATTENTION-2 yields up to 1.3× speedup compared to FLASHATTENTION and 2.8× speedup compared to a baseline without FLASHATTENTION. FLASHATTENTION-2 reaches up to 225 TFLOPs/s (72% model FLOPs utilization) per A100 GPU.

## 4.1 Benchmarking Attention

We measure the runtime of different attention methods on an A100 80GB SXM4 GPU for different settings (without / with causal mask, head dimension 64 or 128). We report the results in Fig. 4, Fig. 5 and Fig. 6, showing that FLASHATTENTION-2 is around 2× faster than FLASHATTENTION and FLASHATTENTION in xformers (the “cutlass” implementation). FLASHATTENTION-2 is around 1.3-1.5× faster than FLASHATTENTION in Triton in the forward pass and around 2× faster in the backward pass. Compared to a standard attention implementation in PyTorch, FLASHATTENTION-2 can be up to 10× faster.

Benchmark setting: we vary the sequence length from 512, 1k, ..., 16k, and set batch size so that the total number of tokens is 16k. We set hidden dimension to 2048, and head dimension to be either 64 or 128 (i.e., 32 heads or 16 heads). To calculate the FLOPs of the forward pass, we use:

$4 \cdot \text{seqlen}^{2} \cdot \text{head dimension} \cdot \text{number of heads}.$

With causal mask, we divide this number by 2 to account for the fact that approximately only half of the entries are calculated. To get the FLOPs of the backward pass, we multiply the forward pass FLOPs by 2.5 (since there are 2 matmuls in the forward pass and 5 matmuls in the backward pass, due to recomputation).

Attention forward + backward speed (A100 80GB SXM4)  
Attention forward + backward speed (A100 80GB SXM4)  
![](images/d910a199fe75336d8d52d73f27b68ed0d1c09764b84f0912f5c1f13f9bab5a58.jpg)

![](images/a4fef772e29dc239227ff26d795fb0ea50824086718d0ce0e9319e820eadce78.jpg)

(a) Without causal mask, head dimension 64  
![](images/3b9f4c2169032102fc6b1083814fd1fd32a7639a46c33050c17caacf9112c228.jpg)  
(c) With causal mask, head dimension 64

(b) Without causal mask, head dimension 128  
![](images/85afa9e49e9b23e860cf980688c53b31316d109016fac5240ce629560cf0fe34.jpg)  
(d) With causal mask, head dimension 128  
Figure 4: Attention forward + backward speed on A100 GPU

Attention forward speed (A100 80GB SXM4)  
![](images/535c482c30395d28568ff7b694962319cdc03fe6c6ec8ccab3a3cea7a4815808.jpg)  
(a) Without causal mask, head dimension 64

Attention forward speed (A100 80GB SXM4)  
![](images/29a37c9a2586c8b565288ce9436bd44577317bd18e9e9a54ca8586955f99bfbb.jpg)  
(b) Without causal mask, head dimension 128

Attention forward speed (A100 80GB SXM4)  
![](images/d3ca612a958268982bce93ce6fd4764fc026ddecb9607091dc2c1befd9d19e27.jpg)  
(c) With causal mask, head dimension 64

Attention forward speed (A100 80GB SXM4)  
![](images/5e59cc7261ecd5033d280345345575a1c81eb7d16c632c5ae25ace44190c6ad9.jpg)  
(d) With causal mask, head dimension 128  
Figure 5: Attention forward speed on A100 GPU

Just running the same implementation on H100 GPUs (using no special instructions to make use of new features such as TMA and 4th-gen Tensor Cores), we obtain up to 335 TFLOPs/s (Fig. 7). We expect that by using new instructions, we can obtain another 1.5x-2x speedup on H100 GPUs. We leave that to future work.

## 4.2 End-to-end Performance

We measure the training throughput of GPT-style models with either 1.3B or 2.7B parameters, on 8×A100 80GB SXM. As shown in Table 1, FLASHATTENTION-2 yields 2.8× speedup compared to a baseline without FlashAttention and 1.3× speedup compared to FLASHATTENTION-2, reaching up to 225 TFLOPs/s per A100 GPU.

Note that we calculate the FLOPs by the formula, following Megatron-LM [16] (and many other papers and libraries):

$$
6 \cdot \mathrm{seqlen} \cdot \mathrm{numberofparams} + 1 2 \cdot \mathrm{numberoflayers} \cdot \mathrm{hiddendim} \cdot \mathrm{seqlen} ^ {2}.
$$

The first term accounts for the FLOPs due to weight–input multiplication, and the second term accounts for the FLOPs due to attention. However, one can argue that the second term should be halved, as with causal mask we only need to compute approximately half the number of elements in attention. We choose to follow the formula from the literature (without dividing the attention FLOPs by 2) for consistency.

## 5 Discussion and Future Directions

FLASHATTENTION-2 is $2\times$ faster than FLASHATTENTION, which means that we can train models with 16k longer context for the same price as previously training a 8k context model. We are excited about how this can be used to understand long books and reports, high resolution images, audio and video. FLASHATTENTION-2 will also speed up training, finetuning, and inference of existing models.

Attention backward speed (A100 80GB SXM4)  
![](images/6be0b2ba7112f14d7e3211a59177fa3ed024d0572a0ce124941d4d2c2909557e.jpg)  
(a) Without causal mask, head dimension 64

Attention backward speed (A100 80GB SXM4)  
![](images/726b13ab9105c06e45aeda85d51a5f7e451abe68461e118b653e1c5a125cab67.jpg)  
(b) Without causal mask, head dimension 128

Attention backward speed (A100 80GB SXM4)  
![](images/b639bf9ca5ca43ad8baa22c51c0f2556b9c2fba18ab64f95cdcbbdafc37afcf5.jpg)  
(c) With causal mask, head dimension 64

Attention backward speed (A100 80GB SXM4)  
![](images/f0ce3e64d7fbd53b3872aabbe10cfb9b6a8bfd2516e4cdf139341ba4c0a8140d.jpg)  
(d) With causal mask, head dimension 128  
Figure 6: Attention backward speed on A100 GPU

Table 1: Training speed (TFLOPs/s/GPU) of GPT-style models on 8×A100 GPUs. FLASHATTENTION-2 reaches up to 225 TFLOPs/s (72% model FLOPs utilization). We compare against a baseline running without FLASHATTENTION.

<table><tr><td>Model</td><td>Without FLASHATTENTION</td><td>FLASHATTENTION</td><td>FLASHATTENTION-2</td></tr><tr><td>GPT3-1.3B 2k context</td><td>142 TFLOPs/s</td><td>189 TFLOPs/s</td><td>196 TFLOPs/s</td></tr><tr><td>GPT3-1.3B 8k context</td><td>72 TFLOPs/s</td><td>170 TFLOPs/s</td><td>220 TFLOPs/s</td></tr><tr><td>GPT3-2.7B 2k context</td><td>149 TFLOPs/s</td><td>189 TFLOPs/s</td><td>205 TFLOPs/s</td></tr><tr><td>GPT3-2.7B 8k context</td><td>80 TFLOPs/s</td><td>175 TFLOPs/s</td><td>225 TFLOPs/s</td></tr></table>

In the near future, we plan to collaborate with researchers and engineers to make FlashAttention widely applicable in different kinds of devices (e.g., H100 GPUs, AMD GPUs), as well as new data types such as FP8. As an immediate next step, we plan to optimize FlashAttention-2 for H100 GPUs to use new hardware features (TMA, 4th-gen Tensor Cores, fp8). Combining the low-level optimizations in FlashAttention-2 with high-level algorithmic changes (e.g., local, dilated, block-sparse attention) could allow us to train AI models with much longer context. We are also excited to work with compiler researchers to make these optimization techniques easily programmable.

Attention forward + backward speed (H100 80GB SXM5)  
![](images/7a0adf5fcd880253fc6d6b4634dd50e7c0152c9e35e032f47ea597d4b30c7940.jpg)  
(a) Without causal mask, head dimension 64

Attention forward + backward speed (H100 80GB SXM5)  
![](images/58f06adef34c30c66635a67d765aa25f49ae4f4c5bbdba8ce9e8c9d185c4dac2.jpg)  
(b) Without causal mask, head dimension 128

Attention forward + backward speed (H100 80GB SXM5)  
![](images/4a2d0bb58161e0c09069d1f735c592ad15877cd440381a87b7cb593a5423aec3.jpg)  
(c) With causal mask, head dimension 64

Attention forward + backward speed (H100 80GB SXM5)  
![](images/cf09a598a455a53f7b872c57ca6e6f1e361368a407e7896d37dbde5aaddc5c96.jpg)  
(d) With causal mask, head dimension 128  
Figure 7: Attention forward + backward speed on H100 GPU

## Acknowledgments

We thank Phil Tillet and Daniel Haziza, who have implemented versions of FLASHATTENTION in Triton $[17]$ and the xformers library $[10]$ . FLASHATTENTION-2 was motivated by exchange of ideas between different ways that attention could be implemented. We are grateful to the Nvidia CUTLASS team (especially Vijay Thakkar, Cris Cecka, Haicheng Wu, and Andrew Kerr) for their CUTLASS library, in particular the CUTLASS 3.x release, which provides clean abstractions and powerful building blocks for the implementation of FLASHATTENTION-2. We thank Driss Guessous for integrating FLASHATTENTION to PyTorch. FLASHATTENTION-2 has benefited from helpful discussions with Phil Wang, Markus Rabe, James Bradbury, Young-Jun Ko, Julien Launay, Daniel Hesslow, Michaël Benesty, Horace He, Ashish Vaswani, and Erich Elsen. Thanks for Stanford CRFM and Stanford NLP for the compute support. We thank Dan Fu and Christopher Ré for their collaboration, constructive feedback, and constant encouragement on this line of work of designing hardware-efficient algorithms. We thank Albert Gu and Beidi Chen for their helpful suggestions on early drafts of this technical report.

## References

[1] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. arXiv preprint arXiv:2305.13245, 2023.

[2] Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020.

[3] Beidi Chen, Tri Dao, Eric Winsor, Zhao Song, Atri Rudra, and Christopher Ré. Scatterbrain: Unifying sparse and low-rank attention. In Advances in Neural Information Processing Systems (NeurIPS), 2021.

[4] Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Quincy Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. Rethinking attention with performers. In International Conference on Learning Representations (ICLR), 2020.

[5] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems, 2022.

[6] Zhe Jia and Peter Van Sandt. Dissecting the Ampere GPU architecture via microbenchmarking. GPU Technology Conference, 2021.

[7] Zhe Jia, Marco Maggioni, Benjamin Staiger, and Daniele P Scarpazza. Dissecting the nvidia Volta GPU architecture via microbenchmarking. arXiv preprint arXiv:1804.06826, 2018.

[8] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are RNNs: Fast autoregressive transformers with linear attention. In International Conference on Machine Learning, pages 5156–5165. PMLR, 2020.

[9] Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. In The International Conference on Machine Learning (ICML), 2020.

[10] Benjamin Lefaudeux, Francisco Massa, Diana Liskovich, Wenhan Xiong, Vittorio Caggiano, Sean Naren, Min Xu, Jieru Hu, Marta Tintore, Susan Zhang, Patrick Labatut, and Daniel Haziza. xformers: A modular and hackable transformer modelling library. https://github.com/facebookresearch/xformers, 2022.

[11] Maxim Milakov and Natalia Gimelshein. Online normalizer calculation for softmax. arXiv preprint arXiv:1805.02867, 2018.

[12] OpenAI. Gpt-4 technical report. ArXiv, abs/2303.08774, 2023.

[13] Markus N Rabe and Charles Staats. Self-attention does not need $O(n^{2})$ memory. arXiv preprint arXiv:2112.05682, 2021.

[14] Aurko Roy, Mohammad Saffar, Ashish Vaswani, and David Grangier. Efficient content-based sparse attention with routing transformers. Transactions of the Association for Computational Linguistics, 9:53–68, 2021.

[15] Noam Shazeer. Fast transformer decoding: One write-head is all you need. arXiv preprint arXiv:1911.02150, 2019.

[16] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-LM: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019.

[17] Philippe Tillet, Hsiang-Tsung Kung, and David Cox. Triton: an intermediate language and compiler for tiled neural network computations. In Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, pages 10–19, 2019.

[18] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[19] Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768, 2020.

[20] Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. Big bird: Transformers for longer sequences. Advances in Neural Information Processing Systems, 33, 2020.