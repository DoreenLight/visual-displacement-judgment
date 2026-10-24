# CVPR 2023 Open Access Repository

> **这是网页快照，不是同行评审论文。**
> 来源：<https://openaccess.thecvf.com/content/CVPR2023/html/Zhou_UDE_A_Unified_Driving_Engine_for_Human_Motion_Generation_CVPR_2023_paper.html>
> 抓取时间：2026-10-01 20:36
> HTTP 状态：200　｜　原始字节：5,987　｜　SHA256：`6eabfe2f5bdedda4`
> **用途**：本条在语料四中的证据等级为「官方一手」。引用时必须标注为网页型证据，
> 不得与同行评审论文并列。抓取日期之后页面可能变化，引用前请核对原链接。



---

- 
 CVPR 2023 Open Access Repository 
 - 

 - 

 CVPR 2023 open access 

 These CVPR 2023 papers are the Open Access versions, provided by the
 Computer Vision Foundation. 

Except for the watermark, they are identical to the accepted versions;
the final published version of the proceedings is available on IEEE Xplore.

 This material is presented to ensure timely dissemination of scholarly and technical work.
Copyright and all rights therein are retained by authors or by other copyright holders.
All persons copying this information are expected to adhere to the terms and constraints invoked 
by each author's copyright.

 Powered by: 

 Sponsored by: 

 UDE: A Unified Driving Engine for Human Motion Generation

 Zixiang Zhou, Baoyuan Wang ; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 5632-5641

 Abstract 

 Generating controllable and editable human motion sequences is a key challenge in 3D Avatar generation. It has been labor-intensive to generate and animate human motion for a long time until learning-based approaches have been developed and applied recently. However, these approaches are still task-specific or modality-specific. In this paper, we propose "UDE", the first unified driving engine that enables generating human motion sequences from natural language or audio sequences (see Fig. 1). Specifically, UDE consists of the following key components: 1) a motion quantization module based on VQVAE that represents continuous motion sequence as discrete latent code, 2) a modality-agnostic transformer encoder that learns to map modality-aware driving signals to a joint space, and 3) a unified token transformer (GPT-like) network to predict the quantized latent code index in an auto-regressive manner. 4) a diffusion motion decoder that takes as input the motion tokens and decodes them into motion sequences with high diversity. We evaluate our method on HumanML3D and AIST++ benchmarks, and the experiment results demonstrate our method achieves state-of-the-art performance.

 Related Material 

[ pdf ]
[ supp ]
[ arXiv ]
 [ bibtex ]
 @InProceedings{Zhou_2023_CVPR,
 author = {Zhou, Zixiang and Wang, Baoyuan},
 title = {UDE: A Unified Driving Engine for Human Motion Generation},
 booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
 month = {June},
 year = {2023},
 pages = {5632-5641}
}