#!/usr/bin/env python3
"""29 篇论文的 PDF → 英文抽取 映射清单（01_参考论文/，不含 09_章隆彬_合作团队）。

每项：
  name   论文目录名（= PDF stem = 目标 _全文翻译.md 的 stem）
  pdf    相对 01_参考论文/ 的 PDF 路径
  en_md  已有的 MinerU 英文抽取 md 路径（None = 尚未抽取，需跑 MinerU）
  pages  PDF 页数
"""
from pathlib import Path

PAPERS_DIR = Path("/Users/xbpd/Projects/视觉位移判断/02-视觉动作分析/01_参考论文")

# (name, pdf_relpath, existing_en_md_relpath_or_None, pages)
ENTRIES = [
    # ---------- 01_参考论文/_translation_outputs（已有英文抽取） ----------
    ("METRO：Transformer 端到端人体姿态与网格重建",
     "METRO/METRO：Transformer 端到端人体姿态与网格重建/METRO：Transformer 端到端人体姿态与网格重建.pdf",
     "_translation_outputs/2012.09760_2020_METRO/2012.09760_2020_METRO/auto/2012.09760_2020_METRO.md", 17),
    ("MarkerlessReview2021：视频无标记动捕临床康复生物力学综述",
     "OpenCap_传统生物力学/MarkerlessReview2021：视频无标记动捕临床康复生物力学综述/MarkerlessReview2021：视频无标记动捕临床康复生物力学综述.pdf",
     "_translation_outputs/Review_2021_VideoBasedMarkerlessMotionCapture/Review_2021_VideoBasedMarkerlessMotionCapture/auto/Review_2021_VideoBasedMarkerlessMotionCapture.md", 44),
    ("OpenCap：从智能手机视频获取人体运动动力学",
     "OpenCap_传统生物力学/OpenCap：从智能手机视频获取人体运动动力学/OpenCap：从智能手机视频获取人体运动动力学.pdf",
     "_translation_outputs/Uhlrich_2023_OpenCap_PLoS/Uhlrich_2023_OpenCap_PLoS/auto/Uhlrich_2023_OpenCap_PLoS.md", 26),
    ("OpenCap-Monocular：单目智能手机视频捕捉人体运动动力学",
     "OpenCap_传统生物力学/OpenCap-Monocular：单目智能手机视频捕捉人体运动动力学/OpenCap-Monocular：单目智能手机视频捕捉人体运动动力学.pdf",
     "_translation_outputs/Sapkota_OpenCapMonocular/Sapkota_OpenCapMonocular/auto/Sapkota_OpenCapMonocular.md", 1),
    ("MotionAGFormer：Transformer-GCNFormer 增强三维人体姿态估计",
     "PoseFormer_MotionBERT/MotionAGFormer：Transformer-GCNFormer 增强三维人体姿态估计/MotionAGFormer：Transformer-GCNFormer 增强三维人体姿态估计.pdf",
     "_translation_outputs/2310.16288_2023_MotionAGFormer/2310.16288_2023_MotionAGFormer/auto/2310.16288_2023_MotionAGFormer.md", 14),
    ("MotionBERT：从异构数据源学习人体运动表征",
     "PoseFormer_MotionBERT/MotionBERT：从异构数据源学习人体运动表征/MotionBERT：从异构数据源学习人体运动表征.pdf",
     "_translation_outputs/2210.06551_2022_MotionBERT/2210.06551_2022_MotionBERT/auto/2210.06551_2022_MotionBERT.md", 18),
    ("PoseFormer：时空 Transformer 的 3D 人体姿态估计",
     "PoseFormer_MotionBERT/PoseFormer：时空 Transformer 的 3D 人体姿态估计/PoseFormer：时空 Transformer 的 3D 人体姿态估计.pdf",
     "_translation_outputs/2103.10455_2021_PoseFormer/2103.10455_2021_PoseFormer/auto/2103.10455_2021_PoseFormer.md", 15),
    ("PoseFormerV2：频域高效鲁棒的 3D 人体姿态估计",
     "PoseFormer_MotionBERT/PoseFormerV2：频域高效鲁棒的 3D 人体姿态估计/PoseFormerV2：频域高效鲁棒的 3D 人体姿态估计.pdf",
     "_translation_outputs/2303.17472_2023_PoseFormerV2/2303.17472_2023_PoseFormerV2/auto/2303.17472_2023_PoseFormerV2.md", 14),
    ("Jamsrandorj2022-Wearable：腰戴混合 CNN-RNN 的步态事件预测",
     "RGB_角度估计_2022_EMBC/Jamsrandorj2022-Wearable：腰戴混合 CNN-RNN 的步态事件预测/Jamsrandorj2022-Wearable：腰戴混合 CNN-RNN 的步态事件预测.pdf",
     "_translation_outputs/Jamsrandorj_2022_arXiv_GaitEventsPrediction/Jamsrandorj_2022_arXiv_GaitEventsPrediction/auto/Jamsrandorj_2022_arXiv_GaitEventsPrediction.md", 10),
    ("Wei2024：基于 sEMG 的上肢康复连续运动意图预测综述",
     "sEMG_原始参照论文/Wei2024：基于 sEMG 的上肢康复连续运动意图预测综述/Wei2024：基于 sEMG 的上肢康复连续运动意图预测综述.pdf",
     "_translation_outputs/Wei_2024_TNSRE_MotionIntentionUpperLimb/Wei_2024_TNSRE_MotionIntentionUpperLimb/auto/Wei_2024_TNSRE_MotionIntentionUpperLimb.md", 19),
    ("TokenHMR：令牌化姿态表示的人体网格恢复",
     "TokenHMR_PoseTokens/TokenHMR：令牌化姿态表示的人体网格恢复/TokenHMR：令牌化姿态表示的人体网格恢复.pdf",
     "_translation_outputs/2404.16752_2024_TokenHMR/2404.16752_2024_TokenHMR/auto/2404.16752_2024_TokenHMR.md", 13),
    ("HMR2.0：4DHumans 基于 Transformer 的人体重建与跟踪",
     "ViT_HMR_2.0/HMR2.0：4DHumans 基于 Transformer 的人体重建与跟踪/HMR2.0：4DHumans 基于 Transformer 的人体重建与跟踪.pdf",
     "_translation_outputs/2305.20091_Goel_2023_HMR2.0/2305.20091_Goel_2023_HMR2.0/auto/2305.20091_Goel_2023_HMR2.0.md", 18),

    # ---------- 腕关节康复感知_D435i/_translation_outputs（已有英文抽取） ----------
    ("改进腕部运动学模型用于人机交互",
     "腕关节康复感知_D435i/腕关节与前臂专项/改进腕部运动学模型用于人机交互/改进腕部运动学模型用于人机交互.pdf",
     "腕关节康复感知_D435i/_translation_outputs/2002.05790/2002.05790/auto/2002.05790.md", 8),
    ("通过分类和平滑度分析简化自动神经康复",
     "腕关节康复感知_D435i/运动质量评价指标/通过分类和平滑度分析简化自动神经康复/通过分类和平滑度分析简化自动神经康复.pdf",
     "腕关节康复感知_D435i/_translation_outputs/2212.14797/2212.14797/auto/2212.14797.md", 14),
    ("自适应学习协作机器人上肢康复训练系统",
     "腕关节康复感知_D435i/腕关节与前臂专项/自适应学习协作机器人上肢康复训练系统/自适应学习协作机器人上肢康复训练系统.pdf",
     "腕关节康复感知_D435i/_translation_outputs/2305.10642/2305.10642/auto/2305.10642.md", 5),
    ("肱二头肌弯举疲劳诱发代偿运动分析",
     "腕关节康复感知_D435i/上肢代偿检测/肱二头肌弯举疲劳诱发代偿运动分析/肱二头肌弯举疲劳诱发代偿运动分析.pdf",
     "腕关节康复感知_D435i/_translation_outputs/2402.11421/2402.11421/auto/2402.11421.md", 11),
    ("RealSense-Kinematics：单 RGBD 相机实时开源上肢肌肉骨骼分析",
     "腕关节康复感知_D435i/深度图姿态估计精度/RealSense-Kinematics：单 RGBD 相机实时开源上肢肌肉骨骼分析/RealSense-Kinematics：单 RGBD 相机实时开源上肢肌肉骨骼分析.pdf",
     "腕关节康复感知_D435i/_translation_outputs/2406.10007_MarkerAssisted_RealSense_Kinematics/2406.10007_MarkerAssisted_RealSense_Kinematics/auto/2406.10007_MarkerAssisted_RealSense_Kinematics.md", 15),
    ("自适应协作机器人减轻假肢使用者代偿运动",
     "腕关节康复感知_D435i/上肢代偿检测/自适应协作机器人减轻假肢使用者代偿运动/自适应协作机器人减轻假肢使用者代偿运动.pdf",
     "腕关节康复感知_D435i/_translation_outputs/2505.01718/2505.01718/auto/2505.01718.md", 6),
    ("骨骼数据卒中代偿检测的图卷积 LSTM 注意力网络",
     "腕关节康复感知_D435i/上肢代偿检测/骨骼数据卒中代偿检测的图卷积 LSTM 注意力网络/骨骼数据卒中代偿检测的图卷积 LSTM 注意力网络.pdf",
     "腕关节康复感知_D435i/_translation_outputs/2512.06736/2512.06736/auto/2512.06736.md", 6),

    # ---------- 需先跑 MinerU 抽取（en_md = None） ----------
    ("Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计",
     "RGB_角度估计_2022_EMBC/Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计/Jamsrandorj2022-EMBC：视角无关的膝肘关节角度深度学习估计.pdf", None, 5),
    ("Kumar2022：深度学习从视觉系统预测下肢运动学",
     "RGB_角度估计_2022_EMBC/Kumar2022：深度学习从视觉系统预测下肢运动学/Kumar2022：深度学习从视觉系统预测下肢运动学.pdf", None, 5),
    ("Jamsrandorj2023：单相机 CNN-Transformer 的步态时空参数估计",
     "RGB_直接竞品_2026_Chao/Jamsrandorj2023：单相机 CNN-Transformer 的步态时空参数估计/Jamsrandorj2023：单相机 CNN-Transformer 的步态时空参数估计.pdf", None, 5),
    ("Chao2026：单目 RGB 相机的下肢关节角度估计",
     "RGB_直接竞品_2026_Chao/Chao2026：单目 RGB 相机的下肢关节角度估计/Chao2026：单目 RGB 相机的下肢关节角度估计.pdf", None, 10),
    ("Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测",
     "sEMG_原始参照论文/Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测/Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测.pdf", None, 13),
    ("Longhi2016：卒中上肢功能评估的仪器化指标效度验证",
     "腕关节康复感知_D435i/运动质量评价指标/Longhi2016：卒中上肢功能评估的仪器化指标效度验证/Longhi2016：卒中上肢功能评估的仪器化指标效度验证.pdf", None, 11),
    ("Seo2024：可穿戴传感器居家监测卒中上肢训练质量",
     "腕关节康复感知_D435i/上肢代偿检测/Seo2024：可穿戴传感器居家监测卒中上肢训练质量/Seo2024：可穿戴传感器居家监测卒中上肢训练质量.pdf", None, 14),
    ("Mei2026：面向康复训练的传感器化运动质量评估与生物反馈综述",
     "腕关节康复感知_D435i/运动质量评价指标/Mei2026：面向康复训练的传感器化运动质量评估与生物反馈综述/Mei2026：面向康复训练的传感器化运动质量评估与生物反馈综述.pdf", None, 23),
    ("Wang2022：卒中上肢代偿评估与检测技术系统综述",
     "腕关节康复感知_D435i/上肢代偿检测/Wang2022：卒中上肢代偿评估与检测技术系统综述/Wang2022：卒中上肢代偿评估与检测技术系统综述.pdf", None, 25),
    ("Li2024：IMU 上肢活动范围测量并行效度综述",
     "腕关节康复感知_D435i/上肢代偿检测/Li2024：IMU 上肢活动范围测量并行效度综述/Li2024：IMU 上肢活动范围测量并行效度综述.pdf", None, 35),
]

# MinerU 抽取产物统一落在这里
NEW_EXTRACT_DIR = PAPERS_DIR / "_translation_outputs"


def rows():
    for name, pdf_rel, en_rel, pages in ENTRIES:
        yield {
            "name": name,
            "pdf": PAPERS_DIR / pdf_rel,
            "paper_dir": (PAPERS_DIR / pdf_rel).parent,
            "en_md": (PAPERS_DIR / en_rel) if en_rel else None,
            "pages": pages,
        }


if __name__ == "__main__":
    import sys
    r = list(rows())
    missing_pdf = [x["name"] for x in r if not x["pdf"].exists()]
    missing_en = [x["name"] for x in r if x["en_md"] is None or not x["en_md"].exists()]
    print(f"清单条目: {len(r)}")
    print(f"总页数: {sum(x['pages'] for x in r)}")
    print(f"PDF 缺失: {missing_pdf or '无'}")
    print(f"需 MinerU 抽取: {len(missing_en)}")
    for n in missing_en:
        print("   -", n)
    if "--list" in sys.argv:
        for x in r:
            print(f"{x['pages']:>3}p  {'EN' if (x['en_md'] and x['en_md'].exists()) else 'NEW'}  {x['name']}")
    sys.exit(1 if missing_pdf else 0)
