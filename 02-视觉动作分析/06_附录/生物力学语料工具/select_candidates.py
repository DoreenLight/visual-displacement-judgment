#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_candidates.py — 人工裁定：把短名单收成候选清单

**这个文件里的一切判断都是人工做的**，脚本只负责：
  1. 用「判别式标题片段」从短名单精确取回完整记录（避免手抄标题出错）
  2. 补齐 schema 字段，输出与语料二同构的 biomech_candidates.json
  3. 报告哪些判别式没命中（= 我记错了标题，必须回头修，不能静默跳过）

口径（与语料二一致，三者严格分开）：
  - 理论搜索空间 = biomech_seeds.json 的 queries 条数
  - 实际命中     = discover.py 落盘的候选池条数
  - 候选清单     = 本文件裁定并写入 papers[] 的条数
  - 实际入库     = 之后 PDF 真实落盘的数
任何一处都不得混写成另一个数。

用法：python3 select_candidates.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHORTLIST = HERE / 'biomech_shortlist.json'
OUT = HERE / 'biomech_candidates.json'

# 第二轴：证据范式（替代语料二的「技术阶段」时间轴）
LENS = {
    'M0': '综述·标准·证据汇总',
    'M1': '测量与验证（精度/信度/有效性研究）',
    'M2': '模型与仿真（肌肉骨骼模型、逆动力学、仿真器）',
    'M3': '方法与算法（学习模型、token 化、可微优化）',
    'M4': '临床与康复（代偿、功能评定、上肢康复工程）',
}

# ---------------------------------------------------------------------------
# 人工裁定表：(线程, 判别式标题片段, 证据范式, 相关度, 裁定理由)
# 判别式取标题的**特征子串**，必须足以在短名单内唯一定位一条。
# ---------------------------------------------------------------------------
PICKS = [
    # ================= B1 视觉运动学：估计与验证方法学 =================
    ('B1', 'Accuracy of a markerless motion capture system in estimating upper extremit', 'M1', '高',
     '上肢标记无关动捕精度实证，与本项目研究范围最直接对应'),
    ('B1', 'SHOULDER AND ELBOW JOINT ANGLE ESTIMATION FOR UPPER LIMB REHABILITATION', 'M1', '高',
     '标题即「上肢康复的肩肘角估计」，与题目主题同构；说明该任务至少 2017 已被做，novelty 需另找'),
    ('B1', 'AN EXTENSION TO THE ISB WRIST JOINT COORDINATE SYSTEM', 'M0', '高',
     '★ 腕关节坐标系标准。决定「腕角」的物理定义，是词表单元设计的地基；未读此文不应定义腕 DoF'),
    ('B1', 'A new kinematic model of the upper extremity based on functional joint p', 'M0', '高',
     '★ 上肢功能性关节参数的运动学模型，直接决定坐标轴/自由度如何取'),
    ('B1', 'BIOMECHANICS OF PRONATION AND SUPINATION OF THE FOREARM', 'M0', '高',
     '★ 前臂旋前旋后是腕关节最主要的 2 个 DoF，也是多数数据集缺失的部分'),
    ('B1', 'Mechanism of limitation of pronation/supination of the forearm', 'M1', '中',
     '旋前旋后的几何限制机制，决定 ROM 语法边界的物理来源'),
    ('B1', 'Limited Forearm Motion Compensated by Thoracohumeral Kinematics', 'M4', '高',
     '★ 腕活动受限由胸肱关节代偿 —— 与本项目 D435i「肩肘代偿」任务包直接同构'),
    ('B1', 'Accuracy of biplane videoradiography for quantifying dynamic wrist kinematics', 'M1', '中',
     '腕关节动态运动学的精度研究，提供腕角误差的量级参照'),
    ('B1', 'Comparison of shoulder kinematics between Theia 3D markerless motion cap', 'M1', '高',
     '2025 年新工作，商业系统 vs 光学动捕，可作对标基线'),
    ('B1', 'Validity and reliability of Kinect skeleton for measuring shoulder joint', 'M1', '高',
     '深度相机（Kinect）测肩角的效度与信度 —— 与 D435i 同属 RGB-D 路线，是最贴近的设备类先例'),
    ('B1', 'Accuracy and repeatability of joint angles measured using a single camera', 'M1', '高',
     '单相机关节角精度与重复性（137 引），单目路线的核心参照'),
    ('B1', 'Analysis of Upper-Limb and Trunk Kinematic Variability: Accuracy and Rel', 'M1', '高',
     '上肢+躯干运动学的精度与信度，覆盖躯干干扰（代偿来源）'),
    ('B1', 'Accuracy evaluation of two markerless motion capture systems for measure', 'M1', '中',
     '两套标记无关系统的横向精度对比'),
    ('B1', 'A Comparative Study of Markerless Systems Based on Color-Depth Cameras', 'M1', '高',
     '★ 基于彩色-深度相机的系统对比 —— 与 D435i（RGB-D）设备形态最接近'),
    ('B1', 'Wearable systems for shoulder kinematics assessment: a systematic review', 'M0', '高',
     '肩关节角可穿戴测量系统综述（127 引），给出该子领域方法版图'),
    ('B1', 'Comparison of modelling and tracking methods for analysing elbow and for', 'M2', '高',
     '肘与前臂的建模 vs 追踪方法对比，兼跨 M1/M2'),
    ('B1', 'Comparison and Reliability of Three Goniometric Methods for Measuring Fore', 'M1', '中',
     '量角器法的前臂测量信度 —— 视觉法要对照的**临床金标准**'),
    ('B1', 'Accuracy, Validity, and Reliability of Markerless Camera-Based 3D Motion Captu', 'M1', '中',
     '2024 年相机式标记无关动捕的效度/信度'),
    ('B1', 'The accuracy of markerless motion capture combined with computer vision techni', 'M0', '中',
     'CV 技术 + 标记无关动捕的准确度综述'),
    ('B1', 'SWOT Analysis of Portable and Low-Cost Markerless Motion Capture Systems', 'M0', '低',
     '低成本标记无关系统的 SWOT，背景性'),
    ('B1', 'Exoskeleton for forearm pronation and supination rehabilitation', 'M4', '中',
     '前臂旋前旋后康复外骨骼 —— 腕 DoF 的应用出口'),
    ('B1', 'wrist units in upper limb prosthesis users', 'M4', '低',
     '假肢腕单元功能，背景性'),

    ('B1', 'Joint coordinate systems of axes for coherence in reporting kinematic and kinetic data', 'M0', '高',
     '★ 关节坐标系轴向一致性的方法学奠基（Clinical Biomechanics 2001）。'
     'V6「跨库腕角归一」必须先有它：不同数据集坐标系不一致时，需要一个权威的轴向约定方法。'
     '**注意：ISB 2002 recommendation 官方题名是 part I「ankle, hip and spine」——只管下肢，'
     '所以腕关节的权威是 2007 那篇 part II（付费，需人工取件）**'),

    # ================= B2 肌肉骨骼动力学与仿真 =================
    ('B2', 'OpenSim Moco: Musculoskeletal optimal control', 'M2', '高',
     '★ OpenSim 官方最优控制模块，是「可微/优化路线」的事实标准工具'),
    ('B2', 'Physics-Informed Deep Learning for Musculoskeletal Modeling: Predicting', 'M3', '高',
     '★★ 物理信息深度学习做肌肉骨骼建模（173 引，TNSRE）—— 物理约束与学习模型的既有桥梁'),
    ('B2', 'A Systematic Review of the Associations Between Inverse Dynamics and Mus', 'M0', '高',
     '★ 逆动力学与肌肉激活关系的系统综述 —— 决定「角度→力矩/激活」这一跃迁的证据基础'),
    ('B2', 'Estimation of Muscle Forces and Joint Moments Using a Forward-Inverse Dy', 'M2', '高',
     '★★ 前向-逆向动力学联立估力矩（206 引）—— 肌肉冗余问题的经典处理'),
    ('B2', 'Inverse Estimation of Multiple Muscle Activations From Joint Moment', 'M2', '高',
     '★★ 由关节力矩反解多肌肉激活 —— 「输出侧冗余」的直接先例，是路径B的理论依据'),
    ('B2', 'Development of a comprehensive musculoskeletal model of the shoulder and', 'M2', '高',
     '★ 肩关节综合肌肉骨骼模型（144 引）—— 上肢动力学模型的几何来源'),
    ('B2', 'Muscle Contributions to Upper-Extremity Movement and Work From a Musculo', 'M2', '高',
     '上肢运动与做功的肌肉贡献分解 —— 可作动力学第二输出头的可解释性模板'),
    ('B2', 'A Real-Time, 3-D Musculoskeletal Model for Dynamic Simulation of Arm Mov', 'M2', '高',
     '★ 手臂运动实时三维肌肉骨骼模型（IEEE TBME，103 引）'),
    ('B2', 'Sensitivity of subject-specific models to Hill muscle–tendon model param', 'M2', '高',
     '★★ 被试特异模型对 Hill 肌肉腱参数的敏感性 —— 直接支撑「群体标签 vs 被试特异」这一取舍'),
    ('B2', 'Moment arms and lengths of human upper limb muscles as functions of join', 'M2', '高',
     '★ 上肢肌肉力臂与长度随关节角的函数关系（126 引）—— 动力学先验数据'),
    ('B2', 'The moment arms of the muscles spanning the glenohumeral joint: a system', 'M0', '中',
     '肩关节跨越肌肉力臂的系统综述'),
    ('B2', 'Predictive multibody dynamic simulation of human neuromusculoskeletal sy', 'M2', '中',
     '神经肌肉系统预测性多体动力学仿真'),
    ('B2', 'A Probabilistic Approach to Quantify the Impact of Uncertainty Propagati', 'M2', '中',
     '不确定性传播的量化 —— 群体标签的误差上限论证'),
    ('B2', 'Simulation of human movement: applications using OpenSim', 'M0', '中',
     'OpenSim 应用综述'),
    ('B2', 'Evaluation of Direct Collocation Optimal Control Problem Formulations fo', 'M2', '中',
     '直接配点法最优控制问题构造评测 —— 「优化式」路线的具体方法学'),
    ('B2', 'Neuro-musculoskeletal simulation of instrumented contracture and spastic', 'M4', '高',
     '★ 痉挛与挛缩的神经肌肉骨骼仿真 —— 康复交叉点，病理运动学的建模入口'),
    ('B2', 'Muscle coordination retraining inspired by musculoskeletal simulations', 'M4', '中',
     '仿真启发的肌肉协调再训练 —— 动力学走向临床应用的实例'),
    ('B2', 'Muscle coactivation: definitions, mechanisms, and functions', 'M0', '中',
     '肌肉共激活定义与机制（345 引）—— 冗余解的非唯一性来源'),
    ('B2', 'Muscle synergies as a predictive framework for the EMG patterns', 'M0', '中',
     '肌肉协同作为 EMG 模式的预测框架（158 引）—— 低维结构假设的来源'),
    ('B2', 'Validity of muscle activation estimated with predicted ground reaction f', 'M1', '中',
     '用预测地面反力估肌肉激活的有效性 —— 标签误差的实证'),
    ('B2', 'Computational performance of musculoskeletal simulation in OpenSim Moco', 'M2', '中',
     'OpenSim Moco 的计算性能 —— 可行性核算'),
    ('B2', 'Automated creation and tuning of personalised muscle paths for OpenSim', 'M2', '中',
     '个体化肌肉路径自动生成与调参 —— 被试特异化的工程手段'),
    ('B2', 'A multibody biomechanical model of the upper limb including the shoulder', 'M2', '中',
     '含肩关节的上肢多体生物力学模型'),
    ('B2', 'Finger Muscle Attachments for an OpenSim Upper-Extremity Model', 'M2', '低',
     'OpenSim 上肢模型的指部肌肉附着点'),
    ('B2', 'Implicit methods for efficient musculoskeletal simulation and optimal co', 'M2', '低',
     '隐式方法加速肌肉骨骼仿真'),

    # ================= B3 上肢康复：代偿与临床评估 =================
    ('B3', 'Vision-based Automatic Detection of Compensatory Postures', 'M4', '高',
     '★★★ 视觉自动检测代偿姿势 —— 与本项目 TP-03 肩肘代偿任务包同题'),
    ('B3', 'Motor compensation and recovery for reaching in stroke patients', 'M4', '高',
     '★★ 卒中取物任务的运动代偿与恢复（280 引）—— 代偿机制的基础文献'),
    ('B3', 'Kinematics and Kinetics of Shoulder Complex, Elbow, and Wrist', 'M4', '高',
     '★ 肩复合体+肘+腕的运动学与动力学，腕肘肩一次覆盖的框架性文献'),
    ('B3', 'A clinically feasible kinematic assessment method of upper extremity', 'M1', '高',
     '★ 临床可行的上肢运动学评定方法 —— 决定「能不能进临床」的可行性门槛'),
    ('B3', 'Kinematic and Clinical Analyses of Upper-Extremity Movements Aft', 'M4', '高',
     '卒中后上肢运动的运动学与临床分析（139 引）'),
    ('B3', 'Kinematic Data Analysis for Post-Stroke Patients Following Bilat', 'M4', '高',
     '双侧训练后卒中患者的运动学数据分析（TNSRE，133 引）'),
    ('B3', 'Kinematic Analysis of the Upper Limb Motor Strategies in Stroke', 'M4', '中',
     '卒中上肢运动策略的运动学分析'),
    ('B3', 'Determining Levels of Upper Extremity Movement Impairment by App', 'M4', '高',
     '用应用/传感器判定上肢运动损伤等级（342 引）—— 连续量表 vs 序数评分的核心问题'),
    ('B3', 'Item Difficulty of Fugl-Meyer Assessment for Upper Extremity', 'M4', '中',
     'Fugl-Meyer 上肢量表条目难度 —— 量表本身的局限，支撑改用连续运动学指标的动机'),
    ('B3', 'Target height influences shoulder and elbow non-use after stroke', 'M4', '高',
     '★ 目标高度影响卒中后肩肘废用 —— 代偿对任务条件的敏感性'),
    ('B3', 'Thoracic position effect on shoulder range of motion, strength', 'M4', '中',
     '胸段位置对肩 ROM 的影响（365 引）—— 躯干-肩耦合，代偿的结构性来源'),
    ('B3', 'Wrist kinematics differ in expert and novice tennis players', 'M1', '中',
     '腕运动学的专家-新手差异 —— 腕角估计的效度场景'),
    ('B3', 'Wearable technology in stroke rehabilitation', 'M0', '中',
     '卒中康复可穿戴技术综述（277 引）'),
    ('B3', 'Robot-Assisted Therapy for Upper Extremity Motor Impairment', 'M0', '中',
     '上肢运动损伤的机器人辅助治疗综述（105 引）'),
    ('B3', 'Fatigue, induced via repetitive upper-limb motor tasks', 'M1', '低',
     '重复上肢任务诱发疲劳的影响 —— 运动质量指标之一'),

    # ================= B4 离散运动学 token 与表征量化 =================
    ('B4', 'VQ-HPS: Human Pose and Shape Estimation in a Vector-Quantized', 'M3', '高',
     '★ 向量量化潜空间做人体姿态与形状估计 —— 与「离散词表」最接近的 CV 先例'),
    ('B4', 'Masked reconstruction model of latent space vector quantization for huma', 'M3', '高',
     '★ 2026 年，掩码重建 + 潜空间 VQ 做人体 —— 需查是否已含运动学 token'),
    ('B4', 'Learning 3D Human Pose from Structure and Motion', 'M3', '中',
     '从结构与运动学习 3D 人体姿态 —— 表征学习基线'),
    ('B4', 'Robot Motion Planning in Learned Latent Spaces', 'M3', '中',
     '学习潜空间中的机器人运动规划 —— 离散潜空间用于控制'),
    ('B4', 'Learning Joint Twist Rotation for 3D Human Pose Estimation', 'M3', '中',
     '关节扭转量学习 —— 与「绕自身轴旋转」这一腕 DoF 表征直接相关'),
    ('B4', 'Best: BERT Pre-training for Sign Language Recognition with Coupling Toke', 'M3', '低',
     '手语识别的耦合 tokenization —— 手部运动 token 化的邻近先例，跨领域参照'),

    # ================= B5 视觉运动学 → 动力学 桥接 =================
    ('B5', 'Differentiable Biomechanics Unlocks Opportunities for Markerless M', 'M3', '高',
     '★★★ 可微生物力学用于 MMC 的开创工作（arXiv 2402.17192 / ICORR 2025）—— 本项目方法学侧最重要的先行'),
    ('B5', 'Differentiable Biomechanics for Markerless Motion Capture in Upper', 'M3', '高',
     '★★★ 上肢卒中可微生物力学 + 与光学动捕对比（IEEE TMRB 2026, 15 引）—— 已决定降级为「验证协议」来源'),
    ('B5', 'Vision-Based Approximate Estimation of Muscle Activation Patterns', 'M3', '高',
     '★★★ 视觉近似估计肌肉激活模式（IEEE RA-L 2023）—— 直接占据「视觉→肌肉激活」输出头'),
    ('B5', 'Towards Human Inverse Dynamics from Real Images: A Dataset and Ben', 'M3', '高',
     '★★ 从真实图像做人体逆动力学：数据集与基准（2025）—— 直接占据「视觉→逆动力学」'),
    ('B5', 'Learning-Based Inverse Dynamics of Human Motion', 'M3', '高',
     '基于学习的运动逆动力学（ICCVW 2017）—— 视觉/姿态→关节力的学习先例'),
    ('B5', 'Machine Learning for Optical Motion Capture-Driven Musculoskeletal', 'M0', '中',
     '动捕驱动的肌肉骨骼建模之机器学习综述'),
    ('B5', 'Evaluating Muscle Activation Models for Elbow Motion Estimation', 'M2', '高',
     '肘运动估计的肌肉激活模型评测 —— 群体模型标签的误差证据'),
    ('B5', 'Optimal-state Dynamics Estimation for Physics-based Human Motion C', 'M3', '中',
     '物理驱动人体动捕的最优状态动力学估计（NeurIPS 2024）'),
    ('B5', 'Applications and limitations of current markerless motion capture', 'M0', '高',
     '★ 现有标记无关动捕的应用与局限（380 引）—— 必引的总览'),
    ('B5', 'PoseAug: A Differentiable Pose Augmentation Framework', 'M3', '中',
     '可微姿态增强（CVPR 2021）—— 可微优化在人体任务中的可迁移性证据'),
    ('B5', 'Advancing human pose estimation through interdisciplinary physics-', 'M3', '中',
     '跨学科物理启发的人体姿态估计（2025）'),
    ('B5', 'Towards a Simplified Estimation of Muscle Activation Pattern from', 'M2', '中',
     '简化的肌肉激活模式估计（Sensors 2020）'),
    ('B5', 'A realistic upper-limb musculoskeletal model for overall muscle ac', 'M2', '高',
     '★ 面向整体肌肉激活的逼真上肢肌肉骨骼模型 —— 群体模型标签的来源候选'),
    ('B5', 'Continuous estimation of finger joint angles using muscle activati', 'M2', '中',
     '由肌肉激活连续估计指关节角 —— 冗余问题的简化范例'),
    ('B5', 'Scaled Motion Dynamics for Markerless Motion Capture', 'M3', '中',
     'CVPR 2007，尺度化运动动力学 —— 视觉+动力学的早期融合尝试'),
    ('B5', 'A neural network model for arm trajectory formation using forward', 'M2', '中',
     '用前向动力学形成手臂轨迹的神经网络模型'),
]


# ---------------------------------------------------------------------------
# 已核实的 arXiv 优先件：不经短名单，直接按 arXiv ID 并入。
# 理由：这些是决定研究定位的关键先行（尤其可微生物力学这一组），
# 必须在候选清单与索引里有一等席位，而不是停留在 _优先件/ 目录里当孤本。
# 元数据由 resolve_multisource.py 从 arXiv 官方 API 取得，不在此手抄。
# ---------------------------------------------------------------------------
# 判别式 → arXiv ID 的显式绑定。
# 为什么必须显式：ARXIV_EXTRA 里的条目在裁定阶段 `title` 是空串（元数据随后由
# resolve_multisource 填入），而 PICKS 里若已用标题判别式收过同一篇，它**没有 arxiv_id**，
# 于是 `if any(p.get('arxiv_id') == aid ...)` 这道去重拦不住 —— 实测产生 2 组重复条目，
# 两条索引指向同一个文件名，PDF 互相覆盖，计数还会虚报 2 篇。
ARXIV_HINTS = {
    'Differentiable Biomechanics Unlocks Opportunities for Markerless M': '2402.17192',
    'Differentiable Biomechanics for Markerless Motion Capture in Upper': '2411.14992',
}

# ---------------------------------------------------------------------------
# DOI_FIX —— component DOI 归一（K7 的复发防线）
#
# 🚨 2026-10-02 实测：Crossref 对同一篇文章同时返回两个标识符
#      10.1109/tmrb.2025.3605962      -> type = journal-article  （正牌）
#      10.1109/tmrb.2025.3605962/mm1  -> type = component       （多媒体组件）
#    候选池里存的是后者，于是**正牌标识被 component 顶掉**，
#    本项目库里会只剩一个**不可引用的组件标识**。
#    → 这就是 K7（component DOI 混入可引用索引）的**同一条老毛病**，
#      只不过上次是混入，这次是**顶替**。
# 📌 为什么用离线映射而不是构建时联网判类型：构建链应当可离线复现，
#    判定依据（Crossref 的 type 字段）已在这里固化，附判定日期。
# ---------------------------------------------------------------------------
DOI_FIX = {
    '10.1109/tmrb.2025.3605962/mm1': '10.1109/tmrb.2025.3605962',  # 2026-10-02 Crossref: component -> journal-article
}


def normalize_doi(doi):
    """把已知的 component DOI 折回正牌标识；未登记的原样返回。"""
    d = (doi or '').strip().lower()
    return DOI_FIX.get(d, d or '')


PRIOR_MANUAL = [
    {
        "thread": "B1",
        "lens": "M0",
        "relevance": "高",
        "year": 2007,
        "venue": "Journal of Shoulder and Elbow Surgery",
        "doi": "10.1016/j.jse.2007.02.134",
        "title": "Test-retest reproducibility of elbow goniometric measurements in a rigid double-blinded protocol: Intervals for distinguishing between measurement error and clinical change",
        "selected_from": "v7d_clinical_threshold_search_2026-10-03",
        "found_via": ["crossref", "europepmc"],
        "note": "★★ 【V7d 的答案：上肢有自己的一套误差地板，且比 E2 的 gait 外推宽约 2 倍】"
                "Cleffken B, van Breukelen G, van Mameren H, Brink P, Olde Damink S。"
                "Crossref 核：J Shoulder Elbow Surg 2007;16(6):788-794.e2。Europe PMC 核：PMID 17967547。"
                "方法：修改 Bland-Altman 法求 smallest detectable differences (SDD)；42 人、每人 144 次评定、"
                "电子数字量角器、双盲 test-retest 交叉设计。"
                "**SDD 原文数值**：主动伸 0±4.2°（无上臂固定）｜主动屈 0±8.2°（无固定）｜"
                "主动伸 0±6.3°｜主动屈 0±5.7°｜被动屈 0±7.4°（有固定）｜主动屈 0±10.1°（上臂后伸）｜"
                "主动 ROM 0±8.5°｜被动 ROM 0±10.8°。"
                "原文结论句可直接引用：*\"Differences smaller than these SDDs found in clinical or research "
                "settings are attributable to measurement error and do not indicate improvement.\"* "
                "📌 **对本项目的三重后果**：(a) V7d 就肘关节闭合 —— 上肢确实有临床阈值，"
                "不必再借用下肢 gait 分档；(b) **E2 的 5°（gait 导出）对上肢 ROM 偏乐观约 2 倍**，"
                "以 5° 为验收线等于把误差压到参考方法自身的噪声地板以下，"
                "此时测到的差异不可归因于算法；(c) **这为「为什么要离散」提供了一个非直觉的论据** —— "
                "不是压缩、也不是抽象的可解释性，而是**输出粒度必须与测量精度匹配**："
                "在 ±5–10° 的测量噪声下，连续角度输出在该带内不可解释，"
                "而锚定在 ROM 边界上的符号能给出「是否接近全伸/全屈」这种在噪声下仍成立的陈述。"
                "⚠️ **边界**：仅肘关节；肩与腕**未找到同规格 SDD**（见 F31）。"
                "正文为 Elsevier 闭取（isOA=N），但**全部数值均在摘要内**，故摘要级即可引用。",
    },
    {
        "thread": "B1",
        "lens": "M0",
        "relevance": "中",
        "year": 1997,
        "venue": "Clinical Rehabilitation",
        "doi": "10.1177/026921559701100408",
        "title": "Joint angle measurement: a comparative study of the reliability of goniometry and wire tracing for the hand",
        "selected_from": "v7d_clinical_threshold_search_2026-10-03",
        "found_via": ["crossref", "europepmc"],
        "note": "★ 【手部，指关节】Ellis B, Bruton A, Goddard JR。"
                "Crossref 核：Clin Rehabil 1997;11(4):314-320。Europe PMC 核：PMID 9408672。"
                "20 名作业治疗师 + 20 名物理治疗师、9 个中心、遮蔽量角器 vs 铁丝描记法、"
                "受试者固定于两个体位重复评定 MCPJ/PIPJ/DIPJ。"
                "摘要级结论：**量角器可靠性优于铁丝描记**；**DIPJ 的可重复性低于其他关节**；"
                "临床与专科经验**不影响**可靠性。"
                "⚠️ **本条的局限必须写清**：摘要提到 repeatability coefficient 但**未给出数值**，"
                "且**研究对象是手指而非腕** → **不能直接当腕的阈值用**。"
                "价值在于**提供腕以下「关节越小越不可靠」的方向性证据**，与 Cleffken 的肘 SDD 同向。",
    },
    {
        "thread": "B1",
        "lens": "M0",
        "relevance": "高",
        "year": 2021,
        "venue": "Journal of Hand Therapy",
        "doi": "10.1016/j.jht.2020.05.001",
        "title": "Remote self-measurement of wrist range of motion performed on normal wrists by a minimally trained individual using the iPhone level application only demonstrated good reliability in measuring wrist flexion and extension",
        "selected_from": "v7d_shoulder_wrist_followup_2026-10-03",
        "found_via": ["crossref", "europepmc"],
        "note": "★★ 【补上 Cleffken 2007 缺的那一半：腕与前臂旋转的可靠性，且序数方向一致】"
                "Alford SL。Crossref 核：J Hand Ther 2021;34(4):549-554。Europe PMC 核：PMID 32883544。"
                "40 名健康受试者（80 腕），两名检查者各测两次，比较通用量角器 vs iPhone Level 应用；"
                "ICC（two-way random，绝对一致性，95% CI）。"
                "**原文可靠性分层（必须照录，不可合并）**："
                "① 腕屈伸 = **good** inter-rater reliability（所有条件均如此）；"
                "② 前臂旋后/旋前/桡偏/尺偏 = **moderate**（由检查者测量时）；"
                "③ **受试者自测**：旋后 moderate，但**旋前/桡偏/尺偏 = poor**。"
                "📌 **对本项目的价值**：这与 Cleffken 的肘 SDD、"
                "TMRB 2026 的零结果、Pohl 2026 的临床清单**方向一致** —— "
                "**腕屈伸 > 前臂旋转，而旋前是其中最差的一个**。"
                "这是第 4 个**类型完全不同**的独立来源（自测可靠性），前三个分别是"
                "CV 基准 / 临床调查 / 测量者重测 SDD。"
                "🚨 **但绝不复活主张 C 的原措辞**：F29 已证伪「没人报过旋前旋后精度」，"
                "**本条讲的是「测的时候有多不可靠」，不是「有没有人测过」—— 两者是不同命题**。"
                "⚠️ **构念限制**：本文用 **ICC（相对一致性）**，Cleffken 用 **SDD（绝对差异）**，"
                "**两者数值不可互换、不可并列比较**，只能比**序数方向**。",
    },
    {
        "thread": "B1",
        "lens": "M0",
        "relevance": "低",
        "year": 2014,
        "venue": "Medical Express",
        "doi": "10.5935/medicalexpress.2014.02.08",
        "title": "Reliability Index of inter- and intra-rater of manual goniometry and computerized biophotogrammetry to assess the range of motion of internal and external shoulder rotation",
        "selected_from": "v7d_shoulder_wrist_followup_2026-10-03",
        "found_via": ["crossref"],
        "note": "⚠️ 【肩内外旋 ROM 的量角器 vs 生物摄影测量可靠性 —— 主题对口，但证据权重低】"
                "Antonietti L, Luna N, Nogueira G, Ito A, Santos M, Alonso A。"
                "Crossref 核：Medical Express 2014;1(2)。"
                "🚨 **低权重的理由（如实登记）**：① **被引仅 3 次**；"
                "② **Europe PMC 无命中、无摘要** → **其可靠性数值目前不可得**；"
                "③ 期刊为区域性小型刊。"
                "📌 **仍登记的理由**：它是本轮检索中**唯一**题名对口「肩内外旋 ROM 测量可靠性」的条目，"
                "**而肩是 V7d 尚未闭合的那一半**。→ **已登记为 P0-8 待取件**；"
                "**取回前不得引用其任何数值**（K19）。",
    },
    {
        "thread": "B1",
        "lens": "M0",
        "relevance": "中",
        "year": 1990,
        "venue": "Physiotherapy",
        "doi": "10.1016/S0031-9406(10)63103-3",
        "title": "Comparison of Visual Estimation and Goniometry in Determination of a Shoulder Joint Angle",
        "selected_from": "v7d_clinical_threshold_search_2026-10-03",
        "found_via": ["crossref"],
        "note": "★ 【肩关节；视觉估计 vs 量角器 —— 与本项目设定最接近的一篇】"
                "Williams JG, Callaghan M。Crossref 核：Physiotherapy 1990;76:655-657。"
                "📌 **相关性**：它比较的是**用眼睛估肩关节角**与**用量角器量**的差别，"
                "正是「视觉能不能替代仪器测量」这个问题在上肢肩关节上的 1990 年版本。"
                "🚨 **本条无摘要可用**（Europe PMC 未收录 Physiotherapy；1990 年刊，无 OA）。"
                "**已登记为 P0 待取件**（编号 P0-7），须用户经学校数据库取回。"
                "**在取回前不得引用其任何数值**（K19 纪律：标识符须反查，数值须回原文）。",
    },
    {
        "thread": "B1",
        "lens": "M0",
        "relevance": "高",
        "year": 2007,
        "venue": "Journal of Shoulder and Elbow Surgery",
        "doi": "10.1016/j.jse.2007.02.134",
        "title": "Test-retest reproducibility of elbow goniometric measurements in a rigid double-blinded protocol: Intervals for distinguishing between measurement error and clinical change",
        "selected_from": "v7d_clinical_threshold_search_2026-10-03",
        "found_via": ["crossref", "europepmc"],
        "note": "★★ 【V7d 的答案：上肢有自己的一套误差地板，且比 E2 的 gait 外推宽约 2 倍】"
                "Cleffken B, van Breukelen G, van Mameren H, Brink P, Olde Damink S。"
                "Crossref 核：J Shoulder Elbow Surg 2007;16(6):788-794.e2。Europe PMC 核：PMID 17967547。"
                "方法：修改 Bland-Altman 法求 smallest detectable differences (SDD)；42 人、每人 144 次评定、"
                "电子数字量角器、双盲 test-retest 交叉设计。"
                "**SDD 原文数值**：主动伸 0±4.2°（无上臂固定）｜主动屈 0±8.2°（无固定）｜"
                "主动伸 0±6.3°｜主动屈 0±5.7°｜被动屈 0±7.4°（有固定）｜主动屈 0±10.1°（上臂后伸）｜"
                "主动 ROM 0±8.5°｜被动 ROM 0±10.8°。"
                "原文结论句可直接引用：*\"Differences smaller than these SDDs found in clinical or research "
                "settings are attributable to measurement error and do not indicate improvement.\"* "
                "📌 **对本项目的三重后果**：(a) V7d 就肘关节闭合 —— 上肢确实有临床阈值，"
                "不必再借用下肢 gait 分档；(b) **E2 的 5°（gait 导出）对上肢 ROM 偏乐观约 2 倍**，"
                "以 5° 为验收线等于把误差压到参考方法自身的噪声地板以下，"
                "此时测到的差异不可归因于算法；(c) **这为「为什么要离散」提供了一个非直觉的论据** —— "
                "不是压缩、也不是抽象的可解释性，而是**输出粒度必须与测量精度匹配**："
                "在 ±5–10° 的测量噪声下，连续角度输出在该带内不可解释，"
                "而锚定在 ROM 边界上的符号能给出「是否接近全伸/全屈」这种在噪声下仍成立的陈述。"
                "⚠️ **边界**：仅肘关节；肩与腕**未找到同规格 SDD**（见 F31）。"
                "正文为 Elsevier 闭取（isOA=N），但**全部数值均在摘要内**，故摘要级即可引用。",
    },
    {
        "thread": "B1",
        "lens": "M0",
        "relevance": "中",
        "year": 1997,
        "venue": "Clinical Rehabilitation",
        "doi": "10.1177/026921559701100408",
        "title": "Joint angle measurement: a comparative study of the reliability of goniometry and wire tracing for the hand",
        "selected_from": "v7d_clinical_threshold_search_2026-10-03",
        "found_via": ["crossref", "europepmc"],
        "note": "★ 【手部，指关节】Ellis B, Bruton A, Goddard JR。"
                "Crossref 核：Clin Rehabil 1997;11(4):314-320。Europe PMC 核：PMID 9408672。"
                "20 名作业治疗师 + 20 名物理治疗师、9 个中心、遮蔽量角器 vs 铁丝描记法、"
                "受试者固定于两个体位重复评定 MCPJ/PIPJ/DIPJ。"
                "摘要级结论：**量角器可靠性优于铁丝描记**；**DIPJ 的可重复性低于其他关节**；"
                "临床与专科经验**不影响**可靠性。"
                "⚠️ **本条的局限必须写清**：摘要提到 repeatability coefficient 但**未给出数值**，"
                "且**研究对象是手指而非腕** → **不能直接当腕的阈值用**。"
                "价值在于**提供腕以下「关节越小越不可靠」的方向性证据**，与 Cleffken 的肘 SDD 同向。",
    },
    {
        "thread": "B1",
        "lens": "M0",
        "relevance": "中",
        "year": 1990,
        "venue": "Physiotherapy",
        "doi": "10.1016/S0031-9406(10)63103-3",
        "title": "Comparison of Visual Estimation and Goniometry in Determination of a Shoulder Joint Angle",
        "selected_from": "v7d_clinical_threshold_search_2026-10-03",
        "found_via": ["crossref"],
        "note": "★ 【肩关节；视觉估计 vs 量角器 —— 与本项目设定最接近的一篇】"
                "Williams JG, Callaghan M。Crossref 核：Physiotherapy 1990;76:655-657。"
                "📌 **相关性**：它比较的是**用眼睛估肩关节角**与**用量角器量**的差别，"
                "正是「视觉能不能替代仪器测量」这个问题在上肢肩关节上的 1990 年版本。"
                "🚨 **本条无摘要可用**（Europe PMC 未收录 Physiotherapy；1990 年刊，无 OA）。"
                "**已登记为 P0 待取件**（编号 P0-7），须用户经学校数据库取回。"
                "**在取回前不得引用其任何数值**（K19 纪律：标识符须反查，数值须回原文）。",
    },
    {
        "thread": "B1",
        "lens": "M0",
        "relevance": "高",
        "year": 2005,
        "venue": "Journal of Biomechanics",
        "doi": "10.1016/j.jbiomech.2004.05.042",
        "title": "ISB recommendation on definitions of joint coordinate systems of various joints for the reporting of human joint motion—Part II: shoulder, elbow, wrist and hand",
        "selected_from": "manual_recovery_from_orphan_card",
        "found_via": ["crossref"],
        "note": "★★ 腕/肘/肩关节坐标系标准本体（V6）。决定腕角与前臂旋转的物理定义，是词表单元设计的地基。此前只有一张不可复现的定位卡游离在盘上（不在任何 candidates/resolved 备份里），本条把它补进可复现链路。Crossref 核：J Biomech 2005, 38(5):981-992, Wu G 等 12 人。本地已取得并核验 12 页出版版 PDF（J Biomech 38, 981–992，DOI 10.1016/j.jbiomech.2004.05.042）。取件缺口已闭合；出版商 OA 元数据仍为 closed，不因本地有原件改标开放获取。坐标系实施及 V6 实验验证仍需独立验收。",
    },
    {
        "thread": "B3",
        "lens": "M4",
        "relevance": "高",
        "year": 2026,
        "venue": "JMIR Rehabilitation and Assistive Technologies",
        "doi": "10.2196/87339",
        "title": "From Metrics to Meaning in Neurological Rehabilitation: Clinicians' Perspectives on Digital Metrics of Upper Limb Functioning—A Focus Group Study",
        "selected_from": "snowball_gen2_flagged_20",
        "found_via": ["open_citations_coci_snowball_gen2"],
        "note": "★★ 【原为 JMIR 预印本 10.2196/preprints.87339（2025-11-07），已正式发表于 JMIR Rehabil Assist Technol 2026】11 名神经康复临床医师（PT/OT/运动科学/医师，临床经验 4–26 年）焦点小组，系统评估 17 个运动学指标 + 10 个真实场景表现指标。**对本项目有双重意义**：(a) 正面 —— 临床侧确实缺「可解释的结构化客观参考值」，Theme 5 明写临床依赖「simply look」，这正是 grammar 层要补的位置；(b) 风险 —— 临床医师并不熟悉 jerk / interjoint coordination（10 人中仅 4–5 人给出评分），且全文 0 次提及 joint angle，说明「离散符号词表 + 力学术语」对临床端**不天然直觉**。已取得 Europe PMC 发表版全文 XML（PMC13293569）",
    },
    {
        "thread": "B4",
        "lens": "M3",
        "relevance": "高",
        "year": 2025,
        "venue": "AAAI Conference on Artificial Intelligence",
        "doi": "10.1609/aaai.v39i5.32511",
        "title": "Multi-View 3D Human Pose Estimation with Weakly Synchronized Images",
        "selected_from": "snowball_gen3_flagged_then_aaai_oa_channel",
        "found_via": ["open_citations_coci_snowball_gen3"],
        "note": "★★ **F11（ACM MM 2024 Hand Labanotation）的同组第三篇**（F17）。作者 Li Ling, Gu Ruiwen, Wang Chongyang, Xing Junliang, Yu Xinchun, Zhang Xiao-Ping —— **与 DigitCode(F5) 同组，连续三篇压在离散符号化运动表示这条线上**。已取得 **AAAI 官方开放 proceedings 全文 PDF**（ojs.aaai.org，合法免费通路）。**该文正文自述多视角 3D 人体姿态估计的应用为「sports and art motion analysis」，并把 F11 引在该语境下** → A 级证据表明该组定位是艺术/体育，**不是临床**。另：其参考文献独立佐证 F11 的 ACM MM 页码 4092–4100",
    },
    {
        "thread": "B1",
        "lens": "M0",
        "relevance": "低",
        "year": 2014,
        "venue": "Medical Express",
        "doi": "10.5935/medicalexpress.2014.02.08",
        "title": "Reliability Index of inter- and intra-rater of manual goniometry and computerized biophotogrammetry to assess the range of motion of internal and external shoulder rotation",
        "selected_from": "v7d_shoulder_wrist_followup_2026-10-03",
        "found_via": ["crossref"],
        "note": "⚠️ 【肩内外旋 ROM 的量角器 vs 生物摄影测量可靠性 —— 主题对口，但证据权重低】"
                "Antonietti L, Luna N, Nogueira G, Ito A, Santos M, Alonso A。"
                "Crossref 核：Medical Express 2014;1(2)。"
                "🚨 **低权重的理由（如实登记）**：① **被引仅 3 次**；"
                "② **Europe PMC 无命中、无摘要** → **其可靠性数值目前不可得**；"
                "③ 期刊为区域性小型刊。"
                "📌 **仍登记的理由**：它是本轮检索中**唯一**题名对口「肩内外旋 ROM 测量可靠性」的条目，"
                "**而肩是 V7d 尚未闭合的那一半**。→ **已登记为 P0-8 待取件**；"
                "**取回前不得引用其任何数值**（K19）。",
    },
    # ================================================================
    # 2026-10-03 用户取回全文后新增（研究者下载 → 本地归置 → 反查入库）
    # 起因：用户经学校数据库取回全部可得论文后，01_参考论文/ 根目录与 20261001/ 倾倒目录
    #       出现 44 份散落 PDF。按 K8/K22 纪律，**不手放定位卡**，一律经本列表回注。
    # ================================================================
    {
        "thread": "B1",
        "lens": "M1",
        "relevance": "高",
        "year": 2001,
        "venue": "Perceptual and Motor Skills",
        "doi": "10.2466/pms.2001.93.2.353",
        "force": True,
        "title": "Comparison and reliability of three goniometric methods for measuring forearm supination and pronation",
        "selected_from": "user_retrieved_fulltext_2026-10-03",
        "found_via": ["crossref", "user_school_db"],
        "note": "★★ 【= 台账 H1 的全文，V7d 前臂侧闭合】Gajdosik RL, 2001。"
                "Crossref 核：Perceptual and Motor Skills 2001;93(2):353-355。"
                "**用户已取回全文**（原 H1 因无摘要长期挂「待人工裁定」）。"
                "📌 **原文 ICC 数值（31 人：10 男 21 女，主动旋前/旋后 ROM，三种量角器方法）**："
                "同一测试内三次评定 **ICC .81–.97**；1 小时后重测 **ICC .86–.96**；"
                "Method One（可动臂对齐手持铅笔）最可靠但**测量上不可用**，"
                "Method Two/Three 在临床上适用。"
                "🚨 **三条使用纪律**：① **这是 ICC 不是 SDD** —— 与 E12（Cleffken 肘 SDD 8.5°）"
                "**构念不同、数值不可并列，只能比序数**；② 样本为**健康人**、非卒中患者，"
                "不可外推到临床人群；③ **Crossref 返回 4 个 DOI 指向同一页**"
                "（`10.2466/pms.2001.93.2.353` / `pms.93.6.353-355` / `pms.93.5.353-355`），"
                "已取**带卷期**的 `10.2466/pms.2001.93.2.353` 为正牌，冲突如实登记。",
    },
    {
        "thread": "B1",
        "lens": "M0",
        "relevance": "中高",
        "year": 2001,
        "venue": "Hand Clinics",
        "doi": "10.1016/s0749-0712(21)00605-3",
        "force": True,
        "title": "Biomechanics of pronation and supination of the forearm",
        "selected_from": "user_retrieved_fulltext_2026-10-03",
        "found_via": ["crossref", "user_school_db"],
        "note": "★ 【V6c「腕 3 DoF」的解剖学依据】Hand Clinics 2001;17(1):111-122。"
                "📌 **原文关键句（支撑腕 = 3 DoF）**：前臂旋前/旋后之外，腕还承担 "
                "*\"flexion-extension, and radial and ulnar inclinations, and makes it a "
                "three-degrees-of-freedom\"* 的复合运动。"
                "→ 与 V6c 裁定（选 A：记在腕，ISB 口径，腕 3 DoF = 屈伸 + 外展内收 + 旋前旋后）"
                "**同向**，且是**解剖层面**的独立佐证（ISB 是标准层面，V6c 选 A 时的依据是 F24/F25，"
                "本条补的是第三条腿）。"
                "🚨 **注意**：本文是**综述**，不是原始测量研究，"
                "**不能当作腕 3 DoF 的定量精度依据**（那要靠 E12/SDD 一路）。",
    },
    {
        "thread": "B1",
        "lens": "M1",
        "relevance": "中",
        "year": 2007,
        "venue": "Journal of Biomechanics",
        "doi": "10.1016/s0021-9290(07)70188-4",
        "force": True,
        "title": "An extension to the ISB wrist joint coordinate system: An in vitro accuracy study",
        "selected_from": "user_retrieved_fulltext_2026-10-03",
        "found_via": ["crossref", "user_school_db"],
        "note": "★ 【V10 可闭合】Lenhoff MW, Carter TI, Pansy B, Wolff A, Cheng RY, Backus SI, "
                "Crisco JJ, Hillstrom HJ 等。Crossref 核：J Biomech 2007;40(Suppl 2):S192。"
                "📌 **本条正是台账 V10 记的那一条** —— V10 原记录：「语料曾把『ISB 腕坐标系标准』"
                "记错，存的是 `10.1016/s0021-9290(07)70188-4` = M. Lenhoff 等 2007 的"
                "**体外精度研究**，不是标准本身」。→ **该误记已被独立反查坐实，本条全文已取回，"
                "V10 的事实部分闭合**。"
                "🚨 **两条限制必须一起写**：① **这是会议摘要（1 页，J Biomeck 增刊 S192），"
                "不是完整论文** —— 证据权重低，只能坐实「存在这样一项体外精度研究」与"
                "「它是对 ISB 的延伸而非标准本体」，**不得引用其任何精度数值**；"
                "② 真正的标准本体仍是 **ISB Part II**（`10.1016/j.jbiomech.2004.05.042`，Wu G 等 12 人，"
                "J Biomech 2005;38(5):981-992），**该文本轮亦由用户取回**（原为 Elsevier 付费闭取，"
                "V6 一直记「坐标定义本身尚未读到」）。",
    },
    {
        "thread": "B3",
        "lens": "M4",
        "relevance": "高",
        "year": 2012,
        "venue": "Journal of Applied Biomechanics",
        "doi": "10.1123/jab.28.2.127",
        "force": True,
        "title": "Limited forearm motion compensated by thoracohumeral kinematics when performing tasks requiring pronation and supination",
        "selected_from": "user_retrieved_fulltext_2026-10-03",
        "found_via": ["crossref", "user_school_db"],
        "note": "★★ 【代偿机制的直接一手证据 + 前臂旋转真实 ROM 实测】Pereira BP, Thambyah A, Lee T。"
                "Crossref 核：J Appl Biomech 2012;28(2):127-138。"
                "📌 **原文数值**：6 名健康者，用夹具分别固定于**固定旋后**与**中立位**，"
                "完成 5 项需要旋前/旋后的日常任务；"
                "**正常前臂旋转 ROM = 77.6°（SD = 30.8°）**，固定旋后降至 11.3°（SD 2.9°）、"
                "中立位 18.2°（SD 6.2°）；代偿发生在**肩**——外展内收 +57.3° / +62.8°、"
                "前屈后伸 +24.3° / +18.2°、内外旋 +37.1° / +44.2°（均 p<.001 或 p<.05）；"
                "**肘几乎不代偿（p = .14）**。"
                "🎯 **对本项目的两处直接价值**："
                "① **代偿发生在肩、发生在近端而非远端** —— 这与 `创新点查重_vs_竞品.md` 里"
                "「近端代偿」那条论证同向，且给出了**方向与量级**，可作 B3 代偿线的对照文献；"
                "② 🚨 **它直接修正 M11 的一个假设** —— M11（符号重测稳定性）原先按 **240°** 的"
                "ROM 跨度估算符号粒度，本条实测**前臂旋前/旋后正常仅 77.6°**，"
                "**240° 这个取值明显偏大**；但 M11 的 σ 仍须逐关节取值（见 E12 与 Alford），"
                "**本条只提供 ROM 跨度，不提供重测 σ**，两者不可混用。",
    },
    {
        "thread": "B2",
        "lens": "M2",
        "relevance": "中",
        "year": 2002,
        "venue": "Clinical Biomechanics",
        "doi": "10.1016/s0268-0033(02)00034-7",
        "force": True,
        "title": "Mechanism of limitation of pronation/supination of the forearm in geometric models of deformities of the forearm bones",
        "selected_from": "user_retrieved_fulltext_2026-10-03",
        "found_via": ["crossref", "user_school_db"],
        "note": "Yasutomi T, Nakatsuchi Y, Koike H, Uchiyama S。Crossref 核：Clin Biomech 2002;17(6):456-463。"
                "用前臂骨（桡/尺）三维几何模型 + 骨间膜轴，研究**前臂骨畸形与骨间隙狭窄"
                "如何限制旋前/旋后**。"
                "📌 **相关性**：为「旋前/旋后是真实解剖自由度、受骨性几何约束」提供机制侧解释，"
                "与 V6c（腕第三 DoF 是填不满的槽位）**同向但层级不同** —— 本条讲的是"
                "**为什么这个 DoF 受限**，不是**怎么定义它**。"
                "🚨 **检出但相关性仅中**：不含任何视觉/估计/精度内容，不进 §0.5 竞品表。",
    },
    {
        "thread": "B3",
        "lens": "M3",
        "relevance": "低",
        "year": 2004,
        "venue": "26th Annual International Conference of the IEEE Engineering in Medicine and Biology Society",
        "doi": "10.1109/iembs.2004.1403778",
        "force": True,
        "title": "Exoskeleton for forearm pronation and supination rehabilitation",
        "selected_from": "user_retrieved_fulltext_2026-10-03",
        "found_via": ["crossref", "user_school_db"],
        "note": "Andreasen DS, Aviles AA, Allen SK, Guthrie KB, Jennings BRJ, Sprigle SH。"
                "Crossref 核：IEEE EMBC 2004, vol 3, pp. 2714-2717。"
                "脊髓损伤/脑卒中后前臂旋前旋后康复外骨骼；传感器可测关节 ROM。"
                "📌 **相关性低但仍登记**：它证明**旋前/旋后被康复工程界当作可独立建模、"
                "可独立训练、需独立测量 ROM 的关节自由度** —— 对 V6c 的定性支撑，"
                "**但无任何精度或视觉成分**，不得进对照表。",
    },
    # ================================================================
    # 2026-10-03 K8 收编：5 张**孤儿定位卡**（由另一会话于 20:44 手工落盘）
    # 起因：自检实测「定位卡 112 张 vs 候选清单 106 条」。逐张回查发现 5 张卡的
    #       `confirm_sources: "—"`、等级为硬编码 A —— **不是 resolve 链路产物**，
    #       任何脚本都复现不出来。按 K8，它们「看起来有元数据有等级」，
    #       「任何人都会当成已收录条目引用」，但重跑只会让它们继续游离。
    # 处理：经 Crossref 逐条反查后固化为 PRIOR_MANUAL，使链路可复现。
    # 🚨 其中 P0-1 的卡片**年份写成 2026，实为 2024**，已按 Crossref 更正。
    # ================================================================
    {
        "thread": "B1",
        "lens": "M3",
        "relevance": "高",
        "year": 2024,
        "venue": "Proceedings of the 32nd ACM International Conference on Multimedia (ACM MM 2024)",
        "doi": "10.1145/3664647.3680568",
        "title": "Translating Motion to Notation: Hand Labanotation for Intuitive and Comprehensive Hand Movement Documentation",
        "force": True,
        "selected_from": "k8_orphan_card_recovery_2026-10-03",
        "found_via": ["crossref", "user_school_db"],
        "note": "★★ 【F11 = 全项目最撞车的一篇；= 台账 P0-1；孤儿卡已于 2026-10-03 收编】"
                "*Translating Motion to Notation: Hand Labanotation for Intuitive and "
                "Comprehensive Hand Movement Documentation*，ACM MM 2024, pp. 4092-4100。"
                "🚨 **两条元数据纠正（均为本轮实测）**："
                "① **孤儿卡把年份写成 2026，Crossref 核为 2024** —— 已更正；"
                "② 🚨🚨 **DOI 用错会让本项目最撞车的一篇永久不可引用（本轮实测踩中）**：孤儿卡登记的是 "
                "**`10.1145/3664647`（父级 = 会议论文集）**，而 Crossref 对该 DOI 返回 "
                "title=*Proceedings of the 32nd ACM International Conference of Multimedia*、"
                "type=**proceedings**、**无独立作者/页码** → **题名比对必然失败**，"
                "实测该条落为 **`unresolved` / D 级（不得引用）**。"
                "**正牌是单篇 DOI `10.1145/3664647.3680568`**（Crossref 核：type=**proceedings-article**、"
                "pp. **4092–4100**、2024、作者 Li L / Yang W / Yu X / Xing J / **Zhang X**（Xiao-Ping Zhang）"
                "—— **与台账 F11 原记的作者完全一致**）。已更正。"
                "📌 **全文已由用户经学校数据库取回**（原为 Unpaywall `closed`、arXiv 按题名 0 命中）。"
                "**F11a 正文级差异辩护的阻塞由此解除**。",
    },
    {
        "thread": "B1",
        "lens": "M3",
        "relevance": "高",
        "year": 2022,
        "venue": "IEEE Transactions on Pattern Analysis and Machine Intelligence",
        "doi": "10.1109/tpami.2021.3087695",
        "title": "Monocular 3D Pose Estimation via Pose Grammar and Data Augmentation",
        "force": True,
        "selected_from": "k8_orphan_card_recovery_2026-10-03",
        "found_via": ["crossref", "user_school_db"],
        "note": "★ 【= 台账 P0-2；pose grammar 线的正式期刊版；F6/F10 的关键对照】"
                "Xu Y, Wang W, Liu T, Liu X, Xie J, Zhu S。Crossref 核："
                "**IEEE TPAMI 2022;44(10):6327-6344**。"
                "📌 **本条的杀伤力**：它明写引入 kinematics → **作废我们「grammar 后继不含力学」"
                "这条依据**（F6/F10）。**全文已由用户取回**（原 OpenAlex `closed`、arXiv 按题名 0 命中），"
                "**「Biomechanical Grammar 还能不能主张力学那一半」现已可查**。",
    },
    {
        "thread": "B1",
        "lens": "M3",
        "relevance": "高",
        "year": 2023,
        "venue": "IEEE Transactions on Biometrics, Behavior, and Identity Science",
        "doi": "10.1109/tbiom.2023.3315509",
        "title": "Progressive Direction-Aware Pose Grammar for Human Pose Estimation",
        "force": True,
        "selected_from": "k8_orphan_card_recovery_2026-10-03",
        "found_via": ["crossref", "user_school_db"],
        "note": "★ 【= 台账 P0-3；pose grammar 线的第三代；F10 的关键对照】"
                "Zhou L, Chen Y, Wang J。Crossref 核：**IEEE TBME 2023;5(4):593-605**。"
                "**全文已由用户取回**。📌 与 P0-2 合并处置：F10 原判「pose grammar 只是 2018 年"
                "一篇孤立的旧文」已被这两篇推翻，**两者必须在 Related Work 正面处理**。",
    },
    {
        "thread": "B3",
        "lens": "M3",
        "relevance": "高",
        "year": 2023,
        "venue": "Mechatronics",
        "doi": "10.1016/j.mechatronics.2022.102919",
        "title": "Performance-based assistance control for robot-mediated upper-limbs rehabilitation",
        "force": True,
        "selected_from": "k8_orphan_card_recovery_2026-10-03",
        "found_via": ["crossref", "user_school_db"],
        "note": "Zhang L, Guo S, Xi F 等。Crossref 核：**Mechatronics 2023;89:102919**。"
                "机器人辅助上肢康复的**按表现给助力**控制。📌 属 B3 上肢康复线程的"
                "**控制侧**对照（与本项目的视觉测量侧互补，不是竞品）。"
                "⚠️ 孤儿卡原有元数据经 Crossref 核对**正确**，仅作链路收编。",
    },
    {
        "thread": "B1",
        "lens": "M3",
        "relevance": "中",
        "year": 2025,
        "venue": "Journal of Biomechanics",
        "doi": "10.1016/j.jbiomech.2025.112758",
        "title": "Do gait patterns normalize concurrently with the recovery of foot arches in children with flatfoot? A longitudinal study",
        "force": True,
        "selected_from": "k8_orphan_card_recovery_2026-10-03",
        "found_via": ["crossref", "user_school_db"],
        "note": "Zhao Y, Yan S, Zhang L, Shi B, Yang L 等。Crossref 核："
                "**Journal of Biomechanics 2025;187:112758**。"
                "🚨 **相关性判定：中，且是本轮【检出但降级】的一例** —— 儿童扁平足足弓恢复的"
                "纵向步态研究，**属下肢步态**，与本项目「上肢 腕-肘-肩」范围不符。"
                "**之所以仍收编**：它经由另一会话落盘为孤儿卡（K8 要求可复现），"
                "**收编的是「链路可复现」这件事，不是它与本项目相关**。"
                "→ **不进 §0.5 上肢竞品表，不在任何论证中引用。**",
    },
]

ARXIV_EXTRA = [
    ('2605.09258', 'B1', 'M3', '高',
     '★★★ 单目 + 生物力学模型在优化环内的手/指关节角跟踪（SAM 3D Body + MuJoCo-MJX 下的 IK 优化；'
     'Shirley Ryan AbilityLab / Northwestern）。**本条由 snowball 追入** —— 它是 `2607.02796` 参考文献 [50]，'
     '而 `2607.02796` 当时被列为「最近的先行」，其引用网络此前从未追过。'
     '**为什么它比 `2607.02796` 更该警惕**：设定上**同为单目**（该文 monocular 出现 28 次），'
     '而 `2607.02796` 是 8 相机多视角。'
     '🚨 **全文级撞红已做（12 页全文，2026-10-02）**：`token` / `vocabulary` / `codebook` / `symbol` / '
     '`discrete` / `quantiz` / `embedding` / `Transformer` / `grammar` / `language model` / `autoregressive` '
     '**全部零命中**；4 次 `translation` 经逐条查上下文**全部是「root translation / 平移量」，非序列翻译**。'
     '→ **不构成机制级撞车**，但**「单目 + 生物力学约束」这个组合已被占**，写贡献时不可再当新意。'
     '📌 **可引的空白陈述（注意其限定）**：摘要原句 *"monocular video approaches for obtaining **hand** '
     'biomechanics remain **under-developed**"* —— 🚨 **它说的是 hand/finger，不可外推到上肢 ISB 角度**。'),
    ('2402.17192', 'B5', 'M3', '高',
     '★★★ 可微生物力学用于 MMC 的开创工作。是本项目「可微路线」的全部先行，'
     '已决定：该路线降级为验证协议来源，不作为创新点。'),
    ('2411.14992', 'B5', 'M3', '高',
     '★★★ 上肢卒中可微生物力学 + 与光学动捕对比（IEEE TMRB 2026, 8(1):90-97, 15 引）。'
     '本项目直接借用其公开基准：卒中饮水任务、关节角 RMSE 2–5°、r>0.95、15 人。'),
    ('2607.23608', 'B3', 'M4', '高',
     '★★ 临床常规评估中嵌入 MMC（20 患者/47 次/1174 任务），证明连续运动学指标'
     '比 ARAT 序数评分更灵敏 —— 自采 D435i 数据的临床价值论证先例。'),
    ('2603.26844', 'B5', 'M1', '高',
     '★★ 3D 关键点→解剖标志物的不确定性量化（ρ≈0.63，ROC-AUC 0.92）。'
     '离散 token 辅助头做置信度/可拒识别的直接方法学参照。'),
    ('2609.18667', 'B1', 'M0', '高',
     '★★ 临床/康复场景视频 MMC 的 PRISMA-ScR 范围综述（117 项已验证研究，'
     '检索至 2026-07）。本语料 B1 线程的总览锚点，也是「是否已饱和」的判定依据。'),
    ('2607.02796', 'B1', 'M3', '高',
     '★ 生物力学感知的 dexterous hand 多视角 MMC —— 腕/手 DoF 表征的最近先行。'),
    ('2502.06486', 'B5', 'M1', '中',
     '多视角 MMC 的生物力学重建与置信区间 —— 不确定性传递的下游版本。'),
    ('2602.13176', 'B3', 'M4', '中',
     '单目 MMC 用于上肢可达空间的定量评估 —— 单目路线 + 上肢的可行性证据。'),
    ('2303.10654', 'B1', 'M3', '中',
     'MMC 与生物力学分析流水线 —— 工程基线。'),
    ('2010.06449', 'B1', 'M0', '中',
     '标记无关动捕用 3D 人体姿态估计算法综述 —— 计算机视觉侧的方法版图。'),
    # ---- 2026-10-01 第八/九轮引文追溯（V9e / gen2）新捞出的撞车工作 ----
    # 这四条**不在 discover 的检索池里**（是被 snowballing 从参考文献里翻出来的），
    # 但必须进语料：**撞红工作不进索引，等于查重没有留证**。
    ('1710.06513', 'B4', 'M3', '高',
     '★★★ 【F6 撞车】pose grammar 编码人体构型（另见 AAAI 2018, 10.1609/aaai.v32i1.12270）。'
     '定义 kinematic/symmetry/coordination 三种 grammar、共 9 条手写规则、实现为 tree-BRNN。'
     '**原文自述「Kinematic grammar describes human body movements WITHOUT CONSIDERING FORCES」** —— '
     '明确排除力学，是本项目主张 A（原句级差异）的直接依据。'
     '⚠️ 其「用关节链约束姿态」已被占且已含 shoulder↔elbow↔wrist 上肢链，不得再作创新点宣称。'
     '⚠️ 发现途径：引文后向追溯 —— 12 条式子检索全部漏掉，因其用词是 pose grammar 而非 vocabulary/token。'),
    ('2608.03127', 'B4', 'M3', '高',
     '★★★ 【F5 撞车】DigitCode：用 Hand Labanotation 作字母表，把手部运动符号化为 T×40 网格、每骨一方向符号。'
     '「离散符号化关节表征」已被占。差异：手部/舞谱字母表/非视觉输入/无生物力学 grammar/非临床。'
     '它提出「符号该覆盖骨、指还是整手一层」的问题 —— 正好对应本项目 V6c 链拓扑，是可正面引述的方法论动机。'),
    ('2601.09949', 'B4', 'M3', '低',
     '⚠️ 【命名占用】Kinematic Tokenization —— 标题与本项目高度相似，**实为金融时间序列**'
     '（资产价格、样条系数、交易决策）。列入是为标记：**该方法名已被占用，论文中不可直接用作方法名**，'
     '否则检索会与此文混淆。领域不撞。'),
    ('2603.09930', 'B4', 'M3', '中',
     'Joint-Angle Motion Images + MaxSim token 交互 + MLM 正则，做文本-动作检索。'
     '相邻但不同：是把关节特征映成结构化**伪图像**喂 ViT，**不是离散符号词表**。'
     '⚠️ 但若本项目采用 MLM 式训练目标，须正面区分此文。'),
]


def main():
    sl = json.loads(SHORTLIST.read_text(encoding='utf-8'))
    by_thread = {t['id']: t for t in sl['threads']}

    # 回退索引：短名单只是裁定辅助，不是硬门槛。
    # 短名单受 venue 配额与模糊去重影响，会把「同刊物但同样重要」的条目挤掉
    # （实测：J Biomech 配额把「biplane videoradiography 腕运动学精度」挤掉）。
    # 判别式若在短名单未命中，则回落到候选池再找一次；只有两处都没有才算未命中。
    pool = json.loads((HERE / 'biomech_pool.json').read_text(encoding='utf-8'))
    by_pool = {t['id']: t for t in pool['threads']}

    def lookup(tid, key):
        t = by_thread.get(tid)
        if t:
            hit = next((b for b in t['shortlist'] if key.lower() in b['title'].lower()), None)
            if hit:
                return hit, 'shortlist'
        for cand_tid in (tid, *[x for x in by_pool if x != tid]):
            t2 = by_pool.get(cand_tid)
            if not t2:
                continue
            hit = next((b for b in t2['items'] if key.lower() in b['title'].lower()), None)
            if hit:
                return hit, 'pool:%s' % cand_tid
        return None, '未命中'

    papers, missed = [], []
    for i, (tid, key, lens, rel, note) in enumerate(PICKS):
        hit, via = lookup(tid, key)
        if hit is None:
            missed.append((tid, key, '短名单与候选池均未命中'))
            continue
        papers.append({
            'idx': len(papers),
            'title': hit['title'],
            'year': hit.get('year'),
            'doi': hit.get('doi', ''),
            'venue': hit.get('venue', ''),
            'thread': tid,
            'lens': lens,
            'relevance': rel,
            'note': note,
            # 以下为人工裁定时的**检索期证据**，不是 arXiv 官方字段
            'found_via': hit.get('sources', []),
            'found_cited_by': hit.get('cited_by', 0),
            'found_oa': hit.get('is_oa'),
            'selected_from': via,
        })
        if key in ARXIV_HINTS:
            # 把 arXiv ID 绑到这条上，ARXIV_EXTRA 里的同篇就会在去重时被跳过
            papers[-1]['arxiv_id'] = ARXIV_HINTS[key]

    for aid, tid, lens, rel, note in ARXIV_EXTRA:
        if any(p.get('arxiv_id') == aid for p in papers):
            continue
        papers.append({
            'idx': len(papers),
            'title': '',                     # 由 resolve_multisource.py 从 arXiv 官方记录填入
            'arxiv_id': aid,
            'year': None,
            'doi': '',
            'venue': '',
            'thread': tid,
            'lens': lens,
            'relevance': rel,
            'note': note,
            'found_via': ['arxiv-direct'],
            'found_cited_by': 0,
            'found_oa': True,
            'selected_from': 'arxiv_priority',
        })

    # ---------------------------------------------------------------
    # PRIOR_MANUAL —— 保住**不走 PICKS/候选池**通道的既有裁定
    #
    # 🚨 2026-10-02 起因（K22）：本脚本是**破坏性重建** —— candidates.json 完全由
    #    PICKS + 候选池 + ARXIV_EXTRA 重新生成，**凡是通过别的路径加进来的条目一律消失**。
    #    实测：一次重跑把下列 3 条**静默删掉**，而其中一条是 ISB Part II
    #    （`10.1016/j.jbiomech.2004.05.042`）—— **本项目腕 DoF 定义的唯一权威**。
    #    它们当初的 selected_from 就说明了来源不在本脚本内：
    #      · manual_recovery_from_orphan_card  （从孤儿定位卡人工恢复）
    #      · snowball_gen2_flagged_20           （第二代引文追溯坐实）
    #      · snowball_gen3_flagged_then_aaai_oa_channel（第三代 + AAAI OA 通道）
    #    📌 修法不是「记得别重跑」，而是**让重跑无损**：把它们连同完整元数据固化在此。
    # ---------------------------------------------------------------
    # component DOI 归一（K7 防线）：在写盘前统一折回正牌标识
    for _p in papers:
        if _p.get('doi'):
            _fixed = normalize_doi(_p['doi'])
            if _fixed != _p['doi']:
                _p['doi'] = _fixed
    prior_keys = {(p.get('doi') or '').lower(): p for p in papers}
    forced = 0
    for rec in PRIOR_MANUAL:
        key = rec['doi'].lower()
        if key in prior_keys:
            # K35：PRIOR_MANUAL 原本是「保命」机制 —— 已在池中的条目**整条跳过**，
            # 于是人工裁定的线程/范式/理由**全部失效**，且不产生任何告警。
            # 2026-10-03 实测：6 条用户取回的论文早已由 shortlist 通道入库，
            # 补写的裁定理由（ICC 数值、ROM 实测、DOI 冲突）一条都没落进数据。
            # 修法：默认行为**不变**（避免动既有基线），仅当记录显式标 force=True 时覆写。
            if not rec.get('force'):
                continue
            tgt = prior_keys[key]
            changed = []
            for f in ('thread', 'lens', 'relevance', 'venue', 'year', 'title', 'note'):
                if f in rec and rec[f] != tgt.get(f):
                    changed.append('%s: %r → %r' % (f, tgt.get(f), rec[f]))
                    tgt[f] = rec[f]
            tgt['selected_from'] = rec.get('selected_from', tgt.get('selected_from'))
            forced += 1
            print('  🔁 覆写既有条目 %s' % rec['doi'])
            for c in changed:
                print('       %s' % c)
            continue
        rec = dict(rec)
        rec.pop('force', None)
        rec['idx'] = len(papers)
        rec.setdefault('arxiv_id', '')
        rec.setdefault('authors', [])
        rec.setdefault('abstract', '')
        rec.setdefault('found_cited_by', 0)
        rec.setdefault('found_oa', True)
        papers.append(rec)
        prior_keys[rec['doi'].lower()] = rec
    if forced:
        print('  （PRIOR_MANUAL 覆写 %d 条；未标 force 的既有条目一律保持原样）' % forced)

    seeds = json.loads((HERE / 'biomech_seeds.json').read_text(encoding='utf-8'))
    nq = sum(len(t['queries']) for t in seeds['threads'])
    pool_n = sum(t['dedup'] for t in pool['threads'])

    out = {
        'meta': {
            'corpus_id': 'biomech',
            'title': seeds['meta']['title'],
            'purpose': seeds['meta']['purpose'],
            'why_new_corpus': seeds['meta']['why_new_corpus'],
            'source_policy': seeds['meta']['source_policy'],
            'axes': {'lens': LENS, 'thread': {t['id']: t['name'] for t in seeds['threads']}},
            'relevance': {'高': '直接支撑既定研究决定或构成新颖性威胁', '中': '方法或论证可借鉴',
                          '低': '背景/里程碑，迁移间接'},
            'counts_strictly_separated': {
                '理论搜索空间_检索式条数': nq,
                '实际命中_候选池去重条数': pool_n,
                '人工裁定_候选清单条数': len(papers),
                '实际入库_PDF数': '待下载后由下载报告复算，不得在此预填',
            },
            'curation': '全部条目由 select_candidates.py 的 PICKS 表人工裁定；'
                        '每条必须能在短名单内被判别式唯一命中，否则计入 missed 并须回修。',
        },
        'papers': papers,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')

    from collections import Counter
    print('候选清单已写出：%s' % OUT)
    print('  理论搜索空间（检索式）      : %d' % nq)
    print('  实际命中（候选池去重）      : %d' % pool_n)
    print('  人工裁定（候选清单）        : %d' % len(papers))
    print('  按线程 : %s' % dict(Counter(p['thread'] for p in papers)))
    print('  按范式 : %s' % dict(Counter(p['lens'] for p in papers)))
    print('  按相关度: %s' % dict(Counter(p['relevance'] for p in papers)))
    if missed:
        print('\n⚠ 未命中的判别式 %d 条（必须回修，不得静默跳过）：' % len(missed))
        for tid, key, why in missed:
            print('   - [%s] %s → %s' % (tid, key[:70], why))


if __name__ == '__main__':
    main()
