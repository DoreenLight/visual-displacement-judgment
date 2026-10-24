#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""研究方向视图层：把散在四条语料里的文献，按**研究方向**重组。

与语料目录（01_参考论文/下按来源分线）的区别：
  - 语料目录回答「这篇文献是哪来的」
  - 本视图回答「我要做这个方向，该读哪些文献」

结构：10_研究方向/<方向编号_方向名>/
        README.md   方向说明（现状、缺口、与本项目的关系）
        建议.md     编者建议（该做什么、不该做什么、证伪判据）
        语料/       软链接 → 原始 PDF，不复制文件

**不复制 PDF**：全部软链。理由：同一篇论文可能被多个方向引用，
复制会产生多份副本，改名/重取件时必然漂移。08_三维动作分析展示台/07_资料库 已用同样做法。
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DEST = ROOT / '10_研究方向'
IDX = ROOT / '01_参考论文' / '决策模型线_JEV' / 'index.json'

# 方向定义：编号 -> (名称, 一句话定位, 语料键列表)
DIRECTIONS = {
    'R1': ('AAN介入决策', '判断什么时候介入、介入多少、不确定时怎么办', ['R1']),
    'R2': ('运动意图回归主干', '3DCNN-TF 主干，对标基线，暂不改', ['R2']),
    'R3': ('轻量化与端侧部署', '延迟与体积预算，含关键反例', ['R3']),
    'R4': ('跨被试泛化与个体化', '已知难点：跨被试只有 50-70%', ['R4']),
    'R5': ('来源核验与证据分级', '方法论：三次来源失真如何被拦下', ['R5']),
}

# 语料键 -> 定位卡里出现过的论文匹配串
TITLES = {
    'when_to_act': 'When to Act',
    'hri2026': 'Human-in-the-Loop Confidence-Aware Failure Recovery',
    'mpc_aan': 'Model Predictive Control-Based Assist-as-Needed',
    'confidence_routing': 'Confidence-Aware Routing',
    'ar2609_28940': 'Calibrated Decision Models for Autonomous Penetration',
    'konak': 'More Features Are Not More Evidence',
    'song_tcn': 'Personalization of Wearable Sensor-Based Joint Kinematic',
    'ude': 'Unified Driving Engine',
}

# 每个方向收哪些条目（用 index.json 的标题子串匹配）
MEMBERSHIP = {
    'R1': ['When to Act', 'Human-in-the-Loop Confidence-Aware', 'Model Predictive Control-Based',
           'Confidence-Aware Routing', 'Calibrated Decision Models'],
    'R2': ['Unified Driving Engine'],
    'R3': ['More Features Are Not More Evidence', 'Calibrated Decision Models'],
    'R4': ['Personalization of Wearable Sensor-Based'],
    'R5': ['More Features Are Not More Evidence', 'Calibrated Decision Models',
           'Confidence-Aware Routing'],
}

# 额外链接：语料二 / 语料三 / 主语料 中与方向相关的 PDF（手工指定相对路径）
EXTRA = {
    'R1': [
        ('sEMG_原始参照论文/Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测/'
         'Wei2026-JBHI：肌肉解剖与 Seq2Seq 翻译的 sEMG 关节运动学预测.pdf', 'Wei2026_JBHI_种子论文'),
    ],
    'R2': [
        ('01_参考论文/Transformer_发展线/T5_动作生成与离散token/'
         '2022_Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text.pdf', 'T2M_CVPR2022'),
        ('01_参考论文/Transformer_发展线/T5_动作生成与离散token/'
         '2023_Zhang_T2M_GPT_Generating_Human_Motion_from_Textual.pdf', 'T2M-GPT_arXiv2301.06052'),
        ('01_参考论文/Transformer_发展线/T1_架构与注意力机制/'
         'A_BERT_Based_Method_for_Continuous_Estimation_of_Cross-Subject_Hand_Kinematics_From_'
         'Surface_Electromyographic_Signals.pdf', 'Lin2023_TNSRE_sBERT_跨被试手部运动学'),
    ],
    'R3': [
        ('01_参考论文/Transformer_发展线/T5_动作生成与离散token/'
         '2025_iTransformer-DTL_坐立起身_sEMG_IMU_轨迹解码.pdf', 'iTransformer-DTL_可学习query非自回归'),
    ],
    'R4': [
        ('01_参考论文/Transformer_发展线/T1_架构与注意力机制/'
         'A_BERT_Based_Method_for_Continuous_Estimation_of_Cross-Subject_Hand_Kinematics_From_'
         'Surface_Electromyographic_Signals.pdf', 'Lin2023_TNSRE_sBERT_跨被试手部运动学'),
        ('01_参考论文/Transformer_发展线/T6_物理约束·世界模型/'
         '2025_消费级摄像头_跨被试上肢代偿检测.pdf', 'Unger2025_Frontiers_跨被试代偿50-70%'),
    ],
    'R5': [],
}


def safe(s, n=60):
    s = re.sub(r'[\\/:*?"<>|]', '_', (s or '').strip())
    return re.sub(r'\s+', ' ', s)[:n].rstrip()


def main():
    idx = json.loads(IDX.read_text(encoding='utf-8'))
    papers = [p for p in idx['papers'] if p.get('pdf_path')]
    DEST.mkdir(exist_ok=True)
    made = []
    for code, (name, tagline, _) in DIRECTIONS.items():
        d = DEST / ('%s_%s' % (code, name))
        corpus = d / '语料'
        corpus.mkdir(parents=True, exist_ok=True)
        n = 0
        # 语料四内部条目
        for p in papers:
            if not any(k in p['title'] for k in MEMBERSHIP[code]):
                continue
            src = (ROOT / p['pdf_path']).resolve()
            if not src.exists():
                continue
            link = corpus / src.name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(src)
            n += 1
        # 跨语料条目
        for rel, alias in EXTRA[code]:
            src = (ROOT / '01_参考论文' / rel) if not rel.startswith(('01_', '08_', '09_')) else (ROOT / rel)
            if not src.exists():
                continue
            link = corpus / ('%s.pdf' % alias)
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(src.resolve())
            n += 1
        made.append((code, name, tagline, n, d))
    print('研究方向视图已生成：')
    for code, name, tagline, n, d in made:
        print('  %s_%s  软链 %d 条  %s' % (code, name, n, tagline))
    print('  目录：%s' % DEST.relative_to(ROOT))


if __name__ == '__main__':
    main()
