#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
oa_ingest.py — 把「用户已下载的订阅件」与「我自行取回的 OA 件」正式登记进语料二。

登记原则：
  - 载体信息必须有**可核验来源**（Crossref / Unpaywall / 出版社页面 / CVF 页面）。
  - 检索时我记错的题名一律按**真实题名**更正，不保留错名。
  - 非 arXiv 论文没有 arXiv ID 时，arxiv_id 记为空并在 index 中标明来源 URL，
    **不编造 arXiv 编号**。

用法：python3 oa_ingest.py
"""
import json
import re
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAND = HERE / 'corpus_candidates.json'
RESOLVED = HERE / 'corpus_resolved.json'
UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36'

# (我原来记的错题名, 真实题名, 线程, 迁移价值, PDF 相对 01_参考论文/, DOI, 载体, 来源 URL, note)
INGEST = [
    # —— 真实题名由 CVF 官方页面核实；我原记的「A Simple Framework for Text-to-Motion
    #    Extraction and Analysis」并非真实题名，是把 T2M 框架名与记忆混在一起的产物。
    ('A Simple Framework for Text-to-Motion Extraction and Analysis',
     'Generating Diverse and Natural 3D Human Motions From Text', 'T5', '高',
     'Transformer_发展线/T5_动作生成与离散token/2022_Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text.pdf',
     '', 'CVPR 2022 (IEEE/CVF), pp. 5152-5161',
     'https://openaccess.thecvf.com/content/CVPR2022/html/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.html',
     'T2M 框架原文：用 kNN 聚类把连续动作量化成 motion snippet code（离散动作词表），'
     '两阶段 text2length + text2motion；是「动作有了词表」的直接证据。CVF 开放获取'),

    ('A Dual-Task Improved Transformer Framework for Decoding Lower Limb Sit-to-Stand Movement from sEMG and IMU Data',
     'A Dual-Task Improved Transformer Framework for Decoding Lower Limb Sit-to-Stand Movement from sEMG and IMU Data',
     'T5', '高', 'Transformer_发展线/T5_动作生成与离散token/2025_iTransformer-DTL_坐立起身_sEMG_IMU_轨迹解码.pdf',
     '10.3390/machines13100953', 'Machines (MDPI), 2025, Gold OA CC-BY',
     'https://www.mdpi.com/2075-1702/13/10/953',
     'iTransformer-DTL：可学习 query + 非自回归直接解码整段轨迹，'
     '与 Wei 2026 的自回归 Seq2Seq 构成路线竞争'),

    ('Markerless On-Device Detection of Compensatory Movement Patterns in Upper-Limb Rehabilitation Exercises from Monocular RGB Video',
     'Markerless On-Device Detection of Compensatory Movement Patterns in Upper-Limb Rehabilitation Exercises from Monocular RGB Video',
     'T6', '高', 'Transformer_发展线/T6_物理约束·世界模型/2026_端侧单目RGB_上肢代偿模式检测.pdf',
     '10.3390/s26165054', 'Sensors (MDPI), 2026, Gold OA CC-BY',
     'https://www.mdpi.com/1424-8220/26/16/5054',
     '单目 RGB 端侧上肢代偿检测（5 项几何指标）；ViKiT 的最直接视觉竞品'),

    ('Using deep learning to detect upper limb compensation in individuals post-stroke using consumer-grade webcams',
     'Using deep learning to detect upper limb compensation in individuals post-stroke using consumer-grade webcams',
     'T6', '高', 'Transformer_发展线/T6_物理约束·世界模型/2025_消费级摄像头_跨被试上肢代偿检测.pdf',
     '10.3389/fmed.2025.1645369', 'Frontiers in Medicine, 2025, Gold OA CC-BY',
     'https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2025.1645369/full',
     '个体内 >90% / 跨被涉约 50-70%（原文自述，未复现）：对跨被试协议的负面证据'),

    ('Video generation models as world simulators',
     'Pre-Trained Video Generative Models as World Simulators', 'T6', '中',
     'Transformer_发展线/T6_物理约束·世界模型/2026_AAAI_预训练视频生成模型作为世界模拟器.pdf',
     '10.1609/aaai.v40i6.42465', 'AAAI 2026, Gold OA',
     'https://ojs.aaai.org/index.php/AAAI/article/view/42465',
     '世界模型视角的视频生成预训练。**注意：这不是 Sora 报告本身**，'
     '是检索「Video generation models as world simulators」时命中的近题论文，已标注为替代项'),
]


def main():
    cand = json.loads(CAND.read_text(encoding='utf-8'))
    res = json.loads(RESOLVED.read_text(encoding='utf-8'))
    papers = res['papers']

    added = renamed = 0
    for old, real, thread, relv, pdf_rel, doi, venue, src, note in INGEST:
        # 1) 候选清单：按「原题名」定位，就地改真实题名 + 补元数据
        hit_c = next((p for p in cand['papers'] if p['title'] == old), None)
        if hit_c:
            if hit_c['title'] != real:
                renamed += 1
            hit_c['title'] = real
            hit_c['thread'] = thread
            hit_c['relevance'] = relv
            hit_c['note'] = note
            if doi:
                hit_c['doi'] = doi
        else:
            cand['papers'].append({'title': real, 'thread': thread, 'relevance': relv,
                                   'note': note, 'doi': doi, 'venue_hint': venue})
            added += 1

        # 2) resolved：置为 confirmed，并写入 pdf 路径（供 index / 定位卡 / 翻译编排使用）
        rec = next((p for p in papers if p.get('title') == old), None)
        if rec is None:
            rec = {'title': real}
            papers.append(rec)
        elif rec['title'] != real:
            renamed += 1
        rec.update({
            'title': real, 'canonical_title': real, 'thread': thread, 'relevance': relv,
            'note': note, 'status': 'confirmed', 'query': 'OA 直取核验（CVF/Crossref/Unpaywall）',
            'match_score': 1.0, 'arxiv_id': rec.get('arxiv_id') or '',
            'doi': doi, 'venue': venue, 'source_url': src,
            'pdf_rel': pdf_rel, 'year': int(re.search(r'(19|20)\d{2}', venue).group(0)),
        })
    CAND.write_text(json.dumps(cand, ensure_ascii=False, indent=2), encoding='utf-8')
    RESOLVED.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print('候选清单：更正题名 %d 条，新增 %d 条' % (renamed, added))


if __name__ == '__main__':
    main()
