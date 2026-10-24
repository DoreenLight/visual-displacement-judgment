#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 01_参考论文/ 根层散落的新下载 PDF 归位到三分法目录，并去重。

只做归置：不改名内容、不改字节（去重删除的是 SHA 完全相同的重复下载）。
目录名格式沿用 09_/12_ 团队语料风格：<首作者姓><年>-<刊缩写>：<中文短题>
中文短题由本脚本从**已核实英文题名**拟定，写在下方 TITLES 中，可逐条复核。
"""
import hashlib
import json
import shutil
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPERS = ROOT / '01_参考论文'
OUTMAP = Path(__file__).parent / '归位映射.json'

# doi -> (目标目录相对 01_参考论文/, 中文短题)
TITLES = {
    # ---------- 语料三 生物力学线（放回原线程，使下载报告转 ok）----------
    '10.1016/s0268-0033(00)00082-6':
        ('整合系统/生物力学线/B1_视觉运动学_估计与验证方法学', '运动学与动力学报告的关节坐标系一致性'),
    '10.1016/j.jse.2007.02.134':
        ('整合系统/生物力学线/B1_视觉运动学_估计与验证方法学', '肘关节量角测量重测信度的双盲方案'),
    '10.1016/j.jht.2020.05.001':
        ('整合系统/生物力学线/B1_视觉运动学_估计与验证方法学', '非受训者用 iPhone 应用自测腕关节活动度的信度'),
    '10.1177/026921559701100408':
        ('整合系统/生物力学线/B1_视觉运动学_估计与验证方法学', '手部关节角度测量中量角器与描线法的信度比较'),
    '10.1016/S0031-9406(10)63103-3':
        ('整合系统/生物力学线/B1_视觉运动学_估计与验证方法学', '肩关节角目视估计与量角器测定的比较'),
    '10.1007/978-981-16-4991-2_8':
        ('整合系统/生物力学线/B1_视觉运动学_估计与验证方法学', '肩复合体、肘与腕的运动学与动力学'),
    '10.1016/0021-9290(94)90062-0':
        ('整合系统/生物力学线/B1_视觉运动学_估计与验证方法学', '网球反手击球中专家与新手的手腕运动学差异'),
    '10.1016/0021-9290(96)00031-0':
        ('整合系统/生物力学线/B2_肌肉骨骼动力学_与可微仿真', '人体上肢肌肉力臂与长度随关节角的变化'),
    '10.1034/j.1600-0404.2003.00021.x':
        ('整合系统/生物力学线/B3_上肢康复_代偿与临床评估', '卒中患者伸手运动的代偿与恢复'),
    '10.1016/s0003-9993(99)90088-6':
        ('整合系统/生物力学线/B3_上肢康复_代偿与临床评估', '胸廓位置对肩活动度、肌力与肩胛三维运动学的影响'),

    # ---------- 宋嵘团队：运动理解 ----------
    '10.1109/TBME.2024.3368021':
        ('运动理解/宋嵘_意图解码与力矩模型', '基于分舱 HD-sEMG 肌骨模型的等长跖屈力矩预测'),
    '10.1109/TMRB.2026.3682646':
        ('运动理解/宋嵘_意图解码与力矩模型', '用于康复机器人控制的实用多模型无扰切换 sEMG 解码器'),
    '10.1109/TNSRE.2024.3371465':
        ('运动理解/宋嵘_意图解码与力矩模型', '基于变遗忘因子递归最小二乘的自适应 Hammerstein FES 力矩预测模型'),
    '10.1109/TNSRE.2021.3066592':
        ('运动理解/宋嵘_意图解码与力矩模型', '基于无扰切换机制的康复机器人人体运动意图描述'),
    '10.1109/TNSRE.2019.2949203':
        ('运动理解/宋嵘_意图解码与力矩模型', '通过切换机制连续描述人体三维运动意图'),

    # ---------- 宋嵘团队：康复辅助与训练强度 ----------
    '10.1109/TNSRE.2020.2986295':
        ('康复辅助/宋嵘_辅助与训练强度', '卒中患者 FES 辅助跑台训练中刺激时相与强度的迭代调整'),
    '10.3389/fbioe.2023.1246014':
        ('康复辅助/宋嵘_辅助与训练强度', '纠正卒中后偏瘫步态的功能性电刺激混合与自适应控制'),
    '10.1002/aisy.202400505':
        ('康复辅助/宋嵘_辅助与训练强度', '基于高斯混合模型的上肢线缆驱动康复机器人肌电控制'),
    '10.1109/TNSRE.2023.3348332':
        ('康复辅助/宋嵘_辅助与训练强度', '面向可穿戴下肢康复机器人的轨迹变形多模态自适应柔顺控制'),
    '10.1109/TBME.2020.3012296':
        ('康复辅助/宋嵘_辅助与训练强度', '健康人与卒中患者用肌电导纳控制踝关节外骨骼'),

    # ---------- 宋嵘团队：双参与与肌间耦合 ----------
    '10.1109/TNSRE.2025.3535928':
        ('整合系统/宋嵘_双参与与肌间耦合', '卒中患者肘等长收缩时皮层-肌电耦合改变与临床评分相关'),
    '10.1088/1741-2552/ac18ae':
        ('整合系统/宋嵘_双参与与肌间耦合', '肌电控制接口调节下卒中皮层-运动系统多层信息传递的改变'),
    '10.1088/1741-2552/abc29a':
        ('整合系统/宋嵘_双参与与肌间耦合', '用二维肌电接口调制并恢复卒中患者的肌间协调'),
    '10.1109/TNSRE.2023.3238758':
        ('整合系统/宋嵘_双参与与肌间耦合', '卒中患者步行中肌电活动与肌间耦合的相位依赖调制'),
    '10.1109/tnsre.2021.3101615':
        ('整合系统/宋嵘_双参与与肌间耦合', '用 Cross-Fuzzy Entropy 研究卒中与老龄相关的全局与瞬时肌间耦合变化'),

    # ---------- 章隆彬团队（其余 5 篇在 09_ 内，不重复取）----------
    '10.1109/TFUZZ.2016.2566676':
        ('整合系统/章隆彬_理解与辅助结合', '脑机接口与视觉压缩感知的外骨骼机器人遥操作控制'),
}

VENUE_ABBR = {
    'IEEE Transactions on Neural Systems and Rehabilitation Engineering': 'TNSRE',
    'IEEE Transactions on Biomedical Engineering': 'TBME',
    'IEEE Transactions on Medical Robotics and Bionics': 'TMRB',
    'IEEE Transactions on Fuzzy Systems': 'TFUZZ',
    'Journal of Neural Engineering': 'JNeuralEng',
    'Advanced Intelligent Systems': 'AdvIntellSys',
    'Frontiers in Bioengineering and Biotechnology': 'FrontBioengBiotech',
    'Clinical Biomechanics': 'ClinBiomech',
    'Journal of Biomechanics': 'JBiomech',
    'Journal of Shoulder and Elbow Surgery': 'JSESurg',
    'Journal of Hand Therapy': 'JHandTher',
    'Clinical Rehabilitation': 'ClinRehab',
    'Physiotherapy': 'Physiotherapy',
    'Archives of Physical Medicine and Rehabilitation': 'ArchPMR',
    'Acta Neurologica Scandinavica': 'ActaNeurolScand',
    'Conceptual Biomechanics and Kinesiology': 'ConceptBiomechKinesiol',
}


def crossref_author(doi):
    req = urllib.request.Request('https://api.crossref.org/works/' + doi,
                                 headers={'User-Agent': 'kb-file-tidy/1.0 (mailto:local)'})
    m = json.load(urllib.request.urlopen(req, timeout=30))['message']
    venue = (m.get('container-title') or [''])[0]
    year = (m.get('published-print') or m.get('published-online')
            or m.get('issued') or {}).get('date-parts', [[None]])[0][0]
    authors = m.get('author') or []
    last = authors[0].get('family', '').strip() if authors else ''
    return last, year, VENUE_ABBR.get(venue, venue[:14])


def norm(s):
    import unicodedata
    return ''.join(c for c in unicodedata.normalize('NFKD', s).casefold() if c.isalnum())


def main():
    import subprocess
    flat = sorted(PAPERS.glob('*.pdf'))
    txt = {}
    for p in flat:
        try:
            txt[p] = subprocess.run(['pdftotext', '-f', '1', '-l', '2', str(p), '-'],
                                   capture_output=True, text=True, timeout=30).stdout
        except Exception:
            txt[p] = ''

    # 1) SHA 去重：保留文件名较规范（短、无下划线堆砌）的一份
    bysha = {}
    for p in flat:
        bysha.setdefault(hashlib.sha256(p.read_bytes()).hexdigest(), []).append(p)
    dropped = []
    for h, ps in bysha.items():
        if len(ps) > 1:
            keep = min(ps, key=lambda q: (len(q.name), q.name))
            for p in ps:
                if p != keep:
                    p.unlink()
                    dropped.append({'dropped': p.name, 'kept': keep.name, 'sha256': h})
                    print('  去重删除重复下载:', p.name[:60])

    # 2) 归位
    ledger = json.loads((ROOT / '10_研究方向/主动参与与训练强度_链路研究/文献台账.json').read_text('utf-8'))
    bydoi = {}
    for r in ledger['papers']:
        if r['id'].startswith('DL-'):
            bydoi[r['doi'].lower()] = r
    moved, missing = [], []
    for doi, (reldir, zh) in TITLES.items():
        rec = bydoi.get(doi.lower())
        probe = norm(rec['title'])[:50] if rec else norm(doi)
        src = next((p for p in flat if p.exists() and probe in norm(txt[p])), None)
        if src is None and rec:
            probe2 = norm(rec['title'].split(':')[0])[:40]
            src = next((p for p in flat if p.exists() and probe2 in norm(txt[p])), None)
        if src is None:
            missing.append(doi)
            print('  缺文件:', doi)
            continue
        last, year, venue = crossref_author(doi)
        time.sleep(0.3)
        dirname = f'{last}{year}-{venue}：{zh}' if last else f'{venue}{year}：{zh}'
        target = PAPERS / reldir / dirname
        target.mkdir(parents=True, exist_ok=True)
        dst = target / f'{dirname}.pdf'
        shutil.move(str(src), str(dst))
        moved.append({'doi': doi, 'from': src.name, 'to': str(dst.relative_to(ROOT)),
                      'dir': dirname, 'zh_title': zh, 'corpus_target': reldir})
        print(f'  OK  {dst.relative_to(ROOT)}')

    OUTMAP.write_text(json.dumps({'moved': moved, 'dropped_duplicates': dropped,
                                  'still_missing': missing}, ensure_ascii=False, indent=1), 'utf-8')
    print(f'\n归位 {len(moved)} 篇；去重删除 {len(dropped)} 个；仍缺 {len(missing)} 个：{missing}')
    print('映射 ->', OUTMAP.relative_to(ROOT))


if __name__ == '__main__':
    main()