#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把「之前无法下载、现在可取」的论文并入唯一台账，随后由 gen_acquisition_list.py 生成清单。

只做一件事：为 33 条已核实 DOI 的论文写入台账记录。
元数据（标题/年份/期刊/出版商）一律取 Crossref，不手写、不猜填。
DOI 纠正会写进 version_note，便于留证。
"""
import hashlib
import json
import shutil
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / '10_研究方向/主动参与与训练强度_链路研究/文献台账.json'
BACKUP = ROOT / '11_归档/_backup_20261008_2155/文献台账_并入待取件前.json'

# (doi, 语料归属, 来源, 优先级, 线程, role, 备注)
ITEMS = [
    # ---- 语料三 生物力学线：confirmed 但全库无全文（此前 Unpaywall 无 OA 直链 / 受限）----
    ('10.1016/j.mechatronics.2022.102919', '生物力学线', '语料三下载报告', 'P0', 'B3', 'assistance',
     '上肢机器人表现型辅助控制，与本轮康复辅助段直接相关；此前状态 unknown，未取得全文'),
    ('10.1016/s0268-0033(00)00082-6', '生物力学线', '语料三下载报告', 'P1', 'B1', 'understanding',
     '上肢关节坐标系一致性，视觉运动学报告口径依据；此前状态 unknown'),
    ('10.1016/j.jse.2007.02.134', '生物力学线', '语料三下载报告', 'P1', 'B1', 'understanding',
     '肘关节量角器重测信度，腕肘角度估计的测量误差基线；此前状态 unknown'),
    ('10.1016/j.jht.2020.05.001', '生物力学线', '语料三下载报告', 'P1', 'B1', 'understanding',
     '腕 ROM 远程自测，腕关节专项的可迁移性参照；此前状态 unknown'),
    ('10.1177/026921559701100408', '生物力学线', '语料三下载报告', 'P1', 'B1', 'understanding',
     '关节角度测量方法比较信度；此前状态 unknown'),
    ('10.1016/S0031-9406(10)63103-3', '生物力学线', '语料三下载报告', 'P2', 'B1', 'understanding',
     '目视估计与量角器对照，视觉角度估计的评价参照；此前状态 unknown'),
    ('10.1016/0021-9290(96)00031-0', '生物力学线', '语料三下载报告', 'P1', 'B2', 'understanding',
     '上肢肌肉力臂与长度，肌骨动力学建模输入；此前 not_oa'),
    ('10.1016/0021-9290(94)90062-0', '生物力学线', '语料三下载报告', 'P2', 'B1', 'understanding',
     '腕运动学（专家/新手），腕关节专项背景；此前 not_oa'),
    ('10.1016/s0003-9993(99)90088-6', '生物力学线', '语料三下载报告', 'P2', 'B3', 'understanding',
     '胸廓位置对上肢 ROM 与肌力的影响，代偿评估的干扰因素；此前 not_oa'),
    ('10.1034/j.1600-0404.2003.00021.x', '生物力学线', '语料三下载报告', 'P0', 'B3', 'understanding',
     '卒中伸手运动的代偿与恢复，直接对应上肢代偿检测；此前状态 blocked'),
    ('10.1007/978-981-16-4991-2_8', '生物力学线', '语料三下载报告', 'P2', 'B1', 'understanding',
     '肩肘腕运动学与动力学综述章；此前 not_oa'),

    # ---- 章隆彬团队：此前 IEEE 反爬/接受稿超时，未取得原文 ----
    ('10.1109/TMECH.2020.3044289', '章隆彬_合作团队', '09 章隆彬 PAPER_GAP_LIST', '高', '结合', 'assistance',
     'Z21；下肢外骨骼人在环控制，连接理解与执行控制；此前 IEEE HTTP 418'),
    ('10.1109/TFUZZ.2016.2566676', '章隆彬_合作团队', '09 章隆彬 PAPER_GAP_LIST', '高', '结合', 'assistance',
     'Z22；脑机接口与视觉压缩感知遥操作；此前 IEEE HTTP 418。源清单标注 TFS，Crossref 核实为 IEEE Transactions on Fuzzy Systems'),
    ('10.1109/TIE.2016.2624260', '章隆彬_合作团队', '09 章隆彬 PAPER_GAP_LIST', '高', '结合', 'assistance',
     'Z20；扰动观测器与自适应神经网络变刚度控制；此前 IEEE HTTP 418'),
    ('10.1109/ICARM52023.2021.9536189', '章隆彬_合作团队', '09 章隆彬 PAPER_GAP_LIST', '中', '理解', 'understanding',
     'Z23；NMS 求解器与神经网络的踝动态力矩估计，视觉→动力学桥接；此前 IEEE 418 / ResearchGate 403。Crossref 核实为 ICARM 2021'),
    ('10.1109/YAC71005.2026.11616179', '章隆彬_合作团队', '09 章隆彬 PAPER_GAP_LIST', '中', '理解', 'understanding',
     'Z24；多任务平衡挑战下的 EEG 与压力中心调节；此前 IEEE HTTP 418'),
    ('10.1109/YAC71005.2026.11615719', '章隆彬_合作团队', '09 章隆彬 PAPER_GAP_LIST', '中', '理解', 'understanding',
     'Z25；多模态 LSTM 地面反力预测；此前 IEEE HTTP 418'),

    # ---- 宋嵘团队：题录已复核，全文从未取件 ----
    ('10.1109/TNSRE.2020.2986295', '宋嵘_合作团队', '12 宋嵘 LIST P0', 'P0', '辅助与训练强度', 'assistance',
     '10 例卒中 + 对照，自适应 FES 强度/时相调整；全台账唯一同时具备患者/真实辅助/强度自适应的条目'),
    ('10.1109/TNSRE.2025.3535928', '宋嵘_合作团队', '12 宋嵘 LIST P0', 'P0', '双参与', 'understanding',
     '上肢肘部皮层-肌电耦合与 FMA 临床评分相关；C1 缺独立心理/神经参照的最对口一篇'),
    ('10.1109/TBME.2024.3368021', '宋嵘_合作团队', '12 宋嵘 LIST P0', 'P0', '双参与', 'understanding',
     'HD-sEMG 分舱肌骨模型力矩预测；C1 物理参照候选。须核清「预测力矩≠主动贡献」'),
    ('10.3389/fbioe.2023.1246014', '宋嵘_合作团队', '12 宋嵘 LIST P1', 'P1', '辅助与训练强度', 'assistance',
     'FES 混合自适应控制纠正偏瘫步态；源清单标注开放获取（PMC10441235），可能无需订阅'),
    ('10.1109/TMRB.2026.3682646', '宋嵘_合作团队', '12 宋嵘 LIST P1', 'P1', '运动理解', 'understanding',
     '该团队 2026 最新方法，多模型无扰切换解码；影响运动理解段基线选择与查新'),
    ('10.1088/1741-2552/ac18ae', '宋嵘_合作团队', '12 宋嵘 LIST P1', 'P1', '双参与', 'understanding',
     '卒中皮层-运动系统多层信息传递随肌电接口维度改变；为「可干预参照」范式'),
    ('10.1002/aisy.202400505', '宋嵘_合作团队', '12 宋嵘 LIST P2', 'P2', '辅助与训练强度', 'assistance',
     '上肢线缆驱动机器人 GMM 抗疲劳增量校准；C2 重标定触发条件依据'),
    ('10.1109/TNSRE.2024.3371465', '宋嵘_合作团队', '12 宋嵘 LIST P2', 'P2', '运动理解', 'understanding',
     'FES 诱发力矩的 Hammerstein 在线预测，真实辅助下的力矩模型'),
    ('10.1109/TNSRE.2023.3348332', '宋嵘_合作团队', '12 宋嵘 LIST P2', 'P2', '辅助与训练强度', 'assistance',
     '人主导/机器人主导/软停三模式，按能力调辅助的范式'),
    ('10.1109/TBME.2020.3012296', '宋嵘_合作团队', '12 宋嵘 LIST P2', 'P2', '辅助与训练强度', 'assistance',
     '踝外骨骼 EMG 导纳主动控制比例，健康+卒中混编'),
    ('10.1109/TNSRE.2020.3042221', '宋嵘_合作团队', '12 宋嵘 LIST P3', 'P3', '运动理解', 'understanding',
     'FES 诱发踝力矩与刚度建模，补覆盖密度'),
    ('10.1109/TNSRE.2021.3066592', '宋嵘_合作团队', '12 宋嵘 LIST P3', 'P3', '运动理解', 'understanding',
     '康复机器人运动意图描述的无扰切换机制'),
    ('10.1109/TNSRE.2019.2949203', '宋嵘_合作团队', '12 宋嵘 LIST P3', 'P3', '运动理解', 'understanding',
     '三维运动意图的切换机制连续描述'),
    ('10.1088/1741-2552/abc29a', '宋嵘_合作团队', '12 宋嵘 LIST P3', 'P3', '双参与', 'understanding',
     '二维肌电接口调制与恢复卒中患者肌间协调；共收缩/代偿分离的候选证据'),
    ('10.1109/TNSRE.2023.3238758', '宋嵘_合作团队', '12 宋嵘 LIST P3', 'P3', '双参与', 'understanding',
     '卒中步行中肌电活动与肌间耦合的相位依赖调制'),
    ('10.1109/tnsre.2021.3101615', '宋嵘_合作团队', '12 宋嵘 LIST P3', 'P3', '双参与', 'understanding',
     '⚠️ DOI 纠正：源清单登记的 10.1109/TNSRE.2021.3079572 在 Crossref 与 doi.org 均不存在（404），'
     '经题名检索确认真实 DOI 为本条；按纠正后的 DOI 取件'),
]


def crossref(doi):
    url = 'https://api.crossref.org/works/' + doi
    req = urllib.request.Request(url, headers={'User-Agent': 'kb-acquisition-verify/1.0 (mailto:local)'})
    m = json.load(urllib.request.urlopen(req, timeout=30))['message']
    year = (m.get('published-print') or m.get('published-online')
            or m.get('issued') or {}).get('date-parts', [[None]])[0][0]
    return {
        'title': (m.get('title') or [''])[0].strip(),
        'venue': (m.get('container-title') or [''])[0].strip(),
        'publisher': (m.get('publisher') or '').strip(),
        'year': year,
        'type': m.get('type', ''),
    }


PLATFORM = {
    'IEEE': 'IEEE Xplore',
    'Elsevier BV': 'ScienceDirect / Elsevier',
    'Elsevier': 'ScienceDirect / Elsevier',
    'Springer Science and Business Media LLC': 'SpringerLink',
    'Springer Nature': 'SpringerLink',
    'SAGE Publications': 'SAGE Journals',
    'SAGE': 'SAGE Journals',
    'Wiley': 'Wiley Online Library',
    'John Wiley and Sons': 'Wiley Online Library',
    'IOP Publishing': 'IOPscience',
    'Frontiers Media SA': 'Frontiers（出版方开放获取平台）',
    'Informa UK Limited': 'Taylor & Francis Online',
    'Wolters Kluwer Health, Inc.': 'Wolters Kluwer / Ovid',
}


def main():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    have = {p['doi'].lower() for p in manifest['papers'] if p.get('doi')}
    print('台账现有条目:', len(manifest['papers']), '| 已有 DOI 数:', len(have))

    shutil.copy2(MANIFEST, BACKUP)
    print('已备份 ->', BACKUP.relative_to(ROOT))

    added, skipped, failed = [], [], []
    for doi, corpus, source, priority, thread, role, note in ITEMS:
        if doi.lower() in have:
            skipped.append(doi)
            continue
        try:
            cr = crossref(doi)
        except Exception as exc:                      # noqa: BLE001
            failed.append((doi, str(exc)[:70]))
            continue
        platform = PLATFORM.get(cr['publisher'], cr['publisher'] or '未核实')
        pid = 'DL-' + hashlib.sha1(doi.lower().encode()).hexdigest()[:8]
        manifest['papers'].append({
            'id': pid,
            'title': cr['title'],
            'doi': doi,
            'arxiv_id': '',
            'year': cr['year'],
            'venue': cr['venue'] or '未核实',
            'corpus': corpus,
            'thread': thread,
            'priority': priority,
            'source_team': source,
            'local_fulltext': None,
            'version_note': note,
            'screening': 'included',
            'index_reference': None,
            'role': role,
            'include_reason': '纳入本轮链路研究；此前因反爬/非OA/未取件而无全文，现已可取',
            'needs_fulltext': True,
            'database': platform,
            'landing_url': 'https://doi.org/' + doi,
            'source_records': [source],
            'kind': 'paper',
        })
        added.append((doi, cr['year'], platform))
        time.sleep(0.35)

    manifest['checked_at'] = manifest.get('checked_at')
    manifest['scope'] = manifest.get('scope')
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')

    print('并入:', len(added), '| 已存在跳过:', len(skipped), '| 核实失败:', len(failed))
    for d, y, p in added:
        print(f'   + {d:38s} {y} {p}')
    for d, e in failed:
        print('   ! 失败', d, e)
    for d in skipped:
        print('   = 已在台账', d)


if __name__ == '__main__':
    main()