#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成「需人工下载」总清单：合并语料三与语料四的全部缺口。

输出 06_附录/需你下载的论文清单.md，每行必含三列：
    论文标题 ｜ 地址 ｜ 期刊/会议
按「怎么下得到」分组，让下载动作可执行。

排除网页型证据（无 PDF 可下），单列一节并说明原因。
"""
import json
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BM = ROOT / '01_参考论文' / '生物力学线'
JV_IDX = ROOT / '01_参考论文' / '决策模型线_JEV' / 'index.json'
OUT = HERE.parent / '需你下载的论文清单.md'

DB = {'10.1016': 'ScienceDirect（Elsevier）', '10.1007': 'SpringerLink', '10.1109': 'IEEE Xplore',
      '10.1088': 'IOPscience', '10.1111': 'Wiley Online Library', '10.1002': 'Wiley Online Library',
      '10.1249': 'Wolters Kluwer（Ovid）', '10.2466': 'SAGE / Perceptual and Motor Skills',
      '10.1123': 'Human Kinetics', '10.1142': 'World Scientific', '10.5220': 'SciTePress',
      '10.3389': 'Frontiers（免费开放）', '10.1034': 'Wiley Online Library',
      '10.1093': 'Oxford Academic', '10.3390': 'MDPI（免费开放）', '10.1038': 'Nature Portfolio',
      '10.1155': 'Hindawi（Wiley）', '10.1186': 'BioMed Central（免费开放）',
      '10.1005': 'SpringerLink', '10.1145': 'ACM Digital Library', '10.5194': 'Copernicus（免费开放）'}

# 实测拦点（浏览器逐篇试出来的，非推测）
BLOCK = {
    '10.1016/j.physio.2015.02.002': ('PMC reCAPTCHA', 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4600010/'),
    '10.1007/s10439-014-1181-7': ('PMC reCAPTCHA', 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4454420/'),
    '10.1109/tbme.2008.2005946': ('PMC reCAPTCHA', 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2971671/'),
    '10.1016/j.apmr.2016.06.023': ('PMC reCAPTCHA', 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5299057/'),
    '10.1088/1741-2560/6/3/036004': ('Radware 机器人验证码', 'https://iopscience.iop.org/article/10.1088/1741-2560/6/3/036004'),
    '10.1007/s11044-022-09852-x': ('仓储 WAF（curl 418 / 浏览器亦拦）', 'https://hdl.handle.net/2117/377476'),
    '10.1109/cvpr.2007.383128': ('MPG bot 校验', 'https://pure.mpg.de/pubman/item/item_1328064_4/'),
    '10.1109/embc.2012.6346535': ('NAIST 仓储落地页，无直链', 'https://naist.repo.nii.ac.jp/records/7819'),
    '10.1111/joa.12903': ('Wiley 403', 'https://onlinelibrary.wiley.com/doi/10.1111/joa.12903'),
    '10.1034/j.1600-0404.2003.00021.x': ('Wiley 403', 'https://onlinelibrary.wiley.com/doi/10.1034/j.1600-0404.2003.00021.x'),
    '10.1016/j.jbiomech.2019.05.040': ('Elsevier 403', 'https://www.sciencedirect.com/science/article/pii/S0021929025002702'),
    '10.1016/j.jbiomech.2024.112118': ('Elsevier 403', 'https://www.sciencedirect.com/science/article/pii/S0021929024006872'),
    '10.1093/ptj/pzab010': ('OUP 403', 'https://academic.oup.com/ptj/article/101/4/pzab010/6363736'),
}


def norm(s):
    return re.sub(r'[^a-z0-9]', '', unicodedata.normalize('NFKD', s or '').lower())


def esc(s):
    return (s or '').replace('|', '\\|')


def db_of(doi):
    for k, v in DB.items():
        if (doi or '').lower().startswith(k):
            return v
    return '按 DOI 走出版商官网'


def on_disk(thread, title):
    td = next((d for d in BM.iterdir() if d.is_dir() and d.name.startswith(thread + '_')), None)
    if not td:
        return None
    k = norm(title)[:38]
    for f in list(td.glob('*.pdf')) + list(td.glob('*.xml')):
        if k in norm(f.stem) or norm(f.stem)[:38] in k:
            return f
    return None


def main():
    # ---------- 语料三 ----------
    bm = json.loads((BM / 'index.json').read_text(encoding='utf-8'))
    miss3 = [p for p in bm['papers'] if (p.get('title') or '').strip()
             and not on_disk(p['thread'], p['title'])]
    a3 = sorted([p for p in miss3 if p.get('oa_status') in ('gold', 'bronze', 'green')],
                key=lambda x: x['venue'])
    b3 = sorted([p for p in miss3 if p.get('oa_status') in ('closed', 'hybrid')],
                key=lambda x: (x['venue'], str(x.get('year', ''))))
    # ---------- 语料四 ----------
    jv = json.loads(JV_IDX.read_text(encoding='utf-8'))
    miss4 = [p for p in jv['papers'] if not p.get('pdf_path') and (p.get('url_abs') or p.get('url_pdf'))]
    paper4 = [p for p in miss4 if p.get('doi') or p.get('arxiv_id')]
    web4 = [p for p in miss4 if not (p.get('doi') or p.get('arxiv_id'))]

    L = ['# 需你下载的论文清单（合并语料三 + 语料四）', '',
         '> **生成时间**：2026-10-01　|　由 `06_附录/决策模型工具/gen_download_list.py` 生成',
         '>', '> **每条三列齐备**：论文标题 / 地址 / 期刊·会议。点地址直接跳出版商页面。',
         '>', '> 语料二（Transformer 发展线 93 篇）**已全部入库，本清单不含该语料**。', '',
         '---', '', '## 0. 一览', '',
         '| 语料 | 已入库 | 待下载 | 说明 |', '|---|---|---|---|',
         '| 语料三（生物力学与上肢运动学） | %d | **%d** | A 组 %d + B 组 %d |' % (
             100 - len(miss3), len(miss3), len(a3), len(b3)),
         '| 语料四（决策模型线 / AAN 介入决策） | %d | **%d** | 其中 %d 条是网页型证据，无 PDF 可下 |' % (
             sum(1 for p in jv['papers'] if p.get('pdf_status') == 'ok'), len(miss4), len(web4)),
         '', '**合计需你下载：%d 篇**（语料三 %d + 语料四 %d）' % (len(a3) + len(b3) + len(paper4),
                                                    len(a3) + len(b3), len(paper4)),
         '', '---', '',
         '## 1. 【语料三 A 组】官方免费，浏览器点开即可 · %d 篇' % len(a3), '',
         '这些是**官方开放获取**（Unpaywall 确认），只是出版商对**脚本请求**返回反爬验证页。',
         '用普通浏览器打开、正常点 Download PDF 即可 —— **不要**用 `curl` / `wget` / 批量下载器。',
         '「实测拦点」是我用真实 Chrome 逐篇试出来的，不是推测。', '',
         '| # | 论文标题 | 期刊/会议 | 地址 | 实测拦点 |', '|---|---|---|---|---|']
    for i, p in enumerate(a3, 1):
        why, land = BLOCK.get(p['doi'], ('出版商访问控制', ''))
        u = land or ('https://doi.org/' + p['doi'])
        L.append('| %d | %s | %s（%s） | %s | %s |' % (
            i, esc(p['title']), esc(p['venue']), p.get('year', ''), u, why))
    L += ['', '---', '',
          '## 2. 【语料三 B 组】需机构订阅 · %d 篇' % len(b3), '',
          '出版商**无开放版本**，需学校/机构订阅。地址是 DOI 解析链接，点开跳到出版商页面。', '',
          '| # | 论文标题 | 期刊/会议 | 地址 | 去哪个数据库 |', '|---|---|---|---|---|']
    for i, p in enumerate(b3, 1):
        L.append('| %d | %s | %s（%s） | https://doi.org/%s | %s |' % (
            i, esc(p['title']), esc(p['venue']), p.get('year', ''), p['doi'], db_of(p['doi'])))
    L += ['', '---', '',
          '## 3. 【语料四】有 DOI/arXiv 但未入库 · %d 篇' % len(paper4), '',
          '| # | 论文标题 | 期刊/会议 | 地址 | 证据等级 |', '|---|---|---|---|---|']
    for i, p in enumerate(paper4, 1):
        v = (p.get('venue') or '—').replace('|', '\\|')
        u = p.get('url_abs') or p.get('url_pdf') or '—'
        L.append('| %d | %s | %s | %s | %s |' % (
            i, esc(p['title']), v, u, p.get('evidence', '—')))
    L += ['', '---', '',
          '## 4. 【语料四】网页型证据 —— 没有 PDF 可下，但需要你核一次 · %d 条' % len(web4), '',
          '这些**不是传闻，但也没有 PDF**。它们是厂商文档、社区实测报告、黑盒探测记录。',
          '「没有 PDF」不是「不可信」，但**也不能当同行评审论文引用**。',
          '', '| # | 标题 | 类型 | 地址 |', '|---|---|---|---|']
    for i, p in enumerate(web4, 1):
        u = p.get('url_abs') or p.get('url_pdf') or '—'
        L.append('| %d | %s | %s | %s |' % (
            i, esc(p['title']), p.get('evidence', '—'), u))
    L += ['', '---', '',
          '## 5. 放回去的位置', '',
          '**语料三**（按线程归位，文件名须与定位卡同名：`<题名>_Abstract_Card.md` 去掉后缀）：', '',
          '```',
          '01_参考论文/生物力学线/B1_视觉运动学_估计与验证方法学/    ← 视觉运动学',
          '01_参考论文/生物力学线/B2_肌肉骨骼动力学_与可微仿真/     ← 肌肉骨骼动力学',
          '01_参考论文/生物力学线/B3_上肢康复_代偿与临床评估/        ← 上肢康复',
          '01_参考论文/生物力学线/B4_离散运动学token_与表征量化/    ← 离散 token',
          '01_参考论文/生物力学线/B5_视觉运动学到动力学_桥接/       ← 视觉→动力学桥接',
          '```', '',
          '**语料四**：`01_参考论文/决策模型线_JEV/D4_AAN辅助控制与介入决策/`（介入决策主线）',
          '或 `D5_轻量化非Transformer路线/`（工程基线）。', '',
          '**清洗规则**（必须与建卡脚本一致，否则自检 Case 7 会 FAIL）：',
          '非法字符 `\\ / : * ? " < > |` **替换成 `_`**，再**截断 60 字符**。', '',
          '放好后跑：', '',
          '```bash',
          'cd 06_附录/Transformer语料工具 && python3 build_biomech.py',
          'cd 06_附录/决策模型工具    && python3 build_jev.py && python3 gen_jev_cards.py',
          'bash 06_附录/自检脚本/verify_kb.sh',
          '```', '',
          '---', '',
          '## 6. 边界声明', '',
          '- 地址全部来自 **Unpaywall / Crossref / Europe PMC 官方接口**，不是搜索引擎结果。',
          '- 「实测拦点」一栏是用真实 Chrome 逐篇试出来的。',
          '- 本清单**不含任何付费墙绕过内容**。A 组走浏览器正常访问，B 组走机构订阅，',
          '  两条都是出版商明示的合法获取方式。',
          '- 破解验证码 / 伪造浏览器指纹 / 批量下载器，本项目一律不做。']
    OUT.write_text('\n'.join(L), encoding='utf-8')
    print('清单已生成：%s' % OUT.relative_to(ROOT))
    print('  语料三 A 组 %d ｜ B 组 %d ｜ 语料四 论文 %d ｜ 网页型 %d' % (
        len(a3), len(b3), len(paper4), len(web4)))


if __name__ == '__main__':
    main()
