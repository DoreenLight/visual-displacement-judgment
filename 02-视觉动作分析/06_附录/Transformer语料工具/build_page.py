#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_page.py — 由 index.json 生成「Transformer 发展线」可视化页面（自包含 HTML）。

设计约束（依据 visual-page 规范 + 用户视觉偏好）：
  - 页面模式：Long-read / Report（研究型知识库，不是仪表盘）
  - 背景暖米色 #f7f5f0，**不使用黑底、不使用紫色**
  - 强调色单一：#2f6f5e（深绿）
  - 图表优先：双轴矩阵（CSS Grid）+ 阶段时间线（竖向轨道）+ 线程占比（bench bar）
  - 字体：DM Serif Display（标题）+ Outfit（正文）+ JetBrains Mono（编号）

用法：python3 build_page.py
"""
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
IDX = (HERE / '../../01_参考论文/运动理解/Transformer_发展线/index.json').resolve()
OUT = (HERE / '../../08_三维动作分析展示台/Transformer_发展线图谱.html').resolve()

STAGE_ORDER = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']
THREAD_ORDER = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']
REL_COLOR = {'高': '#2f6f5e', '中': '#c97b3f', '低': '#8a8a80'}


def e(s):
    return html.escape(str(s or ''), quote=True)


def main():
    data = json.loads(IDX.read_text(encoding='utf-8'))
    papers = data['papers']
    stages = data['axes']['stage']
    threads = data['axes']['thread']

    grid = {(t, s): [] for t in THREAD_ORDER for s in STAGE_ORDER}
    for p in papers:
        k = (p['thread'], p['stage'])
        if k in grid:
            grid[k].append(p)
    for v in grid.values():
        v.sort(key=lambda x: (x.get('year') or 0, x.get('canonical_title') or ''))

    n_high = sum(1 for p in papers if p['relevance'] == '高')
    n_ok = sum(1 for p in papers if p.get('pdf_status') == 'ok')
    by_thread = {t: sum(1 for p in papers if p['thread'] == t) for t in THREAD_ORDER}
    max_thread = max(by_thread.values()) if by_thread else 1

    o = []
    A = o.append
    A('<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">')
    A('<meta name="viewport" content="width=device-width,initial-scale=1">')
    A('<title>Transformer 发展线图谱 · 康复迁移推演</title>')
    A('<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Outfit:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">')
    A('''<style>
:root{--bg:#f7f5f0;--card:#fff;--line:#e9e3d6;--ink:#14171e;--ink2:#3a4256;--muted:#7c7a72;--primary:#2f6f5e;--amber:#c97b3f}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:'Outfit',sans-serif;font-size:14.5px;line-height:1.6;
background-image:radial-gradient(900px 500px at 8% -8%,rgba(47,111,94,.05),transparent 60%),radial-gradient(800px 460px at 100% 0%,rgba(201,123,63,.05),transparent 60%)}
.wrap{max-width:1120px;margin:0 auto;padding:0 24px}
h1,h2{font-family:'DM Serif Display',serif;font-weight:400;letter-spacing:-.005em}
h1{font-size:34px;margin:0 0 6px}
h2{font-size:24px;margin:0}
.mono{font-family:'JetBrains Mono',monospace}
.muted{color:var(--muted)}
.lead{font-size:16px;color:var(--ink2);max-width:74ch;margin:0 0 22px}
section{margin:52px 0}
.sh{display:flex;align-items:center;gap:10px;margin:0 0 6px}
.sh svg{color:var(--primary);flex:none}
.sub{color:var(--muted);font-size:13.5px;margin:0 0 18px;max-width:82ch}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px 22px;
box-shadow:0 2px 8px rgba(0,0,0,.04)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:22px 0 0}
.stat{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px}
.stat b{display:block;font-family:'DM Serif Display',serif;font-size:27px;line-height:1.1;color:var(--primary)}
.stat span{font-size:12.5px;color:var(--muted)}
table.mx{border-collapse:separate;border-spacing:6px;width:100%}
table.mx th{font-size:12px;font-weight:600;color:var(--muted);text-align:left;padding:4px 6px;vertical-align:bottom}
table.mx th.tl{width:118px}
td.cell{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:9px 10px;vertical-align:top;min-width:96px}
td.cell.empty{background:transparent;border-style:dashed;border-color:#dcd6c8;opacity:.55}
td.cell .n{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--muted);display:flex;align-items:center;justify-content:space-between}
.dot{width:7px;height:7px;border-radius:50%;display:inline-block}
p.tt{font-size:12.5px;line-height:1.45;margin:6px 0 0;color:var(--ink2);cursor:pointer}
p.tt:hover{color:var(--primary);text-decoration:underline}
.tl{list-style:none;padding:0;margin:0;position:relative}
.tl::before{content:"";position:absolute;left:9px;top:8px;bottom:8px;width:2px;background:var(--primary);opacity:.32}
.tl li{position:relative;padding:0 0 26px 34px}
.tl li::before{content:"";position:absolute;left:3px;top:6px;width:14px;height:14px;border-radius:50%;background:var(--bg);border:2px solid var(--primary)}
.tl h3{margin:0 0 3px;font-size:16.5px;font-family:'Outfit',sans-serif;font-weight:600}
.tl .yr{font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--primary);margin-right:8px}
.tl p{margin:4px 0 0;font-size:13.5px;color:var(--ink2);max-width:88ch}
.tl .lst{margin:8px 0 0;font-size:12.5px;color:var(--muted)}
.bench{display:grid;grid-template-columns:150px 1fr 68px;gap:12px;align-items:center;padding:6px 0}
.bench-track{height:14px;background:#efece3;border-radius:7px;overflow:hidden}
.bench-fill{height:100%;background:var(--primary);border-radius:7px;opacity:.85}
.bench-val{font-family:'JetBrains Mono',monospace;font-size:12.5px;color:var(--ink2);text-align:right}
.filters{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 14px;align-items:center}
button.f{font-family:'Outfit',sans-serif;font-size:12.5px;padding:5px 12px;border-radius:999px;border:1px solid var(--line);background:var(--card);color:var(--ink2);cursor:pointer;transition:all .18s}
button.f:hover{border-color:var(--primary);color:var(--primary)}
button.f.on{background:var(--primary);border-color:var(--primary);color:#fff}
input.s{font-family:'Outfit',sans-serif;font-size:13.5px;padding:8px 12px;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--ink);min-width:230px;flex:1}
input.s:focus{outline:none;border-color:var(--primary)}
.row{display:grid;grid-template-columns:repeat(auto-fill,minmax(268px,1fr));gap:10px}
.p{border:1px solid var(--line);border-radius:13px;padding:13px 14px;background:var(--card);transition:all .2s}
.p:hover{transform:translateY(-2px);box-shadow:0 6px 18px -6px rgba(0,0,0,.10)}
.p .h{display:flex;align-items:center;gap:7px;font-family:'JetBrains Mono',monospace;font-size:11.5px;color:var(--muted)}
.p h4{margin:6px 0 5px;font-size:14.5px;font-weight:600;line-height:1.35}
.p .au{font-size:12px;color:var(--muted);margin:0 0 6px}
.p .nt{font-size:12.5px;color:var(--ink2);margin:0}
.p .tags{margin-top:8px;display:flex;gap:5px;flex-wrap:wrap}
.tag{font-size:10.5px;padding:2px 7px;border-radius:999px;background:rgba(47,111,94,.09);color:var(--primary);font-family:'JetBrains Mono',monospace}
.tag.a{background:rgba(201,123,63,.11);color:#a2632f}
.tag.g{background:rgba(47,111,94,.14);color:#215a4b;font-weight:600}
.tag.miss{background:#f1e6e2;color:#a2543f}
.hide{display:none!important}
.note{font-size:12.5px;color:var(--muted);margin-top:10px}
svg.i{width:19px;height:19px}
footer{margin:60px 0 40px;padding-top:18px;border-top:1px solid var(--line);font-size:12.5px;color:var(--muted)}
@media(max-width:860px){.wrap{padding:0 16px}table.mx{border-spacing:4px}td.cell{min-width:74px}p.tt{font-size:11.5px}}
</style></head><body>''')

    # ---------------- Hero
    A('<div class="wrap">')
    A('<div style="padding:44px 0 6px">')
    A('<h1>Transformer 发展线图谱</h1>')
    A('<p class="lead">从 2017 年的原始架构到 2026 年的混合架构，本图谱按 <b>时间阶段 × 技术线程</b> 双轴组织 %d 篇已核验论文，'
      '并标注每一条对 <b>上肢腕–肘–肩康复</b>（sEMG / 视觉 → 连续关节角、实时预测、离散 token、物理约束）的迁移价值。'
      '结论先行：<b>康复领域尚未吃透的三块，正好是 Transformer 已经解决的三块</b> —— 离散 token 化（T5）、长时序线性化（T7）、物理约束注入（T6）。</p>' % len(papers))
    A('<div class="stats">')
    A('<div class="stat"><b>%d</b><span>已入库论文（PDF 实测）</span></div>' % n_ok)
    A('<div class="stat"><b>%d</b><span>核验元数据条目</span></div>' % len(papers))
    A('<div class="stat"><b>%d</b><span>康复迁移价值「高」</span></div>' % n_high)
    A('<div class="stat"><b>%d</b><span>技术线程</span></div>' % len(THREAD_ORDER))
    A('</div></div>')

    # ---------------- 双轴矩阵
    A('<section><div class="sh"><svg class="i" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></svg><h2>双轴矩阵：线程 × 阶段</h2></div>')
    A('<p class="sub">纵轴是技术线程，横轴是时间阶段。单元格内的数字是论文数，圆点颜色表示该篇的康复迁移价值。'
      '这张矩阵回答的问题是：<b>哪些线程在康复领域还几乎是空白</b>。</p>')
    A('<table class="mx"><thead><tr><th class="tl">线程 ＼ 阶段</th>')
    for s in STAGE_ORDER:
        A('<th>%s<br><span style="font-weight:400">%s</span></th>' % (s, e(stages[s].split(' ', 1)[1] if ' ' in stages[s] else stages[s])))
    A('</tr></thead><tbody>')
    for t in THREAD_ORDER:
        A('<tr><th class="tl">%s %s</th>' % (t, e(threads[t])))
        for s in STAGE_ORDER:
            items = grid[(t, s)]
            if not items:
                A('<td class="cell empty"><div class="n">·</div></td>')
            else:
                dots = ''.join('<span class="dot" style="background:%s"></span>' % REL_COLOR.get(i['relevance'], '#8a8a80') for i in items)
                A('<td class="cell"><div class="n"><span>%d</span><span>%s</span></div>' % (len(items), dots))
                for i in items[:4]:
                    A('<p class="tt" data-q="%s">%s</p>' % (e(i.get('canonical_title') or i['title']), e(i.get('canonical_title') or i['title'])[:44]))
                if len(items) > 4:
                    A('<p class="tt" data-q="">+%d 更多</p>' % (len(items) - 4))
                A('</td>')
        A('</tr>')
    A('</tbody></table>')
    A('<p class="note">说明：阶段与线程的归属是<b>人工判断</b>（见各线程 LIST.md），arXiv 只提供客观元数据；两者在卡片中分列标注，不混为一谈。</p>')
    A('</section>')

    # ---------------- 阶段时间线
    A('<section><div class="sh"><svg class="i" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg><h2>阶段时间线</h2></div>')
    A('<p class="sub">每个阶段回答一个问题：这一阶段 Transformer 解决或暴露了什么，以及对康复的直接含义。</p>')
    A('<ol class="tl">')
    STAGE_MEANING = {
        'S0': ('原始架构给了「编码器–解码器 + 交叉注意力」的骨架，这正是「源语言→目标语言」翻译范式的原型。',
               'ViKiT 的「翻译」比喻在此有出处；康复领域至今少有真正的 encoder–decoder 结构，多为单塔回归。'),
        'S1': ('预训练 + 稀疏/线性注意力：模型开始能吃长序列，注意力不再是平方级瓶颈。',
               '长时程 sEMG 窗口与长视频段成为可建模对象；但稀疏化会丢细节，正是 Wei 2026 批评 Informer 的点。'),
        'S2': ('位置编码（RoPE/ALiBi）与视频、多模态成型；「空间注意力 + 时间注意力分解」成为人体姿态标准做法。',
               'ViKiT 时空 encoder 的直接蓝本；RoPE 对角度序列的长度外推友好，可缓解「训练 27 帧、测试 81 帧」。'),
        'S3': ('离散动作 token（T2M-GPT / MotionGPT）与状态空间模型（Mamba）同时爆发：一头是「可翻译」，一头是「可长跑」。',
               '离散 kinematic vocabulary 有现成范式；Mamba 线性时间对康复实时闭环是硬需求。'),
        'S4': ('物理先验注入人体运动（PhysPT / PHC）与测试时记忆（Titans）：开始处理「不满足物理规律的输出」。',
               'biomechanical grammar 从「手写 loss」升级为「可学习/可约束的先验」；个体化在线校准有了架构答案。'),
        'S5': ('线性注意力与混合架构复兴（delta rule 等），注意力不再是唯一解。',
               '算力受限的康复端侧部署有了新选项；也提醒我们别把「Transformer」当成唯一范式押注。'),
    }
    for s in STAGE_ORDER:
        items = sorted(grid_by_stage(s, papers), key=lambda x: (x.get('year') or 0, x.get('canonical_title') or ''))
        what, why = STAGE_MEANING[s]
        A('<li><h3><span class="yr">%s</span>%s</h3><p>%s</p><p style="color:var(--primary)">→ 康复含义：%s</p>' % (s, e(stages[s]), e(what), e(why)))
        names = '、'.join((i.get('canonical_title') or i['title'])[:38] for i in items[:6])
        A('<p class="lst">本阶段入库 %d 篇：%s%s</p></li>' % (len(items), e(names), ' …' if len(items) > 6 else ''))
    A('</ol></section>')

    # ---------------- 线程分布
    A('<section><div class="sh"><svg class="i" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19V5M4 19h16"/><rect x="7" y="12" width="3" height="7" rx="1"/><rect x="12" y="8" width="3" height="11" rx="1"/><rect x="17" y="10" width="3" height="9" rx="1"/></svg><h2>线程分布与迁移密度</h2></div>')
    A('<p class="sub">论文数不等于价值。真正要看的是「高迁移价值」的密度 —— 它决定了康复方向的优先级。</p>')
    A('<div class="card">')
    for t in THREAD_ORDER:
        tot = by_thread[t]
        hi = sum(1 for p in papers if p['thread'] == t and p['relevance'] == '高')
        pct = int(round(tot * 100.0 / max_thread))
        A('<div class="bench"><span class="bench-label">%s %s</span><div class="bench-track"><div class="bench-fill" style="width:%d%%"></div></div>'
          '<span class="bench-val">%d·高%d</span></div>' % (t, e(threads[t]), pct, tot, hi))
    A('</div>')
    A('<p class="note">读法：条长 = 该线程总篇数；「高」= 其中康复迁移价值高的篇数。T5（动作生成与离散 token）与 T6（物理约束）在康复领域<b>几乎未被占用</b>，是创新密度最高的洼地。</p>')
    A('</section>')

    # ---------------- 可检索清单
    A('<section id="lib"><div class="sh"><svg class="i" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg><h2>可检索论文清单</h2></div>')
    A('<p class="sub">按标题、作者、线程、阶段、note 全文检索。灰标 = 尚未入库，需人工补件。</p>')
    A('<div class="filters">')
    A('<input class="s" id="q" type="search" placeholder="搜索标题 / 作者 / 迁移作用…">')
    for r in ['全部', '高', '中', '低', '已入库', '需人工取件']:
        A('<button class="f%s" data-r="%s">%s</button>' % (' on' if r == '全部' else '', e(r), e(r)))
    A('</div>')
    A('<div class="row" id="cards">')
    for p in sorted(papers, key=lambda x: (x.get('year') or 0)):
        miss = p.get('pdf_status') != 'ok'
        rel = p['relevance']
        A('<div class="p" data-rel="%s" data-ok="%s" data-t="%s" data-stage="%s" data-note="%s">' % (
            e(rel), '0' if miss else '1',
            e('%s %s %s' % (p.get('canonical_title') or p['title'], ' '.join(p.get('authors', [])[:3]), p.get('note', ''))),
            e('%s %s' % (p['stage'], p['thread'])), e(p.get('note', ''))))
        A('<div class="h"><span>%s</span><span>%s</span></div>' % (e(p.get('year') or ''), e(p['arxiv_id'])))
        A('<h4>%s</h4>' % e(p.get('canonical_title') or p['title']))
        A('<p class="au">%s</p>' % e(', '.join(p.get('authors', [])[:3]) + (' 等' if len(p.get('authors', [])) > 3 else '')))
        A('<p class="nt">%s</p>' % e(p.get('note', '')))
        A('<div class="tags"><span class="tag">%s</span><span class="tag">%s</span>' % (e(p['stage']), e(p['thread'])))
        A('<span class="tag %s">迁移·%s</span>' % ('g' if rel == '高' else ('a' if rel == '中' else ''), e(rel)))
        if miss:
            A('<span class="tag miss">需人工取件</span>')
        A('</div>')
    A('</div>')
    A('<p class="note" id="cnt"></p>')
    A('</section>')

    # ---------------- 康复推演
    A('<section><div class="sh"><svg class="i" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3v18M3 12h18"/><circle cx="12" cy="12" r="9"/></svg><h2>推演：Transformer 已解决 → 康复可用的三个缺口</h2></div>')
    A('<p class="sub">以下为<b>推断</b>，不是既有事实。每条都标注了它在语料中的证据来源，以及需要用实验证伪的判据。</p>')
    A('<div class="card">')
    DIRS = [
        ('D1', '离散 kinematic token 取代连续角度回归', 'T5',
         'T2M-GPT / MotionGPT 已把连续运动量化成离散 token 再用自回归建模；TokenHMR 已把姿态做成词表。',
         'ViKiT 现有 256 桶 bucketize 仍是<b>硬量化</b>，会丢分辨率且训练/推理不匹配。可改用<b>残差 VQ-VAE + EMA 码本更新 + 死码复活</b>，与 T2M-GPT 的 VQ 方案对齐。',
         '判据：同参数量下，VQ 版本 MAE 必须低于 bucketize；且码本利用率（非零码字占比）需 &gt; 80%。'),
        ('D2', '长时序实时预测：线性/混合架构替代纯注意力', 'T7',
         'Mamba 证明选择性状态空间可线性时间建模长序列；2025 的 delta rule 复兴说明注意力不再是唯一解。',
         '康复端侧算力受限（Wei 2026 用 Tesla T4 训练 &lt;2 分钟），300 ms 提前预测 + 长窗口在纯注意力下成本高。可做<b>注意力/SSM 混合编码器</b>，前段注意力抓局部关节耦合、后段 SSM 抓长时程。',
         '判据：同等 MAE 下参数量与推理延迟显著低于纯 Transformer；且延迟抖动（jitter）必须下降。'),
        ('D3', '物理约束从「手写 loss」升级为「可学习先验」', 'T6',
         'PhysPT / PHC 把物理一致性作为预训练约束而非纯后处理；章隆彬团队 NMS+ML 混合已在康复侧验证过物理+学习路线。',
         'ViKiT 现有 6 项 grammar loss 是手写标量惩罚。可升级为<b>能量/约束模型 + projection 投影解码</b>：训练期学约束强度，推理期把预测投影回可行域（ROM + FK），得到<b>硬保证</b>而非软惩罚。',
         '判据：ROM violation rate 必须<b>恒等于 0</b>（硬约束定义），且 MAE 相对软 loss 的退化需在统计显著范围内可接受。'),
        ('D4', '位置编码与长度外推：解决训练/测试窗口不一致', 'T3',
         'RoPE 通过旋转把相对位置编码进注意力，长度外推能力远好于绝对位置编码；ALiBi 走另一条外推路线。',
         '康复数据采集窗口常变（250 ms / 150 ms / 不等），且跨速度、跨被试长度不一致。引入 RoPE 可让模型在<b>未见过的窗口长度</b>上直接推理。',
         '判据：在训练未出现过的窗口长度上，MAE 相对 baseline 劣化 &lt; 10%。'),
        ('D5', '多模态融合：RGB-D + IMU + sEMG 的统一 token 系统', 'T4',
         'Flamingo 的交错块注意力证明「冻结专家 + 轻量适配」可在异构模态间传信息，且不破坏预训练权重。',
         '本项目 D435i（RGB-D + 内置 IMU）与章隆彬团队 sEMG 天然是异构模态。可用<b>各自编码器 + 交错块 cross-attention</b>，先单模态可用，再逐步融合 —— 避免一开始就因多模态对齐失败而全盘皆输。',
         '判据：单模态子模型各自可用；融合后至少不比单模态最差者差。'),
        ('D6', '个体化：测试时记忆 / 参数高效适配', 'T4',
         'LoRA 与 Titans（测试时记忆）提供了「少量样本快速个性化」的架构路径，而非全量重训。',
         '康复个体差异极大（章隆彬路线也强调跨被试迁移难）。可把 anatomy tokenizer + kinematic codebook 用 LoRA 冻结大部分预训练权重，逐个被试用极少数据适配。',
         '判据：每被试 ≤1 分钟适配数据即可显著降低误差，且原始权重不被污染。'),
    ]
    A('<div style="display:grid;gap:14px">')
    for did, title, thread, basis, how, falsify in DIRS:
        A('<div style="border:1px solid var(--line);border-radius:13px;padding:15px 17px;background:#fcfbf8">')
        A('<div class="h" style="font-family:\'JetBrains Mono\',monospace;font-size:11.5px;color:var(--primary)">%s · 证据线程 %s</div>' % (did, e(thread)))
        A('<h4 style="margin:5px 0 7px;font-size:15.5px">%s</h4>' % e(title))
        A('<p style="margin:0 0 6px;font-size:13px;color:var(--ink2)"><b>语料证据：</b>%s</p>' % e(basis))
        A('<p style="margin:0 0 6px;font-size:13px;color:var(--ink2)"><b>可做什么：</b>%s</p>' % e(how))
        A('<p style="margin:0;font-size:12.5px;color:var(--muted)"><b>证伪判据：</b>%s</p>' % e(falsify))
        A('</div>')
    A('</div></div>')
    A('<p class="note"><b>边界声明：</b>以上是「Transformer 已有能力 → 康复可迁移」的<b>推断</b>，尚未经本项目实验验证。'
      '其中 D1/D3 与 ViKiT 现有设计直接冲突或升级，实施前必须先与 <code>03_方法设计/</code> 的既定决策对齐并做查重，不能直接开跑。</p>')
    A('</section>')

    A('<footer>')
    A('语料元数据来源：arXiv 官方 API（逐条标题检索 + 相似度比对，阈值 0.90 自动采信）；PDF 仅取 arXiv 开放获取版本。'
      '阶段/线程归属、迁移价值、推演方向均为<b>人工判断</b>，已在卡片中与官方元数据分列。'
      '本页面为本地自包含 HTML，不引用任何 CDN 脚本（仅 Google Fonts 字体）。')
    A('</footer></div>')

    A('''<script>
(function(){
 var q=document.getElementById('q'),cards=[].slice.call(document.querySelectorAll('#cards .p')),btns=[].slice.call(document.querySelectorAll('button.f')),cnt=document.getElementById('cnt'),cur='全部';
 function run(){
  var s=(q.value||'').toLowerCase(),n=0;
  cards.forEach(function(c){
   var okR=(cur==='全部')||(cur==='已入库'&&c.dataset.ok==='1')||(cur==='需人工取件'&&c.dataset.ok==='0')||(cur===c.dataset.rel);
   var okQ=!s||c.dataset.t.toLowerCase().indexOf(s)>-1;
   var show=okR&&okQ;c.classList.toggle('hide',!show);if(show)n++;
  });
  cnt.textContent='显示 '+n+' / '+cards.length+' 篇';
 }
 btns.forEach(function(b){b.addEventListener('click',function(){btns.forEach(function(x){x.classList.remove('on')});b.classList.add('on');cur=b.dataset.r;run();});});
 q.addEventListener('input',run);run();
})();
</script>''')
    A('</body></html>')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text('\n'.join(o), encoding='utf-8')
    print('已生成：%s（%.1f KB，%d 篇）' % (OUT, OUT.stat().st_size / 1024, len(papers)))


def grid_by_stage(s, papers):
    return [p for p in papers if p['stage'] == s]


if __name__ == '__main__':
    main()
