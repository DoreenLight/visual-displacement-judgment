#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_page_biomech.py — 生成语料三的可视化展示页

产物：08_三维动作分析展示台/生物力学语料图谱.html

包含三块：
  1. 口径面板：理论搜索空间 / 实际命中 / 人工裁定 / 元数据证实 / 已入库 —— 五组数分列
  2. 双轴矩阵：线程（B1–B5）× 证据范式（M0–M4）
  3. 可检索清单：按标题/期刊/DOI/线程/范式/相关度过滤，标注证据等级与取件状态

设计约束（遵循用户偏好）：**浅色底、不用紫色作主色**。
主色取深青 #0f766e，强调色取琥珀 #b45309，文字 #1c1917 / #57534e。
"""
import json
from collections import Counter, defaultdict
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
INDEX = HERE / '../../01_参考论文/生物力学线/index.json'
OUT = HERE / '../../08_三维动作分析展示台/生物力学语料图谱.html'

THREADS = ['B1', 'B2', 'B3', 'B4', 'B5']
LENSES = ['M0', 'M1', 'M2', 'M3', 'M4']


def main():
    idx = json.loads(Path(INDEX).resolve().read_text(encoding='utf-8'))
    meta = idx['meta']
    papers = idx['papers']
    thread_names = meta['axes']['thread']
    lens_names = meta['axes']['lens']
    c = meta['counts_strictly_separated']

    # ---- 双轴矩阵 ----
    matrix = defaultdict(int)
    cell_high = defaultdict(int)
    for p in papers:
        matrix[(p['thread'], p['lens'])] += 1
        if p.get('relevance') == '高':
            cell_high[(p['thread'], p['lens'])] += 1

    maxv = max(matrix.values()) if matrix else 1

    def shade(n):
        if n == 0:
            return '#fafaf9'
        r = min(1.0, n / maxv)
        return 'rgba(15,118,110,%.2f)' % (0.12 + 0.72 * r)

    rows = []
    for t in THREADS:
        cells = []
        for l in LENSES:
            n = matrix.get((t, l), 0)
            hi = cell_high.get((t, l), 0)
            txt = '#ffffff' if n / maxv > 0.55 else '#1c1917'
            cells.append(
                '<div class="cell" style="background:%s;color:%s" '
                'title="%s × %s：%d 篇（其中相关度高 %d）">%s</div>'
                % (shade(n), txt, t, escape(lens_names.get(l, l)), n, hi, n or ''))
        rows.append('<div class="mrow"><div class="mlab">%s <b>%s</b></div>%s</div>'
                    % (t, escape(thread_names.get(t, '')), ''.join(cells)))
    matrix_html = ''.join(rows) + ''.join(
        '<div class="mlab mhead">　</div><div class="cell mhead">%s</div>' % lens_names.get(l, l)
        for l in LENSES)

    # ---- 卡片数据 ----
    items = []
    for p in papers:
        items.append({
            't': escape(p['title']),
            'v': escape((p.get('venue') or '—')[:52]),
            'y': p.get('year') or 0,
            'th': p['thread'],
            'ln': p['lens'],
            'rl': p.get('relevance', ''),
            'g': p['evidence_grade'][:1],
            'd': escape(p.get('doi') or p.get('arxiv_id') or ''),
            'n': escape(p.get('note', '')),
            'u': escape(p.get('url_abs') or ''),
        })

    cards = ''.join(
        '<div class="card" data-th="%s" data-ln="%s" data-rl="%s" data-g="%s" data-y="%d">'
        '<div class="ct">%s</div>'
        '<div class="cm">%s · %s · %s</div>'
        '<div class="cn">%s</div>'
        '<div class="cd">%s</div></div>'
        % (i['th'], i['ln'], i['rl'], i['g'], i['y'],
           i['t'], i['th'], escape(lens_names.get(i['ln'], '')), i['rl'],
           i['n'], ('<a href="%s" target="_blank" rel="noopener">%s</a>' % (i['u'], i['d'])) if i['u'] else i['d'])
        for i in items)

    g = Counter(p['evidence_grade'][0] for p in papers)

    html = """<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>语料三图谱 · 生物力学与上肢运动学</title>
<style>
:root{--bg:#f7f6f3;--card:#fff;--ink:#1c1917;--mute:#57534e;--line:#e7e5e4;
      --teal:#0f766e;--amber:#b45309;--soft:#fafaf9}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:15px/1.65 -apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:34px 22px 70px}
h1{font-size:25px;margin:0 0 6px;letter-spacing:.2px}
h2{font-size:17px;margin:34px 0 12px;padding-left:10px;border-left:3px solid var(--teal)}
.sub{color:var(--mute);font-size:13.5px;margin:0 0 6px}
.warn{background:#fffbeb;border:1px solid #fde68a;border-left:3px solid var(--amber);
      padding:11px 14px;border-radius:5px;margin:16px 0;font-size:13.5px;color:#78350f}
.counts{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));gap:10px;margin:18px 0 6px}
.cnt{background:var(--card);border:1px solid var(--line);border-radius:7px;padding:12px 13px}
.cnt .k{font-size:11.5px;color:var(--mute);letter-spacing:.3px}
.cnt .v{font-size:23px;font-weight:600;color:var(--teal);margin-top:2px;font-variant-numeric:tabular-nums}
.cnt .d{font-size:11px;color:var(--mute);margin-top:2px}
.mx{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px;overflow-x:auto}
.mrow{display:flex;align-items:stretch;gap:5px;margin-bottom:5px;min-width:640px}
.mlab{flex:0 0 168px;display:flex;align-items:center;font-size:12.5px;color:var(--mute)}
.mlab b{color:var(--ink)}
.cell{flex:1;min-height:42px;border-radius:5px;display:flex;align-items:center;justify-content:center;
      font-size:14px;font-weight:600;font-variant-numeric:tabular-nums;min-width:0}
.mhead{background:none!important;color:var(--mute)!important;font-size:11.5px;font-weight:500;
       min-height:30px;text-align:center;line-height:1.3;padding:0 4px}
.bar{display:flex;gap:9px;flex-wrap:wrap;align-items:center;margin:12px 0}
select,input{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:7px 10px;
     font:inherit;font-size:13.5px;color:var(--ink);outline:none}
input{flex:1;min-width:190px}
select:focus,input:focus{border-color:var(--teal)}
button{background:var(--teal);color:#fff;border:0;border-radius:6px;padding:7px 14px;font:inherit;
       font-size:13.5px;cursor:pointer}
button.ghost{background:var(--card);color:var(--mute);border:1px solid var(--line)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:11px;margin-top:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:7px;padding:12px 13px}
.card[data-rl="高"]{border-left:3px solid var(--teal)}
.card[data-rl="中"]{border-left:3px solid #a8a29e}
.card[data-rl="低"]{border-left:3px solid #d6d3d1}
.ct{font-size:13.5px;font-weight:600;line-height:1.45;margin-bottom:5px}
.cm{font-size:11.5px;color:var(--mute);margin-bottom:5px}
.cm b{color:var(--teal);font-weight:600}
.cn{font-size:12px;color:var(--mute);line-height:1.55;margin-bottom:6px}
.cd{font-size:11px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.cd a{color:var(--teal);text-decoration:none;word-break:break-all}
.cd a:hover{text-decoration:underline}
.badge{display:inline-block;font-size:10.5px;padding:1px 6px;border-radius:3px;background:#eef2f1;
       color:#115e59;margin-left:5px;vertical-align:1px}
.foot{margin-top:38px;padding-top:14px;border-top:1px solid var(--line);
      color:var(--mute);font-size:12px;line-height:1.8}
.hide{display:none}
#cnt{color:var(--mute);font-size:13px;margin-left:auto}
</style></head><body><div class="wrap">

<h1>语料三图谱 · 生物力学与上肢运动学</h1>
<p class="sub">支撑「视觉图像→关节角的语言翻译（ViKiT）」论文的两块知识底座：视觉运动学估计与验证 · 肌肉骨骼动力学与仿真</p>

<div class="warn">
<b>口径警告</b>：下列五组数字含义不同，<b>不可相加、不可互相替代</b>。
「候选池 3567 条」是检索命中，不是收录；只有「已交叉证实」才可被论文引用。
相关度、证据范式、线程归属均为<b>人工裁定</b>，不是来源官方字段。
</div>

<div class="counts">
  <div class="cnt"><div class="k">理论搜索空间</div><div class="v">%(nq)d</div><div class="d">检索式条数</div></div>
  <div class="cnt"><div class="k">实际命中</div><div class="v">%(pool)d</div><div class="d">候选池去重条数</div></div>
  <div class="cnt"><div class="k">人工裁定</div><div class="v">%(cur)d</div><div class="d">进入候选清单</div></div>
  <div class="cnt"><div class="k">元数据证实</div><div class="v">%(ok)d</div><div class="d">可被引用</div></div>
  <div class="cnt"><div class="k">已入库 PDF</div><div class="v">%(ing)d</div><div class="d">通过三重校验</div></div>
</div>
<p class="sub">证据等级：<b>A %(gA)d 篇</b>（≥2 独立来源确认标题）· <b>B %(gB)d 篇</b>（单来源确认 ≥0.90）· 另有 %(pend)d 条 C/D 级待人工裁定，不写入可引用索引。</p>

<h2>双轴矩阵：线程 × 证据范式</h2>
<div class="mx">%(matrix)s</div>

<h2>可检索清单（%(total)d 篇已证实）</h2>
<div class="bar">
  <input id="q" type="search" placeholder="搜索标题、期刊、DOI…">
  <select id="fth"><option value="">全部线程</option>%(opts_th)s</select>
  <select id="fln"><option value="">全部范式</option>%(opts_ln)s</select>
  <select id="frl"><option value="">全部相关度</option><option>高</option><option>中</option><option>低</option></select>
  <button class="ghost" id="rs">重置</button>
  <span id="cnt"></span>
</div>
<div class="grid" id="g">%(cards)s</div>

<div class="foot">
<b>已知限制（如实声明）</b><br>
· B5 线程候选池仅由 Crossref 生成（OpenAlex 在本机 IP 硬 429），召回面窄于其他线程 —— <b>其「未命中」不等于「不存在」</b>。<br>
· 定位卡为<b>摘要级</b>，不是全文翻译；任何数值、方法细节、结论都必须回原文核对。<br>
· 「OA 但被出版商拦截」条目未下载：MDPI 返回 <code>bm-verify</code> 反爬令牌页、Springer/BMC 返回 JS 挑战页 —— <b>程序化绕过属规避访问控制，本项目不做</b>，已在取件清单附精确 OA 链接交人工浏览器下载。<br>
· 非 OA 条目须经机构订阅取件；不使用影子图书馆。<br>
· 检索为英文单语，<b>未覆盖中文与非英语文献</b>。<br>
· 数据源：Crossref（标题+DOI 权威记录）· Unpaywall（按 DOI 的 OA 定位）· arXiv API 与 abs 页面（预印本元数据）。
</div>
</div>
<script>
const cards=[...document.querySelectorAll('.card')],q=document.getElementById('q'),
th=document.getElementById('fth'),ln=document.getElementById('fln'),rl=document.getElementById('frl'),
cnt=document.getElementById('cnt');
function apply(){
  const s=q.value.trim().toLowerCase();
  let n=0;
  cards.forEach(c=>{
    const ok=(!s||c.textContent.toLowerCase().includes(s))
      &&(!th.value||c.dataset.th===th.value)
      &&(!ln.value||c.dataset.ln===ln.value)
      &&(!rl.value||c.dataset.rl===rl.value);
    c.classList.toggle('hide',!ok); if(ok)n++;
  });
  cnt.textContent=n+' / '+cards.length+' 篇';
}
[q,th,ln,rl].forEach(e=>e.addEventListener('input',apply));
document.getElementById('rs').onclick=()=>{q.value='';th.value='';ln.value='';rl.value='';apply();};
apply();
</script>
</body></html>""" % {
        'nq': c['理论搜索空间_检索式条数'],
        'pool': c['实际命中_候选池去重条数'],
        'cur': c['人工裁定_候选清单条数'],
        'ok': c['元数据证实_可引用条数'],
        'ing': c.get('已入库_PDF数', 0),
        'gA': g.get('A', 0), 'gB': g.get('B', 0),
        'pend': c['待裁定_C或D级条数'],
        'matrix': matrix_html,
        'cards': cards,
        'total': len(papers),
        'opts_th': ''.join('<option value="%s">%s %s</option>' % (t, t, escape(thread_names.get(t, '')))
                           for t in THREADS),
        'opts_ln': ''.join('<option value="%s">%s %s</option>' % (l, l, escape(lens_names.get(l, '')))
                           for l in LENSES),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding='utf-8')
    print('展示页已生成：%s' % OUT.resolve())
    print('  条目 %d · 矩阵 %d 格 · 字节 %d' % (len(papers), len(THREADS) * len(LENSES), len(html.encode())))


if __name__ == '__main__':
    main()
