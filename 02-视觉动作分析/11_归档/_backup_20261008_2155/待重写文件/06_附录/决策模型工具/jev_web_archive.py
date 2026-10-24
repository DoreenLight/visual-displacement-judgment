#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""语料四：网页型证据存档器。

语料四有相当比例条目是**网页型证据**（厂商文档 / 社区实测 / 黑盒探测记录），
它们没有 PDF 可下载，但也不能当作传闻忽略。本脚本把公开可访问的页面
抓成 `<条目名>_网页存档.md`，与 PDF 同级存放。

**纪律**：
- 只抓公开可访问页面，不绕过任何验证、不使用影子来源
- 抓不动的（403 / 反爬 / 需订阅）如实记为 blocked，不伪造内容
- 每份存档带抓取元数据：HTTP 状态、字节数、SHA256、抓取时间
- 存档开头显式声明「这是网页快照，不是同行评审论文」

隔离点：只写 01_参考论文/决策模型线_JEV/ 与自己的报告文件。
"""
import hashlib
import html
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
JV = ROOT / '01_参考论文' / '决策模型线_JEV'
REPORT = HERE / 'jev_web_archive_report.json'
UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126 Safari/537.36'

SKIP = re.compile(r'<script.*?</script>|<style.*?</style>|<nav.*?</nav>|<footer.*?</footer>|'
                  r'<svg.*?</svg>|<noscript.*?</noscript>', re.S | re.I)
DROP_TAGS = re.compile(r'<(br|/p|/div|/li|/tr|/h[1-6]|/section|/article)\s*/?>', re.I)


def fetch(url, timeout=35):
    """返回 (正文, HTTP 状态, 字节数, 错误说明)。"""
    try:
        out = subprocess.run(
            ['curl', '-sL', '--max-time', str(timeout), '-A', UA,
             '-w', '\n%{http_code}', url],
            capture_output=True, timeout=timeout + 10)
        raw = out.stdout
        if b'\n' not in raw[-5:]:
            return '', 0, 0, '响应异常'
        body, _, code = raw.rpartition(b'\n')
        return body, int(code.strip() or 0), len(body), ''
    except Exception as e:                                            # noqa: BLE001
        return '', 0, 0, type(e).__name__


def html_to_md(raw, url):
    t = raw.decode('utf-8', 'ignore')
    t = re.sub(r'<!--.*?-->', '', t, flags=re.S)
    t = SKIP.sub('\n', t)
    m = re.search(r'<title[^>]*>(.*?)</title>', t, re.S | re.I)
    title = html.unescape(re.sub(r'\s+', ' ', m.group(1)).strip()) if m else url
    m = re.search(r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']', t, re.S | re.I)
    desc = html.unescape(m.group(1).strip()) if m else ''
    t = DROP_TAGS.sub('\n', t)
    t = re.sub(r'<li[^>]*>', '- ', t, flags=re.I)
    t = re.sub(r'<h([1-6])[^>]*>', lambda m: '\n' + '#' * int(m.group(1)) + ' ', t, flags=re.I)
    t = re.sub(r'<[^>]+>', ' ', t)
    t = html.unescape(t)
    t = re.sub(r'[ \t\xa0]+', ' ', t)
    t = re.sub(r'\n\s*\n\s*\n+', '\n\n', t).strip()
    if len(t) < 400:                       # 多半是 SPA 空壳或挑战页
        return title, desc, ''
    return title, desc, t


HEADER = """# {title}

> **这是网页快照，不是同行评审论文。**
> 来源：<{url}>
> 抓取时间：{ts}
> HTTP 状态：{code}　｜　原始字节：{size}　｜　SHA256：`{sha}`
> **用途**：本条在语料四中的证据等级为「{evidence}」。引用时必须标注为网页型证据，
> 不得与同行评审论文并列。抓取日期之后页面可能变化，引用前请核对原链接。

{desc_block}

---

"""


def main():
    idx = json.loads((JV / 'index.json').read_text(encoding='utf-8'))
    items = [p for p in idx['papers'] if not p.get('pdf_path') and (p.get('url_abs') or '').startswith('http')]
    items = [p for p in items if p.get('evidence') != '编者判断']   # 编者判断是自己写的，不抓
    ok = blocked = 0
    out = []
    for n, p in enumerate(items, 1):
        url = p['url_abs']
        th = JV / ('%s_%s' % (p['thread'], p['thread_name']))
        if not th.is_dir():
            continue
        stem = re.sub(r'[\\/:*?"<>|]', '_', p['title'].strip())[:60].rstrip()
        dst = th / (stem + '_网页存档.md')
        if dst.exists():
            print('[%2d] = 已存在 %s' % (n, dst.name[:52])); ok += 1; continue
        body, code, size, err = fetch(url)
        title, desc, text = ('', '', '')
        if body and code == 200:
            title, desc, text = html_to_md(body, url)
        if len(text) < 400:
            blocked += 1
            print('[%2d] ⛔ HTTP %s %s  %s' % (n, code, err or '(空壳/挑战页)', url[:60]))
            out.append({'title': p['title'], 'url': url, 'status': 'blocked',
                        'http': code, 'bytes': size, 'error': err or '空壳或反爬挑战页'})
            continue
        sha = hashlib.sha256(body).hexdigest()[:16]
        desc_block = ('**页面描述**：%s\n' % desc) if desc else ''
        dst.write_text(HEADER.format(title=title or p['title'], url=url,
                                     ts=datetime.now().strftime('%Y-%m-%d %H:%M'),
                                     code=code, size=f'{size:,}', sha=sha,
                                     evidence=p.get('evidence', '—'),
                                     desc_block=desc_block) + text, encoding='utf-8')
        ok += 1
        print('[%2d] ✅ HTTP %s  %7d B → %s' % (n, code, len(text.encode()), dst.name[:46]))
        out.append({'title': p['title'], 'url': url, 'status': 'ok', 'http': code,
                    'bytes': size, 'sha256_16': sha, 'path': str(dst.relative_to(ROOT)),
                    'text_chars': len(text)})
        time.sleep(1.2)                        # 礼貌间隔
    REPORT.write_text(json.dumps({'generated': 'jev_web_archive.py',
                                  'ok': ok, 'blocked': blocked, 'items': out},
                                 ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n网页存档：成功 %d ／ 拦截 %d　报告 %s' % (ok, blocked, REPORT.relative_to(ROOT)))


if __name__ == '__main__':
    main()
