#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""执行 title_of 大写归一带来的重命名（K44）。

⚠️ 执行顺序不能反：新旧文件名**仅大小写不同**
（`AN EXTENSION TO THE ISB WRIST…_AN IN` vs `An extension to the ISB wrist…_An in`），
在 macOS 默认的大小写不敏感 APFS 上二者是**同一个路径**。
若先跑 build 再处理旧卡，新卡会直接覆盖旧卡、旧的进回收站证据就没了；
若先改 PDF 名再跑 build，则 `dstat()` 的「报告 OK 但磁盘缺」判定会误报。
故采用：**① 改 PDF 名 → ② 回收旧卡 → ③ 跑 build 生成新卡**。
"""
import glob
import hashlib
import os
import shutil
import subprocess
import sys

BM = '01_参考论文/生物力学线'
PAIRS = [
    ('B1_视觉运动学_估计与验证方法学',
     'AN EXTENSION TO THE ISB WRIST JOINT COORDINATE SYSTEM_ AN IN',
     'An extension to the ISB wrist joint coordinate system_ An in'),
    ('B1_视觉运动学_估计与验证方法学',
     'BIOMECHANICS OF PRONATION AND SUPINATION OF THE FOREARM',
     'Biomechanics of pronation and supination of the forearm'),
]
TRASH = '/Users/xbpd/.minimax/bin/mavis-trash'


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


for sub, old, new in PAIRS:
    pdf_old = os.path.join(BM, sub, old + '.pdf')
    pdf_new = os.path.join(BM, sub, new + '.pdf')
    card_old = os.path.join(BM, sub, old + '_Abstract_Card.md')
    h = sha(pdf_old) if os.path.exists(pdf_old) else None
    print('── %s' % old[:60])
    print('   PDF  存在=%s' % bool(h))
    print('   旧卡 存在=%s' % os.path.exists(card_old))
    if h is None or not os.path.exists(card_old):
        print('   ⚠️ 缺件，跳过'); continue
    if '--apply' not in sys.argv:
        print('   （DRY-RUN）'); continue
    os.rename(pdf_old, pdf_new)
    print('   %s PDF 改名，hash %s' % ('OK  ' if sha(pdf_new) == h else 'HASH MISMATCH', h[:12]))
    r = subprocess.run([TRASH, '--', card_old], capture_output=True, text=True)
    print('   %s 旧卡回收' % ('OK  ' if r.returncode == 0 else 'FAILED ' + r.stderr[:80]))
