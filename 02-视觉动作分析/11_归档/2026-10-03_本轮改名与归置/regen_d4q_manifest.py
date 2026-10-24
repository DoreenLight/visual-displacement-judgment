#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重算 D435i 四问任务包的 文件清单.sha256。

为什么必须重算：
  - 2026-10-03 上一轮给 `Agent.md` §4 增补第 8–10 条时**没有同步本清单**，
    验收命令 `(cd 04_实验设计/D435i_四问任务包 && shasum -a 256 -c 文件清单.sha256)`
    自那时起就是 FAILED —— 属既存漂移，不是本轮引入。
  - 本轮 C13 的 ID 前缀化又改动了 01/02/03/04 四个任务包与风险清单共 5 个文件。

纪律：**只按清单里已登记的文件列表重算**，不增删条目 ——
清单覆盖哪些文件是 AGENT.md §3 的结构定义决定的，不是本脚本的裁量。
历史归档的「原件哈希清单」是**原件**的证据（AGENT.md §4-3），
它记录的是归档当时的历史字节，**永不重算**。
"""
import hashlib
import os
import sys

BASE = '04_实验设计/D435i_四问任务包'
MANIFEST = os.path.join(BASE, '文件清单.sha256')

rows = []
for line in open(MANIFEST, encoding='utf-8'):
    line = line.rstrip('\n')
    if not line.strip():
        continue
    _, rel = line.split('  ', 1)
    p = os.path.join(BASE, rel)
    if not os.path.exists(p):
        print('!! 清单登记的文件不存在：%s' % rel)
        sys.exit(1)
    h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    rows.append('%s  %s' % (h, rel))

if '--apply' in sys.argv:
    open(MANIFEST, 'w', encoding='utf-8').write('\n'.join(rows) + '\n')
    print('已重算 %d 条' % len(rows))
else:
    old = {}
    for line in open(MANIFEST, encoding='utf-8'):
        if line.strip():
            o, r = line.rstrip('\n').split('  ', 1)
            old[r] = o
    ch = sum(1 for r in rows if old.get(r) != r.split('  ')[0])
    print('登记 %d 条，将变更 %d 条（%s）' % (len(rows), ch, 'APPLY' if '--apply' in sys.argv else 'DRY-RUN'))
    for r in rows:
        if old.get(r) != r.split('  ')[0]:
            print('  ~ %s' % r.split('  ')[1])
