#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C13 剩余 8 份文档的本地 ID 前缀化（第一阶段：定义位 + 同文件内引用）。

依据 06_附录/ID命名约定.md §7。本轮对原计划做了 3 处修正（均为实测纠错，非计划变更）：
  ① 检索渠道记录.md 内部存在**两套同号异义的 A 系列**（K34 复发）：
     §「最终结果」A1..A4,A7 = 5 篇论文；§0 结果总表 A1/A2..A9/A10 = 10 个渠道槽位。
     原计划整文件加 `RCA-` 会把歧义**固化**成同前缀。改为**按行区段分别归属**。
  ② 待下载论文总清单.md 第 204 行的 `B4` 指的是**语料三的 B4 线程**，
     不是本文档的下载分组 B4 —— 加前缀会把线程引用改成下载分组引用（属主写错）。
  ③ 风险清单的编号是 **R01–R10（补零两位）**，约定 §6 记作「R1–R10」不准确。
另外纠正 §6 另一处：Anatomy_aware 的 R 系列实为 **R1–R4**（非 R1–R3）。
"""
import re, sys

SIMPLE = {
    '03_方法设计/Anatomy_aware_Region_Encoder.md': ('ARE', ['F1','F2','F3','F4','F5','F6','F7','R1','R2','R3','R4']),
    '03_方法设计/Spatio_Temporal_Encoder.md':     ('STE', ['F1','F2','F3','F4','F5','F6','F7','R1','R2','R3']),
    '04_实验设计/D435i_四问任务包/01_上肢关键点三维坐标任务包.md': ('TP01', ['D1','D2','D3','D4','D5','K1','K2','K3','K4']),
    '04_实验设计/D435i_四问任务包/02_腕关节运动轨迹任务包.md':       ('TP02', ['D1','D2','D3','D4','D5']),
    '04_实验设计/D435i_四问任务包/03_肩肘代偿识别任务包.md':         ('TP03', ['D1','D2','D3','D4','D5','C1','C2','C3','C4']),
    '04_实验设计/D435i_四问任务包/04_运动质量评价任务包.md':         ('TP04', ['D1','D2','D3','D4','D5','Q1','Q2','Q3','Q4']),
    '04_实验设计/D435i_四问任务包/D435i_实验方案与风险清单.md':       ('D4Q',  ['R01','R02','R03','R04','R05','R06','R07','R08','R09','R10']),
    '04_实验设计/README.md':                       ('EXR', ['M1','M2','M3','M4','M5']),
    '00_研究背景/JEV路线可行性评估.md':               ('JEV', ['E1','E2','E3']),
    '02_创新点矩阵/创新点查重_vs_竞品.md':             ('DUP', ['A1','A2','A3','A4','A5']),
}
HIST = ('编号更正','直接撞号','归还词表健康度','本节三条原用','本就该另起编号',
        '跨文件引用已核','变更记录','ID 命名空间冲突','C13')

def sub(line, pre, ids):
    for x in ids:
        line = re.compile(r'(?<![A-Za-z0-9_-])'+re.escape(x)+r'(?![0-9A-Za-z_-])').sub('%s-%s'%(pre,x), line)
    return line

APPLY = '--apply' in sys.argv
n = 0
for path,(pre,ids) in SIMPLE.items():
    lines = open(path,encoding='utf-8').read().split('\n'); c = 0
    for i,l in enumerate(lines):
        if any(h in l for h in HIST): continue
        new = sub(l, pre, ids)
        if new != l: lines[i] = new; c += 1
    n += c; print('%-52s 改 %2d 行' % (path, c))
    if APPLY: open(path,'w',encoding='utf-8').write('\n'.join(lines))

# ---- 行区段归属：检索渠道记录.md 的两套 A 系列 ----
P = '06_附录/检索渠道尝试记录.md'
lines = open(P,encoding='utf-8').read().split('\n'); c = 0
PAPER = ['A1','A2','A3','A4','A7']          # §「最终结果」：5 篇论文
SLOT  = ['A%d'%i for i in range(1,11)]     # §0 结果总表：10 个渠道槽位
for i,l in enumerate(lines, 1):
    if any(h in l for h in HIST): continue
    new = l
    if 15 <= i <= 19:  new = sub(l,'RCA',PAPER)
    elif 46 <= i <= 50: new = sub(l,'RCS',SLOT)
    if new != l: lines[i-1] = new; c += 1
n += c; print('%-52s 改 %2d 行  [行区段：15-19→RCA / 46-50→RCS]' % (P, c))
if APPLY: open(P,'w',encoding='utf-8').write('\n'.join(lines))

# ---- 待下载论文总清单.md：B1-B5 是下载分组，但提到「语料三」的行里的 B* 指的是线程 ----
P = '06_附录/待下载论文总清单.md'
lines = open(P,encoding='utf-8').read().split('\n'); c = 0
for i,l in enumerate(lines):
    if any(h in l for h in HIST) or '语料三' in l: continue
    new = sub(l,'DL',['B1','B2','B3','B4','B5'])
    if new != l: lines[i] = new; c += 1
n += c; print('%-52s 改 %2d 行  [含「语料三」的行跳过：其 B* 属线程]' % (P, c))
if APPLY: open(P,'w',encoding='utf-8').write('\n'.join(lines))
print('合计改动行：%d（模式：%s）' % (n, 'APPLY' if APPLY else 'DRY-RUN'))
