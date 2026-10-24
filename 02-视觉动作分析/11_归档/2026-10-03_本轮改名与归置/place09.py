#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 09_章隆彬_合作团队/papers/ 下 7 篇未译论文归置到该线既有命名规范。

规范（取自该线已译的 6 篇，非自拟）：
    papers/Zhang20XX-Venue：中文描述/
        ├── <目录名>.pdf
        ├── <目录名>_全文翻译.md
        ├── <目录名>_翻译导读.md
        └── images/

篇目与著录信息取自该线 `LIST.md` §一 的 13 篇总览；两个 `1-s2.0-*` 的题名已从 PDF 首页正文逐条核验
（不靠文件名推测）：S0021-9290 = J Biomech 2025「Do gait patterns normalize…flatfoot?」，
S0966-6362 = Gait & Posture 2023「An approach to establishing the thresholds of plantar loading…」。
"""
import hashlib
import os
import shutil
import sys

BASE = '09_章隆彬_合作团队/papers'

# 原路径（相对 BASE） -> 目标目录名
PLAN = {
    'Global_and_Local_Spatio-Temporal_Encoder_for_3D_Human_Pose_Estimation.pdf':
        'Zhang2024-TMM：全局与局部时空编码器的三维人体姿态估计',
    'Design and Evaluation of a Single-Sided Mobility Assistive Exoskeleton'
    '                                                           for Hemiplegia .pdf':
        'Zhang2025-ICORR：偏瘫单侧移动辅助外骨骼的设计与评估',
    'Muscle_Activation_and_Postural_Sway_in_Response_to_Task_Complexity_'
    'A_Study_of_Balance_Control_in_Older_Adults.pdf':
        'Zhang2025-ICORR：任务复杂度下的肌电激活与姿态摇摆',
    'Ankle_Joint_Torque_Estimation_Using_an_EMG-Driven_Neuromusculoskeletal_Model_'
    'and_an_Artificial_Neural_Network_Model.pdf':
        'Zhang2021-TASE：肌电驱动 NMS 模型与 ANN 的踝关节力矩估计',
    '1-s2.0-S0021929025002702-main.pdf':
        'Zhang2025-JBiomech：扁平足儿童足弓与步态同步恢复的前瞻性队列研究',
    '1-s2.0-S0966636223012936-main.pdf':
        'Zhang2023-GaitPosture：肥胖儿童足底负荷阈值的确定方法',
    # ICARM 的目标目录已存在，只是里面的 PDF 仍是出版商原名
    'Zhang2023-ICARM：NMS 求解器与高斯过程的膝关节力矩预测/NMS_Solver_Informed_GP_Knee_Joint_Torque_2023_ICARM.pdf':
        'Zhang2023-ICARM：NMS 求解器与高斯过程的膝关节力矩预测',
}

APPLY = '--apply' in sys.argv


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


plan = []
for src_rel, dirname in PLAN.items():
    src = os.path.join(BASE, src_rel)
    dstdir = os.path.join(BASE, dirname)
    dst = os.path.join(dstdir, dirname + '.pdf')
    plan.append({'src': src, 'dst': dst, 'dir': dstdir, 'hash': sha(src)})

print('待归置 %d 篇' % len(plan))
for p in plan:
    print('  %s' % p['src'][:78])
    print('    → %s' % os.path.relpath(p['dst'], BASE)[:78])

if not APPLY:
    sys.exit(0)

for p in plan:
    os.makedirs(p['dir'], exist_ok=True)
    if os.path.exists(p['dst']) and os.path.abspath(p['dst']) != os.path.abspath(p['src']):
        print('!! 目标已存在，跳过：%s' % p['dst'])
        continue
    shutil.move(p['src'], p['dst'])
    ok = sha(p['dst']) == p['hash']
    print('%s %s' % ('OK  ' if ok else 'HASH MISMATCH ', os.path.relpath(p['dst'], BASE)[:70]))
