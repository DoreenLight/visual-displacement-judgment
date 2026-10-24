#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 09 线 7 篇已存在的 `papers/*_zh.md` 归位到各篇目录并改名为 `<目录名>_全文翻译.md`。

📌 本脚本落地的前提是一个**推翻 LIST.md 记载的实测结论**：
`09_章隆彬_合作团队/LIST.md` 写「首批 6 篇已全译；**增补 7 篇待译**」，
但实测 `papers/` 顶层**本来就有 7 个 `*_zh.md`**，逐个核验体量为 8,492–59,113 字符，
与该线既有 6 篇的「全文翻译」（18,165–65,363 字符）同量级（GaitPosture 那篇是 1 页会议摘要，
译文短属正常），并非占位文件。
⇒ 7 篇**早就译完了**，只是该线当时用了 `*_zh.md` 这一套旁路命名，
而 `verify_kb.sh` Case 7 只认 `<stem>_全文翻译.md` / `<stem>_reconstruction.md`。

因此**不需要重跑翻译管线**（省掉 MinerU 抽取 + MiniMax-M3 全文翻译 + 校验的整轮开销），
只需把命名归一到该线既有规范。

图片引用已逐篇核验：7 个文件合计只有 1 处 `![...](placeholder)`，无真实图片依赖 ⇒ 移动不产生断图。
"""
import hashlib
import os
import shutil
import sys

BASE = '09_章隆彬_合作团队/papers'

MAP = {
    'Zhang2021-TASE-AnkleTorque_zh.md':
        'Zhang2021-TASE：肌电驱动 NMS 模型与 ANN 的踝关节力矩估计',
    'Zhang2023-GaitPosture-PlantarLoading_zh.md':
        'Zhang2023-GaitPosture：肥胖儿童足底负荷阈值的确定方法',
    'Zhang2023-ICARM-KneeTorqueGP_zh.md':
        'Zhang2023-ICARM：NMS 求解器与高斯过程的膝关节力矩预测',
    # 原名 Zhang2023-TMM 用的 online-first 年份；Crossref 核 issued=2024、IEEE TMM VOL.26 (2024)
    'Zhang2023-TMM-3DPoseEstimator_zh.md':
        'Zhang2024-TMM：全局与局部时空编码器的三维人体姿态估计',
    'Zhang2025-ICORR-PosturalSway_zh.md':
        'Zhang2025-ICORR：任务复杂度下的肌电激活与姿态摇摆',
    'Zhang2025-ICORR-SMAEXO_zh.md':
        'Zhang2025-ICORR：偏瘫单侧移动辅助外骨骼的设计与评估',
    'Zhang2025-JBiomech-FootArches_zh.md':
        'Zhang2025-JBiomech：扁平足儿童足弓与步态同步恢复的前瞻性队列研究',
}

APPLY = '--apply' in sys.argv


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


items = []
for src_name, dirname in MAP.items():
    src = os.path.join(BASE, src_name)
    dst = os.path.join(BASE, dirname, dirname + '_全文翻译.md')
    items.append((src, dst, sha(src)))

print('待归位 %d 份译文' % len(items))
for src, dst, h in items:
    print('  %-44s → %s' % (os.path.basename(src), os.path.basename(dst)[:56]))

if not APPLY:
    sys.exit(0)

for src, dst, h in items:
    if not os.path.exists(os.path.dirname(dst)):
        print('!! 目标目录不存在：%s' % os.path.dirname(dst))
        continue
    if os.path.exists(dst):
        print('!! 目标已存在，跳过：%s' % dst)
        continue
    shutil.move(src, dst)
    print('%s %s' % ('OK  ' if sha(dst) == h else 'HASH MISMATCH ', os.path.basename(dst)[:60]))
