#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""【已退役 2026-10-03】本脚本不再生成任何文件。

## 为什么退役

它曾输出 `06_附录/需你下载的论文清单.md`。那个文件是**五份并存清单之一**，
且不是最新的一份 —— 同一诉求有五处副本时，没有一处能保证是最新的
（实测：五份唯一 DOI 仅 61 条，重叠 24/13/3 条）。

取件清单现已合并为**唯一一份**：

    产物    06_附录/待下载论文总清单.md
    生成器  06_附录/gen_acquisition_list.py
    叠层    06_附录/取件清单_人工标注.json

## 为什么保留这个存根而不是直接删掉

直接删掉的话，下一个人看到 `决策模型工具/` 里「没有下载清单生成器」，
很可能重新写一个 —— 然后又造出第二份真相。

所以留一个**会说话但不产出**的存根：任何人重跑它，都会立刻被指向唯一入口。
并且 `verify_kb.sh` Case 19 断言 `需你下载的论文清单.md` **不得存在** ——
真有人改了这段代码把输出打开，自检当场 FAIL。

原脚本已原样留档：
    11_归档/2026-10-03_待下载清单合并/gen_download_list.已退役.py
其硬编码的 13 条实测拦点（`BLOCK`）是实验事实；其中 4 条 PMC reCAPTCHA
记录对应的论文均已入库，拦点已转写进叠层的 `拦截类型实测汇总`。

## 用法

    python3 gen_download_list.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MASTER = ROOT / '06_附录' / '待下载论文总清单.md'
GEN = ROOT / '06_附录' / 'gen_acquisition_list.py'

print(__doc__)
print('=' * 72)
print('退役后的唯一入口：')
print('  产物   %s' % MASTER.relative_to(ROOT) if MASTER.exists()
      else '  ⚠️ 产物缺失，请跑：python3 %s' % GEN.relative_to(ROOT))
print('  生成器 %s  →  python3 %s' % (GEN.relative_to(ROOT), GEN.relative_to(ROOT)))
print('=' * 72)

sys.exit(0)
