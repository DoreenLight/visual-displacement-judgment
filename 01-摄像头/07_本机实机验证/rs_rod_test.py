#!/usr/bin/env python3
"""细杆可用性专项测试：3mm 金属杆在 20cm 处能否被有效测量。

针对性强，因为三重风险叠加：
  1. 杆宽 6.4px（左右各 1px 边缘占 31% 宽度）
  2. 金属镜面反射 → 红外散斑被反射走
  3. 20cm 压在 D435i Min-Z 边界

本脚本回答
----------
Q1. 杆在深度图里到底是什么样：有读数？空洞？还是被背景"吞掉"？
Q2. 杆的有效像素数与深度一致性如何？
Q3. 为了去掉边缘污染，内核（去掉左右各 N 列）后精度是多少？
Q4. 杆的**长轴方向**能否用来平均？（261px 长是最大优势）
Q5. 换 4 种表面处理建议的预期改善（用实测数据反推）

用法
----
sudo -E .../rs_rod_test.py                    # 自动找画面中最细的竖直条状目标
sudo -E .../rs_rod_test.py --roi 380,120,20,240
sudo -E .../rs_rod_test.py --frames 300
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import macos_usb_fix
    macos_usb_fix.ensure_ready(verbose=False)
except Exception:
    pass

try:
    import pyrealsense2 as rs
except ImportError:
    sys.exit("缺少 pyrealsense2。")

W, H, FPS = 848, 480, 30


def make_pipe(alpha, cw=640, ch=480):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.color, cw, ch, rs.format.bgr8, FPS)
    prof = pipe.start(cfg)
    tf = None
    if alpha > 0:
        tf = rs.temporal_filter()
        tf.set_option(rs.option.filter_smooth_alpha, alpha)
        tf.set_option(rs.option.filter_smooth_delta, 20)
    align = rs.align(rs.stream.color)
    return pipe, tf, align, prof


def grab(pipe, tf, align):
    fs = pipe.wait_for_frames(timeout_ms=10000)
    fs = align.process(fs)
    df = fs.get_depth_frame()
    if tf is not None:
        df = tf.process(df)
    d = np.asanyarray(df.get_data())
    c = np.asanyarray(fs.get_color_frame().get_data())
    return d, c


def find_thin_target(med):
    """自动定位最细的竖直条状候选：宽 <12px、高 >60px、有效率高。"""
    # 与背景深度差异明显的区域
    v = (med > 0) & (med < 65535)
    if v.sum() < 1000:
        return None
    cands = []
    # 用形态学思路：对每个列区间找连续有效段
    colvalid = v.sum(axis=0)
    for x0 in range(0, med.shape[1] - 8, 4):
        for wid in (6, 8, 10, 12):
            x1 = x0 + wid
            if x1 >= med.shape[1]:
                continue
            sub = v[:, x0:x1]
            rowok = sub.all(axis=1)
            # 最长连续段
            best = cur = 0
            for k in rowok:
                cur = cur + 1 if k else 0
                best = max(best, cur)
            if best >= 60:
                zs = med[rowok, x0:x1]
                if zs.size and 100 < np.median(zs) < 600:
                    cands.append((best, wid, x0, float(np.median(zs))))
    if not cands:
        return None
    cands.sort(key=lambda t: (-t[0], t[1]))
    return cands[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi", default=None, help="x,y,w,h（基于彩色 640x480）")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--warmup", type=int, default=45)
    args = ap.parse_args()

    print("=" * 98)
    print("3mm 金属杆 @ 20cm 可用性专项测试")
    print("=" * 98)

    pipe, tf, align, prof = make_pipe(args.alpha)
    for _ in range(args.warmup):
        pipe.wait_for_frames(timeout_ms=10000)

    stack = np.stack([grab(pipe, tf, align)[0].astype(np.float32)
                      for _ in range(15)])
    med = np.median(stack, axis=0)
    h, w = med.shape
    print(f"\n对齐后深度图 {w}x{h}（与彩色同尺寸）")

    v = (med > 0) & (med < 65535)
    print(f"整体有效率 {100*v.mean():.1f}%")

    if args.roi:
        x, y, rw, rh = (int(t) for t in args.roi.split(","))
    else:
        got = find_thin_target(med)
        if got is None:
            print("\n❌ 没能自动找到合适的细杆目标。")
            print("   请用 --roi x,y,w,h 手动指定（坐标基于彩色 640x480）。")
            pipe.stop()
            return 1
        best, wid, x0, zmed = got
        x, rw = x0, wid
        y, rh = 0, h
        # 收紧到实际有读数的行
        sub = v[:, x:x + rw].all(axis=1)
        rows = np.where(sub)[0]
        y, rh = int(rows.min()), int(rows.max() - rows.min() + 1)
        print(f"\n自动定位: 列 {x}..{x+rw} 宽 {rw}px, 行 {y}..{y+rh} "
              f"长 {rh}px, 中位深度 {zmed:.0f}mm")

    print(f"\n使用 ROI: x={x} y={y} w={rw} h={rh}  "
          f"（理论面积 {rw*rh}px）")

    # ── Q1: 杆区域到底长什么样 ──
    print("\n" + "=" * 98)
    print("Q1. 杆区域的深度结构（逐列统计）")
    print("=" * 98)
    print(f"{'列':>5} {'有效px':>7} {'有效率':>7} {'中位Z':>8} {'σ':>7} {'备注'}")
    print("-" * 98)
    col_stats = []
    for c in range(x, x + rw):
        col = med[y:y + rh, c]
        cv = (col > 0) & (col < 65535)
        rate = 100.0 * cv.mean()
        if cv.sum() >= 3:
            zc = col[cv]
            zmed, zsig = float(np.median(zc)), float(np.std(zc))
        else:
            zmed, zsig = np.nan, np.nan
        note = ""
        if rate < 50:
            note = "← 大量空洞"
        col_stats.append((c, int(cv.sum()), rate, zmed, zsig, note))
        print(f"{c:>5} {int(cv.sum()):>7} {rate:>6.1f}% "
              f"{zmed if np.isfinite(zmed) else 0:>8.0f} "
              f"{zsig if np.isfinite(zsig) else 0:>7.1f} {note}")

    # ── Q2: 整体统计 ──
    print("\n" + "=" * 98)
    print("Q2. 杆区域整体可用性")
    print("=" * 98)
    roi = med[y:y + rh, x:x + rw]
    rv = (roi > 0) & (roi < 65535)
    tot = roi.size
    print(f"  区域总面积        {tot} px")
    print(f"  有效深度像素      {int(rv.sum())} px ({100*rv.sum()/tot:.1f}%)")
    print(f"  空洞（=0）        {int((roi==0).sum())} px ({100*(roi==0).sum()/tot:.1f}%)")
    print(f"  离群（=65535）    {int((roi==65535).sum())} px")
    if rv.sum():
        vals = roi[rv]
        print(f"  深度中位          {np.median(vals):.0f} mm")
        print(f"  深度分布 σ        {np.std(vals):.2f} mm")
        print(f"  p5 / p95          {np.percentile(vals,5):.0f} / "
              f"{np.percentile(vals,95):.0f} mm")
    else:
        print("  ❌ 完全无有效深度 —— 该目标不可测")
        pipe.stop()
        return 2

    # ── Q3: 内核 vs 全宽 ──
    print("\n" + "=" * 98)
    print("Q3. 去边缘影响：内核 vs 全宽（时间序列对比）")
    print("=" * 98)

    variants = {"全宽（含边缘）": (x, x + rw)}
    if rw >= 6:
        variants["去左右各1列"] = (x + 1, x + rw - 1)
    if rw >= 8:
        variants["去左右各2列"] = (x + 2, x + rw - 2)
    if rw >= 10:
        variants["去左右各3列"] = (x + 3, x + rw - 3)

    series = {k: np.full(args.frames, np.nan) for k in variants}
    cnts = {k: np.zeros(args.frames, np.int32) for k in variants}

    for t in range(args.frames):
        d, _ = grab(pipe, tf, align)
        for k, (a, b) in variants.items():
            sub = d[y:y + rh, a:b]
            vals = sub[(sub > 0) & (sub < 65535)]
            cnts[k][t] = int(vals.size)
            if vals.size >= 10:
                series[k][t] = float(vals.mean())

    print(f"{'变体':>16} {'宽px':>5} {'平均有效px':>10} {'空间σ':>8} "
          f"{'MDR_diff':>10} {'MDR_scat':>10} {'漂移':>8}")
    print("-" * 98)
    for k, (a, b) in variants.items():
        s = series[k]
        if not np.isfinite(s).all():
            print(f"{k:>16} {b-a:>5} {'—':>10} ⚠️ 数据不完整")
            continue
        dif = np.diff(s)
        mdr_d = 3 * float(np.std(dif, ddof=1)) / np.sqrt(2) if dif.size > 1 else np.nan
        scat = float(np.std(s, ddof=1))
        nw = len(s) // FPS
        drift = (float(np.ptp(s[:nw*FPS].reshape(nw, FPS).mean(axis=1)))
                 if nw >= 2 else np.nan)
        sp = np.std(med[y:y+rh, a:b][(med[y:y+rh, a:b] > 0) &
                                     (med[y:y+rh, a:b] < 65535)])
        print(f"{k:>16} {b-a:>5} {cnts[k].mean():>10.0f} {sp:>8.1f} "
              f"{mdr_d:>10.3f} {3*scat:>10.3f} {drift:>8.3f}")

    # ── Q4: 长轴分段平均 ──
    print("\n" + "=" * 98)
    print("Q4. 沿杆长分段（检验长轴方向的空间均匀性）")
    print("=" * 98)
    inner = (max(x + 1, x), min(x + rw - 1, x + rw)) if rw > 4 else (x, x + rw)
    nseg = 4
    print(f"{'段':>4} {'行范围':>12} {'中位Z':>8} {'段内σ':>8} {'与整体差':>9}")
    print("-" * 98)
    seg_z = []
    overall = float(np.median(med[y:y+rh, inner[0]:inner[1]][
        (med[y:y+rh, inner[0]:inner[1]] > 0) &
        (med[y:y+rh, inner[0]:inner[1]] < 65535)]))
    for i in range(nseg):
        a = y + i * rh // nseg
        b = y + (i + 1) * rh // nseg
        sub = med[a:b, inner[0]:inner[1]]
        vals = sub[(sub > 0) & (sub < 65535)]
        if vals.size < 3:
            print(f"{i+1:>4} {'—':>12} (无效)")
            continue
        seg_z.append(float(np.median(vals)))
        print(f"{i+1:>4} {a:>5}-{b:<6} {np.median(vals):>8.0f} "
              f"{np.std(vals):>8.1f} {np.median(vals)-overall:>9.1f}")
    if len(seg_z) >= 2:
        print(f"\n  各段中位深度的散布 σ = {np.std(seg_z):.2f} mm")
        print(f"  ← 若此值大，说明杆在深度上不是恒定值（倾斜/弯曲/边缘效应）")

    # ── Q5: 判定 ──
    print("\n" + "=" * 98)
    print("Q5. 判定与建议")
    print("=" * 98)
    best_k = None
    best_mdr = 1e9
    for k in variants:
        s = series[k]
        if np.isfinite(s).all():
            dif = np.diff(s)
            m = 3 * float(np.std(dif, ddof=1)) / np.sqrt(2)
            if m < best_mdr:
                best_mdr, best_k = m, k

    rr = roi[rv]
    holes = 100.0 * (roi == 0).sum() / roi.size
    print(f"  最优统计方式: {best_k}   MDR = {best_mdr:.3f} mm")
    print(f"  空洞率: {holes:.1f}%")
    print()
    if best_mdr <= 2 and holes < 30:
        print(f"  ✅ 达标：MDR {best_mdr:.3f}mm ≤ 2mm（老师要求）")
        print(f"     可直接进入测试计划阶段 1。")
    elif best_mdr <= 2:
        print(f"  ⚠️ MDR 达标（{best_mdr:.3f}mm）但空洞率偏高（{holes:.0f}%）")
        print(f"     风险：移动时可能整片失效。建议先做表面处理。")
    else:
        print(f"  ❌ 未达标：MDR {best_mdr:.3f}mm > 2mm")
        print(f"     建议按下列顺序排查：")
        print(f"     1. 表面处理（金属→哑光）：贴哑光白纸/喷哑光漆/磨砂")
        print(f"     2. 加大直径：3mm→6-10mm（成像 12-21px）")
        print(f"     3. 拉远距离：20cm→30cm（但需重新标定基准）")

    # ── 表面处理预期 ──
    print("\n" + "=" * 98)
    print("表面处理方案对比（基于红外反射率的一般经验，非本机实测）")
    print("=" * 98)
    rows = [
        ("原状金属", "镜面反射，散斑被反射走", "差", "不推荐"),
        ("贴哑光白纸", "漫反射，红外反射率高", "优", "最简单，推荐先试"),
        ("喷哑光漆", "漫反射，均匀", "优", "需等干，约10分钟"),
        ("贴磨砂胶带", "漫反射，略有色", "良", "临时方案"),
        ("磨砂处理", "漫反射", "良", "需工具"),
        ("贴反光标记(Retroreflective)", "定向反射强，官方推荐", "最优", "需采购，但效果最好"),
    ]
    print(f"{'处理':>16} {'原理':>28} {'预期':>6} {'备注'}")
    print("-" * 98)
    for a, b, c, d in rows:
        print(f"{a:>16} {b:>28} {c:>6} {d}")

    pipe.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
