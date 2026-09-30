#!/usr/bin/env python3
"""Z 轴位移的「最小可检测位移」与误差来源分解。

回答三个决定方案设计的关键问题
------------------------------
Q1. 最小可检测位移 (MDR) 是多少？
    用帧间 ROI 均值差分统计：若两帧之间目标静止，差分应只含噪声。
    Δ 的分布宽度直接给出「单次测量能分辨的最小位移」。

Q2. 为什么空间平均收益远低于理论 1/√N？
    分别考察：
      - 同一距离箱内不同子区域（左/中/右）是否同步波动  → 共模 vs 独立
      - 共模占比 = 1 - (独立部分方差 / 总方差)

Q3. 误差是传感器噪声，还是相机自身振动？
    若为相机振动，则**所有距离箱会同步漂移**；若为传感器噪声，
    则各箱独立。用箱间相关系数区分。这决定了方案要不要做相机固定/减振。

用法
----
sudo -E /opt/anaconda3/envs/realsense/bin/python3 rs_mdr_probe.py --seconds 30
"""

import argparse
import os
import sys
import time

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
    sys.exit("缺少 pyrealsense2，请用 conda 环境 realsense 的 python 运行。")

BINS = [(0, 300), (300, 500), (500, 800), (800, 1200), (1200, 1800),
        (1800, 2800), (2800, 5000), (5000, 9000)]
MIN_PX = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=848)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--no-emitter", action="store_true")
    args = ap.parse_args()

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, args.w, args.h, rs.format.z16, args.fps)
    profile = pipe.start(cfg)
    ds = profile.get_device().first_depth_sensor()
    if args.no_emitter:
        ds.set_option(rs.option.emitter_enabled, 0)

    nframes = int(args.seconds * args.fps)
    print(f"深度 {args.w}x{args.h}@{args.fps}   采集 {args.seconds}s ≈ {nframes} 帧")
    print(f"投射器 {'关' if args.no_emitter else '开'}   "
          f"深度单位 {ds.get_depth_scale()*1000:.4f} mm")

    print(f"预热 {args.warmup} 帧…")
    for _ in range(args.warmup):
        pipe.wait_for_frames(timeout_ms=10000)

    # 建立掩膜
    pre = []
    for _ in range(10):
        f = pipe.wait_for_frames(timeout_ms=10000)
        pre.append(np.asanyarray(f.get_depth_frame().get_data()).astype(np.float32))
    med = np.median(np.stack(pre), axis=0)

    labels = [f"{lo}-{hi}" if hi <= 1000 else f"{lo/1000:.1f}-{hi/1000:.1f}m"
              for lo, hi in BINS]
    masks = [((med > 0) & (med < 65535) & (med >= lo) & (med < hi))
             for lo, hi in BINS]

    # 时域采集
    series = np.full((nframes, len(BINS)), np.nan)
    t0 = time.time()
    for t in range(nframes):
        frames = pipe.wait_for_frames(timeout_ms=10000)
        d = np.asanyarray(frames.get_depth_frame().get_data())
        for i, m in enumerate(masks):
            sel = d[m & (d > 0) & (d < 65535)]
            if sel.size >= MIN_PX:
                series[t, i] = sel.mean()
    dur = time.time() - t0
    pipe.stop()
    print(f"完成 {dur:.1f}s → {nframes/dur:.1f} fps\n")

    keep = [i for i in range(len(BINS))
            if np.isfinite(series[:, i]).all()]
    if not keep:
        print("没有可用的距离箱。")
        return 1

    # ---------- Q1: 最小可检测位移 ----------
    print("=" * 92)
    print("Q1. 最小可检测位移 MDR（帧间差分，目标静止时差分=纯噪声）")
    print("=" * 92)
    print(f"{'距离箱':>12} {'中位Z':>9} {'帧间Δ σ':>10} {'MDR(3σ)':>10} "
          f"{'静置漂移':>10} {'可分辨':>8}")
    print(f"{'':>12} {'(mm)':>9} {'(mm)':>10} {'(mm)':>10} {'(mm)':>10}")
    print("-" * 92)

    mdr_rows = []
    for i in keep:
        s = series[:, i]
        z = float(np.nanmedian(s))
        dif = np.diff(s)
        dsig = float(np.nanstd(dif, ddof=1))
        # 相邻两帧差分含两帧噪声，换算到单帧
        single = dsig / np.sqrt(2)
        mdr3 = 3.0 * single
        # 静置漂移: 用 1 秒窗口均值的极差
        w = args.fps
        nw = len(s) // w
        if nw >= 2:
            wm = s[:nw * w].reshape(nw, w).mean(axis=1)
            drift = float(wm.max() - wm.min())
        else:
            drift = float("nan")
        print(f"{labels[i]:>12} {z:>9.1f} {single:>10.4f} {mdr3:>10.3f} "
              f"{drift:>10.3f} {'✅' if mdr3 <= 2 else ('⚠️' if mdr3 <= 5 else '❌')}")
        mdr_rows.append((labels[i], z, single, mdr3, drift))

    # ---------- Q2: 共模 vs 独立 ----------
    print("\n" + "=" * 92)
    print("Q2. 空间平均收益分解（同一距离箱内切成若干子块，看是否同步波动）")
    print("=" * 92)
    print(f"{'距离箱':>12} {'子块数':>8} {'单像素σ':>10} {'子块均值σ':>11} "
          f"{'独立分量σ':>11} {'共模占比':>10}")
    print("-" * 92)

    # 用第一组掩膜的子块, 重新采集一小段逐像素
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, args.w, args.h, rs.format.z16, args.fps)
    pipe.start(cfg)
    if args.no_emitter:
        pipe.get_device().first_depth_sensor().set_option(
            rs.option.emitter_enabled, 0)
    for _ in range(30):
        pipe.wait_for_frames(timeout_ms=10000)

    nq = 60
    h, w = med.shape
    stack = np.full((nq, h, w), np.nan, np.float32)
    for t in range(nq):
        frames = pipe.wait_for_frames(timeout_ms=10000)
        d = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32)
        stack[t] = np.where((d > 0) & (d < 65535), d, np.nan)
    pipe.stop()

    for i in keep[:6]:
        m = masks[i]
        ys, xs = np.where(m)
        if ys.size < 900:
            continue
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        # 切成 3x3 子块
        sub_series = []
        for ky in range(3):
            for kx in range(3):
                syr = (ys >= y0 + ky * (y1 - y0 + 1) // 3) & \
                      (ys < y0 + (ky + 1) * (y1 - y0 + 1) // 3)
                sxr = (xs >= x0 + kx * (x1 - x0 + 1) // 3) & \
                      (xs < x0 + (kx + 1) * (x1 - x0 + 1) // 3)
                sel = syr & sxr
                if sel.sum() < 50:
                    continue
                yy, xx = ys[sel], xs[sel]
                vals = stack[:, yy, xx]
                sub_series.append(np.nanmean(vals, axis=1))
        if len(sub_series) < 4:
            continue
        M = np.vstack(sub_series)           # (nsub, nq)
        px = stack[:, ys, xs]
        pix_sig = float(np.nanmedian(np.nanstd(px, axis=0, ddof=1)))
        sub_sig = float(np.nanstd(M.mean(axis=0), ddof=1))
        # 子块均值序列的平均方差 → 共模部分
        sub_mean_var = float(np.nanmean(np.var(M, axis=1, ddof=1)))
        nsub = M.shape[0]
        indep = np.sqrt(sub_mean_var / nsub)      # 若完全独立, 均值σ应为这个
        common_frac = 1.0 - (indep ** 2) / (sub_sig ** 2) if sub_sig > 0 else np.nan
        print(f"{labels[i]:>12} {nsub:>8} {pix_sig:>10.3f} {sub_sig:>11.4f} "
              f"{indep:>11.4f} {100*common_frac:>9.1f}%")

    # ---------- Q3: 箱间相关性 ----------
    print("\n" + "=" * 92)
    print("Q3. 误差来源：相机振动（箱间同步）vs 传感器噪声（箱间独立）")
    print("=" * 92)
    S = series[:, keep]
    C = np.corrcoef(S.T)
    off = C[np.triu_indices_from(C, 1)]
    print(f"  箱间相关系数：均值 {np.nanmean(off):+.3f}   "
          f"中位 {np.nanmedian(off):+.3f}   范围 [{np.nanmin(off):+.3f}, {np.nanmax(off):+.3f}]")
    print(f"  最大相关对：")
    kk = keep
    pairs = []
    for a in range(len(kk)):
        for b in range(a + 1, len(kk)):
            pairs.append((C[a, b], labels[kk[a]], labels[kk[b]]))
    pairs.sort(reverse=True)
    for c, l1, l2 in pairs[:4]:
        print(f"    {l1:>12} ↔ {l2:<12} r = {c:+.3f}")
    print()
    if np.nanmean(off) > 0.5:
        print("  → 高度同步：主导误差是**相机/目标的整体运动或系统级共模漂移**，")
        print("     而非各像素独立噪声。方案必须包含相机刚性固定与减振要求。")
    elif np.nanmean(off) > 0.15:
        print("  → 部分同步：共模与独立噪声共存。")
    else:
        print("  → 基本独立：误差以传感器噪声为主。")

    # ---------- 1/sqrt(N) 收益（用真实 ROI 时间序列）----------
    print("\n" + "=" * 92)
    print("时间平均收益（对 MDR 的实际改善）")
    print("=" * 92)
    best = min(mdr_rows, key=lambda r: r[3])
    bi = [labels[i] for i in keep].index(best[0])
    s = series[:, keep[bi]]
    print(f"取最优箱 {best[0]}（中位 {best[1]:.0f} mm）\n")
    print(f"{'平均帧数':>9} {'σ':>10} {'理论 1/√N':>11} {'实际/理论':>10} {'MDR(3σ)':>10}")
    print("-" * 54)
    base = None
    for k in (1, 2, 4, 8, 16, 32, 64, 128):
        nw = len(s) // k
        if nw < 3:
            break
        wm = s[:nw * k].reshape(nw, k).mean(axis=1)
        sig = float(np.nanstd(wm, ddof=1))
        if base is None:
            base = sig
        th = base / np.sqrt(k)
        print(f"{k:>9} {sig:>10.4f} {th:>11.4f} {sig/th:>10.2f} {3*sig:>10.3f}")

    # ---------- 结论 ----------
    print("\n" + "=" * 92)
    print("结论摘要")
    print("=" * 92)
    ok2 = [(l, z, m) for l, z, _, m, _ in mdr_rows if m <= 2]
    ok5 = [(l, z, m) for l, z, _, m, _ in mdr_rows if m <= 5]
    if ok2:
        print(f"  单帧 MDR ≤ 2mm 的距离范围: 最远 {max(z for _, z, _ in ok2):.0f} mm")
        print(f"    {', '.join(f'{l}({z:.0f}mm,{m:.2f}mm)' for l, z, m in ok2)}")
    else:
        print("  无单帧 MDR ≤ 2mm 的距离箱")
    if ok5:
        print(f"  单帧 MDR ≤ 5mm 的距离范围: 最远 {max(z for _, z, _ in ok5):.0f} mm")
    else:
        print("  无单帧 MDR ≤ 5mm 的距离箱")
    print(f"\n  数据文件可另行导出；本次未做真实位移标定。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
