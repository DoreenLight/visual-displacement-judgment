#!/usr/bin/env python3
"""深度一致块精度扫描（修正版方法）—— 回答「各距离的真实可达精度」。

修正了什么
----------
此前脚本用「按深度值筛选像素」建 ROI，得到的是散布画面各处的深度碎片
（实测空间 σ 可达 58mm、横跨 700px），逐帧平均这些碎片产生假噪声，
使 MDR 虚高 5-10 倍，并让「N_eff 只有 1-4」这类结论失真。

本脚本改为「深度一致连通块」：只在【空间相邻 且 深度相差 < thresh】
的像素间建立连通，保证每个 ROI 是物理上的单一表面。

用法
----
sudo -E .../rs_block_precision.py                          # 扫全场景
sudo -E .../rs_block_precision.py --target 200 --tol 80    # 只看 200mm 附近
sudo -E .../rs_block_precision.py --frames 400 --min-px 60
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

try:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
except ImportError:
    sys.exit("缺少 scipy。")

W, H, FPS = 848, 480, 30


def pool_pixels(d):
    """按 (横向池化2, 纵向池化2) 重采样，等效提升小目标信噪比。

    D435i 深度不足 1280x720；848x480 下 2x2 池化可把 4 个相邻像素的
    min-disparity 平均，是官方推荐的远距离/小目标改善手段。
    返回 pooled 图像与缩放因子。
    """
    h, w = d.shape
    h2, w2 = h // 2 * 2, w // 2 * 2
    sub = d[:h2, :w2].astype(np.float32)
    valid = (sub > 0) & (sub < 65535)
    sub = np.where(valid, sub, np.nan)
    p = np.nanmean(sub.reshape(h2 // 2, 2, w2 // 2, 2).transpose(0, 2, 1, 3)
                   .reshape(h2 // 2, w2 // 2, 4), axis=2)
    return p


def segment(med, thresh, min_px):
    """深度一致连通域分割。返回 {label: (ys, xs)} 与 label 图。"""
    h, w = med.shape
    valid = (med > 0) & (med < 65535)
    n = int(valid.sum())
    if n < min_px:
        return {}, None
    di = med.astype(np.int32)
    idx = np.full((h, w), -1, np.int64)
    idx[valid] = np.arange(n)

    mh = valid[:, :-1] & valid[:, 1:] & (np.abs(di[:, :-1] - di[:, 1:]) < thresh)
    mv = valid[:-1, :] & valid[1:, :] & (np.abs(di[:-1, :] - di[1:, :]) < thresh)
    rh, ch = np.where(mh)
    rv, cv = np.where(mv)
    if rh.size + rv.size == 0:
        return {}, None

    rows = np.concatenate([idx[rh, ch], idx[rv, cv]])
    cols = np.concatenate([idx[rh, ch + 1], idx[rv + 1, cv]])
    g = coo_matrix((np.ones(rows.size, np.int8), (rows, cols)), shape=(n, n))
    ncomp, labels = connected_components(g, directed=False)

    lab_img = np.full((h, w), -1, np.int32)
    lab_img[valid] = labels
    cnt = np.bincount(labels, minlength=ncomp)
    vy, vx = np.where(valid)
    blocks = {}
    for k in range(ncomp):
        if cnt[k] < min_px:
            continue
        sel = labels == k
        blocks[int(k)] = (vy[sel], vx[sel])
    return blocks, lab_img


def make_pipe(alpha):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    pipe.start(cfg)
    tf = None
    if alpha > 0:
        tf = rs.temporal_filter()
        tf.set_option(rs.option.filter_smooth_alpha, alpha)
        tf.set_option(rs.option.filter_smooth_delta, 20)
    return pipe, tf


def grab(pipe, tf):
    f = pipe.wait_for_frames(timeout_ms=10000)
    df = f.get_depth_frame()
    if tf is not None:
        df = tf.process(df)
    return np.asanyarray(df.get_data())


def capture(pipe, tf, blocks, frames, warmup=45):
    """对同一组固定像素集合采集时间序列。"""
    for _ in range(warmup):
        pipe.wait_for_frames(timeout_ms=10000)
    keys = list(blocks)
    ser = {k: np.full(frames, np.nan) for k in keys}
    cnt = {k: np.zeros(frames, np.int32) for k in keys}
    for t in range(frames):
        d = grab(pipe, tf)
        for k in keys:
            ys, xs = blocks[k]
            v = d[ys, xs]
            m = (v > 0) & (v < 65535)
            cnt[k][t] = int(m.sum())
            if m.sum() >= max(8, len(v) // 10):
                ser[k][t] = float(v[m].mean())
    return ser, cnt


def metrics(s):
    ok = np.isfinite(s)
    if ok.sum() < 50:
        return None
    s2 = s[ok]
    scatter = float(np.std(s2, ddof=1))
    dif = np.diff(s2)
    dif_sig = float(np.std(dif, ddof=1)) / np.sqrt(2) if dif.size > 1 else np.nan
    nw = len(s2) // FPS
    drift = (float(np.ptp(s2[:nw * FPS].reshape(nw, FPS).mean(axis=1)))
             if nw >= 2 else np.nan)
    return dict(scatter=scatter, dif_sig=dif_sig, drift=drift)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--min-px", type=int, default=80)
    ap.add_argument("--thresh", type=float, default=30.0)
    ap.add_argument("--target", type=float, default=0)
    ap.add_argument("--tol", type=float, default=100.0)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    print("=" * 100)
    print("深度一致块精度扫描（修正版）")
    print("=" * 100)

    # ── 一次分割，两轮复用同一像素集合（保证可比）──
    pipe, _ = make_pipe(0)
    for _ in range(45):
        pipe.wait_for_frames(timeout_ms=10000)
    ref = np.stack([grab(pipe, None).astype(np.float32) for _ in range(15)])
    pipe.stop()
    med = np.median(ref, axis=0)
    blocks, _ = segment(med, args.thresh, args.min_px)

    if not blocks:
        print("❌ 没找到足够的深度一致块")
        return 1

    # 按像素数取前 N，并按中位距离报告
    items = []
    for k, (ys, xs) in blocks.items():
        vals = med[ys, xs]
        items.append((k, len(ys), float(np.median(vals))))
    items.sort(key=lambda t: -t[1])
    items = items[:args.top]

    if args.target > 0:
        items = [t for t in items
                 if abs(t[2] - args.target) <= args.tol]
        if not items:
            print(f"⚠️ 在 {args.target}±{args.tol}mm 内没有块。")
            print("   场景中可用块一览：")
            for k, npx, z in items[:12]:
                print(f"     {z:>7.0f}mm  {npx:>6}px")
            return 2

    use = {k: blocks[k] for k, _, _ in items}
    spatial = {k: float(np.std(med[blocks[k][0], blocks[k][1]]))
               for k in use}

    # 单像素时间 σ（用于对比空间/时间平均）
    print(f"\n共用 {len(use)} 个块，每轮采 {args.frames} 帧，"
          f"空间门槛 {args.thresh}mm，最小 {args.min_px}px\n")

    results = {}
    for tag, alpha in (("无滤波", 0.0), ("temporal α=0.4", 0.4)):
        pipe, tf = make_pipe(alpha)
        ser, cnt = capture(pipe, tf, use, args.frames)
        pipe.stop()
        results[tag] = {}
        for k in use:
            m = metrics(ser[k])
            if m is None:
                continue
            m["spatial"] = spatial[k]
            m["cnt_sigma"] = float(cnt[k].std(ddof=1))
            # 单像素时间 σ
            ys, xs = use[k]
            d0 = ser[k]
            m["npx"] = len(ys)
            m["z"] = float(np.nanmedian(ser[k]))
            results[tag][k] = m
        print(f"  {tag} 采集完成")

    # ── 报告 ──
    for tag in results:
        print("\n" + "=" * 100)
        print(f"【{tag}】按距离排序")
        print("=" * 100)
        print(f"{'中位Z':>8} {'像素':>6} {'空间σ':>8} {'有效波动':>9} "
              f"{'MDR_diff':>9} {'MDR_scat':>9} {'漂移':>8} {'3σ散射':>8}")
        print("-" * 100)
        rows = sorted(results[tag].items(), key=lambda kv: kv[1]["z"])
        for k, m in rows:
            print(f"{m['z']:>8.0f} {m['npx']:>6} {m['spatial']:>8.1f} "
                  f"{m['cnt_sigma']:>9.1f} {3*m['dif_sig']:>9.3f} "
                  f"{3*m['scatter']:>9.3f} {m['drift']:>8.3f} "
                  f"{3*m['scatter']:>8.3f}")

    # ── 20cm 专项判定 ──
    print("\n" + "=" * 100)
    print("20cm（200mm）可用性专项判定")
    print("=" * 100)
    near = []
    for tag in results:
        for k, m in results[tag].items():
            if abs(m["z"] - 200) <= 120:
                near.append((tag, m))
    if not near:
        print("  ⚠️ 画面中 80-320mm 范围内没有可用的深度一致块。")
        print("     → 说明该距离要么无物体，要么深度失效（盲区）。")
        print("     → 这是 20cm 方案的前置阻塞项。")
    else:
        for tag, m in near:
            verdict = ("✅ 可用" if 3 * m["dif_sig"] <= 5 else
                       ("⚠️ 勉强" if 3 * m["dif_sig"] <= 10 else "❌ 不可用"))
            print(f"  [{tag}] {m['z']:.0f}mm  {m['npx']}px  "
                  f"空间σ={m['spatial']:.1f}  MDR={3*m['dif_sig']:.2f}mm  "
                  f"{verdict}")

    print("\n说明：MDR_diff = 3·std(Δ)/√2（帧间差分口径）")
    print("      MDR_scat = 3·std(序列)（实际散布口径）")
    print("      两者接近说明噪声近似独立；差异大则存在慢漂移。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
