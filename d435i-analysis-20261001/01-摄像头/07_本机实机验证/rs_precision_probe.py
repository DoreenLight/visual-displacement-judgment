#!/usr/bin/env python3
"""D435i 精度探底 —— 实测 Z 轴位移的可达精度及其随距离的变化。

用途
----
为「Z 轴位移测量测试方案」提供实测数据基础，回答五个问题：

  1. 静态目标的深度噪声底是多少（mm）？
  2. 精度随距离如何变化？
  3. 1mm 量化是否成为精度瓶颈？
  4. 多帧 / 多像素平均能改善多少？是否存在上限？
  5. 是否存在随时间漂移的系统误差（平均不掉的误差）？

方法
----
对准静止场景连续采集 N 帧深度。场景内不同距离的表面天然构成多个
「被测目标」，按距离分箱后分别统计各自的时间噪声。

对每个距离箱计算：
  - per-pixel σ : 单像素的时间标准差（单点噪声）
  - ROI-mean σ  : 该箱所有像素逐帧求均值后，均值序列的时间标准差
                  ★ 这就是「测量目标 Z 位移」时的实际分辨率底
  - N_eff       : 等效独立像素数，反映空间平均的真实收益
  - drift       : 前半段均值与后半段均值之差，反映系统漂移

用法
----
sudo -E /opt/anaconda3/envs/realsense/bin/python3 rs_precision_probe.py
sudo -E .../rs_precision_probe.py --w 1280 --h 720 --frames 120
sudo -E .../rs_precision_probe.py --no-emitter      # 关红外投射器对比
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

# 距离分箱（mm）。覆盖 D435i 的有效量程并细分近端。
BINS = [
    (0, 300), (300, 400), (400, 500), (500, 700), (700, 1000),
    (1000, 1500), (1500, 2000), (2000, 3000), (3000, 5000),
    (5000, 8000), (8000, 65535),
]
MIN_PIXELS = 200        # 每箱至少这么多有效像素才统计


def bin_labels():
    out = []
    for lo, hi in BINS:
        if hi >= 65535:
            out.append(f"{lo/1000:.0f}m+")
        elif hi <= 1000:
            out.append(f"{lo}-{hi}mm")
        else:
            out.append(f"{lo/1000:.1f}-{hi/1000:.1f}m")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=848)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frames", type=int, default=150,
                    help="采集帧数（越多统计越稳）")
    ap.add_argument("--warmup", type=int, default=40,
                    help="预热帧数（等自动曝光稳定）")
    ap.add_argument("--no-emitter", action="store_true",
                    help="关闭红外投射器（对比纹理依赖）")
    args = ap.parse_args()

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, args.w, args.h, rs.format.z16, args.fps)

    print(f"深度流 {args.w}x{args.h} @ {args.fps}   目标帧数 {args.frames}")
    profile = pipe.start(cfg)

    dev = profile.get_device()
    ds = dev.first_depth_sensor()

    if args.no_emitter:
        try:
            ds.set_option(rs.option.emitter_enabled, 0)
            print("红外投射器: 已关闭")
        except Exception as e:
            print(f"⚠️  关投射器失败: {e}")
    else:
        try:
            print(f"红外投射器: 开启 (power={ds.get_option(rs.option.laser_power)})")
        except Exception:
            pass

    try:
        print(f"深度单位: {ds.get_depth_scale()}  →  1 单位 = "
              f"{ds.get_depth_scale()*1000:.4f} mm")
    except Exception:
        pass

    print(f"\n预热 {args.warmup} 帧…")
    for _ in range(args.warmup):
        pipe.wait_for_frames(timeout_ms=10000)

    # --- 预采集：确定每个像素的距离归属 ---
    pre = []
    for _ in range(10):
        f = pipe.wait_for_frames(timeout_ms=10000)
        pre.append(np.asanyarray(f.get_depth_frame().get_data()).astype(np.float32))
    med = np.median(np.stack(pre), axis=0)

    h, w = med.shape
    masks = []
    labels = bin_labels()
    for lo, hi in BINS:
        masks.append((med > 0) & (med < 65535) & (med >= lo) & (med < hi))

    print("距离分箱像素分布：")
    for lbl, m in zip(labels, masks):
        if m.sum() > 0:
            center = float(np.median(med[m]))
            print(f"  {lbl:>12} : {int(m.sum()):>7} px   中位 {center:8.1f} mm")

    # --- 主采集：逐帧累计 ---
    print(f"\n采集 {args.frames} 帧…")
    sum_px = np.zeros((h, w), np.float64)
    sq_px = np.zeros((h, w), np.float64)
    cnt_px = np.zeros((h, w), np.int32)
    n_bins = len(BINS)
    series = np.full((args.frames, n_bins), np.nan, np.float64)
    valid_frac = []

    t0 = time.time()
    for t in range(args.frames):
        frames = pipe.wait_for_frames(timeout_ms=10000)
        d = np.asanyarray(frames.get_depth_frame().get_data())
        v = (d > 0) & (d < 65535)
        valid_frac.append(v.mean())
        df = d.astype(np.float64)
        sum_px += np.where(v, df, 0.0)
        sq_px += np.where(v, df * df, 0.0)
        cnt_px += v.astype(np.int32)
        for i, m in enumerate(masks):
            sel = d[m & v]
            if sel.size:
                series[t, i] = sel.mean()
    dur = time.time() - t0
    pipe.stop()

    print(f"完成，耗时 {dur:.1f}s  →  {args.frames/dur:.1f} fps")
    print(f"平均有效率 {100*np.mean(valid_frac):.1f}%")

    # --- 逐像素统计 ---
    ok = cnt_px == args.frames
    mean_px = np.where(cnt_px > 0, sum_px / np.maximum(cnt_px, 1), np.nan)
    var_px = np.where(cnt_px > 1,
                      (sq_px - cnt_px * mean_px ** 2) / np.maximum(cnt_px - 1, 1),
                      np.nan)
    std_px = np.sqrt(np.maximum(var_px, 0))

    print("\n" + "=" * 96)
    print("各距离箱的噪声统计")
    print("=" * 96)
    print(f"{'距离箱':>12} {'像素数':>8} {'中位Z':>9} {'单像素σ':>9} "
          f"{'ROI均值σ':>10} {'N_eff':>8} {'半段漂移':>9} {'合计误差':>9}")
    print(f"{'':>12} {'':>8} {'(mm)':>9} {'(mm)':>9} "
          f"{'(mm)':>10} {'':>8} {'(mm)':>9} {'(mm)':>9}")
    print("-" * 96)

    results = []
    nframes = args.frames
    half = nframes // 2
    for i, (lbl, m) in enumerate(zip(labels, masks)):
        mm = m & ok
        npx = int(mm.sum())
        if npx < MIN_PIXELS:
            continue
        z = float(np.median(mean_px[mm]))
        psig = float(np.median(std_px[mm]))
        s = series[:, i]
        if np.isnan(s).any():
            continue
        rmean = float(np.nanmean(s))
        rsig = float(np.nanstd(s, ddof=1))
        neff = (psig / rsig) ** 2 if rsig > 0 else float("nan")
        drift = abs(float(np.nanmean(s[half:])) - float(np.nanmean(s[:half])))
        total = float(np.hypot(rsig, drift))
        print(f"{lbl:>12} {npx:>8} {z:>9.1f} {psig:>9.2f} "
              f"{rsig:>10.3f} {neff:>8.0f} {drift:>9.3f} {total:>9.3f}")
        results.append(dict(lbl=lbl, npx=npx, z=z, psig=psig, rsig=rsig,
                            neff=neff, drift=drift, total=total,
                            series=s, mean_px=mean_px, std_px=std_px, mask=mm))

    if not results:
        print("\n没有足够像素的距离箱，无法统计。")
        return 1

    # --- 量化分析 ---
    print("\n" + "=" * 96)
    print("量化分析（1mm 步长是否构成瓶颈）")
    print("=" * 96)
    allv = mean_px[ok]
    allv = allv[(allv > 0) & (allv < 65535)]
    is_int = np.isclose(allv, np.round(allv), atol=1e-6)
    print(f"  全部有效像素 {allv.size} 个")
    print(f"  整数值占比   {100*is_int.mean():.2f}%   "
          f"({'✅ 输出为整数毫米' if is_int.mean()>0.999 else '⚠️ 存在非整数'})")
    uniq = np.unique(np.round(allv).astype(np.int32))
    print(f"  不同取值数   {uniq.size}")
    # 空间抖动：同一距离箱内，量化误差是否被空间平均掉
    for r in results[:4]:
        v = np.round(r["mean_px"][r["mask"]]).astype(np.int32)
        frac = np.abs(v - v.mean()) % 1.0
        print(f"  {r['lbl']:>12}: 箱内均值 {v.mean():8.2f} mm, "
              f"超出整数的部分 {frac.mean():.3f}  ← 非零说明空间抖动真实存在")

    # --- 平均律：多帧平均的收益 ---
    print("\n" + "=" * 96)
    print("时间平均收益（检验是否遵循 1/sqrt(N)）")
    print("=" * 96)
    best = min(results, key=lambda r: r["total"])
    s = best["series"]
    print(f"取最优距离箱 {best['lbl']}（中位 {best['z']:.1f} mm，"
          f"单帧 ROI 均值 σ={best['rsig']:.3f} mm）\n")
    print(f"{'窗口帧数':>10} {'窗口σ':>10} {'理论1/√N':>10} {'实际/理论':>10}")
    print("-" * 46)
    base1 = None
    for k in (1, 2, 4, 8, 16, 32, 64):
        if k > len(s):
            break
        nw = len(s) // k
        if nw < 2:
            continue
        wm = s[:nw * k].reshape(nw, k).mean(axis=1)
        wsig = float(np.nanstd(wm, ddof=1))
        if base1 is None:
            base1 = wsig
        theory = base1 / np.sqrt(k)
        ratio = wsig / theory if theory > 0 else float("nan")
        print(f"{k:>10} {wsig:>10.3f} {theory:>10.3f} {ratio:>10.2f}")

    # 漂移检验：最优箱的均值轨迹
    print(f"\n最优箱均值轨迹（每 {max(1,len(s)//10)} 帧取样）：")
    step = max(1, len(s) // 10)
    traj = [f"{np.nanmean(s[j:j+step]):.2f}" for j in range(0, len(s), step)]
    print("  " + "  ".join(traj))
    print(f"  全段极差 {np.nanmax(s)-np.nanmin(s):.3f} mm")

    # --- 结论 ---
    print("\n" + "=" * 96)
    print("结论")
    print("=" * 96)
    for r in results:
        verdict = ("✅ 满足 2-5mm" if r["total"] <= 5
                   else ("⚠️  接近 5mm" if r["total"] <= 8 else "❌ 超出 5mm"))
        print(f"  {r['lbl']:>12}  中位 {r['z']:>7.1f} mm   "
              f"单帧 ROI 精度 {r['total']:>7.3f} mm   {verdict}")
    print(f"\n  参考：rsig 为单帧 ROI 均值噪声；drift 为半段系统漂移；"
          f"total=√(rsig²+drift²)")
    print(f"  单帧 ROI 精度 ≤5mm 的最远距离: "
          f"{max([r['z'] for r in results if r['total']<=5], default=0):.0f} mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
