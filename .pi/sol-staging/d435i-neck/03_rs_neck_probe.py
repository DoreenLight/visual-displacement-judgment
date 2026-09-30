#!/usr/bin/env python3
"""人体颈部/头部 Z 轴位移测量 —— 可行性探测与 ROI 优化。

被测对象特殊性
--------------
与刚性平面不同，人体颈部/头部目标带来五个必须单独处理的约束：

  1. 曲面：颈部近似圆柱面，固定 ROI 内像素深度天然有分布 → 抬高 ROI 均值噪声
  2. 呼吸：颈/锁骨区随呼吸起伏，若测得不是呼吸它就是干扰
  3. 遮挡：头发（黑色丝状 → 深度空洞）、衣领、下颌
  4. 肤色：深色皮肤红外反射率低，汗液镜面反射 → 深度空洞
  5. 不自主运动：姿态抖动是 cm 级，会与 mm 级测量目标混叠

本脚本的作用
------------
不预设「哪里是颈部」。而是把画面按距离分层，逐层报告：
深度有效性、ROI 内自然分布宽度、时间噪声、以及**分离出不同运动分量**。

然后由人来指定真正的颈部 ROI，脚本给出该 ROI 的优化参数与可达精度。

用法
----
# 1. 探索：看画面里有哪些深度层，各自噪声水平
sudo -E .../rs_neck_probe.py --explore

# 2. 优化：在指定 ROI 上测 MDR，并对比不同 ROI 尺寸/平均帧数
sudo -E .../rs_neck_probe.py --roi 380,180,120,120

# 3. 分离运动分量：区分呼吸（高频）与整体移动（低频）
sudo -E .../rs_neck_probe.py --roi 380,180,120,120 --decompose

# 4. 导出时域数据供离线分析
sudo -E .../rs_neck_probe.py --roi 380,180,120,120 --dump /tmp/neck.npz
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
    sys.exit("缺少 pyrealsense2。")


def start(fps=30, temporal_alpha=0.4, lock_exposure=False):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, fps)
    profile = pipe.start(cfg)
    ds = profile.get_device().first_depth_sensor()

    if lock_exposure:
        try:
            ds.set_option(rs.option.enable_auto_exposure, 0)
        except Exception:
            pass

    tf = None
    if temporal_alpha > 0:
        tf = rs.temporal_filter()
        tf.set_option(rs.option.filter_smooth_alpha, temporal_alpha)
        tf.set_option(rs.option.filter_smooth_delta, 20)
    return pipe, profile, ds, tf


def grab(pipe, tf):
    fr = pipe.wait_for_frames(timeout_ms=10000)
    df = fr.get_depth_frame()
    if tf is not None:
        df = tf.process(df)
    return np.asanyarray(df.get_data())


# ---------------------------------------------------------------- explore
def explore(pipe, tf, ds, nframes=150, warmup=45):
    print("=" * 92)
    print("探索模式：先看画面里的深度层次（不做任何 ROI 假设）")
    print("=" * 92)

    for _ in range(warmup):
        pipe.wait_for_frames(timeout_ms=10000)

    stack = []
    for _ in range(30):
        stack.append(grab(pipe, tf).astype(np.float32))
    st = np.stack(stack)
    med = np.median(st, axis=0)
    v = (med > 0) & (med < 65535)

    print(f"\n深度有效性: {100*v.mean():.1f}%  "
          f"({int(v.sum())}/{med.size} 像素有读数)")
    if v.sum() == 0:
        print("❌ 完全没有有效深度。检查相机对准、距离、遮挡。")
        return

    vv = med[v]
    print(f"有效距离分布: p5={np.percentile(vv,5):.0f}  "
          f"p25={np.percentile(vv,25):.0f}  p50={np.percentile(vv,50):.0f}  "
          f"p75={np.percentile(vv,75):.0f}  p95={np.percentile(vv,95):.0f} mm")

    # 前景检测：画面中最近的那一簇像素通常就是被测者
    p10 = np.percentile(vv, 10)
    near = v & (med < p10 * 1.6)
    print(f"\n前景候选（最近的 10% 深度 + 60% 容差）: {int(near.sum())} 像素 "
          f"({100*near.sum()/v.sum():.1f}% 的有效像素)")
    if near.sum() > 100:
        ys, xs = np.where(near)
        print(f"  包围盒: x=[{xs.min()},{xs.max()}] y=[{ys.min()},{ys.max()}]  "
              f"中心=({int(xs.mean())},{int(ys.mean())})  "
              f"质心深度={np.median(med[near]):.0f}mm")

        # 分块给出前景不同区域的有效率（用于避开头发/衣领）
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        print(f"\n  前景分区有效率（4×4 网格，用于找深度最稳的区域）:")
        print(f"  {'':>6}" + "".join(f"{'col'+str(c):>10}" for c in range(4)))
        for r in range(4):
            row = []
            for c in range(4):
                sy = slice(y0 + r*(y1-y0+1)//4, y0 + (r+1)*(y1-y0+1)//4)
                sx = slice(x0 + c*(x1-x0+1)//4, x0 + (c+1)*(x1-x0+1)//4)
                blk = med[sy, sx]
                bv = (blk > 0) & (blk < 65535)
                row.append(100*bv.mean() if bv.size else 0)
            print(f"  {'row'+str(r):>6}" + "".join(f"{x:>9.1f}%" for x in row))

    # 时间噪声：按距离层分组
    print("\n" + "=" * 92)
    print("各距离层的时间噪声（30 帧窗口，检验哪一层最稳）")
    print("=" * 92)
    series = np.full((nframes, 0), np.nan)
    layers = []
    edges = [0, 200, 300, 400, 500, 700, 1000, 1500, 3000, 100000]
    masks = [v & (med >= edges[i]) & (med < edges[i+1])
             for i in range(len(edges)-1)]
    ser = np.full((nframes, len(masks)), np.nan)
    for t in range(nframes):
        d = grab(pipe, tf)
        for i, m in enumerate(masks):
            sel = d[m & (d > 0) & (d < 65535)]
            if sel.size >= 200:
                ser[t, i] = sel.mean()

    print(f"{'层':>14} {'像素':>8} {'中位Z':>9} {'σ':>9} {'MDR(3σ)':>10} {'有效率':>9}")
    print("-" * 92)
    for i in range(len(masks)):
        m = masks[i]
        s = ser[:, i]
        if m.sum() < 200 or not np.isfinite(s).all():
            continue
        sig = float(np.std(np.diff(s), ddof=1)) / np.sqrt(2)
        lbl = f"{edges[i]}-{edges[i+1]}" if edges[i+1] < 100000 else f"{edges[i]}+"
        seld = med[m]
        print(f"{lbl:>14} {int(m.sum()):>8} {np.median(seld):>9.0f} "
              f"{sig:>9.3f} {3*sig:>10.3f} "
              f"{100*np.mean(s>0):>8.0f}%")
    print("\n提示：用 --roi x,y,w,h 在上面的包围盒内指定真正的颈部区域。")


# ------------------------------------------------------------------- ROI
def run_roi(pipe, tf, ds, roi, nframes, fps, warmup, decompose, dump):
    x, y, w, h = roi
    print("=" * 92)
    print(f"ROI 模式: x={x} y={y} w={w} h={h}   (画面 848x480)")
    print("=" * 92)

    for _ in range(warmup):
        pipe.wait_for_frames(timeout_ms=10000)

    # 固定掩膜 = ROI ∩ 有效深度
    stack = [grab(pipe, tf).astype(np.float32) for _ in range(15)]
    med = np.median(np.stack(stack), axis=0)
    roi_mask = np.zeros(med.shape, bool)
    roi_mask[y:y+h, x:x+w] = True
    valid_mask = roi_mask & (med > 0) & (med < 65535)
    nvalid = int(valid_mask.sum())
    print(f"\nROI 内有效像素: {nvalid}/{w*h} ({100*nvalid/(w*h):.1f}%)")
    if nvalid < 50:
        print("❌ ROI 内有效像素太少。避开头发/衣领/空洞区域，或调整 ROI。")
        return
    zr = med[valid_mask]
    print(f"ROI 深度: 中位 {np.median(zr):.0f}mm  "
          f"p5={np.percentile(zr,5):.0f}  p95={np.percentile(zr,95):.0f}  "
          f"空间分布宽度(σ)={np.std(zr):.2f}mm   ← 曲面导致的自然宽度")

    # 时间序列
    ser_mean = np.full(nframes, np.nan)
    ser_med = np.full(nframes, np.nan)
    ser_p25 = np.full(nframes, np.nan)
    ser_p75 = np.full(nframes, np.nan)
    valid_frac = np.full(nframes, np.nan)
    frames_store = np.full((nframes, h, w), np.nan, np.float32)

    t0 = time.time()
    for t in range(nframes):
        d = grab(pipe, tf)
        sub = d[y:y+h, x:x+w].astype(np.float32)
        m = (d[valid_mask.argwhere().T[0], valid_mask.argwhere().T[1]]
             if False else None)
        vsub = (sub > 0) & (sub < 65535)
        valid_frac[t] = float(vsub.mean())
        sel = sub[vsub]
        if sel.size >= 20:
            ser_mean[t] = sel.mean()
            ser_med[t] = np.median(sel)
            ser_p25[t] = np.percentile(sel, 25)
            ser_p75[t] = np.percentile(sel, 75)
        frames_store[t] = np.where(vsub, sub, np.nan)
    dur = time.time() - t0
    print(f"采集 {nframes} 帧，耗时 {dur:.1f}s → {nframes/dur:.1f} fps")
    print(f"ROI 有效率: 均值 {100*np.nanmean(valid_frac):.1f}%  "
          f"最低 {100*np.nanmin(valid_frac):.1f}%")

    # --- 核心指标 ---
    print("\n" + "=" * 92)
    print("核心指标")
    print("=" * 92)

    def metrics(s, name):
        if not np.isfinite(s).all():
            print(f"  {name}: 数据不完整，跳过")
            return None
        sig = float(np.std(np.diff(s), ddof=1)) / np.sqrt(2)
        mdr = 3 * sig
        # 漂移
        nw = max(1, nframes // fps)
        wm = s[:nw*fps].reshape(nw, fps).mean(axis=1) if nw >= 2 else s
        drift = float(wm.max() - wm.min())
        print(f"  {name:16} σ={sig:7.3f}mm  MDR(3σ)={mdr:8.3f}mm  "
              f"漂移={drift:7.3f}mm  抖动范围={np.ptp(s):7.3f}mm")
        return sig, mdr, drift

    r_mean = metrics(ser_mean, "ROI 均值")
    r_med = metrics(ser_med, "ROI 中位数")
    r_iqr = metrics(ser_p75 - ser_p25, "ROI IQR 宽度")

    # --- 不同 ROI 尺寸的对比 ---
    print("\n" + "=" * 92)
    print("ROI 尺寸的影响（同中心，不同尺寸 → 找最小够用的 ROI）")
    print("=" * 92)
    print(f"{'尺寸':>10} {'有效px':>8} {'ROI内σ':>9} {'时间σ':>9} {'MDR(3σ)':>10}")
    print("-" * 92)
    cx, cy = x + w // 2, y + h // 2
    for frac in (0.25, 0.5, 0.75, 1.0):
        ww, hh = max(8, int(w * frac)), max(8, int(h * frac))
        xx, yy = max(0, cx - ww//2), max(0, cy - hh//2)
        ssz = frames_store[:, yy:yy+hh, xx:xx+ww]
        vv = np.isfinite(ssz)
        cnt = vv.sum(axis=(1, 2))
        good = cnt >= max(10, ww*hh//10)
        if good.sum() < 10:
            continue
        ss = np.array([np.nanmean(ssz[i]) for i in np.where(good)[0]])
        if len(ss) < 10:
            continue
        sig = float(np.std(np.diff(ss), ddof=1)) / np.sqrt(2)
        inner = ssz[good]
        roi_sd = float(np.nanstd(np.nanmean(inner, axis=0)))
        print(f"{ww}x{hh:<6} {int(cnt.mean()):>8} {roi_sd:>9.2f} "
              f"{sig:>9.3f} {3*sig:>10.3f}")

    # --- 运动分量分离 ---
    if decompose and np.isfinite(ser_mean).all():
        print("\n" + "=" * 92)
        print("运动分量分离（呼吸 vs 整体移动）")
        print("=" * 92)
        s = ser_mean - np.nanmean(ser_mean)
        # 简单 FFT 找主要周期
        n = len(s)
        sp = np.abs(np.fft.rfft(s * np.hanning(n)))
        fr_axis = np.fft.rfftfreq(n, d=1.0/fps)
        # 只看 0.05-2 Hz（呼吸 0.1-0.5Hz，姿态抖动更慢）
        band = (fr_axis >= 0.05) & (fr_axis <= 2.0)
        if band.sum() > 2:
            pk = np.argmax(sp[band])
            fpk = fr_axis[band][pk]
            print(f"  主要频率成分: {fpk:.3f} Hz "
                  f"(周期 {1/fpk:.1f}s)" if fpk > 0 else "  主要频率: DC")
            if 0.1 <= fpk <= 0.6:
                print(f"  → 落在呼吸频段 (0.1-0.6Hz)")
            elif fpk < 0.1:
                print(f"  → 低频，更像整体漂移/姿态缓变")
            else:
                print(f"  → 高频，可能是抖动或噪声")
            print(f"  幅度(峰峰值): {np.ptp(ser_mean):.2f} mm")
            print(f"  该频段能量占比: "
                  f"{100*sp[band][pk]**2/np.sum(sp[band]**2):.1f}%")
        else:
            print("  数据太短，无法做频谱分析（建议 --frames 600 以上）")

    # --- 平均帧数收益 ---
    if np.isfinite(ser_mean).all():
        print("\n" + "=" * 92)
        print("时间平均收益（离线分析可用的杠杆）")
        print("=" * 92)
        print(f"{'平均帧数':>9} {'σ':>10} {'理论1/√N':>11} {'实际/理论':>10} "
              f"{'MDR(3σ)':>10} {'延迟':>8}")
        print("-" * 70)
        base = None
        for k in (1, 2, 4, 8, 16, 32, 64):
            nw = nframes // k
            if nw < 4:
                break
            wm = ser_mean[:nw*k].reshape(nw, k).mean(axis=1)
            sig = float(np.std(wm, ddof=1))
            if base is None:
                base = float(np.std(np.diff(ser_mean, ddof=1))) / np.sqrt(2)
            th = base / np.sqrt(k)
            print(f"{k:>9} {sig:>10.3f} {th:>11.3f} {sig/th:>10.2f} "
                  f"{3*sig:>10.3f} {k/fps*1000:>7.0f}ms")

    # --- 导出 ---
    if dump and np.isfinite(ser_mean).all():
        np.savez_compressed(
            dump,
            roi=np.array(roi),
            t=np.arange(nframes) / fps,
            roi_mean_z=ser_mean,
            roi_median_z=ser_med,
            roi_p25=ser_p25,
            roi_p75=ser_p75,
            valid_frac=valid_frac,
            fps=fps,
        )
        print(f"\n✅ 时域数据已导出: {dump}")
        print("   含 roi_mean_z / roi_median_z / valid_frac / 时间轴")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explore", action="store_true", help="探索模式")
    ap.add_argument("--roi", default=None, help="x,y,w,h")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=45)
    ap.add_argument("--alpha", type=float, default=0.4,
                    help="temporal 滤波 alpha；0 = 关闭")
    ap.add_argument("--lock-exposure", action="store_true")
    ap.add_argument("--decompose", action="store_true", help="分离呼吸/整体运动")
    ap.add_argument("--dump", default=None, help="导出 npz 路径")
    args = ap.parse_args()

    if not args.explore and not args.roi:
        ap.error("需要 --explore 或 --roi x,y,w,h")

    pipe, profile, ds, tf = start(args.fps, args.alpha, args.lock_exposure)
    try:
        print(f"深度 848x480@{args.fps}  temporal={'on α='+str(args.alpha) if tf else 'off'}  "
              f"曝光={'锁定' if args.lock_exposure else '自动'}")
        if args.explore:
            explore(pipe, tf, ds, args.frames, args.warmup)
        else:
            roi = tuple(int(v) for v in args.roi.split(","))
            if len(roi) != 4:
                ap.error("--roi 需要 4 个整数: x,y,w,h")
            run_roi(pipe, tf, ds, roi, args.frames, args.fps, args.warmup,
                    args.decompose, args.dump)
    finally:
        try:
            pipe.stop()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
