#!/usr/bin/env python3
"""细粒度分辨力曲线 + 曝光锁定验证。

背景
----
用户需求：测量结构件 Z 轴形变，幅值 ±50mm 以上，工作距离 300-500mm，
验收标准为「分辨力/重复性」。这产生一个关键问题：

    目标沿 Z 轴移动 ±Δ 时，相机到目标的实际距离在 [D-Δ, D+Δ] 区间变化。
    而分辨力 MDR 是距离的函数（实测 300mm 处 1.38mm，500mm 处 3.75mm）。
    因此必须知道：在哪个距离区间内能保证 MDR ≤ 目标值。

本脚本回答
----------
Q1. 细化到 25mm 粒度的 MDR-距离曲线（找到 MDR ≤ 1/2/3/5mm 的距离边界）
Q2. 表面倾角对 MDR 的影响（分离「距离效应」与「倾角效应」）
Q3. 锁定曝光能否消除之前观察到的 16 倍异常（风险 R5）
Q4. 在选定工作区间内的重复性是否稳定

方法
----
利用场景中同一材质表面（桌面）在不同距离的连续区域，
按 25mm 粒度分箱，分别统计各箱的时间噪声与空间梯度。

用法
----
sudo -E /opt/anaconda3/envs/realsense/bin/python3 rs_resolution_curve.py
sudo -E .../rs_resolution_curve.py --rounds 3        # 更稳的重复性统计
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

# 25mm 粒度分箱，聚焦 250-600mm；之后粗粒度补全覆盖
BINS = [(250 + 25 * i, 275 + 25 * i) for i in range(14)]   # 250..600
BINS += [(600, 700), (700, 900), (900, 1300)]

MIN_PX = 150


def label(lo, hi):
    return f"{lo}-{hi}"


def setup_pipe(fps=30, temporal=True):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, fps)
    profile = pipe.start(cfg)
    return pipe, profile


def get_masks(pipe):
    """用中值帧确定各距离箱的掩膜。"""
    stack = []
    for _ in range(10):
        f = pipe.wait_for_frames(timeout_ms=10000)
        stack.append(np.asanyarray(f.get_depth_frame().get_data()))
    med = np.median(np.stack(stack).astype(np.float32), axis=0)
    masks = [((med > 0) & (med < 65535) & (med >= lo) & (med < hi))
             for lo, hi in BINS]
    return med, masks


def measure(pipe, masks, nframes, temporal=True, warmup=45):
    """采集并返回逐箱时间序列 + 逐像素统计。"""
    tf = None
    if temporal:
        tf = rs.temporal_filter()
        tf.set_option(rs.option.filter_smooth_alpha, 0.4)
        tf.set_option(rs.option.filter_smooth_delta, 20)

    for _ in range(warmup):
        pipe.wait_for_frames(timeout_ms=10000)

    nb = len(BINS)
    series = np.full((nframes, nb), np.nan)
    h, w = 480, 848
    sum_px = np.zeros((h, w), np.float64)
    cnt_px = np.zeros((h, w), np.int32)
    valid_frac = []

    for t in range(nframes):
        fr = pipe.wait_for_frames(timeout_ms=10000)
        df = fr.get_depth_frame()
        if tf is not None:
            df = tf.process(df)
        d = np.asanyarray(df.get_data())
        v = (d > 0) & (d < 65535)
        valid_frac.append(float(v.mean()))
        sum_px += np.where(v, d.astype(np.float64), 0.0)
        cnt_px += v.astype(np.int32)
        for i, m in enumerate(masks):
            sel = d[m & v]
            if sel.size >= MIN_PX:
                series[t, i] = sel.mean()

    mean_px = np.where(cnt_px > 0, sum_px / np.maximum(cnt_px, 1), np.nan)
    return series, mean_px, float(np.mean(valid_frac))


def analyze(series, mean_px, med, masks, nframes, fps=30):
    """逐箱计算 MDR、倾角代理、有效率。"""
    rows = []
    for i, (lo, hi) in enumerate(BINS):
        m = masks[i]
        npx = int(m.sum())
        if npx < MIN_PX:
            continue
        s = series[:, i]
        if not np.isfinite(s).all():
            continue
        z = float(np.median(med[m]))

        # 帧间差分 → 单帧噪声（差分含两帧噪声，除以 sqrt2）
        dif = np.diff(s)
        single = float(np.std(dif, ddof=1)) / np.sqrt(2)
        mdr = 3.0 * single

        # 空间梯度 → 倾角代理（mm per pixel）
        d_px = mean_px.copy()
        d_px[~np.isfinite(d_px)] = 0
        gy, gx = np.gradient(d_px)
        gmag = np.sqrt(gx ** 2 + gy ** 2)
        grad = float(np.median(gmag[m]))

        # 倾角（度）：tan θ = (mm/px) / (mm/px 的水平像素尺寸)
        px_mm = z / 428.0            # fx=428，水平像素的物理尺寸
        tilt_deg = float(np.degrees(np.arctan(grad / px_mm))) if px_mm > 0 else np.nan

        # ROI 内空间标准差（表面起伏）
        roi_std = float(np.nanstd(mean_px[m]))

        # 漂移：1 秒窗口均值的极差
        nw = nframes // fps
        if nw >= 2:
            wm = s[:nw * fps].reshape(nw, fps).mean(axis=1)
            drift = float(wm.max() - wm.min())
        else:
            drift = np.nan

        rows.append(dict(lo=lo, hi=hi, z=z, npx=npx, sigma=single, mdr=mdr,
                         grad=grad, tilt=tilt_deg, roi_std=roi_std,
                         drift=drift, series=s))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    print("=" * 100)
    print("细粒度分辨力曲线 + 曝光锁定验证")
    print("=" * 100)

    # ---------- 阶段 1：自动曝光 ----------
    print("\n【阶段 1】自动曝光模式")
    pipe, profile = setup_pipe(args.fps)
    ds = profile.get_device().first_depth_sensor()
    auto_exp = None
    try:
        auto_exp = ds.get_option(rs.option.enable_auto_exposure)
        print(f"  自动曝光 = {auto_exp}")
        print(f"  当前曝光 = {ds.get_option(rs.option.exposure):.0f} μs")
        print(f"  当前增益 = {ds.get_option(rs.option.gain):.0f}")
    except Exception as e:
        print(f"  读取曝光参数失败: {e}")

    med, masks = get_masks(pipe)
    print(f"  有效分箱数: {sum(1 for m in masks if m.sum() >= MIN_PX)}")

    auto_rows = []
    for r in range(args.rounds):
        s, mp, vf = measure(pipe, masks, args.frames, temporal=True)
        rows = analyze(s, mp, med, masks, args.frames, args.fps)
        auto_rows.append(rows)
        key = next((x for x in rows if x["lo"] == 300), None)
        if key:
            print(f"  轮 {r+1}: 有效率 {100*vf:.1f}%  "
                  f"300-325mm MDR={key['mdr']:.3f}mm  "
                  f"曝光={ds.get_option(rs.option.exposure):.0f}")
    pipe.stop()

    # ---------- 锁定曝光 ----------
    print("\n【阶段 2】锁定曝光模式")
    pipe, profile = setup_pipe(args.fps)
    ds = profile.get_device().first_depth_sensor()
    try:
        cur_exp = ds.get_option(rs.option.exposure)
        cur_gain = ds.get_option(rs.option.gain)
        ds.set_option(rs.option.enable_auto_exposure, 0)
        time.sleep(0.5)
        locked_exp = ds.get_option(rs.option.exposure)
        locked_gain = ds.get_option(rs.option.gain)
        print(f"  已锁定: 曝光={locked_exp:.0f} μs (原 {cur_exp:.0f}), "
              f"增益={locked_gain:.0f} (原 {cur_gain:.0f})")
    except Exception as e:
        print(f"  锁定失败: {e}")

    medL, masksL = get_masks(pipe)
    locked_rows = []
    for r in range(args.rounds):
        s, mp, vf = measure(pipe, masksL, args.frames, temporal=True)
        rows = analyze(s, mp, medL, masksL, args.frames, args.fps)
        locked_rows.append(rows)
        key = next((x for x in rows if x["lo"] == 300), None)
        if key:
            print(f"  轮 {r+1}: 有效率 {100*vf:.1f}%  "
                  f"300-325mm MDR={key['mdr']:.3f}mm")
    pipe.stop()

    # ---------- 汇总对比 ----------
    print("\n" + "=" * 100)
    print("Q3. 自动曝光 vs 锁定曝光（检验 16 倍异常是否由曝光引起）")
    print("=" * 100)

    def summarize(rounds, name):
        vals = []
        for rows in rounds:
            for x in rows:
                if x["lo"] == 300:
                    vals.append(x["mdr"])
        if len(vals) >= 2:
            a = np.array(vals)
            print(f"  {name:10} 300-325mm MDR: 均值 {a.mean():7.3f}  "
                  f"σ {a.std(ddof=1):6.3f}  范围 [{a.min():.3f}, {a.max():.3f}]  "
                  f"波动 {a.max()/max(a.min(),1e-9):5.1f}×  (n={len(a)})")
        elif vals:
            print(f"  {name:10} 300-325mm MDR: {vals[0]:.3f}  (n=1)")

    summarize(auto_rows, "自动曝光")
    summarize(locked_rows, "锁定曝光")

    # ---------- Q1: 细粒度曲线 ----------
    print("\n" + "=" * 100)
    print("Q1. 细粒度 MDR-距离曲线（锁定曝光，取各箱轮间中位数）")
    print("=" * 100)
    print(f"{'距离箱':>12} {'像素':>7} {'中位Z':>8} {'σ':>8} {'MDR(3σ)':>9} "
          f"{'倾角°':>7} {'ROI内σ':>8} {'漂移':>7} {'判级':>6}")
    print("-" * 100)

    merged = {}
    for rows in locked_rows:
        for x in rows:
            merged.setdefault((x["lo"], x["hi"]), []).append(x)

    curve = []
    for (lo, hi), xs in sorted(merged.items()):
        z = float(np.median([x["z"] for x in xs]))
        npx = int(np.median([x["npx"] for x in xs]))
        sig = float(np.median([x["sigma"] for x in xs]))
        mdr = float(np.median([x["mdr"] for x in xs]))
        tilt = float(np.median([x["tilt"] for x in xs]))
        roi_std = float(np.median([x["roi_std"] for x in xs]))
        drift = float(np.median([x["drift"] for x in xs]))
        grade = "1mm" if mdr <= 1 else ("2mm" if mdr <= 2 else
                                        ("3mm" if mdr <= 3 else
                                         ("5mm" if mdr <= 5 else ">5mm")))
        print(f"{label(lo,hi):>12} {npx:>7} {z:>8.0f} {sig:>8.3f} {mdr:>9.3f} "
              f"{tilt:>7.1f} {roi_std:>8.2f} {drift:>7.3f} {grade:>6}")
        curve.append(dict(lo=lo, hi=hi, z=z, mdr=mdr, tilt=tilt,
                          roi_std=roi_std, npx=npx))

    # ---------- 边界推断 ----------
    print("\n" + "=" * 100)
    print("距离边界推断（用于确定 Z 轴可用跨度）")
    print("=" * 100)
    for thr in (1, 2, 3, 5):
        ok = [c for c in curve if c["mdr"] <= thr]
        if ok:
            zmin = min(c["z"] for c in ok)
            zmax = max(c["z"] for c in ok)
            # 只取连续区间
            print(f"  MDR ≤ {thr}mm : {zmin:.0f}-{zmax:.0f}mm   "
                  f"跨度 {zmax-zmin:.0f}mm  (可用 ±{(zmax-zmin)/2:.0f}mm)")
        else:
            print(f"  MDR ≤ {thr}mm : 无满足的箱")

    # ---------- 倾角效应 ----------
    print("\n" + "=" * 100)
    print("Q2. 分离距离效应与倾角效应")
    print("=" * 100)
    good = [c for c in curve if c["npx"] >= MIN_PX]
    if len(good) >= 3:
        zs = np.array([c["z"] for c in good])
        ms = np.array([c["mdr"] for c in good])
        ts = np.array([c["tilt"] for c in good])
        if len(zs) >= 3:
            # MDR 与距离、倾角的相关系数
            cz = np.corrcoef(zs, ms)[0, 1] if zs.std() > 0 else np.nan
            ct = np.corrcoef(ts, ms)[0, 1] if ts.std() > 0 else np.nan
            print(f"  MDR ↔ 距离   r = {cz:+.3f}")
            print(f"  MDR ↔ 倾角   r = {ct:+.3f}")
            print()
            if abs(cz) > abs(ct):
                print("  → 距离是主导因素，倾角影响次之。")
                print("    方案应以「控制距离跨度」为主要手段。")
            else:
                print("  → 倾角影响不亚于距离。")
                print("    方案需同时约束目标表面法线方向（尽量正对相机）。")
        # 拟合 MDR ~ a*z + b
        A = np.vstack([zs, np.ones_like(zs)]).T
        coef, *_ = np.linalg.lstsq(A, ms, rcond=None)
        print(f"\n  线性拟合 MDR ≈ {coef[0]*1000:.4f} mm/μm·z + {coef[1]:.3f}")
        print(f"    即距离每增加 100mm，MDR 增加约 {coef[0]*100:.3f} mm")

    # ---------- 结论 ----------
    print("\n" + "=" * 100)
    print("结论：对 ±50mm 以上形变测量的可用跨度")
    print("=" * 100)
    best = [c for c in curve if c["mdr"] <= 2]
    if best:
        zlo = min(c["z"] for c in best)
        zhi = max(c["z"] for c in best)
        print(f"  保证 MDR ≤ 2mm 的距离区间: {zlo:.0f} - {zhi:.0f} mm")
        span = zhi - zlo
        print(f"  对应 Z 轴可用跨度: {span:.0f} mm  →  ±{span/2:.0f} mm")
        print(f"  建议基准距离: {(zlo+zhi)/2:.0f} mm（区间中心，余量最大）")
    ok3 = [c for c in curve if c["mdr"] <= 3]
    if ok3:
        zlo3 = min(c["z"] for c in ok3)
        zhi3 = max(c["z"] for c in ok3)
        print(f"\n  保证 MDR ≤ 3mm 的距离区间: {zlo3:.0f} - {zhi3:.0f} mm")
        print(f"  对应 Z 轴可用跨度: {zhi3-zlo3:.0f} mm  →  ±{(zhi3-zlo3)/2:.0f} mm")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
