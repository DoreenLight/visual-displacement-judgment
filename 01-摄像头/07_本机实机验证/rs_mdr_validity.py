#!/usr/bin/env python3
"""验证 temporal 滤波是否让 std(diff)/sqrt(2) 的 MDR 估计系统性偏小。

Sol 的指控
----------
我用 `std(diff)/sqrt(2)` 反推单帧噪声，这隐含「相邻帧独立」的假设。
temporal filter 人为制造强时间正相关，此时：

    var(f[t] - f[t-1]) = 2*sigma^2*(1 - rho)

所以 `std(diff)/sqrt(2) = sigma*sqrt(1-rho)`，当 rho > 0 时**系统性偏小**。

本脚本验证
----------
对同一静止场景，同时计算三种度量：

  A. std(series)              输出的实际散布 ← 静止目标测量时真正关心的量
  B. std(diff)/sqrt(2)        我原来用的 MDR 基础（隐含独立性假设）
  C. 去趋势后 std             排除缓慢漂移后的散布

并测量 lag-1 自相关 rho，直接检验相关性假设。

对「滤波 on / off」两种配置分别做，对比结论是否被影响。

用法
----
sudo -E /opt/anaconda3/envs/realsense/bin/python3 rs_mdr_validity.py
"""

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


def detrend(s, win=None):
    """滑动均值去趋势（保持长度）。"""
    n = len(s)
    if win is None:
        win = max(3, n // 10)
    if win < 3:
        return s - s.mean()
    k = np.ones(win) / win
    pad = win // 2
    sp = np.pad(s, pad, mode="edge")
    trend = np.convolve(sp, k, mode="valid")[:n]
    return s - trend


def collect(pipe, tf, mask, nframes, warmup=45):
    """采集 ROI 逐帧均值序列 + 逐帧有效集合信息。"""
    for _ in range(warmup):
        pipe.wait_for_frames(timeout_ms=10000)
    out = []
    nvalid = []
    for _ in range(nframes):
        fr = pipe.wait_for_frames(timeout_ms=10000)
        df = fr.get_depth_frame()
        if tf is not None:
            df = tf.process(df)
        d = np.asanyarray(df.get_data())
        sel = d[mask & (d > 0) & (d < 65535)]
        # 固定像素集合：只在预先确定的 mask 内取值（不随帧变化）
        if mask.sum() > 0:
            sub = d[mask]
            subv = sub[(sub > 0) & (sub < 65535)]
            nvalid.append(int(subv.size))
            out.append(float(subv.mean()) if subv.size > 20 else np.nan)
        else:
            out.append(np.nan)
    return np.array(out), np.array(nvalid)


def analyze(s, name):
    if not np.isfinite(s).all():
        print(f"  {name}: 数据不完整")
        return None
    s = s.astype(np.float64)
    n = len(s)

    a = float(np.std(s, ddof=1))                       # 实际散布
    b = float(np.std(np.diff(s), ddof=1)) / np.sqrt(2)  # 我原来的估计
    r = a / b if b > 0 else np.nan

    # lag-1 自相关
    sm = s - s.mean()
    rho1 = float(np.sum(sm[:-1] * sm[1:]) / np.sum(sm * sm)) if np.sum(sm*sm) > 0 else np.nan

    # 去趋势后
    d1 = detrend(s)
    c = float(np.std(d1, ddof=1))

    print(f"  {name}")
    print(f"    A std(series)         = {a:8.4f} mm   ← 实际散布（真实关心量）")
    print(f"    B std(diff)/sqrt(2)   = {b:8.4f} mm   ← 我原来的 MDR 基础")
    print(f"    A/B 放大系数          = {r:8.3f}      {'⚠️ B 偏小' if r > 1.2 else '✅ 接近'}")
    print(f"    lag-1 自相关 rho      = {rho1:+8.3f}     {'⚠️ 强正相关' if rho1 > 0.3 else '✅ 弱'}") 
    print(f"    C 去趋势后 std        = {c:8.4f} mm")
    return dict(A=a, B=b, ratio=r, rho1=rho1, C=c, n=n)


def main():
    print("=" * 84)
    print("MDR 估计有效性验证：temporal 滤波是否使 std(diff)/√2 系统性偏小")
    print("=" * 84)
    print("\nSol 的指控：temporal filter 制造时间正相关，")
    print("使 std(diff)/√2 = σ√(1-ρ) 系统性偏小，我报告的 MDR 过于乐观。\n")

    nframes = 600
    result = {}

    for label, alpha in (("无滤波", 0.0), ("temporal α=0.4", 0.4),
                         ("temporal α=0.1", 0.1)):
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        profile = pipe.start(cfg)

        tf = None
        if alpha > 0:
            tf = rs.temporal_filter()
            tf.set_option(rs.option.filter_smooth_alpha, alpha)
            tf.set_option(rs.option.filter_smooth_delta, 20)

        # 用中值帧选一块稳定 ROI（画面中央偏下，通常是桌面）
        stack = []
        for _ in range(10):
            f = pipe.wait_for_frames(timeout_ms=10000)
            d = np.asanyarray(f.get_depth_frame().get_data())
            stack.append(d.astype(np.float32))
        med = np.median(np.stack(stack), axis=0)

        # 找画面中位深度附近的稳定区域
        v = (med > 0) & (med < 65535)
        if v.sum() < 1000:
            print("有效像素太少，跳过")
            pipe.stop()
            continue
        zmed = np.median(med[v])
        band = v & (np.abs(med - zmed) < 50)   # 同一深度层 ±50mm

        # 取该层中最大的连通块近似：直接用中心窗口
        ys, xs = np.where(band)
        cy, cx = int(np.median(ys)), int(np.median(xs))
        H = W = 60
        y0 = max(0, min(med.shape[0] - H, cy - H // 2))
        x0 = max(0, min(med.shape[1] - W, cx - W // 2))
        mask = np.zeros_like(v)
        mask[y0:y0+H, x0:x0+W] = True
        mask &= v

        print(f"\n{label}: ROI {W}x{H} @ ({x0},{y0})  "
              f"固定像素 {int(mask.sum())} 个, 中位 Z={np.median(med[mask]):.0f}mm")

        s, nv = collect(pipe, tf, mask, nframes)
        pipe.stop()

        # 逐帧有效像素是否波动
        nv_ok = nv[np.isfinite(s)]
        if nv_ok.size:
            print(f"    逐帧有效像素: 均值 {nv_ok.mean():.0f}  "
                  f"σ {nv_ok.std(ddof=1):.1f}  范围 "
                  f"[{nv_ok.min()}, {nv_ok.max()}]  "
                  f"{'⚠️ 有效集合在变' if nv_ok.std(ddof=1) > 1 else '✅ 固定'}")

        result[label] = analyze(s, label)

    # ---------- 结论 ----------
    print("\n" + "=" * 84)
    print("结论")
    print("=" * 84)

    # 用 A（实际散布）重新计算 MDR
    print("\n用两种口径重算 MDR(3σ)：\n")
    print(f"{'配置':>18} {'B 口径(我原来)':>16} {'A 口径(实际散布)':>18} {'放大':>8}")
    print("-" * 64)
    for label, r in result.items():
        if r:
            print(f"{label:>18} {3*r['B']:>14.3f}mm {3*r['A']:>16.3f}mm "
                  f"{r['ratio']:>7.2f}×")

    print("\n" + "=" * 84)
    verdict = []
    for label, r in result.items():
        if r and r["ratio"] > 1.2:
            verdict.append((label, r["ratio"]))
    if verdict:
        print("✅ Sol 的指控成立：temporal 滤波下 std(diff)/√2 系统性偏小")
        for label, ratio in verdict:
            print(f"   {label}: 实际散布是原估计的 {ratio:.2f} 倍")
        print("\n   → 我之前报告的 filtered MDR 偏乐观，需要用 std(series) 重算。")
    else:
        print("❌ Sol 的指控不成立：两种口径接近，原 MDR 估计可用。")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    sys.exit(main())
