#!/usr/bin/env python3
"""确证 16 倍异常根因：get_supported_options() 是否污染后续采集。

假设
----
`get_supported_options()` 在 pipeline 运行中调用会阻塞帧流，
且 stop() 后残留不洁状态，污染紧接着的下一次采集。

A/B 对照
--------
A 组（污染）：start → get_supported_options() → stop → 立即采 300 帧
B 组（对照）：start → （什么都不做）        → stop → 立即采 300 帧

交替重复多轮，排除时间顺序造成的偏差。

判定
----
若 A 组 MDR 显著高于 B 组，根因确证。

用法
----
sudo -E /opt/anaconda3/envs/realsense/bin/python3 rs_pollution_ab.py
sudo -E .../rs_pollution_ab.py --rounds 4 --frames 300
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

W, H, FPS = 848, 480, 30
BIN = (300, 500)   # 取 300-500mm 作为主要观察区间（人体测量工作区）


def warm_pipeline(pollute: bool):
    """按需制造污染：启动 pipeline 后（可选）调用 get_supported_options。"""
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    prof = pipe.start(cfg)
    ds = prof.get_device().first_depth_sensor()

    if pollute:
        try:
            n = len(ds.get_supported_options())   # ← 被怀疑的调用
            note = f"枚举 {n} 项"
        except Exception as e:
            note = f"枚举异常 {e}"

    # 无论是否污染，都尝试收几帧（污染时会超时）
    got = 0
    t0 = time.time()
    while time.time() - t0 < 3.0:
        try:
            pipe.wait_for_frames(timeout_ms=1500)
            got += 1
            if got >= 30:
                break
        except Exception:
            break

    try:
        pipe.stop()
    except Exception:
        pass

    return got, (note if pollute else "-")


def measure(nframes=300, warmup=30):
    """采集并返回该次运行的 MDR 指标。"""
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    prof = pipe.start(cfg)

    for _ in range(warmup):
        pipe.wait_for_frames(timeout_ms=10000)

    # 中值帧建掩膜
    stack = []
    for _ in range(10):
        d = np.asanyarray(
            pipe.wait_for_frames(timeout_ms=10000).get_depth_frame().get_data())
        stack.append(d.astype(np.float32))
    med = np.median(np.stack(stack), axis=0)
    v = (med > 0) & (med < 65535)
    m = v & (med >= BIN[0]) & (med < BIN[1])

    if m.sum() < 100:
        # 回退：取最大的距离箱
        hist, edges = np.histogram(med[v], bins=20)
        k = int(np.argmax(hist))
        m = v & (med >= edges[k]) & (med < edges[k + 1])
    npx = int(m.sum())

    ser = np.full(nframes, np.nan)
    cnt = np.zeros(nframes, np.int32)
    for t in range(nframes):
        d = np.asanyarray(
            pipe.wait_for_frames(timeout_ms=10000).get_depth_frame().get_data())
        sel = d[m & (d > 0) & (d < 65535)]
        cnt[t] = sel.size
        if sel.size >= 20:
            ser[t] = sel.mean()
    pipe.stop()

    if not np.isfinite(ser).all():
        return dict(ok=False, npx=npx, z=np.nan, mdr=np.nan,
                    scatter=np.nan, drift=np.nan, cnt_sigma=np.nan)

    s = ser.astype(np.float64)
    mdr_diff = 3 * float(np.std(np.diff(s), ddof=1)) / np.sqrt(2)
    scatter = float(np.std(s, ddof=1))
    nw = nframes // FPS
    drift = float(np.ptp(s[:nw * FPS].reshape(nw, FPS).mean(axis=1))) \
        if nw >= 2 else np.nan
    return dict(ok=True, npx=npx, z=float(np.median(s)), mdr=mdr_diff,
                scatter=scatter, drift=drift,
                cnt_sigma=float(cnt.std(ddof=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--frames", type=int, default=300)
    args = ap.parse_args()

    print("=" * 94)
    print("A/B 对照：get_supported_options() 是否污染后续采集")
    print("=" * 94)
    print(f"观察区间 {BIN[0]}-{BIN[1]}mm   每轮 {args.frames} 帧   "
          f"交替 {args.rounds} 轮\n")
    print(f"{'轮':>3} {'组':>5} {'预热收帧':>9} {'备注':>12} {'像素':>7} "
          f"{'Z(mm)':>9} {'MDR(3σ)':>10} {'scatter':>9} {'漂移':>8}")
    print("-" * 94)

    A, B = [], []
    for r in range(1, args.rounds + 1):
        for grp, pollute in (("A污染", True), ("B对照", False)):
            got, note = warm_pipeline(pollute)
            time.sleep(1.0)
            m = measure(args.frames)
            line = (f"{r:>3} {grp:>5} {got:>9} {note:>12} {m['npx']:>7} "
                    f"{m['z']:>9.1f} ")
            if m["ok"]:
                line += (f"{m['mdr']:>10.3f} {m['scatter']:>9.3f} "
                         f"{m['drift']:>8.3f}")
            else:
                line += f"{'FAIL':>10} {'-':>9} {'-':>8}"
            print(line)
            if m["ok"]:
                (A if pollute else B).append(m["mdr"])

    print("\n" + "=" * 94)
    print("统计")
    print("=" * 94)

    def stat(name, arr):
        if not arr:
            print(f"  {name}: 无有效数据")
            return None
        a = np.array(arr)
        print(f"  {name}: MDR 均值 {a.mean():8.3f}  σ {a.std(ddof=1) if len(a)>1 else 0:7.3f}  "
              f"范围 [{a.min():.3f}, {a.max():.3f}]  (n={len(a)})")
        return a

    aA = stat("A 污染组", A)
    aB = stat("B 对照组", B)

    print()
    if aA is not None and aB is not None and aB.mean() > 0:
        ratio = aA.mean() / aB.mean()
        print(f"  A/B 比值 = {ratio:.2f}×")
        if ratio > 2.0:
            print("\n  ✅ 根因确证：get_supported_options() 确实污染后续采集。")
            print("     → 之前 16 倍异常的解释成立。")
            print("     → 既有数据中凡经过该调用的配置需重测。")
        elif ratio > 1.3:
            print("\n  ⚠️ 有影响但不显著（1.3-2.0×）。")
            print("     → 该调用有副作用，但不是 16 倍差异的主因，需继续排查。")
        else:
            print("\n  ❌ 未复现污染效应（比值 <1.3）。")
            print("     → 16 倍异常另有原因，需继续排查（参考 Sol 建议的其他方向：")
            print("       sensor 状态差异、温度、USB 丢帧、pipeline 重建次数）。")
    else:
        print("  ⚠️ 数据不足，无法判定。")
    print("=" * 94)
    return 0


if __name__ == "__main__":
    sys.exit(main())
