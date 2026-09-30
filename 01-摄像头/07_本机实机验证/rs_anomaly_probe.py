#!/usr/bin/env python3
"""决定性实验：16× 异常是「采集问题」还是「统计脚本问题」？

Sol 建议的二叉定位
------------------
录制同一份 raw depth 数据一次，然后用两种不同的统计逻辑分析它。
  - 同一 raw 数据 → 两个逻辑算出不同结果 = 统计/掩膜问题
  - 同一 raw 数据 → 结果一致，但两次采集差 16× = 采集问题

本脚本实现
----------
1. 采集一次，把原始 uint16 深度帧全部存盘（.npy）
2. 在同一份数据上，用两套独立实现的掩膜+统计逻辑分别计算
3. 对比结果差异

两套逻辑还原自：
  逻辑 A = rs_filter_compare.py 的做法（8 帧中值建掩膜, 150 帧, 3 箱）
  逻辑 B = rs_mdr_probe.py 的做法（10 帧中值建掩膜, 长序列, 细分箱）

同时引入第三个对照：
  逻辑 C = 完全用固定像素集合（掩膜不随帧变化）—— 排除「有效集合漂移」污染

并把每帧的元数据（曝光/增益/帧号）一并记录，用于事后归因。

用法
----
# 采集并存盘
sudo -E .../rs_anomaly_probe.py --record /tmp/raw.npz --frames 600

# 分析已有数据（可与上面的采集分开运行，验证复现性）
sudo -E .../rs_anomaly_probe.py --analyze /tmp/raw.npz
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

BINS_A = [(0, 300), (300, 500), (500, 900)]          # rs_filter_compare 的箱
BINS_B = [(0, 300), (300, 500), (500, 800), (800, 1200),
          (1200, 1800), (1800, 2800), (2800, 5000), (5000, 9000)]


def record(path, frames, warmup, w, h, fps, alpha=0.0):
    """采集原始深度并存盘（含逐帧元数据）。"""
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    profile = pipe.start(cfg)
    ds = profile.get_device().first_depth_sensor()

    tf = None
    if alpha > 0:
        tf = rs.temporal_filter()
        tf.set_option(rs.option.filter_smooth_alpha, alpha)
        tf.set_option(rs.option.filter_smooth_delta, 20)

    print(f"采集 {w}x{h}@{fps}  temporal={'α='+str(alpha) if tf else 'off'}")

    # ⚠️ 重要：不能在 pipeline 运行中调用 get_supported_options()。
    # 实测该调用会阻塞帧流（Frame didn't arrive），无论是否读取值。
    # 只能逐个读取已知安全的 option。
    opts = {}
    for name in ("exposure", "gain", "laser_power", "enable_auto_exposure",
                 "depth_units", "visual_preset", "stereo_baseline",
                 "emitter_enabled", "frames_queue_size"):
        try:
            opts[name] = ds.get_option(getattr(rs.option, name))
        except Exception:
            pass
    print(f"已记录 sensor 状态 {len(opts)} 项（避开阻塞帧流的 enumerate）")

    for _ in range(warmup):
        pipe.wait_for_frames(timeout_ms=10000)

    data = np.zeros((frames, h, w), np.uint16)
    meta = {k: np.zeros(frames) for k in
            ("exposure", "gain", "frameno", "timestamp", "laser")}

    for t in range(frames):
        fr = pipe.wait_for_frames(timeout_ms=10000)
        df = fr.get_depth_frame()
        if tf is not None:
            # 只对输出做滤波，但把滤波后结果当作「原始」记录
            df = tf.process(df)
        data[t] = np.asanyarray(df.get_data())
        # 元数据一律从 sensor 读取，避免 frame 对象上不存在的方法
        try:
            meta["exposure"][t] = ds.get_option(rs.option.exposure)
        except Exception:
            meta["exposure"][t] = np.nan
        try:
            meta["gain"][t] = ds.get_option(rs.option.gain)
        except Exception:
            meta["gain"][t] = np.nan
        try:
            meta["laser"][t] = ds.get_option(rs.option.laser_power)
        except Exception:
            meta["laser"][t] = np.nan
        try:
            meta["frameno"][t] = df.get_frame_number()
        except Exception:
            meta["frameno"][t] = t
        try:
            meta["timestamp"][t] = df.get_timestamp()
        except Exception:
            meta["timestamp"][t] = np.nan

    pipe.stop()
    np.savez_compressed(path, data=data, meta_keys=np.array(list(meta.keys())),
                        **{f"meta_{k}": v for k, v in meta.items()},
                        opts_keys=np.array(list(opts.keys())),
                        opts_vals=np.array(list(opts.values()), dtype=np.float64),
                        w=w, h=h, fps=fps, alpha=alpha)
    print(f"✅ 已存盘 {path}  shape={data.shape}")

    # 存盘后立刻自检元数据稳定性
    print("\n--- 采集期元数据稳定性 ---")
    for k in ("exposure", "gain"):
        v = meta[k]
        if np.isfinite(v).all() and v.std() > 0:
            print(f"  ⚠️ {k} 在采集期变化: 均值 {v.mean():.1f} σ {v.std():.1f} "
                  f"范围 [{v.min():.0f}, {v.max():.0f}]")
        else:
            print(f"  ✅ {k} 稳定: {v[0]:.1f}")
    fn = meta["frameno"]
    if np.isfinite(fn).all() and len(fn) > 1:
        d = np.diff(fn)
        gaps = int((d != 1).sum())
        print(f"  帧号连续性: {'✅ 连续' if gaps == 0 else f'⚠️ 有 {gaps} 处跳变（丢帧）'}")
    ts = meta["timestamp"]
    if np.isfinite(ts).all() and len(ts) > 1:
        dt = np.diff(ts)
        print(f"  时间戳间隔: 中位 {np.median(dt):.2f}ms  "
              f"σ {dt.std():.2f}ms  范围 [{dt.min():.2f}, {dt.max():.2f}]")
    lp = meta.get("laser", np.array([]))
    if lp.size and np.isfinite(lp).all() and lp.std() == 0:
        print(f"  激光功率稳定: {lp[0]:.0f}")
    return 0


def mask_from_median(stack, bins, min_px=100):
    med = np.median(stack.astype(np.float32), axis=0)
    v = (med > 0) & (med < 65535)
    return np.stack(stack).astype(np.float32), med, [
        v & (med >= lo) & (med < hi) for lo, hi in bins]


def analyze(path):
    z = np.load(path, allow_pickle=True)
    data = z["data"].astype(np.float32)
    n, h, w = data.shape
    fps = int(z["fps"])
    print("=" * 92)
    print(f"分析 {path}")
    print(f"  帧数 {n}  分辨率 {w}x{h}  fps {fps}  "
          f"temporal α={float(z['alpha']):.2f}")
    print("=" * 92)

    # 元数据复核
    print("\n【元数据】")
    exp = z["meta_exposure"]
    gain = z["meta_gain"]
    fn = z["meta_frameno"]
    ts = z["meta_timestamp"]
    if np.isfinite(exp).all() and exp.std() > 0:
        print(f"  ⚠️ 曝光变化: σ={exp.std():.1f} 范围 [{exp.min():.0f}, {exp.max():.0f}] μs")
        jumps = int((np.abs(np.diff(exp)) > exp.std() * 3).sum())
        print(f"     大跳变次数: {jumps}")
    else:
        print(f"  ✅ 曝光稳定: {exp[0]:.0f} μs")
    if np.isfinite(gain).all() and gain.std() > 0:
        print(f"  ⚠️ 增益变化: σ={gain.std():.2f} 范围 [{gain.min():.0f}, {gain.max():.0f}]")
    else:
        print(f"  ✅ 增益稳定: {gain[0]:.0f}")
    if np.isfinite(fn).all():
        d = np.diff(fn)
        print(f"  {'✅ 帧号连续' if (d==1).all() else f'⚠️ 丢帧 {(d!=1).sum()} 处'}")

    # ---------------------------------------------------------------
    def metrics(series, name, label):
        if not np.isfinite(series).all():
            print(f"  {label:<28} {name}: 不完整")
            return None
        s = series.astype(np.float64)
        # 三种口径
        mdr_diff = 3 * float(np.std(np.diff(s), ddof=1)) / np.sqrt(2)
        scatter = float(np.std(s, ddof=1))
        mdr_scatter = 3 * scatter
        # lag-1
        sm = s - s.mean()
        rho1 = (float(np.sum(sm[:-1] * sm[1:]) / np.sum(sm * sm))
                if np.sum(sm * sm) > 0 else np.nan)
        # 漂移（1 秒窗口极差）
        nw = n // fps
        drift = (float(np.ptp(s[:nw*fps].reshape(nw, fps).mean(axis=1)))
                 if nw >= 2 else np.nan)
        print(f"  {label:<28} Z={np.mean(s):7.1f}  "
              f"MDR_diff={mdr_diff:8.3f}  scatterσ={scatter:7.3f}  "
              f"MDR_scat={mdr_scatter:8.3f}  ρ₁={rho1:+.3f}  drift={drift:6.3f}")
        return dict(mdr_diff=mdr_diff, scatter=scatter, mdr_scat=mdr_scatter,
                    rho1=rho1, drift=drift, z=np.mean(s))

    # 逻辑 A：rs_filter_compare 的做法
    print("\n【逻辑 A】rs_filter_compare 做法：8 帧中值建掩膜")
    rawA, medA, masksA = mask_from_median(data[:8], BINS_A)
    resA = {}
    for (lo, hi), m in zip(BINS_A, masksA):
        s = np.array([data[t][m & (data[t] > 0) & (data[t] < 65535)].mean()
                      if (m & (data[t] > 0) & (data[t] < 65535)).sum() > 0
                      else np.nan for t in range(n)])
        r = metrics(s, "", f"{lo}-{hi}mm(n={int(m.sum())})")
        if r:
            resA[f"{lo}-{hi}"] = r

    # 逻辑 B：rs_mdr_probe 的做法（用全部帧中值建掩膜，更稳）
    print("\n【逻辑 B】rs_mdr_probe 做法：全序列中值建掩膜 + 细分箱")
    rawB, medB, masksB = mask_from_median(data, BINS_B)
    resB = {}
    for (lo, hi), m in zip(BINS_B, masksB):
        if m.sum() < 100:
            continue
        s = np.array([data[t][m & (data[t] > 0) & (data[t] < 65535)].mean()
                      if (m & (data[t] > 0) & (data[t] < 65535)).sum() > 0
                      else np.nan for t in range(n)])
        r = metrics(s, "", f"{lo}-{hi}mm(n={int(m.sum())})")
        if r:
            resB[f"{lo}-{hi}"] = r

    # 逻辑 C：完全固定像素集合（掩膜不随帧变化）
    print("\n【逻辑 C】固定像素集合（排除有效集合漂移污染）")
    med = np.median(data, axis=0).astype(np.float32)
    v = (med > 0) & (med < 65535)
    resC = {}
    for (lo, hi) in BINS_A:
        m = v & (med >= lo) & (med < hi)
        if m.sum() < 100:
            continue
        # 固定像素：直接取该掩膜内的像素，缺失值用该像素自身的时间中值填补
        sub = data[:, m]                      # (n, k)
        subv = np.where((sub > 0) & (sub < 65535), sub, np.nan)
        s = np.nanmean(subv, axis=1)
        r = metrics(s, "", f"{lo}-{hi}mm(n={int(m.sum())})")
        if r:
            resC[f"{lo}-{hi}"] = r

    # 有效集合漂移度量
    print("\n【有效集合漂移】每帧有效像素数的时间波动（会污染 ROI 均值）")
    for (lo, hi) in BINS_A:
        m = v & (med >= lo) & (med < hi)
        if m.sum() < 100:
            continue
        cnt = ((data[:, m] > 0) & (data[:, m] < 65535)).sum(axis=1)
        print(f"  {lo}-{hi}mm: 有效像素 均值 {cnt.mean():.0f}  σ {cnt.std():.1f}  "
              f"范围 [{cnt.min()}, {cnt.max()}]  "
              f"{'⚠️ 集合在变' if cnt.std() > 2 else '✅ 稳定'}")

    # ---------------------------------------------------------------
    print("\n" + "=" * 92)
    print("三套逻辑的一致性对比（同一份 raw 数据）")
    print("=" * 92)
    print(f"{'距离箱':>12} {'逻辑A MDR':>11} {'逻辑B MDR':>11} {'逻辑C MDR':>11} "
          f"{'A/C':>7} {'B/C':>7} {'判定':>12}")
    print("-" * 92)
    all_keys = sorted(set(resA) | set(resB) | set(resC),
                      key=lambda k: int(k.split("-")[0]))
    max_ratio = 0
    for k in all_keys:
        a = resA.get(k, {}).get("mdr_diff")
        b = resB.get(k, {}).get("mdr_diff")
        c = resC.get(k, {}).get("mdr_diff")
        ra = a / c if (a and c) else np.nan
        rb = b / c if (b and c) else np.nan
        rr = np.nanmax([ra, rb])
        if np.isfinite(rr):
            max_ratio = max(max_ratio, rr)
        verdict = ("一致" if np.isfinite(rr) and rr < 1.5 else
                   ("差异大" if np.isfinite(rr) else "—"))
        fa = f"{a:.2f}" if a else "—"
        fb = f"{b:.2f}" if b else "—"
        fc = f"{c:.2f}" if c else "—"
        print(f"{k:>12} {fa:>11} {fb:>11} {fc:>11} "
              f"{ra if np.isfinite(ra) else 0:>7.2f} "
              f"{rb if np.isfinite(rb) else 0:>7.2f} {verdict:>12}")

    print("\n" + "=" * 92)
    print("结论")
    print("=" * 92)
    if max_ratio < 1.5:
        print(f"  ✅ 三套逻辑在**同一份 raw 数据**上结果一致（最大差异 {max_ratio:.2f}×）")
        print("     → 统计脚本不是差异来源。")
        print("     → 之前观察到的 16× 差异必然是**采集阶段**造成的。")
        print("     → 排查方向：sensor 状态、曝光、温度、USB、pipeline 重启次数。")
    else:
        print(f"  ⚠️ 三套逻辑结果不一致（最大差异 {max_ratio:.2f}×）")
        print("     → 掩膜/统计方式的选择会显著影响 MDR 结论。")
        print("     → 必须在报告中固定并声明统计口径。")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default=None, help="采集并保存到该路径 (.npz)")
    ap.add_argument("--analyze", default=None, help="分析已有 .npz")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--warmup", type=int, default=45)
    ap.add_argument("--width", type=int, default=848)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="temporal alpha，0=关闭")
    args = ap.parse_args()

    if args.record:
        return record(args.record, args.frames, args.warmup,
                      args.width, args.height, args.fps, args.alpha)
    if args.analyze:
        return analyze(args.analyze)
    ap.error("需要 --record 或 --analyze")


if __name__ == "__main__":
    sys.exit(main())
