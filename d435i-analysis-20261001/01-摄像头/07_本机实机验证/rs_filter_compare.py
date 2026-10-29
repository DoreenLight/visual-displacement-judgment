#!/usr/bin/env python3
"""对比不同 SDK 配置对 Z 轴测量精度的改善。

测试矩阵
--------
  A. visual_preset : Default / HighAccuracy / HighDensity
  B. temporal_filter  (时间滤波, alpha / delta)
  C. spatial_filter   (空间滤波)
  D. decimation       (降采样)
  E. 高帧率 + 时间平均

对每种配置，用同一静止场景测 ROI-mean 的时间噪声 σ 与 MDR(3σ)，
直接给出「哪个配置最省误差」。

用法
----
sudo -E /opt/anaconda3/envs/realsense/bin/python3 rs_filter_compare.py
"""

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

# 关心的近距离箱（mm）
TARGET_BINS = [(0, 300), (300, 500), (500, 900)]
NFRAMES = 150


def make_filters(kind, w, h):
    """返回 (filters_list, note)"""
    if kind == "none":
        return [], ""
    if kind == "temporal":
        f = rs.temporal_filter()
        f.set_option(rs.option.filter_smooth_alpha, 0.4)
        f.set_option(rs.option.filter_smooth_delta, 20)
        return [f], "alpha=0.4 delta=20"
    if kind == "temporal_strong":
        f = rs.temporal_filter()
        f.set_option(rs.option.filter_smooth_alpha, 0.1)
        f.set_option(rs.option.filter_smooth_delta, 20)
        return [f], "alpha=0.1 delta=20"
    if kind == "spatial":
        f = rs.spatial_filter()
        f.set_option(rs.option.filter_magnitude, 2)
        f.set_option(rs.option.filter_smooth_alpha, 0.5)
        f.set_option(rs.option.filter_smooth_delta, 20)
        return [f], "mag=2"
    if kind == "both":
        t = rs.temporal_filter()
        t.set_option(rs.option.filter_smooth_alpha, 0.4)
        s = rs.spatial_filter()
        s.set_option(rs.option.filter_magnitude, 2)
        return [s, t], "spatial+temporal"
    if kind == "spatial_temporal":
        s = rs.spatial_filter()
        s.set_option(rs.option.filter_magnitude, 2)
        t = rs.temporal_filter()
        t.set_option(rs.option.filter_smooth_alpha, 0.4)
        return [s, t], "spatial→temporal"
    raise ValueError(kind)


def run_config(preset, filt, w, h, fps, nframes=NFRAMES, warmup=45):
    """跑一种配置，返回 {bin_label: (z, sigma, mdr, drift)} 或 None"""
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    try:
        profile = pipe.start(cfg)
    except Exception as e:
        print(f"    ❌ pipeline.start 失败: {e}")
        return None

    ds = profile.get_device().first_depth_sensor()
    preset_note = ""
    if preset is not None:
        try:
            ds.set_option(rs.option.visual_preset, preset)
            preset_note = f"{preset}"
        except Exception as e:
            preset_note = f"设置失败({e})"

    try:
        for _ in range(warmup):
            pipe.wait_for_frames(timeout_ms=8000)

        med_stack = []
        for _ in range(8):
            f = pipe.wait_for_frames(timeout_ms=8000)
            med_stack.append(np.asanyarray(f.get_depth_frame().get_data()))
        # 注意：用了滤波器时这里也走 filter
        med = np.median(np.stack(med_stack).astype(np.float32), axis=0)

        masks = [((med > 0) & (med < 65535) & (med >= lo) & (med < hi))
                 for lo, hi in TARGET_BINS]

        series = np.full((nframes, len(TARGET_BINS)), np.nan)
        filters, _ = make_filters(filt, w, h)

        for t in range(nframes):
            frames = pipe.wait_for_frames(timeout_ms=8000)
            df = frames.get_depth_frame()
            for flt in filters:
                df = flt.process(df)
            d = np.asanyarray(df.get_data())
            for i, m in enumerate(masks):
                sel = d[m & (d > 0) & (d < 65535)]
                if sel.size >= 100:
                    series[t, i] = sel.mean()
    except Exception as e:
        print(f"    ❌ 采集出错: {e}")
        return None
    finally:
        try:
            pipe.stop()
        except Exception:
            pass

    out = {}
    for i, (lo, hi) in enumerate(TARGET_BINS):
        s = series[:, i]
        if not np.isfinite(s).all():
            continue
        z = float(np.mean(s))
        dif = np.diff(s)
        single = float(np.std(dif, ddof=1)) / np.sqrt(2)
        mdr = 3 * single
        w_ = min(30, len(s) // 2)
        nw = len(s) // w_
        wm = s[:nw * w_].reshape(nw, w_).mean(axis=1) if nw >= 2 else s
        drift = float(wm.max() - wm.min())
        out[f"{lo}-{hi}"] = (z, single, mdr, drift)
    return out


def main():
    print("=" * 100)
    print("SDK 配置对比：Z 轴 ROI 测量精度（静止场景）")
    print(f"每种配置采集 {NFRAMES} 帧，统计 ROI-mean 的帧间噪声 σ 与 MDR(3σ)")
    print("=" * 100)

    # 预设枚举值
    # ⚠️ 注意：不要在 pipeline 运行中调用 ds.get_supported_options()。
    # 实测该调用会阻塞帧流（Frame didn't arrive），导致该 pipeline 无法正常
    # 收帧，且 stop() 后可能残留不洁的 USB 状态，污染紧接着的第一次采集。
    # 这很可能是之前观察到的「同一配置两次运行 MDR 差 16 倍」的根因。
    presets = {
        "default": None,
        "high_accuracy": 3,     # D400 预设编号：3 = HighAccuracy
        "high_density": 4,      # 4 = HighDensity
        "medium_density": 2,
    }

    matrix = [
        ("30fps 848x480", 848, 480, 30, None, "none"),
        ("30fps + temporal", 848, 480, 30, None, "temporal"),
        ("30fps + temporal(强)", 848, 480, 30, None, "temporal_strong"),
        ("30fps + spatial", 848, 480, 30, None, "spatial"),
        ("30fps + spatial+temporal", 848, 480, 30, None, "spatial_temporal"),
        ("90fps 848x480", 848, 480, 90, None, "none"),
        ("90fps + temporal", 848, 480, 90, None, "temporal"),
        ("90fps + spatial+temporal", 848, 480, 90, None, "spatial_temporal"),
        ("30fps HighAccuracy", 848, 480, 30, 3, "none"),
        ("30fps HighAccuracy+temporal", 848, 480, 30, 3, "temporal"),
        ("30fps HighDensity", 848, 480, 30, 4, "none"),
        ("30fps 640x480", 640, 480, 30, None, "none"),
        ("30fps 640x480+temporal", 640, 480, 30, None, "temporal"),
        ("90fps 640x480+temporal", 640, 480, 90, None, "temporal"),
    ]

    results = {}
    for name, w, h, fps, preset, filt in matrix:
        print(f"\n▶ {name}   (filters={filt})")
        r = run_config(preset, filt, w, h, fps)
        if r:
            results[name] = r
            for k, (z, sig, mdr, dr) in r.items():
                flag = "✅" if mdr <= 2 else ("⚠️ " if mdr <= 5 else "❌")
                print(f"    {k:>9} : Z={z:7.1f}mm  σ={sig:6.3f}  "
                      f"MDR={mdr:6.3f}mm {flag}  drift={dr:5.3f}")

    # ---- 汇总 ----
    print("\n" + "=" * 100)
    print("汇总：按最近距离箱（0-300mm）的 MDR 排序")
    print("=" * 100)
    rows = []
    for name, r in results.items():
        if "0-300" in r:
            z, sig, mdr, dr = r["0-300"]
            rows.append((mdr, name, z, sig, dr))
    rows.sort()
    print(f"{'排名':>4} {'配置':<32} {'Z(mm)':>8} {'σ(mm)':>8} {'MDR(mm)':>9} {'drift':>7}")
    print("-" * 76)
    for i, (mdr, name, z, sig, dr) in enumerate(rows, 1):
        print(f"{i:>4} {name:<32} {z:>8.1f} {sig:>8.3f} {mdr:>9.3f} {dr:>7.3f}")

    if rows:
        best = rows[0]
        base = next((r for r in rows if r[1].startswith("30fps 848x480")), None)
        print(f"\n最优: {best[1]}  MDR={best[0]:.3f} mm")
        if base and best[0] > 0:
            print(f"相对 30fps 848x480 (MDR={base[0]:.3f} mm) 改善 "
                  f"{base[0]/best[0]:.2f}×")

    # 各距离箱的最优
    print("\n各距离箱最优配置：")
    for lo, hi in TARGET_BINS:
        key = f"{lo}-{hi}"
        cands = [(r[key][2], n) for n, r in results.items() if key in r]
        if cands:
            cands.sort()
            print(f"  {key:>9} : {cands[0][1]:<32} MDR={cands[0][0]:.3f} mm")

    return 0


if __name__ == "__main__":
    sys.exit(main())
