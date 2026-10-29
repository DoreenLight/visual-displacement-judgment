#!/usr/bin/env python3
"""D435i 流测试 —— 用「单 pipeline 多流并发」的正常用法测试。

与 rs_check.py 的区别
---------------------
rs_check.py 每条流各建一个 pipeline（start/stop 三次）。
macOS 上设备状态脆弱，反复 start/stop 会让 SDK 上下文失效，
报 "No device connected"。本脚本用**一条 pipeline 同时开多条流**，
这才是实际采集时的用法。

用法（需要 sudo）
-----------------
    # 标准三流：深度 + 彩色 + 红外
    sudo -E /opt/anaconda3/envs/realsense/bin/python3 rs_stream_test.py

    # 只测深度 + 彩色（位移测量最常用）
    sudo -E .../rs_stream_test.py --streams depth,color

    # 先复位 USB 再测
    sudo -E .../rs_stream_test.py --reset

    # 连续采集 N 秒，统计帧率与丢帧
    sudo -E .../rs_stream_test.py --seconds 10
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np

# 必须在 import pyrealsense2 之前修好 USB
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

HERE = os.path.dirname(os.path.abspath(__file__))

# 每条流的配置：名称 -> (stream_type, 宽, 高, 格式, 帧率)
STREAM_DEFS = {
    "depth": (rs.stream.depth, 848, 480, rs.format.z16, 30),
    "color": (rs.stream.color, 640, 480, rs.format.bgr8, 30),
    "ir": (rs.stream.infrared, 848, 480, rs.format.y8, 30),
}


def maybe_reset():
    """调用 rs_usb_reset.py 做一次 USB 复位。"""
    print("--- 先做 USB 复位 ---")
    rl = os.path.join(HERE, "rs_usb_reset.py")
    if not os.path.exists(rl):
        print("  (找不到 rs_usb_reset.py，跳过)")
        return
    r = subprocess.run([sys.executable, rl], capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    if "复位成功" in out or "可以访问设备" in out:
        print("  ✅ 复位成功")
    else:
        print("  ⚠️  复位结果：")
        for ln in out.splitlines()[-6:]:
            print("    ", ln)
    time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default="depth,color,ir",
                    help="逗号分隔，可选 depth/color/ir")
    ap.add_argument("--reset", action="store_true", help="先复位 USB")
    ap.add_argument("--seconds", type=float, default=0,
                    help=">0 则连续采集该秒数并统计帧率")
    ap.add_argument("--warmup", type=int, default=30, help="预热丢弃帧数")
    args = ap.parse_args()

    if args.reset:
        maybe_reset()

    names = [s.strip() for s in args.streams.split(",") if s.strip()]
    for n in names:
        if n not in STREAM_DEFS:
            sys.exit(f"未知流: {n}（可选 {list(STREAM_DEFS)}）")

    # 1. 枚举设备
    print("\n" + "=" * 60)
    print("1. 枚举设备")
    print("=" * 60)
    ctx = rs.context()
    try:
        devs = list(ctx.devices)
    except RuntimeError as e:
        devs = []
        print(f"  ❌ 枚举失败: {e}")
        print("     → 先跑 rs_usb_reset.py 做 USB 复位，或加 --reset 参数")
        return 1
    if not devs:
        print("  ❌ 没有设备。先插好相机 / 跑 rs_usb_reset.py")
        return 1

    dev = devs[0]
    info = {}
    for k in ("name", "serial_number", "firmware_version", "usb_type_descriptor",
              "product_line", "physical_port"):
        try:
            info[k] = dev.get_info(getattr(rs.camera_info, k))
        except Exception:
            info[k] = "?"
    for k, v in info.items():
        print(f"  {k:20} = {v}")

    usb = info["usb_type_descriptor"]
    print(f"  {'✅ USB 链路正常' if usb.startswith('3') else '⚠️  USB 链路降级'} ({usb})")

    # 2. 单 pipeline 多流并发
    print("\n" + "=" * 60)
    print(f"2. 单 pipeline 同时开流: {', '.join(names)}")
    print("=" * 60)

    pipe = rs.pipeline()
    cfg = rs.config()
    for n in names:
        st, w, h, fmt, fps = STREAM_DEFS[n]
        cfg.enable_stream(st, w, h, fmt, fps)

    t0 = time.time()
    try:
        profile = pipe.start(cfg)
    except Exception as e:
        print(f"  ❌ pipeline.start 失败: {e}")
        print("\n  排查：")
        print("    1. sudo -E .../rs_usb_reset.py          # 复位 USB")
        print("    2. 换直插机身雷雳口，别走扩展坞")
        return 2

    print(f"  ✅ 管线启动成功（耗时 {time.time()-t0:.2f}s）")

    # 实际生效的配置
    print("\n  实际协商的流：")
    for n in names:
        st, w, h, fmt, fps = STREAM_DEFS[n]
        sp = profile.get_stream(st)
        vsp = sp.as_video_stream_profile()
        ii = vsp.get_intrinsics()
        print(f"    {n:6} {ii.width}x{ii.height} @ {sp.fps() if hasattr(sp,'fps') else fps} "
              f"fx={ii.fx:.1f} fy={ii.fy:.1f} "
              f"ppx={ii.ppx:.1f} ppy={ii.ppy:.1f} {ii.model}")

    # 3. 采集
    print("\n" + "=" * 60)
    print("3. 采集")
    print("=" * 60)

    nframes = 0
    counts = {n: 0 for n in names}
    first_frames = {}
    dropped_note = []

    try:
        for _ in range(args.warmup):
            fs = pipe.wait_for_frames(timeout_ms=5000)
            nframes += 1

        print(f"  预热 {args.warmup} 帧完成")

        if args.seconds > 0:
            t0 = time.time()
            while time.time() - t0 < args.seconds:
                fs = pipe.wait_for_frames(timeout_ms=5000)
                nframes += 1
                for n in names:
                    st = STREAM_DEFS[n][0]
                    if fs.first_or_default(st) is not None:
                        counts[n] += 1
                    else:
                        dropped_note.append(f"{n} 缺帧 @ {time.time()-t0:.1f}s")
            dur = time.time() - t0
            print(f"\n  连续采集 {dur:.1f}s，共 {nframes} 个帧集")
            print(f"  实测帧率: {nframes/dur:.1f} fps（帧集）")
            print("\n  各流有效帧数：")
            for n in names:
                exp = int(dur * STREAM_DEFS[n][4])
                got = counts[n]
                pct = 100.0 * got / exp if exp else 0
                flag = "✅" if pct > 90 else ("⚠️ " if pct > 50 else "❌")
                print(f"    {flag} {n:6} {got:5}/{exp:5} ({pct:5.1f}%)")
            if dropped_note:
                print(f"\n  缺帧记录（前 5 条）: {dropped_note[:5]}")
        else:
            fs = pipe.wait_for_frames(timeout_ms=5000)
            nframes += 1
            for n in names:
                st = STREAM_DEFS[n][0]
                f = fs.first_or_default(st)
                if f is not None:
                    first_frames[n] = f
                    counts[n] += 1
    except Exception as e:
        print(f"  ❌ 采集出错: {e}")
    finally:
        try:
            pipe.stop()
            print("\n  管线已停止")
        except Exception as e:
            print(f"\n  ⚠️ 停止管线出错: {e}")

    # 4. 结果
    print("\n" + "=" * 60)
    print("4. 结果")
    print("=" * 60)
    ok = [n for n in names if counts[n] > 0]
    bad = [n for n in names if counts[n] == 0]
    for n in ok:
        print(f"  ✅ {n}")
    for n in bad:
        print(f"  ❌ {n} 没出数据")

    if not args.seconds and first_frames:
        print("\n  数据摘要：")
        if "depth" in first_frames:
            d = np.asanyarray(first_frames["depth"].get_data())
            v = d[d > 0]
            if v.size:
                print(f"    深度 {d.shape} 有效 {100.0*v.size/d.size:.1f}% "
                      f"范围 {v.min()}~{v.max()} mm 中位 {int(np.median(v))} mm")
            else:
                print(f"    深度 {d.shape} 全为 0（无有效深度）")
        if "color" in first_frames:
            c = np.asanyarray(first_frames["color"].get_data())
            print(f"    彩色 {c.shape} 均值 {c.mean():.1f}")
        if "ir" in first_frames:
            i = np.asanyarray(first_frames["ir"].get_data())
            print(f"    红外 {i.shape} 均值 {i.mean():.1f}")

    if ok:
        print(f"\n✅ 可用流: {', '.join(ok)}")
        return 0
    print("\n❌ 没有可用流")
    return 3


if __name__ == "__main__":
    sys.exit(main())
