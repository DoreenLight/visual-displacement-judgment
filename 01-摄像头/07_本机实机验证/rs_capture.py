#!/usr/bin/env python3
"""D435i 采集脚本：抓取一帧深度 + 彩色 + 红外，存成 PNG 并打印统计。

用法（macOS 12+ 需要 sudo）：
    sudo -E /opt/anaconda3/envs/realsense/bin/python3 rs_capture.py
    sudo -E /opt/anaconda3/envs/realsense/bin/python3 rs_capture.py --out data/frame01

产出：
    <out>_color.png   彩色图
    <out>_depth.png   深度伪彩色（便于肉眼看）
    <out>_depth.npy   深度原始数据（毫米，uint16，16位精度）
    <out>_ir.png      左红外图

macOS 注意：本脚本会自动调用 macos_usb_fix 解绑抢占 USB 接口的系统 UVC 驱动。
"""

import argparse
import os
import sys

import numpy as np

# 必须在 import pyrealsense2 之前修好 USB
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import macos_usb_fix
    macos_usb_fix.ensure_ready(verbose=True)
except Exception as _e:
    print(f"[warn] macos_usb_fix 未生效: {_e}")

try:
    import pyrealsense2 as rs
except ImportError:
    sys.exit("缺少 pyrealsense2，请用 conda 环境 realsense 的 python 运行。")


HERE = os.path.dirname(os.path.abspath(__file__))

# 深度伪彩色可视化的最远距离（mm）。D435i 在室内可靠量程约 0.3~3m，
# 取 3m 作为上限，超出部分按最远处理。
VIS_MAX_MM = 3000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="captures/frame", help="输出路径前缀")
    ap.add_argument("--warmup", type=int, default=30, help="预热丢弃帧数（自动曝光需要）")
    args = ap.parse_args()

    outdir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(outdir, exist_ok=True)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 30)

    print("启动管线…")
    profile = pipe.start(cfg)

    # 对齐彩色到深度坐标系（做视觉位移测量时通常需要）
    align = rs.align(rs.stream.color)

    print(f"预热 {args.warmup} 帧（等自动曝光稳定）…")
    frames = None
    for _ in range(args.warmup):
        frames = pipe.wait_for_frames(timeout_ms=10000)

    frames = align.process(frames)
    depth = frames.get_depth_frame()
    color = frames.get_color_frame()
    ir = frames.get_infrared_frame(1)

    if not depth or not color:
        sys.exit("❌ 未能同时拿到深度和彩色帧")

    d = np.asanyarray(depth.get_data())
    c = np.asanyarray(color.get_data())
    irr = np.asanyarray(ir.get_data()) if ir else None

    # 存彩色
    try:
        from PIL import Image
        Image.fromarray(c[:, :, ::-1]).save(f"{args.out}_color.png")
        print(f"✅ {args.out}_color.png   {c.shape}")
    except ImportError:
        np.save(f"{args.out}_color.npy", c)
        print(f"✅ {args.out}_color.npy   {c.shape} (无 PIL，存 npy)")

    # 存深度原始数据（位移测量要用这个，别用伪彩色图）
    np.save(f"{args.out}_depth.npy", d)
    print(f"✅ {args.out}_depth.npy   {d.shape} dtype={d.dtype}")

    # 存深度伪彩色，方便肉眼看
    # 注意：不能用 min-max 归一化。离群值（>10m 的杂散读数）会把
    # 1~2m 内的真实场景压成接近全黑。改用分位数 + 固定上限。
    valid = d[(d > 0) & (d < 65535)]
    if valid.size:
        dmin = int(np.percentile(valid, 1))
        dmax = min(int(np.percentile(valid, 99)), VIS_MAX_MM)
    else:
        dmin, dmax = 0, VIS_MAX_MM
    if dmax <= dmin:
        dmax = dmin + 1
    # 超出可视化上限的当最远
    clipped = np.clip(d.astype(np.float32), dmin, dmax)
    norm = (clipped - dmin) / (dmax - dmin)
    norm[(d == 0) | (d == 65535)] = 0  # 无效像素标黑
    gray = (norm * 255).astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(gray).save(f"{args.out}_depth.png")
        print(f"✅ {args.out}_depth.png   （伪彩色，归一化区间 "
              f"{dmin}~{dmax} mm，超出按 {dmax} 处理）")
    except ImportError:
        np.save(f"{args.out}_depth_vis.npy", gray)

    if irr is not None:
        try:
            from PIL import Image
            Image.fromarray(irr).save(f"{args.out}_ir.png")
            print(f"✅ {args.out}_ir.png      {irr.shape}")
        except ImportError:
            np.save(f"{args.out}_ir.npy", irr)

    # 统计
    dev = profile.get_device()
    nz = int((d == 0).sum())
    nmax = int((d == 65535).sum())
    real = d[(d > 0) & (d < 65535)]
    print("\n--- 深度统计 ---")
    print(f"  分辨率     {d.shape[1]}x{d.shape[0]}")
    print(f"  无效(0)    {nz}/{d.size} ({100.0*nz/d.size:.1f}%)")
    print(f"  离群(65535) {nmax}/{d.size} ({100.0*nmax/d.size:.1f}%)")
    print(f"  有效读数   {real.size}/{d.size} ({100.0*real.size/d.size:.1f}%)")
    if real.size:
        print(f"  p1 / p50 / p99  {np.percentile(real,1):.0f} / "
              f"{np.percentile(real,50):.0f} / "
              f"{np.percentile(real,99):.0f} mm")
        print(f"  最近 / 最远     {real.min():.0f} / {real.max():.0f} mm")
        cy, cx = d.shape[0] // 2, d.shape[1] // 2
        cval = d[cy, cx]
        print(f"  画面中心        {cval} mm"
              + ("  (无效)" if cval in (0, 65535) else ""))
    print(f"\n  序列号 {dev.get_info(rs.camera_info.serial_number)}")
    print(f"  固件   {dev.get_info(rs.camera_info.firmware_version)}")
    print(f"  链路   {dev.get_info(rs.camera_info.usb_type_descriptor)}")

    pipe.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
