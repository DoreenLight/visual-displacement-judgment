#!/usr/bin/env python3
"""D435i 连通性自检。

用法（macOS 12+ 需要 sudo 才能访问 USB）：
    sudo -E /opt/anaconda3/envs/realsense/bin/python3 rs_check.py

依次检查：设备识别 -> USB 链路速率 -> 传感器 -> 每条流能否出帧。
不修改相机任何设置。

macOS 注意：本脚本会自动调用 macos_usb_fix 解绑抢占 USB 接口的系统 UVC 驱动。
"""

import os
import sys

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
    sys.exit("缺少 pyrealsense2。用 /opt/anaconda3/envs/realsense/bin/python3 运行本脚本。")


def line(title=""):
    print("\n" + "=" * 58)
    if title:
        print(title)
        print("=" * 58)


def main():
    line("1. 设备识别")
    ctx = rs.context()
    devs = list(ctx.devices)
    if not devs:
        print("❌ 没有检测到任何 RealSense 设备。")
        print("\n排查顺序：")
        print("  a. 相机 USB-C 线是否插紧（两端都插紧）")
        print("  b. 是否用了 USB 3 数据线（不是充电线）")
        print("  c. 是否直插 MacBook 雷雳口（不要走扩展坞/ Hub）")
        print("  d. 是否用 sudo 运行（macOS 12+ 必须）")
        print("  e. 相机侧面指示灯是否亮")
        return 1

    print(f"✅ 检测到 {len(devs)} 台设备\n")
    for i, d in enumerate(devs):
        print(f"--- 设备 {i} ---")
        for key in (
            "name", "serial_number", "firmware_version",
            "product_line", "usb_type_descriptor", "physical_port",
        ):
            try:
                print(f"  {key:22} = {d.get_info(getattr(rs.camera_info, key))}")
            except Exception:
                print(f"  {key:22} = <不可用>")

        usb = d.get_info(rs.camera_info.usb_type_descriptor)
        if "3" in usb:
            print(f"  ✅ USB 链路正常 ({usb})")
        else:
            print(f"  ⚠️  USB 链路降级为 {usb}（应为 3.x）")
            print("     → 深度/彩色帧率会受限，请换 USB 3 线 + 直插机身口")

    line("2. 传感器清单")
    dev = devs[0]
    sensors = list(dev.query_sensors())
    print(f"共 {len(sensors)} 个传感器：")
    for s in sensors:
        try:
            name = s.get_info(rs.camera_info.name)
        except Exception:
            name = "<未命名>"
        cams = []
        for p in s.get_stream_profiles():
            cams.append(str(p.stream_type()).replace("stream_type.", ""))
        uniq = sorted(set(cams))
        print(f"  - {name:28} 提供: {', '.join(uniq)}")

    line("3. 逐流出帧测试（各抓 30 帧）")
    print("  注意：macOS 上请优先用 rs_stream_test.py（单 pipeline 多流）。")
    print("  本段为兼容性保留，会反复 start/stop pipeline。\n")
    tests = [
        ("深度 Depth 848x480@30", rs.stream.depth, 848, 480, rs.format.z16, 30),
        ("彩色 Color 640x480@30", rs.stream.color, 640, 480, rs.format.bgr8, 30),
        ("红外 IR1   848x480@30", rs.stream.infrared, 848, 480, rs.format.y8, 30),
    ]

    results = {}
    for label, stream, w, h, fmt, fps in tests:
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(stream, w, h, fmt, fps)
        n, err, first = 0, None, None
        try:
            pipe.start(cfg)
            for _ in range(30):
                frames = pipe.wait_for_frames(timeout_ms=5000)
                f = frames.first_or_default(stream)
                if f is None:
                    raise RuntimeError("该流无帧返回")
                if first is None:
                    first = f
                n += 1
            results[label] = True
            print(f"  ✅ {label}  → {n} 帧")
        except Exception as e:
            err = e
            results[label] = False
            print(f"  ❌ {label}  → {err}")
        finally:
            try:
                pipe.stop()
            except Exception:
                pass

        if first is not None:
            prof = first.get_profile().as_video_stream_profile()
            intr = prof.get_intrinsics()
            print(f"     内参: fx={intr.fx:.2f} fy={intr.fy:.2f} "
                  f"ppx={intr.ppx:.2f} ppy={intr.ppy:.2f} "
                  f"畸变={intr.model}")

    line("4. IMU（D435i 独有）")
    imu_ok = False
    pipe = rs.pipeline()
    cfg = rs.config()
    try:
        cfg.enable_stream(rs.stream.accel)
        cfg.enable_stream(rs.stream.gyro)
        pipe.start(cfg)
        for _ in range(20):
            frames = pipe.wait_for_frames(timeout_ms=5000)
            if frames.first_or_default(rs.stream.accel) is not None:
                imu_ok = True
                break
        if imu_ok:
            print("  ✅ IMU 可以出数据（加速度计 + 陀螺仪）")
        else:
            print("  ⚠️  开了流但没有数据")
    except Exception as e:
        print(f"  ⚠️  IMU 不可用: {e}")
        print("     → macOS 上 IMU 官方不支持，这是预期行为，不影响深度/彩色")
    finally:
        try:
            pipe.stop()
        except Exception:
            pass

    line("结论")
    core_ok = results.get("深度 Depth 848x480@30") and results.get("彩色 Color 640x480@30")
    if core_ok:
        print("✅ 深度 + 彩色全部可用 —— 相机可以正常用了")
        print(f"   序列号 {dev.get_info(rs.camera_info.serial_number)} "
              f"| 固件 {dev.get_info(rs.camera_info.firmware_version)} "
              f"| 链路 {dev.get_info(rs.camera_info.usb_type_descriptor)}")
    else:
        print("❌ 核心流有问题，看上面 ❌ 那几行")
    print(f"\nIMU: {'可用' if imu_ok else '不可用（macOS 预期）'}")
    return 0 if core_ok else 2


if __name__ == "__main__":
    sys.exit(main())
