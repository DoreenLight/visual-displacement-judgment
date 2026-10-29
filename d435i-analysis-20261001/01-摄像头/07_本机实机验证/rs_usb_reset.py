#!/usr/bin/env python3
"""D435i USB 复位工具 —— 解决 macOS 上 `failed to set power state`。

背景
----
macOS 12+ 的系统 UVC 驱动（CoreMediaIO / VDCAssistant / cameracaptured）
会抢占 D435i 的 interface 0（bInterfaceClass=14，视频类），
导致 librealsense 的 libusb 后端拿不到接口，报：

    failed to claim usb interface: 0, error: RS2_USB_STATUS_ACCESS
    acquire_power failed: failed to set power state

而且驱动会残留状态，导致**第一次调用能成功、之后全部失败**。
本脚本用 libusb 对设备做一次 USB 端口复位（不重新枚举），
把被 UVC 驱动占用的接口释放掉。

用法（需要 sudo）
-----------------
    sudo -E /opt/anaconda3/envs/realsense/bin/python3 rs_usb_reset.py

    # 复位后立刻跑自检
    sudo -E /opt/anaconda3/envs/realsense/bin/python3 rs_usb_reset.py --then-check

    # 只看状态，不复位
    sudo -E /opt/anaconda3/envs/realsense/bin/python3 rs_usb_reset.py --dry-run
"""

import argparse
import ctypes
import ctypes.util
import os
import subprocess
import sys
import time

# RealSense 厂商 ID
INTEL_VID = 0x8086
# D435i 的 PID（USB 3 模式）
D435I_PIDS = (0x0B3A, 0x0B07, 0x0AD3, 0x0AD2, 0x0AD1)

# libusb 常量
LIBUSB_SUCCESS = 0
LIBUSB_CLASS_VIDEO = 14


def find_libusb():
    """找到可用的 libusb-1.0.dylib。"""
    candidates = [
        "/opt/anaconda3/envs/realsense/lib/libusb-1.0.0.dylib",
        "/opt/anaconda3/envs/realsense/lib/libusb-1.0.dylib",
        "/opt/homebrew/opt/libusb/lib/libusb-1.0.0.dylib",
        "/opt/homebrew/lib/libusb-1.0.dylib",
        "/usr/local/lib/libusb-1.0.dylib",
        ctypes.util.find_library("usb-1.0"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


class LibUSB:
    def __init__(self, path):
        self.lib = ctypes.CDLL(path)
        L = self.lib
        L.libusb_init.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        L.libusb_init.restype = ctypes.c_int
        L.libusb_exit.argtypes = [ctypes.c_void_p]
        L.libusb_get_device_list.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)),
        ]
        L.libusb_get_device_list.restype = ctypes.c_ssize_t
        L.libusb_free_device_list.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        L.libusb_get_device_descriptor.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p]
        L.libusb_get_device_descriptor.restype = ctypes.c_int
        L.libusb_get_bus_number.argtypes = [ctypes.c_void_p]
        L.libusb_get_bus_number.restype = ctypes.c_uint8
        L.libusb_get_device_address.argtypes = [ctypes.c_void_p]
        L.libusb_get_device_address.restype = ctypes.c_uint8
        L.libusb_get_port_numbers.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int]
        L.libusb_get_port_numbers.restype = ctypes.c_int
        L.libusb_open.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        L.libusb_open.restype = ctypes.c_int
        L.libusb_close.argtypes = [ctypes.c_void_p]
        L.libusb_reset_device.argtypes = [ctypes.c_void_p]
        L.libusb_reset_device.restype = ctypes.c_int
        L.libusb_error_name.argtypes = [ctypes.c_int]
        L.libusb_error_name.restype = ctypes.c_char_p
        L.libusb_strerror.argtypes = [ctypes.c_int]
        L.libusb_strerror.restype = ctypes.c_char_p

    def err(self, code):
        try:
            return self.lib.libusb_strerror(code).decode()
        except Exception:
            return f"code {code}"


class DeviceDescriptor(ctypes.Structure):
    _fields_ = [
        ("bLength", ctypes.c_uint8),
        ("bDescriptorType", ctypes.c_uint8),
        ("bcdUSB", ctypes.c_uint16),
        ("bDeviceClass", ctypes.c_uint8),
        ("bDeviceSubClass", ctypes.c_uint8),
        ("bDeviceProtocol", ctypes.c_uint8),
        ("bMaxPacketSize0", ctypes.c_uint8),
        ("idVendor", ctypes.c_uint16),
        ("idProduct", ctypes.c_uint16),
        ("bcdDevice", ctypes.c_uint16),
        ("iManufacturer", ctypes.c_uint8),
        ("iProduct", ctypes.c_uint8),
        ("iSerialNumber", ctypes.c_uint8),
        ("bNumConfigurations", ctypes.c_uint8),
    ]


def check_root():
    if os.geteuid() != 0:
        print("❌ 需要 root 权限（macOS 上必须）。请用：")
        print("   sudo -E /opt/anaconda3/envs/realsense/bin/python3 "
              + os.path.basename(__file__))
        return False
    return True


def enumerate_realsense(usb):
    """返回 [(handle, desc, bus, addr, ports), ...]"""
    found = []
    ctx = ctypes.c_void_p()
    rc = usb.lib.libusb_init(ctypes.byref(ctx))
    if rc != LIBUSB_SUCCESS:
        print(f"❌ libusb_init 失败: {usb.err(rc)}")
        return None, None
    try:
        devs = ctypes.POINTER(ctypes.c_void_p)()
        n = usb.lib.libusb_get_device_list(ctx, ctypes.byref(devs))
        if n < 0:
            print(f"❌ libusb_get_device_list 失败: {usb.err(n)}")
            return None, ctx
        try:
            for i in range(n):
                dev = devs[i]
                desc = DeviceDescriptor()
                if usb.lib.libusb_get_device_descriptor(
                        dev, ctypes.byref(desc)) != LIBUSB_SUCCESS:
                    continue
                if desc.idVendor != INTEL_VID:
                    continue
                if desc.idProduct not in D435I_PIDS:
                    continue
                bus = usb.lib.libusb_get_bus_number(dev)
                addr = usb.lib.libusb_get_device_address(dev)
                pbuf = (ctypes.c_uint8 * 8)()
                nport = usb.lib.libusb_get_port_numbers(dev, pbuf, 8)
                ports = [pbuf[j] for j in range(nport)] if nport > 0 else []
                found.append((dev, desc, bus, addr, ports))
        finally:
            usb.lib.libusb_free_device_list(devs, 1)
    finally:
        pass
    return found, ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只看状态，不复位")
    ap.add_argument("--then-check", action="store_true",
                    help="复位后立刻运行 rs_check.py")
    ap.add_argument("--rounds", type=int, default=3, help="最多复位轮数")
    args = ap.parse_args()

    libpath = find_libusb()
    if not libpath:
        print("❌ 找不到 libusb-1.0.dylib")
        return 1
    print(f"libusb: {libpath}")

    if not check_root():
        return 1

    usb = LibUSB(libpath)

    found, ctx = enumerate_realsense(usb)
    if found is None:
        return 1
    if not found:
        print("❌ libusb 没看到任何 D435i。")
        print("   检查相机是否插好，或先跑 rs_check.py 看系统层识别情况。")
        return 1

    print(f"✅ libusb 看到 {len(found)} 个 RealSense 设备：\n")
    for i, (dev, d, bus, addr, ports) in enumerate(found):
        portstr = ".".join(str(p) for p in ports) or "?"
        print(f"  [{i}] VID=0x{d.idVendor:04x} PID=0x{d.idProduct:04x} "
              f"bus={bus} addr={addr} port={portstr} "
              f"USB={'3.x' if d.bcdUSB >= 0x0300 else '2.x'}(0x{d.bcdUSB:04x})")

    if args.dry_run:
        print("\n(--dry-run，不做复位)")
        usb.lib.libusb_exit(ctx)
        return 0

    # 逐个复位
    print()
    ok_any = False
    for rounds in range(1, args.rounds + 1):
        all_ok = True
        for i, (dev, d, bus, addr, ports) in enumerate(found):
            h = ctypes.c_void_p()
            rc = usb.lib.libusb_open(dev, ctypes.byref(h))
            if rc != LIBUSB_SUCCESS:
                print(f"  第{rounds}轮 [{i}] libusb_open 失败: {usb.err(rc)}")
                all_ok = False
                continue
            try:
                rc = usb.lib.libusb_reset_device(h)
                if rc == LIBUSB_SUCCESS:
                    print(f"  第{rounds}轮 [{i}] ✅ 复位成功 (bus={bus} addr={addr})")
                    ok_any = True
                elif rc == -4:  # LIBUSB_ERROR_NOT_FOUND
                    print(f"  第{rounds}轮 [{i}] ✅ 复位触发（设备已重新枚举）")
                    ok_any = True
                else:
                    print(f"  第{rounds}轮 [{i}] ❌ 复位失败: {usb.err(rc)}")
                    all_ok = False
            finally:
                try:
                    usb.lib.libusb_close(h)
                except Exception:
                    pass

        time.sleep(2.0)
        if all_ok:
            print(f"\n第 {rounds} 轮全部复位成功。")
            break
        print(f"  等待 2 秒后重试…（libusb 看到的设备列表需要刷新）")
        # 重新枚举：复位后设备可能换了地址
        try:
            found, ctx2 = enumerate_realsense(usb)
            if ctx2:
                usb.lib.libusb_exit(ctx2)
            if not found:
                print("  ⚠️  重新枚举后看不到设备了（可能正在重新枚举），等待后重试")
                time.sleep(3)
                found, ctx2 = enumerate_realsense(usb)
                if ctx2:
                    usb.lib.libusb_exit(ctx2)
                if not found:
                    print("  ❌ 设备没回来")
                    break
        except Exception as e:
            print(f"  重新枚举出错: {e}")
            break

    usb.lib.libusb_exit(ctx)

    if not ok_any:
        print("\n❌ 所有复位尝试都失败。")
        print("   → 这是 macOS 上 UVC 驱动抢占接口的已知问题，见 issue #9718 / #14325。")
        return 2

    # 等一下让系统重新枚举，然后验证
    print("\n等待 3 秒让系统重新枚举…")
    time.sleep(3)

    print("\n--- 复位后验证 ---")
    out = subprocess.run(
        ["/opt/homebrew/bin/rs-enumerate-devices", "-s"],
        capture_output=True, text=True)
    combined = (out.stdout or "") + (out.stderr or "")
    if "failed to set power" in combined or "failed to claim" in combined:
        print("⚠️  仍然报 power state / claim 错误：")
        for ln in combined.splitlines():
            if "ERROR" in ln or "failed" in ln.lower():
                print("   ", ln.strip()[:150])
        print("\n   → 复位没能解决。可能是 macOS 版本兼容性问题。")
        return 3

    print("✅ librealsense 可以访问设备了！")
    print(combined[:800])

    if args.then_check:
        print("\n--- 运行完整自检 ---")
        here = os.path.dirname(os.path.abspath(__file__))
        subprocess.run(
            [sys.executable, os.path.join(here, "rs_check.py")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
