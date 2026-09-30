#!/usr/bin/env python3
"""D435i USB 接口占用探测 —— 定位 macOS UVC 驱动抢占了哪些接口。

原理
----
D435i 在 USB 上暴露 6 个接口：

    interface 0,1  = Depth（深度/红外）
    interface 2    = Y（红外 Y 通道）
    interface 3,4  = RGB（彩色）
    interface 5    = HID（IMU）

macOS 12+ 的 CoreMediaIO 体系（cameracaptured / VDCAssistant）会把
RGB 接口当成系统 UVC 摄像头绑定，并注册给 AVFoundation。
这样 librealsense 的 libusb 后端就 claim 不到这些接口，
报 `failed to claim usb interface: N, error: RS2_USB_STATUS_ACCESS`。

本脚本用 libusb 逐个探测：
  - kernel_driver_active：系统驱动是否已绑定该接口
  - detach_kernel_driver：能否把它踢掉（需要 root）
  - claim_interface：最终能否独占

用法（需要 sudo）
-----------------
    sudo -E /opt/anaconda3/envs/realsense/bin/python3 rs_iface_probe.py
    sudo -E .../rs_iface_probe.py --detach    # 尝试踢掉系统驱动后 claim
"""

import argparse
import ctypes
import ctypes.util
import os
import sys

INTEL_VID = 0x8086
D435I_PIDS = (0x0B3A, 0x0B07, 0x0AD3, 0x0AD2, 0x0AD1)

IFACE_NAMES = {
    0: "Depth  (深度/红外)",
    1: "Depth  (深度/红外)",
    2: "Y      (红外 Y)",
    3: "RGB    (彩色)",
    4: "RGB    (彩色)",
    5: "HID    (IMU)",
}

LIBUSB_SUCCESS = 0
LIBUSB_ERROR_NOT_FOUND = -5


def find_libusb():
    for c in (
        "/opt/anaconda3/envs/realsense/lib/libusb-1.0.0.dylib",
        "/opt/anaconda3/envs/realsense/lib/libusb-1.0.dylib",
        "/opt/homebrew/opt/libusb/lib/libusb-1.0.0.dylib",
        "/opt/homebrew/lib/libusb-1.0.dylib",
        ctypes.util.find_library("usb-1.0"),
    ):
        if c and os.path.exists(c):
            return c
    return None


class USB:
    def __init__(self, path):
        L = ctypes.CDLL(path)
        self.L = L
        L.libusb_init.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        L.libusb_init.restype = ctypes.c_int
        L.libusb_exit.argtypes = [ctypes.c_void_p]
        L.libusb_get_device_list.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))]
        L.libusb_get_device_list.restype = ctypes.c_ssize_t
        L.libusb_free_device_list.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        L.libusb_get_device_descriptor.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        L.libusb_get_device_descriptor.restype = ctypes.c_int
        L.libusb_get_bus_number.argtypes = [ctypes.c_void_p]
        L.libusb_get_bus_number.restype = ctypes.c_uint8
        L.libusb_get_device_address.argtypes = [ctypes.c_void_p]
        L.libusb_get_device_address.restype = ctypes.c_uint8
        L.libusb_open.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        L.libusb_open.restype = ctypes.c_int
        L.libusb_close.argtypes = [ctypes.c_void_p]
        L.libusb_kernel_driver_active.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.libusb_kernel_driver_active.restype = ctypes.c_int
        L.libusb_detach_kernel_driver.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.libusb_detach_kernel_driver.restype = ctypes.c_int
        L.libusb_attach_kernel_driver.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.libusb_attach_kernel_driver.restype = ctypes.c_int
        L.libusb_claim_interface.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.libusb_claim_interface.restype = ctypes.c_int
        L.libusb_release_interface.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.libusb_release_interface.restype = ctypes.c_int
        L.libusb_get_string_descriptor_ascii.argtypes = [
            ctypes.c_void_p, ctypes.c_uint8, ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int]
        L.libusb_get_string_descriptor_ascii.restype = ctypes.c_int
        L.libusb_strerror.argtypes = [ctypes.c_int]
        L.libusb_strerror.restype = ctypes.c_char_p

    def err(self, c):
        try:
            return self.L.libusb_strerror(c).decode()
        except Exception:
            return f"code {c}"


class DD(ctypes.Structure):
    _fields_ = [
        ("bLength", ctypes.c_uint8), ("bDescriptorType", ctypes.c_uint8),
        ("bcdUSB", ctypes.c_uint16), ("bDeviceClass", ctypes.c_uint8),
        ("bDeviceSubClass", ctypes.c_uint8), ("bDeviceProtocol", ctypes.c_uint8),
        ("bMaxPacketSize0", ctypes.c_uint8), ("idVendor", ctypes.c_uint16),
        ("idProduct", ctypes.c_uint16), ("bcdDevice", ctypes.c_uint16),
        ("iManufacturer", ctypes.c_uint8), ("iProduct", ctypes.c_uint8),
        ("iSerialNumber", ctypes.c_uint8), ("bNumConfigurations", ctypes.c_uint8),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detach", action="store_true",
                    help="尝试 detach 系统驱动后再 claim")
    ap.add_argument("--reattach", action="store_true",
                    help="测试完把驱动装回去（默认不装，让 librealsense 用）")
    args = ap.parse_args()

    libpath = find_libusb()
    if not libpath:
        sys.exit("找不到 libusb-1.0.dylib")
    print(f"libusb: {libpath}\n")

    if os.geteuid() != 0:
        sys.exit("需要 root：sudo -E /opt/anaconda3/envs/realsense/bin/python3 "
                 + os.path.basename(__file__))

    usb = USB(libpath)
    ctx = ctypes.c_void_p()
    if usb.L.libusb_init(ctypes.byref(ctx)) != LIBUSB_SUCCESS:
        sys.exit("libusb_init 失败")

    devs = ctypes.POINTER(ctypes.c_void_p)()
    n = usb.L.libusb_get_device_list(ctx, ctypes.byref(devs))
    if n < 0:
        sys.exit(f"libusb_get_device_list 失败: {usb.err(n)}")

    target = None
    for i in range(n):
        d = DD()
        if usb.L.libusb_get_device_descriptor(devs[i], ctypes.byref(d)) != 0:
            continue
        if d.idVendor == INTEL_VID and d.idProduct in D435I_PIDS:
            target = devs[i]
            bus = usb.L.libusb_get_bus_number(devs[i])
            addr = usb.L.libusb_get_device_address(devs[i])
            print(f"找到 D435i: bus={bus} addr={addr} "
                  f"PID=0x{d.idProduct:04x} USB=0x{d.bcdUSB:04x}\n")
            break
    usb.L.libusb_free_device_list(devs, 1)

    if target is None:
        usb.L.libusb_exit(ctx)
        sys.exit("libusb 没看到 D435i")

    h = ctypes.c_void_p()
    rc = usb.L.libusb_open(target, ctypes.byref(h))
    if rc != LIBUSB_SUCCESS:
        usb.L.libusb_exit(ctx)
        sys.exit(f"libusb_open 失败: {usb.err(rc)}（这是最关键的失败点）")

    print("=" * 70)
    print("接口占用情况")
    print("=" * 70)
    print(f"{'接口':<6}{'名称':<22}{'系统驱动':<10}{'可claim':<10}{'说明'}")
    print("-" * 70)

    claimed = []
    detach_failed = []
    results = {}

    for iface in range(6):
        name = IFACE_NAMES.get(iface, "?")

        # 1. 系统驱动是否绑定
        kd = usb.L.libusb_kernel_driver_active(h, iface)
        if kd == 1:
            kd_txt = "已绑定"
        elif kd == 0:
            kd_txt = "未绑定"
        else:
            kd_txt = f"err({kd})"

        # 2. 尝试 claim
        rc = usb.L.libusb_claim_interface(h, iface)
        if rc == LIBUSB_SUCCESS:
            claimed.append(iface)
            results[iface] = "ok"
            print(f"{iface:<6}{name:<22}{kd_txt:<10}{'✅ 是':<10}接口空闲")
            usb.L.libusb_release_interface(h, iface)
            continue

        # claim 失败，看是否因系统驱动
        note = usb.err(rc)
        if args.detach and kd == 1:
            drc = usb.L.libusb_detach_kernel_driver(h, iface)
            if drc == LIBUSB_SUCCESS:
                rc2 = usb.L.libusb_claim_interface(h, iface)
                if rc2 == LIBUSB_SUCCESS:
                    claimed.append(iface)
                    results[iface] = "detached_ok"
                    print(f"{iface:<6}{name:<22}{kd_txt:<10}{'✅ 是':<10}"
                          f"detach 系统驱动后成功")
                    usb.L.libusb_release_interface(h, iface)
                    if args.reattach:
                        pass  # 留着不装回
                    continue
                note = f"detach 后仍失败: {usb.err(rc2)}"
            else:
                note = f"detach 失败: {usb.err(drc)}"
                detach_failed.append(iface)

        results[iface] = "fail"
        print(f"{iface:<6}{name:<22}{kd_txt:<10}{'❌ 否':<10}{note}")

    print("-" * 70)
    print(f"\n可 claim 的接口: {claimed if claimed else '无'}")
    if detach_failed:
        print(f"detach 失败的接口: {detach_failed}")

    # 结论
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)

    rgb_ok = any(i in claimed for i in (3, 4))
    depth_ok = any(i in claimed for i in (0, 1))

    if depth_ok and rgb_ok:
        print("✅ 深度与彩色接口都可访问")
        print("   → 设备当前状态正常，可以直接跑 rs_stream_test.py")
    elif depth_ok and not rgb_ok:
        print("⚠️  深度可访问，**彩色不可访问**")
        print("   → 这是 macOS UVC 驱动占用 RGB 接口的典型表现")
        print("   → AVFoundation 已把相机注册为系统摄像头，CoreMediaIO 持有 RGB 接口")
        print("   → 纯视觉位移测量：深度可用，彩色需另想办法")
    elif not depth_ok and rgb_ok:
        print("⚠️  彩色可访问，深度不可访问")
    else:
        print("❌ 所有接口都拿不到")
        print("   → 设备被系统完全接管，或之前访问残留了坏状态")
        print("   → 尝试：物理拔插相机（不带扩展坞），然后立刻跑本脚本")

    usb.L.libusb_close(h)
    usb.L.libusb_exit(ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
