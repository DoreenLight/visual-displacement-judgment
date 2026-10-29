# macOS D435i 连通问题与修复（实测）

> 本机环境：MacBook Air (Apple M5, arm64) / macOS 27.0 (Build 26A428)
> 实测相机：Intel RealSense D435i，序列号 `244222076984`，固件 `5.15.1.55`
> 记录时间：2025-09-22
> **结论：已完全跑通。深度 + 彩色 + 红外三流并发 32fps，15 秒 100% 无丢帧。**

本文是 `macOS环境与验证.md` 的续篇，专门记录**插上相机之后**遇到的问题与完整解法。

---

## 1. 最终状态（实测证据）

```
name                 = RealSense D435I
serial_number        = 244222076984
firmware_version     = 5.15.1.55
usb_type_descriptor  = 3.2          ← USB 3.2 链路，正常
product_line         = D400

depth  848x480 @ 30  fx=428.0 fy=428.0 ppx=425.8 ppy=236.8  brown_conrady
color  640x480 @ 30  fx=606.7 fy=606.5 ppx=327.3 ppy=259.0  inverse_brown_conrady
ir     848x480 @ 30  fx=428.0 fy=428.0 ppx=425.8 ppy=236.8  brown_conrady
```

连续采集验证：

```
连续采集 15.0s，共 480 个帧集
实测帧率: 32.0 fps（帧集）
  ✅ depth    450/450 (100.0%)
  ✅ color    450/450 (100.0%)
```

深度单位实测 **正好 1.0 mm**：

```
depth_scale = 0.0010000000474974513   → 1 原始单位 = 1.0000 mm
```

> 所以深度 `.npy` 里的整数**直接就是毫米值**，无需换算。

---

## 2. 核心问题：macOS 系统 UVC 驱动抢占 USB 接口

### 2.1 现象

相机会被 macOS 正确识别（`ioreg` 能看到，链路 5 Gb/s），LLM 也能看到设备名，
但 librealsense 报错：

```
failed to claim usb interface: 0, error: RS2_USB_STATUS_ACCESS
acquire_power failed: failed to set power state
Could not create device - failed to set power state
No device detected. Is it plugged in?
```

**关键特征：间歇性。** 第一次调用可能成功，之后持续失败：

```
$ rs-enumerate-devices -s          # 第 1 次：✅ 成功
RealSense D435I  244222076984  5.15.1.55
ERROR [dispatcher.cpp:34] Dispatcher exception caught: mutex lock failed

$ rs-enumerate-devices -s          # 第 2~6 次：❌ 全部失败
failed to claim usb interface: 0, error: RS2_USB_STATUS_ACCESS
```

### 2.2 根因（用 `rs_iface_probe.py` 实测确认）

D435i 在 USB 上暴露 6 个接口：

| 接口 | 用途 | 类型 |
|---|---|---|
| 0, 1 | Depth（深度/红外） | UVC 视频类 |
| 2 | Y（红外 Y 通道） | UVC 视频类 |
| 3, 4 | RGB（彩色） | UVC 视频类 |
| 5 | HID（IMU） | HID |

macOS 12+ 的 **CoreMediaIO** 体系（`cameracaptured` / `VDCAssistant`）
把 UVC 视频类接口视为系统摄像头并绑定驱动，还注册给 AVFoundation：

```
$ system_profiler SPCameraDataType
    Intel(R) RealSense(TM) Depth Camera 435i RGB:
      Model ID: UVC Camera VendorID_32902 ProductID_2874    ← 被当成系统摄像头
```

实测接口占用情况：

```
接口    名称                    系统驱动      可claim    说明
----------------------------------------------------------------------
0     Depth  (深度/红外)        已绑定       ✅ 是       detach 系统驱动后成功   ← 元凶
1     Depth  (深度/红外)        未绑定       ✅ 是       接口空闲
2     Y      (红外 Y)         未绑定       ✅ 是       接口空闲
3     RGB    (彩色)           未绑定       ✅ 是       接口空闲
4     RGB    (彩色)           未绑定       ✅ 是       接口空闲
5     HID    (IMU)          未绑定       ✅ 是       接口空闲
```

**interface 0 被系统驱动绑定**，libusb 抢不到它 → `RS2_USB_STATUS_ACCESS`。
由于 interface 0 是深度/红外流的入口，这直接导致整个设备创建失败。

### 2.3 解法

在 `import pyrealsense2` **之前**，用 `libusb_detach_kernel_driver()`
把系统驱动从接口上解绑：

```python
import macos_usb_fix
macos_usb_fix.ensure_ready()      # 必须在 import pyrealsense2 之前
import pyrealsense2 as rs
```

解绑后效果（同一台机器，同一根线，同一端口）：

```
可 claim 的接口: [0, 1, 2, 3, 4, 5]        ← 全部 6 个
✅ 管线启动成功（耗时 0.07s）
✅ depth  ✅ color  ✅ ir
```

**这是唯一有效的解法。** 以下方法实测**都无效**：

| 尝试 | 结果 |
|---|---|
| 反复重试（10 轮，每轮间隔 1.5s） | ❌ 全部失败 |
| `libusb_reset_device()` USB 端口复位 | ⚠️ 能短暂恢复，但 `pipeline.start` 随即再失败 |
| 只靠 `sudo` | ❌ sudo 是必需条件但不充分 |
| 换直插雷雳口（不走扩展坞） | ❌ 已是直插（独立 XHCI 控制器），仍失败 |
| 加 `--reset` 后再跑 | ❌ 复位不解决接口被绑定的问题 |

---

## 3. 其他实测结论

### 3.1 `sudo` 无法免除

解绑驱动后，**非 root 仍然看不到设备**：

```
$ python3 -c "import pyrealsense2 as rs; print(len(list(rs.context().devices)))"
非 root 设备数: 0
```

这是 macOS 12+ 的硬限制（官方文档明确说明），与驱动解绑无关。
**所有访问相机的脚本都必须 `sudo -E`。**

### 3.2 不要反复 start/stop pipeline

macOS 上设备状态脆弱，早期的 `rs_check.py` 为每条流各建一个 pipeline
（start/stop 三次），结果：

```
✅ 深度 Depth 848x480@30  → 30 帧
❌ 彩色 Color 640x480@30  → Frame didn't arrive within 5000
❌ 红外 IR1   848x480@30  → No device connected      ← 设备"消失"了
```

**正确做法是用一条 pipeline 同时开多条流**（`rs_stream_test.py`）。
修复后三流并发完全稳定。

### 3.3 有两个"序列号"，含义不同

```
ioreg 里的 USB Serial Number : 254843063163
SDK 报的 serial_number      : 244222076984
```

两者不一致是正常现象：USB 描述符里的序列号由 USB 控制器固件提供，
SDK 报的是相机模组自身的校准序列号（**标定时要用 SDK 这个**）。

### 3.4 深度值有三种状态，统计时必须区分

初版脚本只统计 `d > 0`，得出"有效 33.6%、中位数 339mm"，
但生成的可视化图几乎全黑。分析后发现是**可视化归一化 bug**：

```
总数        307200
  ==0        203995 ( 66.4%)   无读数（无效）
  ==65535       141 (  0.0%)   离群/无效
  正常值      103064 ( 33.5%)

有效值分位：p0=174  p50=339  p95=9644  p99=11605  p100=20139 mm
```

用 `min-max(174~20139)` 归一化时，339mm 被映射到 **0.6/255 ≈ 全黑**。

**正确做法**：用分位数定区间 + 固定上限（本机取 3m）：

```python
valid = d[(d > 0) & (d < 65535)]
dmin = int(np.percentile(valid, 1))
dmax = min(int(np.percentile(valid, 99)), 3000)   # ≤3m
```

修正后深度图能正常看出结构。

> **做位移测量时必须用 `_depth.npy`**（uint16 原始毫米值），
> 不能用 `_depth.png`（归一化 + 8 位量化，精度已丢失）。

### 3.5 深度/彩色对齐可用

D435i 的深度与彩色有轻微基线偏移，`rs.align` 可把深度投到彩色坐标系：

```
未对齐:  depth (480,640), 有效 104073
align 后: depth (480,640), 有效 120584     ← 有效像素更多
深度->彩色 外参平移: [0.015, -0.0, 0.0] mm
```

做视觉位移测量时，**如果要叠加彩色纹理，必须先 align**。

### 3.6 IMU 确认不可用（符合官方说明）

```
⚠️  IMU 不可用: No device connected
   → macOS 上 IMU 官方不支持，这是预期行为
```

接口 5（HID）在 libusb 层是空闲可 claim 的，但 SDK 主动禁用了 macOS 上的运动传感器。
本机为纯视觉位移用途，不影响。

### 3.7 USB 3.2 链路正常，硬件条件达标

```
bus=1 addr=2 port=2  USB=3.x(0x0320)
UsbLinkSpeed = 5000000000        (5 Gb/s)
物理端口: 1-2-2
拓扑: 直插 AppleT8142USBXHCI@01000000 独立控制器（不在 USB Hub 下）
```

---

## 4. 文件清单

| 文件 | 用途 |
|---|---|
| **`macos_usb_fix.py`** | **核心修复模块。** 解绑系统 UVC 驱动。其他脚本 `import` 它即可自动生效 |
| `rs_iface_probe.py` | 诊断工具。逐个探测 6 个接口的占用状态，定位被抢占的接口 |
| `rs_stream_test.py` | **推荐的主测试脚本。** 单 pipeline 多流并发，支持长期采集与丢帧统计 |
| `rs_usb_reset.py` | USB 端口复位（备选，效果不如 `macos_usb_fix`） |
| `rs_check.py` | 兼容性自检（会反复 start/stop pipeline，macOS 上不推荐） |
| `rs_capture.py` | 采集一帧并落盘（深度/彩色/红外 + 统计） |
| `captures/` | 实测样例数据 |

### 4.1 命令速查

```bash
cd /Users/xbpd/Projects/视觉位移判断/摄像头/07_本机实机验证

# ① 诊断：看哪个接口被系统驱动占了
osascript -e 'do shell script "/opt/anaconda3/envs/realsense/bin/python3 \
  '"$PWD"'/rs_iface_probe.py --detach > /tmp/i.txt 2>&1; echo done" \
  with administrator privileges'; cat /tmp/i.txt

# ② 主测试：三流并发（自动修 USB）
sudo -E /opt/anaconda3/envs/realsense/bin/python3 rs_stream_test.py

# ③ 稳定性：连续采集 15 秒并统计丢帧
sudo -E /opt/anaconda3/envs/realsense/bin/python3 rs_stream_test.py \
    --seconds 15 --streams depth,color

# ④ 采集一帧落盘
sudo -E /opt/anaconda3/envs/realsense/bin/python3 rs_capture.py --out captures/frame01
```

> macOS 终端里 `sudo` 会要密码。若在非交互环境（如自动化脚本）中调用，
> 可用 `osascript -e 'do shell script "..." with administrator privileges'`
> 弹出系统授权框替代。

### 4.2 在自己的代码里集成

```python
import sys
sys.path.insert(0, "/Users/xbpd/Projects/视觉位移判断/摄像头/07_本机实机验证")
import macos_usb_fix
macos_usb_fix.ensure_ready()          # 必须在下面这行之前

import pyrealsense2 as rs
# ... 正常使用 pyrealsense2
```

---

## 5. 恢复系统默认绑定

解绑后，**其他 App 无法再把 D435i 当普通摄像头用**。
恢复方法（任选其一）：

```bash
# 方法一：脚本装回驱动
sudo -E /opt/anaconda3/envs/realsense/bin/python3 macos_usb_fix.py --reattach

# 方法二：物理拔插相机（最简单，系统会自动重新绑定）
```

---

## 6. 对「纯视觉位移测量」的可用性判断

| 能力 | 状态 | 说明 |
|---|---|---|
| 深度流 848x480@30 | ✅ 可用 | 单位 1mm，1.00 精确 |
| 彩色流 640x480@30 | ✅ 可用 | 需 align 才能与深度像素对应 |
| 红外流 848x480@30 | ✅ 可用 | 能看到投射器点阵，结构光正常 |
| 三流并发 | ✅ 稳定 | 32fps，15s 100% 无丢帧 |
| 深度-彩色对齐 | ✅ 可用 | `rs.align(rs.stream.color)` |
| 帧时间戳 | ✅ 可用 | `frame.get_timestamp()` |
| 多相机同步 | ❌ 未验证 | macOS 上硬件同步未测试 |
| IMU | ❌ 不可用 | macOS 官方禁用 |
| VIO / SLAM | ❌ 不可行 | 依赖 IMU，需换 Ubuntu |

**结论：纯视觉位移测量的硬件与软件条件已完全具备。**

---

## 7. 参考

- [RealSense 官方 macOS 安装文档](https://dev.realsenseai.com/installation/macos-installation-for-realsense-sdk/) — sudo 要求、Viewer 与 IMU 限制
- [librealsense #9718](https://github.com/realsenseai/librealsense/issues/9718) — `failed to set power state` 完全相同报错
- [librealsense #11815](https://github.com/realsenseai/librealsense/issues/11815) — macOS Monterey+ 需要 sudo
- [librealsense #14325](https://github.com/realsenseai/librealsense/issues/14325) — "Works when running as root, but unstable"
- [librealsense #13903](https://github.com/realsenseai/librealsense/issues/13903) — USB 2.1 识别问题
- [librealsense #14648](https://github.com/realsenseai/librealsense/issues/14648) — macOS 上 IMU 相关问题

---

## 8. 相关文档

- [macOS 环境与验证](macOS环境与验证.md) — 安装过程、平台限制、PyPI 踩坑
- [SDK 安装](../02_SDK与编程/SDK安装.md) — 通用安装方式
- [Python API](../02_SDK与编程/Python_API.md) — pyrealsense2 接口用法
