# macOS 实机验证记录

> 本机环境：MacBook Air (Apple M5, arm64) / macOS 27.0 (Build 26A428)
> 记录时间：2025-09-22
> 目标场景：**纯视觉位移测量**（不需要 IMU）

本目录记录 D435i 在**本机 macOS** 上的真实安装与验证结果，与 `02_SDK与编程/SDK安装.md` 的通用说明互补。
所有结论均为本机实测，不是文档转述。

---

## 1. 环境安装结果（已实测）

### 1.1 SDK：Homebrew 有现成 bottle，不需要源码编译

```bash
brew install librealsense
```

实测结果：

| 项 | 值 |
|---|---|
| 版本 | 2.58.4 |
| 安装路径 | `/opt/homebrew/Cellar/librealsense/2.58.4` |
| bottle | `arm64_golden_gate`（Apple Silicon 原生，无需编译） |
| 依赖 | glfw 3.5.1、libusb 1.0.30 |
| 体积 | 97 文件 / 41.3 MB |
| 耗时 | 约 2 分钟（纯下载解包） |

提供 **37 个命令行工具**，位于 `/opt/homebrew/bin/`：

```
rs-enumerate-devices   rs-capture      rs-depth        rs-color
rs-infrared            rs-measure      rs-pointcloud   rs-distance
rs-align               rs-align-advanced               rs-convert
rs-fw-update           rs-fw-logger    rs-motion       rs-multicam
rs-record              rs-record-playback              rs-save-to-disk
rs-benchmark           rs-depth-quality                rs-hdr
rs-embedded-filters    rs-post-processing              rs-on-chip-calib
rs-object-detection    rs-labeled-pointcloud           rs-data-collect
rs-sensor-control      rs-terminal     rs-gpu-frame    rs-embed
rs-eth-config          rs-callback     rs-hello-realsense
rs-software-device     rs-rosbag-inspector
```

> ⚠️ **修正**：`02_SDK与编程/SDK安装.md` 中把 macOS 标为「源码编译（仅旧版）⭐⭐⭐」，
> 这与实测不符。Apple Silicon 上有官方 bottle，一条命令装完。
> 源码编译反而是**不必要**的（除非需要 IMU 相关的自定义补丁）。

### 1.2 Python 绑定：走 conda-forge，不要走 PyPI

```bash
conda create -y -n realsense -c conda-forge --override-channels \
    python=3.12 pyrealsense2
conda install -y -n realsense -c conda-forge --override-channels \
    opencv pillow scipy matplotlib
```

实测结果：

| 包 | 版本 | 备注 |
|---|---|---|
| pyrealsense2 | 2.58.3 | conda-forge `osx-arm64` 原生构建 |
| python | 3.12 | |
| opencv | 5.0.0 | |
| pillow | 12.3.0 | |
| scipy | 1.18.1 | |
| matplotlib | 3.11.2 | |

环境路径：`/opt/anaconda3/envs/realsense`

> ⚠️ **关键坑**：**PyPI 上的 `pyrealsense2` 没有 macOS wheel**。
> 查 PyPI JSON 确认，最新版 2.58.4.10922 只提供：
> - `manylinux2014_aarch64` / `manylinux1_x86_64`（Linux）
> - `win_amd64`（Windows）
>
> **没有 `macosx_*` 条目**，`pip install pyrealsense2` 在 macOS 上一定失败。
> **必须走 conda-forge**，它是唯一提供 osx-arm64 原生构建的渠道。
> 直接用 `pip install pyrealsense2` 会浪费大量时间，这是本机验证过的踩坑点。

> ⚠️ **第二个坑**：安装时**不要**把 `opencv` 和 `pyrealsense2` 写在同一条
> `conda create` 命令里。opencv 会拖入 qt6 / ffmpeg / sdl2 / x264 / x265 / svt-av1
> 等近百个包，依赖求解会卡住数分钟甚至超时。
> 正确做法：先 `conda create ... pyrealsense2`，装完再单独装 opencv。

---

## 2. macOS 平台的两条硬限制（官方明确）

来源：[RealSense 官方 macOS 安装文档](https://dev.realsenseai.com/installation/macos-installation-for-realsense-sdk/)

> **Current Limitations:**
> - **RealSense Viewer is not supported** on macOS in the current release
> - **Motion sensors (IMU) are disabled** on macOS in the current release

### 2.1 没有 realsense-viewer

Homebrew 的 librealsense 只安装 37 个 `rs-*` 命令行工具，**不含 GUI viewer**。
实测确认 `/opt/homebrew/Cellar/librealsense/2.58.4/bin/` 下没有 `realsense-viewer`。

替代方案：用 Python + OpenCV 自己写显示窗口（见 `rs_capture.py`）。

### 2.2 IMU 不能用 —— 对 D435i 是重大功能损失

D435i 相比 D435 多出来的**唯一硬件差异就是 Bosch BMI055 六轴 IMU**。
在 macOS 上这个 IMU **被 SDK 主动禁用**，意味着：

- ❌ 不能做视觉惯性里程计（VIO）
- ❌ 不能做 IMU 辅助的 SLAM
- ❌ 不能跑 `rs-motion`
- ❌ `05_校准/IMU校准.md` 的全部内容在本机**无法验证**

**Linux 上 IMU 正常。** 如果后续需要 VIO/SLAM，必须换 Ubuntu 机器。

### 2.3 需要 sudo

macOS 12+ 由于 USB 安全策略变更 + 需要覆盖默认 UVC 驱动，
几乎所有用 libusb 的 librealsense 工具**必须提权运行**：

```bash
sudo -E /opt/anaconda3/envs/realsense/bin/python3 rs_check.py
sudo rs-enumerate-devices
```

`-E` 保留环境变量（否则 conda 环境的库路径会丢）。

---

## 3. 自检脚本

### 3.1 `rs_check.py` —— 连通性自检

依次检查：设备识别 → USB 链路速率 → 传感器清单 → 深度/彩色/红外逐流出帧 → IMU。

```bash
cd /Users/xbpd/Projects/视觉位移判断
sudo -E /opt/anaconda3/envs/realsense/bin/python3 07_本机实机验证/rs_check.py
```

关键检查项：**USB 链路必须是 3.x**。
若显示 `2.1` 说明线材或接口不达标，深度帧率和分辨率会受限。
`02_SDK与编程/SDK安装.md` 里提到的 USB 2.1 问题（对应 librealsense issue #13903）就是这个现象。

### 3.2 `rs_capture.py` —— 采集一帧并落盘

```bash
sudo -E /opt/anaconda3/envs/realsense/bin/python3 07_本机实机验证/rs_capture.py --out captures/frame01
```

产出 4 个文件：

| 文件 | 用途 |
|---|---|
| `frame01_color.png` | 彩色图 |
| `frame01_depth.npy` | **深度原始数据（毫米，uint16）** |
| `frame01_depth.png` | 深度伪彩色，仅供肉眼看 |
| `frame01_ir.png` | 左红外图 |

> ⚠️ **做位移测量必须用 `_depth.npy`**，不能用 `_depth.png`。
> 伪彩色经过归一化 + 8 位量化，**已经丢失了原始毫米精度**。
> 深度值本身就是距离（单位 mm），直接做差即可得到位移。

---

## 4. 硬件连接要点

实测过程中踩过的问题：**相机未识别**。

排查顺序（按概率从高到低）：

1. **USB 线材**：必须用支持 USB 3 的**数据线**。D435i 全功能峰值电流约 2A，
   普通充电线（只有 USB 2 的线芯）会导致识别失败或降速到 2.1。
2. **直插机身**：不要走 USB Hub / 扩展坞。本机接的是 VIA Labs `USB3.1 Hub`，
   这类扩展坞容易供电不足，且会把设备协商到 USB 2.0。
   **直插 MacBook 侧面的雷雳口。**
3. **提权**：确认用了 `sudo`。
4. **指示灯**：相机侧面指示灯应亮起。

验证是否被系统认到（不需要 sudo）：

```bash
ioreg -w0 -l | grep -ci realsense     # 返回 0 说明系统层面就没认到
ioreg -p IOUSB -w0 -l | grep '"USB Product Name"'
```

---

## 5. 本机状态

| 项 | 状态 | 时间 |
|---|---|---|
| librealsense 2.58.4 (brew) | ✅ 已安装 | 2025-09-22 |
| conda env `realsense` | ✅ 已创建 | 2025-09-22 |
| pyrealsense2 2.58.3 | ✅ 可 import | 2025-09-22 |
| 自检脚本 | ✅ 已就绪 | 2025-09-22 |
| **相机连通性** | ⏳ **待验证（采集时相机未连接）** | — |

> 连通性一栏必须在**相机实际插上并跑通 `rs_check.py`** 之后才可改为 ✅。
> 在拿到实测输出前，不得声称「相机已可用」。

---

## 6. 相关文档

- [SDK 安装（通用）](../02_SDK与编程/SDK安装.md) — Linux/Windows/Jetson 安装方式，macOS 部分见本文修正
- [Python API](../02_SDK与编程/Python_API.md) — pyrealsense2 接口用法
- [IMU 校准](../05_校准/IMU校准.md) — ⚠️ **仅 Linux 可用**，macOS 上 IMU 被禁用
- [官方 macOS 安装文档](https://dev.realsenseai.com/installation/macos-installation-for-realsense-sdk/)
- [librealsense issue #13903](https://github.com/realsenseai/librealsense/issues/13903) — USB 2.1 识别问题
- [librealsense issue #14648](https://github.com/realsenseai/librealsense/issues/14648) — macOS 上 IMU 导致 viewer 崩溃
