# librealsense2 SDK 安装

> Librealsense2 是 Intel 官方的 D435i SDK。本页汇总 Linux / Windows / Jetson / Raspberry Pi 的安装方式。

## 1. 安装方式概览

| 系统 | 推荐方式 | 难度 |
|------|---------|------|
| Ubuntu 20.04/22.04 (x86_64) | apt 官方源 | ⭐ |
| Ubuntu 24.04+ | 源码编译 | ⭐⭐ |
| Windows 10/11 | exe 安装包 | ⭐ |
| macOS (Apple Silicon) | **`brew install librealsense`** | ⭐ |
| NVIDIA Jetson | apt (JetPack 4.6+) | ⭐ |
| Raspberry Pi 4/5 | 源码编译 | ⭐⭐⭐ |

## 2. Ubuntu x86_64 (apt 推荐)

### 2.1 注册 Intel 源

```bash
# 添加 Intel 官方源
sudo apt-get install -y curl
curl -sSf https://librealsense.intel.com/rs-utils-v2/upgrade-repo-format | sudo bash

# 或手动添加
sudo add-apt-repository "deb https://librealsense.intel.com/librealsense/apt-repo $(lsb_release -cs) main" -u
sudo apt-get update
```

### 2.2 安装 SDK 与工具

```bash
# SDK 库 + 头文件 + 工具
sudo apt-get install -y librealsense2-dkms \
                        librealsense2-utils \
                        librealsense2-dev \
                        librealsense2-gl

# 可选：ROS2 包装包（ROS 用户另装，见 ROS 章节）
# 可选：Python binding
sudo apt-get install -y python3-pyrealsense2
```

### 2.3 验证

```bash
# 检查设备
rs-enumerate-devices

# 看图像
realsense-viewer

# 测距 demo
rs-distance
```

输出应该能看到 D435i 的 serial_number 与所有流配置。

## 3. Ubuntu 源码编译

适合需要最新功能或定制化场景：

```bash
# 1. 安装依赖
sudo apt-get install -y git cmake libssl-dev libusb-1.0-0-dev \
    pkg-config libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev \
    libeigen3-dev

# 2. 克隆源码
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense

# 3. 配置（带 IMU、点云等高级选项）
mkdir build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_EXAMPLES=true \
    -DBUILD_TOOLS=true \
    -DBUILD_GRAPHICAL_EXAMPLES=true \
    -DCHECK_FOR_UPDATES=false

# 4. 编译（4-8 核，约 10-30 分钟）
make -j$(nproc)
sudo make install
```

**验证安装**：

```bash
# 检查 pkg-config
pkg-config --modversion librealsense2

# 检查库文件
ls /usr/local/lib/librealsense2*

# 检查头文件
ls /usr/local/include/librealsense2/
```

## 4. Windows 安装

### 4.1 安装包方式

从 [librealsense releases](https://github.com/IntelRealSense/librealsense/releases) 下载 exe 安装包：

- `Intel.RealSense.SDK-Win10.exe`（Windows 10+）
- 双击安装，会自动安装驱动 + SDK + viewer

### 4.2 验证

```powershell
# PowerShell
cd "C:\Program Files (x86)\Intel RealSense SDK 2.0\tools"
.\realsense-viewer.exe
```

### 4.3 Python 环境

```powershell
# 用 conda 装 pyrealsense2
conda create -n rs python=3.10
conda activate rs
pip install pyrealsense2
```

## 5. NVIDIA Jetson 安装

Jetson 是 SLAM 部署的主流平台。

### 5.1 JetPack 4.6 / 5.x

```bash
# 1. JetPack 已包含 USB 驱动
# 2. 装 SDK
sudo apt-get install -y librealsense2-utils librealsense2-dev \
    librealsense2-dkms python3-pyrealsense2

# 3. 验证
realsense-viewer
```

### 5.2 Jetson Orin Nano / AGX Orin

兼容性最好的组合，**USB 3.0 带宽足够**跑 1280×720@30 双流。

### 4.3 性能参考

| 平台 | 深度 1280×720@30 + RGB 1080p@30 | CPU 占用 |
|------|-------------------------------|----------|
| Jetson Nano | ⚠️ 卡顿（建议降到 848×480）| > 80% |
| Jetson Orin Nano | ✅ 流畅 | ~30% |
| Jetson AGX Orin | ✅ 流畅 | ~15% |
| x86 i7 | ✅ 流畅 | ~10% |

## 6. Raspberry Pi 安装

⚠️ **不推荐**：D435i 在 Pi 4 上只能跑 640×480@30，且 IMU 数据会丢。

如果必须：

```bash
# 编译（耗时 2-4 小时）
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TOOLS=ON -DBUILD_EXAMPLES=OFF
make -j4
sudo make install
```

## 7. Docker 容器化

适合 CI / 复现环境：

```dockerfile
# Dockerfile 示例
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    librealsense2-dkms librealsense2-utils librealsense2-dev \
    python3-pyrealsense2 \
    && rm -rf /var/lib/apt/lists/*

# USB 设备映射（运行时）
# docker run --device /dev/bus/usb ...
```

## 9. 常见安装报错

### 9.1 USB 权限问题

```bash
# 现象：能 enumerate 但看不到流
# 解决：复制 udev 规则
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### 9.2 dkms 编译失败

```bash
# 现象：librealsense2-dkms 安装失败
# 解决：手动重装
sudo apt-get remove --purge librealsense2-dkms
sudo apt-get install librealsense2-dkms
```

### 9.3 多版本冲突

```bash
# 现象：cmake 找到旧版
# 解决：清理后重装
sudo apt-get remove --purge librealsense2*
sudo rm -rf /usr/local/include/librealsense2 /usr/local/lib/librealsense2*
# 重新安装
```

### 9.4 USB 带宽不足

```bash
# 现象：Viewer 卡顿、深度丢帧
# 解决：
#   1. 换 USB 3.0 端口（蓝舌）
#   2. 换更短的线
#   3. 关闭其他 USB 3.0 设备
#   4. 降低分辨率到 848x480
```

## 10. 安装后必做

1. **跑 `rs-enumerate-devices`**：确认能识别 D435i
2. **打开 `realsense-viewer`**：可视化所有流（深度、RGB、IMU）
3. **运行 IMU 校准**：[05_校准/IMU校准.md](../05_校准/IMU校准.md)
4. **记录 serial_number**：ROS2 多机配置需要

> ⚠️ **macOS 用户注意**：上面第 2、3 步在 macOS 上**不可用**。
> RealSense Viewer 官方不支持 macOS，IMU 也被官方禁用。
> macOS 的完整安装与验证记录见 [07_本机实机验证/macOS环境与验证.md](../07_本机实机验证/macOS环境与验证.md)。

## macOS (Apple Silicon) 专用说明

Homebrew 提供 **arm64 原生 bottle**，无需源码编译：

```bash
brew install librealsense
```

Python 绑定**必须走 conda-forge**（PyPI 上没有 macOS wheel）：

```bash
conda create -y -n realsense -c conda-forge --override-channels \
    python=3.12 pyrealsense2
```

macOS 的两条硬限制：

| 限制 | 影响 |
|------|------|
| RealSense Viewer 不支持 | 无 GUI，只能用 `rs-*` CLI + Python |
| IMU 被禁用 | **D435i 的 IMU 用不了**，无法做 VIO/SLAM |

另需注意：macOS 12+ 运行 librealsense 工具**必须 `sudo`**。

详见 [07_本机实机验证/macOS环境与验证.md](../07_本机实机验证/macOS环境与验证.md)。

## 11. 升级与降级

```bash
# 升级到最新稳定版（不破坏依赖）
sudo apt-get update && sudo apt-get upgrade librealsense2*

# 降级（如 ROS2 Jazzy 需要特定版本）
sudo apt-get install librealsense2=2.55.1-1~ubuntu22.04
```

## 参考

- [librealsense 安装文档](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md)
- [librealsense releases](https://github.com/IntelRealSense/librealsense/releases)
- [Intel RealSense SDK 2.0 文档](https://dev.intelrealsense.com/docs/docs-get-started)