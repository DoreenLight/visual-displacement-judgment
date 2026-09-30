# D435i 资料库 README

> 本资料库系统整理 Intel RealSense D435i 深度相机的硬件、SDK、ROS 集成与应用方案。
> 适合：机器人开发者、SLAM 研究者、机械臂视觉工程师。

---

## 📌 这是什么

这是一个**面向开发者查阅**的 D435i 知识库，不是 SDK 也不是代码仓库。
所有内容都是**已整理的 Markdown 文档**，按主题分类，配备目录跳转与关键词检索。

## 🎯 适用场景

- 机器人室内导航与建图（SLAM）
- 机械臂视觉抓取（深度+目标检测）
- 物体测距与体积估算
- RGB-D 数据采集与回放
- 多相机阵列（立体 / 拼接）
- 手势 / 人体骨架识别（搭配第三方算法）

## ⚡ 快速上手（5 分钟）

### 1️⃣ 连接相机

```bash
# 检查 USB 3.0（必须 USB 3.0+，否则帧率/分辨率受限）
lsusb | grep Intel
# 应看到 "Intel Corp. Intel RealSense D4XX" 之类字样
```

### 2️⃣ 看图像（最快验证）

```bash
# 安装 librealsense2 viewer
sudo apt install realsense-viewer
realsense-viewer
```

### 3️⃣ ROS2 启动（项目推荐方式）

```bash
# Humble
sudo apt install ros-humble-realsense2-*

# 启动（默认 640x480x30 深度 + 640x480x30 RGB）
ros2 launch realsense2_camera rs_launch.py

# 启用对齐深度 + 点云
ros2 launch realsense2_camera rs_launch.py \
    depth_module.profile:=640x480x30 \
    pointcloud.enable:=true \
    align_depth.enable:=true
```

更多细节见 [02_SDK与编程/SDK安装.md](02_SDK与编程/SDK安装.md) 与 [03_ROS集成/ROS2_Wrapper.md](03_ROS集成/ROS2_Wrapper.md)。

## 🧭 阅读路径建议

| 你是 | 看这里 |
|------|--------|
| 第一次接触 D435i | [00_产品概述/产品定位与规格.md](00_产品概述/产品定位与规格.md) → [README.md](README.md) |
| **macOS 用户（Apple Silicon）** | [07_本机实机验证/macOS环境与验证.md](07_本机实机验证/macOS环境与验证.md) ⚠️ IMU 被官方禁用，仅适合纯视觉 |
| 准备接入到机器人 | [03_ROS集成/ROS2_Wrapper.md](03_ROS集成/ROS2_Wrapper.md) → [04_应用场景/SLAM建图.md](04_应用场景/SLAM建图.md) |
| 做应用开发 | [02_SDK与编程/示例代码.md](02_SDK与编程/示例代码.md) → [04_应用场景/](04_应用场景/) |
| 调相机 / 排查问题 | [05_校准/IMU校准.md](05_校准/IMU校准.md) → [00_产品概述/坐标系与外参.md](00_产品概述/坐标系与外参.md) |

## 🗂️ 目录结构

```
视觉位移判断/
├── INDEX.md
├── README.md
├── 00_产品概述/        产品定位、规格、应用场景、坐标系约定
├── 01_硬件详解/        相机模组、IMU BMI055、多机同步
├── 02_SDK与编程/      librealsense 安装与 C++/Python 开发
├── 03_ROS集成/         realsense2_camera 包、话题、配置
├── 04_应用场景/        SLAM、点云测距、物体检测、多相机阵列
├── 05_校准/             IMU 校准、相机内外参标定
├── 06_参考资料/         官方文档链接、社区项目
└── 07_本机实机验证/    macOS 实测记录 + 自检脚本
```

详细目录与跳转见 [INDEX.md](INDEX.md)。

## ⚠️ 重要前置知识

1. **必须 USB 3.0+**：D435i 在 USB 2.0 下只能跑低分辨率/低帧率。
2. **IMU 出厂未精细校准**：直接读数会有偏置与重力误差，建议运行 [rs-imu-calibration](05_校准/IMU校准.md)。⚠️ macOS 官方禁用 IMU，需 Linux 才能用 IMU 相关功能。
3. **坐标系方向**：x 右 / y 下 / z 前，符合 OpenCV pinhole 模型，但 **加速度计静止时读 -9.8**（详见 [坐标系与外参](00_产品概述/坐标系与外参.md)）。
4. **红外投射器**：D435i 投射静态随机图案，对纹理缺失表面（白墙）友好；对透明/反光物体仍然失效。

## 🔧 推荐开发环境

| 组件 | 推荐版本 |
|------|---------|
| OS | Ubuntu 22.04 / 20.04 / 24.04 |
| librealsense | 2.55+（支持 D435i） |
| ROS2 | Humble / Iron / Jazzy |
| Python | 3.8+（pyrealsense2） |
| 算力 | x86_64 / Jetson Orin / Raspberry Pi 4（受限） |

## 📚 资料来源

主要来源：
- [librealsense GitHub](https://github.com/IntelRealSense/librealsense)
- [realsense-ros GitHub](https://github.com/IntelRealSense/realsense-ros)
- [Intel ARK D435i 页面](https://www.intel.com/content/www/us/en/products/sku/190004/intel-realsense-depth-camera-d435i/specifications.html)
- [D400 Series Datasheet PDF](https://www.intel.com/content/dam/develop/public/us/en/documents/intel-realsense-d400-series-product-datasheet-rev-2-0.pdf)

完整清单见 [06_参考资料/官方文档汇总.md](06_参考资料/官方文档汇总.md)。

## 📝 维护

- 任何官方 SDK / ROS 包大版本更新时回访校对
- 新增应用场景时，在 `04_应用场景/` 下加一篇
- 文档错误请在仓库 issue 标记