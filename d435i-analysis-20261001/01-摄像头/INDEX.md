# D435i 资料库索引

> Intel® RealSense™ Depth Camera D435i — 机器人/SLAM/机械臂视觉的资料库
> 最后更新：2026-09-22

## 🔖 当前交接

**接手本库请先读 [`HANDOFF.md`](HANDOFF.md)** —— 全库唯一当前交接入口，含：
当前阶段、环境基线、已完成工作、验证证据、关键文件 SHA256、已知风险、唯一下一步。

> 状态（2026-09-22）：D435i 已在 macOS 上**完全跑通**（深度+彩色+红外三流并发 32fps，
> 15s 100% 无丢帧），环境与脚本均已就绪，**位移测量算法尚未开始**。

---

## 📌 一句话定位

D435i 是 Intel 出品的 **带 IMU 的双目结构光深度相机**，主打室内 0.3-3m 测距，配合 ROS2 与 librealsense SDK，可一站式搭建 RGB-D + IMU 的 SLAM / 测距 / 物体检测管线。

## 🗂️ 目录结构

```
视觉位移判断/
├── HANDOFF.md                    ← ⭐ 当前交接入口，接手先读这个
├── INDEX.md                      ← 你正在看这个
├── README.md                     项目说明与快速上手
├── 00_产品概述/
│   ├── 产品定位与规格.md         核心参数、SKU、对标产品
│   ├── 应用场景.md               典型用例：SLAM/测距/机械臂
│   └── 坐标系与外参.md           深度坐标系、IMU 坐标系、外参
├── 01_硬件详解/
│   ├── 相机模组.md               双目 IR + RGB + 投射器结构
│   ├── IMU_BMI055.md             Bosch BMI055 6 轴 IMU 详解
│   └── 多相机同步.md             多机硬件同步、时戳对齐
├── 02_SDK与编程/
│   ├── SDK安装.md                librealsense2 安装（Linux/Win/macOS/Jetson）
│   ├── C++_API.md                C++ pipeline / frame / sensor
│   ├── Python_API.md             pyrealsense2 入门
│   └── 示例代码.md               深度+点云+IMU 采集代码
├── 03_ROS集成/
│   ├── ROS2_Wrapper.md           realsense2_camera 包说明
│   ├── 话题与参数.md             话题列表 / rs_launch.py 参数
│   └── 常用配置.md               对齐、点云、多机、点云滤波
├── 04_应用场景/
│   ├── SLAM建图.md               RTAB-Map / ORB-SLAM3 集成
│   ├── 点云与测距.md             点云生成、物体距离测量
│   ├── 物体检测.md               YOLO + Depth 融合方案
│   └── 多相机阵列.md             双目 / 多 D435i 阵列方案
├── 05_校准/
│   ├── IMU校准.md                rs-imu-calibration 工具
│   └── 内外参标定.md             相机内外参、IMU-Cam 联合标定
└── 06_参考资料/
    ├── 官方文档汇总.md           datasheet、白皮书、SDK 文档链接
    └── 社区资源.md               GitHub 项目、博客、论坛

07_本机实机验证/                    ← 本机 macOS 实测记录
├── 杆边界测试报告.md              ⭐ 最新：20cm/3mm杆 MDR=0.307mm（达标）
├── Sol咨询与验证记录.md          外部专家审阅与逐条验证
├── 16倍异常根因报告.md            异常排查全过程（已定位）
├── 实测数据汇总.md               早期数据（部分已作废，见文首声明）
├── 扰动极限测试计划_v3.md         测试计划（⚠️已被取代）
├── 颈部位移测量方案_v2.md         颈部件方案（需求已变更为杆）
├── Z轴位移测量方案_v1.md          早期方案（⚠️已作废）
├── rs_measure.py                 ⭐ 核心库：S4 算法 + 调零
├── rs_rod_boundary.py            6 种统计策略对比
├── rs_verify_roi.py              ROI 身份验证（防定位错目标）
├── rs_verify_s4.py               S4 端到端验证
├── rs_live_view.py               实时预览（点击设 ROI + 块检测）
├── rs_zero.py                    调零工具
├── rs_rod_test.py                细杆可用性专项
├── rs_block_precision.py         深度一致块精度扫描
├── rs_mdr_validity.py            MDR 估计有效性验证
├── macOS环境与验证.md             brew/conda 安装实测、平台限制
├── macOS连通问题与修复.md         UVC 驱动抢占接口的根因与解法
├── macos_usb_fix.py              核心修复模块（解绑系统 UVC 驱动）
├── rs_iface_probe.py             诊断：USB 接口占用状态
├── rs_stream_test.py             单 pipeline 多流并发测试
├── rs_capture.py                 采集一帧并落盘
├── rs_usb_reset.py               USB 复位（备选）
├── rs_check.py                   兼容性自检
├── zero_20cm.json                20cm 调零基准（含稳定像素坐标）
└── captures/                     实测样例数据

08_实时数控展示台/                    ← ⭐ 教学实验交作业展示系统（2x2网格）
├── README.md                     展示台操作手册与实验报告
├── 启动展示台.command             macOS Finder 双击一键启动
├── start.sh                      终端一键启动脚本
├── app.py                        数控展示台服务器 (实时图/红外图/位移图/数控台)
├── export_excel.py               一键导出 (x,z) Excel 报表与嵌入折线图
├── 界面展示效果.png               系统完整运行全景截图
└── 示例导出数据.xlsx              真实测试导出的标准 Excel 数据表
```
```

### 📋 文档状态说明

本目录文档按有效性分为三类：

| 状态 | 文档 | 说明 |
|---|---|---|
| **✅ 当前有效** | `尺度标定报告.md`、`后续实验计划.md`、`杆边界测试报告.md`、`Sol咨询与验证记录.md`、`16倍异常根因报告.md`、`macOS*.md` | 可据以决策 |
| **⚠️ 已作废** | `Z轴位移测量方案_v1.md` | ROI 方法错误，含被推翻结论（文首有作废声明） |
| **⚠️ 已被取代** | `扰动极限测试计划_v3.md`、`颈部位移测量方案_v2.md` | 已被 `后续实验计划.md` 取代 |
| **📜 数据记录** | `实测数据汇总.md` | 早期数据，部分作废（文首有声明） |

## 🔍 按主题快速跳转

### 🚀 想快速跑通
- [README.md → 快速上手](README.md)
- [SDK 安装 →](02_SDK与编程/SDK安装.md)
- [ROS2 启动 →](03_ROS集成/ROS2_Wrapper.md)

### 🎯 典型应用
- **SLAM 建图**：[04_应用场景/SLAM建图.md](04_应用场景/SLAM建图.md)
- **物体测距**：[04_应用场景/点云与测距.md](04_应用场景/点云与测距.md)
- **机械臂视觉**：[04_应用场景/物体检测.md](04_应用场景/物体检测.md)
- **多相机阵列**：[04_应用场景/多相机阵列.md](04_应用场景/多相机阵列.md)

### 🔧 遇到问题
- **IMU 数据漂移**：[05_校准/IMU校准.md](05_校准/IMU校准.md)
- **深度与 RGB 不对齐**：[03_ROS集成/常用配置.md](03_ROS集成/常用配置.md)
- **坐标方向不对**：[00_产品概述/坐标系与外参.md](00_产品概述/坐标系与外参.md)

### 📐 开发参考
- **C++ 开发**：[02_SDK与编程/C++_API.md](02_SDK与编程/C++_API.md)
- **Python 开发**：[02_SDK与编程/Python_API.md](02_SDK与编程/Python_API.md)
- **示例代码**：[02_SDK与编程/示例代码.md](02_SDK与编程/示例代码.md)
- **ROS2 参数**：[03_ROS集成/话题与参数.md](03_ROS集成/话题与参数.md)

### 🧰 硬件与底层
- **相机模组**：[01_硬件详解/相机模组.md](01_硬件详解/相机模组.md)
- **IMU BMI055**：[01_硬件详解/IMU_BMI055.md](01_硬件详解/IMU_BMI055.md)
- **多机同步**：[01_硬件详解/多相机同步.md](01_硬件详解/多相机同步.md)

### 📚 资料库元信息
- **官方文档**：[06_参考资料/官方文档汇总.md](06_参考资料/官方文档汇总.md)
- **社区资源**：[06_参考资料/社区资源.md](06_参考资料/社区资源.md)

### 🚀 作业交付与实时数控展示台（一键启动）
- **展示台操作手册**：[08_实时数控展示台/README.md](08_实时数控展示台/README.md)
- **Finder 双击启动**：`08_实时数控展示台/启动展示台.command`
- **终端一键启动**：`cd 08_实时数控展示台 && ./start.sh`
- **一键导出 Excel**：数控台按钮直接生成带时序图表的 `.xlsx`
- **作业打包提交文件**：`视觉位移监控数控系统_作业提交包.zip` (根目录，可直接交付老师)

### 💻 本机实机验证（macOS）
- **⭐ 最新结果（首次实验结束 2026-09-23）**：**尺度误差 0.045%，2mm 判据通过**（余量 30–2222×）
- **尺度标定报告**：[07_本机实机验证/尺度标定报告.md](07_本机实机验证/尺度标定报告.md)　← **先看这篇（含汇报图）**
- **汇报图**：[07_本机实机验证/图表/report.png](07_本机实机验证/图表/report.png)（4 子图）
- **后续实验计划**：[07_本机实机验证/后续实验计划.md](07_本机实机验证/后续实验计划.md)　← 含 10 个未验证缺口与优先级
- **杆边界测试报告**：[07_本机实机验证/杆边界测试报告.md](07_本机实机验证/杆边界测试报告.md)　← ⚠️ **先看这篇**
- **Sol 咨询记录**：[07_本机实机验证/Sol咨询与验证记录.md](07_本机实机验证/Sol咨询与验证记录.md)
- **环境与验证记录**：[07_本机实机验证/macOS环境与验证.md](07_本机实机验证/macOS环境与验证.md)
- **连通问题与修复**：[07_本机实机验证/macOS连通问题与修复.md](07_本机实机验证/macOS连通问题与修复.md)

```bash
# 核心测量库（S4 中值锚定截尾均值）
import rs_measure as M
m = M.Measurement(roi=(385,104,39,83), physical_mm=200.0)
m.calibrate(depth_frames)   # 调零
z = m.read(depth_frame)     # 读数 (mm)

# 实时预览
sudo -E /opt/anaconda3/envs/realsense/bin/python3 07_本机实机验证/rs_live_view.py --port 8899 --temporal
```

> ⚠️ **关键方法警告**：绝不能用「按深度值筛选像素」建 ROI（会拿到散布各处的
> 深度碎片，产生假噪声，实测可将 MDR 从 0.3mm 虚高到 **199mm**）。
> 必须用 `rs_measure.measure_s4()` 或深度一致连通块。详见报告 §2。
>
> ⚠️ macOS 上若报 `failed to set power state` / `failed to claim usb interface`，
> 是系统 UVC 驱动抢占了 interface 0，跑 `macos_usb_fix.ensure_ready()` 即可。
>
> ⚠️ 本机（macOS）**IMU 被官方禁用**，若需 VIO/SLAM 请换 Ubuntu。

## 🔑 常用检索关键词

| 关键词 | 命中页 |
|--------|--------|
| ROS2 launch | `03_ROS集成/ROS2_Wrapper.md`、`03_ROS集成/话题与参数.md` |
| 点云 | `02_SDK与编程/示例代码.md`、`04_应用场景/点云与测距.md` |
| IMU 校准 | `05_校准/IMU校准.md`、`01_硬件详解/IMU_BMI055.md` |
| ORB-SLAM3 | `04_应用场景/SLAM建图.md` |
| RTAB-Map | `04_应用场景/SLAM建图.md` |
| align_depth | `03_ROS集成/常用配置.md` |
| 坐标系 | `00_产品概述/坐标系与外参.md` |
| macOS 安装 | `07_本机实机验证/macOS环境与验证.md`、`02_SDK与编程/SDK安装.md` |
| USB 报错/power state | `07_本机实机验证/macOS连通问题与修复.md`、`macos_usb_fix.py` |
| 位移测量 | `07_本机实机验证/rs_capture.py`（深度值即 mm 距离） |

## ⚠️ 选型提示

| 如果你要 | 推荐方案 |
|----------|---------|
| 室内 0.3-3m 测距 + SLAM | D435i（双目+IMU）— 本文主角 |
| 户外 / 远距离 6m+ | 考虑 D455 / Livox |
| 高精度小视场 | D405 |
| 不需要 IMU | D435（同款无 IMU） |
| ROS2 原生支持 | realsense2_camera 包（ros-<dist>-realsense2-*） |

## 📝 文档维护

- 内容来源：librealsense 官方文档、realsense-ros GitHub、Intel ARK、SLAM 社区实践
- 更新策略：librealsense 大版本升级时回访本页校对
- 反馈：在仓库内开 issue 标记文档错误或补充