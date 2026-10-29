# SLAM 建图

> 用 D435i 做室内 SLAM 的主流方案与集成步骤。

## 1. SLAM 方案选型

| 方案 | 类型 | 优势 | 限制 | 推荐度 |
|------|------|------|------|--------|
| **RTAB-Map** | RGB-D + 闭环 | 全开源、ROS 原生、易用 | 大场景需调 | ⭐⭐⭐⭐⭐ |
| **ORB-SLAM3** | Visual-Inertial | 精度高 | ROS2 集成折腾 | ⭐⭐⭐⭐ |
| **VINS-Fusion** | Visual-Inertial | 紧耦合 | 配置门槛 | ⭐⭐⭐ |
| **Cartographer** | 3D SLAM | Google 出品 | 调参复杂 | ⭐⭐⭐ |
| **LIO-SAM** | LiDAR-Visual | 高精度 | 需 LiDAR | ⭐⭐⭐⭐（多传感器）|

## 2. RTAB-Map（推荐入门）

### 2.1 安装

```bash
sudo apt install ros-<dist>-rtabmap ros-<dist>-rtabmap-examples
```

### 2.2 一行命令式 launch

```bash
# 启动相机
ros2 launch realsense2_camera rs_launch.py \
    depth_module.depth_profile:=640x480x30 \
    rgb_camera.color_profile:=640x480x30 \
    pointcloud.enable:=true \
    align_depth.enable:=true

# 启动 RTAB-Map
ros2 launch rtabmap_examples realsense_d435i_infra_composition.launch.py
```

> `composition` 表示将 RealSense + RTAB-Map 跑在同一进程，省通信开销。

### 2.3 标准 launch（非 composition）

```bash
ros2 launch rtabmap_launch rtabmap.launch.py \
    rtabmap_args:="--delete_db_on_start" \
    depth_topic:=/camera/depth_registered/image_rect_raw \
    frame_id:=camera_link \
    approx_sync:=true \
    rviz:=true
```

### 2.4 RTAB-Map 参数

```yaml
# /rtabmap node
rtabmap:
  ros__parameters:
    Rtabmap/DetectionRate: "1.0"          # 检测频率
    Rtabmap/DetectionMaxFeature: "500"   # 特征数
    Rtabmap/MemoryThr: "100"             # 工作内存回点上限
    Rtabmap/LoopThr: "0.11"              # 闭环阈值
    Rtabmap/OptimizerIterations: "50"
    
    # RGB-D 模式
    Rtabmap/PublishRAM: "true"
    
    # 可视化
    Rtabmap/Publish3DMap: "true"
```

### 2.5 保存地图

```bash
# 命令行导出
ros2 service call /rtabmap/export_map \
    rtabmap_msgs/srv/ExportMap "{path: '/path/to/directory', name: 'my_map.db'}"

# 地图文件是 .db（SQLite 格式），可用 rtabmap-databaseViewer 打开
```

## 3. ORB-SLAM3

### 3.1 安装

```bash
# ROS2 版本（社区维护）
mkdir -p ~/orb_slam3_ws/src && cd ~/orb_slam3_ws/src
git clone https://github.com/gjcliff/ORB_SLAM3_ROS2.git
# 或 EOLab-HSRW 版本
# git clone https://github.com/EOLab-HSRW/orbslam3_ros2.git

# 编译（按各自仓库说明）
cd ~/orb_slam3_ws
colcon build --symlink-install
```

### 3.2 启动 D435i

```bash
# 相机启动
ros2 launch realsense2_camera rs_launch.py \
    depth_module.depth_profile:=640x480x30 \
    rgb_camera.color_profile:=640x480x30

# ORB-SLAM3（mono-inertial）
ros2 run orbslam3_ros2 mono_inertial \
    /path/to/ORBvoc.txt \
    /path/to/D435i.yaml
```

### 3.3 相机配置文件（D435i.yaml）

```yaml
# D435i.yaml（参考值）
File.version: "1.0"
Camera.type: "PinHole"

# 相机内参（640×480）
Camera1.fx: 386.738
Camera1.fy: 386.738
Camera1.cx: 320.0
Camera1.cy: 240.0

# 畸变系数
Camera1.k1: 0.0
Camera1.k2: 0.0
Camera1.p1: 0.0
Camera1.p2: 0.0

# 分辨率
Camera.width: 640
Camera.height: 480
Camera.fps: 30

# 颜色顺序（RGB）
Camera.RGB: 1

# 近距离阈值
ThDepth: 30.0  # mm
DepthMapFactor: 1.0

# IMU 参数（BMI055 校准后）
IMU.NoiseGyro: 1.7e-4
IMU.NoiseAcc: 2.0e-3
IMU.GyroWalk: 1.9e-5
IMU.AccWalk: 3.0e-3
IMU.Frequency: 200

# 相机-IMU 外参（需 Kalibr 标定）
TBC: [-0.00764, -0.00173, 0.00990]
```

## 4. VINS-Fusion

### 4.1 安装

```bash
# 源码编译
mkdir -p ~/vins_ws/src && cd ~/vins_ws/src
git clone https://github.com/HKUST-Aerial-Robotics/VINS-Fusion.git

cd ~/vins_ws
catkin_make  # ROS1 风格，VINS-Fusion 暂以 ROS1 为主
# 或用 catkin-tools 兼容 ROS2
```

### 4.2 D435i 配置

VINS-Fusion 主要工作在 Mono/Stereo + IMU 模式：

```yaml
# vins.yaml
imu: 1                   # 用 IMU
num_of_features: 150
pose_graph_enable: 1

# IMU 噪声（BMI055）
acc_n: 0.02              # 加速度白噪声
gyr_n: 0.005              # 陀螺白噪声
acc_w: 0.002              # 加速度随机游走
gyr_w: 2.0e-5             # 陀螺随机游走

# 相机
model_type: PINHOLE
distortion_type: RADTAN
fx: 386.738
fy: 386.738
cx: 320.0
cy: 240.0
k1: 0.0
k2: 0.0
p1: 0.0
p2: 0.0

# 相机-IMU 外参（Kalibr 输出）
estimate_extrinsic: 2   # 0:固定 1:在线 2:外部标定
```

## 5. 实战：RTAB-Map + D435i 全流程

### 5.1 启动

```bash
# Terminal 1: 相机
ros2 launch realsense2_camera rs_launch.py \
    camera_name:=camera \
    depth_module.depth_profile:=640x480x30 \
    rgb_camera.color_profile:=640x480x30 \
    pointcloud.enable:=true \
    align_depth.enable:=true \
    spatial_filter.enable:=true \
    temporal_filter.enable:=true

# Terminal 2: RTAB-Map
ros2 launch rtabmap_launch rtabmap.launch.py \
    rtabmap_args:="--delete_db_on_start" \
    depth_topic:=/camera/depth_registered/image_rect_raw \
    rgb_topic:=/camera/color/image_raw \
    camera_info_topic:=/camera/color/camera_info \
    frame_id:=camera_link \
    approx_sync:=true \
    rviz:=true

# Terminal 3 (可选): 可视化
rviz2
# 在 Rviz 里添加 PointCloud2 订阅 /rtabmap/cloud_map
```

### 5.2 移动建图

```
1. 把 D435i 拿在手里 / 装到移动平台上
2. 慢慢绕房间走（不要太快，RTAB-Map 适合 < 1 m/s）
3. 在 Rviz 里实时看到 3D 地图
5. 完成时按 Ctrl+C 停止，地图自动保存为 ~/.ros/rtabmap.db
```

### 5.3 导出地图

```bash
# 用 RTAB-Map 自带工具
rtabmap-export --images ~/.ros/rtabmap.db /path/to/output/dir

# 输出：
# - depth/*.png    深度图
# - rgb/*.png      RGB 图
# - groundtruth.txt 相机位姿
# - mesh.ply       三角网格
```

### 5.4 加载地图定位

```bash
ros2 launch rtabmap_launch rtabmap.launch.py \
    rtabmap_args:="--udw true" \
    --load ~/path/to/map.db
```

`--udw true` 启用定位模式（不重建图）。

## 6. 性能调优

### 6.1 RTAB-Map 调参

```yaml
# 大场景
Rtabmap/MemoryThr: 500          # 提高工作内存阈值
Rtabmap/DetectionRate: 1.0      # 每帧检测
Rtabmap/LoopThr: 0.20           # 提高闭环阈值（防误闭环）

# 小场景/室内
Rtabmap/MemoryThr: 100          # 默认
Rtabmap/DetectionMaxFeature: 300  # 减少特征点
```

### 6.2 ORB-SLAM3 调参

```yaml
# 精度优先
ORBextractor.nFeatures: 1500
ORBextractor.nLevels: 8
ORBextractor.scaleFactor: 1.2

# 速度优先
ORBextractor.nFeatures: 500
ORBextractor.nLevels: 6
```

## 7. 闭环检测失败排查

| 现象 | 原因 | 解决 |
|------|------|------|
| 地图漂移严重 | 闭环阈值太小 | `Rtabmap/LoopThr: 0.20` |
| 闭环后地图扭曲 | 错闭环 | 提高特征数 / `Vis/MinInliers` |
| TF 跳变 | IMU 噪声大 | 重跑 IMU 校准 |
| 跟踪丢失 | 运动太快 / 图像模糊 | 降低移动速度 |
| 地图碎片化 | 闭环未生效 | 检查 `Rtabmap/MemoryThr` |

## 8. SLAM 评估指标

### 8.1 绝对轨迹误差（ATE）

```bash
# 用 evo 工具
pip install evo

# 估计轨迹 vs 真值
evo_ape tum estimated.txt groundtruth.txt -va --plot
```

### 8.2 相对位姿误差（RPE）

```bash
evo_rpe tum estimated.txt groundtruth.txt -va --plot
```

## 9. 数据集参考

| 数据集 | 链接 |
|-------|------|
| TUM RGB-D | https://vision.in.tum.de/data/datasets/rgbd-dataset |
| EuRoC MAV | https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets |
| Replica | https://github.com/facebookresearch/Replica-Dataset |
| TUM VI | https://vision.in.tum.de/data/datasets/visual-inertial-dataset |

## 参考

- [RTAB-Map ROS2 文档](http://wiki.ros.org/rtabmap_ros/Tutorials)
- [ORB-SLAM3 GitHub](https://github.com/UZ-SLAMLab/ORB_SLAM3)
- [VINS-Fusion GitHub](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion)
- [rtabmap_examples launch](https://github.com/introlab/rtabmap_ros/tree/ros2/rtabmap_examples/launch)