# ROS2 Wrapper (realsense2_camera)

> Intel 官方维护的 ROS2 包 `realsense2_camera`，把 D435i 接入 ROS2 生态。

## 1. 包介绍

| 项 | 说明 |
|----|------|
| 包名 | `realsense2_camera` |
| GitHub | [IntelRealSense/realsense-ros](https://github.com/IntelRealSense/realsense-ros) |
| ROS 2 分支 | `ros2-development` |
| 支持的 Distro | Foxy / Humble / Iron / Jazzy / Kilted / Rolling |
| 支持的设备 | D400 / L500 / SR300 / T265 |

## 2. 安装

### 2.1 apt 方式（推荐）

```bash
# 替换 <distro> 为实际版本（Humble/Iron/Jazzy）
sudo apt install ros-<distro>-realsense2-*

# 这会装所有 realsense 相关包：
# - ros-<distro>-realsense2-camera        主包
# - ros-<distro>-realsense2-description   URDF
# - ros-<distro>-realsense2-examples      示例 launch
# - ros-<distro>-realsense2-msgs          消息定义
```

验证：

```bash
ros2 pkg list | grep realsense
# 应看到：realsense2_camera, realsense2_description, realsense2_examples, realsense2_msgs
```

### 2.2 源码编译（开发版）

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src

# 克隆（注意 ros2 分支）
git clone -b ros2-development https://github.com/IntelRealSense/realsense-ros.git

# 安装依赖
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y

# 编译
colcon build --packages-select realsense2_camera --symlink-install

# source
source install/setup.bash
```

## 3. 启动

### 3.1 最简启动

```bash
ros2 launch realsense2_camera rs_launch.py
```

默认配置：
- 深度：640×480@30fps
- RGB：640×480@30fps
- IMU：200Hz
- TF：camera_link / camera_depth_frame 等

### 3.2 常用参数

```bash
ros2 launch realsense2_camera rs_launch.py \
    depth_module.profile:=640x480x30 \
    rgb_camera.profile:=1280x720x30 \
    pointcloud.enable:=true \
    align_depth.enable:=true \
    depth_module.emitter_enabled:=true
```

## 4. 启动文件清单

`realsense2_camera/launch/` 下所有 launch：

| Launch 文件 | 用途 |
|------------|------|
| `rs_launch.py` | 单相机主入口 |
| `rs_multi_camera_launch.py` | 多相机（软件同步） |
| `rs_multi_camera_launch_sync.py` | 多相机（硬件触发同步） |
| `rs_from_file.py` | 从 bag 回放 |
| `rs_aligned_depth.py` | 对齐深度 + 点云 |
| `rs_d435i_examples/rs_d435i_pointcloud.launch.py` | D435i 点云示例 |
| `rs_d435i_examples/rs_d435i_depth_and_color.launch.py` | 深度+彩色 |
| `rs_d435i_examples/rs_d435i_imu_launch.py` | 仅 IMU |

### 4.1 D435i 专用

```bash
# 深度 + 彩色 + IMU
ros2 launch realsense2_camera rs_d435i_examples/rs_d435i_depth_and_color_and_IMU_launch.py

# 点云
ros2 launch realsense2_camera rs_d435i_examples/rs_d435i_pointcloud_launch.py

# IMU
ros2 launch realsense2_camera rs_d435i_examples/rs_d435i_IMU_launch.py
```

## 5. 话题清单

启动 `rs_launch.py` 后发布的话题（默认命名空间 `/camera/`）：

### 5.1 图像话题

| 话题 | 类型 | 用途 |
|------|------|------|
| `/camera/color/image_raw` | `sensor_msgs/Image` | RGB 原始图像 |
| `/camera/depth/image_rect_raw` | `sensor_msgs/Image` | 校正后深度图（Z16, mm） |
| `/camera/depth_registered/image_rect_raw` | `sensor_msgs/Image` | 对齐到 RGB 的深度图 |
| `/camera/stereo/left/image_raw` | `sensor_msgs/Image` | 左红外相机 |
| `/camera/stereo/right/image_raw` | `sensor_msgs/Image` | 右红外相机 |
| `/camera/infra1/image_raw` | `sensor_msgs/Image` | IR1 |
| `/camera/infra2/image_raw` | `sensor_msgs/Image` | IR2 |

### 5.2 点云与配准

| 话题 | 类型 | 用途 |
|------|------|------|
| `/camera/depth/color/points` | `sensor_msgs/PointCloud2` | 彩色点云（XYZRGB） |
| `/camera/depth/points` | `sensor_msgs/PointCloud2` | 深度坐标系点云 |

> `pointcloud.enable:=true` 才会有这两个话题。

### 5.3 IMU

| 话题 | 类型 | 频率 | 用途 |
|------|------|------|------|
| `/camera/imu` | `sensor_msgs/Imu` | 200 Hz | 合并加速度 + 陀螺 |
| `/camera/accel/imu_info` | `realsense2_camera_msgs/IMUInfo` | - | 加速度计元数据 |
| `/camera/gyro/imu_info` | `realsense2_camera_msgs/IMUInfo` | - | 陀螺仪元数据 |

### 5.4 相机信息

| 话题 | 类型 | 用途 |
|------|------|------|
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | RGB 内参 |
| `/camera/depth/camera_info` | `sensor_msgs/CameraInfo` | 深度内参 |
| `/camera/extrinsics/depth_to_color` | ±30 Hz | 深度→RGB 外参 |
| `/camera/extrinsics/depth_to_imu` | ±30 Hz | 深度→IMU 外参 |

### 5.5 元数据

| 话题 | 类型 | 频率 | 用途 |
|------|------|------|------|
| `/camera/metadata` | `realsense2_camera_msgs/Metadata` | 每帧 | 帧元数据 |

## 6. TF 树

启动后默认发布的 TF：

```
world
└── camera_link               ← 相机物理坐标系
    ├── camera_depth_frame    ← 深度光学坐标系（左 IR）
    │   └── camera_depth_optical_frame
    ├── camera_color_frame    ← RGB 光学坐标系
    │   └── camera_color_optical_frame
    ├── camera_infra1_frame   ← 左 IR 光学坐标系
    │   └── camera_infra1_optical_frame
    ├── camera_infra2_frame   ← 右 IR 光学坐标系
    │   └── camera_infra2_optical_frame
    └── camera_accel_frame    ← IMU 加速度计坐标系
        └── camera_accel_optical_frame
    └── camera_gyro_frame     ← IMU 陀螺仪坐标系
        └── camera_gyro_optical_frame
```

`_frame` 是 ROS REP-103 约定（X 前 / Y 左 / Z 上），`_optical_frame` 是 OpenCV 约定（X 右 / Y 下 / Z 前）。

## 7. 服务

| 服务 | 类型 | 用途 |
|------|------|------|
| `/camera/device_info` | `realsense2_camera_msgs/DeviceInfo` | 设备信息 |
| `/camera/hardware_monitor` | ±10 Hz | 温度、功耗监控 |
| `/camera/calib_config` | - | 校准参数读写 |
| `/camera/application_config` | - | 应用配置读写 |

```bash
# 查看设备信息
ros2 service call /camera/device_info realsense2_camera_msgs/srv/DeviceInfo
```

## 8. 录制 ROS bag

```bash
# 录制所有话题
ros2 bag record -a

# 只录制指定话题
ros2 bag record /camera/depth/image_rect_raw \
                /camera/color/image_raw \
                /camera/imu
```

输出格式：`.mcap`（ROS2 默认）或 `.db3`（旧格式）。

## 9. 回放

```bash
# 启动回放 launch
ros2 launch realsense2_camera rs_from_file.py \
    rosbag_filename:=/path/to/recording.mcap

# 也可单独 bag play + rs_from_file 组合
ros2 bag play recording.mcap &
ros2 launch realsense2_camera rs_launch.py &
```

## 10. 多相机

```bash
ros2 launch realsense2_camera rs_multi_camera_launch.py \
    camera_name1:=cam_left \
    camera_name2:=cam_right \
    camera1.serial_no:=213722067000 \
    camera2.serial_no:=213722068000

# 所有话题会有命名空间：/cam_left/..., /cam_right/...
```

详细多相机方案见 [01_硬件详解/多相机同步.md](../01_硬件详解/多相机同步.md)。

## 11. 自定义参数

所有可调参数可通过 launch 参数传入：

```bash
ros2 launch realsense2_camera rs_launch.py \
    depth_module.depth_profile:=640x480x30 \
    depth_module.infra_profile:=640x480x30 \
    rgb_camera.color_profile:=1280x720x30 \
    depth_module.emitter_enabled:=true \
    depth_module.laser_pulse:=0 \
    depth_module.inter_cam_sync_mode:=0 \
    align_depth.enable:=true \
    pointcloud.enable:=true \
    pointcloud.stream_filter:=2 \
    spatial_filter.enable:=true \
    temporal_filter.enable:=true
```

完整参数见 [话题与参数.md](话题与参数.md)。

## 12. 与 ROS1 互操作（不要选这条路）

ROS1 `realsense2_camera` 仍可用但官方已不再维护。新项目**只做 ROS2**。

## 13. 常见问题

| 问题 | 解决 |
|------|------|
| launch 启动后没话题 | 检查 USB / 设备 serial 是否被占用 |
| IMU 话题没数据 | `enable_gyro:=true enable_accel:=true` |
| 点云没发布 | `pointcloud.enable:=true` |
| TF 找不到 | `publish_tf:=true` |
| 帧率低 | 降分辨率 / 关闭不用的流 |
| 多相机冲突 | 用 `serial_no` 区分 |

## 参考

- [realsense-ros GitHub](https://github.com/IntelRealSense/realsense-ros)
- [realsense2_camera README](https://github.com/IntelRealSense/realsense-ros/blob/ros2-development/realsense2_camera/README.md)
- [realsense2_camera launch 源码](https://github.com/IntelRealSense/realsense-ros/tree/ros2-development/realsense2_camera/launch)