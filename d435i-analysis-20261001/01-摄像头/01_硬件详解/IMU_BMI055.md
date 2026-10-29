# D435i 内置 IMU — Bosch BMI055 详解

> D435i 内置的 6 轴 IMU 是 Bosch BMI055，是相机做视觉-惯性融合的关键。

## 1. BMI055 简介

BMI055 是 Bosch Sensortec 出品的**消费级 6 轴 IMU**：

- 3 轴 ±16 g 加速度计
- 3 轴 ±2000 °/s 陀螺仪
- I²C / SPI 接口
- 16-bit ADC
- 内置温度传感器

> 📄 Datasheet：[Bosch BMI055 Product Page](https://www.bosch-sensortec.com/bst/products/all_products/bmi055)

## 2. 在 D435i 中的角色

D435i 的 BMI055 **直接焊在主板上**，通过板级 I²C 连接到主 ASIC。SDK 把 IMU 当成一个普通 sensor：

```
stream 类型：
- RS2_STREAM_ACCEL → 加速度
- RS2_STREAM_GYRO → 角速度
```

每个 IMU 数据包用深度传感器的硬件时钟打时戳，**保证 IMU 与深度/帧时序同步**。

## 3. 关键规格

### 3.1 加速度计

| 范围 | 分辨率 | 噪声密度 |
|------|--------|----------|
| ±2 g | 0.98 mg | 190 μg/√Hz |
| ±4 g | 1.95 mg | - |
| ±8 g | 3.91 mg | - |
| ±16 g | 7.81 mg | - |

### 3.2 陀螺仪

| 范围 | 分辨率 | 零偏稳定性 |
|------|--------|----------|
| ±125 °/s | 0.004 °/s | ~5 °/h |
| ±250 °/s | 0.008 °/s | - |
| ±500 °/s | 0.016 °/s | - |
| ±1000 °/s | 0.031 °/s | - |
| ±2000 °/s | 0.062 °/s | - |

### 3.3 输出速率

| 模式 | 速率 |
|------|------|
| Normal | 100 Hz / 200 Hz |
| Fast | 1000 Hz（BMI055 自身上限） |
| D435i 默认 | 200 Hz |

## 4. 坐标系与时序

### 4.1 加速度计静止读数

⚠️ **这是新手最容易踩的坑**。

D435i 默认安装朝向（板子正面朝上、相机朝前）：

```
静止时加速度读数（世界系下）：
  ax = 0
  ay = -9.80665   ← 负的！因为加速度计测"惯性反作用力"，正 Y 轴朝下
  az = 0
```

**逻辑**：

- 重力方向永远是 -Z（世界系）
- D435i 的 Y 轴朝下
- 重力反向作用到 Y 轴 = -Y 方向
- 但 Y 是正向，所以读数是 **-9.8**

**判断标准**：

```python
import pyrealsense2 as rs
import numpy as np

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.accel)
pipe.start(cfg)

for _ in range(50):
    f = pipe.wait_for_frames()
    a = f.first_or_default(rs.stream.accel).as_motion_frame().get_motion_data()
    norm = np.linalg.norm([a.x, a.y, a.z])
    assert 9.0 < norm < 10.5, f"静止加速度异常: {norm}"
print("✅ IMU 加速度计读数正常")
```

### 4.2 时戳

所有 IMU 数据包都带 `frame.get_timestamp()`，**单位是毫秒**，相对于设备上电时间。

```python
ts_ms = frame.get_timestamp()        # 设备时间（ms）
ts_domain = frame.get_frame_timestamp_domain()  # 硬件时钟
```

## 5. 校准（核心！）

### 5.1 为什么要校准

D435i 出厂**没有对 IMU 做精细校准**：

- 陀螺仪偏置可能 ±0.1-1 °/s（足以让 1 秒积分出 0.1-1° 的姿态误差）
- 加速度计偏置可能 ±5-50 mg

> 来源：librealsense 官方文档 *"The D435i IMU sensor does not include internal calibration"*

校准后数据会写入相机 EEPROM，**永久生效**。

### 5.2 校准工具

```bash
# 编译 rs-imu-calibration
git clone https://github.com/IntelRealSense/librealsense
cd librealsense
mkdir build && cd build
cmake .. -DBUILD_TOOLS=ON
make rs-imu-calibration -j$(nproc)

# 运行（按提示摆放相机各种姿态）
./tools/rs-imu-calibration/rs-imu-calibration
```

工具会引导你做约 30-60 秒的姿态变化，自动计算陀螺零偏、加速度零偏、刻度因子误差，写回 EEPROM。

**校准前后的差异**（典型值）：

| 项 | 校准前 | 校准后 |
|------|------|------|
| 陀螺零偏 | ±0.5 °/s | ±0.05 °/s |
| 加速度零偏 | ±30 mg | ±3 mg |

### 5.3 二次校准与漂移

如果发现：

- 长时间静止时陀螺有缓慢漂移
- 加速度计读数与 9.80665 偏差持续增大

可再次运行校准。温度变化也会影响，建议每 6 个月校准一次。

详细步骤见 [05_校准/IMU校准.md](../05_校准/IMU校准.md)。

## 6. 与 VIO/SLAM 系统的集成

### 6.1 ORB-SLAM3 with D435i

ORB-SLAM3 需要 IMU 噪声参数：

```
IMU.NoiseGyro: 1.7e-4    # 典型 BMI055 噪声密度 rad/s/√Hz
IMU.NoiseAcc:  2.0e-3    # 典型 BMI055 噪声密度 m/s^2/√Hz
IMU.GyroWalk:  1.9e-5    # 陀螺随机游走 rad/s^2/√Hz
IMU.AccWalk:   3.0e-3    # 加速度随机游走 m/s^3/√Hz
IMU.Frequency: 200       # D435i IMU 默认输出频率

# 校准后参数（参考值）：
TBC: [-0.00764, -0.00173, 0.00990]  # IMU->Camera 平移
RBC: [0.9996, -0.0014, -0.0283, ...]  # IMU->Camera 旋转（四元数）
```

### 6.2 VINS-Mono / VINS-Fusion

VINS 系列通常通过动态初始化估计 IMU 偏置，不需要精确出厂参数，但需要相机-IMU 外参。

### 6.3 RTAB-Map

RTAB-Map 可以直接订阅 `/camera/imu` topic 做松耦合（可选）。做紧耦合需要 OPENVINS 后端。

## 7. SDK API（C++）

```cpp
#include <librealsense2/rs.hpp>

rs2::pipeline pipe;
rs2::config cfg;
cfg.enable_stream(RS2_STREAM_ACCEL);
cfg.enable_stream(RS2_STREAM_GYRO);
pipe.start(cfg);

while (true) {
    rs2::frameset frames = pipe.wait_for_frames();
    
    if (auto accel = frames.first_or_default(RS2_STREAM_ACCEL)) {
        rs2_vector a = accel.as<rs2::motion_frame>().get_motion_data();
        // a.x, a.y, a.z in m/s^2
    }
    
    if (auto gyro = frames.first_or_default(RS2_STREAM_GYRO)) {
        rs2_vector g = gyro.as<rs2::motion_frame>().get_motion_data();
        // g.x, g.y, g.z in rad/s
    }
}
```

## 8. SDK API（Python）

```python
import pyrealsense2 as rs
import numpy as np

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.accel)
cfg.enable_stream(rs.stream.gyro)
pipe.start(cfg)

try:
    while True:
        frames = pipe.wait_for_frames()
        
        accel_frame = frames.first_or_default(rs.stream.accel)
        if accel_frame:
            a = accel_frame.as_motion_frame().get_motion_data()
            a_np = np.array([a.x, a.y, a.z])
            # 注意：静止时 ||a|| ≈ 9.80665
            
        gyro_frame = frames.first_or_default(rs.stream.gyro)
        if gyro_frame:
            g = gyro_frame.as_motion_frame().get_motion_data()
            # g.x, g.y, g.z in rad/s
finally:
    pipe.stop()
```

完整示例见 [02_SDK与编程/示例代码.md](../02_SDK与编程/示例代码.md)。

## 9. 常见坑

| 现象 | 原因 |
|------|------|
| 加速度计静止读数 +9.8 | 相机倒装 / 翻转安装 |
| ORB-SLAM3 初始化失败 | IMU 噪声参数不对 |
| VIO 漂移严重 | 没校准或外参错误 |
| IMU 跳变 | USB 带宽不足 → 丢包 |
| ROS 里 `/camera/imu` 没数据 | 没 enable stream 或参数错误 |

## 10. 替代 IMU 选型

如果 BMI055 不够用，可外接：

| IMU | 优势 | 劣势 |
|-----|------|------|
| BMI088 | 升级版，零漂更好 | 需自己接线 |
| ICM-20948 | 9 轴（含磁力计） | 体积大 |
| VectorNav VN-100 | 工业级 | 价格 $1000+ |

但 D435i 自带的 BMI055 对 **大多数 SLAM / 移动机器人场景够用**。

## 参考

- [librealsense doc/d435i.md](https://github.com/IntelRealSense/librealsense/blob/master/doc/d435i.md)
- [Bosch BMI055 Datasheet](https://www.bosch-sensortec.com/bst/products/all_products/bmi055)
- [Intel 白皮书: IMU calibration for D435i](https://www.intelrealsense.com/download/2741/)