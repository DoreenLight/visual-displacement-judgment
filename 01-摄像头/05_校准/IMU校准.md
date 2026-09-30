# IMU 校准

> D435i 出厂**没有对 IMU 做精细校准**，跑 VIO/SLAM 之前必须校准。

## 1. 为什么必须校准

> 来源：[librealsense doc/d435i.md](https://github.com/IntelRealSense/librealsense/blob/master/doc/d435i.md)
> "The D435i IMU sensor does not include internal calibration, which may manifest itself with non-zero angular velocities produced at idle mode and the gravity (accel) force measured being not equal to 9.80665."

具体影响：

| 问题 | 现象 |
|------|------|
| 陀螺零偏 | 静止时仍输出角速度 → VIO 缓慢漂移 |
| 加速度零偏 | 重力测量不准 → 姿态估计偏差 |
| 刻度因子误差 | 真实旋转 1° 时测量 0.95° → 累计误差 |

校准后写入相机 EEPROM，**永久生效**。

## 2. 校准工具：rs-imu-calibration

### 2.1 编译

```bash
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
mkdir build && cd build
cmake .. -DBUILD_TOOLS=ON
make rs-imu-calibration -j$(nproc)

# 二进制路径：
# build/tools/rs-imu-calibration/rs-imu-calibration
```

### 2.2 运行

```bash
# 1. 连接相机（注意：静止平放）
# 2. 运行工具
./rs-imu-calibration

# 输出会引导你做：
# - 静止 5 秒（采集陀螺零偏）
# - 沿 X 轴慢转 180°（重力测量 → X 方向）
# - 沿 Y 轴慢转 180°（Y 方向）
# - 沿 Z 轴慢转 180°（Z 方向）
# - 静止 5 秒（结束）
```

### 2.3 操作要点

| 步骤 | 操作 |
|------|------|
| 静止 | 把 D435i 平放在桌面，**不要触碰** 5 秒 |
| X 轴转 | 把相机**绕**相机前后轴转 90°，保持 5 秒 |
| Y 轴转 | 把相机**绕**右左轴转 90°，保持 5 秒 |
| Z 轴转 | 把相机**绕**垂直轴转 90°，保持 5 秒 |

工具会显示：

```
Step 1: Static
Position: [0.000, 0.000, 9.806]    ← 当前读数
Reference: [0.000, 0.000, 9.806]   ← 理论值
```

每步完成后会闪烁 LED 或打印 OK，等待下一步。

### 2.4 校准结果

```bash
# 工具完成后
# 数据写入相机 NVRAM（永久）

# 验证（重启 SDK 后）
rs-enumerate-devices -c
# 输出含：
#    Gyro Bias: [0.0012, 0.0009, -0.0023]    ← 校准后的偏置
#    Accel Bias: [0.05, -0.02, 0.03]
```

## 4. 校准前后对比（典型值）

| 指标 | 校准前 | 校准后 |
|------|--------|--------|
| 陀螺零偏 X | 0.5 °/s | 0.05 °/s |
| 陀螺零偏 Y | 0.3 °/s | 0.04 °/s |
| 陀螺零偏 Z | 0.4 °/s | 0.06 °/s |
| 加速度零偏 X | 30 mg | 3 mg |
| 加速度零偏 Y | 25 mg | 2 mg |
| 加速度零偏 Z | 20 mg | 2 mg |

## 5. 进阶：温度补偿

BMI055 受温度影响：

- 陀螺零偏温度系数 ~ 0.01 °/s / °C
- 加速度零偏温度系数 ~ 0.1 mg / °C

D435i 工作 30 分钟后温度上升 5-10°C，**数据会有轻微漂移**。建议：

1. 校准时让相机预热 5 分钟
2. 长期运行时考虑外加温度传感器做实时补偿

## 7. 何时需要重新校准

| 场景 | 建议 |
|------|------|
| 首次使用 | ✅ 必做 |
| 长时间未用 (> 6 月) | ✅ 重做 |
| 跌落 / 撞击 | ✅ 重做 |
| 工作温度大幅变化 | ⚠️ 观察后再决定 |
| 换主板 | ✅ 必做 |
| 日常使用 | ❌ 不需要 |

## 8. 校准失败排查

| 现象 | 原因 |
|------|------|
| 工具无法识别设备 | USB 权限 / librealsense 版本不匹配 |
| 步骤超时 | 没按提示转动相机 |
| 数据偏差过大 | 转得太快/有振动 |
| 写入 EEPROM 失败 | udev 权限问题 |

udev 权限修复：

```bash
# 添加规则
sudo tee /etc/udev/rules.d/99-realsense-libusb.rules << 'EOF'
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b07", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b3a", MODE="0666"
EOF

sudo udevadm control --reload-rules && sudo udevadm trigger
```

## 9. 校准数据备份

校准数据存在相机 EEPROM，**换相机需要重做**。如需备份：

- 不能直接读取 EEPROM（Intel 未公开协议）
- 校准后导出 SDK 内的 intrinsic 参数：

```python
import pyrealsense2 as rs

ctx = rs.context()
dev = ctx.query_devices()[0]
sensors = dev.query_sensors()

for s in sensors:
    print(f"\n{s.get_info(rs.camera_info.name)}:")
    for opt in s.get_supported_options():
        val = s.get_option(opt)
        print(f"  {opt}: {val}")
```

## 10. 用 ROS 验证校准

### 10.1 看静止加速度

```bash
ros2 topic echo /camera/accel/sample
# 应输出：
# linear_acceleration:
#   x: 0.0
#   y: -9.8
#   z: 0.0
```

### 10.2 看静止陀螺

```bash
ros2 topic echo /camera/gyro/sample
# 应输出：
# angular_velocity:
#   x: 0.0 ± 0.05
#   y: 0.0 ± 0.05
#   z: 0.0 ± 0.05
```

### 10.3 自动检查脚本

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import numpy as np

class ImuCheck(Node):
    def __init__(self):
        super().__init__('imu_check')
        self.sub_accel = self.create_subscription(
            Imu, '/camera/accel/sample', self.accel_cb, 10
        )
        self.sub_gyro = self.create_subscription(
            Imu, '/camera/gyro/sample', self.gyro_cb, 10
        )
        self.accel_samples = []
        self.gyro_samples = []
    
    def accel_cb(self, msg):
        a = np.array([msg.linear_acceleration.x,
                      msg.linear_acceleration.y,
                      msg.linear_acceleration.z])
        self.accel_samples.append(a)
        
        if len(self.accel_samples) > 200:
            mean_a = np.mean(self.accel_samples, axis=0)
            norm = np.linalg.norm(mean_a)
            if not (9.0 < norm < 10.5):
                self.get_logger().warn(f'加速度异常 {norm:.3f}')
            else:
                self.get_logger().info(f'✓ 加速度正常 ({norm:.3f})')
            self.accel_samples = []
    
    def gyro_cb(self, msg):
        g = np.array([msg.angular_velocity.x,
                      msg.angular_velocity.y,
                      msg.angular_velocity.z])
        self.gyro_samples.append(g)
        
        if len(self.gyro_samples) > 200:
            mean_g = np.mean(self.gyro_samples, axis=0)
            if np.linalg.norm(mean_g) > 0.05:
                self.get_logger().warn(f'陀螺漂移 {mean_g}')
            self.gyro_samples = []

def main():
    rclpy.init()
    node = ImuCheck()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
```

## 11. 常见问题

| Q | A |
|----|----|
| 校准后数据反而变差？ | 可能操作过快，重做时慢慢转动 |
| 同一台相机多次校准结果不一样？ | 正常，每次转动方式略有差异，结果会收敛 |
| 校准数据可以移植吗？ | 不可以，每个 EEPROM 独立 |
| 室温变化后需要重新校准？ | 建议每 6 个月一次 |
| 是否需要专业设备？ | 不需要，手动转动即可 |

## 参考

- [librealsense tools/rs-imu-calibration](https://github.com/IntelRealSense/librealsense/tree/development/tools/rs-imu-calibration)
- [Intel 白皮书: IMU calibration for D435i](https://www.intelrealsense.com/download/2741/)
- [Bosch BMI055 Datasheet](https://www.bosch-sensortec.com/bst/products/all_products/bmi055)