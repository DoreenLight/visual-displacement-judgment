# Python API（pyrealsense2）详解

> pyrealsense2 是 librealsense2 的 Python 绑定，接口与 C++ 几乎一致。

## 1. 安装

```bash
# Ubuntu 22.04 / 20.04
sudo apt-get install python3-pyrealsense2

# 或 pip（任意平台）
pip install pyrealsense2
```

验证：

```python
import pyrealsense2 as rs
print(rs.__version__)  # 应输出 2.x.x
```

## 2. 第一个程序

```python
import pyrealsense2 as rs
import numpy as np

pipeline = rs.pipeline()
config = rs.config()

# 配置要启用的流
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.accel)
config.enable_stream(rs.stream.gyro)

# 启动
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        
        if depth:
            print(f"Depth: {depth.get_width()}x{depth.get_height()}")
        
        if color:
            print(f"Color: {color.get_width()}x{color.get_height()}")
        
        # IMU
        accel_frame = frames.first_or_default(rs.stream.accel)
        if accel_frame:
            a = accel_frame.as_motion_frame().get_motion_data()
            print(f"Accel: {a.x:.3f}, {a.y:.3f}, {a.z:.3f}")
        
        gyro_frame = frames.first_or_default(rs.stream.gyro)
        if gyro_frame:
            g = gyro_frame.as_motion_frame().get_motion_data()
            print(f"Gyro: {g.x:.3f}, {g.y:.3f}, {g.z:.3f}")
finally:
    pipeline.stop()
```

## 3. numpy 互操作（最常用）

```python
import numpy as np

# 深度图 → numpy array
depth_image = np.asanyarray(depth_frame.get_data())
# shape: (480, 640), dtype: uint16 (Z16 = 深度毫米值)

# 彩色图 → numpy array (OpenCV 兼容)
color_image = np.asanyarray(color_frame.get_data())
# shape: (480, 640, 3), dtype: uint8, BGR

# 直接显示
import cv2
cv2.imshow('Color', color_image)
cv2.imshow('Depth', depth_image)  # uint16，需 colormap
cv2.waitKey(1)
```

## 4. 点云生成

```python
pc = rs.pointcloud()
points = pc.process(depth_frame)

# 获取顶点数组
vtx = np.asanyarray(points.get_vertices())
# vtx[i] = [x, y, z]
# shape: (N, 3)

# 获取纹理坐标
tex = np.asanyarray(points.get_texture_coordinates())
# shape: (N, 2), values in [0, 1]

# 获取带颜色的点云（需要传入 color_frame）
points = pc.process(depth_frame)
points.consumer.process(color_frame)  # 颜色附加

vtx = np.asanyarray(points.get_vertices())
# vtx[i] = [x, y, z, r, g, b]  # RGBA
```

### 用 open3d 可视化

```python
import open3d as o3d

# 构造点云
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(vtx[:, :3])

# 颜色（如果有）
if vtx.shape[1] >= 6:
    pcd.colors = o3d.utility.Vector3dVector(vtx[:, 3:6] / 255.0)

o3d.visualization.draw_geometries([pcd])
```

## 5. 像素 → 3D 转换

```python
# 获取深度图内参
intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
# intrinsics.fx, fy, ppx, ppy, coeffs, width, height

# 像素 → 3D
def pixel_to_3d(depth_frame, x, y):
    depth = depth_frame.get_distance(x, y)  # 米
    pt = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
    return pt  # [X, Y, Z] 米

# 3D → 像素
def point_3d_to_pixel(point_3d):
    px = rs.rs2_project_point_to_pixel(intrinsics, point_3d)
    return px  # [u, v]
```

## 6. 对齐（align）

```python
align = rs.align(rs.stream.color_frame)

frames = pipeline.wait_for_frames()
aligned = align.process(frames)

# aligned 里的 depth 已与 color 对齐
aligned_depth = aligned.get_depth_frame()
aligned_color = aligned.get_color_frame()
```

## 7. 滤波

```python
# 各种滤波器（pipeline 形式）
dec = rs.decimation_filter()     # 下采样（速度快）
spat = rs.spatial_filter()       # 空间滤波
temp = rs.temporal_filter()      # 时序滤波
hole = rs.hole_filling_filter()  # 空洞填充

filtered_depth = dec.process(depth)
filtered_depth = spat.process(filtered_depth)
filtered_depth = temp.process(filtered_depth)
filtered_depth = hole.process(filtered_depth)
```

## 8. 颜色映射（可视化）

```python
colorizer = rs.colorizer()

colorized = colorizer.process(depth_frame)
# 输出 BGR 图像
color_img = np.asanyarray(colorized.get_data())
cv2.imshow('Depth Colorized', color_img)
```

## 9. 录制与回放

### 9.1 录制

```python
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 写到 .bag
config.enable_record_to_file('output.bag')

pipeline.start(config)
# ... 主循环 ...
pipeline.stop()
```

### 9.2 回放

```python
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file('output.bag')

pipeline.start(config)
while True:
    frames = pipeline.wait_for_frames()
    depth = frames.get_depth_frame()
    color = frames.get_color_frame()
    # ... 处理 ...
pipeline.stop()
```

## 10. 多设备

```python
ctx = rs.context()
for dev in ctx.query_devices():
    serial = dev.get_info(rs.camera_info.serial_number)
    name = dev.get_info(rs.camera_info.name)
    print(f"{name}: {serial}")

# 启动指定设备
config = rs.config()
config.enable_device('213722067000')
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline = rs.pipeline()
pipeline.start(config)
```

## 11. 读取 IMU 数据

```python
import numpy as np

config = rs.config()
config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

pipeline = rs.pipeline()
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        
        for stream in [rs.stream.accel, rs.stream.gyro]:
            f = frames.first_or_default(stream)
            if f:
                data = f.as_motion_frame().get_motion_data()
                t = f.get_timestamp()  # 设备时间 ms
                print(f"{stream}: t={t}ms, x={data.x:.3f}, y={data.y:.3f}, z={data.z:.3f}")
finally:
    pipeline.stop()
```

完整示例见 [示例代码.md](示例代码.md)。

## 12. 帧同步回调（异步模式）

```python
import pyrealsense2 as rs
from threading import Lock

class FrameCallback:
    def __init__(self):
        self.lock = Lock()
        self.frames = None
    
    def callback(self, frame):
        # 处理单帧（多线程）
        if frame.is_depth_frame():
            with self.lock:
                # ... 处理深度 ...
                pass

callback = FrameCallback()

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config, callback.callback)
```

## 13. 异常处理

```python
try:
    pipeline.start(config)
except RuntimeError as e:
    print(f"启动失败: {e}")
    # 常见原因：
    # - 相机未连接
    # - USB 带宽不足
    # - 已有程序占用
```

## 14. 常用库组合

| 库 | 用途 |
|----|------|
| `numpy` | 数组处理 |
| `opencv-python` | 图像处理、可视化 |
| `open3d` | 点云处理与可视化 |
| `matplotlib` | 绘图（深度直方图等）|
| `pyqtgraph` | 实时 IMU 曲线 |

## 15. 性能 Tips

1. **`np.asanyarray` 不拷贝数据**：性能等同直接访问
2. **避免每帧 `.get_data()` 多次**：一次缓存
3. **用 `frames.combine` 合并帧**：少一次遍历
4. **批量 deproject**：用 `rs2_deproject_pixel_to_point` 批量版（如果可用）

## 参考

- [librealsense Python wrapper 文档](https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python)
- [pyrealsense2 examples](https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python/examples)
- [Python Tutorial 1 - Depth](https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/python-tutorial-1-depth.py)