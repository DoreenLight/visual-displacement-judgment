# C++ API 详解

> librealsense2 C++ 编程的核心 API 包括 pipeline、frameset、frame、sensor 等。

## 1. 核心对象

```
rs2::context          → 设备管理器（全局）
   ↓
rs2::device           → 单台相机
   ↓
rs2::sensor           → 单个传感器（depth / rgb / motion）
   ↓
rs2::pipeline         → 数据采集管道（推荐入口）
   ↓
rs2::frameset         → 一帧多流（depth + rgb + imu）
   ↓
rs2::frame            → 单个流的一帧
```

## 2. 第一个程序：打印所有流

```cpp
#include <librealsense2/rs.hpp>
#include <iostream>

int main() {
    // 1. 创建 pipeline
    rs2::pipeline pipe;
    
    // 2. 配置要启用的流
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_ACCEL);
    cfg.enable_stream(RS2_STREAM_GYRO);
    
    // 3. 启动
    rs2::pipeline_profile profile = pipe.start(cfg);
    
    // 4. 主循环
    while (true) {
        rs2::frameset frames = pipe.wait_for_frames();
        
        // 深度
        if (auto depth = frames.get_depth_frame()) {
            std::cout << "Depth: " 
                      << depth.get_width() << "x" << depth.get_height() 
                      << " @ " << depth.get_profile().fps() << " fps" << std::endl;
        }
        
        // 彩色
        if (auto color = frames.get_color_frame()) {
            std::cout << "Color: "
                      << color.get_width() << "x" << color.get_height() << std::endl;
        }
        
        // IMU
        if (auto accel = frames.first_or_default(RS2_STREAM_ACCEL)) {
            rs2_vector a = accel.as<rs2::motion_frame>().get_motion_data();
            std::cout << "Accel: " << a.x << ", " << a.y << ", " << a.z << std::endl;
        }
        
        if (auto gyro = frames.first_or_default(RS2_STREAM_GYRO)) {
            rs2_vector g = gyro.as<rs2::motion_frame>().get_motion_data();
            std::cout << "Gyro: " << g.x << ", " << g.y << ", " << g.z << std::endl;
        }
    }
    
    pipe.stop();
    return 0;
}
```

## 3. 编译

```bash
g++ -std=c++14 main.cpp -o app \
    -I/usr/local/include \
    -L/usr/local/lib \
    -lrealsense2 \
    $(pkg-config --cflags --libs gl glfw3)
```

或者 CMake：

```cmake
cmake_minimum_required(VERSION 3.10)
project(realsense_demo)

set(CMAKE_CXX_STANDARD 14)

find_package(realsense2 REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} realsense2)
```

## 4. 常用 API 速查

### 4.1 pipeline

```cpp
rs2::pipeline pipe;

// 启动（用默认配置）
pipe.start();

// 启动（自定义配置）
rs2::config cfg;
cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 30);
pipe.start(cfg);

// 等待帧（阻塞，超时毫秒）
rs2::frameset frames = pipe.wait_for_frames(5000);

// 异步帧（队列模式）
pipe.start(cfg, [callback](rs2::frame f) {
    // 处理帧
});

// 停止
pipe.stop();
```

### 4.2 frameset

```cpp
// 获取深度
rs2::depth_frame depth = frames.get_depth_frame();

// 获取 RGB
rs2::video_frame color = frames.get_color_frame();

// 获取 IMU（注意 IMU 帧不在 frameset 中，是单独的）
rs2::frame accel = frames.first_or_default(RS2_STREAM_ACCEL);

// 帧计数
size_t size = frames.size();
```

### 4.3 depth_frame 特有 API

```cpp
// 获取 (x, y) 处深度值（米）
float dist = depth.get_distance(x, y);

// 获取分辨率
int w = depth.get_width();
int h = depth.get_height();

// 距离单位转换
// 默认输出为米
```

### 4.4 点云生成

```cpp
rs2::pointcloud pc;
rs2::points points = pc.process(depth_frame);

// 每个点云点的结构
auto vertices = points.get_vertices();
// vertices[i].x, .y, .z (float, meters)
// vertices[i].r, .g, .b (uint8_t) — 仅当传入 color frame 时有效
```

### 4.5 deproject（像素 → 3D）

```cpp
// 获取相机内参
auto intrinsics = depth_frame.get_profile().as<rs2::video_stream_profile>().get_intrinsics();

// 像素坐标 → 3D 坐标
float pixel[2] = {320, 240};
float distance = depth_frame.get_distance(320, 240);
rs2_vector p3d = rs2_deproject_pixel_to_point(intrinsics, pixel, distance);
// p3d.x, .y, .z (米)
```

### 4.6 motion_frame（IMU）

```cpp
rs2::motion_frame accel_frame = frames.first_or_default(RS2_STREAM_ACCEL).as<rs2::motion_frame>();
rs2_vector a = accel_frame.get_motion_data();
// a.x, .y, .z 单位 m/s^2
```

## 5. 处理块（Processing Blocks）

### 5.1 点云

```cpp
rs2::pointcloud pc;
rs2::points points = pc.process(frameset);
```

### 5.2 对齐（depth → color）

```cpp
rs2::align align_to_color(RS2_STREAM_COLOR);
auto aligned = align_to_color.process(frameset);
// aligned.get_depth_frame() 现在与 color 帧对齐
```

### 5.3 颜色映射（深度可视化）

```cpp
rs2::colorizer color_map;
rs2::video_frame colorized = color_map.process(frameset.get_depth_frame());
// colorized 是 BGR 图像，可直接 cv::imshow
```

### 5.4 滤波

```cpp
rs2::decimation_filter dec;     // 下采样
rs2::spatial_filter spat;       // 空间滤波（去噪）
rs2::temporal_filter temp;      // 时序滤波（去闪）
rs2::hole_filling_filter hole;  // 空洞填充

rs2::frame filtered = dec.process(spat.process(temp.process(hole.process(depth))));
```

## 6. 录制与回放

### 6.1 录制到文件

```cpp
rs2::pipeline pipe;
pipe.start();

std::ofstream file("record.bag", std::ios::binary);
rs2::frame f = pipe.wait_for_frames();
file.write((const char*)f.get_data(), f.get_data_size());
```

或用 SDK 自带工具：

```bash
rs-record -f output.bag
```

### 6.2 回放文件

```cpp
rs2::pipeline pipe;
rs2::config cfg;
cfg.enable_device_from_file("output.bag");

pipe.start(cfg);
while (true) {
    rs2::frameset frames = pipe.wait_for_frames();
    // 处理帧
}
```

## 7. 多设备

```cpp
rs2::context ctx;
auto devices = ctx.query_devices();
for (auto&& dev : devices) {
    std::cout << dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) << std::endl;
}

// 按 serial number 启动指定设备
rs2::config cfg;
cfg.enable_device("213722067000");
cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

rs2::pipeline pipe;
pipe.start(cfg);
```

## 8. 高级：自定义 IMU 输出频率

```cpp
// 启动后修改 IMU 频率
rs2::pipeline pipe;
pipe.start();

auto sensors = pipe.get_active_profile().get_device().query_sensors();
for (auto&& sensor : sensors) {
    if (sensor.get_info(RS2_CAMERA_INFO_NAME) == "Motion Module") {
        sensor.set_option(RS2_OPTION_MOTION_MODULE_FREQUENCY, 200);  // Hz
    }
}
```

## 9. 错误处理

```cpp
try {
    pipe.start(cfg);
} catch (const rs2::error& e) {
    std::cerr << "RealSense error: " << e.what() << std::endl;
}
```

## 10. 性能 Tips

1. **复用 frameset**：避免每帧都 enable/disable 流
2. **用 align 而不是 deproject**：批量对齐比逐像素快 10×
3. **异步模式**：`pipe.start(cfg, callback)` 减少阻塞
4. **关闭不需要的流**：节省 USB 带宽
5. **用 `frames.first_or_default()`**：避免不必要的 cast

## 11. 与 OpenCV 互操作

```cpp
#include <opencv2/opencv.hpp>

// video_frame → cv::Mat（不拷贝）
rs2::video_frame color = frames.get_color_frame();
cv::Mat img(cv::Size(640, 480), CV_8UC3, (void*)color.get_data(), color.get_stride_in_bytes());
cv::imshow("Color", img);
cv::waitKey(1);

// depth_frame → cv::Mat
rs2::depth_frame depth = frames.get_depth_frame();
cv::Mat depth_img(cv::Size(640, 480), CV_16UC1, (void*)depth.get_data(), depth.get_stride_in_bytes());
```

完整示例见 [示例代码.md](示例代码.md)。

## 参考

- [librealsense C++ API 文档](https://intelrealsense.github.io/librealsense2/doxygen/)
- [librealsense API 概念](https://github.com/IntelRealSense/librealsense/blob/master/doc/api-concepts.md)
- [librealsense examples](https://github.com/IntelRealSense/librealsense/tree/master/examples)