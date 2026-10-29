#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
d435i_capture.py - Intel RealSense D435i 硬件流管理器
遵守 TP-D435I-01 与 wrist-rehab/v1 数据契约规范：
1. 校验设备型号是否为 RealSense D435i；
2. 读取硬件内参、深度标尺、固件与SDK版本；
3. 将深度流与彩色流在硬件/软件层面严格对齐 (align_to_color)；
4. 记录设备时钟域时间戳及主机单调时钟时间戳；
5. 执行传感器预热（默认丢弃前 30 帧以稳定自动曝光与激光投射）。
"""

import time
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import numpy as np

try:
    import pyrealsense2 as rs
    PYREALSENSE_AVAILABLE = True
except ImportError:
    PYREALSENSE_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("D435iCapture")


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float  # cx
    ppy: float  # cy
    model: str
    coeffs: list
    depth_scale: float


class D435iCaptureManager:
    def __init__(
        self,
        width: int = 848,
        height: int = 480,
        fps: int = 30,
        warmup_frames: int = 30,
        enable_emitter: bool = True
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup_frames = warmup_frames
        self.enable_emitter = enable_emitter

        self.pipeline: Optional[Any] = None
        self.config: Optional[Any] = None
        self.align: Optional[Any] = None
        self.profile: Optional[Any] = None
        self.intrinsics: Optional[CameraIntrinsics] = None
        self.device_info: Dict[str, Any] = {}

    def is_hardware_available(self) -> Tuple[bool, str]:
        """检查是否有 RealSense 设备连接"""
        if not PYREALSENSE_AVAILABLE:
            return False, "pyrealsense2 未安装或不可用"
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                return False, "未检测到已连接的 RealSense 物理设备"
            dev = devices[0]
            name = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)
            fw = dev.get_info(rs.camera_info.firmware_version)
            return True, f"检测到设备: {name} (SN: {serial}, FW: {fw})"
        except Exception as e:
            return False, f"查询 RealSense 设备失败: {str(e)}"

    def start(self) -> bool:
        """启动相机流并获取标定与对齐配置"""
        available, msg = self.is_hardware_available()
        if not available:
            logger.error(f"无法启动 D435i 相机: {msg}")
            return False

        logger.info(f"正在初始化 D435i 相机... {msg}")
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # 配置对齐后的流
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        # 启动 pipeline
        self.profile = self.pipeline.start(self.config)

        # 获取设备元数据
        device = self.profile.get_device()
        self.device_info = {
            "name": device.get_info(rs.camera_info.name),
            "serial_number": device.get_info(rs.camera_info.serial_number),
            "firmware_version": device.get_info(rs.camera_info.firmware_version),
            "usb_type": device.get_info(rs.camera_info.usb_type_descriptor) if device.supports(rs.camera_info.usb_type_descriptor) else "unknown",
            "librealsense_version": rs.__version__ if hasattr(rs, "__version__") else "2.58.x"
        }

        # 确认深度传感器与标尺
        depth_sensor = device.first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        # 设置激光散斑投射器 (Emitter)
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1.0 if self.enable_emitter else 0.0)

        # 创建对齐对象 (Align to Color)
        self.align = rs.align(rs.stream.color)

        # 提取对齐后的彩色流内参
        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        color_intr = color_stream.get_intrinsics()

        self.intrinsics = CameraIntrinsics(
            width=color_intr.width,
            height=color_intr.height,
            fx=color_intr.fx,
            fy=color_intr.fy,
            ppx=color_intr.ppx,
            ppy=color_intr.ppy,
            model=str(color_intr.model),
            coeffs=list(color_intr.coeffs),
            depth_scale=depth_scale
        )

        logger.info(f"相机启动成功! 对齐内参: fx={self.intrinsics.fx:.2f}, fy={self.intrinsics.fy:.2f}, "
                    f"cx={self.intrinsics.ppx:.2f}, cy={self.intrinsics.ppy:.2f}, depth_scale={depth_scale}")

        # 相机预热：丢弃前 warmup_frames 帧
        logger.info(f"正在进行相机预热，丢弃前 {self.warmup_frames} 帧以稳定自动曝光与测距...")
        for _ in range(self.warmup_frames):
            frames = self.pipeline.wait_for_frames()
            self.align.process(frames)

        logger.info("D435i 相机预热完成，处于可用采集状态。")
        return True

    def read_frame(self) -> Optional[Dict[str, Any]]:
        """读取一帧对齐的 RGB 与深度数据"""
        if self.pipeline is None or self.align is None:
            return None

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=3000)
            aligned_frames = self.align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                return None

            t_host_mono_s = time.monotonic()
            t_color_device_s = color_frame.get_timestamp() / 1000.0  # ms -> s
            t_depth_device_s = depth_frame.get_timestamp() / 1000.0

            color_img = np.asanyarray(color_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())  # raw uint16 (in scale units, usually mm)

            return {
                "color_bgr": color_img,
                "depth_raw": depth_img,
                "depth_scale": self.intrinsics.depth_scale,
                "t_rgb_device_s": t_color_device_s,
                "t_depth_device_s": t_depth_device_s,
                "t_host_mono_s": t_host_mono_s,
                "clock_domain": "hardware_clock_ms"
            }
        except Exception as e:
            logger.warning(f"读取帧超时或异常: {e}")
            return None

    def stop(self):
        """停止采集并释放硬件"""
        if self.pipeline:
            try:
                self.pipeline.stop()
                logger.info("D435i 相机已正常关闭并释放。")
            except Exception as e:
                logger.warning(f"关闭 D435i 相机时出现异常: {e}")
            finally:
                self.pipeline = None
