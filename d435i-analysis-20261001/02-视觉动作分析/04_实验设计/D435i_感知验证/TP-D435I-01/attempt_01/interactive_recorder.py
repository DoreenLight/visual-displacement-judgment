#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interactive_recorder.py - D435i 交互式多动作多模态数据录制系统
完全由用户手动掌控录制启停：
1. 实时预览：彩色 RGB 与彩色化对齐深度图同屏并排显示；
2. 动作快捷切换：按键盘 1~8 快速切换动作标签 (S1静止/S2握柄/屈伸FE/偏摆RU/代偿等)；
3. 空格键启停：按一次空格开始录制，再按一次空格停止并自动保存；
4. 数据保存：自动存入 07_资料库/D435i_感知验证/<action_name>_<timestamp>/
   - color.mp4 (RGB高清视频)
   - depth_raw.npz (无损16位原始深度时序数据，单位毫米)
   - timestamps.csv (硬件时钟与单调时钟时间戳)
   - run.json (相机内参、型号、标尺、动作名称与环境元数据)
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import cv2

try:
    import pyrealsense2 as rs
    PYREALSENSE_AVAILABLE = True
except ImportError:
    PYREALSENSE_AVAILABLE = False


# 预设的八大标准测试动作
ACTION_PRESETS = [
    ("S1_natural_sitting", "自然坐姿静止 (S1 - 测稳定性)"),
    ("S2_grip_posture", "握柄姿态静止 (S2 - 测握柄抗遮挡)"),
    ("wrist_FE_flex_ext", "标准腕屈伸 (FE - 10次往返，肩肘不动)"),
    ("wrist_RU_deviation", "标准桡尺偏 (RU - 10次往返，肩肘不动)"),
    ("wrist_circumduction", "腕部圆周回转 (顺时针5圈，逆时针5圈)"),
    ("shoulder_compensation", "肩部代偿动作 (故意耸肩/身体前倾带动手腕)"),
    ("elbow_compensation", "肘部代偿动作 (故意飞肘/肘屈伸带动手腕)"),
    ("combined_compensation", "肩肘协同代偿 (耸肩+飞肘复合代偿)")
]


class D435iInteractiveRecorder:
    def __init__(
        self,
        width: int = 848,
        height: int = 480,
        fps: int = 30,
        save_root: str = "../../../../07_资料库/D435i_感知验证"
    ):
        self.width = width
        self.height = height
        self.fps = fps
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_root = os.path.abspath(os.path.join(curr_dir, save_root))
        os.makedirs(self.save_root, exist_ok=True)

        self.current_action_idx = 0
        self.is_recording = False
        self.record_frames_color = []
        self.record_frames_depth = []
        self.record_timestamps = []
        self.start_record_time = 0.0

        self.pipeline = None
        self.align = None
        self.depth_scale = 0.001
        self.intrinsics_dict = {}
        self.device_info = {}

    def init_camera(self) -> bool:
        """初始化 D435i 物理相机并启动对齐流"""
        if not PYREALSENSE_AVAILABLE:
            print("[错误] 未安装 pyrealsense2 库！")
            return False

        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            print("[错误] 未检测到连接的 Intel RealSense 设备！")
            print("请检查：1. USB 3.0 数据线是否插紧；2. 是否插在电脑的 USB 3.0 (蓝色/SS) 接口上。")
            return False

        dev = devices[0]
        self.device_info = {
            "name": dev.get_info(rs.camera_info.name),
            "serial_number": dev.get_info(rs.camera_info.serial_number),
            "firmware": dev.get_info(rs.camera_info.firmware_version)
        }
        print(f"[成功] 识别到相机: {self.device_info['name']} (SN: {self.device_info['serial_number']}, FW: {self.device_info['firmware']})")

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        # 深度传感器标尺
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # 激光发射器开启
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1.0)

        # 提取对齐后的彩色内参
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        self.intrinsics_dict = {
            "width": intr.width,
            "height": intr.height,
            "fx": intr.fx,
            "fy": intr.fy,
            "ppx": intr.ppx,
            "ppy": intr.ppy,
            "model": str(intr.model),
            "coeffs": list(intr.coeffs),
            "depth_scale": self.depth_scale
        }

        print(f"[就绪] 深度与彩色流已对齐 ({self.width}x{self.height} @ {self.fps}fps)")
        print("正在进行相机自动曝光与测距预热，请稍候...")
        for _ in range(30):
            frames = self.pipeline.wait_for_frames()
            self.align.process(frames)
        print("[完成] 预热完毕，进入实时交互预览！\n")
        return True

    def start_recording(self):
        """开始录制当前动作"""
        self.is_recording = True
        self.record_frames_color = []
        self.record_frames_depth = []
        self.record_timestamps = []
        self.start_record_time = time.time()
        action_name, action_desc = ACTION_PRESETS[self.current_action_idx]
        print(f"\n>>> [● 开始录制] 当前动作: {action_name} ({action_desc})")
        print(">>> 动作完成后，再次按下【空格键】即可停止录制并保存。")

    def stop_and_save(self):
        """停止录制并将当前片段打包存盘"""
        self.is_recording = False
        duration = time.time() - self.start_record_time
        total_frames = len(self.record_frames_color)
        action_name, action_desc = ACTION_PRESETS[self.current_action_idx]
        print(f"\n>>> [■ 停止录制] 录制时长: {duration:.2f} 秒，有效帧数: {total_frames} 帧")

        if total_frames < 10:
            print("[警告] 录制帧数过短 (<10帧)，已放弃保存。")
            return

        # 创建目标存储目录
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        run_id = f"RUN_{action_name}_{timestamp_str}"
        run_dir = os.path.join(self.save_root, run_id)
        os.makedirs(run_dir, exist_ok=True)
        print(f"正在保存数据至: {run_dir} ...")

        # 1. 保存彩色视频 (color.mp4)
        video_path = os.path.join(run_dir, "color.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, float(self.fps), (self.width, self.height))
        for img in self.record_frames_color:
            out.write(img)
        out.release()

        # 2. 保存原始 16位 毫米深度时序 (depth_raw.npz - 无损压缩)
        depth_array = np.array(self.record_frames_depth, dtype=np.uint16)
        depth_path = os.path.join(run_dir, "depth_raw.npz")
        np.savez_compressed(depth_path, depth_mm=depth_array)

        # 3. 保存时间戳长表 (timestamps.csv)
        ts_path = os.path.join(run_dir, "timestamps.csv")
        with open(ts_path, "w", encoding="utf-8") as f:
            f.write("frame_id,t_rgb_device_s,t_depth_device_s,t_host_mono_s\n")
            for ts in self.record_timestamps:
                f.write(f"{ts['frame_id']},{ts['t_rgb_device_s']:.6f},{ts['t_depth_device_s']:.6f},{ts['t_host_mono_s']:.6f}\n")

        # 4. 保存 run.json 元数据
        metadata = {
            "schema_version": "wrist-rehab/v1",
            "run_id": run_id,
            "action_id": action_name,
            "action_description": action_desc,
            "device": self.device_info,
            "intrinsics": self.intrinsics_dict,
            "total_frames": total_frames,
            "duration_s": round(duration, 3),
            "fps": self.fps,
            "timestamp_saved": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "files": {
                "color_video": "color.mp4",
                "depth_data": "depth_raw.npz",
                "timestamps": "timestamps.csv"
            }
        }
        json_path = os.path.join(run_dir, "run.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"[保存成功!] 该片段已归档完成。")
        print(f"提示：可按数字键 1~8 切换下一个动作，准备就绪后按【空格键】继续录制。\n")

    def run_gui(self):
        """主交互循环"""
        if not self.init_camera():
            return

        window_name = "D435i Multi-Action Interactive Recorder"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 500)

        frame_idx = 0
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)

                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                t_host_mono = time.monotonic()
                t_rgb_dev = color_frame.get_timestamp() / 1000.0
                t_depth_dev = depth_frame.get_timestamp() / 1000.0

                color_bgr = np.asanyarray(color_frame.get_data())
                depth_raw = np.asanyarray(depth_frame.get_data())  # uint16 (mm)

                # 录制缓存
                if self.is_recording:
                    self.record_frames_color.append(color_bgr.copy())
                    self.record_frames_depth.append(depth_raw.copy())
                    self.record_timestamps.append({
                        "frame_id": len(self.record_frames_color) - 1,
                        "t_rgb_device_s": t_rgb_dev,
                        "t_depth_device_s": t_depth_dev,
                        "t_host_mono_s": t_host_mono
                    })

                # 生成用于显示的彩色化深度图
                # 将 0.4m ~ 2.0m 的深度映射为伪彩色图
                depth_vis = np.clip(depth_raw, 400, 2200)
                depth_norm = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

                # 拼接左右视图 (左RGB，右深度)
                display_img = np.hstack((color_bgr, depth_color))

                # 叠加屏幕 HUD 状态栏
                action_name, action_desc = ACTION_PRESETS[self.current_action_idx]
                h_disp, w_disp = display_img.shape[:2]

                # 顶部遮罩条
                cv2.rectangle(display_img, (0, 0), (w_disp, 45), (20, 20, 20), -1)
                cv2.putText(
                    display_img,
                    f"Action [{self.current_action_idx + 1}/8]: {action_desc}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                # 录制状态标识
                if self.is_recording:
                    elapsed = time.time() - self.start_record_time
                    frames_cnt = len(self.record_frames_color)
                    cv2.circle(display_img, (w_disp - 240, 24), 9, (0, 0, 255), -1)
                    cv2.putText(
                        display_img,
                        f"REC: {elapsed:.1f}s ({frames_cnt}f)",
                        (w_disp - 220, 31),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
                else:
                    cv2.putText(
                        display_img,
                        "[PAUSED / READY]",
                        (w_disp - 230, 31),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (100, 255, 100),
                        2
                    )

                # 底部操作指引
                cv2.rectangle(display_img, (0, h_disp - 32), (w_disp, h_disp), (15, 15, 15), -1)
                tip_text = "SPACE: Start/Stop Record | 1-8: Switch Action | TAB: Next Action | Q/ESC: Quit"
                cv2.putText(display_img, tip_text, (15, h_disp - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                cv2.imshow(window_name, display_img)

                # 键盘响应
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:  # Q 或 ESC 退出
                    if self.is_recording:
                        self.stop_and_save()
                    break
                elif key == 32:  # 空格键：启停录制
                    if not self.is_recording:
                        self.start_recording()
                    else:
                        self.stop_and_save()
                elif key == 9:  # TAB 键：循环切换动作
                    if not self.is_recording:
                        self.current_action_idx = (self.current_action_idx + 1) % len(ACTION_PRESETS)
                        print(f"切换动作 -> {ACTION_PRESETS[self.current_action_idx][1]}")
                elif ord('1') <= key <= ord('8'):  # 数字键 1-8 切换动作
                    if not self.is_recording:
                        self.current_action_idx = key - ord('1')
                        print(f"切换动作 -> {ACTION_PRESETS[self.current_action_idx][1]}")

                frame_idx += 1

        finally:
            cv2.destroyAllWindows()
            if self.pipeline:
                self.pipeline.stop()
                print("\n[退出] 相机流已安全释放。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D435i Interactive Multi-Action Recorder")
    parser.add_argument("--width", type=int, default=848, help="流分辨率宽")
    parser.add_argument("--height", type=int, default=480, help="流分辨率高")
    parser.add_argument("--fps", type=int, default=30, help="帧率")
    args = parser.parse_args()

    recorder = D435iInteractiveRecorder(width=args.width, height=args.height, fps=args.fps)
    recorder.run_gui()
