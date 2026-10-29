#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
realtime_biofeedback_app.py - D435i 实时闭环视觉生物反馈康复训练系统
针对右手腕关节屈伸 (FE) 训练：
1. 持续获取 RealSense D435i 硬件对齐彩色与深度画面 (848x480 @ 30FPS)
2. 持续解算上肢运动链 3D 物理空间坐标 (肩-肘-腕-指)
3. 实时计算腕关节有效屈伸角度 (FE Angle) 与运动进度
4. 毫秒级实时代偿识别：耸肩代偿 (Shoulder Shrugging)、飞肘代偿 (Flying Elbow)、躯干侧倾
5. 镜像级生物反馈可视化界面 (HUD)：
   - 镜面翻转呈现，符合直觉反应
   - 动态变色骨骼（正常亮绿，代偿瞬间闪烁高亮红）
   - 实时环形角度仪表盘（直观显示角度刻度与目标线）
   - 醒目警报横幅与即时教练指导语（中文高清渲染）
   - 自动动作周期识别与标准/代偿达标计数
"""

import os
import sys
import time
import math
import json
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

try:
    import pyrealsense2 as rs
    PYREALSENSE_AVAILABLE = True
except ImportError:
    PYREALSENSE_AVAILABLE = False

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 确保本脚本所在目录始终位于 sys.path 首位
curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

# 引入生物力学引擎与自动主控导演引擎
from biomechanics_engine import BiomechanicsEngine
from autonomous_director import AutonomousRehabDirector


class ChineseHudRenderer:
    """中文高清 HUD 渲染器，支持透明遮罩、圆环仪表盘与变色骨骼"""

    def __init__(self, width: int = 848, height: int = 480):
        self.width = width
        self.height = height

        # 加载 Windows 系统微软雅黑字体
        font_candidates = [
            "C:/Windows/Fonts/msyhbd.ttc",   # 微软雅黑粗体
            "C:/Windows/Fonts/msyh.ttc",     # 微软雅黑常规
            "C:/Windows/Fonts/simhei.ttf"    # 黑体
        ]
        self.font_path = None
        for p in font_candidates:
            if os.path.exists(p):
                self.font_path = p
                break

        if self.font_path:
            self.font_sm = ImageFont.truetype(self.font_path, 15)
            self.font_md = ImageFont.truetype(self.font_path, 19)
            self.font_lg = ImageFont.truetype(self.font_path, 25)
            self.font_xl = ImageFont.truetype(self.font_path, 34)
        else:
            self.font_sm = self.font_md = self.font_lg = self.font_xl = None

    def draw_alpha_panel(
        self,
        img: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color_bgr: Tuple[int, int, int] = (25, 25, 30),
        alpha: float = 0.72
    ):
        """绘制带圆角/平滑半透明面板"""
        sub = img[y1:y2, x1:x2]
        rect = np.full_like(sub, color_bgr, dtype=np.uint8)
        cv2.addWeighted(rect, alpha, sub, 1.0 - alpha, 0, sub)
        img[y1:y2, x1:x2] = sub

    def draw_circular_gauge(
        self,
        img: np.ndarray,
        center: Tuple[int, int],
        radius: int,
        current_deg: float,
        target_deg: float,
        has_comp: bool
    ):
        """绘制右侧实时角度表盘"""
        cx, cy = center

        # 背景圆环
        cv2.circle(img, (cx, cy), radius, (45, 45, 50), 10, cv2.LINE_AA)

        # 进度圆弧
        ratio = max(0.0, min(1.2, current_deg / max(1.0, target_deg)))
        start_angle = -90
        span_angle = int(ratio * 360)
        end_angle = start_angle + span_angle

        if has_comp:
            arc_color = (40, 40, 255)      # 代偿红色
        elif ratio >= 1.0:
            arc_color = (50, 240, 80)      # 达标绿色
        else:
            arc_color = (255, 200, 30)     # 运动中天蓝

        if span_angle > 0:
            cv2.ellipse(
                img,
                (cx, cy),
                (radius, radius),
                0,
                start_angle,
                end_angle,
                arc_color,
                10,
                cv2.LINE_AA
            )

        # 目标刻度线
        rad_target = math.radians(start_angle + 360)
        tx = int(cx + (radius + 8) * math.cos(rad_target))
        ty = int(cy + (radius + 8) * math.sin(rad_target))
        cv2.circle(img, (tx, ty), 4, (0, 255, 255), -1, cv2.LINE_AA)

    def draw_hud(
        self,
        disp_bgr: np.ndarray,
        eval_result: Dict[str, Any],
        joints_disp: Dict[str, Tuple[int, int]],
        fps: float,
        director_cmd: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """主 HUD 叠加渲染 (支持全自动主控导演指令)"""
        h, w = disp_bgr.shape[:2]

        # 1. 骨骼可视化 (带代偿变色预警)
        self.render_skeleton(disp_bgr, eval_result, joints_disp)

        # 2. 顶部综合状态条 (半透明黑底)
        self.draw_alpha_panel(disp_bgr, 0, 0, w, 52, color_bgr=(18, 20, 24), alpha=0.88)

        # 3. 右侧仪表盘区域 (半透明底板)
        panel_w = 210
        panel_x = w - panel_w - 15
        panel_y = 65
        panel_h = 270
        self.draw_alpha_panel(
            disp_bgr,
            panel_x,
            panel_y,
            panel_x + panel_w,
            panel_y + panel_h,
            color_bgr=(20, 24, 30),
            alpha=0.78
        )

        # 绘制仪表盘表盘
        gauge_cx = panel_x + panel_w // 2
        gauge_cy = panel_y + 80
        gauge_radius = 52
        self.draw_circular_gauge(
            disp_bgr,
            (gauge_cx, gauge_cy),
            gauge_radius,
            eval_result["fe_angle"],
            eval_result["target_angle"],
            eval_result["has_compensation"]
        )

        # 4. 底部导演大字命令横幅 (加高至 82px)
        tip_h = 84
        tip_y = h - tip_h - 10
        banner_color = (18, 22, 28)
        if director_cmd and director_cmd.get("is_task_complete", False):
            banner_color = (15, 65, 25)   # 全通关翠绿
        elif eval_result["has_compensation"]:
            banner_color = (18, 18, 90)   # 代偿深红底
        elif eval_result["status_level"] == "SUCCESS":
            banner_color = (15, 60, 22)   # 达标深绿底

        self.draw_alpha_panel(disp_bgr, 15, tip_y, w - 15, tip_y + tip_h, color_bgr=banner_color, alpha=0.92)

        # 5. 高清中文字体渲染 (使用 Pillow 绘制并写回)
        pil_img = Image.fromarray(cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # 5.1 顶部文字渲染
        stage_title = director_cmd.get("stage_name", "腕关节背伸训练 (右手 FE)") if director_cmd else "腕关节背伸训练 (右手 FE)"
        draw.text((20, 14), stage_title, font=self.font_md, fill=(255, 235, 100))

        # 顶部计数与性能
        std_cnt = eval_result["standard_reps"]
        comp_cnt = eval_result["compensated_reps"]
        counts_str = f"考核进度: {std_cnt}/5 次  |  代偿: {comp_cnt} 次  |  {fps:.0f} FPS"
        draw.text((w - 360, 16), counts_str, font=self.font_sm, fill=(210, 225, 240))

        # 5.2 仪表盘中心数字与信息
        deg_str = f"{eval_result['fe_angle']:.0f}°"
        draw.text((gauge_cx - 24, gauge_cy - 22), deg_str, font=self.font_xl, fill=(255, 255, 255))
        draw.text((gauge_cx - 28, gauge_cy + 16), "当前背伸角", font=self.font_sm, fill=(180, 195, 210))

        info_y = panel_y + 155
        target_str = f"目标角度: {eval_result['target_angle']:.0f}°"
        draw.text((panel_x + 20, info_y), target_str, font=self.font_sm, fill=(200, 240, 255))

        peak_str = f"本组峰值: {eval_result['history_peak_angle']:.0f}°"
        draw.text((panel_x + 20, info_y + 26), peak_str, font=self.font_sm, fill=(255, 220, 100))

        shrug_str = f"耸肩差分: {eval_result['shrug_displacement_mm']:+.0f} mm"
        shrug_fill = (255, 90, 90) if eval_result["is_shrugging"] else (160, 200, 160)
        draw.text((panel_x + 20, info_y + 52), shrug_str, font=self.font_sm, fill=shrug_fill)

        elbow_str = f"飞肘外展: {eval_result['abduction_displacement_mm']:+.0f} mm"
        elbow_fill = (255, 140, 40) if eval_result["is_flying_elbow"] else (160, 200, 160)
        draw.text((panel_x + 20, info_y + 78), elbow_str, font=self.font_sm, fill=elbow_fill)

        # 5.3 骨骼旁侧悬浮标签 (直接在人体关节旁提示)
        if eval_result["is_shrugging"] and "r_shoulder" in joints_disp:
            sx, sy = joints_disp["r_shoulder"]
            draw.text((sx + 15, sy - 12), f"⚠️ 耸肩 (+{eval_result['shrug_displacement_mm']:.0f}mm)", font=self.font_sm, fill=(255, 60, 60))

        if eval_result["is_flying_elbow"] and "r_elbow" in joints_disp:
            ex, ey = joints_disp["r_elbow"]
            draw.text((ex + 15, ey - 12), f"⚠️ 飞肘 (+{eval_result['abduction_displacement_mm']:.0f}mm)", font=self.font_sm, fill=(255, 150, 40))

        if "r_wrist" in joints_disp and eval_result["fe_angle"] > 8.0:
            wx, wy = joints_disp["r_wrist"]
            draw.text((wx + 15, wy - 10), f"{eval_result['fe_angle']:.0f}°", font=self.font_sm, fill=(120, 255, 120))

        # 5.4 底部主控导演命令提示 (第一行超大标题命令，第二行动作指导)
        if director_cmd:
            headline = director_cmd["director_headline"]
            subtext = director_cmd["director_subtext"]
            accent = director_cmd.get("accent_color", (50, 240, 80))
            accent_rgb = (accent[2], accent[1], accent[0]) # BGR -> RGB
        else:
            headline = eval_result["coach_tip"]
            subtext = "请平稳做动作，系统实时分析中。"
            accent_rgb = (255, 255, 255)

        # 第一行：超大主控命令 (24px 字体)
        draw.text((30, tip_y + 12), headline, font=self.font_lg, fill=accent_rgb)

        # 第二行：动作要领与详细指引 (17px 字体)
        draw.text((30, tip_y + 45), subtext, font=self.font_md, fill=(235, 240, 250))

        # 快捷键小贴士 (右上角/底部右侧)
        draw.text((w - 320, tip_y + 50), "[C]重置标定 | [Q/ESC]退出", font=self.font_sm, fill=(160, 170, 180))

        # 转回 OpenCV 格式
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def render_skeleton(
        self,
        img: np.ndarray,
        eval_result: Dict[str, Any],
        joints_disp: Dict[str, Tuple[int, int]]
    ):
        """在镜像图像上绘制智能变色骨骼与关键点"""
        # 默认高饱和绿色
        col_normal = (50, 235, 80)
        col_shrug = (30, 40, 255)       # 红色警报
        col_elbow = (30, 140, 255)      # 橙色警报

        is_shrug = eval_result["is_shrugging"]
        is_elbow = eval_result["is_flying_elbow"]

        # 骨骼连线配置
        bones = [
            ("l_shoulder", "r_shoulder", col_shrug if is_shrug else col_normal, 3),
            ("r_shoulder", "r_elbow", col_shrug if is_shrug else (col_elbow if is_elbow else col_normal), 4),
            ("r_elbow", "r_wrist", col_elbow if is_elbow else col_normal, 4),
            ("r_wrist", "r_index", col_normal, 3)
        ]

        # 绘制骨骼线
        for j1, j2, color, thick in bones:
            if j1 in joints_disp and j2 in joints_disp:
                pt1 = joints_disp[j1]
                pt2 = joints_disp[j2]
                cv2.line(img, pt1, pt2, color, thick, cv2.LINE_AA)

        # 绘制关节圆点
        for name, pt in joints_disp.items():
            if name == "r_shoulder" and is_shrug:
                # 耸肩光晕脉冲
                cv2.circle(img, pt, 14, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.circle(img, pt, 9, (0, 0, 255), -1, cv2.LINE_AA)
            elif name == "r_elbow" and is_elbow:
                # 飞肘光晕
                cv2.circle(img, pt, 14, (0, 140, 255), 2, cv2.LINE_AA)
                cv2.circle(img, pt, 9, (0, 140, 255), -1, cv2.LINE_AA)
            else:
                cv2.circle(img, pt, 6, col_normal, -1, cv2.LINE_AA)
                cv2.circle(img, pt, 8, (255, 255, 255), 1, cv2.LINE_AA)


class RealtimeBiofeedbackApp:
    def __init__(
        self,
        width: int = 848,
        height: int = 480,
        fps: int = 30,
        target_angle_deg: float = 40.0
    ):
        self.width = width
        self.height = height
        self.fps = fps

        # 核心引擎
        self.biomechanics = BiomechanicsEngine(
            target_action="wrist_FE",
            target_angle_deg=target_angle_deg,
            calib_frames_required=45
        )
        self.hud = ChineseHudRenderer(width=self.width, height=self.height)

        # 相机与内参
        self.pipeline = None
        self.align = None
        self.depth_scale = 0.001
        self.fx = 615.0
        self.fy = 615.0
        self.cx = 424.0
        self.cy = 240.0

        # MediaPipe Pose 检测器
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(curr_dir, "pose_landmarker_lite.task")
        with open(model_path, "rb") as f:
            model_bytes = f.read()

        base_options = python.BaseOptions(model_asset_buffer=model_bytes)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.35,
            min_tracking_confidence=0.35
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

        # 3D 关节平滑滤波器
        self.joints_3d_filtered: Dict[str, np.ndarray] = {}
        self.filter_alpha = 0.35

        # 实时遥测数据导出路径 (供 AI 与后台实时监测指挥)
        self.telemetry_path = os.path.join(curr_dir, "live_telemetry.json")

        # 全自动主控导演引擎 (5阶段全流程自动推进与考核)
        self.director = AutonomousRehabDirector(
            target_angle_deg=target_angle_deg,
            required_clean_reps=5,
            save_dir=curr_dir
        )

    def init_camera(self) -> bool:
        """初始化 D435i 硬件对齐流"""
        if not PYREALSENSE_AVAILABLE:
            print("[错误] 未检测到 pyrealsense2！")
            return False

        ctx = rs.context()
        devs = ctx.query_devices()
        if len(devs) == 0:
            print("[错误] 未发现已连接的 RealSense 相机！")
            return False

        dev = devs[0]
        print(f"[相机识别] {dev.get_info(rs.camera_info.name)} (SN: {dev.get_info(rs.camera_info.serial_number)})")

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)

        # 获取深度标尺与内参
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        self.fx = intr.fx
        self.fy = intr.fy
        self.cx = intr.ppx
        self.cy = intr.ppy
        print(f"[相机配置] 深度标尺: {self.depth_scale}, 内参: fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}")

        # 相机预热
        print("正在进行曝光与测距预热 (30帧)...")
        for _ in range(30):
            fs = self.pipeline.wait_for_frames()
            self.align.process(fs)
        print("[就绪] 硬件通信完毕，启动实时反馈系统！\n")
        return True

    def extract_patch_depth_m(self, depth_raw: np.ndarray, u_px: int, v_px: int) -> float:
        """7x7 邻域深度中值滤波提取"""
        h, w = depth_raw.shape
        u_px = max(0, min(w - 1, u_px))
        v_px = max(0, min(h - 1, v_px))

        r = 3
        u1, u2 = max(0, u_px - r), min(w, u_px + r + 1)
        v1, v2 = max(0, v_px - r), min(h, v_px + r + 1)
        patch = depth_raw[v1:v2, u1:u2]
        valid_mask = patch > 0
        if np.count_nonzero(valid_mask) > 0:
            median_val = np.median(patch[valid_mask])
            depth_m = float(median_val) * self.depth_scale
            if 0.4 <= depth_m <= 2.2:
                return depth_m
        return 0.0

    def deproject_to_3d(self, u_px: float, v_px: float, depth_m: float) -> np.ndarray:
        """针孔反投影"""
        x_m = (u_px - self.cx) * depth_m / self.fx
        y_m = (v_px - self.cy) * depth_m / self.fy
        return np.array([x_m, y_m, depth_m], dtype=np.float64)

    def run(self):
        """主实时闭环循环"""
        if not self.init_camera():
            return

        window_name = "D435i Real-Time Visual Biofeedback (Mirror Mode)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        # 目标关键点列表与 MediaPipe ID 映射
        # 右侧上肢运动链
        joint_id_map = {
            "l_shoulder": 11,
            "r_shoulder": 12,
            "r_elbow": 14,
            "r_wrist": 16,
            "r_pinky": 18,
            "r_index": 20
        }

        prev_time = time.time()
        fps = 30.0
        frame_idx = 0

        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)

                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                color_bgr = np.asanyarray(color_frame.get_data())
                depth_raw = np.asanyarray(depth_frame.get_data())

                curr_time = time.time()
                dt = curr_time - prev_time
                prev_time = curr_time
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)

                # 1. MediaPipe Pose 骨骼提取
                rgb_img = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
                detection_result = self.detector.detect(mp_img)

                joints_3d = {}
                joints_disp_mirrored = {}
                landmarks = None
                if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
                    landmarks = detection_result.pose_landmarks[0]

                # 2. 空间三维坐标解算与滤波
                if landmarks is not None:
                    for name, lm_id in joint_id_map.items():
                        if lm_id < len(landmarks):
                            lm = landmarks[lm_id]
                            # 仅保留可见度合格的点
                            vis = float(lm.visibility if lm.visibility is not None else 1.0)
                            if vis >= 0.25:
                                u_px = lm.x * self.width
                                v_px = lm.y * self.height

                                # 镜像屏幕像素坐标 (镜子效果)
                                u_disp = int(round(self.width - 1 - u_px))
                                v_disp = int(round(v_px))
                                joints_disp_mirrored[name] = (u_disp, v_disp)

                                # 提取真实深度并反投影为相机三维物理坐标 (米)
                                depth_m = self.extract_patch_depth_m(depth_raw, int(round(u_px)), int(round(v_px)))
                                if depth_m > 0:
                                    pt_3d = self.deproject_to_3d(u_px, v_px, depth_m)

                                    # 一阶 EMA 滤波
                                    if name not in self.joints_3d_filtered:
                                        self.joints_3d_filtered[name] = pt_3d
                                    else:
                                        self.joints_3d_filtered[name] = (
                                            self.filter_alpha * pt_3d +
                                            (1.0 - self.filter_alpha) * self.joints_3d_filtered[name]
                                        )
                                    joints_3d[name] = self.joints_3d_filtered[name].copy()

                # 3. 生物力学分析与代偿判定
                eval_result = self.biomechanics.evaluate_frame(joints_3d)

                # 全自动主控导演状态机单帧推进 (指挥测试者动作与阶段流转)
                director_cmd = self.director.update(eval_result, joints_3d)

                # 实时遥测数据导出 (供 AI 临床监督与参数优化)
                frame_idx += 1
                if frame_idx % 8 == 0:
                    try:
                        telemetry_data = {
                            "timestamp": time.time(),
                            "frame_idx": frame_idx,
                            "fps": round(fps, 1),
                            "director_stage": director_cmd["stage_id"],
                            "director_headline": director_cmd["director_headline"],
                            "is_calibrated": eval_result["is_calibrated"],
                            "fe_angle": eval_result["fe_angle"],
                            "target_angle": eval_result["target_angle"],
                            "progress_ratio": eval_result["progress_ratio"],
                            "shrug_displacement_mm": eval_result["shrug_displacement_mm"],
                            "is_shrugging": eval_result["is_shrugging"],
                            "abduction_displacement_mm": eval_result["abduction_displacement_mm"],
                            "is_flying_elbow": eval_result["is_flying_elbow"],
                            "trunk_tilt_diff_deg": eval_result["trunk_tilt_diff_deg"],
                            "has_compensation": eval_result["has_compensation"],
                            "comp_reasons": eval_result["comp_reasons"],
                            "status_text": eval_result["status_text"],
                            "coach_tip": eval_result["coach_tip"],
                            "standard_reps": eval_result["standard_reps"],
                            "compensated_reps": eval_result["compensated_reps"],
                            "total_cycles": eval_result["total_cycles"],
                            "history_peak_angle": eval_result["history_peak_angle"],
                            "motion_state": eval_result["motion_state"],
                            "joints_detected": list(joints_3d.keys())
                        }
                        tmp_f = self.telemetry_path + ".tmp"
                        with open(tmp_f, "w", encoding="utf-8") as f:
                            json.dump(telemetry_data, f, ensure_ascii=False, indent=2)
                        os.replace(tmp_f, self.telemetry_path)
                    except Exception:
                        pass

                # 4. 画面镜面翻转 (如同看镜子)
                mirror_bgr = cv2.flip(color_bgr, 1)

                # 5. HUD 高清视觉反馈叠加 (全自动导演指令投射)
                output_bgr = self.hud.draw_hud(mirror_bgr, eval_result, joints_disp_mirrored, fps, director_cmd=director_cmd)

                # 显示窗口
                cv2.imshow(window_name, output_bgr)

                # 键盘事件交互
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q'), 27]:  # Q 或 ESC 退出
                    break
                elif key in [ord('c'), ord('C')]:    # C 键：重新标定坐姿基准
                    self.biomechanics.reset_calibration()
                    print("\n>>> 已重置标定基准，请端正坐姿，正在重新采集...")
                elif key in [ord('r'), ord('R')]:    # R 键：重置计数器
                    self.biomechanics.reset_counters()
                    print("\n>>> 已重置训练计数器！")

        finally:
            self.pipeline.stop()
            self.detector.close()
            cv2.destroyAllWindows()
            print("[退出] 实时反馈系统安全关闭。")


def main():
    app = RealtimeBiofeedbackApp(width=848, height=480, fps=30, target_angle_deg=40.0)
    app.run()


if __name__ == "__main__":
    main()
