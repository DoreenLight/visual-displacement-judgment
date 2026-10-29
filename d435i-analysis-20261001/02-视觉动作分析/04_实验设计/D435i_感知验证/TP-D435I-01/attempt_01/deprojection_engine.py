#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deprojection_engine.py - 上肢关键点检测与深度点云反投影核心算法引擎
遵循 TP-D435I-01 与 wrist-rehab/v1 契约规范：
1. MediaPipe Tasks PoseLandmarker 定位训练侧 shoulder/elbow/wrist 二维像素坐标；
2. 局部自适应滤波提取稳健有效深度 Z (剔除空洞与边缘飞点)；
3. 针孔相机几何反投影到 camera_color_optical 光学坐标系 (米制)；
4. 异常检测：置信度不足、深度缺失、几何肢长突变等标记为 invalid，坐标置 NaN；
5. 在线因果平滑滤波 (EMA/移动窗口)，不借用未来帧，缺失时不伪造插值。
"""

import os
import math
import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class JointFilterState:
    """单个关节的在线因果滤波状态 (Causal Online EMA Filter)"""
    def __init__(self, alpha: float = 0.35, max_gap_reset: int = 6):
        self.alpha = alpha
        self.max_gap_reset = max_gap_reset
        self.filtered_pos: Optional[np.ndarray] = None
        self.consecutive_invalid: int = 0

    def update(self, raw_pos: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if raw_pos is None or np.isnan(raw_pos).any():
            self.consecutive_invalid += 1
            if self.consecutive_invalid >= self.max_gap_reset:
                self.filtered_pos = None  # 连续丢点超限时重置状态，防止后续出现拖拽迟滞
            return None

        self.consecutive_invalid = 0
        if self.filtered_pos is None:
            self.filtered_pos = raw_pos.copy()
        else:
            self.filtered_pos = self.alpha * raw_pos + (1.0 - self.alpha) * self.filtered_pos

        return self.filtered_pos.copy()


class UpperLimbDeprojectionEngine:
    # 关节点索引对照表
    JOINTS_CONFIG = {
        "right": {
            "shoulder": 12,
            "elbow": 14,
            "wrist": 16
        },
        "left": {
            "shoulder": 11,
            "elbow": 13,
            "wrist": 15
        }
    }

    def __init__(
        self,
        training_side: str = "right",
        model_path: Optional[str] = None,
        min_detection_confidence: float = 0.4,
        min_tracking_confidence: float = 0.4,
        depth_patch_radius: int = 3,       # 7x7 patch
        min_depth_m: float = 0.4,          # 最小工作距离 (m)
        max_depth_m: float = 2.5,          # 最大工作距离 (m)
        filter_alpha: float = 0.35
    ):
        self.training_side = training_side.lower()
        if self.training_side not in self.JOINTS_CONFIG:
            raise ValueError(f"training_side 必须为 'right' 或 'left', 当前为: {training_side}")

        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.depth_patch_radius = depth_patch_radius
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m

        self.filters = {
            "shoulder": JointFilterState(alpha=filter_alpha),
            "elbow": JointFilterState(alpha=filter_alpha),
            "wrist": JointFilterState(alpha=filter_alpha)
        }

        # 确定模型文件路径
        if model_path is None:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(curr_dir, "pose_landmarker_lite.task")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到 MediaPipe 模型文件: {model_path}")

        # 使用二进制缓冲方式加载模型，彻底规避 Windows 非 ASCII 路径编码问题
        with open(model_path, "rb") as f:
            model_bytes = f.read()

        base_options = python.BaseOptions(model_asset_buffer=model_bytes)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def close(self):
        if self.detector:
            self.detector.close()
            self.detector = None

    def extract_patch_depth(
        self,
        depth_map: np.ndarray,
        depth_scale: float,
        u_px: int,
        v_px: int
    ) -> Tuple[Optional[float], bool, str]:
        """
        在关键点周围以自适应 patch 提取深度中位数
        返回: (depth_m, is_valid, reason)
        """
        h, w = depth_map.shape[:2]
        if u_px < 0 or u_px >= w or v_px < 0 or v_px >= h:
            return None, False, "pixel_coordinates_out_of_bounds"

        r = self.depth_patch_radius
        u_min = max(0, u_px - r)
        u_max = min(w, u_px + r + 1)
        v_min = max(0, v_px - r)
        v_max = min(h, v_px + r + 1)

        patch = depth_map[v_min:v_max, u_min:u_max]
        valid_mask = patch > 0
        valid_vals = patch[valid_mask]

        if len(valid_vals) == 0:
            return None, False, "depth_missing_zero_hole"

        # 中位数深度剔除飞点
        depth_m = float(np.median(valid_vals)) * depth_scale

        if depth_m < self.min_depth_m or depth_m > self.max_depth_m:
            return None, False, f"depth_out_of_range_{depth_m:.3f}m"

        return depth_m, True, "valid"

    def deproject_pixel_to_camera_optical(
        self,
        u_px: float,
        v_px: float,
        depth_m: float,
        fx: float,
        fy: float,
        cx: float,
        cy: float
    ) -> np.ndarray:
        """
        相机光学坐标系针孔反投影:
        X向右, Y向下, Z向前 (单位: 米)
        """
        X = (u_px - cx) * depth_m / fx
        Y = (v_px - cy) * depth_m / fy
        Z = depth_m
        return np.array([X, Y, Z], dtype=np.float64)

    def process_frame(
        self,
        color_bgr: np.ndarray,
        depth_raw: np.ndarray,
        depth_scale: float,
        intrinsics: Any,
        run_id: str,
        frame_id: int,
        t_rgb_device_s: float,
        t_depth_device_s: float,
        t_host_mono_s: float,
        clock_domain: str = "hardware_clock_ms"
    ) -> List[Dict[str, Any]]:
        """
        处理单帧并返回符合 wrist-rehab/v1 标准长表结构的 3 个关节记录 (shoulder, elbow, wrist)
        """
        records: List[Dict[str, Any]] = []
        h, w = color_bgr.shape[:2]

        # 1. 2D 姿态检测
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_rgb)
        pose_results = self.detector.detect(mp_image) if self.detector else None

        landmarks = None
        if pose_results and pose_results.pose_landmarks and len(pose_results.pose_landmarks) > 0:
            landmarks = pose_results.pose_landmarks[0]

        joint_ids_map = self.JOINTS_CONFIG[self.training_side]
        extracted_joints = {}

        for joint_name, lm_id in joint_ids_map.items():
            record = {
                "run_id": run_id,
                "frame_id": frame_id,
                "joint_id": joint_name,
                "t_rgb_device_s": t_rgb_device_s,
                "t_depth_device_s": t_depth_device_s,
                "t_host_mono_s": t_host_mono_s,
                "clock_domain": clock_domain,
                "u_px": math.nan,
                "v_px": math.nan,
                "x_raw_m": math.nan,
                "y_raw_m": math.nan,
                "z_raw_m": math.nan,
                "x_filt_m": math.nan,
                "y_filt_m": math.nan,
                "z_filt_m": math.nan,
                "keypoint_score": 0.0,
                "visibility": 0.0,
                "depth_valid": False,
                "measurement_valid": False,
                "imputed": False,              # 实测数据不作伪造插值
                "valid_for_analysis": False,
                "invalid_reason": "none",
                "coord_frame": "camera_color_optical"
            }

            if landmarks is None or lm_id >= len(landmarks):
                record["invalid_reason"] = "mediapipe_detection_failed"
                self.filters[joint_name].update(None)
                records.append(record)
                continue

            lm = landmarks[lm_id]
            u_px = lm.x * w
            v_px = lm.y * h
            record["u_px"] = round(u_px, 2)
            record["v_px"] = round(v_px, 2)
            record["keypoint_score"] = round(float(lm.presence if hasattr(lm, "presence") and lm.presence is not None else 1.0), 3)
            record["visibility"] = round(float(lm.visibility if lm.visibility is not None else 1.0), 3)

            # 检查 2D 识别置信度
            if record["visibility"] < 0.35:
                record["invalid_reason"] = f"low_visibility_{record['visibility']:.2f}"
                self.filters[joint_name].update(None)
                records.append(record)
                continue

            # 提取局部深度
            depth_m, depth_valid, depth_reason = self.extract_patch_depth(
                depth_raw, depth_scale, int(round(u_px)), int(round(v_px))
            )
            record["depth_valid"] = depth_valid

            if not depth_valid:
                record["invalid_reason"] = depth_reason
                self.filters[joint_name].update(None)
                records.append(record)
                continue

            # 针孔几何反投影
            raw_pt = self.deproject_pixel_to_camera_optical(
                u_px=u_px,
                v_px=v_px,
                depth_m=depth_m,
                fx=intrinsics.fx,
                fy=intrinsics.fy,
                cx=intrinsics.ppx,
                cy=intrinsics.ppy
            )

            record["x_raw_m"] = round(float(raw_pt[0]), 5)
            record["y_raw_m"] = round(float(raw_pt[1]), 5)
            record["z_raw_m"] = round(float(raw_pt[2]), 5)
            record["measurement_valid"] = True

            extracted_joints[joint_name] = (raw_pt, record)

        # 2. 上肢解剖几何一致性检验 (Upper Limb Geometric Consistency)
        if "shoulder" in extracted_joints and "elbow" in extracted_joints:
            p_s = extracted_joints["shoulder"][0]
            p_e = extracted_joints["elbow"][0]
            len_upper_arm = float(np.linalg.norm(p_e - p_s))
            if len_upper_arm < 0.10 or len_upper_arm > 0.65:
                extracted_joints["elbow"][1]["measurement_valid"] = False
                extracted_joints["elbow"][1]["invalid_reason"] = f"upper_arm_length_anomaly_{len_upper_arm:.3f}m"

        if "elbow" in extracted_joints and "wrist" in extracted_joints:
            p_e = extracted_joints["elbow"][0]
            p_w = extracted_joints["wrist"][0]
            len_forearm = float(np.linalg.norm(p_w - p_e))
            if len_forearm < 0.06 or len_forearm > 0.55:
                extracted_joints["wrist"][1]["measurement_valid"] = False
                extracted_joints["wrist"][1]["invalid_reason"] = f"forearm_length_anomaly_{len_forearm:.3f}m"

        # 3. 滤波更新与最终写入
        for joint_name, lm_id in joint_ids_map.items():
            if joint_name in extracted_joints:
                raw_pt, record = extracted_joints[joint_name]
                if record["measurement_valid"]:
                    filt_pt = self.filters[joint_name].update(raw_pt)
                    if filt_pt is not None:
                        record["x_filt_m"] = round(float(filt_pt[0]), 5)
                        record["y_filt_m"] = round(float(filt_pt[1]), 5)
                        record["z_filt_m"] = round(float(filt_pt[2]), 5)
                    record["valid_for_analysis"] = True
                    record["invalid_reason"] = "valid"
                else:
                    self.filters[joint_name].update(None)
                    record["valid_for_analysis"] = False
                    record["x_raw_m"] = math.nan
                    record["y_raw_m"] = math.nan
                    record["z_raw_m"] = math.nan
                records.append(record)

        return records
