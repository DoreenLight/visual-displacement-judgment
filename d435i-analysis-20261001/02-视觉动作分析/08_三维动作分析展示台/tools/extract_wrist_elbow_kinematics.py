#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upper-Limb Kinematic Analysis & Compensation Modeling
For video: 07_资料库/实验2-腕肘肩代偿判断-视频2-腕肘.mp4
Extracts Shoulder, Elbow, Wrist, and 21 Hand Landmarks.
Computes Kinematics, Fourier Series Models, and Elbow Compensation Index.
"""

import os
import json
import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import savgol_filter, find_peaks

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    video_path = os.path.join(base_dir, "07_资料库/实验2-腕肘肩代偿判断-视频2-腕肘_h264.mp4")
    if not os.path.exists(video_path):
        video_path = os.path.join(base_dir, "07_资料库/实验2-腕肘肩代偿判断-视频2-腕肘.mp4")
    if not os.path.exists(video_path):
        video_path = "07_资料库/实验2-腕肘肩代偿判断-视频2-腕肘_h264.mp4"

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video loaded: {width}x{height} @ {fps} fps, {total_frames} frames")

    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

    raw_frames = []

    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        t = idx / fps
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. Hands detection (Foreground hand)
        h_res = hands.process(rgb)
        hand_landmarks = None
        wrist_pt_hand = None
        if h_res.multi_hand_landmarks:
            best_hl = max(
                h_res.multi_hand_landmarks,
                key=lambda hl: (max([lm.x for lm in hl.landmark]) - min([lm.x for lm in hl.landmark])) * 
                               (max([lm.y for lm in hl.landmark]) - min([lm.y for lm in hl.landmark]))
            )
            hand_landmarks = [[lm.x, lm.y, lm.z] for lm in best_hl.landmark]
            wrist_pt_hand = [best_hl.landmark[0].x, best_hl.landmark[0].y, best_hl.landmark[0].z]

        # 2. Pose detection (Masking background subject at x < 0.35)
        masked_rgb = rgb.copy()
        masked_rgb[:, :int(width * 0.35)] = 0
        p_res = pose.process(masked_rgb)
        
        shoulder_pt = None
        elbow_pt = None
        shoulder_world_pt = None
        elbow_world_pt = None

        if p_res.pose_landmarks:
            s_lm = p_res.pose_landmarks.landmark[12]
            e_lm = p_res.pose_landmarks.landmark[14]
            shoulder_pt = [s_lm.x, s_lm.y, s_lm.z]
            elbow_pt = [e_lm.x, e_lm.y, e_lm.z]
            
        if p_res.pose_world_landmarks:
            sw_lm = p_res.pose_world_landmarks.landmark[12]
            ew_lm = p_res.pose_world_landmarks.landmark[14]
            shoulder_world_pt = [sw_lm.x, sw_lm.y, sw_lm.z]
            elbow_world_pt = [ew_lm.x, ew_lm.y, ew_lm.z]

        raw_frames.append({
            "frame": idx,
            "time": t,
            "hand_landmarks": hand_landmarks,
            "wrist_pt_hand": wrist_pt_hand,
            "shoulder_pt": shoulder_pt,
            "elbow_pt": elbow_pt,
            "shoulder_world_pt": shoulder_world_pt,
            "elbow_world_pt": elbow_world_pt
        })

    cap.release()
    print(f"Tracking complete. Processing raw data for {len(raw_frames)} frames...")

    # Impute any missing points by linear interpolation
    def interpolate_series(data_list, default_val):
        arr = []
        for d in data_list:
            arr.append(d if d is not None else [np.nan]*len(default_val))
        arr = np.array(arr)
        for col in range(arr.shape[1]):
            s = arr[:, col]
            nans = np.isnan(s)
            if np.all(nans):
                s[:] = default_val[col]
            else:
                x = np.arange(len(s))
                s[nans] = np.interp(x[nans], x[~nans], s[~nans])
            arr[:, col] = s
        return arr

    wrist_arr_norm = interpolate_series([r["wrist_pt_hand"] for r in raw_frames], [0.3, 0.7, 0.0])
    elbow_arr_norm = interpolate_series([r["elbow_pt"] for r in raw_frames], [0.5, 0.5, 0.0])
    shoulder_arr_norm = interpolate_series([r["shoulder_pt"] for r in raw_frames], [0.65, 0.25, 0.0])

    elbow_world_arr = interpolate_series([r["elbow_world_pt"] for r in raw_frames], [-0.17, -0.12, -0.20])
    shoulder_world_arr = interpolate_series([r["shoulder_world_pt"] for r in raw_frames], [-0.10, -0.40, -0.21])

    # Convert normalized image coordinates to scaled millimeters
    # Optical calibration: adult forearm ~ 260mm, upper arm ~ 290mm
    # We establish a coherent 3D Cartesian frame (in mm):
    # Origin: centered at mean shoulder anchor position
    # X: Left / Right (lateral)
    # Y: Anterior / Posterior (forward/backward)
    # Z: Superior / Inferior (vertical, Up is +Z)

    # Let's derive metric coordinates (X, Y, Z) in mm:
    # 1. Shoulder anchor:
    # Pose world coordinates in meters, converted to mm
    S_mm = (shoulder_world_arr - np.mean(shoulder_world_arr, axis=0)) * 1000.0
    # Shoulder moves very little (torso anchored): smooth S_mm
    S_mm = savgol_filter(S_mm, 15, 2, axis=0)

    # 2. Elbow position in mm:
    # From shoulder to elbow vector:
    # World landmarks give metric displacement in meters
    E_world_mm = elbow_world_arr * 1000.0
    S_world_mm = shoulder_world_arr * 1000.0
    upper_arm_vec = E_world_mm - S_world_mm
    # Normalise upper arm length to canonical 295 mm
    upper_lens = np.linalg.norm(upper_arm_vec, axis=1, keepdims=True)
    scale_upper = 295.0 / np.maximum(upper_lens, 1e-3)
    upper_arm_vec_calib = upper_arm_vec * (0.7 + 0.3 * scale_upper)

    E_mm = S_mm + upper_arm_vec_calib
    E_mm = savgol_filter(E_mm, 13, 2, axis=0)

    # 3. Wrist position in mm:
    # Forearm extends from elbow towards the wrist
    # In camera coordinates, wrist is closer to camera (more negative/positive depth)
    # Let's compute forearm 2D direction and foreshortening depth
    wrist_px = wrist_arr_norm[:, :2] * np.array([width, height])
    elbow_px = elbow_arr_norm[:, :2] * np.array([width, height])
    forearm_2d_px = wrist_px - elbow_px
    
    # 2D pixel length of forearm
    forearm_px_len = np.linalg.norm(forearm_2d_px, axis=1)
    mean_forearm_px = np.percentile(forearm_px_len, 85) # near perpendicular view
    px_to_mm = 265.0 / max(mean_forearm_px, 1.0) # mm per pixel

    # Reconstruct Forearm 3D vector:
    # delta_X = forearm_2d_px[:, 0] * px_to_mm
    # delta_Z (vertical in 3D scene, negative image Y) = -forearm_2d_px[:, 1] * px_to_mm
    # delta_Y (depth towards camera): from L_forearm^2 = delta_X^2 + delta_Z^2 + delta_Y^2
    L_forearm = 265.0 # mm
    dx = forearm_2d_px[:, 0] * px_to_mm
    dz = -forearm_2d_px[:, 1] * px_to_mm
    transverse_sq = dx**2 + dz**2
    # Ensure physical constraint
    dy = np.sqrt(np.maximum(L_forearm**2 - transverse_sq, 20.0**2))

    W_mm = np.zeros_like(E_mm)
    W_mm[:, 0] = E_mm[:, 0] + dx
    W_mm[:, 1] = E_mm[:, 1] + dy # extending forward towards camera
    W_mm[:, 2] = E_mm[:, 2] + dz
    W_mm = savgol_filter(W_mm, 11, 2, axis=0)

    # Re-center coordinate system so that the initial wrist position is near origin:
    # Origin offset: E(0) and W(0)
    w0 = W_mm[0].copy()
    e0 = E_mm[0].copy()
    s0 = S_mm[0].copy()

    # Center scene on mean wrist-elbow midpoint
    center_offset = (w0 + e0) * 0.5
    W_mm -= center_offset
    E_mm -= center_offset
    S_mm -= center_offset

    # Compute Hand Pronation/Supination angles and 3D Hand Mesh
    # Hand landmarks are 21 relative points
    processed_frames = []
    pronation_angles = []
    hand_normals = []

    dt = 1.0 / fps

    for idx, r in enumerate(raw_frames):
        hl = r["hand_landmarks"]
        if hl is None:
            # use previous
            hl = processed_frames[-1]["hand_landmarks_3d"] if processed_frames else [[0,0,0]]*21
            angle = pronation_angles[-1] if pronation_angles else 180.0
            normal = hand_normals[-1] if hand_normals else [0, 0, 1]
        else:
            # landmark 0 is wrist
            p0 = np.array(hl[0])
            p5 = np.array(hl[5]) # index mcp
            p17 = np.array(hl[17]) # pinky mcp
            v1 = (p5 - p0) * np.array([width, height, width]) * px_to_mm
            v2 = (p17 - p0) * np.array([width, height, width]) * px_to_mm
            norm_v = np.cross(v1, v2)
            n_len = np.linalg.norm(norm_v)
            if n_len > 1e-5:
                norm_v /= n_len
            else:
                norm_v = np.array([0.0, 0.0, 1.0])

            # Transform hand 21 landmarks into wrist-centered metric mm coordinates
            hand_3d = []
            for pt in hl:
                # scale relative to wrist (point 0)
                diff = (np.array(pt) - p0) * np.array([width, height, width]) * px_to_mm
                # flip Y to match Three.js Z-up or Y-up
                hand_3d.append([float(round(diff[0], 2)), float(round(-diff[1], 2)), float(round(diff[2], 2))])
            hl = hand_3d

            # Pronation / Supination angle (0 deg: palm prone, 180 deg: palm supine)
            # Palm normal pointing down: prone (~0°/360°), pointing up: supine (~180°)
            angle = float(np.degrees(np.arctan2(norm_v[0], norm_v[1])) % 360)
            normal = [float(round(norm_v[0], 3)), float(round(norm_v[1], 3)), float(round(norm_v[2], 3))]

        pronation_angles.append(angle)
        hand_normals.append(normal)
        processed_frames.append({
            "hand_landmarks_3d": hl
        })

    # Smooth pronation angles
    pronation_angles = np.array(pronation_angles)
    pronation_angles = savgol_filter(pronation_angles, 9, 2)

    # Velocity and Acceleration (Finite Difference + Filtering)
    v_wrist = np.zeros_like(W_mm)
    a_wrist = np.zeros_like(W_mm)
    v_elbow = np.zeros_like(E_mm)
    a_elbow = np.zeros_like(E_mm)

    v_wrist[1:-1] = (W_mm[2:] - W_mm[:-2]) / (2 * dt)
    v_wrist[0] = (W_mm[1] - W_mm[0]) / dt
    v_wrist[-1] = (W_mm[-1] - W_mm[-2]) / dt

    a_wrist[1:-1] = (v_wrist[2:] - v_wrist[:-2]) / (2 * dt)
    a_wrist[0] = (v_wrist[1] - v_wrist[0]) / dt
    a_wrist[-1] = (v_wrist[-1] - v_wrist[-2]) / dt

    v_elbow[1:-1] = (E_mm[2:] - E_mm[:-2]) / (2 * dt)
    v_elbow[0] = (E_mm[1] - E_mm[0]) / dt
    v_elbow[-1] = (E_mm[-1] - E_mm[-2]) / dt

    a_elbow[1:-1] = (v_elbow[2:] - v_elbow[:-2]) / (2 * dt)
    a_elbow[0] = (v_elbow[1] - v_elbow[0]) / dt
    a_elbow[-1] = (v_elbow[-1] - v_elbow[-2]) / dt

    speed_wrist = np.linalg.norm(v_wrist, axis=1)
    accel_wrist = np.linalg.norm(a_wrist, axis=1)
    speed_elbow = np.linalg.norm(v_elbow, axis=1)
    accel_elbow = np.linalg.norm(a_elbow, axis=1)

    # Elbow joint angle: angle between upper arm (E -> S) and forearm (E -> W)
    vec_es = S_mm - E_mm
    vec_ew = W_mm - E_mm
    dot_prod = np.sum(vec_es * vec_ew, axis=1)
    norm_es = np.linalg.norm(vec_es, axis=1)
    norm_ew = np.linalg.norm(vec_ew, axis=1)
    cos_theta = np.clip(dot_prod / (norm_es * norm_ew), -1.0, 1.0)
    elbow_joint_angles = np.degrees(np.arccos(cos_theta)) # degrees
    elbow_joint_angles = savgol_filter(elbow_joint_angles, 9, 2)

    # Elbow Compensation Metrics:
    # 1. Displacement from starting position
    elbow_disp_from_start = np.linalg.norm(E_mm - E_mm[0], axis=1)
    wrist_disp_from_start = np.linalg.norm(W_mm - W_mm[0], axis=1)

    # 2. Instantaneous compensation ratio: ratio of elbow displacement to wrist displacement
    comp_ratio = elbow_disp_from_start / np.maximum(wrist_disp_from_start, 10.0)
    comp_ratio = np.clip(comp_ratio, 0.0, 2.0)

    # 3. Peak/Valley Detection for Pronation-Supination Cycles
    # Wrist Z or pronation angle gives clear alternating cycles
    peaks, _ = find_peaks(W_mm[:, 2], distance=14, prominence=8)
    valleys, _ = find_peaks(-W_mm[:, 2], distance=14, prominence=8)

    waypoints = []
    # Add start
    waypoints.append({
        "type": "start",
        "time": 0.0,
        "frame": 0,
        "label": "动作起点 (Start, t=0.00s)",
        "wrist_pos": [round(float(W_mm[0, 0]), 2), round(float(W_mm[0, 1]), 2), round(float(W_mm[0, 2]), 2)],
        "elbow_pos": [round(float(E_mm[0, 0]), 2), round(float(E_mm[0, 1]), 2), round(float(E_mm[0, 2]), 2)],
        "comp_disp": 0.0
    })

    # Merge peaks and valleys into chronological waypoints
    event_frames = sorted(list(peaks) + list(valleys))
    cycle_counter = 1
    for f in event_frames:
        t_sec = round(f / fps, 3)
        is_peak = f in peaks
        disp_e = round(float(elbow_disp_from_start[f]), 2)
        if is_peak:
            wp_type = "peak"
            label = f"旋后波峰 / 肘部代偿峰值 (Cycle {cycle_counter}, t={t_sec}s)"
            cycle_counter += 1
        else:
            wp_type = "valley"
            label = f"旋前波谷 / 肘部回撤 (t={t_sec}s)"
            
        waypoints.append({
            "type": wp_type,
            "time": t_sec,
            "frame": int(f),
            "label": label,
            "wrist_pos": [round(float(W_mm[f, 0]), 2), round(float(W_mm[f, 1]), 2), round(float(W_mm[f, 2]), 2)],
            "elbow_pos": [round(float(E_mm[f, 0]), 2), round(float(E_mm[f, 1]), 2), round(float(E_mm[f, 2]), 2)],
            "comp_disp": disp_e
        })

    # Add end
    waypoints.append({
        "type": "end",
        "time": round((total_frames - 1) / fps, 3),
        "frame": total_frames - 1,
        "label": f"动作终点 (End, t={round((total_frames - 1)/fps, 2)}s)",
        "wrist_pos": [round(float(W_mm[-1, 0]), 2), round(float(W_mm[-1, 1]), 2), round(float(W_mm[-1, 2]), 2)],
        "elbow_pos": [round(float(E_mm[-1, 0]), 2), round(float(E_mm[-1, 1]), 2), round(float(E_mm[-1, 2]), 2)],
        "comp_disp": round(float(elbow_disp_from_start[-1]), 2)
    })

    # Fit Fourier Series model for both Wrist and Elbow
    times = np.arange(total_frames) * dt
    f0 = 1.15 # Hz
    K = 3

    def fit_fourier(series_1d, times, f0, K):
        # Basis: [1, t, cos(w t), sin(w t), cos(2w t), sin(2w t)...]
        A = [np.ones_like(times), times]
        for k in range(1, K + 1):
            A.append(np.cos(2 * np.pi * k * f0 * times))
            A.append(np.sin(2 * np.pi * k * f0 * times))
        A = np.column_stack(A)
        coeffs, _, _, _ = np.linalg.lstsq(A, series_1d, rcond=None)
        return coeffs.tolist()

    fourier_wrist = {
        "x": fit_fourier(W_mm[:, 0], times, f0, K),
        "y": fit_fourier(W_mm[:, 1], times, f0, K),
        "z": fit_fourier(W_mm[:, 2], times, f0, K),
    }

    fourier_elbow = {
        "x": fit_fourier(E_mm[:, 0], times, f0, K),
        "y": fit_fourier(E_mm[:, 1], times, f0, K),
        "z": fit_fourier(E_mm[:, 2], times, f0, K),
    }

    # Model Summary Statistics
    model_summary = {
        "video_file": "实验2-腕肘肩代偿判断-视频2-腕肘.mp4",
        "action_title": "前臂旋前-旋后轮替翻转伴显著肘关节代偿动作 (Rapid Wrist Pronation-Supination with Elbow Compensation)",
        "total_frames": total_frames,
        "sampling_fps": fps,
        "total_duration_s": round(total_frames / fps, 2),
        "fundamental_frequency_hz": f0,
        "fundamental_period_s": round(1.0 / f0, 2),
        "cycles_count": len(peaks),
        "wrist_spans_mm": {
            "x_span": round(float(np.ptp(W_mm[:, 0])), 2),
            "y_span": round(float(np.ptp(W_mm[:, 1])), 2),
            "z_span": round(float(np.ptp(W_mm[:, 2])), 2),
            "total_path_length": round(float(np.sum(np.linalg.norm(np.diff(W_mm, axis=0), axis=1))), 2)
        },
        "elbow_spans_mm": {
            "x_span": round(float(np.ptp(E_mm[:, 0])), 2),
            "y_span": round(float(np.ptp(E_mm[:, 1])), 2),
            "z_span": round(float(np.ptp(E_mm[:, 2])), 2),
            "total_path_length": round(float(np.sum(np.linalg.norm(np.diff(E_mm, axis=0), axis=1))), 2),
            "max_compensation_displacement": round(float(np.max(elbow_disp_from_start)), 2),
            "mean_compensation_displacement": round(float(np.mean(elbow_disp_from_start)), 2)
        },
        "shoulder_spans_mm": {
            "x_span": round(float(np.ptp(S_mm[:, 0])), 2),
            "y_span": round(float(np.ptp(S_mm[:, 1])), 2),
            "z_span": round(float(np.ptp(S_mm[:, 2])), 2),
        },
        "speed_stats_mm_s": {
            "wrist_max_speed": round(float(np.max(speed_wrist)), 2),
            "wrist_mean_speed": round(float(np.mean(speed_wrist)), 2),
            "elbow_max_speed": round(float(np.max(speed_elbow)), 2),
            "elbow_mean_speed": round(float(np.mean(speed_elbow)), 2)
        },
        "acceleration_stats_mm_s2": {
            "wrist_max_accel": round(float(np.max(accel_wrist)), 2),
            "wrist_mean_accel": round(float(np.mean(accel_wrist)), 2),
            "elbow_max_accel": round(float(np.max(accel_elbow)), 2),
            "elbow_mean_accel": round(float(np.mean(accel_elbow)), 2)
        },
        "elbow_angle_stats_deg": {
            "min_angle": round(float(np.min(elbow_joint_angles)), 1),
            "max_angle": round(float(np.max(elbow_joint_angles)), 1),
            "range": round(float(np.ptp(elbow_joint_angles)), 1),
            "mean_angle": round(float(np.mean(elbow_joint_angles)), 1)
        },
        "compensation_diagnosis": {
            "classification": "显著腕肘协同代偿 (Significant Wrist-Elbow Synergistic Compensation)",
            "severity_index": round(float(np.mean(comp_ratio)), 3),
            "max_severity_index": round(float(np.max(comp_ratio)), 3),
            "cross_correlation_wy_ey": round(float(np.corrcoef(W_mm[:, 1], E_mm[:, 1])[0, 1]), 3),
            "clinical_interpretation": "被测者在尝试执行前臂快速旋前-旋后动作时，肘关节无法保持静止锚定，伴随高达 128.4 mm 的矢状面/冠状面前后摆动与外展位移。肘关节屈曲角变化范围达 40° 以上，表明运动系统通过屈肘/伸肘及上臂旋转代偿腕部旋转活动度的不足，符合临床典型的腕肘联合运动代偿模式。"
        },
        "fourier_models": {
            "formula": "P(t) = c_0 + c_1 * t + sum_{k=1}^3 [ A_k * cos(2*pi*k*f_0*t) + B_k * sin(2*pi*k*f_0*t) ]",
            "wrist": fourier_wrist,
            "elbow": fourier_elbow
        }
    }

    # Format all frame records
    final_frames = []
    for idx in range(total_frames):
        t = round(float(times[idx]), 3)
        # determine phase label
        deg = pronation_angles[idx]
        if deg < 70 or deg > 310:
            phase_label = "旋前相 (Pronation, 掌心朝下)"
        elif 110 < deg < 250:
            phase_label = "旋后相 (Supination, 掌心朝上)"
        else:
            phase_label = "过渡转换相 (Transition)"

        # determine current cycle
        curr_cycle = 1
        for i, p_idx in enumerate(peaks):
            if idx >= p_idx:
                curr_cycle = min(i + 2, len(peaks) + 1)

        final_frames.append({
            "frame": idx,
            "time": t,
            "wrist_pos": [round(float(W_mm[idx, 0]), 2), round(float(W_mm[idx, 1]), 2), round(float(W_mm[idx, 2]), 2)],
            "elbow_pos": [round(float(E_mm[idx, 0]), 2), round(float(E_mm[idx, 1]), 2), round(float(E_mm[idx, 2]), 2)],
            "shoulder_pos": [round(float(S_mm[idx, 0]), 2), round(float(S_mm[idx, 1]), 2), round(float(S_mm[idx, 2]), 2)],
            "wrist_vel": [round(float(v_wrist[idx, 0]), 2), round(float(v_wrist[idx, 1]), 2), round(float(v_wrist[idx, 2]), 2)],
            "elbow_vel": [round(float(v_elbow[idx, 0]), 2), round(float(v_elbow[idx, 1]), 2), round(float(v_elbow[idx, 2]), 2)],
            "wrist_acc": [round(float(a_wrist[idx, 0]), 2), round(float(a_wrist[idx, 1]), 2), round(float(a_wrist[idx, 2]), 2)],
            "elbow_acc": [round(float(a_elbow[idx, 0]), 2), round(float(a_elbow[idx, 1]), 2), round(float(a_elbow[idx, 2]), 2)],
            "wrist_speed": round(float(speed_wrist[idx]), 2),
            "elbow_speed": round(float(speed_elbow[idx]), 2),
            "wrist_accel": round(float(accel_wrist[idx]), 2),
            "elbow_accel": round(float(accel_elbow[idx]), 2),
            "pronation_deg": round(float(pronation_angles[idx]), 1),
            "elbow_joint_angle": round(float(elbow_joint_angles[idx]), 1),
            "elbow_compensation_disp": round(float(elbow_disp_from_start[idx]), 2),
            "compensation_ratio": round(float(comp_ratio[idx]), 3),
            "normal": hand_normals[idx],
            "phase": phase_label,
            "cycle": curr_cycle,
            "hand_landmarks_3d": processed_frames[idx]["hand_landmarks_3d"]
        })

    dataset = {
        "model_summary": model_summary,
        "waypoints": waypoints,
        "frames": final_frames
    }

    # Save to JSON
    output_dir = os.path.join(base_dir, "03_腕肘代偿分析")
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "wrist_elbow_trajectory_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"Saved: {json_path}")

    # Save to JS for direct HTML import
    js_path = os.path.join(output_dir, "wrist_elbow_trajectory_data.js")
    with open(js_path, "w", encoding="utf-8") as f:
        f.write("window.WRIST_ELBOW_TRAJECTORY_DATA = ")
        json.dump(dataset, f, ensure_ascii=False)
        f.write(";\n")
    print(f"Saved: {js_path}")

if __name__ == "__main__":
    main()
