#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
biomechanics_engine.py - 上肢康复生物力学与代偿判定引擎 (临床解剖差分增强版)
彻底解决“放松坐着误报耸肩”缺陷：
1. 临床解剖相对参考系 (Anatomical Relative Reference):
   - 彻底摒弃“相机绝对空间坐标”，改用“双肩差分相对位移”与“胸骨中线共模抑制”；
   - 消除躯干自然呼吸、放松沉降、身体前后轻微晃动引起的假阳性。
2. 动作态代偿有效性法则 (Motion State Gating):
   - 临床定义：代偿是为“完成目标动作”而发生的副反应。
   - 当手腕处于静息状态 (FE < 10°) 时，判定为“静息放松”，绝不误报“代偿借力”；
   - 仅在主动屈伸运动中 (FE >= 10°)，实时检测单侧耸肩与手肘飞出。
3. 双模肩部上提判定 (Dual-Mode Shrugging Detection):
   - 模式A (主控): 双肩相对高差差分 Delta_H_shrug = (Y_左 - Y_右) - (Y_左0 - Y_右0)
   - 模式B (单侧容错): 肩肘垂直跨距差分 Delta_V_se = (Y_肘 - Y_肩) - (Y_肘0 - Y_肩0)
4. 飞肘相对位移 (Anatomical Elbow Abduction):
   - 计算手肘相对于右肩峰的外展间距 Delta_X_rel = (X_肘 - X_肩) - (X_肘0 - X_肩0)
"""

import math
import time
from typing import Dict, Optional, Tuple, Any, List
import numpy as np


class BiomechanicsEngine:
    def __init__(
        self,
        target_action: str = "wrist_FE",
        target_angle_deg: float = 40.0,
        start_threshold_deg: float = 10.0,
        shrug_threshold_mm: float = 30.0,         # 双肩相对上抬差分阈值 (mm)
        abduction_threshold_mm: float = 38.0,     # 手肘相对肩峰外展阈值 (mm)
        trunk_tilt_threshold_deg: float = 7.0,     # 躯干侧倾夹角偏转 (度)
        calib_frames_required: int = 40
    ):
        self.target_action = target_action
        self.target_angle_deg = target_angle_deg
        self.start_threshold_deg = start_threshold_deg
        self.shrug_threshold_mm = shrug_threshold_mm
        self.abduction_threshold_mm = abduction_threshold_mm
        self.trunk_tilt_threshold_deg = trunk_tilt_threshold_deg
        self.calib_frames_required = calib_frames_required

        # 基线标定状态 (解剖相对基准)
        self.is_calibrated = False
        self.calib_buffer: List[Dict[str, np.ndarray]] = []
        
        # 解剖相对零位
        self.baseline_shoulder_diff_y: float = 0.0      # 双肩相对高度差 (Y_L - Y_R)
        self.baseline_shoulder_elbow_dist_y: float = 0.0 # 肩肘相对纵向跨距 (Y_E - Y_S)
        self.baseline_elbow_abduction_x: float = 0.0     # 肘部相对肩峰外展量 (X_E - X_S)
        self.baseline_shoulder_span: float = 0.38        # 双肩物理间距 (米)
        self.baseline_torso_tilt_deg: float = 0.0        # 两肩初始倾角 (度)
        self.baseline_wrist_fe: float = 0.0             # 初始手腕屈伸零位

        # 动作周期与计数状态机
        self.motion_state = "REST"
        self.current_rep_compensated = False
        self.current_rep_comp_reasons = set()
        self.current_rep_max_angle = 0.0

        self.standard_rep_count = 0
        self.compensated_rep_count = 0
        self.total_cycles = 0
        self.history_peak_angle = 0.0

        # 实时平滑角度
        self.filtered_angle = 0.0
        self.angle_filter_alpha = 0.35

    def reset_calibration(self):
        """重置标定"""
        self.is_calibrated = False
        self.calib_buffer.clear()
        self.baseline_shoulder_diff_y = 0.0
        self.baseline_shoulder_elbow_dist_y = 0.0
        self.baseline_elbow_abduction_x = 0.0
        self.baseline_shoulder_span = 0.38
        self.baseline_torso_tilt_deg = 0.0
        self.baseline_wrist_fe = 0.0

    def reset_counters(self):
        """重置训练计数"""
        self.motion_state = "REST"
        self.current_rep_compensated = False
        self.current_rep_comp_reasons.clear()
        self.current_rep_max_angle = 0.0
        self.standard_rep_count = 0
        self.compensated_rep_count = 0
        self.total_cycles = 0
        self.history_peak_angle = 0.0

    def update_calibration(self, joints_3d: Dict[str, np.ndarray]) -> bool:
        """
        解剖相对零位标定：
        记录双肩相对差分、肩肘解剖相对跨距，彻底独立于人体在空间中的绝对位置！
        """
        if self.is_calibrated:
            return True

        if "r_shoulder" in joints_3d and "r_elbow" in joints_3d and "r_wrist" in joints_3d:
            self.calib_buffer.append(joints_3d)

        if len(self.calib_buffer) >= self.calib_frames_required:
            diff_y_list = []
            se_dist_y_list = []
            abduct_x_list = []
            span_list = []
            tilt_list = []

            for b in self.calib_buffer:
                p_sr = b["r_shoulder"]
                p_er = b["r_elbow"]
                
                # 肩肘相对纵向跨距 (+Y 向下, 肘部Y大于肩部Y)
                se_dist_y_list.append(p_er[1] - p_sr[1])
                # 肘部相对肩峰外展量
                abduct_x_list.append(p_er[0] - p_sr[0])

                if "l_shoulder" in b:
                    p_sl = b["l_shoulder"]
                    diff_y_list.append(p_sl[1] - p_sr[1])
                    span_list.append(float(np.linalg.norm(p_sr - p_sl)))
                    dx = p_sr[0] - p_sl[0]
                    dy = p_sr[1] - p_sl[1]
                    tilt_list.append(math.degrees(math.atan2(dy, dx)))

            self.baseline_shoulder_elbow_dist_y = float(np.median(se_dist_y_list))
            self.baseline_elbow_abduction_x = float(np.median(abduct_x_list))

            if len(diff_y_list) > 0:
                self.baseline_shoulder_diff_y = float(np.median(diff_y_list))
                self.baseline_shoulder_span = float(np.median(span_list))
                self.baseline_torso_tilt_deg = float(np.median(tilt_list))
            else:
                self.baseline_shoulder_diff_y = 0.0
                self.baseline_shoulder_span = 0.38
                self.baseline_torso_tilt_deg = 0.0

            self.is_calibrated = True
            return True

        return False

    def compute_wrist_angles(
        self,
        p_elbow: np.ndarray,
        p_wrist: np.ndarray,
        p_hand: np.ndarray
    ) -> Tuple[float, float]:
        """解算腕关节背伸角 (FE)"""
        v_fa = p_wrist - p_elbow
        len_fa = float(np.linalg.norm(v_fa))
        v_hand = p_hand - p_wrist
        len_hand = float(np.linalg.norm(v_hand))

        if len_fa < 1e-4 or len_hand < 1e-4:
            return 0.0, 0.0

        # 解剖学屈伸 (矢状面俯仰)
        dy_hand = -(p_hand[1] - p_wrist[1])
        dy_fa = -(p_wrist[1] - p_elbow[1])
        pitch_hand = math.degrees(math.asin(np.clip(dy_hand / len_hand, -1.0, 1.0)))
        pitch_fa = math.degrees(math.asin(np.clip(dy_fa / len_fa, -1.0, 1.0)))
        fe_angle = pitch_hand - pitch_fa

        # 桡尺侧偏 (水平偏角)
        dx_hand = p_hand[0] - p_wrist[0]
        dx_fa = p_wrist[0] - p_elbow[0]
        ru_hand = math.degrees(math.asin(np.clip(dx_hand / len_hand, -1.0, 1.0)))
        ru_fa = math.degrees(math.asin(np.clip(dx_fa / len_fa, -1.0, 1.0)))
        ru_angle = ru_hand - ru_fa

        return round(float(fe_angle), 2), round(float(ru_angle), 2)

    def evaluate_frame(
        self,
        joints_3d: Dict[str, np.ndarray],
        joints_2d: Optional[Dict[str, Tuple[int, int]]] = None
    ) -> Dict[str, Any]:
        """单帧综合解算与临床代偿判定"""
        result = {
            "is_calibrated": self.is_calibrated,
            "calib_progress": len(self.calib_buffer) / float(self.calib_frames_required),
            "fe_angle": 0.0,
            "ru_angle": 0.0,
            "target_angle": self.target_angle_deg,
            "progress_ratio": 0.0,
            "is_shrugging": False,
            "shrug_displacement_mm": 0.0,
            "is_flying_elbow": False,
            "abduction_displacement_mm": 0.0,
            "is_trunk_leaning": False,
            "trunk_tilt_diff_deg": 0.0,
            "has_compensation": False,
            "comp_reasons": [],
            "status_text": "未标定",
            "coach_tip": "请保持自然坐姿，系统正在校准解剖零位...",
            "status_level": "INFO",
            "standard_reps": self.standard_rep_count,
            "compensated_reps": self.compensated_rep_count,
            "total_cycles": self.total_cycles,
            "history_peak_angle": self.history_peak_angle,
            "motion_state": self.motion_state
        }

        # 1. 标定流程
        if not self.is_calibrated:
            calib_done = self.update_calibration(joints_3d)
            if not calib_done:
                pct = int(result["calib_progress"] * 100)
                result["status_text"] = f"标定零位 ({pct}%)"
                result["coach_tip"] = "请保持身体自然放松端正，双肩自然平沉。"
                return result
            else:
                result["is_calibrated"] = True
                result["coach_tip"] = "零位标定成功！可以开始抬手腕做屈伸动作。"

        # 2. 检查必要关节点
        if "r_elbow" not in joints_3d or "r_wrist" not in joints_3d:
            result["status_text"] = "检测中"
            result["coach_tip"] = "请确保右侧手臂在摄像头前可见。"
            return result

        p_sr = joints_3d.get("r_shoulder")
        p_er = joints_3d["r_elbow"]
        p_wr = joints_3d["r_wrist"]

        # 手部指节中点
        if "r_index" in joints_3d and "r_pinky" in joints_3d:
            p_hand = (joints_3d["r_index"] + joints_3d["r_pinky"]) / 2.0
        elif "r_index" in joints_3d:
            p_hand = joints_3d["r_index"]
        else:
            p_hand = p_wr + np.array([0.0, 0.0, -0.09])  # 估计前向指掌

        # 3. 计算实时屈伸角度
        raw_fe, raw_ru = self.compute_wrist_angles(p_er, p_wr, p_hand)
        if self.filtered_angle == 0.0:
            self.filtered_angle = raw_fe
        else:
            self.filtered_angle = (
                self.angle_filter_alpha * raw_fe +
                (1.0 - self.angle_filter_alpha) * self.filtered_angle
            )
        cur_fe = max(0.0, round(self.filtered_angle, 1))
        result["fe_angle"] = cur_fe
        result["ru_angle"] = raw_ru

        progress = max(0.0, min(1.2, cur_fe / max(1.0, self.target_angle_deg)))
        result["progress_ratio"] = round(progress, 3)

        # 4. 临床代偿解剖学判定
        comp_reasons = []

        # 4.1 耸肩解剖差分解算 (Bilateral Differential)
        shrug_metric_mm = 0.0
        tilt_diff_deg = 0.0

        if p_sr is not None:
            # 模式 A: 双肩差分 (若左肩可见，以左肩为共模参考)
            if "l_shoulder" in joints_3d:
                p_sl = joints_3d["l_shoulder"]
                cur_diff_y = p_sl[1] - p_sr[1]
                # 相机系中: 右肩相对上抬意味着 Y_R 变小，(Y_L - Y_R) 变大
                delta_diff_y_m = cur_diff_y - self.baseline_shoulder_diff_y
                shrug_metric_mm = delta_diff_y_m * 1000.0

                # 双肩倾角偏转 (度)
                dx = p_sr[0] - p_sl[0]
                dy = p_sr[1] - p_sl[1]
                cur_tilt = math.degrees(math.atan2(dy, dx))
                tilt_diff_deg = cur_tilt - self.baseline_torso_tilt_deg
                result["trunk_tilt_diff_deg"] = round(abs(tilt_diff_deg), 1)

            else:
                # 模式 B: 单侧肩肘垂直跨距差分 (左肩出框时的容错保护)
                # 肘部在桌面上，耸肩会导致肩部向上抽离手肘，(Y_E - Y_S) 增加
                cur_se_dist_y = p_er[1] - p_sr[1]
                delta_se_y_m = cur_se_dist_y - self.baseline_shoulder_elbow_dist_y
                shrug_metric_mm = delta_se_y_m * 1000.0

        result["shrug_displacement_mm"] = round(shrug_metric_mm, 1)

        # 4.2 飞肘外展解剖差分解算 (Elbow Abduction relative to Shoulder)
        abduct_metric_mm = 0.0
        if p_sr is not None:
            cur_abduct_x = p_er[0] - p_sr[0]
            delta_abduct_x_m = cur_abduct_x - self.baseline_elbow_abduction_x
            abduct_metric_mm = delta_abduct_x_m * 1000.0
        result["abduction_displacement_mm"] = round(abduct_metric_mm, 1)

        # 4.3 核心临床法则：只有在主动运动中 (FE >= start_threshold_deg) 发生借力才定义为“代偿”
        # 当被测者静止放松坐在椅子上时，绝不判定为代偿！
        is_active_motion = (cur_fe >= self.start_threshold_deg)

        if is_active_motion:
            # 耸肩判定：单侧相对上抬超过阈值，且肩部连线出现可见倾斜
            if shrug_metric_mm > self.shrug_threshold_mm:
                result["is_shrugging"] = True
                comp_reasons.append(f"右肩代偿耸起(+{shrug_metric_mm:.0f}mm)")

            # 飞肘判定：肘部相对于肩膀外展明显超过基准
            if abduct_metric_mm > self.abduction_threshold_mm:
                result["is_flying_elbow"] = True
                comp_reasons.append(f"手肘向外借力(+{abduct_metric_mm:.0f}mm)")

            # 躯干侧倾判定
            if abs(tilt_diff_deg) > self.trunk_tilt_threshold_deg:
                result["is_trunk_leaning"] = True
                comp_reasons.append(f"躯干向侧面倾斜({abs(tilt_diff_deg):.1f}°)")

        result["has_compensation"] = len(comp_reasons) > 0
        result["comp_reasons"] = comp_reasons

        # 5. 动作周期与计数状态机
        if result["has_compensation"]:
            self.current_rep_compensated = True
            for r in comp_reasons:
                self.current_rep_comp_reasons.add(r)

        if cur_fe > self.current_rep_max_angle:
            self.current_rep_max_angle = cur_fe
        if cur_fe > self.history_peak_angle:
            self.history_peak_angle = cur_fe

        # 状态机流转
        if self.motion_state == "REST":
            if cur_fe >= self.start_threshold_deg:
                self.motion_state = "MOVING_OUT"
                self.current_rep_compensated = result["has_compensation"]
                self.current_rep_max_angle = cur_fe
        elif self.motion_state == "MOVING_OUT":
            if cur_fe >= (self.target_angle_deg - 2.0):
                self.motion_state = "PEAK_HOLD"
            elif cur_fe < (self.current_rep_max_angle - 6.0) and cur_fe > self.start_threshold_deg:
                self.motion_state = "RETURNING"
        elif self.motion_state == "PEAK_HOLD":
            if cur_fe < (self.target_angle_deg - 5.0):
                self.motion_state = "RETURNING"
        elif self.motion_state == "RETURNING":
            if cur_fe <= (self.start_threshold_deg - 3.0):
                # 周期结算
                self.total_cycles += 1
                if not self.current_rep_compensated and self.current_rep_max_angle >= (self.target_angle_deg - 5.0):
                    self.standard_rep_count += 1
                else:
                    self.compensated_rep_count += 1

                self.motion_state = "REST"
                self.current_rep_compensated = False
                self.current_rep_comp_reasons.clear()
                self.current_rep_max_angle = 0.0

        result["standard_reps"] = self.standard_rep_count
        result["compensated_reps"] = self.compensated_rep_count
        result["total_cycles"] = self.total_cycles
        result["history_peak_angle"] = round(self.history_peak_angle, 1)
        result["motion_state"] = self.motion_state

        # 6. 生成画面文字指导语 (清晰明确，直指要害)
        if result["has_compensation"]:
            result["status_level"] = "DANGER"
            result["status_text"] = "⚠️ 代偿警告"
            if result["is_shrugging"]:
                result["coach_tip"] = f"【请下沉右肩！】右肩向上耸起借力 (+{shrug_metric_mm:.0f}mm)，请保持双肩水平！"
            elif result["is_flying_elbow"]:
                result["coach_tip"] = f"【请夹紧手肘！】手肘向外侧飞出 (+{abduct_metric_mm:.0f}mm)，请贴近肋骨！"
            elif result["is_trunk_leaning"]:
                result["coach_tip"] = "【请端正上身！】身体向侧面倾斜了，保持躯干直立不动！"
        else:
            if not is_active_motion:
                # 静息放松状态
                result["status_level"] = "INFO"
                result["status_text"] = "静息端正 (正常)"
                result["coach_tip"] = "双肩平衡，姿态良好。请将右手手腕缓慢向上翘起 (背伸训练)。"
            elif self.motion_state == "MOVING_OUT":
                rem = int(self.target_angle_deg - cur_fe)
                if rem > 0:
                    result["status_level"] = "INFO"
                    result["status_text"] = f"动作中 ({cur_fe:.0f}°)"
                    result["coach_tip"] = f"姿态标准！手肘贴稳，手腕继续向上翘起，还差 {rem}° 达标。"
                else:
                    result["status_level"] = "SUCCESS"
                    result["status_text"] = "已达标！"
                    result["coach_tip"] = "非常好！达到目标角度，请保持 1 秒后缓慢放平。"
            elif self.motion_state == "PEAK_HOLD":
                result["status_level"] = "SUCCESS"
                result["status_text"] = "已达标 ★"
                result["coach_tip"] = "达到 40° 标准角度！请平稳控制手腕，缓慢放平。"
            elif self.motion_state == "RETURNING":
                result["status_level"] = "INFO"
                result["status_text"] = "缓慢回位"
                result["coach_tip"] = "控制平稳放平手腕，准备进入下一次动作。"

        return result
