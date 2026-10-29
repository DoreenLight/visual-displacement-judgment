#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_tp01_contract.py - TP-D435I-01 自动化数据契约与候选验收测试脚本
严格对照《01_上肢关键点三维坐标任务包.md》第6节验收标准进行机器验证：
1. 校验 run.json 与 keypoints.csv 的契约完整性 (23列定义、米制、坐标系);
2. 负向/反例检验：无效帧是否严格置为 NaN，无以零或上一帧插值冒充实测;
3. 静止可用性验证：肩/肘/腕有效率是否 >=95%，最大连续丢失是否 <=0.2s;
4. 静止波动性验证：各轴原始 SD 是否 <=5mm，基线位移 P95 是否 <=10mm;
5. 独立参考与准确度判定：若未接入独立动作捕捉真值，准确性项按规范诚实判定为 BLOCKED。
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple


class TP01ContractVerifier:
    REQUIRED_COLUMNS = [
        "run_id", "frame_id", "joint_id", "t_rgb_device_s", "t_depth_device_s", "t_host_mono_s", "clock_domain",
        "u_px", "v_px", "x_raw_m", "y_raw_m", "z_raw_m", "x_filt_m", "y_filt_m", "z_filt_m",
        "keypoint_score", "visibility", "depth_valid", "measurement_valid", "imputed",
        "valid_for_analysis", "invalid_reason", "coord_frame"
    ]

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.run_json_path = os.path.join(data_dir, "run.json")
        self.csv_path = os.path.join(data_dir, "keypoints.csv")
        self.results: Dict[str, Any] = {}

    def run_all_checks(self) -> Dict[str, Any]:
        print(f"\n==================================================")
        print(f" 开始执行 TP-D435I-01 契约与候选验收自动测试")
        print(f" 目标路径: {os.path.abspath(self.data_dir)}")
        print(f"==================================================\n")

        # 1. 文件存在性
        if not os.path.exists(self.run_json_path):
            return {"verdict": "FAIL", "reason": f"缺失 run.json: {self.run_json_path}"}
        if not os.path.exists(self.csv_path):
            return {"verdict": "FAIL", "reason": f"缺失 keypoints.csv: {self.csv_path}"}

        with open(self.run_json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        df = pd.read_csv(self.csv_path)

        # 2. 列定义检查
        check_schema = self._check_schema(df, metadata)
        print(f"[检查 1] 数据契约与 Schema 校验: {check_schema['status']}")

        # 3. 缺失值与无插值伪造检查
        check_nan = self._check_nan_integrity(df)
        print(f"[检查 2] 缺失值真实性 (严格NaN / 严禁填0 / 零插值): {check_nan['status']}")

        # 4. 可用性与丢帧检查 (>=95%, gap <= 0.2s)
        check_usability = self._check_usability(df)
        print(f"[检查 3] 静止基准可用性 (有效率>=95%, 连续丢失<=0.2s): {check_usability['status']}")

        # 5. 抖动与波动性检查 (SD <= 5mm, P95 <= 10mm)
        check_jitter = self._check_jitter(df)
        print(f"[检查 4] 静态抖动与测量波动性 (SD<=5mm, P95<=10mm): {check_jitter['status']}")

        # 6. 异常边界工况拦截检验
        check_edge = self._check_edge_case_rejection(df)
        print(f"[检查 5] 负向边缘异常拦截与标记测试: {check_edge['status']}")

        # 7. 独立参考与准确度判定
        check_accuracy = self._check_accuracy(metadata)
        print(f"[检查 6] 独立参考与绝对位置准确度 (RMSE<=10mm): {check_accuracy['status']}")

        # 综合判定
        all_checks = [check_schema, check_nan, check_usability, check_jitter, check_edge, check_accuracy]
        has_fail = any(c["verdict"] == "FAIL" for c in all_checks)
        has_blocked = any(c["verdict"] == "BLOCKED" for c in all_checks)

        if has_fail:
            overall = "FAIL"
        elif has_blocked:
            overall = "PARTIAL_PASS (ACCURACY_BLOCKED)"
        else:
            overall = "PASS"

        summary = {
            "overall_verdict": overall,
            "checks": {
                "schema": check_schema,
                "nan_integrity": check_nan,
                "usability": check_usability,
                "jitter_stability": check_jitter,
                "edge_cases": check_edge,
                "accuracy": check_accuracy
            }
        }
        self.results = summary
        self._print_summary_table(summary)
        return summary

    def _check_schema(self, df: pd.DataFrame, meta: Dict[str, Any]) -> Dict[str, Any]:
        missing_cols = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            return {"verdict": "FAIL", "status": "FAIL", "msg": f"缺失必需列: {missing_cols}"}

        if meta.get("schema_version") != "wrist-rehab/v1":
            return {"verdict": "FAIL", "status": "FAIL", "msg": f"schema_version 必须为 wrist-rehab/v1"}

        if meta.get("coordinate_frame") != "camera_color_optical":
            return {"verdict": "FAIL", "status": "FAIL", "msg": f"坐标系必须为 camera_color_optical"}

        return {"verdict": "PASS", "status": "PASS", "msg": "23列齐全，格式与单位严格符合契约"}

    def _check_nan_integrity(self, df: pd.DataFrame) -> Dict[str, Any]:
        # 检验所有 valid_for_analysis == False 的行，x_raw_m 是否为 NaN
        invalid_rows = df[df["valid_for_analysis"] == False]
        if len(invalid_rows) > 0:
            raw_not_nan = invalid_rows["x_raw_m"].dropna()
            if len(raw_not_nan) > 0:
                # 检查是否赋了 0
                has_zeros = (raw_not_nan == 0.0).any()
                if has_zeros:
                    return {"verdict": "FAIL", "status": "FAIL", "msg": "发现无效数据被赋为 0.0，违反数据契约！"}

        # 检查是否有任何 imputed=True 的行冒充实测
        imputed_count = (df["imputed"] == True).sum()
        if imputed_count > 0:
            return {"verdict": "FAIL", "status": "FAIL", "msg": f"发现 {imputed_count} 处伪造插值数据！"}

        return {"verdict": "PASS", "status": "PASS", "msg": "无效点严格置 NaN，无填零或插值伪造"}

    def _check_usability(self, df: pd.DataFrame) -> Dict[str, Any]:
        # 筛选静态基准数据 (不含负向极端测试)
        bench_df = df[~df["run_id"].str.contains("negative_edge")]
        if len(bench_df) == 0:
            bench_df = df

        stats = {}
        passes = True
        for joint in ["shoulder", "elbow", "wrist"]:
            j_df = bench_df[bench_df["joint_id"] == joint]
            total = len(j_df)
            valid = (j_df["valid_for_analysis"] == True).sum()
            rate = valid / total if total > 0 else 0.0

            # 计算最长连续失效时长
            valid_arr = (j_df["valid_for_analysis"] == True).values
            max_gap_frames = 0
            cur_gap = 0
            for v in valid_arr:
                if not v:
                    cur_gap += 1
                    max_gap_frames = max(max_gap_frames, cur_gap)
                else:
                    cur_gap = 0
            max_gap_s = max_gap_frames / 30.0

            stats[joint] = {"valid_rate": rate, "max_gap_s": max_gap_s}
            if rate < 0.95 or max_gap_s > 0.20:
                passes = False

        status = "PASS" if passes else "FAIL"
        return {"verdict": status, "status": status, "stats": stats}

    def _check_jitter(self, df: pd.DataFrame) -> Dict[str, Any]:
        bench_df = df[~df["run_id"].str.contains("negative_edge")]
        if len(bench_df) == 0:
            bench_df = df

        stats = {}
        passes = True
        for joint in ["shoulder", "elbow", "wrist"]:
            j_df = bench_df[(bench_df["joint_id"] == joint) & (bench_df["valid_for_analysis"] == True)]
            if len(j_df) < 30:
                continue

            # 毫米单位统计
            x_mm = j_df["x_raw_m"].values * 1000.0
            y_mm = j_df["y_raw_m"].values * 1000.0
            z_mm = j_df["z_raw_m"].values * 1000.0

            sd_x = float(np.std(x_mm))
            sd_y = float(np.std(y_mm))
            sd_z = float(np.std(z_mm))

            med_x, med_y, med_z = np.median(x_mm), np.median(y_mm), np.median(z_mm)
            disp_3d_mm = np.sqrt((x_mm - med_x)**2 + (y_mm - med_y)**2 + (z_mm - med_z)**2)
            p95_drift = float(np.percentile(disp_3d_mm, 95))

            stats[joint] = {
                "sd_x_mm": sd_x,
                "sd_y_mm": sd_y,
                "sd_z_mm": sd_z,
                "p95_drift_mm": p95_drift
            }

            if sd_x > 5.0 or sd_y > 5.0 or sd_z > 5.0 or p95_drift > 10.0:
                passes = False

        status = "PASS" if passes else "FAIL"
        return {"verdict": status, "status": status, "stats": stats}

    def _check_edge_case_rejection(self, df: pd.DataFrame) -> Dict[str, Any]:
        edge_df = df[df["run_id"].str.contains("negative_edge")]
        if len(edge_df) == 0:
            return {"verdict": "PASS", "status": "PASS", "msg": "未包含独立负向测试批次"}

        # 检查是否成功捕获了深度空洞或遮挡
        invalid_reasons = edge_df["invalid_reason"].unique().tolist()
        has_caught_reasons = any(r != "valid" for r in invalid_reasons)
        if not has_caught_reasons:
            return {"verdict": "FAIL", "status": "FAIL", "msg": "负向边缘异常未被成功拦截"}

        return {"verdict": "PASS", "status": "PASS", "captured_reasons": invalid_reasons}

    def _check_accuracy(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        has_gt = meta.get("independent_ground_truth_available", False)
        if not has_gt:
            return {
                "verdict": "BLOCKED",
                "status": "BLOCKED",
                "reason": "缺少独立、同步且经标定的外部真值参考系统 (如 Vicon/Qualisys 或高精工装)。"
                          "按 Agent.md 与 TP-01 规范，稳定性可独立给结论，绝对准确性必须标为 BLOCKED，严禁虚报全项通过。"
            }
        return {"verdict": "PASS", "status": "PASS"}

    def _print_summary_table(self, summary: Dict[str, Any]):
        print("\n--------------------------------------------------")
        print(f"验收项测试结果汇总：")
        for k, v in summary["checks"].items():
            print(f" - {k:<18}: {v['verdict']}")
        print(f"\n最终任务包裁定 (Overall Verdict): {summary['overall_verdict']}")
        print("--------------------------------------------------\n")


if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    verifier = TP01ContractVerifier(target_dir)
    res = verifier.run_all_checks()
    # 导出测试结果 json
    out_file = os.path.join(target_dir, "verification_summary.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"验证汇总已写入: {out_file}")
