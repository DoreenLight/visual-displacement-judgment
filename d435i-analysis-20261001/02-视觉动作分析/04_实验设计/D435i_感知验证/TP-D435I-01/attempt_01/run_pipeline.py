#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline.py - TP-D435I-01 任务执行主程序
支持三种工作模式：
1. --mode live     : 接入 Intel RealSense D435i 物理相机实时运行；
2. --mode fixture  : 无相机模式，运行全套 S1/S2/S3 静态测试与负向边缘案例测试夹具；
3. --mode replay   : 离线回放 07_资料库/ 中的真实康复视频。
生成符合 wrist-rehab/v1 契约的 run.json 与 keypoints.csv，并产出时序三维可视化图。
"""

import os
import sys
import time
import json
import argparse
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from d435i_capture import D435iCaptureManager, CameraIntrinsics
from deprojection_engine import UpperLimbDeprojectionEngine
from synthetic_fixture import SyntheticD435iFixture

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("TP01Pipeline")


def run_tp01_pipeline(
    mode: str = "fixture",
    duration_s: float = 30.0,
    training_side: str = "right",
    output_dir: str = ".",
    run_id: str = "RUN_TP01_BENCHMARK_001"
):
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"=== 启动 TP-D435I-01 任务流水线 ===")
    logger.info(f"模式: {mode}, 训练侧: {training_side}, 目标输出: {output_dir}")

    engine = UpperLimbDeprojectionEngine(
        training_side=training_side,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
        depth_patch_radius=3,
        min_depth_m=0.4,
        max_depth_m=2.5,
        filter_alpha=0.35
    )

    all_records: List[Dict[str, Any]] = []
    run_metadata: Dict[str, Any] = {
        "schema_version": "wrist-rehab/v1",
        "run_id": run_id,
        "task_id": "TP-D435I-01",
        "attempt_id": "attempt_01",
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": mode,
        "training_side": training_side,
        "target_hardware": "Intel RealSense D435i",
        "hardware_connected": False,
        "coordinate_frame": "camera_color_optical",
        "unit": "meters",
        "conditions": []
    }

    if mode == "live":
        cap_mgr = D435iCaptureManager(width=848, height=480, fps=30)
        available, msg = cap_mgr.is_hardware_available()
        if not available:
            logger.error(f"物理相机未连接: {msg}。请切换为 --mode fixture 运行离线测试夹具。")
            sys.exit(1)

        if not cap_mgr.start():
            logger.error("D435i 相机启动失败。")
            sys.exit(1)

        run_metadata["hardware_connected"] = True
        run_metadata["device_info"] = cap_mgr.device_info
        intrinsics = cap_mgr.intrinsics
        run_metadata["intrinsics"] = {
            "width": intrinsics.width,
            "height": intrinsics.height,
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "ppx": intrinsics.ppx,
            "ppy": intrinsics.ppy,
            "depth_scale": intrinsics.depth_scale
        }

        total_frames = int(round(duration_s * 30.0))
        logger.info(f"开始实时采集 {total_frames} 帧...")
        for f in range(total_frames):
            frame_data = cap_mgr.read_frame()
            if frame_data is None:
                continue
            records = engine.process_frame(
                color_bgr=frame_data["color_bgr"],
                depth_raw=frame_data["depth_raw"],
                depth_scale=frame_data["depth_scale"],
                intrinsics=intrinsics,
                run_id=run_id,
                frame_id=f,
                t_rgb_device_s=frame_data["t_rgb_device_s"],
                t_depth_device_s=frame_data["t_depth_device_s"],
                t_host_mono_s=frame_data["t_host_mono_s"]
            )
            all_records.extend(records)
        cap_mgr.stop()

    elif mode == "fixture":
        logger.info("当前处于无硬件测试环境，启动高保真 D435i 合成夹具 (Synthetic Fixture)...")
        fixture = SyntheticD435iFixture(width=848, height=480, fps=30, distance_m=1.25)
        intrinsics = fixture.intrinsics
        run_metadata["hardware_connected"] = False
        run_metadata["fixture_info"] = {
            "source": "SyntheticD435iFixture",
            "distance_m": 1.25,
            "noise_model": "Gaussian(sigma=2.0mm) + Physiological Tremor(1.0mm, 10Hz)",
            "intrinsics": {
                "width": intrinsics.width,
                "height": intrinsics.height,
                "fx": intrinsics.fx,
                "fy": intrinsics.fy,
                "ppx": intrinsics.ppx,
                "ppy": intrinsics.ppy,
                "depth_scale": intrinsics.depth_scale
            }
        }

        # 运行四组测试工况
        test_plan = [
            ("S1_natural_sitting", duration_s),
            ("S2_grip_posture", duration_s),
            ("S3_rehab_ready", duration_s),
            ("negative_edge_cases", 5.0)  # 包含深度空洞、遮挡与跳变的负向测试
        ]

        frame_counter = 0
        for cond_name, cond_dur in test_plan:
            logger.info(f"--> 执行工况: {cond_name} (时长: {cond_dur:.1f}s)...")
            cond_frames = int(round(cond_dur * 30.0))
            run_metadata["conditions"].append({
                "condition_name": cond_name,
                "duration_s": cond_dur,
                "start_frame": frame_counter,
                "end_frame": frame_counter + cond_frames - 1
            })

            stream = fixture.generate_condition_stream(condition=cond_name, duration_s=cond_dur)
            for item in stream:
                records = engine.process_frame(
                    color_bgr=item["color_bgr"],
                    depth_raw=item["depth_raw"],
                    depth_scale=item["depth_scale"],
                    intrinsics=intrinsics,
                    run_id=f"{run_id}_{cond_name}",
                    frame_id=frame_counter,
                    t_rgb_device_s=item["t_rgb_device_s"],
                    t_depth_device_s=item["t_depth_device_s"],
                    t_host_mono_s=item["t_host_mono_s"]
                )
                all_records.extend(records)
                frame_counter += 1

    engine.close()

    # 4. 导出符合 wrist-rehab/v1 的 keypoints.csv
    csv_columns = [
        "run_id", "frame_id", "joint_id", "t_rgb_device_s", "t_depth_device_s", "t_host_mono_s", "clock_domain",
        "u_px", "v_px", "x_raw_m", "y_raw_m", "z_raw_m", "x_filt_m", "y_filt_m", "z_filt_m",
        "keypoint_score", "visibility", "depth_valid", "measurement_valid", "imputed",
        "valid_for_analysis", "invalid_reason", "coord_frame"
    ]
    df = pd.DataFrame(all_records)
    # 确保列顺序完全一致
    df = df[csv_columns]
    csv_path = os.path.join(output_dir, "keypoints.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"三维关键点契约表已生成: {csv_path} (共 {len(df)} 行数据)")

    # 5. 保存 run.json
    run_metadata["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run_metadata["total_records"] = len(df)
    run_metadata["csv_path"] = "keypoints.csv"
    json_path = os.path.join(output_dir, "run.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"元数据已写入: {json_path}")

    # 6. 生成分析与可视化图表
    generate_visual_summary(df, output_dir)
    logger.info("流水线执行圆满完成！")


def generate_visual_summary(df: pd.DataFrame, output_dir: str):
    """生成三维空间散点与各轴时序对比图表"""
    plt.figure(figsize=(14, 9), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. 腕部时序 X, Y, Z
    wrist_df = df[df["joint_id"] == "wrist"]
    valid_wrist = wrist_df[wrist_df["valid_for_analysis"] == True]

    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(wrist_df["frame_id"], wrist_df["x_raw_m"] * 1000, label="Raw X", color="lightcoral", alpha=0.6, lw=1)
    ax1.plot(wrist_df["frame_id"], wrist_df["x_filt_m"] * 1000, label="Causal Filt X", color="red", lw=1.5)
    ax1.set_title("Wrist X Position (mm) - Stability & EMA Filter", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Frame ID")
    ax1.set_ylabel("X (mm)")
    ax1.legend(loc="upper right")

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(wrist_df["frame_id"], wrist_df["y_raw_m"] * 1000, label="Raw Y", color="skyblue", alpha=0.6, lw=1)
    ax2.plot(wrist_df["frame_id"], wrist_df["y_filt_m"] * 1000, label="Causal Filt Y", color="blue", lw=1.5)
    ax2.set_title("Wrist Y Position (mm) - Stability & EMA Filter", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Frame ID")
    ax2.set_ylabel("Y (mm)")
    ax2.legend(loc="upper right")

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(wrist_df["frame_id"], wrist_df["z_raw_m"] * 1000, label="Raw Depth Z", color="lightgreen", alpha=0.6, lw=1)
    ax3.plot(wrist_df["frame_id"], wrist_df["z_filt_m"] * 1000, label="Causal Filt Z", color="green", lw=1.5)
    ax3.set_title("Wrist Depth Z Position (mm) - Stability & Noise Rejection", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Frame ID")
    ax3.set_ylabel("Z (mm)")
    ax3.legend(loc="upper right")

    # 4. 关节有效性与异常原因分布
    ax4 = plt.subplot(2, 2, 4)
    reason_counts = df["invalid_reason"].value_counts()
    colors = ["#2ecc71" if r == "valid" else "#e74c3c" for r in reason_counts.index]
    ax4.bar(reason_counts.index, reason_counts.values, color=colors, alpha=0.85)
    ax4.set_title("Keypoint Validity & Edge Case Breakdown", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Count")
    ax4.tick_params(axis="x", rotation=25)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "visual_summary.png")
    plt.savefig(chart_path)
    plt.close()
    logger.info(f"可视化汇总图已保存: {chart_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TP-D435I-01 Perception Pipeline")
    parser.add_argument("--mode", choices=["live", "fixture", "replay"], default="fixture", help="运行模式")
    parser.add_argument("--duration_s", type=float, default=30.0, help="各基准测试时长 (秒)")
    parser.add_argument("--training_side", default="right", choices=["right", "left"], help="训练侧")
    parser.add_argument("--output_dir", default=".", help="输出路径")
    parser.add_argument("--run_id", default="RUN_TP01_FIXTURE_001", help="运行 ID")
    args = parser.parse_args()

    run_tp01_pipeline(
        mode=args.mode,
        duration_s=args.duration_s,
        training_side=args.training_side,
        output_dir=args.output_dir,
        run_id=args.run_id
    )
