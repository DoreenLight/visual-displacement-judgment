import numpy as np
from biomechanics_engine import BiomechanicsEngine

def main():
    engine = BiomechanicsEngine(target_angle_deg=40.0, calib_frames_required=10)

    # 1. 标定阶段
    base_pose = {
        'r_shoulder': np.array([0.15, -0.20, 0.77]),
        'l_shoulder': np.array([-0.20, -0.20, 0.77]),
        'r_elbow': np.array([0.22, 0.05, 0.75]),
        'r_wrist': np.array([0.18, 0.08, 0.65]),
        'r_index': np.array([0.16, 0.09, 0.58]),
        'r_pinky': np.array([0.20, 0.09, 0.58])
    }

    for _ in range(10):
        engine.evaluate_frame({k: v.copy() for k, v in base_pose.items()})

    assert engine.is_calibrated, "Engine should be calibrated!"
    print("1. Calibration OK! Baseline shoulder diff:", engine.baseline_shoulder_diff_y)

    # 2. 关键测试：测试者【放松坐着】（身体自然整体下沉25mm，身体后靠30mm）
    slump_pose = {
        'r_shoulder': np.array([0.15, -0.175, 0.80]),   # 沉降+后移
        'l_shoulder': np.array([-0.20, -0.175, 0.80]),  # 双肩同步沉降
        'r_elbow': np.array([0.22, 0.075, 0.78]),
        'r_wrist': np.array([0.18, 0.08, 0.65]),
        'r_index': np.array([0.16, 0.09, 0.58]),
        'r_pinky': np.array([0.20, 0.09, 0.58])
    }
    res_slump = engine.evaluate_frame(slump_pose)
    print(f"2. 放松坐姿测试: status='{res_slump['status_text']}', is_shrugging={res_slump['is_shrugging']}, comp={res_slump['has_compensation']}")
    assert not res_slump['is_shrugging'], "【严重缺陷复现】放松坐姿绝对不应该判定为耸肩！"
    assert not res_slump['has_compensation'], "放松坐姿绝对不应该有代偿警告！"

    # 3. 正常手腕屈伸动作 (手腕抬起25度，无代偿)
    active_clean = {
        'r_shoulder': np.array([0.15, -0.20, 0.77]),
        'l_shoulder': np.array([-0.20, -0.20, 0.77]),
        'r_elbow': np.array([0.22, 0.05, 0.75]),
        'r_wrist': np.array([0.18, 0.08, 0.65]),
        'r_index': np.array([0.16, -0.05, 0.58]),       # 手指上翘 (背伸)
        'r_pinky': np.array([0.20, -0.05, 0.58])
    }
    res_clean = engine.evaluate_frame(active_clean)
    print(f"3. 正常背伸动作: fe={res_clean['fe_angle']}°, status='{res_clean['status_text']}', comp={res_clean['has_compensation']}")
    assert res_clean['fe_angle'] > 20.0, "应检测到背伸角"
    assert not res_clean['has_compensation'], "正常动作无代偿"

    # 4. 真实耸肩借力动作 (手腕上翘的同时，右肩单侧向上提 35mm，左肩不动)
    active_shrug = {
        'r_shoulder': np.array([0.15, -0.238, 0.77]),   # 右肩单侧上提 (Y减少38mm)
        'l_shoulder': np.array([-0.20, -0.20, 0.77]),   # 左肩保持不动
        'r_elbow': np.array([0.22, 0.05, 0.75]),
        'r_wrist': np.array([0.18, 0.08, 0.65]),
        'r_index': np.array([0.16, -0.05, 0.58]),       # 动作中
        'r_pinky': np.array([0.20, -0.05, 0.58])
    }
    res_shrug = engine.evaluate_frame(active_shrug)
    print(f"4. 真实耸肩测试: is_shrugging={res_shrug['is_shrugging']}, diff={res_shrug['shrug_displacement_mm']}mm, tip='{res_shrug['coach_tip']}'")
    assert res_shrug['is_shrugging'], "真实单侧耸肩必须被准确捕获！"

    # 5. 飞肘借力动作 (肘部外展 45mm)
    active_elbow = {
        'r_shoulder': np.array([0.15, -0.20, 0.77]),
        'l_shoulder': np.array([-0.20, -0.20, 0.77]),
        'r_elbow': np.array([0.268, 0.05, 0.75]),       # 肘部外展 (X增加48mm)
        'r_wrist': np.array([0.18, 0.08, 0.65]),
        'r_index': np.array([0.16, -0.05, 0.58]),       # 动作中
        'r_pinky': np.array([0.20, -0.05, 0.58])
    }
    res_elbow = engine.evaluate_frame(active_elbow)
    print(f"5. 飞肘测试: is_flying={res_elbow['is_flying_elbow']}, diff={res_elbow['abduction_displacement_mm']}mm, tip='{res_elbow['coach_tip']}'")
    assert res_elbow['is_flying_elbow'], "飞肘动作必须被捕获！"

    print("\n>>> 临床生物力学差分增强版所有测试 100% 通过！ <<<")

if __name__ == "__main__":
    main()
