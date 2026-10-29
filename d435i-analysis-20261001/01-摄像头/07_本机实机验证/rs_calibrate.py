#!/usr/bin/env python3
"""位移标定：验证「目标真的移动 X mm 时，读数是否也变化 X mm」。

这是整个项目最关键的一步
------------------------
之前所有测试测的都是**静置噪声底（重复性）**。
本脚本测的是**准确度**：输入已知位移，看输出是否正确。

方法论
------
1. 先用当前位置建立零点
2. 你手动移动杆到一个新位置，保持不动
3. 脚本测出「读数变化量」
4. 你告诉我「实际移动了多少」
5. 脚本计算：尺度误差、线性度、判定 2mm 是否可靠

为什么需要多轮
--------------
单次测量无法区分「尺度误差」与「随机误差」。
需要在多个位移量上重复，才能拟合出：
    读数变化 = a × 真实位移 + b
其中 a 是尺度系数，b 是零点偏移。

用法
----
# 交互式（推荐）：脚本提示你移动，你输入实际移动量
sudo -E .../rs_calibrate.py --interactive --physical 207

# 批量模式：每个位置按回车采样，最后输入实际值
sudo -E .../rs_calibrate.py --positions 0,2,5,10,20 --physical 207
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import macos_usb_fix
    macos_usb_fix.ensure_ready(verbose=False)
except Exception:
    pass

try:
    import pyrealsense2 as rs
except ImportError:
    sys.exit("缺少 pyrealsense2。")

import rs_measure as M

# 目标 ROI（由逐行剖面确定：杆的有效段）
DEFAULT_ROI = (342, 66, 43, 108)
# 参考面 ROI（桌面，用于检测相机是否被碰动）
DEFAULT_REF = (531, 236, 44, 81)


def start(alpha=0.4):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    prof = pipe.start(cfg)
    al = rs.align(rs.stream.color)
    tf = rs.temporal_filter()
    tf.set_option(rs.option.filter_smooth_alpha, alpha)
    tf.set_option(rs.option.filter_smooth_delta, 20)
    return pipe, al, tf


def grab(pipe, al, tf):
    fs = al.process(pipe.wait_for_frames(timeout_ms=10000))
    return np.asanyarray(tf.process(fs.get_depth_frame()).get_data())


def sample(pipe, al, tf, mT, mR, n=60, label=""):
    """采 n 帧，返回 (目标均值, 参考均值, 目标MDR)。"""
    sT, sR = [], []
    for i in range(n):
        d = grab(pipe, al, tf)
        sT.append(mT.read(d))
        sR.append(mR.read(d))
    sT, sR = np.array(sT), np.array(sR)
    st = M.series_mdr(sT)
    return (float(np.mean(sT)), float(np.mean(sR)),
            st["mdr"] if st else float("nan"), st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi", default=None, help="目标 ROI x,y,w,h")
    ap.add_argument("--ref", default=None, help="参考面 ROI x,y,w,h")
    ap.add_argument("--physical", type=float, default=227.0,
                    help="目标到相机的物理距离 mm")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--positions", default=None,
                    help="批量模式：逗号分隔的位移量 mm，如 0,2,5,10,20")
    ap.add_argument("--save", default="displacement_cal.json")
    args = ap.parse_args()

    troi = tuple(int(v) for v in args.roi.split(",")) if args.roi else DEFAULT_ROI
    rroi = tuple(int(v) for v in args.ref.split(",")) if args.ref else DEFAULT_REF

    print("=" * 92)
    print("位移标定 —— 验证「真实位移能否被正确测量」")
    print("=" * 92)
    print(f"目标 ROI   : {troi}")
    print(f"参考面 ROI : {rroi}")
    print(f"物理距离   : {args.physical} mm")
    print(f"每点采样   : {args.frames} 帧\n")

    pipe, al, tf = start(args.alpha)
    try:
        print("预热…")
        for _ in range(45):
            pipe.wait_for_frames(timeout_ms=10000)

        mT = M.Measurement(roi=troi, trim_mm=20.0)
        mR = M.Measurement(roi=rroi, trim_mm=20.0)

        # ── 零点 ──
        print(f"\n【零点】请确保杆静止在初始位置，采 {args.frames} 帧…")
        z0, r0, mdr0, st0 = sample(pipe, al, tf, mT, mR, args.frames)
        print(f"  目标 z0 = {z0:.2f} mm")
        print(f"  参考 r0 = {r0:.2f} mm")
        print(f"  静置 MDR = {mdr0:.3f} mm")
        if st0:
            print(f"  漂移 = {st0['drift']:.3f} mm")
        k0 = args.physical / z0
        print(f"  初步尺度系数 k = {args.physical}/{z0:.2f} = {k0:.5f}")

        print("\n" + "=" * 92)
        print("开始标定。每一步会提示你移动杆。")
        print("=" * 92)

        records = []

        if args.interactive:
            print("""
说明：
  - 每次移动后，脚本会采 60 帧
  - 然后询问你「相对初始位置移动了多少 mm」
  - 输入正数表示远离相机（Z 增大），负数表示靠近
  - 输入 q 结束标定
""")
            idx = 0
            while True:
                idx += 1
                print(f"\n--- 第 {idx} 个位置 ---")
                cmd = input("移动杆到新位置，然后按回车采样（输入 q 结束）: ").strip()
                if cmd.lower() == "q":
                    break
                z, r, mdr, st = sample(pipe, al, tf, mT, mR, args.frames)
                dz = z - z0
                dr = r - r0
                print(f"  目标读数     = {z:.2f} mm   (变化 {dz:+.3f})")
                print(f"  参考面读数   = {r:.2f} mm   (变化 {dr:+.3f})")
                print(f"  本次 MDR     = {mdr:.3f} mm")
                if abs(dr) > 3 * 2.472:
                    print(f"  ⚠️ 参考面漂移 {dr:+.2f}mm 较大，相机可能被碰动")
                    print(f"     扣除共模后目标净变化 = {dz-dr:+.3f} mm")
                try:
                    truth = float(input("  实际移动了多少 mm？（如 2.0 / -5.0）: "))
                except ValueError:
                    print("  输入无效，跳过此点")
                    continue
                records.append(dict(truth=truth, measured=dz,
                                    measured_net=dz - dr,
                                    ref_drift=dr, mdr=mdr, z=z, r=r))
                print(f"  → 记录: 真实 {truth:+.1f}mm  测得 {dz:+.3f}mm  "
                      f"误差 {dz-truth:+.3f}mm")

        elif args.positions:
            # 批量模式：按提示移动，每次回车采样，最后输入实际值
            pos_list = [float(v) for v in args.positions.split(",")]
            print(f"将在 {len(pos_list)} 个位置采样: {pos_list}")
            print("注意：每个位置采完后需输入你实际移动的距离\n")
            for i, intended in enumerate(pos_list, 1):
                print(f"\n--- 位置 {i}/{len(pos_list)}: 计划移动 {intended:+.1f} mm ---")
                input("把杆移到该位置，按回车采样…")
                z, r, mdr, st = sample(pipe, al, tf, mT, mR, args.frames)
                dz = z - z0
                dr = r - r0
                print(f"  目标读数变化 = {dz:+.3f} mm")
                print(f"  参考面变化   = {dr:+.3f} mm")
                print(f"  本次 MDR     = {mdr:.3f} mm")
                try:
                    s = input(f"  实际移动量（回车=按计划 {intended}）: ").strip()
                    truth = float(s) if s else intended
                except ValueError:
                    truth = intended
                records.append(dict(truth=truth, measured=dz,
                                    measured_net=dz - dr,
                                    ref_drift=dr, mdr=mdr, z=z, r=r))
                print(f"  → 记录: 真实 {truth:+.1f}  测得 {dz:+.3f}  "
                      f"误差 {dz-truth:+.3f} mm")
        else:
            print("需要 --interactive 或 --positions")
            return 1

    finally:
        try:
            pipe.stop()
        except Exception:
            pass

    if len(records) < 2:
        print("\n数据不足（<2 个点），无法拟合。")
        return 2

    # ── 分析 ──
    truth = np.array([r["truth"] for r in records])
    meas = np.array([r["measured"] for r in records])
    meas_net = np.array([r["measured_net"] for r in records])
    mdr = np.array([r["mdr"] for r in records])

    print("\n" + "=" * 92)
    print("标定结果")
    print("=" * 92)
    print(f"{'真实(mm)':>10}{'测得(mm)':>11}{'误差(mm)':>11}"
          f"{'扣共模(mm)':>12}{'本次MDR':>10}")
    print("-" * 92)
    for r in records:
        print(f"{r['truth']:>10.2f}{r['measured']:>11.3f}"
              f"{r['measured']-r['truth']:>11.3f}"
              f"{r['measured_net']:>12.3f}{r['mdr']:>10.3f}")

    # 线性拟合
    def fit(x, y, tag):
        if len(x) < 2:
            return
        A = np.vstack([x, np.ones_like(x)]).T
        coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        a, b = coef
        pred = a * x + b
        resid = y - pred
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else float("nan")
        print(f"\n{tag}:  测得 = {a:.4f} × 真实 + {b:+.3f}")
        print(f"  尺度系数 a = {a:.4f}   (理想 1.0000，偏差 {(a-1)*100:+.2f}%)")
        print(f"  零点偏移 b = {b:+.3f} mm")
        print(f"  RMSE       = {rmse:.3f} mm")
        print(f"  R²         = {r2:.5f}")
        return dict(a=float(a), b=float(b), rmse=rmse, r2=float(r2))

    f_raw = fit(truth, meas, "原始读数")
    f_net = fit(truth, meas_net, "扣除参考面共模")

    # ── 2mm 判据评估 ──
    print("\n" + "=" * 92)
    print("2mm 判据评估")
    print("=" * 92)
    err = np.abs(meas - truth)
    err_net = np.abs(meas_net - truth)
    print(f"  原始读数误差:  均值 {err.mean():.3f}  最大 {err.max():.3f} mm")
    print(f"  扣共模后误差:  均值 {err_net.mean():.3f}  最大 {err_net.max():.3f} mm")
    print(f"  单点 MDR 均值: {mdr.mean():.3f} mm")
    print()
    worst = max(err_net.max(), err.max())
    if worst <= 2.0:
        print(f"  ✅ 最大误差 {worst:.3f}mm ≤ 2mm —— 2mm 判据成立")
    else:
        print(f"  ❌ 最大误差 {worst:.3f}mm > 2mm —— 2mm 判据不成立")
        print(f"     需检查：尺度误差是否已修正？目标移动量是否准确？")

    # 判定「能可靠区分的最小位移」
    small = [(t, m, mn) for t, m, mn in zip(truth, meas, meas_net) if abs(t) <= 5]
    if small:
        print(f"\n  小位移样本（|真实|≤5mm，共 {len(small)} 个）:")
        for t, m, mn in sorted(small, key=lambda x: abs(x[0])):
            ok = abs(mn - t) <= 2.0
            print(f"    真实 {t:+5.1f} → 测得 {m:+7.3f} (扣共模 {mn:+7.3f})  "
                  f"误差 {mn-t:+.2f}  {'✅' if ok else '❌'}")

    # 保存
    out = dict(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        target_roi=list(troi), ref_roi=list(rroi),
        physical_mm=args.physical, z0=z0, r0=r0, k0=float(k0),
        static_mdr=float(mdr0),
        records=[{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                  for k, v in r.items()} for r in records],
        fit_raw=f_raw, fit_net=f_net,
        max_err=float(worst),
        verdict_2mm=bool(worst <= 2.0),
    )
    with open(args.save, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 标定结果已保存: {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
