#!/usr/bin/env python3
"""小位移标定：验证 0.5/1/2/3/5mm 能否被正确测量。

这是项目最关键的一步
--------------------
此前所有数据都是「重复性」（静置噪声底，MDR=0.148mm）。
本脚本测的是「准确度」：输入已知位移，看输出是否正确。

读数精度限制（必须知道）
------------------------
用钢尺目视读数，2mm 量级的读数误差约 ±0.5mm（25% 相对误差）。
这意味着：
  - 若测得误差 <0.5mm，无法判断是真误差还是尺子读数误差
  - 要突破这个限制，需要平移台或塞尺

本脚本的对策：每个位移点重复 5 次，用多次平均压低读数随机误差。
若 5 次的标准差很小，说明读数一致，可信度提高。

用法
----
sudo -E /opt/anaconda3/envs/realsense/bin/python3 rs_calib_small.py

流程
----
1. 建立零点（60 帧静置）
2. 脚本提示「请移动杆，使钢尺读数为 +2mm」
3. 你移动后按回车 → 脚本采 60 帧
4. 脚本记录：尺子读数 + 测量值
5. 全部完成后自动拟合、判定
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

# 黑背景后的确定配置
TARGET_ROI = (414, 105, 24, 70)     # 杆上的白纸段
REF_ROI = (493, 229, 51, 94)        # 桌面参考面
PHYSICAL_MM = 227.0                 # 由白纸 17mm 几何反推


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


def make_grabber(pipe, al, tf):
    def grab():
        fs = al.process(pipe.wait_for_frames(timeout_ms=10000))
        return np.asanyarray(tf.process(fs.get_depth_frame()).get_data())
    return grab


def sample(grab, mT, mR, n):
    """采 n 帧，返回 (目标序列, 参考序列, 逐帧像素数)。"""
    sT, sR, cnt = [], [], []
    for _ in range(n):
        d = grab()
        z, np_, sp = mT.read_full(d)
        sT.append(z)
        cnt.append(np_)
        if mR is not None:
            sR.append(mR.read(d))
    return np.array(sT), (np.array(sR) if sR else None), np.array(cnt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi", default=None)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--physical", type=float, default=PHYSICAL_MM)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--repeat", type=int, default=5,
                    help="每个位移点的重复次数（压低尺子读数误差）")
    ap.add_argument("--targets", default="2,3,5,8",
                    help="要标定的位移量（mm）；用钢尺时建议 ≥2mm（尺子误差±0.5mm）")
    ap.add_argument("--save", default="calib_small.json")
    ap.add_argument("--quick", action="store_true",
                    help="快速模式：每点只测 1 次")
    args = ap.parse_args()

    troi = tuple(int(v) for v in args.roi.split(",")) if args.roi else TARGET_ROI
    rroi = tuple(int(v) for v in args.ref.split(",")) if args.ref else REF_ROI
    targets = [float(v) for v in args.targets.split(",")]
    repeat = 1 if args.quick else args.repeat

    print("=" * 88)
    print("小位移标定 —— 验证 2mm 判据")
    print("=" * 88)
    print(f"目标 ROI   : {troi}")
    print(f"参考面 ROI : {rroi}")
    print(f"物理距离   : {args.physical} mm")
    print(f"位移点     : {targets} mm")
    print(f"每点重复   : {repeat} 次")
    print()
    print("⚠️ 使用钢尺读数的精度约 ±0.5mm。")
    print("   本脚本每点重复多次，用平均压低随机误差。")
    print()

    pipe, al, tf = start()
    grab = make_grabber(pipe, al, tf)
    try:
        print("预热…")
        for _ in range(60):
            grab()

        mT = M.Measurement(roi=troi, trim_mm=20.0)
        mR = M.Measurement(roi=rroi, trim_mm=20.0)

        # ── 零点 ──
        print(f"\n【第 1 步】建立零点")
        print(f"请确保杆静止在初始位置，按回车开始采 {args.frames} 帧…")
        input()
        sT, sR, cnt = sample(grab, mT, mR, args.frames)
        st0 = M.series_mdr(sT)
        z0 = st0["z_ref"]
        r0 = float(np.mean(sR)) if sR is not None else 0.0
        print(f"  目标 z0    = {z0:.3f} mm")
        print(f"  参考 r0    = {r0:.3f} mm")
        print(f"  静置 MDR   = {st0['mdr']:.4f} mm")
        print(f"  静置漂移   = {st0['drift']:.4f} mm")
        print(f"  参与像素   = {cnt.mean():.0f}")
        k0 = args.physical / z0
        print(f"  尺度系数 k = {k0:.5f}  ({(k0-1)*100:+.3f}%)")

        # ── 逐点标定 ──
        print("\n" + "=" * 88)
        print("【第 2 步】逐点标定")
        print("=" * 88)
        print("""
操作说明：
  脚本会提示目标位移量（如 +2mm）
  你把杆沿 Z 轴移动到该位置（用钢尺读数确认）
  按回车采样
  若中途想结束，输入 q + 回车
""")
        records = []
        for tgt in targets:
            print(f"\n{'─'*88}")
            print(f"目标位移 = {tgt:+.1f} mm  （{repeat} 次采样）")
            print(f"{'─'*88}")
            reps = []
            for r in range(repeat):
                cmd = input(f"  第 {r+1}/{repeat} 次："
                            f"把杆移到 {tgt:+.1f}mm 位置，按回车采样"
                            f"（q=结束）: ").strip()
                if cmd.lower() == "q":
                    print("  提前结束标定")
                    break
                sT, sR, cnt = sample(grab, mT, mR, args.frames)
                st = M.series_mdr(sT)
                z = st["z_ref"]
                dz = z - z0
                dr = (float(np.mean(sR)) - r0) if sR is not None else 0.0
                print(f"    测得 {dz:+.3f} mm   参考漂移 {dr:+.3f} mm   "
                      f"MDR {st['mdr']:.3f}   像素 {cnt.mean():.0f}")
                reps.append(dict(dz=float(dz), z=float(z), dr=float(dr),
                                 mdr=float(st["mdr"]),
                                 drift=float(st["drift"]),
                                 npx=float(cnt.mean()),
                                 drift_r=float(dr)))
            if not reps:
                continue
            dzs = np.array([r["dz"] for r in reps])
            nets = np.array([r["dz"] - r["dr"] for r in reps])
            # 询问实际移动量（若重复多次，只问一次）
            try:
                s = input(f"  实际移动量（钢尺读数，回车={tgt}）: ").strip()
                truth = float(s) if s else tgt
            except ValueError:
                truth = tgt
            records.append(dict(
                truth=truth, target=tgt,
                dz_mean=float(np.mean(dzs)), dz_std=float(np.std(dzs, ddof=1))
                if len(dzs) > 1 else 0.0,
                net_mean=float(np.mean(nets)),
                net_std=float(np.std(nets, ddof=1)) if len(nets) > 1 else 0.0,
                mdr=float(np.mean([r["mdr"] for r in reps])),
                npx=float(np.mean([r["npx"] for r in reps])),
                reps=reps))
            print(f"  → 记录: 真实 {truth:+.2f}  测得 {np.mean(dzs):+.3f}  "
                  f"误差 {np.mean(dzs)-truth:+.3f}  "
                  f"重复σ {records[-1]['dz_std']:.3f}")

    finally:
        try:
            pipe.stop()
        except Exception:
            pass

    if len(records) < 2:
        print("\n数据不足，无法拟合。")
        return 1

    # ── 分析 ──
    truth = np.array([r["truth"] for r in records])
    meas = np.array([r["dz_mean"] for r in records])
    nets = np.array([r["net_mean"] for r in records])

    print("\n" + "=" * 88)
    print("标定结果")
    print("=" * 88)
    print(f"{'真实(mm)':>10}{'测得(mm)':>11}{'误差(mm)':>11}"
          f"{'重复σ':>9}{'扣共模(mm)':>12}{'MDR':>9}{'像素':>7}")
    print("-" * 88)
    for r in records:
        print(f"{r['truth']:>10.2f}{r['dz_mean']:>11.3f}"
              f"{r['dz_mean']-r['truth']:>11.3f}{r['dz_std']:>9.3f}"
              f"{r['net_mean']:>12.3f}{r['mdr']:>9.3f}{r['npx']:>7.0f}")

    def fit(x, y, tag):
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        a, b = coef
        pred = a * x + b
        resid = y - pred
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        ss = np.sum((y - y.mean()) ** 2)
        r2 = float(1 - np.sum(resid ** 2) / ss) if ss > 0 else float("nan")
        print(f"\n【{tag}】测得 = {a:.5f} × 真实 + {b:+.4f}")
        print(f"  尺度系数 a  = {a:.5f}   （理想 1.0，偏差 {(a-1)*100:+.2f}%）")
        print(f"  零点偏移 b  = {b:+.4f} mm")
        print(f"  RMSE        = {rmse:.4f} mm")
        print(f"  R²          = {r2:.6f}")
        return dict(a=float(a), b=float(b), rmse=rmse, r2=r2)

    f1 = fit(truth, meas, "原始读数")
    f2 = fit(truth, nets, "扣除参考面共模")

    # ── 2mm 判据 ──
    print("\n" + "=" * 88)
    print("2mm 判据评估")
    print("=" * 88)
    err = np.abs(meas - truth)
    err_n = np.abs(nets - truth)
    print(f"  原始误差:    均值 {err.mean():.3f}  最大 {err.max():.3f} mm")
    print(f"  扣共模误差:  均值 {err_n.mean():.3f}  最大 {err_n.max():.3f} mm")
    print(f"  静置 MDR:    {st0['mdr']:.4f} mm")
    print(f"  尺子读数精度: ±0.5 mm （这是本标定的精度上限）")
    print()
    worst = max(err_n.max(), err.max())
    if worst <= 2.0:
        print(f"  ✅ 最大误差 {worst:.3f}mm ≤ 2mm —— 2mm 判据成立")
    else:
        print(f"  ❌ 最大误差 {worst:.3f}mm > 2mm")
    if worst <= 0.5:
        print(f"  ⚠️ 但误差 {worst:.3f}mm 已在钢尺读数精度（±0.5mm）以内")
        print(f"     → 无法区分「相机误差」与「尺子读数误差」")
        print(f"     → 要突破需用平移台或塞尺")

    # 小位移专项
    small = [r for r in records if abs(r["truth"]) <= 2.0]
    if small:
        print(f"\n  【1-2mm 专项】（老师要求的关键区间）")
        for r in sorted(small, key=lambda x: abs(x["truth"])):
            e = r["dz_mean"] - r["truth"]
            ok = abs(e) <= 2.0
            print(f"    真实 {r['truth']:+.2f}mm → 测得 {r['dz_mean']:+.3f}mm  "
                  f"误差 {e:+.3f}mm  重复σ {r['dz_std']:.3f}  "
                  f"{'✅' if ok else '❌'}")

    out = dict(
        ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
        target_roi=list(troi), ref_roi=list(rroi),
        physical_mm=args.physical, z0=float(z0), r0=float(r0),
        static_mdr=float(st0["mdr"]), static_drift=float(st0["drift"]),
        ruler_precision=0.5,
        records=records, fit_raw=f1, fit_net=f2,
        max_err=float(worst), verdict_2mm=bool(worst <= 2.0),
    )
    with open(args.save, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 已保存: {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
