#!/usr/bin/env python3
"""步进式位移标定 —— 每步一条命令，无需终端交互。

为什么这样设计
--------------
rs_calib_small.py 是交互式的（input()），在非交互环境下无法使用。
本脚本把每个步骤拆成独立命令，由操作者按需调用：

  1. baseline  : 建立零点，保存到 JSON
  2. measure   : 采一次数据，与零点比对，输出位移
  3. report    : 汇总所有记录，拟合 + 判定

用法
----
# 第1步 建立零点
sudo -E .../rs_calib_step.py baseline

# 第2步 移动杆到目标位置，然后
sudo -E .../rs_calib_step.py measure --truth 2.0

# 重复第2步若干次...
sudo -E .../rs_calib_step.py measure --truth 3.0

# 最后汇总
sudo -E .../rs_calib_step.py report
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

TARGET_ROI = (302, 96, 52, 251)
REF_ROI = (541, 223, 30, 57)      # 桌面（878mm），非黑板反光区
PHYSICAL_MM = 227.0
STATE_FILE = "calib_state.json"


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


def make_grab(pipe, al, tf):
    def grab():
        fs = al.process(pipe.wait_for_frames(timeout_ms=10000))
        return np.asanyarray(tf.process(fs.get_depth_frame()).get_data())
    return grab


def collect(grab, mT, mR, n, warmup=40):
    for _ in range(warmup):
        grab()
    sT, sR, cnt = [], [], []
    for _ in range(n):
        d = grab()
        z, np_, sp = mT.read_full(d)
        sT.append(z)
        cnt.append(np_)
        if mR:
            sR.append(mR.read(d))
    return (np.array(sT),
            np.array(sR) if sR else None,
            np.array(cnt))


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return None


def save_state(st):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)


def cmd_baseline(args):
    pipe, al, tf = start()
    grab = make_grab(pipe, al, tf)
    try:
        mT = M.Measurement(roi=TARGET_ROI, trim_mm=20.0)
        mR = M.Measurement(roi=REF_ROI, trim_mm=20.0) if not args.no_ref else None
        print(f"建立零点（采 {args.frames} 帧）…")
        sT, sR, cnt = collect(grab, mT, mR, args.frames)
    finally:
        try:
            pipe.stop()
        except Exception:
            pass

    a = M.series_mdr(sT)
    z0 = a["z_ref"]
    r0 = float(np.mean(sR)) if sR is not None else 0.0
    st = dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"),
              target_roi=list(TARGET_ROI), ref_roi=list(REF_ROI),
              physical_mm=PHYSICAL_MM,
              z0=float(z0), r0=float(r0),
              static_mdr=float(a["mdr"]), static_drift=float(a["drift"]),
              static_scatter=float(a["scatter"]),
              npx=float(cnt.mean()),
              records=[])
    save_state(st)

    k = PHYSICAL_MM / z0
    print(f"\n{'='*70}")
    print("零点已建立")
    print(f"{'='*70}")
    print(f"  z0        = {z0:.3f} mm")
    print(f"  r0        = {r0:.3f} mm")
    print(f"  静置 MDR  = {a['mdr']:.4f} mm")
    print(f"  静置 σ    = {a['scatter']:.4f} mm")
    print(f"  漂移      = {a['drift']:.4f} mm")
    print(f"  参与像素  = {cnt.mean():.0f}")
    print(f"  尺度系数  = {k:.5f}")
    print(f"\n✅ 已保存 {STATE_FILE}")
    print(f"\n下一步：把杆移动到目标位移处，然后运行：")
    print(f"  sudo -E .../rs_calib_step.py measure --truth 2.0")
    return 0


def cmd_measure(args):
    st = load_state()
    if st is None:
        print("❌ 未找到 state 文件，请先运行 baseline")
        return 1

    pipe, al, tf = start()
    grab = make_grab(pipe, al, tf)
    try:
        mT = M.Measurement(roi=tuple(st["target_roi"]), trim_mm=20.0)
        mR = (M.Measurement(roi=tuple(st["ref_roi"]), trim_mm=20.0)
              if st.get("ref_roi") and not args.no_ref else None)
        print(f"采 {args.frames} 帧…")
        sT, sR, cnt = collect(grab, mT, mR, args.frames)
    finally:
        try:
            pipe.stop()
        except Exception:
            pass

    a = M.series_mdr(sT)
    z = a["z_ref"]
    dz = z - st["z0"]
    dr = (float(np.mean(sR)) - st["r0"]) if sR is not None else 0.0
    net = dz - dr

    print(f"\n{'='*70}")
    print(f"第 {len(st['records'])+1} 次测量")
    print(f"{'='*70}")
    print(f"  z         = {z:.3f} mm   (零点 {st['z0']:.3f})")
    print(f"  位移 dz   = {dz:+.3f} mm")
    if sR is not None:
        print(f"  参考漂移  = {dr:+.3f} mm")
        print(f"  扣共模    = {net:+.3f} mm")
    print(f"  本次 MDR  = {a['mdr']:.4f} mm")
    print(f"  像素      = {cnt.mean():.0f}")

    if args.truth is not None:
        err = dz - args.truth
        print(f"\n  真实位移  = {args.truth:+.3f} mm")
        print(f"  误差      = {err:+.3f} mm")
        ok = abs(err) <= 2.0
        print(f"  {'✅ 误差 ≤2mm' if ok else '❌ 误差 >2mm'}")

    st["records"].append(dict(
        ts=time.strftime("%H:%M:%S"),
        truth=float(args.truth) if args.truth is not None else None,
        z=float(z), dz=float(dz), dr=float(dr), net=float(net),
        mdr=float(a["mdr"]), scatter=float(a["scatter"]),
        npx=float(cnt.mean()), note=args.note or "",
    ))
    save_state(st)
    print(f"\n✅ 已记录（累计 {len(st['records'])} 条）")
    print(f"\n继续移动杆并 measure，或运行 report 汇总")
    return 0


def cmd_report(args):
    st = load_state()
    if st is None:
        print("❌ 无 state 文件")
        return 1
    recs = [r for r in st["records"] if r.get("truth") is not None]
    if len(recs) < 2:
        print(f"❌ 有效记录不足（{len(recs)} 条，需 ≥2）")
        print("   请用 measure --truth <mm> 记录至少 2 个位移点")
        return 1

    truth = np.array([r["truth"] for r in recs])
    dz = np.array([r["dz"] for r in recs])
    net = np.array([r["net"] for r in recs])

    print("=" * 84)
    print("位移标定结果")
    print("=" * 84)
    print(f"{'真实(mm)':>10}{'测得(mm)':>11}{'误差(mm)':>11}"
          f"{'扣共模(mm)':>12}{'MDR':>9}{'像素':>7}")
    print("-" * 84)
    for r in recs:
        print(f"{r['truth']:>10.2f}{r['dz']:>11.3f}"
              f"{r['dz']-r['truth']:>11.3f}{r['net']:>12.3f}"
              f"{r['mdr']:>9.4f}{r['npx']:>7.0f}")

    def fit(x, y, tag):
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        a, b = coef
        resid = y - (a * x + b)
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        ss = np.sum((y - y.mean()) ** 2)
        r2 = float(1 - np.sum(resid ** 2) / ss) if ss > 0 else float("nan")
        print(f"\n【{tag}】测得 = {a:.5f} × 真实 + {b:+.4f}")
        print(f"  尺度系数 a = {a:.5f}   (理想 1.0，偏差 {(a-1)*100:+.2f}%)")
        print(f"  零点偏移 b = {b:+.4f} mm")
        print(f"  RMSE       = {rmse:.4f} mm")
        print(f"  R²         = {r2:.6f}")
        return dict(a=float(a), b=float(b), rmse=rmse, r2=r2)

    f1 = fit(truth, dz, "原始读数")
    f2 = fit(truth, net, "扣除参考面共模")

    err = np.abs(dz - truth)
    err_n = np.abs(net - truth)
    print("\n" + "=" * 84)
    print("2mm 判据评估")
    print("=" * 84)
    print(f"  原始误差  均值 {err.mean():.3f}  最大 {err.max():.3f} mm")
    print(f"  扣共模误差 均值 {err_n.mean():.3f}  最大 {err_n.max():.3f} mm")
    print(f"  静置 MDR  {st['static_mdr']:.4f} mm")
    print(f"  钢尺精度  ±0.5 mm（本标定的上限）")
    print()
    worst = max(err.max(), err_n.max())
    if worst <= 2.0:
        print(f"  ✅ 最大误差 {worst:.3f}mm ≤ 2mm —— 2mm 判据成立")
    else:
        print(f"  ❌ 最大误差 {worst:.3f}mm > 2mm")
    if worst <= 0.5:
        print(f"  ⚠️ 误差 {worst:.3f}mm 在钢尺精度内，无法区分来源")
    else:
        print(f"  ⚠️ 误差 {worst:.3f}mm > 钢尺精度 0.5mm，可信")

    st["fit_raw"] = f1
    st["fit_net"] = f2
    st["max_err"] = float(worst)
    st["verdict"] = bool(worst <= 2.0)
    save_state(st)
    print(f"\n✅ 结果已写入 {STATE_FILE}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("baseline", help="建立零点")
    b.add_argument("--frames", type=int, default=90)
    b.add_argument("--no-ref", action="store_true")
    b.set_defaults(func=cmd_baseline)

    m = sub.add_parser("measure", help="采一次测量")
    m.add_argument("--truth", type=float, default=None,
                   help="真实位移（钢尺读数）")
    m.add_argument("--frames", type=int, default=90)
    m.add_argument("--note", default=None)
    m.add_argument("--no-ref", action="store_true")
    m.set_defaults(func=cmd_measure)

    r = sub.add_parser("report", help="汇总拟合")
    r.set_defaults(func=cmd_report)
    r.add_argument("--x", type=int, default=0, help=argparse.SUPPRESS)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
