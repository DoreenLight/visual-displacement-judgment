#!/usr/bin/env python3
"""盲测记录器：只记读数与质量指标，真值事后补填。

为什么这样设计
--------------
若操作者每次都说真值，可能无意中影响读数判读。
盲测模式：先连续记录 N 个位置的读数（不含真值），
最后一次性补填真值，再拟合。这样避免引导。

用法
----
sudo -E .../rs_blind.py take              # 记录当前位置（自动编号）
sudo -E .../rs_blind.py list              # 查看已记录
sudo -E .../rs_blind.py fill --index 1 --physical 200.0   # 补填真值
sudo -E .../rs_blind.py report            # 拟合
sudo -E .../rs_blind.py reset             # 清空
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

DATA = "blind_calib.json"
ROI_FILE = "roi.json"


def load_roi():
    if os.path.exists(ROI_FILE):
        with open(ROI_FILE, encoding="utf-8") as f:
            return tuple(json.load(f)["roi"])
    return (307, 94, 55, 253)


def start():
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    prof = pipe.start(cfg)
    al = rs.align(rs.stream.color)
    tf = rs.temporal_filter()
    tf.set_option(rs.option.filter_smooth_alpha, 0.4)
    tf.set_option(rs.option.filter_smooth_delta, 20)
    return pipe, al, tf


def load():
    if os.path.exists(DATA):
        with open(DATA, encoding="utf-8") as f:
            return json.load(f)
    return dict(points=[], created=time.strftime("%Y-%m-%dT%H:%M:%S"))


def save(d):
    with open(DATA, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


def cmd_take(args):
    roi = load_roi()
    pipe, al, tf = start()
    try:
        for _ in range(60):
            pipe.wait_for_frames(timeout_ms=10000)

        def grab():
            fs = al.process(pipe.wait_for_frames(timeout_ms=10000))
            return np.asanyarray(tf.process(fs.get_depth_frame()).get_data())

        m = M.Measurement(roi=roi, trim_mm=20.0)
        ser, nus, sps = [], [], []
        for _ in range(args.frames):
            z, n, sp = m.read_full(grab())
            ser.append(z)
            nus.append(n)
            sps.append(sp)
        d_last = grab()
    finally:
        try:
            pipe.stop()
        except Exception:
            pass

    ser = np.array(ser)
    a = M.series_mdr(ser)
    z = a["z_ref"]
    npx = float(np.mean(nus))
    ok, msg = M.check_aligned(d_last, roi)

    problems = []
    if not ok:
        problems.append(msg)
    if npx < 1000:
        problems.append(f"参与像素仅 {npx:.0f}（<1000）")
    if a["mdr"] > 1.0:
        problems.append(f"MDR={a['mdr']:.3f}mm 过大")

    d = load()
    idx = len(d["points"]) + 1

    print("=" * 74)
    print(f"盲测记录  第 {idx} 次")
    print("=" * 74)
    print(f"  深度读数 z   = {z:.4f} mm")
    print(f"  MDR          = {a['mdr']:.4f} mm")
    print(f"  σ            = {a['scatter']:.4f} mm")
    print(f"  漂移         = {a['drift']:.4f} mm")
    print(f"  参与像素     = {npx:.0f}")
    print(f"  窗口空间σ    = {np.mean(sps):.2f} mm")
    print(f"  校验         = {'✅ ' + msg if ok else '⚠️ ' + msg}")

    if problems:
        print(f"\n❌ 数据不可信，拒绝记录：")
        for p in problems:
            print(f"   - {p}")
        return 1

    d["points"].append(dict(
        idx=idx, z=float(z), mdr=float(a["mdr"]),
        sigma=float(a["scatter"]), drift=float(a["drift"]),
        npx=npx, ts=time.strftime("%H:%M:%S"),
        physical=None, note=args.note or ""))
    save(d)
    print(f"\n✅ 已记录（第 {idx} 条，真值待补）")
    _list(d)
    return 0


def _list(d):
    if not d["points"]:
        print("  (无记录)")
        return
    print(f"\n  {'#':>3}{'读数(mm)':>12}{'真值(mm)':>12}{'相对首点(mm)':>14}"
          f"{'MDR':>9}")
    print("  " + "-" * 52)
    z0 = d["points"][0]["z"]
    for p in d["points"]:
        ph = f"{p['physical']:.2f}" if p.get("physical") is not None else "—"
        print(f"  {p['idx']:>3}{p['z']:>12.4f}{ph:>12}"
              f"{p['z']-z0:>+14.4f}{p['mdr']:>9.4f}")


def cmd_list(args):
    d = load()
    print("=" * 74)
    print(f"盲测记录（{DATA}）")
    print("=" * 74)
    _list(d)
    n_filled = sum(1 for p in d["points"] if p.get("physical") is not None)
    print(f"\n  已记录 {len(d['points'])} 点，其中 {n_filled} 点已补真值")
    return 0


def cmd_fill(args):
    d = load()
    if args.index < 1 or args.index > len(d["points"]):
        print(f"❌ index 越界（1..{len(d['points'])}）")
        return 1
    p = d["points"][args.index - 1]
    p["physical"] = float(args.physical)
    if args.note:
        p["note"] = args.note
    save(d)
    print(f"✅ 第 {args.index} 点真值已补：{args.physical:.2f} mm"
          f"  (读数 {p['z']:.4f})")
    _list(d)
    return 0


def cmd_fillall(args):
    """一次补填所有真值：--values 200,202,205,210,220"""
    d = load()
    vals = [float(v) for v in args.values.split(",")]
    if len(vals) != len(d["points"]):
        print(f"❌ 数量不匹配：给 {len(vals)} 个值，"
              f"但有 {len(d['points'])} 个点")
        return 1
    for p, v in zip(d["points"], vals):
        p["physical"] = v
    save(d)
    print(f"✅ 已补填 {len(vals)} 个真值")
    _list(d)
    return 0


def cmd_reset(args):
    if os.path.exists(DATA):
        os.remove(DATA)
        print(f"✅ 已清空 {DATA}")
    else:
        print("(无文件)")
    return 0


def cmd_report(args):
    d = load()
    pts = [p for p in d["points"] if p.get("physical") is not None]
    if len(pts) < 2:
        print(f"❌ 已补真值的点不足（{len(pts)}，需 ≥2）")
        _list(d)
        return 1

    phys = np.array([p["physical"] for p in pts])
    z = np.array([p["z"] for p in pts])

    print("=" * 82)
    print("盲测标定报告（尺度检验）")
    print("=" * 82)
    _list(d)

    A = np.vstack([phys, np.ones_like(phys)]).T
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    a, b = coef
    resid = z - (a * phys + b)
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss = np.sum((z - z.mean()) ** 2)
    r2 = float(1 - np.sum(resid ** 2) / ss) if ss > 0 else float("nan")

    print("\n" + "=" * 82)
    print(f"拟合：深度读数 = {a:.6f} × 物理距离 {b:+.4f}")
    print("=" * 82)
    print(f"  尺度系数 a = {a:.6f}   误差 {(a-1)*100:+.3f}%")
    print(f"  系统偏移 b = {b:+.4f} mm（不影响相对位移）")
    print(f"  RMSE       = {rmse:.4f} mm")
    print(f"  R²         = {r2:.8f}")

    # 相邻点差分（最直接的尺度检验）
    if len(pts) >= 2:
        print("\n" + "=" * 82)
        print("相邻点差分（消除 b，最直接验证「读数变化 vs 真实变化」）")
        print("=" * 82)
        print(f"  {'对比':>10}{'物理变化':>12}{'读数变化':>12}"
              f"{'比例':>10}{'误差':>11}")
        print("  " + "-" * 58)
        ratios = []
        for i in range(len(pts) - 1):
            dp = pts[i+1]["physical"] - pts[i]["physical"]
            dz = pts[i+1]["z"] - pts[i]["z"]
            if abs(dp) < 0.05:
                continue
            r = dz / dp
            ratios.append(r)
            print(f"  #{i+1}→#{i+2:<5}{dp:>+12.2f}{dz:>+12.4f}"
                  f"{r:>10.4f}{dz-dp:>+11.4f}")
        if ratios:
            rr = np.array(ratios)
            print(f"\n  比例均值 = {rr.mean():.5f}")
            if len(rr) > 1:
                print(f"  比例 σ   = {rr.std(ddof=1):.5f}")

    # 对 2mm 判据的影响
    print("\n" + "=" * 82)
    print("2mm 判据影响评估")
    print("=" * 82)
    print(f"  {'真实位移':>10}{'读数变化':>12}{'误差':>11}{'占2mm判据':>12}")
    print("  " + "-" * 48)
    for dd in (1, 2, 3, 5, 10, 20):
        rc = a * dd
        print(f"  {dd:>9.1f}mm{rc:>11.4f}mm{rc-dd:>+10.4f}mm"
              f"{100*abs(rc-dd)/2:>11.1f}%")
    err2 = abs(a - 1) * 2.0
    print()
    if err2 <= 0.2:
        print(f"  ✅ 2mm 位移时尺度误差 {err2:.4f}mm，占判据 {100*err2/2:.1f}% —— 可忽略")
    elif err2 <= 2.0:
        print(f"  ⚠️ 2mm 位移时尺度误差 {err2:.4f}mm，占判据 {100*err2/2:.1f}%")
    else:
        print(f"  ❌ 2mm 位移时尺度误差 {err2:.4f}mm —— 已超出判据")

    d["fit"] = dict(a=float(a), b=float(b), rmse=rmse, r2=r2)
    save(d)
    print(f"\n✅ 已写入 {DATA}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("take", help="记录当前位置")
    t.add_argument("--frames", type=int, default=90)
    t.add_argument("--note", default=None)
    t.set_defaults(func=cmd_take)

    L = sub.add_parser("list", help="列出记录")
    L.set_defaults(func=cmd_list)

    f = sub.add_parser("fill", help="补填某点真值")
    f.add_argument("--index", type=int, required=True)
    f.add_argument("--physical", type=float, required=True)
    f.add_argument("--note", default=None)
    f.set_defaults(func=cmd_fill)

    fa = sub.add_parser("fillall", help="一次补填全部真值")
    fa.add_argument("--values", required=True,
                    help="逗号分隔，按记录顺序")
    fa.set_defaults(func=cmd_fillall)

    r = sub.add_parser("report", help="拟合分析")
    r.set_defaults(func=cmd_report)

    rs = sub.add_parser("reset", help="清空")
    rs.set_defaults(func=cmd_reset)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
