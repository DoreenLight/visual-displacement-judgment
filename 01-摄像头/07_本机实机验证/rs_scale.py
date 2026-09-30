#!/usr/bin/env python3
"""尺度标定：验证「真实移动多少 mm，读数就变化多少 mm」。

这是整个项目的核心问题
----------------------
分辨力（MDR=0.10mm）只说明「读数很稳」。
但真正要回答的是：**真实移动 4mm 时，读数是否也变化 4mm？**

若存在尺度误差（如真实 4mm 被读成 3.6mm），则：
  - 测 2mm 会得到 1.8mm（误差 0.2mm）
  - 测 20mm 会得到 18mm（误差 2mm —— 已经超出 2mm 判据！）

标定方法
--------
1. 用钢尺把相机移到若干「已知物理距离」的位置
   （如 200 / 205 / 210 / 220 / 227 mm —— 从同一基准点量起）
2. 每个位置记录：物理距离 + 深度读数
3. 拟合：读数 = a × 物理 + b
   - a = 尺度系数（理想 1.0）
   - b = 系统偏移（对相对位移无影响）

为什么 b 不影响相对位移
----------------------
位移 = 读数₂ - 读数₁ = a×(物理₂ - 物理₁)
b 被消掉了。所以**只有 a 决定位移准确度**。

用法
----
# 记录一个点
sudo -E .../rs_scale.py add --physical 200.0

# 查看已记录的点
sudo -E .../rs_scale.py list

# 拟合分析
sudo -E .../rs_scale.py report

# 清空重来
sudo -E .../rs_scale.py reset
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

DATA_FILE = "scale_calib.json"
ROI_FILE = "roi.json"


def load_roi():
    if os.path.exists(ROI_FILE):
        with open(ROI_FILE, encoding="utf-8") as f:
            d = json.load(f)
        return tuple(d["roi"])
    return (307, 94, 55, 253)


def start():
    """创建 pipeline。必须同时开 depth+color，否则 align 静默失效。"""
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


def make_grab(pipe, al, tf):
    def grab():
        fs = al.process(pipe.wait_for_frames(timeout_ms=10000))
        return np.asanyarray(tf.process(fs.get_depth_frame()).get_data())
    return grab


def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, encoding="utf-8") as f:
            return json.load(f)
    return dict(points=[], created=time.strftime("%Y-%m-%dT%H:%M:%S"))


def save_data(d):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


def cmd_add(args):
    roi = load_roi()
    pipe, al, tf = start()
    grab = make_grab(pipe, al, tf)
    try:
        for _ in range(60):
            grab()
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

    # 可信度门控
    problems = []
    if not ok:
        problems.append(msg)
    if npx < 1000:
        problems.append(f"参与像素仅 {npx:.0f}（<1000）")
    if a["mdr"] > 1.0:
        problems.append(f"MDR={a['mdr']:.3f}mm 过大（>1mm）")

    print("=" * 76)
    print("尺度标定点")
    print("=" * 76)
    print(f"  ROI          = {roi}")
    print(f"  物理距离     = {args.physical:.2f} mm")
    print(f"  深度读数 z   = {z:.4f} mm")
    print(f"  差值 (z-物理)= {z-args.physical:+.3f} mm")
    print(f"  本次 MDR     = {a['mdr']:.4f} mm")
    print(f"  σ            = {a['scatter']:.4f} mm")
    print(f"  漂移         = {a['drift']:.4f} mm")
    print(f"  参与像素     = {npx:.0f}")
    print(f"  窗口空间σ    = {np.mean(sps):.2f} mm")

    if problems:
        print(f"\n❌ 数据不可信，拒绝记录：")
        for p in problems:
            print(f"   - {p}")
        return 1

    print(f"  ✅ 校验通过")

    data = load_data()
    data["points"].append(dict(
        physical=float(args.physical), z=float(z),
        mdr=float(a["mdr"]), sigma=float(a["scatter"]),
        drift=float(a["drift"]), npx=npx,
        ts=time.strftime("%H:%M:%S"), note=args.note or ""))
    save_data(data)

    print(f"\n✅ 已记录（累计 {len(data['points'])} 点）")
    _print_list(data)
    if len(data["points"]) >= 3:
        print(f"\n已有 ≥3 点，可运行 report 拟合")
    return 0


def _print_list(data):
    if not data["points"]:
        print("  (无记录)")
        return
    print(f"\n  {'#':>3}{'物理(mm)':>11}{'读数(mm)':>11}{'差(mm)':>10}"
          f"{'MDR':>9}{'像素':>7}")
    print("  " + "-" * 54)
    for i, p in enumerate(data["points"], 1):
        print(f"  {i:>3}{p['physical']:>11.2f}{p['z']:>11.3f}"
              f"{p['z']-p['physical']:>10.3f}{p['mdr']:>9.4f}{p['npx']:>7.0f}")


def cmd_list(args):
    data = load_data()
    print("=" * 76)
    print(f"尺度标定记录（{DATA_FILE}）")
    print("=" * 76)
    _print_list(data)
    return 0


def cmd_reset(args):
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
        print(f"✅ 已清空 {DATA_FILE}")
    else:
        print("(无文件)")
    return 0


def cmd_report(args):
    data = load_data()
    pts = data["points"]
    if len(pts) < 2:
        print(f"❌ 记录点不足（{len(pts)} 个，需 ≥2）")
        return 1

    phys = np.array([p["physical"] for p in pts])
    z = np.array([p["z"] for p in pts])

    print("=" * 80)
    print("尺度标定报告")
    print("=" * 80)
    _print_list(data)

    # 线性拟合 z = a*phys + b
    A = np.vstack([phys, np.ones_like(phys)]).T
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    a, b = coef
    pred = a * phys + b
    resid = z - pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss = np.sum((z - z.mean()) ** 2)
    r2 = float(1 - np.sum(resid ** 2) / ss) if ss > 0 else float("nan")

    print("\n" + "=" * 80)
    print("拟合：深度读数 = a × 物理距离 + b")
    print("=" * 80)
    print(f"  尺度系数 a       = {a:.6f}")
    print(f"  尺度误差         = {(a-1)*100:+.3f}%")
    print(f"  系统偏移 b       = {b:+.4f} mm")
    print(f"  RMSE             = {rmse:.4f} mm")
    print(f"  R²               = {r2:.8f}")

    print(f"\n  【物理含义】")
    print(f"  b = {b:+.2f}mm 是「相机前壳/镜头到深度模块」的内部偏移 +")
    print(f"     量测基准点差异。它【不影响】相对位移测量。")
    if abs(a - 1) < 0.005:
        print(f"  a ≈ 1（偏差 {abs(a-1)*100:.2f}%），尺度准确 —— "
              f"读数变化量 = 真实位移量")
    else:
        print(f"  ⚠️ a = {a:.4f}（偏差 {(a-1)*100:+.2f}%），存在尺度误差。")
        print(f"     真实移动 Δ 时，读数变化 = {a:.4f} × Δ")
        print(f"     若用读数直接当位移，会有 {(a-1)*100:+.2f}% 的比例误差：")
        print(f"       {'真实位移':>10}{'读数变化':>12}{'误差':>10}")
        for dd in (2, 5, 10, 20, 50):
            read_change = a * dd
            print(f"       {dd:>9.1f}mm{read_change:>11.3f}mm{read_change-dd:>+9.3f}mm")

    # 对 2mm 判据的影响
    print("\n" + "=" * 80)
    print("对 2mm 判据的影响")
    print("=" * 80)
    # 2mm 位移时的读数误差
    err_2mm = abs(a - 1) * 2.0
    print(f"  真实移动 2mm 时，读数误差 = |a-1| × 2 = {err_2mm:.4f} mm")
    if err_2mm <= 0.2:
        print(f"  ✅ 远小于 2mm 判据（占 {100*err_2mm/2:.1f}%）")
    elif err_2mm <= 2.0:
        print(f"  ⚠️ 占 2mm 判据的 {100*err_2mm/2:.1f}%")
    else:
        print(f"  ❌ 已超出 2mm 判据")

    # 相邻点差分（避免 b 的影响，直接看斜率）
    if len(pts) >= 3:
        print("\n" + "=" * 80)
        print("相邻点差分（最直接的尺度检验，自动消掉 b）")
        print("=" * 80)
        print(f"  {'对比':>12}{'物理变化':>11}{'读数变化':>11}{'比例':>9}")
        print("  " + "-" * 46)
        ratios = []
        for i in range(len(pts) - 1):
            dp = pts[i+1]["physical"] - pts[i]["physical"]
            dz = pts[i+1]["z"] - pts[i]["z"]
            if abs(dp) < 0.05:
                continue
            r = dz / dp
            ratios.append(r)
            print(f"  #{i+1}→#{i+2:<6}{dp:>+11.2f}{dz:>+11.3f}{r:>9.4f}")
        if ratios:
            rr = np.array(ratios)
            print(f"\n  比例均值 = {rr.mean():.4f}   σ = {rr.std(ddof=1) if len(rr)>1 else 0:.4f}")
            print(f"  → 这直接验证了「读数变化 / 真实变化」的比值")

    data["fit"] = dict(a=float(a), b=float(b), rmse=rmse, r2=r2)
    save_data(data)
    print(f"\n✅ 结果已写入 {DATA_FILE}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    a1 = sub.add_parser("add", help="记录一个标定点")
    a1.add_argument("--physical", type=float, required=True,
                    help="物理距离 mm（钢尺读数，从同一基准量起）")
    a1.add_argument("--frames", type=int, default=90)
    a1.add_argument("--note", default=None)
    a1.set_defaults(func=cmd_add)

    a2 = sub.add_parser("list", help="列出记录")
    a2.set_defaults(func=cmd_list)

    a3 = sub.add_parser("report", help="拟合分析")
    a3.set_defaults(func=cmd_report)

    a4 = sub.add_parser("reset", help="清空记录")
    a4.set_defaults(func=cmd_reset)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
