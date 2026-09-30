#!/usr/bin/env python3
"""位移测量：同时测「目标」与「固定参考面」，用于分离目标运动与相机漂移。

为什么要双目标
--------------
单看目标的深度变化，无法区分两种情况：
  (a) 目标真的动了
  (b) 相机/支架动了（或被碰了）

加测一个画面内的**固定参考面**即可判别：
  目标变 + 参考不变 → 目标真实位移 ✅
  目标变 + 参考同向变 → 相机在动，应作废或补偿
  都不变         → 未记录到运动

用法
----
# 先标定（目标静置，确定参考面与零点）
sudo -E .../rs_displace.py --calibrate --save disp_cal.json

# 然后移动目标，再测
sudo -E .../rs_displace.py --load disp_cal.json

# 连续监测模式（实时显示目标位移与参考漂移）
sudo -E .../rs_displace.py --load disp_cal.json --monitor
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

try:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def start(alpha=0.4):
    """创建 pipeline。filter 实例只建一次并复用（每帧新建会耗尽资源）。"""
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    prof = pipe.start(cfg)
    al = rs.align(rs.stream.color)
    tf = None
    if alpha > 0:
        tf = rs.temporal_filter()
        tf.set_option(rs.option.filter_smooth_alpha, alpha)
        tf.set_option(rs.option.filter_smooth_delta, 20)
    return pipe, al, tf


def grab(pipe, al, tf):
    fs = al.process(pipe.wait_for_frames(timeout_ms=10000))
    df = fs.get_depth_frame()
    if tf is not None:
        df = tf.process(df)
    return (np.asanyarray(df.get_data()),
            np.asanyarray(fs.get_color_frame().get_data()))


def list_blocks(med, min_px=200):
    """列出所有深度一致块（用于选参考面）。"""
    if not HAS_SCIPY:
        return []
    valid = (med > 0) & (med < 65535)
    n = int(valid.sum())
    if n < min_px:
        return []
    h, w = med.shape
    di = med.astype(np.int32)
    idx = np.full((h, w), -1, np.int64)
    idx[valid] = np.arange(n)
    mh = valid[:, :-1] & valid[:, 1:] & (np.abs(di[:, :-1] - di[:, 1:]) < 30)
    mv = valid[:-1, :] & valid[1:, :] & (np.abs(di[:-1, :] - di[1:, :]) < 30)
    rh, ch = np.where(mh)
    rv, cv = np.where(mv)
    if rh.size + rv.size == 0:
        return []
    rows = np.concatenate([idx[rh, ch], idx[rv, cv]])
    cols = np.concatenate([idx[rh, ch + 1], idx[rv + 1, cv]])
    g = coo_matrix((np.ones(rows.size, np.int8), (rows, cols)), shape=(n, n))
    ncomp, labels = connected_components(g, directed=False)
    cnt = np.bincount(labels, minlength=ncomp)
    vy, vx = np.where(valid)
    out = []
    for k in range(ncomp):
        if cnt[k] < min_px:
            continue
        sel = labels == k
        ys, xs = vy[sel], vx[sel]
        bw = int(xs.max() - xs.min() + 1)
        bh = int(ys.max() - ys.min() + 1)
        # 取块的中心区域作为可用 ROI（只取内部 60%，避开边缘闪烁）
        cx0 = int(xs.min() + bw * 0.2)
        cy0 = int(ys.min() + bh * 0.2)
        out.append(dict(
            roi=[cx0, cy0, max(8, int(bw * 0.6)), max(8, int(bh * 0.6))],
            z=float(np.median(med[ys, xs])),
            npx=int(len(ys)), bw=bw, bh=bh,
            sigma=float(np.std(med[ys, xs]))))
    out.sort(key=lambda d: -d["npx"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true", help="标定模式")
    ap.add_argument("--save", default=None)
    ap.add_argument("--load", default=None)
    ap.add_argument("--target-roi", default=None, help="目标 x,y,w,h")
    ap.add_argument("--ref-roi", default=None, help="参考面 x,y,w,h")
    ap.add_argument("--auto-ref", action="store_true",
                    help="自动选参考面（画面中最大的稳定块，且与目标距离差 >200mm）")
    ap.add_argument("--physical", type=float, default=None,
                    help="目标到相机的物理距离 mm（用于尺度校准）")
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--cal-frames", type=int, default=90)
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--monitor", action="store_true")
    args = ap.parse_args()

    cfg = {}
    if args.load:
        with open(args.load, encoding="utf-8") as f:
            cfg = json.load(f)
        print(f"已载入标定: {args.load}")
        print(f"  目标 ROI {cfg['target_roi']}  z0={cfg['target_z0']:.2f}mm")
        print(f"  参考 ROI {cfg['ref_roi']}  z0={cfg['ref_z0']:.2f}mm")
        if cfg.get("scale_k"):
            print(f"  尺度系数 k={cfg['scale_k']:.5f}")

    pipe, al, tf = start(args.alpha)
    try:
        print("\n预热…")
        for _ in range(45):
            pipe.wait_for_frames(timeout_ms=10000)

        # ── 确定目标 ROI ──
        if args.target_roi:
            troi = tuple(int(v) for v in args.target_roi.split(","))
        elif cfg.get("target_roi"):
            troi = tuple(cfg["target_roi"])
        else:
            st = [grab(pipe, al, tf)[0].astype(np.float32) for _ in range(20)]
            med = np.median(np.stack(st), axis=0)
            tgt = M.find_target_block(med, min_px=300, rng=(120, 400))
            if tgt is None:
                print("❌ 未定位到目标，请用 --target-roi 指定")
                return 1
            troi = tgt["roi"]
            print(f"自动定位目标: roi={troi}  z={tgt['z']:.0f}mm  "
                  f"{tgt['npx']}px  σ={tgt['sigma']:.1f}mm")

        # ── 确定参考面 ROI ──
        if args.ref_roi:
            rroi = tuple(int(v) for v in args.ref_roi.split(","))
        elif cfg.get("ref_roi"):
            rroi = tuple(cfg["ref_roi"])
        elif args.auto_ref or args.calibrate:
            st = [grab(pipe, al, tf)[0].astype(np.float32) for _ in range(20)]
            med = np.median(np.stack(st), axis=0)
            blocks = list_blocks(med)
            tz = None
            if len(st) and troi:
                sub = med[troi[1]:troi[1] + troi[3], troi[0]:troi[0] + troi[2]]
                vv = sub[(sub > 0) & (sub < 65535)]
                tz = float(np.median(vv)) if vv.size else None
            # 参考面：远离目标距离、面积大、σ 小
            cand = []
            for b in blocks:
                if tz is not None and abs(b["z"] - tz) < 200:
                    continue
                if b["npx"] < 2000 or b["sigma"] > 20:
                    continue
                # 不与目标重叠
                if not (b["roi"][0] + b["roi"][2] < troi[0] or
                        b["roi"][0] > troi[0] + troi[2] or
                        b["roi"][1] + b["roi"][3] < troi[1] or
                        b["roi"][1] > troi[1] + troi[3]):
                    continue
                cand.append(b)
            cand.sort(key=lambda b: -b["npx"])
            print(f"\n候选参考面（共 {len(cand)} 个，按面积排序）:")
            for b in cand[:6]:
                print(f"  roi={b['roi']}  z={b['z']:.0f}mm  {b['npx']}px  "
                      f"σ={b['sigma']:.1f}")
            if not cand:
                print("❌ 没找到合适参考面（需：与目标距离差>200mm、"
                      ">2000px、σ<20mm、不与目标重叠）")
                print("   请用 --ref-roi 手动指定")
                return 2
            rroi = tuple(cand[0]["roi"])
            print(f"自动选参考面: {rroi}  z={cand[0]['z']:.0f}mm")
        else:
            print("❌ 需要参考面：用 --auto-ref 或 --ref-roi")
            return 2

        mT = M.Measurement(roi=troi, trim_mm=20.0)
        mR = M.Measurement(roi=rroi, trim_mm=20.0)

        # ── 标定 ──
        if args.calibrate or not cfg:
            print(f"\n采标定基线 {args.cal_frames} 帧（目标与参考面都应静止）…")
            sT, sR = [], []
            for _ in range(args.cal_frames):
                d, _ = grab(pipe, al, tf)
                sT.append(mT.read(d))
                sR.append(mR.read(d))
            sT, sR = np.array(sT), np.array(sR)
            stT = M.series_mdr(sT)
            stR = M.series_mdr(sR)

            print("\n" + "=" * 92)
            print("标定结果")
            print("=" * 92)
            print(f"{'':14}{'z0(mm)':>10}{'MDR(3σ)':>11}{'σ散布':>9}"
                  f"{'漂移':>9}")
            print(f"{'目标':14}{stT['z_ref']:>10.2f}{stT['mdr']:>11.3f}"
                  f"{stT['scatter']:>9.3f}{stT['drift']:>9.3f}")
            print(f"{'参考面':14}{stR['z_ref']:>10.2f}{stR['mdr']:>11.3f}"
                  f"{stR['scatter']:>9.3f}{stR['drift']:>9.3f}")

            k = None
            if args.physical:
                k = args.physical / stT["z_ref"]
                print(f"\n尺度系数 k = {args.physical}/{stT['z_ref']:.2f} = {k:.5f}")

            # 相关性：目标与参考是否同步（同步=相机在动）
            L = min(len(sT), len(sR))
            c = float(np.corrcoef(sT[:L], sR[:L])[0, 1]) if L > 5 else np.nan
            print(f"\n目标↔参考 相关系数 = {c:+.3f}")
            if abs(c) > 0.5:
                print("  ⚠️ 高度相关：相机/支架在动，静态精度受影响")
            else:
                print("  ✅ 低相关：两者独立，可用作相互校验")

            cfg = dict(target_roi=list(troi), ref_roi=list(rroi),
                       target_z0=stT["z_ref"], ref_z0=stR["z_ref"],
                       target_mdr=stT["mdr"], ref_mdr=stR["mdr"],
                       corr=c, scale_k=k,
                       timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))

            if args.save:
                with open(args.save, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, ensure_ascii=False, indent=2)
                print(f"\n✅ 标定已保存: {args.save}")

        # ── 测量 ──
        tz0, rz0 = cfg["target_z0"], cfg["ref_z0"]
        k = cfg.get("scale_k") or 1.0

        if args.monitor:
            print("\n" + "=" * 92)
            print("实时监测（Ctrl-C 退出）")
            print("=" * 92)
            print(f"{'时间s':>7}{'目标z':>10}{'目标Δ':>10}"
                  f"{'参考z':>10}{'参考Δ':>10}{'判定':>16}")
            t0 = time.time()
            try:
                while True:
                    d, _ = grab(pipe, al, tf)
                    tz, rz = mT.read(d), mR.read(d)
                    dt_ = (tz - tz0) * k
                    dr = (rz - rz0) * k
                    if abs(dt_) < 3 * cfg["target_mdr"] and abs(dr) < 3 * cfg["ref_mdr"]:
                        v = "静止"
                    elif abs(dr) > 3 * cfg["ref_mdr"] and abs(dt_ - dr) < 3 * cfg["target_mdr"]:
                        v = "⚠️相机在动"
                    else:
                        v = "✓ 目标位移"
                    print(f"{time.time()-t0:>7.1f}{tz:>10.2f}{dt_:>10.3f}"
                          f"{rz:>10.2f}{dr:>10.3f}{v:>16}")
            except KeyboardInterrupt:
                print("\n结束。")
        else:
            print(f"\n采 {args.frames} 帧…")
            sT, sR = [], []
            for _ in range(args.frames):
                d, _ = grab(pipe, al, tf)
                sT.append(mT.read(d))
                sR.append(mR.read(d))
            sT, sR = np.array(sT), np.array(sR)
            dt_ = (np.mean(sT) - tz0) * k
            dr = (np.mean(sR) - rz0) * k
            stT = M.series_mdr(sT)

            print("\n" + "=" * 92)
            print("位移测量结果")
            print("=" * 92)
            print(f"  目标 z         = {np.mean(sT):.2f} mm   (基线 {tz0:.2f})")
            print(f"  目标位移 Δ     = {dt_:+.3f} mm")
            print(f"  参考面 z       = {np.mean(sR):.2f} mm   (基线 {rz0:.2f})")
            print(f"  参考面漂移 Δ   = {dr:+.3f} mm")
            print(f"  本次测量 MDR   = {stT['mdr']:.3f} mm")
            print(f"  尺度系数 k     = {k:.5f}")
            print()
            thrT = 3 * cfg["target_mdr"]
            thrR = 3 * cfg["ref_mdr"]
            print("判定:")
            if abs(dr) > thrR and abs(dt_ - dr) < thrT:
                print(f"  ⚠️ 相机/支架在动（参考面漂移 {dr:+.2f}mm > {thrR:.2f}mm）")
                print(f"     扣除共模后目标净位移 = {dt_ - dr:+.3f} mm")
            elif abs(dt_) < thrT:
                print(f"  · 未检测到显著位移（|Δ|={abs(dt_):.3f} < {thrT:.3f}mm）")
            else:
                print(f"  ✅ 目标位移 = {dt_:+.3f} mm "
                      f"(参考面稳定 {dr:+.3f}mm)")
                if abs(dt_) < 2.0:
                    print(f"     ← 该位移 {abs(dt_):.2f}mm 接近 2mm 判据边界，")
                    print(f"       需多次重复才能可靠判定")
    finally:
        try:
            pipe.stop()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
