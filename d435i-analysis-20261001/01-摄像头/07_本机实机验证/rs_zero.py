#!/usr/bin/env python3
"""20cm 基准调零 + 位移测量工具。

调零的两层含义
--------------
第 1 层 硬件基准 z_ref：
    用实测的深度值定义「物理零点」。同时用钢尺量出的物理距离 d_ref
    用于计算尺度系数 k = d_ref / z_ref。若 k 显著偏离 1，说明存在系统偏差。

第 2 层 软件零点 z_base：
    每段数据开始时，用静置基线（前 N 帧中位数）作零点。
    displacement[t] = (roi_median_z[t] - z_base) * k

本工具做什么
------------
1. 自动定位「白纸杆」（或在 ROI 内锁定深度一致块）
2. 锁定 ROI 的像素集合（避免逐帧漂移污染统计）
3. 做 60 帧静置，输出 z_ref / σ / MDR / 漂移
4. 建立零点，输出基准报告
5. 选做：连续监测模式，实时输出位移

用法
----
# 调零（静置 60 帧，输出基准报告）
sudo -E .../rs_zero.py --auto --physical 200

# 指定 ROI（基于彩色 640x480）
sudo -E .../rs_zero.py --roi 367,48,58,127 --physical 200

# 调零后进入监测模式（实时打印位移，Ctrl-C 退出）
sudo -E .../rs_zero.py --auto --physical 200 --monitor

# 保存基准到文件，供后续实验复用
sudo -E .../rs_zero.py --auto --physical 200 --save zero_20cm.json
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

try:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

W, H, FPS = 848, 480, 30
DEGLITCH_MM = 200      # 单帧偏离中值超过此值视为离群并剔除


def start(alpha=0.4):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, FPS)
    prof = pipe.start(cfg)
    tf = None
    if alpha > 0:
        tf = rs.temporal_filter()
        tf.set_option(rs.option.filter_smooth_alpha, alpha)
        tf.set_option(rs.option.filter_smooth_delta, 20)
    return pipe, tf, rs.align(rs.stream.color), prof


def grab(pipe, tf, al):
    fs = al.process(pipe.wait_for_frames(timeout_ms=10000))
    df = fs.get_depth_frame()
    if tf is not None:
        df = tf.process(df)
    return (np.asanyarray(df.get_data()),
            np.asanyarray(fs.get_color_frame().get_data()))


def find_target(pipe, tf, al, warmup=45):
    """自动定位最像「竖直白纸杆」的深度一致块。"""
    for _ in range(warmup):
        pipe.wait_for_frames(timeout_ms=10000)
    st = []
    for _ in range(15):
        d, _ = grab(pipe, tf, al)
        st.append(d.astype(np.float32))
    med = np.median(np.stack(st), axis=0)
    if not HAS_SCIPY:
        return None, med
    valid = (med > 0) & (med < 65535)
    n = int(valid.sum())
    if n < 200:
        return None, med
    h, w = med.shape
    di = med.astype(np.int32)
    idx = np.full((h, w), -1, np.int64)
    idx[valid] = np.arange(n)
    mh = valid[:, :-1] & valid[:, 1:] & (np.abs(di[:, :-1] - di[:, 1:]) < 30)
    mv = valid[:-1, :] & valid[1:, :] & (np.abs(di[:-1, :] - di[1:, :]) < 30)
    rh, ch = np.where(mh)
    rv, cv = np.where(mv)
    if rh.size + rv.size == 0:
        return None, med
    rows = np.concatenate([idx[rh, ch], idx[rv, cv]])
    cols = np.concatenate([idx[rh, ch + 1], idx[rv + 1, cv]])
    g = coo_matrix((np.ones(rows.size, np.int8), (rows, cols)), shape=(n, n))
    ncomp, labels = connected_components(g, directed=False)
    cnt = np.bincount(labels, minlength=ncomp)
    vy, vx = np.where(valid)

    best = None
    for k in range(ncomp):
        if cnt[k] < 200:
            continue
        sel = labels == k
        ys, xs = vy[sel], vx[sel]
        z = float(np.median(med[ys, xs]))
        if not (140 < z < 350):
            continue
        sig = float(np.std(med[ys, xs]))
        bw = int(xs.max() - xs.min() + 1)
        bh = int(ys.max() - ys.min() + 1)
        if sig > 20:
            continue
        # 打分：像素多、σ 小、形状偏高（杆特征）
        score = len(ys) - sig * 50 + (bh / max(1, bw)) * 100
        if best is None or score > best[0]:
            best = (score, k, z, len(ys), sig,
                    int(xs.min()), int(xs.max()),
                    int(ys.min()), int(ys.max()), bw, bh)
    if best is None:
        return None, med
    _, k, z, npx, sig, x0, x1, y0, y1, bw, bh = best
    return dict(label=k, z=z, npx=npx, sigma=sig,
                roi=[x0, y0, x1 - x0 + 1, y1 - y0 + 1]), med


def series_stats(series, valid_cnt, fps=FPS):
    """对时间序列计算统计量。

    重要：调用者应传入【均值序列】（浮点）而非中值序列。
    深度原始值是 uint16（整数 mm），中值也是整数，
    静态场景下中值会恒为同一整数 → σ 虚假地等于 0，
    完全掩盖亚毫米变化。必须用均值才能看到真实噪声。
    """
    s = series[np.isfinite(series)]
    if s.size < 20:
        return None
    # 去离群：偏离中值 > DEGLITCH_MM 的剔除
    med = float(np.median(s))
    keep = np.abs(s - med) < DEGLITCH_MM
    s2 = s[keep]
    n_glitch = int((~keep).sum())
    scatter = float(np.std(s2, ddof=1))
    dif = np.diff(s2)
    dif_sig = float(np.std(dif, ddof=1)) / np.sqrt(2) if dif.size > 1 else np.nan
    nw = len(s2) // fps
    drift = (float(np.ptp(s2[:nw * fps].reshape(nw, fps).mean(axis=1)))
             if nw >= 2 else np.nan)
    # 量化能被观测到的最小步进：看唯一值个数
    uniq = int(np.unique(np.round(s2, 3)).size)
    return dict(n=len(s2), z_ref=float(np.median(s2)), mean=float(np.mean(s2)),
                scatter=scatter, dif_sig=dif_sig, drift=drift,
                mdr_diff=3 * dif_sig, mdr_scat=3 * scatter,
                glitch=n_glitch, uniq=uniq,
                cnt_mean=float(np.mean(valid_cnt)) if len(valid_cnt) else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi", default=None, help="x,y,w,h（彩色 640x480）")
    ap.add_argument("--auto", action="store_true", help="自动定位杆")
    ap.add_argument("--physical", type=float, default=200.0,
                    help="钢尺实测的物理距离 mm")
    ap.add_argument("--baseline", type=int, default=60,
                    help="静置基线帧数（用于建立零点）")
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--save", default=None, help="保存基准 JSON")
    ap.add_argument("--monitor", action="store_true",
                    help="调零后进入实时监测")
    args = ap.parse_args()

    print("=" * 96)
    print("20cm 基准调零")
    print("=" * 96)

    pipe, tf, al, prof = start(args.alpha)
    ds = prof.get_device().first_depth_sensor()
    try:
        ex = ds.get_option(rs.option.exposure)
        gn = ds.get_option(rs.option.gain)
        print(f"\n传感器: 曝光={ex:.0f}μs  增益={gn:.0f}  "
              f"激光={ds.get_option(rs.option.laser_power):.0f}  "
              f"temporal α={args.alpha}")
    except Exception:
        pass

    # ── 1. 确定 ROI ──
    if args.auto or not args.roi:
        if not args.auto:
            print("\n未给 --roi，自动定位…")
        tgt, med = find_target(pipe, tf, al)
        if tgt is None:
            print("\n❌ 自动定位失败。请用 --roi x,y,w,h 指定。")
            print("   提示：在浏览器预览上点击目标读坐标，"
                  "或运行 rs_rod_test.py 看候选块。")
            pipe.stop()
            return 1
        x, y, rw, rh = tgt["roi"]
        print(f"\n自动定位到目标: 中位 {tgt['z']:.0f}mm  "
              f"{tgt['npx']}px  σ={tgt['sigma']:.1f}mm")
    else:
        x, y, rw, rh = (int(v) for v in args.roi.split(","))

    print(f"ROI = x={x} y={y} w={rw} h={rh}   (面积 {rw*rh}px)")

    # ── 2. 锁定像素集合 ──
    # 用多帧中值确定「稳定有效」的像素，之后固定使用这批像素
    # （避免逐帧有效集合漂移污染均值）
    for _ in range(20):
        pipe.wait_for_frames(timeout_ms=10000)
    st = []
    for _ in range(20):
        d, _ = grab(pipe, tf, al)
        st.append(d[y:y + rh, x:x + rw].astype(np.float32))
    stackp = np.stack(st)
    patch_med = np.median(stackp, axis=0)
    pm_valid = (patch_med > 0) & (patch_med < 65535)
    # 逐像素稳定性筛选（三道栅栏，缺一不可）：
    #   1) 时间 std 小       → 像素本身不跳变
    #   2) 时间极差小         → 排除慢跳变（std 会低估偶发跳变）
    #   3) 与 ROI 中值偏差小  → 排除读到背景/远端的像素
    # 仅用 (1) 不够：实测出现过均值被 4000mm 级离群像素拉偏 750mm。
    with np.errstate(invalid="ignore"):
        masked = np.where((stackp > 0) & (stackp < 65535), stackp, np.nan)
        pstd = np.nanstd(masked, axis=0)
        pmax = np.nanmax(masked, axis=0)
        pmin = np.nanmin(masked, axis=0)
    ptp = pmax - pmin
    ref_z = float(np.median(patch_med[pm_valid])) if pm_valid.sum() else 0.0
    stable = (pm_valid & np.isfinite(pstd) & (pstd < 5.0)
              & (ptp < 15.0)
              & (np.abs(patch_med - ref_z) < 30.0))
    n_stable = int(stable.sum())
    print(f"\n参考深度（ROI 中值）: {ref_z:.1f} mm")
    print(f"像素筛选: ROI {rw*rh}px → 有效 {int(pm_valid.sum())}px "
          f"→ 稳定 {n_stable}px ({100*n_stable/max(1,rw*rh):.0f}%)")
    if n_stable < 50:
        print("⚠️ 稳定像素太少（<50）。该目标是细杆时，"
              "考虑扩大 ROI 或提高白纸覆盖率。")
        if n_stable < 20:
            print("❌ 不足以测量，退出。")
            pipe.stop()
            return 2

    # ── 3. 采基线 ──
    # 同时记录「均值序列」（浮点，能看到亚毫米变化）与
    # 「中值序列」（整数，抗离群但被量化到 1mm）
    print(f"\n采静置基线 {args.baseline} 帧…")
    base_mean = np.full(args.baseline, np.nan)
    base_med = np.full(args.baseline, np.nan)
    bcnt = np.zeros(args.baseline, np.int32)
    for t in range(args.baseline):
        d, _ = grab(pipe, tf, al)
        vals = d[y:y + rh, x:x + rw][stable]
        v = vals[(vals > 0) & (vals < 65535)]
        bcnt[t] = v.size
        if v.size >= max(10, n_stable // 5):
            base_mean[t] = float(v.mean())
            base_med[t] = float(np.median(v))

    st_base = series_stats(base_mean, bcnt)      # ← 主统计基于均值
    st_med = series_stats(base_med, bcnt)        # ← 中值仅作对比
    if st_base is None:
        print("❌ 基线数据不足")
        pipe.stop()
        return 3

    # 一致性校验：均值与中值差异过大 → 仍有离群像素未清干
    # （均值被远端像素拉动时，中值口径反而可靠，因为它是抗离群的）
    consistent = abs(st_base["mean"] - st_med["z_ref"]) <= 20
    if not consistent:
        print(f"\n⚠️ 均值({st_base['mean']:.1f})与中值({st_med['z_ref']:.1f}) "
              f"相差 {abs(st_base['mean']-st_med['z_ref']):.1f}mm")
        print("   → 仍有离群像素未剔除，均值口径不可信。")
        print("   → 下面统计以中值口径为主；建议收紧阈值或换更干净的 ROI。")

    z_base = st_base["z_ref"] if consistent else st_med["z_ref"]
    k = args.physical / z_base if z_base > 0 else np.nan

    print("\n" + "=" * 96)
    print("调零结果")
    print("=" * 96)
    print(f"  z_ref（深度读数中位数）  {z_base:8.2f} mm")
    print(f"  d_ref（钢尺实测）        {args.physical:8.1f} mm")
    print(f"  尺度系数 k = d_ref/z_ref {k:8.5f}   "
          f"{'✅ 接近1' if abs(k-1) < 0.02 else '⚠️ 偏离1'}")
    print()
    print(f"  {'指标':<24}{'均值口径':>12}{'中值口径':>12}")
    print(f"  {'-'*48}")
    print(f"  {'深度散布 σ':<24}{st_base['scatter']:>12.3f}"
          f"{st_med['scatter']:>12.3f} mm")
    print(f"  {'MDR(3σ 帧差) ← 主指标':<24}{st_base['mdr_diff']:>12.3f}"
          f"{st_med['mdr_diff']:>12.3f} mm")
    print(f"  {'MDR(3σ 散布)':<24}{st_base['mdr_scat']:>12.3f}"
          f"{st_med['mdr_scat']:>12.3f} mm")
    print(f"  {'静置漂移':<24}{st_base['drift']:>12.3f}"
          f"{st_med['drift']:>12.3f} mm")
    print(f"  {'不同取值个数':<24}{st_base['uniq']:>12d}"
          f"{st_med['uniq']:>12d}")
    print(f"  {'剔除离群帧':<24}{st_base['glitch']:>12d}"
          f"{st_med['glitch']:>12d}")
    print(f"  {'平均有效像素':<24}{st_base['cnt_mean']:>12.0f}"
          f"{st_med['cnt_mean']:>12.0f} / {n_stable}")
    if st_med["scatter"] == 0.0:
        print(f"\n  ℹ️ 中值口径 σ=0 是因为 uint16 量化（整数 mm）——")
        print(f"     静置时中值恒为 {st_med['z_ref']:.0f}，看不到亚毫米变化。")
        print(f"     均值口径 σ={st_base['scatter']:.3f}mm 才是真实噪声水平。")

    print("\n" + "=" * 96)
    print("判定")
    print("=" * 96)
    mdr = st_base["mdr_diff"] if consistent else st_med["mdr_diff"]
    if mdr <= 2.0:
        print(f"  ✅ MDR = {mdr:.3f}mm ≤ 2mm（老师要求），"
              f"余量 {2.0/max(mdr,1e-6):.1f}×")
    else:
        print(f"  ⚠️ MDR = {mdr:.3f}mm > 2mm，需改善")
    print(f"  {'✅' if st_base['drift'] < 1 else '⚠️'} 静置漂移 "
          f"{st_base['drift']:.3f}mm "
          f"{'可忽略' if st_base['drift'] < 1 else '需考虑调零周期'}")

    # ── 4. 保存 ──
    if args.save:
        rec = dict(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            roi=[x, y, rw, rh], aligned_size=[640, 480],
            stable_mask_count=n_stable,
            physical_mm=args.physical, z_ref=z_base, scale_k=float(k),
            baseline=st_base, alpha=args.alpha,
            exposure=float(ex) if 'ex' in dir() else None,
            gain=float(gn) if 'gn' in dir() else None,
        )
        # 同时保存稳定掩膜的坐标，便于后续脚本精确复现
        ys, xs = np.where(stable)
        rec["stable_pixels_xy"] = [[int(a), int(b)] for a, b in zip(xs, ys)]
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 基准已保存: {args.save}")
        print(f"   含 {n_stable} 个稳定像素的坐标，可精确复现")

    # ── 5. 监测模式 ──
    if args.monitor:
        print("\n" + "=" * 96)
        print("实时监测（Ctrl-C 退出）   displacement = (z - z_base) * k")
        print("=" * 96)
        print(f"{'时间s':>7} {'z(mm)':>9} {'位移(mm)':>10} {'有效px':>7} {'σ窗':>7}")
        t0 = time.time()
        hist = []
        try:
            while True:
                d, _ = grab(pipe, tf, al)
                vals = d[y:y + rh, x:x + rw][stable]
                v = vals[(vals > 0) & (vals < 65535)]
                if v.size < 10:
                    time.sleep(0.03)
                    continue
                z = float(v.mean())          # 均值口径（浮点，能看亚毫米）
                disp = (z - z_base) * k
                hist.append(disp)
                if len(hist) > 30:
                    hist.pop(0)
                if len(hist) % 10 == 0:
                    print(f"{time.time()-t0:>7.1f} {z:>9.2f} {disp:>10.3f} "
                          f"{v.size:>7} {np.std(hist, ddof=1) if len(hist)>1 else 0:>7.3f}")
        except KeyboardInterrupt:
            print("\n监测结束。")
            if hist:
                h = np.array(hist)
                print(f"  {len(h)} 个样本  位移范围 [{h.min():.3f}, {h.max():.3f}] mm  "
                      f"σ={h.std(ddof=1):.3f}mm")

    pipe.stop()
    print("\n完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
