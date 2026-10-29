#!/usr/bin/env python3
"""细杆边界测试：解决「逐帧闪烁」导致的假 MDR。

问题机制（已定位）
------------------
3mm 细杆在 20cm 处，单像素时间 σ 只有 0.20mm（很好），
但块级均值的 MDR 却是 25mm（很差）—— 矛盾。

根因：杆的深度读数**逐帧闪烁**。某些像素这一帧有效、下一帧失效。
逐帧用「当前帧所有有效像素」求均值时，被平均的集合每帧都不同：
  少量背景像素（4000mm）漏进来，就能把 3200 像素的均值拉动上百毫米。
这不是深度真的变了，是集合在变。

中位数对此免疫（所以中值恒为 228mm），但中值被量化到 1mm，
看不到亚毫米变化 —— 对 2mm 判据不够用。

本脚本对比 6 种统计策略
-----------------------
  S1 裸有效逐帧均值       （已知最差，作为基线）
  S2 裸有效逐帧中值       （抗泄漏但量化到 1mm）
  S3 固定像素集合均值     （缺像素用时间中值填补）
  S4 中值锚定截尾均值     ★ 提案：以中值为中心，只对 ±W 内像素求均值
  S5 逐像素差分再平均     （对闪烁最鲁棒）
  S6 25/75 分位中点       （介于中值与均值之间）

并对比 temporal 开/关，以及不同 ROI 宽度。

用法
----
sudo -E .../rs_rod_boundary.py
sudo -E .../rs_rod_boundary.py --roi 369,48,54,125 --frames 400
"""

import argparse
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


def start(alpha):
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
    return np.asanyarray(df.get_data())


def find_rod(pipe, tf, al, warmup=45):
    """自动定位最像「白纸杆」的深度一致块。"""
    for _ in range(warmup):
        pipe.wait_for_frames(timeout_ms=10000)
    st = [grab(pipe, tf, al).astype(np.float32) for _ in range(20)]
    med = np.median(np.stack(st), axis=0)
    if not HAS_SCIPY:
        return None, med, st

    valid = (med > 0) & (med < 65535)
    n = int(valid.sum())
    if n < 300:
        return None, med, st
    h, w = med.shape
    di = med.astype(np.int32)
    idx = np.full((h, w), -1, np.int64)
    idx[valid] = np.arange(n)
    mh = valid[:, :-1] & valid[:, 1:] & (np.abs(di[:, :-1] - di[:, 1:]) < 30)
    mv = valid[:-1, :] & valid[1:, :] & (np.abs(di[:-1, :] - di[1:, :]) < 30)
    rh, ch = np.where(mh)
    rv, cv = np.where(mv)
    if rh.size + rv.size == 0:
        return None, med, st
    rows = np.concatenate([idx[rh, ch], idx[rv, cv]])
    cols = np.concatenate([idx[rh, ch + 1], idx[rv + 1, cv]])
    g = coo_matrix((np.ones(rows.size, np.int8), (rows, cols)), shape=(n, n))
    ncomp, labels = connected_components(g, directed=False)
    cnt = np.bincount(labels, minlength=ncomp)
    vy, vx = np.where(valid)

    best = None
    for k in range(ncomp):
        if cnt[k] < 300:
            continue
        sel = labels == k
        ys, xs = vy[sel], vx[sel]
        z = float(np.median(med[ys, xs]))
        sig = float(np.std(med[ys, xs]))
        bw = int(xs.max() - xs.min() + 1)
        bh = int(ys.max() - ys.min() + 1)
        # 白纸杆特征：15-60px 较近的距离、偏高、σ 小
        score = 0
        if 150 < z < 350:
            score += 1000
        if 15 <= bw <= 70:
            score += 500
        if bh > 60:
            score += 300
        score -= sig * 30
        score += min(len(ys), 5000) / 100.0
        if best is None or score > best[0]:
            best = (score, k, z, len(ys), sig,
                    int(xs.min()), int(ys.min()), bw, bh)
    if best is None:
        return None, med, st
    _, k, z, npx, sig, x0, y0, bw, bh = best
    return dict(roi=[x0, y0, bw, bh], z=z, npx=npx, sigma=sig,
                label=int(k)), med, st


def build_fixed_mask(st, y, x, h, w, valid_ratio=0.9, dev=40.0):
    """从参考帧构建固定像素掩膜：要求像素在 ≥valid_ratio 比例帧内有效，
    且深度接近 ROI 中值（排除背景像素）。"""
    sub = np.stack([s[y:y + h, x:x + w] for s in st])
    ok = (sub > 0) & (sub < 65535)
    ratio_map = ok.mean(axis=0)
    with np.errstate(invalid="ignore"):
        tmed = np.nanmedian(np.where(ok, sub, np.nan), axis=0)
    roi_med = float(np.median(tmed[ratio_map > 0.5])) if (ratio_map > 0.5).sum() else 0.0
    mask = ((ratio_map >= valid_ratio)
            & np.isfinite(tmed)
            & (np.abs(tmed - roi_med) < dev))
    return mask, tmed, roi_med, ratio_map


def series_stats(s, fps=FPS, glitch=200.0):
    ok = np.isfinite(s)
    if ok.sum() < 40:
        return None
    s2 = s[ok]
    med = float(np.median(s2))
    keep = np.abs(s2 - med) < glitch
    s3 = s2[keep]
    if s3.size < 30:
        s3 = s2
    dif = np.diff(s3)
    dif_sig = float(np.std(dif, ddof=1)) / np.sqrt(2) if dif.size > 1 else np.nan
    nw = len(s3) // fps
    drift = (float(np.ptp(s3[:nw * fps].reshape(nw, fps).mean(axis=1)))
             if nw >= 2 else np.nan)
    return dict(n=len(s3), z=float(np.median(s3)),
                mean=float(np.mean(s3)),
                scatter=float(np.std(s3, ddof=1)),
                mdr=3 * dif_sig, mdr_scat=3 * float(np.std(s3, ddof=1)),
                drift=drift, glitch=int((~keep).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi", default=None, help="x,y,w,h（彩色 640x480）")
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--trim", type=float, default=20.0,
                    help="S4 中值锚定截尾宽度 mm")
    args = ap.parse_args()

    print("=" * 100)
    print("细杆边界测试：6 种统计策略对比")
    print("=" * 100)

    # ── 阶段 1：用 temporal 定位杆 ──
    pipe, tf, al, prof = start(args.alpha)
    try:
        tgt, med_ref, st_ref = find_rod(pipe, tf, al)
    finally:
        try:
            pipe.stop()
        except Exception:
            pass
    if tgt is None:
        print("\n❌ 未找到杆。请用 --roi x,y,w,h 指定。")
        return 1
    x, y, rw, rh = tgt["roi"]
    print(f"\n自动定位: ROI x={x} y={y} w={rw} h={rh}")
    print(f"  中位 {tgt['z']:.0f}mm  {tgt['npx']}px  空间σ={tgt['sigma']:.1f}mm")

    fixed, tmed, roi_med, ratio_map = build_fixed_mask(st_ref, y, x, rh, rw)
    nfix = int(fixed.sum())
    print(f"\n固定像素掩膜（要求 ≥90% 帧有效且距中值 <40mm）: {nfix}px / {rw*rh}px"
          f"  ({100*nfix/max(1,rw*rh):.0f}%)")
    if nfix < 30:
        print("⚠️ 固定像素太少，改用阈值 0.7 重试")
        fixed, tmed, roi_med, ratio_map = build_fixed_mask(
            st_ref, y, x, rh, rw, valid_ratio=0.7)
        nfix = int(fixed.sum())
        print(f"   重试后: {nfix}px")

    # 闪烁率统计
    print(f"\n像素有效率分布: 中位 {np.median(ratio_map)*100:.0f}%  "
          f"≥90% 的像素占 {100*(ratio_map>=0.9).mean():.0f}%  "
          f"≥70% 占 {100*(ratio_map>=0.7).mean():.0f}%")
    print(f"  ← 有效率低的像素就是「闪烁」来源")

    # ── 阶段 2：主体采集（不滤波，看原始能力）──
    time.sleep(2)

    for alpha, tag in ((0.0, "无滤波"), (args.alpha, f"temporal α={args.alpha}")):
        pipe2, tf2, al2, _ = start(alpha)
        for _ in range(35):
            pipe2.wait_for_frames(timeout_ms=10000)

        N = args.frames
        strategies = {k: np.full(N, np.nan) for k in
                      ("S1 裸有效均值", "S2 裸有效中值",
                       "S3 固定像素均值", "S4 中值锚定均值",
                       "S5 逐像素差分", "S6 分位中点")}
        extra = {"有效像素数": np.zeros(N, np.int32),
                 "固定集有效数": np.zeros(N, np.int32)}

        # S5 需要预存每帧的逐像素值
        pxbuf = np.full((N, nfix), np.nan, np.float32)
        # 固定掩膜对应的「时间中值」也拉平成同样长度的向量
        tmed_flat = tmed[fixed].astype(np.float32)

        t0 = time.time()
        for t in range(N):
            d = grab(pipe2, tf2, al2)
            sub = d[y:y + rh, x:x + rw].astype(np.float32)

            # S1/S2: 该帧所有有效像素
            vv = sub[(sub > 0) & (sub < 65535)]
            extra["有效像素数"][t] = vv.size
            if vv.size >= 20:
                strategies["S1 裸有效均值"][t] = vv.mean()
                strategies["S2 裸有效中值"][t] = np.median(vv)
                p25, p75 = np.percentile(vv, [25, 75])
                strategies["S6 分位中点"][t] = 0.5 * (p25 + p75)
                med_v = np.median(vv)
                # S4: 以中值为中心，只对 ±trim 内像素求均值
                win = vv[np.abs(vv - med_v) <= args.trim]
                if win.size >= 10:
                    strategies["S4 中值锚定均值"][t] = win.mean()

            # S3/S5: 固定像素集合
            vals = sub[fixed]
            m = np.isfinite(tmed_flat) & (vals > 0) & (vals < 65535)
            extra["固定集有效数"][t] = int(m.sum())
            if m.sum() >= 20:
                # S3: 缺失像素用其时间中值填补（保持集合恒定）
                filled = np.where(m, vals, tmed_flat)
                strategies["S3 固定像素均值"][t] = float(np.nanmean(filled))
                pxbuf[t, :] = np.where(m, vals, np.nan)
        dur = time.time() - t0

        print("\n" + "=" * 100)
        print(f"【{tag}】{N} 帧，耗时 {dur:.1f}s")
        print("=" * 100)
        ec = extra["有效像素数"]
        fc = extra["固定集有效数"]
        print(f"  有效像素数波动: {ec.mean():.0f} ± {ec.std():.1f} "
              f"(范围 {ec.min()}–{ec.max()}, 相对波动 "
              f"{100*ec.std()/max(1,ec.mean()):.1f}%)")
        print(f"  固定集有效数:   {fc.mean():.0f} ± {fc.std():.1f} "
              f"(范围 {fc.min()}–{fc.max()})")
        print()
        print(f"{'策略':<20} {'MDR(3σ)':>10} {'σ散布':>9} {'3σ散布':>9} "
              f"{'漂移':>8} {'离群':>6} {'判定':>8}")
        print("-" * 100)

        results = {}
        for name in strategies:
            stt = series_stats(strategies[name])
            if stt is None:
                print(f"{name:<20} {'数据不足':>10}")
                continue
            results[name] = stt
            v = ("✅达标" if stt["mdr"] <= 2 else
                 ("⚠️接近" if stt["mdr"] <= 5 else "❌超标"))
            print(f"{name:<20} {stt['mdr']:>10.3f} {stt['scatter']:>9.3f} "
                  f"{stt['mdr_scat']:>9.3f} {stt['drift']:>8.3f} "
                  f"{stt['glitch']:>6} {v:>8}")

        # S5 单独算：逐像素去均值后取平均（对闪烁最鲁棒）
        if np.isfinite(pxbuf).sum() > 100:
            base = np.nanmedian(pxbuf, axis=0)
            dev = pxbuf - base
            s5 = np.nanmean(dev, axis=1)
            stt = series_stats(s5)
            if stt:
                results["S5 逐像素差分"] = stt
                v = ("✅达标" if stt["mdr"] <= 2 else
                     ("⚠️接近" if stt["mdr"] <= 5 else "❌超标"))
                print(f"{'S5 逐像素差分':<20} {stt['mdr']:>10.3f} "
                      f"{stt['scatter']:>9.3f} {stt['mdr_scat']:>9.3f} "
                      f"{stt['drift']:>8.3f} {stt['glitch']:>6} {v:>8}")

        # 最优
        if results:
            best = min(results.items(), key=lambda kv: kv[1]["mdr"])
            print(f"\n  ★ 本组最优: {best[0]}  MDR={best[1]['mdr']:.3f}mm")
        try:
            pipe2.stop()
        except Exception:
            pass
        time.sleep(1.5)

    print("\n" + "=" * 100)
    print("结论")
    print("=" * 100)
    print("  S1 是此前使用的策略，若它明显差于 S3/S4/S5，")
    print("  则证明「逐帧集合漂移」是假 MDR 的主因，应以固定集合或锚定策略替代。")
    print(f"\n  注：S4 截尾宽度 = ±{args.trim}mm；S3 缺像素用时间中值填补。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
