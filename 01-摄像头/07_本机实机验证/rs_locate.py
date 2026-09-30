#!/usr/bin/env python3
"""定位杆 + 评估白纸质量 —— 重贴白纸后先用这个。

为什么需要专门的定位脚本
------------------------
之前多次出现「自动定位选错目标」，根因有三：
  1. 黑板有了读数（反光），与杆在图像上相邻
  2. 白纸卷曲导致深度断续，连通块被切碎
  3. 按面积排序会选中黑板（69804px）而非杆（4547px）

本脚本的改进
------------
  1. 用「彩色图上白纸很亮」作为辅助判据（白纸在灰度图上显著偏亮）
  2. 形状判据：竖直细长（高/宽 > 1.3）
  3. σ 判据：白纸平整时 σ 应该小
  4. 逐行剖面：报告白纸的「连续可用行数」，这是质量关键指标
  5. 画标注图，人工可核对

用法
----
sudo -E /opt/anaconda3/envs/realsense/bin/python3 rs_locate.py
sudo -E .../rs_locate.py --save roi.json
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
    from scipy import ndimage as ndi
    HAS_SCI = True
except ImportError:
    HAS_SCI = False


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--near", type=float, default=150.0,
                    help="杆的深度下限 mm")
    ap.add_argument("--far", type=float, default=400.0,
                    help="杆的深度上限 mm")
    ap.add_argument("--save", default=None)
    ap.add_argument("--annotate", default="/tmp/locate.png")
    args = ap.parse_args()

    print("=" * 84)
    print("定位杆 + 评估白纸质量")
    print("=" * 84)

    pipe, al, tf = start()
    try:
        for _ in range(60):
            pipe.wait_for_frames(timeout_ms=10000)

        st, col = [], None
        for _ in range(25):
            fs = al.process(pipe.wait_for_frames(timeout_ms=10000))
            st.append(np.asanyarray(
                tf.process(fs.get_depth_frame()).get_data()).astype(np.float32))
            col = np.asanyarray(fs.get_color_frame().get_data()).copy()
        med = np.median(np.stack(st), axis=0)
    finally:
        try:
            pipe.stop()
        except Exception:
            pass

    gray = col.mean(axis=2)
    V = (med > 0) & (med < 65535)
    print(f"\n整体有效率 {100*V.mean():.1f}%")

    # ── 候选分割 ──
    band = (med >= args.near) & (med < args.far)
    print(f"{args.near:.0f}-{args.far:.0f}mm 像素: {int(band.sum())}")
    if band.sum() < 200:
        print("❌ 该深度范围内几乎没有像素。检查：杆在不在？距离是多少？")
        print(f"   场景深度分布: p5={np.percentile(med[V],5):.0f} "
              f"p50={np.percentile(med[V],50):.0f} "
              f"p95={np.percentile(med[V],95):.0f} mm")
        return 1

    if not HAS_SCI:
        print("⚠️ 无 scipy，跳过连通分析")
        return 1

    lab, n = ndi.label(band)
    sizes = ndi.sum(band, lab, range(1, n + 1))
    order = np.argsort(-sizes)

    print(f"\n{'#':>3}{'像素':>7}{'中位Z':>8}{'σ':>7}{'宽':>5}{'高':>5}"
          f"{'高/宽':>7}{'灰度':>7}{'连续行':>7}  {'bbox':>22}  {'形状':>6}")
    print("-" * 84)

    cands = []
    for i, idx in enumerate(order[:12]):
        k = int(idx) + 1
        if sizes[idx] < 300:
            break
        b = lab == k
        ys, xs = np.where(b)
        z = float(np.median(med[b]))
        sig = float(np.std(med[b]))
        bw = int(xs.max() - xs.min() + 1)
        bh = int(ys.max() - ys.min() + 1)
        ar = bh / max(1, bw)
        gmean = float(gray[b].mean())
        # 连续行数：该块内「每行都有有效像素」的最长连续段
        rows_have = b.any(axis=1)
        best_run = cur = 0
        for v in rows_have:
            cur = cur + 1 if v else 0
            best_run = max(best_run, cur)
        shape = ("细高" if ar > 1.3 else ("横条" if ar < 0.7 else "方块"))
        print(f"{i+1:>3}{int(sizes[idx]):>7}{z:>8.0f}{sig:>7.1f}"
              f"{bw:>5}{bh:>5}{ar:>7.2f}{gmean:>7.0f}{best_run:>7}  "
              f"x[{xs.min():>3},{xs.max():>3}] y[{ys.min():>3},{ys.max():>3}]  {shape:>6}")
        cands.append(dict(idx=i + 1, npx=int(sizes[idx]), z=z, sig=sig,
                          bw=bw, bh=bh, ar=ar, gray=gmean, run=best_run,
                          x0=int(xs.min()), y0=int(ys.min()), mask=b))

    if not cands:
        print("❌ 无足够大的候选")
        return 1

    # ── 打分选杆 ──
    # 白纸特征：竖直细长、灰度偏高、σ 小、连续行多
    for c in cands:
        s = 0.0
        s += 10.0 if c["ar"] > 1.3 else -10.0        # 竖直
        s += (c["gray"] - 100) / 10.0                 # 白纸亮
        s -= c["sig"] * 2.0                           # 表面平整
        s += c["run"] / 5.0                           # 纵向连续
        s += min(c["npx"], 3000) / 300.0
        c["score"] = s
    cands.sort(key=lambda c: -c["score"])
    best = cands[0]

    print("\n" + "=" * 84)
    print(f"★ 选中 #{best['idx']}（得分 {best['score']:.1f}）")
    print(f"  {best['npx']}px  中位 {best['z']:.0f}mm  σ={best['sig']:.1f}  "
          f"{best['bw']}x{best['bh']}  灰度 {best['gray']:.0f}  连续行 {best['run']}")
    print("=" * 84)

    # ── 白纸质量评估 ──
    b = best["mask"]
    print(f"\n【白纸质量评估】")
    print(f"  有效像素率: {100*b.sum()/max(1,best['bw']*best['bh']):.1f}%")
    print(f"  最长连续行: {best['run']} / {best['bh']} 行 "
          f"({100*best['run']/max(1,best['bh']):.0f}%)")
    if best["run"] < best["bh"] * 0.7:
        print(f"  ⚠️ 纵向不连续 —— 白纸可能还是卷曲/有空洞")
        print(f"     建议重贴，使纸面平整正对相机")
    if best["gray"] < 120:
        print(f"  ⚠️ 灰度偏低（{best['gray']:.0f}）—— 目标可能偏暗，"
              f"影响红外匹配")
    print(f"\n  逐行剖面:")
    ys, xs = np.where(b)
    for yy in range(int(ys.min()), int(ys.max()) + 1, 10):
        row = med[yy:yy + 10, :]
        rm = b[yy:yy + 10, :]
        if rm.sum() < 3:
            print(f"    y={yy:>3}: (无)")
            continue
        vals = row[rm]
        vals = vals[(vals > 0) & (vals < 65535)]
        if vals.size < 3:
            print(f"    y={yy:>3}: (无有效)")
            continue
        print(f"    y={yy:>3}: {vals.size:>3}px  中位{np.median(vals):>6.0f}mm  "
              f"σ={np.std(vals):>5.1f}")

    # ── 建议 ROI（内部收紧，避边缘） ──
    x0, y0, bw, bh = best["x0"], best["y0"], best["bw"], best["bh"]
    ix = int(x0 + bw * 0.2)
    iy = int(y0 + bh * 0.15)
    iw = max(8, int(bw * 0.6))
    ih = max(8, int(bh * 0.7))
    roi = (ix, iy, iw, ih)
    print(f"\n★ 建议 ROI = {roi}")
    print(f"  即 --roi {roi[0]},{roi[1]},{roi[2]},{roi[3]}")

    # ── 实测该 ROI ──
    print(f"\n【实测 {roi}（90 帧）】")
    pipe2, al2, tf2 = start()
    try:
        for _ in range(50):
            pipe2.wait_for_frames(timeout_ms=10000)

        def grab():
            fs = al2.process(pipe2.wait_for_frames(timeout_ms=10000))
            return np.asanyarray(tf2.process(fs.get_depth_frame()).get_data())

        m = M.Measurement(roi=roi, trim_mm=20.0)
        ser, nus = [], []
        for _ in range(90):
            z, nn, sp = m.read_full(grab())
            ser.append(z)
            nus.append(nn)
        a = M.series_mdr(np.array(ser))
    finally:
        try:
            pipe2.stop()
        except Exception:
            pass

    print(f"  z0    = {a['z_ref']:.3f} mm")
    print(f"  MDR   = {a['mdr']:.4f} mm   "
          f"{'✅ 达标' if a['mdr'] <= 2 else '❌'}")
    print(f"  σ     = {a['scatter']:.4f} mm")
    print(f"  漂移  = {a['drift']:.4f} mm")
    print(f"  像素  = {np.mean(nus):.0f}")

    if args.save:
        out = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
                   roi=list(roi), roi_loose=[x0, y0, bw, bh],
                   z0=a["z_ref"], mdr=a["mdr"], scatter=a["scatter"],
                   drift=a["drift"], npx=float(np.mean(nus)),
                   paper_quality=dict(valid_ratio=float(b.sum()/max(1, bw*bh)),
                                      run_ratio=float(best["run"]/max(1, bh)),
                                      gray=best["gray"]))
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 已保存 {args.save}")

    # ── 标注图 ──
    try:
        from PIL import Image, ImageDraw
        img = Image.fromarray(col[:, :, ::-1]).convert("RGB")
        d = ImageDraw.Draw(img)
        colors = [(255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 255, 0),
                  (255, 0, 255), (0, 255, 255)]
        for i, c in enumerate(cands[:6]):
            cc = colors[i % len(colors)]
            d.rectangle([c["x0"], c["y0"], c["x0"] + c["bw"], c["y0"] + c["bh"]],
                        outline=cc, width=1)
            d.text((c["x0"] + 2, max(0, c["y0"] - 12)),
                   f"#{c['idx']} {c['z']:.0f}", fill=cc)
        d.rectangle([roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3]],
                    outline=(255, 255, 255), width=2)
        d.text((roi[0], roi[1] + roi[3] + 2), "SELECTED", fill=(255, 255, 255))
        img.save(args.annotate)
        print(f"✅ 标注图: {args.annotate}")
    except ImportError:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
