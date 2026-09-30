#!/usr/bin/env python3
"""D435i Z 轴位移测量核心库 —— 已修正统计方法。

为什么需要这个库
----------------
3mm 细杆在 20cm 处的实测揭示：常规的「逐帧对有效像素求均值」会产生
高达 200mm 的**假 MDR**（真实值 0.334mm），差 598 倍。

两个叠加机制：
  机制一（集合漂移）：细杆深度读数逐帧闪烁，被平均的像素集合每帧不同。
                     200 个背景像素（4000mm）足以把 3200 像素的均值拉动 20mm+。
  机制二（背景复活）：用「时间中值」填补固定像素集合，会把少量背景像素
                     固化为常驻参与者，它们「复活」时注入 4000mm 的值。

正确方法（S4 中值锚定截尾均值）：
  med = median(该帧所有有效像素)          # 抗离群锚点
  win = 像素中 |值 - med| <= trim          # 排除背景/离群
  result = mean(win)                       # 对干净窗口求均值（浮点，看亚毫米）

效果：MDR 从 199.694mm → 0.334mm（598× 改善）

使用方法
--------
    from rs_measure import Measurement, find_target_block

    m = Measurement(roi=(356, 48, 70, 134), trim_mm=20.0)
    m.calibrate(n_frames=60)                 # 建立基线
    z = m.read(depth_frame)                  # 单次读数（mm, float）
    zs = m.read_series(frames)               # 批量

    # 或自动定位目标
    roi = find_target_block(depth_frame)
"""

from __future__ import annotations

import numpy as np

# ── 默认参数（基于实测标定）──
DEFAULT_TRIM_MM = 20.0      # S4 截尾窗口半宽
DEFAULT_DEGLITCH_MM = 200.0  # 帧级离群剔除阈值


def check_aligned(d, roi):
    """校验深度图尺寸与 ROI 是否匹配（防止 align 静默失败）。

    背景：rs.align(rs.stream.color) 在 pipeline 没有启用彩色流时
    会【静默不对齐】—— 不报错，深度图仍是 848x480。
    若 ROI 是按 640x480（彩色对齐坐标系）定义的，就会落在错误位置。
    实测后果：参与像素从 12109 掉到 74，读数完全错误。

    返回 (ok, msg)
    """
    h, w = d.shape[:2]
    x, y, rw, rh = roi
    problems = []
    if w == 848:
        problems.append(
            f"深度图为 {w}x{h}（未对齐），但 ROI 通常在 640x480 彩色"
            f"对齐坐标系定义。请确认已 enable_stream(color) 并 align。")
    if x + rw > w or y + rh > h:
        problems.append(
            f"ROI {roi} 超出图像边界 {w}x{h}。")
    # 检查 ROI 是否有足够有效像素
    sub = d[y:y + rh, x:x + rw]
    valid = ((sub > 0) & (sub < 65535)).mean()
    if valid < 0.3:
        problems.append(
            f"ROI 内有效像素仅 {100*valid:.1f}%（<30%）。"
            f"目标可能不在该 ROI，或坐标系数错。")
    if problems:
        return False, " / ".join(problems)
    return True, f"OK（{w}x{h}, ROI 有效率 {100*valid:.0f}%）"


def safe_measure(d, roi, trim_mm=DEFAULT_TRIM_MM, min_px=50):
    """带校验的测量。返回 (z, n, spread, warn)。warn 非空表示可疑。"""
    z, n, spread = measure_s4(d, roi, trim_mm)
    ok, msg = check_aligned(d, roi)
    warn = "" if ok else msg
    if n < min_px:
        warn = (warn + " / " if warn else "") + f"参与像素仅 {n}（<{min_px}）"
    return z, n, spread, warn


def measure_s4(d: np.ndarray, roi: tuple[int, int, int, int],
               trim_mm: float = DEFAULT_TRIM_MM) -> tuple[float, int, float]:
    """S4 中值锚定截尾均值。

    参数
    ----
    d       : 深度图（uint16，毫米）
    roi     : (x, y, w, h)，基于对齐后的彩色分辨率
    trim_mm : 截尾窗口半宽（mm）

    返回
    ----
    (z_mm, n_used, spread_mm)
      z_mm     : 测量值（mm，float）
      n_used   : 窗口内参与平均的像素数
      spread_mm: 窗口内像素的空间标准差（mm）
    """
    x, y, w, h = roi
    sub = d[y:y + h, x:x + w].astype(np.float32)
    vv = sub[(sub > 0) & (sub < 65535)]
    if vv.size < 10:
        return float("nan"), 0, float("nan")
    med = float(np.median(vv))
    sel = vv[np.abs(vv - med) <= trim_mm]
    if sel.size < 5:
        # 窗口内样本不足，退回中值（会有 1mm 量化损失，但不至于失败）
        return med, int(vv.size), float(np.std(vv))
    return float(sel.mean()), int(sel.size), float(np.std(sel))


def measure_median(d: np.ndarray, roi: tuple[int, int, int, int]):
    """对照用：裸中值（抗离群但量化到 1mm）。"""
    x, y, w, h = roi
    sub = d[y:y + h, x:x + w].astype(np.float32)
    vv = sub[(sub > 0) & (sub < 65535)]
    if vv.size < 10:
        return float("nan"), 0
    return float(np.median(vv)), int(vv.size)


def series_mdr(series: np.ndarray, fps: int = 30,
               deglitch_mm: float = DEFAULT_DEGLITCH_MM) -> dict | None:
    """对时间序列算 MDR 等指标。

    注意：series 应是**均值序列**（浮点）。若传入中值序列（整数），
    静置时 σ 会虚假地等于 0（uint16 量化）。
    """
    s = np.asarray(series, dtype=np.float64)
    ok = np.isfinite(s)
    if ok.sum() < 40:
        return None
    s2 = s[ok]
    med = float(np.median(s2))
    keep = np.abs(s2 - med) < deglitch_mm
    s3 = s2[keep] if (s2[keep].size >= 30) else s2
    dif = np.diff(s3)
    dif_sig = float(np.std(dif, ddof=1)) / np.sqrt(2) if dif.size > 1 else np.nan
    nw = len(s3) // max(1, fps)
    drift = (float(np.ptp(s3[:nw * fps].reshape(nw, fps).mean(axis=1)))
             if nw >= 2 else float("nan"))
    return dict(
        n=int(s3.size),
        z_ref=float(np.median(s3)),
        mean=float(np.mean(s3)),
        scatter=float(np.std(s3, ddof=1)),
        mdr=3 * dif_sig,
        mdr_scatter=3 * float(np.std(s3, ddof=1)),
        drift=drift,
        glitch=int((~keep).sum()),
        uniq=int(np.unique(np.round(s3, 3)).size),
    )


def find_target_block(d: np.ndarray, min_px: int = 300,
                      rng: tuple[float, float] = (140.0, 400.0),
                      prefer_vertical: bool = True):
    """在深度图中定位最像「竖直白纸杆」的深度一致块。

    返回 dict(roi, z, npx, sigma) 或 None。
    """
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
    except ImportError:
        return _find_target_simple(d, min_px, rng)

    med = d.astype(np.float32)
    valid = (med > 0) & (med < 65535)
    n = int(valid.sum())
    if n < min_px:
        return None
    h, w = med.shape
    di = med.astype(np.int32)
    idx = np.full((h, w), -1, np.int64)
    idx[valid] = np.arange(n)

    mh = valid[:, :-1] & valid[:, 1:] & (np.abs(di[:, :-1] - di[:, 1:]) < 30)
    mv = valid[:-1, :] & valid[1:, :] & (np.abs(di[:-1, :] - di[1:, :]) < 30)
    rh, ch = np.where(mh)
    rv, cv = np.where(mv)
    if rh.size + rv.size == 0:
        return _find_target_simple(d, min_px, rng)

    rows = np.concatenate([idx[rh, ch], idx[rv, cv]])
    cols = np.concatenate([idx[rh, ch + 1], idx[rv + 1, cv]])
    g = coo_matrix((np.ones(rows.size, np.int8), (rows, cols)), shape=(n, n))
    ncomp, labels = connected_components(g, directed=False)
    cnt = np.bincount(labels, minlength=ncomp)
    vy, vx = np.where(valid)

    best = None
    for k in range(ncomp):
        if cnt[k] < min_px:
            continue
        sel = labels == k
        ys, xs = vy[sel], vx[sel]
        z = float(np.median(med[ys, xs]))
        if not (rng[0] < z < rng[1]):
            continue
        sig = float(np.std(med[ys, xs]))
        bw = int(xs.max() - xs.min() + 1)
        bh = int(ys.max() - ys.min() + 1)
        score = min(len(ys), 6000) / 100.0 - sig * 30
        if prefer_vertical and bh > bw * 1.5:
            score += 300
        if best is None or score > best[0]:
            best = (score, dict(
                roi=(int(xs.min()), int(ys.min()), bw, bh),
                z=z, npx=int(len(ys)), sigma=sig))
    return best[1] if best else None


def _find_target_simple(d, min_px, rng):
    """无 scipy 时的退路：按深度值找最大连续区域。"""
    med = d.astype(np.float32)
    v = (med > 0) & (med < 65535)
    best = None
    for lo in range(int(rng[0]), int(rng[1]), 20):
        m = v & (med >= lo) & (med < lo + 20)
        if m.sum() < min_px:
            continue
        ys, xs = np.where(m)
        z = float(np.median(med[m]))
        sig = float(np.std(med[m]))
        score = m.sum() / 100.0 - sig * 30
        if best is None or score > best[0]:
            best = (score, dict(roi=(int(xs.min()), int(ys.min()),
                                     int(xs.max() - xs.min() + 1),
                                     int(ys.max() - ys.min() + 1)),
                                z=z, npx=int(m.sum()), sigma=sig))
    return best[1] if best else None


class Measurement:
    """Z 轴位移测量器（含调零）。

    用法
    ----
        m = Measurement(roi=(356, 48, 70, 134))
        m.calibrate(frames)            # frames: list of depth arrays
        z = m.read(d)                  # 单帧
        d_mm = m.displacement(z)       # 相对基线的位移
    """

    def __init__(self, roi, trim_mm=DEFAULT_TRIM_MM,
                 physical_mm=None, fps=30):
        self.roi = tuple(int(v) for v in roi)
        self.trim_mm = float(trim_mm)
        self.fps = int(fps)
        self.physical_mm = physical_mm
        self.z_base = None
        self.scale_k = 1.0
        self.baseline_stats = None

    def read(self, d):
        """对一帧深度图做 S4 测量，返回 mm（float）。"""
        z, n, spread = measure_s4(d, self.roi, self.trim_mm)
        return z

    def read_full(self, d):
        """同 read()，但同时返回参与像素数与窗口内空间 std。"""
        return measure_s4(d, self.roi, self.trim_mm)

    def read_series(self, frames):
        """对深度图序列做 S4 测量，返回 mm 数组。

        注意：frames 必须是【深度图数组】的序列，不是已算好的标量。
        传了标量会报 TypeError —— 这是接口设计的常见误用点。
        """
        out = []
        for i, d in enumerate(frames):
            if not hasattr(d, "shape"):
                raise TypeError(
                    f"read_series 需要深度图数组，但第 {i} 个元素是 "
                    f"{type(d).__name__}。若已有算好的标量序列，"
                    f"直接用 series_mdr() 而不是 read_series()。")
            out.append(self.read(d))
        return np.array(out, dtype=np.float64)

    def calibrate(self, frames, deglitch=DEFAULT_DEGLITCH_MM):
        """用静置数据建立零点。

        frames 可以是：
          - 深度图数组的序列（会自动走 read_series）
          - 已算好的标量序列（会直接用 series_mdr）
        """
        frames = list(frames)
        if frames and hasattr(frames[0], "shape"):
            s = self.read_series(frames)
        else:
            s = np.asarray(frames, dtype=np.float64)
        st = series_mdr(s, self.fps, deglitch)
        if st is None:
            raise ValueError("基线数据不足（需要 ≥40 帧）")
        self.z_base = st["z_ref"]
        self.baseline_stats = st
        if self.physical_mm and self.z_base > 0:
            self.scale_k = self.physical_mm / self.z_base
        return st

    def displacement(self, z):
        """相对零点的位移（mm）。"""
        if self.z_base is None:
            raise RuntimeError("请先 calibrate()")
        return (z - self.z_base) * self.scale_k

    def summary(self):
        if self.baseline_stats is None:
            return "未标定"
        st = self.baseline_stats
        return (f"z_ref={self.z_base:.2f}mm  k={self.scale_k:.5f}  "
                f"MDR={st['mdr']:.3f}mm  漂移={st['drift']:.3f}mm  "
                f"n={st['n']}")
