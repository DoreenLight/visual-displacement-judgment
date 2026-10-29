#!/usr/bin/env python3
"""实时预览：浏览器查看 D435i 彩色 + 深度，用于调整相机与被测目标位置。

用法（需要 sudo）
-----------------
sudo -E /opt/anaconda3/envs/realsense/bin/python3 rs_live_view.py
# 浏览器打开 http://127.0.0.1:8899/

画面布局
--------
左：彩色 (640x480)      右：深度伪彩色 (640x480)，黑=无读数
两图都画有：坐标网格、ROI 框、中心十字
底部状态条：ROI 中位深度、有效率、帧率

页面端点
--------
/            交互页面
/stream      MJPEG 视频流
/frame.jpg   单帧快照（便于脚本抓取）
/stats       JSON 状态

参数
----
--port        监听端口，默认 8899
--temporal    开 temporal 滤波（画面更平滑）
--lock-exposure  锁定曝光（AE 稳定后）
--roi x,y,w,h 指定 ROI，默认画面中心 120x120
--grid        画坐标网格（默认开）
--w h         深度分辨率，默认 848x480
"""

import argparse
import json
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import macos_usb_fix
    macos_usb_fix.ensure_ready(verbose=False)
except Exception:
    pass

try:
    import cv2
except ImportError:
    sys.exit("缺少 opencv，请用 conda 环境 realsense 的 python 运行。")

try:
    import pyrealsense2 as rs
except ImportError:
    sys.exit("缺少 pyrealsense2。")

try:
    from scipy import ndimage as ndi
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ------------------------------------------------------------------ 共享状态
LOCK = threading.Lock()
STATE = {
    "jpeg": None,          # 合成画面 JPEG
    "jpeg_color": None,    # 彩色单帧
    "roi": [260, 180, 120, 120],   # 可由页面点击修改
    "stats": {"fps": 0.0, "roi_median": 0, "valid_pct": 0.0,
              "center": 0, "n_frames": 0, "error": None},
    "stop": False,
}


def draw_grid(img, step=100):
    """画坐标网格与刻度，便于指定 ROI 坐标。"""
    h, w = img.shape[:2]
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), (70, 70, 70), 1)
        cv2.putText(img, str(x), (x + 3, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 170, 170), 1,
                    cv2.LINE_AA)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), (70, 70, 70), 1)
        if y > 0:
            cv2.putText(img, str(y), (3, y + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 170, 170), 1,
                        cv2.LINE_AA)


def colorize_depth(d, vmin, vmax):
    """深度 → 伪彩色 BGR。无效像素（0/65535）为黑。"""
    invalid = (d == 0) | (d == 65535)
    if vmax <= vmin:
        vmax = vmin + 1
    n = np.clip((d.astype(np.float32) - vmin) / (vmax - vmin), 0, 1)
    n[invalid] = 0
    g = (n * 255).astype(np.uint8)
    img = cv2.applyColorMap(g, cv2.COLORMAP_JET)
    img[invalid] = 0
    return img


def depth_segment(d, thresh=30):
    """深度感知连通域分割。

    关键区别：只在【空间相邻 且 深度相差 < thresh】的像素间建立连通。

    为什么不能只用空间连通：手边缘与它后面的背景在图像上紧邻，
    纯空间连通会把它们并成一块（实测该块的深度 σ 可达 1171mm），
    用这种「假连续块」逐帧平均会产生巨大假噪声。

    返回 (lab_img, info)：lab_img 中 -1 表示无效像素；
    info: {label: dict(n, med, sigma, span)}
    """
    h, w = d.shape
    valid = (d > 0) & (d < 65535)
    n = int(valid.sum())
    lab_img = np.full((h, w), -1, np.int32)
    if n < 200:
        return lab_img, {}

    di = d.astype(np.int32)
    idx = np.full((h, w), -1, np.int64)
    idx[valid] = np.arange(n)

    # 水平/垂直邻接（仅当深度相近时才算连通）
    mh = valid[:, :-1] & valid[:, 1:] & (np.abs(di[:, :-1] - di[:, 1:]) < thresh)
    mv = valid[:-1, :] & valid[1:, :] & (np.abs(di[:-1, :] - di[1:, :]) < thresh)
    rh, chh = np.where(mh)
    rv, cvv = np.where(mv)

    if rh.size + rv.size == 0:
        lab_img[valid] = 0
        vals = d[valid].astype(np.float64)
        return lab_img, {0: dict(n=n, med=float(np.median(vals)),
                                 sigma=float(np.std(vals)),
                                 span=float(np.ptp(vals)))}

    rows = np.concatenate([idx[rh, chh], idx[rv, cvv]])
    cols = np.concatenate([idx[rh, chh + 1], idx[rv + 1, cvv]])
    g = coo_matrix((np.ones(rows.size, np.int8), (rows, cols)), shape=(n, n))
    ncomp, labels = connected_components(g, directed=False)
    lab_img[valid] = labels

    dv = d[valid].astype(np.float64)
    cnt = np.bincount(labels, minlength=ncomp).astype(np.float64)
    s1 = np.bincount(labels, weights=dv, minlength=ncomp)
    s2 = np.bincount(labels, weights=dv * dv, minlength=ncomp)
    mean = s1 / np.maximum(cnt, 1)
    var = np.maximum(s2 / np.maximum(cnt, 1) - mean * mean, 0)

    info = {}
    for k in range(ncomp):
        c = int(cnt[k])
        if c < 50:
            continue
        info[k] = dict(n=c, med=float(mean[k]),
                       sigma=float(np.sqrt(var[k])), span=None)
    return lab_img, info


def best_window(mask, max_side):
    """在二值 mask 中找「内部像素最多」的方形窗口。

    比用质心可靠：细长/环状块的质心可能落在块外。
    用积分图实现，复杂度与图像面积线性相关。

    返回 (x0, y0, side, count)；无有效窗口时返回 None。
    """
    n = int(mask.sum())
    if n < 100:
        return None
    # ROI 边长自适应：块小就用小 ROI，不硬撑到 max_side
    side = int(min(max_side, max(40, (n ** 0.5) * 0.8)))
    side = max(20, side)
    H, W = mask.shape
    if side > H or side > W:
        side = min(H, W)
    ii = cv2.integral(mask.astype(np.uint8))
    S = (ii[side:, side:] - ii[:-side, side:]
         - ii[side:, :-side] + ii[:-side, :-side])
    if S.size == 0:
        return None
    k = int(np.argmax(S))
    y0, x0 = np.unravel_index(k, S.shape)
    return int(x0), int(y0), int(side), int(S[y0, x0])


def capture_loop(cfg):
    """采集线程：持续更新 STATE。"""
    try:
        pipe = rs.pipeline()
        conf = rs.config()
        conf.enable_stream(rs.stream.depth, cfg["w"], cfg["h"],
                           rs.format.z16, cfg["fps"])
        conf.enable_stream(rs.stream.color, cfg["cw"], cfg["ch"],
                           rs.format.bgr8, cfg["fps"])
        profile = pipe.start(conf)
        ds = profile.get_device().first_depth_sensor()

        if cfg["lock_exposure"]:
            # 正确做法：先让 AE 稳定，再读值，再关闭 AE 并写回
            time.sleep(3.0)
            try:
                e = ds.get_option(rs.option.exposure)
                g = ds.get_option(rs.option.gain)
                ds.set_option(rs.option.enable_auto_exposure, 0)
                time.sleep(0.3)
                ds.set_option(rs.option.exposure, e)
                ds.set_option(rs.option.gain, g)
                print(f"曝光已锁定: exposure={e:.0f} gain={g:.0f}")
            except Exception as ex:
                print(f"锁定曝光失败: {ex}")

        al = rs.align(rs.stream.color)
        tf = None
        if cfg["temporal"]:
            tf = rs.temporal_filter()
            tf.set_option(rs.option.filter_smooth_alpha, 0.4)
            tf.set_option(rs.option.filter_smooth_delta, 20)

        # 预热
        for _ in range(40):
            pipe.wait_for_frames(timeout_ms=10000)

        x, y, w, h = cfg["roi"]
        nframes = 0
        seg_counter = 0
        seg_lab = None
        seg_info = {}
        seg_sigma = 0.0
        seg_px = 0
        seg_span = 0.0
        seg_global = []
        t0 = time.time()
        fps_ema = 0.0
        vmin, vmax = 200, 3000

        while not STATE["stop"]:
            # ROI 可被页面实时修改
            with LOCK:
                x, y, w, h = STATE["roi"]
            fs = pipe.wait_for_frames(timeout_ms=10000)
            fs = al.process(fs)
            df = fs.get_depth_frame()
            if tf is not None:
                df = tf.process(df)
            cf = fs.get_color_frame()
            if not df or not cf:
                continue

            d = np.asanyarray(df.get_data())
            color = np.asanyarray(cf.get_data()).copy()

            # 自适应深度范围（分位数，避免被离群值拉伸）
            v = d[(d > 0) & (d < 65535)]
            if v.size > 500:
                lo = float(np.percentile(v, 2))
                hi = float(np.percentile(v, 98))
                vmin = int(max(0, lo))
                vmax = int(max(hi, lo + 200))

            depth_vis = colorize_depth(d, vmin, vmax)

            # ROI 统计
            sub = d[y:y + h, x:x + w]
            sv = sub[(sub > 0) & (sub < 65535)]
            roi_med = int(np.median(sv)) if sv.size else 0
            roi_valid = 100.0 * sv.size / max(1, sub.size)
            cy, cx = cfg["h"] // 2 * 0, 0
            ch, cw = color.shape[:2]
            cc = d[ch // 2, cw // 2] if (ch // 2 < d.shape[0] and cw // 2 < d.shape[1]) else 0

            # ── 深度感知分割：每 15 帧重算一次（~2Hz）──
            # 不用纯空间连通：手边缘与背景相邻会造成假连续块。
            seg_counter += 1
            if HAS_SCIPY and (seg_counter % 15 == 1 or seg_lab is None):
                seg_lab, seg_info = depth_segment(d, 30)
                # 全局候选：按面积排序，标出各块中位深度，供用户选 ROI
                cand = sorted(seg_info.items(), key=lambda kv: -kv[1]["n"])[:6]
                seg_global = []
                for lb, v in cand:
                    m = seg_lab == lb
                    ys, xs = np.where(m)
                    if ys.size == 0:
                        continue
                    seg_global.append(dict(
                        n=v["n"], med=round(v["med"]),
                        sigma=round(v["sigma"], 1), label=int(lb),
                        cx=int(np.median(xs)), cy=int(np.median(ys))))
                with LOCK:
                    STATE["block_cands"] = seg_global
                    STATE["seg_lab"] = seg_lab

            # ROI 中心所在的深度一致块（指标均限制在 ROI 范围内计算）
            if seg_lab is not None:
                cy0, cx0 = y + h // 2, x + w // 2
                cy0 = min(max(cy0, 0), seg_lab.shape[0] - 1)
                cx0 = min(max(cx0, 0), seg_lab.shape[1] - 1)
                lab = int(seg_lab[cy0, cx0])
                if lab >= 0:
                    # 只看 ROI 内的部分，避免整块远大于 ROI 时指标失真
                    roi_win = seg_lab[y:y + h, x:x + w]
                    d_win = d[y:y + h, x:x + w]
                    m_win = (roi_win == lab) & (d_win > 0) & (d_win < 65535)
                    seg_px = int(m_win.sum())
                    if seg_px >= 20:
                        vals = d_win[m_win].astype(np.float32)
                        seg_sigma = float(np.std(vals))
                        seg_span = float(np.ptp(vals))
                    else:
                        seg_sigma = seg_span = 0.0
                else:
                    seg_px = seg_sigma = seg_span = 0

            nframes += 1
            el = time.time() - t0
            if el >= 0.5:
                inst = nframes / el
                fps_ema = inst if fps_ema == 0 else 0.7 * fps_ema + 0.3 * inst
                nframes = 0
                t0 = time.time()

            # 绘制
            for img in (color, depth_vis):
                if cfg["grid"]:
                    draw_grid(img)
                # ROI 框
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.drawMarker(img, (cw // 2, ch // 2), (255, 255, 255),
                               cv2.MARKER_CROSS, 22, 2)

            cv2.putText(color, "COLOR", (8, ch - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(depth_vis, "DEPTH", (8, ch - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            # 连续块质量提示条（基于 ROI 内统计）
            ratio = 100.0 * seg_px / max(1, w * h)
            good = (ratio > 70 and seg_span < 60)
            mid = (ratio > 40 and seg_span < 200)
            cc_col = (0, 220, 0) if good else ((0, 190, 255) if mid else (0, 0, 255))
            cv2.putText(depth_vis,
                        f"ROIblk {ratio:.0f}% span={seg_span:.0f}mm s={seg_sigma:.1f}",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cc_col, 2,
                        cv2.LINE_AA)
            if not good:
                cv2.putText(depth_vis, "ROI NOT on single surface",
                            (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2, cv2.LINE_AA)

            combo = np.hstack([color, depth_vis])

            # 状态条
            bar = np.zeros((34, combo.shape[1], 3), np.uint8)
            msg = (f"ROI({x},{y},{w},{h}) median={roi_med}mm  "
                   f"valid={roi_valid:.0f}%   center={int(cc)}mm  "
                   f"range={vmin}-{vmax}mm  fps={fps_ema:.1f}")
            cv2.putText(bar, msg, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            out = np.vstack([combo, bar])

            ok_c, buf_c = cv2.imencode(".jpg", combo,
                                       [cv2.IMWRITE_JPEG_QUALITY, 78])
            ok_o, buf_o = cv2.imencode(".jpg", out,
                                       [cv2.IMWRITE_JPEG_QUALITY, 78])
            with LOCK:
                if ok_o:
                    STATE["jpeg"] = buf_o.tobytes()
                if ok_c:
                    STATE["jpeg_color"] = buf_c.tobytes()
                STATE["stats"].update({
                    "fps": round(fps_ema, 1),
                    "roi_median": roi_med,
                    "valid_pct": round(roi_valid, 1),
                    "center": int(cc),
                    "n_frames": STATE["stats"]["n_frames"] + 1,
                    "error": None,
                    "range": [vmin, vmax],
                    "roi": [x, y, w, h],
                    "blk_pct": round(ratio, 1),
                    "blk_px": seg_px,
                    "blk_sigma": round(seg_sigma, 2),
                    "blk_span": round(seg_span, 1),
                    "blk_ok": bool(good),
                    "top_blocks": seg_global,
                })
    except Exception as e:
        with LOCK:
            STATE["stats"]["error"] = str(e)
        print(f"采集线程异常: {e}", file=sys.stderr)
    finally:
        try:
            pipe.stop()
        except Exception:
            pass


PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>D435i 实时预览</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#111; color:#ddd;
         font:13px/1.5 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif; }
  header { padding:10px 14px; background:#1b1b1b; border-bottom:1px solid #2a2a2a;
           display:flex; align-items:center; gap:16px; flex-wrap:wrap; }
  h1 { font-size:15px; margin:0; font-weight:600; }
  .pill { padding:3px 9px; border-radius:99px; background:#222; border:1px solid #333;
          font-variant-numeric:tabular-nums; }
  .ok { color:#7dd87d; } .warn { color:#e8b84b; } .bad { color:#e8624b; }
  main { padding:12px; }
  img { width:100%; height:auto; display:block; border:1px solid #2a2a2a; border-radius:6px; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  .note { color:#888; font-size:12px; margin-top:10px; }
  code { background:#222; padding:1px 5px; border-radius:4px; }
</style>
</head>
<body>
<header>
  <h1>D435i 实时预览</h1>
  <span class="pill">ROI 中位: <b id="rm">—</b> mm</span>
  <span class="pill">有效率: <b id="vr">—</b> %</span>
  <span class="pill">中心: <b id="ce">—</b> mm</span>
  <span class="pill">fps: <b id="fp">—</b></span>
  <span class="pill" id="errbox" style="display:none"></span>
</header>
<main>
  <img id="v" src="/stream" alt="实时画面" style="cursor:crosshair">
  <div class="note">
    <b>在图上点击</b>即可把 ROI 移到该处（按住 Shift 点击可改小 ROI）。<br>
    左=彩色，右=深度伪彩色（黑=无读数）。绿框=ROI，白十字=画面中心。
  </div>
  <div class="grid2" style="margin-top:12px">
    <div class="pill">ROI: <b id="roi">—</b></div>
    <div class="pill">深度范围: <b id="rng">—</b></div>
    <div class="pill">块占比: <b id="cont">—</b></div>
    <div class="pill">块深度跨度: <b id="csig">—</b> mm</div>
    <div class="pill">块内σ: <b id="csig2">—</b> mm</div>
  </div>
  <div style="margin-top:10px; display:flex; gap:8px; flex-wrap:wrap">
    <button id="btnbest" style="cursor:pointer; padding:6px 12px">跳到最平整块</button>
    <button id="btnb900" style="cursor:pointer; padding:6px 12px">跳到 ~900mm</button>
    <button id="btnb400" style="cursor:pointer; padding:6px 12px">跳到 ~400mm</button>
  </div>
  <div class="note" style="margin-top:10px">
    <b>块占比</b>：ROI 中心所在的「深度一致连通块」占 ROI 面积的比例。<br>
    <b>块深度跨度</b>：该块的 (max-min) 深度。>60mm 就说明这块跨了多个物体。
    <b>这两个指标必须同时看</b>：占比高但跨度大 = 假连续块（手边缘+背景）。<br>
    目标：块占比 <b>&gt;70%</b> 且 跨度 <b>&lt;60mm</b>。<br>
    <span id="blocks" style="color:#9ad"></span>
  </div>
</main>
<script>
async function tick(){
  try{
    const r = await fetch('/stats', {cache:'no-store'});
    const s = await r.json();
    document.getElementById('rm').textContent = s.roi_median || '—';
    document.getElementById('vr').textContent = (s.valid_pct ?? '—');
    document.getElementById('ce').textContent = s.center ?? '—';
    document.getElementById('fp').textContent = s.fps ?? '—';
    const vr = s.valid_pct || 0;
    document.getElementById('vr').className = vr > 60 ? 'ok' : (vr > 25 ? 'warn' : 'bad');
    const eb = document.getElementById('errbox');
    if (s.error){ eb.style.display=''; eb.className='pill bad'; eb.textContent='错误: '+s.error; }
    else { eb.style.display='none'; }
    if (s.roi) document.getElementById('roi').textContent = s.roi.join(',');
    if (s.range) document.getElementById('rng').textContent = s.range[0]+'-'+s.range[1]+' mm';
    if (s.blk_pct != null) {
      const el = document.getElementById('cont');
      el.textContent = s.blk_pct + '%';
      el.className = (s.blk_ok) ? 'ok' : (s.blk_pct > 40 ? 'warn' : 'bad');
    }
    if (s.blk_span != null) {
      const e1 = document.getElementById('csig');
      e1.textContent = s.blk_span;
      e1.className = s.blk_span < 60 ? 'ok' : (s.blk_span < 200 ? 'warn' : 'bad');
    }
    if (s.blk_sigma != null) {
      const e2 = document.getElementById('csig2');
      e2.textContent = s.blk_sigma;
      e2.className = s.blk_sigma < 10 ? 'ok' : (s.blk_sigma < 40 ? 'warn' : 'bad');
    }
    if (s.top_blocks && s.top_blocks.length) {
      document.getElementById('blocks').textContent =
        '场景中较大的深度一致块: ' +
        s.top_blocks.map(b => `${b.med}mm(${b.n}px,σ${b.sigma})`).join(' / ');
    }
  }catch(e){}
}
setInterval(tick, 600); tick();

async function jump(target, tol){
  const r = await fetch(`/best?target=${target}&tol=${tol}`, {cache:'no-store'});
  const j = await r.json();
  if (!j.ok) alert('未找到合适的块: ' + (j.msg || j.err || ''));
  tick();
}
document.getElementById('btnbest').onclick = () => jump(0, 99999);
document.getElementById('btnb900').onclick = () => jump(900, 250);
document.getElementById('btnb400').onclick = () => jump(400, 200);

// 点击画面设置 ROI（坐标映射到彩色图）
const img = document.getElementById('v');
img.addEventListener('click', async (ev) => {
  const r = img.getBoundingClientRect();
  const fx = (ev.clientX - r.left) / r.width;
  if (fx > 0.5) return;                            // 点右半边无效
  const cx = Math.round(fx * 2 * 640);
  const cy = Math.round(((ev.clientY - r.top) / r.height) * 480);
  const sz = ev.shiftKey ? 60 : 120;
  await fetch(`/setroi?cx=${cx}&cy=${cy}&s=${sz}`, {cache:'no-store'});
  tick();
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/stats":
            with LOCK:
                body = json.dumps(STATE["stats"]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/setroi"):
            # /setroi?cx=..&cy=..&s=..  把 ROI 居中到指定点
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            try:
                cx = int(q.get("cx", [320])[0])
                cy = int(q.get("cy", [240])[0])
                s = int(q.get("s", [120])[0])
                s = max(20, min(400, s))
                x = max(0, min(640 - s, cx - s // 2))
                y = max(0, min(480 - s, cy - s // 2))
                with LOCK:
                    STATE["roi"] = [x, y, s, s]
                body = json.dumps({"ok": True, "roi": [x, y, s, s]}).encode()
            except Exception as e:
                body = json.dumps({"ok": False, "err": str(e)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/best"):
            # /best?target=900&tol=150  把 ROI 跳到最接近 target 的平整块中心
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            try:
                target = float(q.get("target", [0])[0])
                tol = float(q.get("tol", [200])[0])
                with LOCK:
                    cands = STATE.get("block_cands", [])
                    cur = STATE["roi"]
                pick = None
                for c in cands:
                    if c["sigma"] > 40 or c["n"] < 2000:
                        continue
                    if target > 0 and abs(c["med"] - target) > tol:
                        continue
                    pick = c
                    break
                if pick is None and cands:
                    pick = min(cands, key=lambda c: c["sigma"])
                if pick is None:
                    body = json.dumps({"ok": False, "msg": "没有候选块"}).encode()
                else:
                    # 在该块内找内部像素最多的方形窗口（而非质心）
                    with LOCK:
                        sl = STATE.get("seg_lab")
                        want = cur[2]
                    win = None
                    if sl is not None:
                        win = best_window(sl == pick["label"], want)
                    if win is None:
                        cx, cy = pick["cx"], pick["cy"]
                        s = want
                        x = max(0, min(640 - s, cx - s // 2))
                        y = max(0, min(480 - s, cy - s // 2))
                        cnt = 0
                    else:
                        x, y, s, cnt = win
                    with LOCK:
                        STATE["roi"] = [x, y, s, s]
                    body = json.dumps({
                        "ok": True, "roi": [x, y, s, s],
                        "inside_px": cnt,
                        "inside_pct": round(100.0 * cnt / max(1, s * s), 1),
                        "picked": pick,
                    }).encode()
            except Exception as e:
                body = json.dumps({"ok": False, "err": str(e)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/frame.jpg":
            with LOCK:
                j = STATE["jpeg_color"] or STATE["jpeg"]
            if not j:
                self.send_error(503, "no frame yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(j)))
            self.end_headers()
            self.wfile.write(j)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while not STATE["stop"]:
                    with LOCK:
                        j = STATE["jpeg"]
                    if not j:
                        time.sleep(0.05)
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(j)}\r\n\r\n".encode())
                    self.wfile.write(j)
                    self.wfile.write(b"\r\n")
                    time.sleep(1.0 / 25)   # 显示限速
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--temporal", action="store_true")
    ap.add_argument("--lock-exposure", action="store_true")
    ap.add_argument("--roi", default=None, help="x,y,w,h")
    ap.add_argument("--no-grid", dest="grid", action="store_false", default=True)
    ap.add_argument("--w", type=int, default=848)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--cw", type=int, default=640)
    ap.add_argument("--ch", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    # ROI 基于对齐后的彩色分辨率
    if args.roi:
        roi = tuple(int(v) for v in args.roi.split(","))
        if len(roi) != 4:
            ap.error("--roi 需要 4 个整数 x,y,w,h")
    else:
        w0, h0 = 120, 120
        roi = ((args.cw - w0) // 2, (args.ch - h0) // 2, w0, h0)

    cfg = dict(w=args.w, h=args.h, cw=args.cw, ch=args.ch, fps=args.fps,
               roi=roi, temporal=args.temporal,
               lock_exposure=args.lock_exposure, grid=args.grid)

    t = threading.Thread(target=capture_loop, args=(cfg,), daemon=True)
    t.start()

    # 等第一帧
    for _ in range(200):
        with LOCK:
            if STATE["jpeg"]:
                break
            if STATE["stats"]["error"]:
                print(f"❌ 采集失败: {STATE['stats']['error']}")
                return 1
        time.sleep(0.1)
    else:
        print("❌ 等待首帧超时")
        return 1

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"✅ 实时预览已就绪: http://127.0.0.1:{args.port}/")
    print(f"   ROI = {roi}   temporal={'on' if args.temporal else 'off'}")
    print(f"   (Ctrl-C 停止)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        STATE["stop"] = True
        srv.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
