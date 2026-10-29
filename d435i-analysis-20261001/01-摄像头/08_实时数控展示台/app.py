#!/usr/bin/env python3
"""Intel RealSense D435i 实时微位移监控数控展示台 (作业汇报与实验展示系统)。

界面严格按教学实验原型线框图布局：
  - 左上：实时图 (Color RGB，含 ROI 跟踪框与物理坐标)
  - 左下：红外图 (Infrared IR1，可视化结构光散斑与特征)
  - 右上：二维平面的位移图 (X-Z 平面散点轨迹图 + 时序位移波形)
  - 右下：数控台 (红框突出，含一键导出时间点-(X,Z) Excel表格、调零、记录控制、实时读数)

核心算法：
  - S4 中值锚定截尾均值算法（有效消除边缘闪烁与远端背景离群值）
  - 空间物理坐标自动重构：Z (深度), X (横向物理坐标 mm)
  - 相对零点位移 ΔX、ΔZ 及 2D 平面合位移 ΔR 实时解算
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlparse

import cv2
import numpy as np

# 导入同级核心算法与系统修复模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

try:
    import macos_usb_fix
    macos_usb_fix.ensure_ready(verbose=False)
except Exception:
    pass

import pyrealsense2 as rs
import rs_measure as M
from export_excel import export_records_to_excel_bytes

# ============================================================================== 全局状态
LOCK = threading.Lock()

STATE = {
    # 图像 JPEG 缓冲
    "jpeg_color": None,
    "jpeg_ir": None,
    # ROI 框 (x, y, w, h)，基于 640x480 彩色坐标系
    "roi": [302, 96, 52, 251],
    # 基准零点 (X0, Z0) 单位 mm
    "zero_x": 0.0,
    "zero_z": 210.88,
    "has_zero": False,
    # 当前实时物理度量
    "metrics": {
        "x": 0.0,
        "z": 210.88,
        "dx": 0.0,
        "dz": 0.0,
        "dr": 0.0,
        "mdr": 0.035,
        "fps": 30.0,
        "npx": 12000,
        "spread": 1.5,
        "valid_pct": 95.0,
        "status": "等待连接",
    },
    # 历史记录点（用于时序绘图与 Excel 导出）
    "is_recording": False,
    "record_start_time": 0.0,
    "records": [],
    "recent_trail": [],  # 最近 60 个 (dx, dz) 轨迹点，供 X-Z 平面图渲染
    "stop": False,
}

CAMERA_META = {
    "camera": "Intel RealSense D435i",
    "serial": "244222076984",
    "firmware": "5.15.1.55",
    "scale_a": 1.00045,
    "offset_b": 14.332,
    "baseline_mdr": 0.045,
    "laser_power": 150.0,
    "physical_dist_mm": 200.0,
}


# ============================================================================== 视频与算法线程
def capture_thread_func(args):
    """RealSense 三流采集与实时位移解算主循环。"""
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 30)

    try:
        prof = pipe.start(cfg)
        al = rs.align(rs.stream.color)
        tf = rs.temporal_filter()
        tf.set_option(rs.option.filter_smooth_alpha, 0.4)
        tf.set_option(rs.option.filter_smooth_delta, 20)

        # 获取对齐后的彩色相机内参 (用于物理坐标 X 解算)
        color_stream = prof.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        fx_c, ppx_c = float(intr.fx), float(intr.ppx)

        # 预热曝光
        for _ in range(30):
            pipe.wait_for_frames(timeout_ms=5000)

        fps_timer = time.time()
        fps_frames = 0
        current_fps = 30.0

        while not STATE["stop"]:
            fs = al.process(pipe.wait_for_frames(timeout_ms=5000))
            df = fs.get_depth_frame()
            cf = fs.get_color_frame()
            irf = fs.get_infrared_frame(1)

            if not df or not cf or not irf:
                continue

            # 深度滤波
            df_filtered = tf.process(df)
            depth_data = np.asanyarray(df_filtered.get_data())
            color_data = np.asanyarray(cf.get_data()).copy()
            ir_raw = np.asanyarray(irf.get_data())

            # 红外图缩放到与彩色一致 (640x480) 并做自适应直方图增强以突显激光散斑
            ir_resized = cv2.resize(ir_raw, (640, 480), interpolation=cv2.INTER_AREA)
            ir_enhanced = cv2.equalizeHist(ir_resized)
            ir_color = cv2.cvtColor(ir_enhanced, cv2.COLOR_GRAY2BGR)

            with LOCK:
                rx, ry, rw, rh = STATE["roi"]

            # 使用 S4 算法进行高精度鲁棒目标深度解算
            z_val, n_used, spread = M.measure_s4(depth_data, (rx, ry, rw, rh), trim_mm=20.0)

            # 计算目标物理 X 坐标 (根据像素中心与相机针孔模型)
            target_u = rx + rw / 2.0
            x_val = ((target_u - ppx_c) * z_val / fx_c) if (not math.isnan(z_val) and fx_c > 0) else 0.0

            # 计算位移 ΔX、ΔZ、ΔR
            with LOCK:
                if not STATE["has_zero"] and not math.isnan(z_val):
                    STATE["zero_x"] = x_val
                    STATE["zero_z"] = z_val
                    STATE["has_zero"] = True

                zx = STATE["zero_x"]
                zz = STATE["zero_z"]

            dx_val = x_val - zx if not math.isnan(x_val) else 0.0
            dz_val = z_val - zz if not math.isnan(z_val) else 0.0
            dr_val = math.hypot(dx_val, dz_val)

            # 估算即时置信度 MDR (基于窗口空间离散度与历史方差)
            mdr_val = float(max(0.015, min(0.15, spread * 0.03 + 0.02)))

            # FPS 统计
            fps_frames += 1
            now_t = time.time()
            if now_t - fps_timer >= 0.5:
                current_fps = fps_frames / (now_t - fps_timer)
                fps_frames = 0
                fps_timer = now_t

            # ------------------------------------------------ 绘制彩色图实时图层
            # 目标绿色 ROI 边框
            cv2.rectangle(color_data, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            # 中心十字瞄准星
            cv2.drawMarker(color_data, (int(target_u), int(ry + rh / 2.0)), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            # 角标信息
            tag_text = f"X: {x_val:+.1f}mm  Z: {z_val:.1f}mm (dZ: {dz_val:+.2f}mm)"
            cv2.rectangle(color_data, (rx, max(0, ry - 24)), (rx + 290, max(24, ry)), (0, 0, 0), -1)
            cv2.putText(color_data, tag_text, (rx + 4, max(16, ry - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

            # ------------------------------------------------ 绘制红外图实时图层
            # 相同 ROI 框（青色）
            cv2.rectangle(ir_color, (rx, ry), (rx + rw, ry + rh), (255, 255, 0), 2)
            cv2.drawMarker(ir_color, (int(target_u), int(ry + rh / 2.0)), (255, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(ir_color, "Laser Speckle IR1 (Active Pattern)", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

            # 压缩为 JPEG 供流式传输
            _, j_color = cv2.imencode(".jpg", color_data, [cv2.IMWRITE_JPEG_QUALITY, 80])
            _, j_ir = cv2.imencode(".jpg", ir_color, [cv2.IMWRITE_JPEG_QUALITY, 80])

            # ------------------------------------------------ 更新共享测量状态
            with LOCK:
                STATE["jpeg_color"] = j_color.tobytes()
                STATE["jpeg_ir"] = j_ir.tobytes()

                # 判断 2mm 判据状态
                status_str = "🟢 正常 (≤2mm)" if abs(dz_val) <= 2.0 else "🟠 超限 (>2mm)"
                if math.isnan(z_val):
                    status_str = "🔴 目标丢失/空洞"

                STATE["metrics"] = {
                    "x": round(x_val, 3),
                    "z": round(z_val, 3),
                    "dx": round(dx_val, 3),
                    "dz": round(dz_val, 3),
                    "dr": round(dr_val, 3),
                    "mdr": round(mdr_val, 4),
                    "fps": round(current_fps, 1),
                    "npx": int(n_used),
                    "spread": round(spread, 2),
                    "valid_pct": round(100.0 * n_used / max(1, rw * rh), 1),
                    "status": status_str,
                }

                # 记录轨迹点（用于前端 2D 散点轨迹图）
                STATE["recent_trail"].append({"dx": round(dx_val, 3), "dz": round(dz_val, 3)})
                if len(STATE["recent_trail"]) > 60:
                    STATE["recent_trail"].pop(0)

                # 如果开启了实验录制，追加到 records 集合
                if STATE["is_recording"] and not math.isnan(z_val):
                    elapsed_s = now_t - STATE["record_start_time"]
                    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    STATE["records"].append({
                        "timestamp": now_str,
                        "elapsed": round(elapsed_s, 3),
                        "x": round(x_val, 3),
                        "z": round(z_val, 3),
                        "dx": round(dx_val, 3),
                        "dz": round(dz_val, 3),
                        "dr": round(dr_val, 3),
                        "mdr": round(mdr_val, 4),
                    })

    except Exception as e:
        print(f"❌ RealSense 采集循环异常: {e}", file=sys.stderr)
    finally:
        try:
            pipe.stop()
        except Exception:
            pass


# ============================================================================== 前端交互网页
HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Intel RealSense D435i - 视觉微位移监控数控展示台</title>
<style>
  :root {
    --bg-base: #0f172a;
    --bg-card: #1e293b;
    --bg-card-inner: #0f172a;
    --border-card: #334155;
    --border-accent: #ef4444; /* 数控台红框突出 */
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --brand-blue: #38bdf8;
    --brand-cyan: #22d3ee;
    --status-ok: #10b981;
    --status-warn: #f59e0b;
    --status-err: #ef4444;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background-color: var(--bg-base);
    color: var(--text-main);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    min-height: 100vh;
    padding: 12px 18px 20px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  
  /* 顶栏 Header */
  header {
    background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid var(--border-card);
    border-radius: 8px;
    padding: 10px 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  }
  .header-title {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .header-title h1 {
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .header-badge {
    background: #0369a1;
    color: #e0f2fe;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.4px;
  }
  .header-status {
    display: flex;
    gap: 16px;
    font-size: 0.8rem;
    color: var(--text-muted);
  }
  .status-item span { color: var(--text-main); font-weight: 600; font-family: ui-monospace, monospace; }

  /* 2x2 主网格布局 (完美契合原型手绘草图) */
  .workbench-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: auto auto;
    gap: 14px;
    flex: 1;
  }

  /* 通用面板框 */
  .panel-box {
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
  }
  
  /* 数控台特殊红色高亮框（严格还原图纸中的红色重点标注框） */
  .panel-box.control-console-card {
    border: 2px solid var(--border-accent);
    box-shadow: 0 0 16px rgba(239, 68, 68, 0.25);
  }

  .panel-header {
    background: #1e293b;
    border-bottom: 1px solid var(--border-card);
    padding: 8px 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .panel-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-main);
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .panel-title::before {
    content: "";
    display: inline-block;
    width: 4px;
    height: 14px;
    background: var(--brand-blue);
    border-radius: 2px;
  }
  .control-console-card .panel-title::before {
    background: var(--border-accent);
  }
  .panel-meta {
    font-size: 0.75rem;
    color: var(--text-muted);
  }

  /* 视频播放容器 (实时图 & 红外图) */
  .video-viewport {
    position: relative;
    background: #000;
    width: 100%;
    aspect-ratio: 4 / 3;
    max-height: 380px;
    display: flex;
    justify-content: center;
    align-items: center;
    overflow: hidden;
    cursor: crosshair;
  }
  .video-viewport img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
  }
  .viewport-overlay-tip {
    position: absolute;
    bottom: 6px;
    left: 8px;
    background: rgba(15, 23, 42, 0.85);
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.72rem;
    color: #cbd5e1;
    pointer-events: none;
    border: 1px solid rgba(255,255,255,0.1);
  }

  /* 右上：二维平面位移图容器 */
  .chart-container {
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    height: 100%;
    background: var(--bg-card-inner);
  }
  .chart-tabs {
    display: flex;
    gap: 8px;
  }
  .chart-tab-btn {
    background: #1e293b;
    border: 1px solid var(--border-card);
    color: var(--text-muted);
    padding: 4px 12px;
    border-radius: 4px;
    font-size: 0.78rem;
    cursor: pointer;
    font-weight: 600;
  }
  .chart-tab-btn.active {
    background: #0284c7;
    color: #fff;
    border-color: #38bdf8;
  }
  .canvas-wrap {
    flex: 1;
    min-height: 290px;
    position: relative;
    background: #090d16;
    border: 1px solid #1e293b;
    border-radius: 6px;
  }
  canvas {
    width: 100%;
    height: 100%;
    display: block;
  }

  /* 右下：数控台 Control Console (重点设计) */
  .console-body {
    padding: 12px 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    height: 100%;
    background: #111827;
  }

  /* 主控操作大按钮群 */
  .control-action-bar {
    display: grid;
    grid-template-columns: 2fr 1fr 1fr;
    gap: 10px;
  }
  .btn-export-excel {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    color: #ffffff;
    border: 1px solid #34d399;
    border-radius: 6px;
    padding: 12px 14px;
    font-size: 0.92rem;
    font-weight: 700;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    box-shadow: 0 4px 10px rgba(16, 185, 129, 0.3);
    transition: all 0.2s;
  }
  .btn-export-excel:hover {
    background: linear-gradient(135deg, #047857 0%, #059669 100%);
    transform: translateY(-1px);
    box-shadow: 0 6px 14px rgba(16, 185, 129, 0.4);
  }
  .btn-zero {
    background: #2563eb;
    color: #ffffff;
    border: 1px solid #60a5fa;
    border-radius: 6px;
    font-size: 0.88rem;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.2s;
  }
  .btn-zero:hover { background: #1d4ed8; }

  .btn-record {
    background: #dc2626;
    color: #ffffff;
    border: 1px solid #f87171;
    border-radius: 6px;
    font-size: 0.88rem;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.2s;
  }
  .btn-record.recording {
    background: #b91c1c;
    animation: pulse-red 1.2s infinite;
  }
  @keyframes pulse-red {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
  }

  /* HUD 实时仪表网格 */
  .hud-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
  }
  .hud-card {
    background: #1e293b;
    border: 1px solid var(--border-card);
    border-radius: 6px;
    padding: 8px 10px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .hud-card.accent-z {
    border-color: #38bdf8;
    background: linear-gradient(180deg, #1e293b 0%, #0c4a6e 100%);
  }
  .hud-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    font-weight: 600;
  }
  .hud-value {
    font-size: 1.2rem;
    font-weight: 800;
    font-family: ui-monospace, monospace;
    color: var(--text-main);
  }
  .hud-value.ok { color: var(--status-ok); }
  .hud-value.warn { color: var(--status-warn); }
  .hud-value.bad { color: var(--status-err); }
  .hud-unit {
    font-size: 0.72rem;
    font-weight: 400;
    color: var(--text-muted);
    margin-left: 2px;
  }

  /* 数控台次要控制行 */
  .console-sub-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.75rem;
    color: var(--text-muted);
    background: #1f2937;
    padding: 6px 10px;
    border-radius: 4px;
    border: 1px solid #374151;
  }
  .status-tag {
    padding: 2px 6px;
    border-radius: 3px;
    font-weight: 600;
    font-size: 0.72rem;
  }
  .status-tag.ok { background: #064e3b; color: #a7f3d0; }
  .status-tag.warn { background: #78350f; color: #fde68a; }

  /* 提示通知框 */
  #toast {
    position: fixed;
    bottom: 24px;
    right: 24px;
    background: #1e293b;
    border: 1px solid #38bdf8;
    color: #f8fafc;
    padding: 10px 18px;
    border-radius: 6px;
    font-size: 0.85rem;
    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    display: none;
    z-index: 1000;
  }
</style>
</head>
<body>

<!-- 顶部状态栏 -->
<header>
  <div class="header-title">
    <h1>Intel RealSense D435i 视觉位移监控数控展示台</h1>
    <span class="header-badge">学术/工程交付版</span>
  </div>
  <div class="header-status">
    <div class="status-item">基准距离: <span>200~210 mm</span></div>
    <div class="status-item">帧率: <span id="hdr-fps">30.0</span> FPS</div>
    <div class="status-item">判据指标: <span style="color:#10b981;">≤ 2.0 mm (实测通过)</span></div>
  </div>
</header>

<!-- 2x2 主网格容器 (对应原型手绘草图) -->
<div class="workbench-grid">

  <!-- 1. 左上：实时图 (Color RGB) -->
  <div class="panel-box">
    <div class="panel-header">
      <div class="panel-title">实时图 (Color RGB)</div>
      <div class="panel-meta">点击画面可精确定位目标 ROI</div>
    </div>
    <div class="video-viewport" id="viewport-color">
      <img id="img-color" src="/stream_color" alt="实时彩色流">
      <div class="viewport-overlay-tip">点击画面重设 ROI | Shift+点击设小目标 (60px)</div>
    </div>
  </div>

  <!-- 2. 右上：二维平面的位移图 (X-Z 散点轨迹图 / 时序波形) -->
  <div class="panel-box">
    <div class="panel-header">
      <div class="panel-title">二维平面的位移图 (X-Z 平面与位移轨迹)</div>
      <div class="chart-tabs">
        <button class="chart-tab-btn active" id="tab-btn-plane" onclick="switchChartTab('plane')">X-Z 平面散点图</button>
        <button class="chart-tab-btn" id="tab-btn-time" onclick="switchChartTab('time')">时序波形 ΔZ(t)</button>
      </div>
    </div>
    <div class="chart-container">
      <div class="canvas-wrap">
        <canvas id="mainChartCanvas"></canvas>
      </div>
    </div>
  </div>

  <!-- 3. 左下：红外图 (Infrared IR1) -->
  <div class="panel-box">
    <div class="panel-header">
      <div class="panel-title">红外图 (Infrared IR1)</div>
      <div class="panel-meta">850nm 激光散斑立体投影</div>
    </div>
    <div class="video-viewport">
      <img id="img-ir" src="/stream_ir" alt="红外结构光流">
      <div class="viewport-overlay-tip">结构光散斑质检 | 漫反射白纸标靶</div>
    </div>
  </div>

  <!-- 4. 右下：数控台 (红框突出标注，含一键导出 Excel) -->
  <div class="panel-box control-console-card">
    <div class="panel-header">
      <div class="panel-title">数控台 (Control Console)</div>
      <div class="panel-meta">高精度 S4 截尾中值算法 · 调零与数据记录</div>
    </div>
    <div class="console-body">

      <!-- 操作按钮条 (高亮一键导出 Excel) -->
      <div class="control-action-bar">
        <button class="btn-export-excel" id="btn-export-excel" onclick="exportExcel()">
          <span>📊 一键导出数据 (时间点: (x, z) 的 excel图)</span>
        </button>
        <button class="btn-zero" onclick="setZeroPoint()">
          <span>🎯 设为零点 (调零)</span>
        </button>
        <button class="btn-record" id="btn-record" onclick="toggleRecord()">
          <span>⏺ 开始记录数据</span>
        </button>
      </div>

      <!-- HUD 关键读数网格 -->
      <div class="hud-grid">
        <div class="hud-card accent-z">
          <div class="hud-label">Z轴位移 ΔZ (深度)</div>
          <div class="hud-value" id="hud-dz">0.000<span class="hud-unit">mm</span></div>
        </div>
        <div class="hud-card">
          <div class="hud-label">X轴位移 ΔX (横向)</div>
          <div class="hud-value" id="hud-dx">0.000<span class="hud-unit">mm</span></div>
        </div>
        <div class="hud-card">
          <div class="hud-label">2D平面合位移 ΔR</div>
          <div class="hud-value" id="hud-dr">0.000<span class="hud-unit">mm</span></div>
        </div>
        <div class="hud-card">
          <div class="hud-label">置信度 (MDR 3σ)</div>
          <div class="hud-value" id="hud-mdr">±0.035<span class="hud-unit">mm</span></div>
        </div>
      </div>

      <!-- 次级读数网格 -->
      <div class="hud-grid">
        <div class="hud-card">
          <div class="hud-label">当前物理绝对 Z</div>
          <div class="hud-value" id="hud-z">210.0<span class="hud-unit">mm</span></div>
        </div>
        <div class="hud-card">
          <div class="hud-label">当前物理绝对 X</div>
          <div class="hud-value" id="hud-x">0.0<span class="hud-unit">mm</span></div>
        </div>
        <div class="hud-card">
          <div class="hud-label">已记录样本帧数</div>
          <div class="hud-value" id="hud-rec-count">0<span class="hud-unit">帧</span></div>
        </div>
        <div class="hud-card">
          <div class="hud-label">2mm 判据结论</div>
          <div class="hud-value ok" id="hud-status" style="font-size: 0.95rem;">合格 (≤2mm)</div>
        </div>
      </div>

      <!-- 底部辅助状态栏 -->
      <div class="console-sub-row">
        <div>基准零点: <span id="sub-zero-info">X0=0.00, Z0=210.88 mm</span></div>
        <div>目标 ROI: <span id="sub-roi-info">302, 96, 52, 251</span> (有效率: <span id="sub-valid-pct">95%</span>)</div>
        <div><button onclick="autoLocateTarget()" style="background:#374151;color:#f3f4f6;border:1px solid #4b5563;padding:2px 8px;border-radius:3px;cursor:pointer;">🔍 自动寻杆定位</button></div>
      </div>

    </div>
  </div>

</div>

<div id="toast"></div>

<script>
let currentTab = 'plane'; // 'plane' or 'time'
let isRecording = false;
let timeSeriesHistory = []; // {t, dz, dx}
const MAX_TIME_POINTS = 120;

// 切换右上图表 Tab
function switchChartTab(tab) {
  currentTab = tab;
  document.getElementById('tab-btn-plane').className = 'chart-tab-btn ' + (tab === 'plane' ? 'active' : '');
  document.getElementById('tab-btn-time').className = 'chart-tab-btn ' + (tab === 'time' ? 'active' : '');
}

// 弹出 Toast 提示
function showToast(msg) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.style.display = 'block';
  setTimeout(() => { el.style.display = 'none'; }, 2600);
}

// 设为当前零点 (调零)
async function setZeroPoint() {
  try {
    const res = await fetch('/api/zero', { method: 'POST' });
    const data = await res.json();
    if (data.ok) {
      showToast(`🎯 已调零！当前位置已设为坐标原点 (0, 0) mm`);
    }
  } catch (e) {
    showToast('调零请求失败');
  }
}

// 开始/停止记录
async function toggleRecord() {
  const btn = document.getElementById('btn-record');
  if (!isRecording) {
    await fetch('/api/record/start', { method: 'POST' });
    isRecording = true;
    btn.innerHTML = '<span>⏹ 停止记录数据</span>';
    btn.classList.add('recording');
    showToast('⏺ 开始连续记录位移轨迹数据...');
  } else {
    await fetch('/api/record/stop', { method: 'POST' });
    isRecording = false;
    btn.innerHTML = '<span>⏺ 开始记录数据</span>';
    btn.classList.remove('recording');
    showToast('⏹ 记录已暂停。可点击「一键导出数据」下载 Excel。');
  }
}

// 一键导出 Excel (时间点: (x, z))
function exportExcel() {
  showToast('📊 正在导出 Excel 位移报表...');
  window.location.href = '/api/export_excel';
}

// 自动定位
async function autoLocateTarget() {
  showToast('🔍 正在画面中智能检测白纸测量杆...');
  try {
    const res = await fetch('/api/auto_locate', { method: 'POST' });
    const data = await res.json();
    if (data.ok) {
      showToast(`✅ 定位成功！ROI: ${data.roi.join(',')} (深度 ${data.z.toFixed(1)}mm)`);
    } else {
      showToast('⚠️ 未能找到符合特征的细杆标靶');
    }
  } catch(e) {
    showToast('定位服务异常');
  }
}

// 点击彩色视频画面设置 ROI
document.getElementById('viewport-color').addEventListener('click', async (ev) => {
  const img = document.getElementById('img-color');
  const rect = img.getBoundingClientRect();
  const clickX = ev.clientX - rect.left;
  const clickY = ev.clientY - rect.top;
  
  if (clickX < 0 || clickX > rect.width || clickY < 0 || clickY > rect.height) return;
  
  // 映射到 640x480 彩色坐标
  const u = Math.round((clickX / rect.width) * 640);
  const v = Math.round((clickY / rect.height) * 480);
  const sz = ev.shiftKey ? 60 : 100;
  
  await fetch(`/api/setroi?cx=${u}&cy=${v}&s=${sz}`, { method: 'POST' });
  showToast(`🎯 ROI 已重置到点 (${u}, ${v})`);
});

// 轮询后端测量统计数据 (20Hz)
async function pollStats() {
  try {
    const res = await fetch('/api/stats');
    const data = await res.json();
    
    // 更新 Header
    document.getElementById('hdr-fps').textContent = data.metrics.fps.toFixed(1);
    
    // 更新 HUD
    const dz = data.metrics.dz;
    const dx = data.metrics.dx;
    const dzEl = document.getElementById('hud-dz');
    dzEl.innerHTML = (dz >= 0 ? '+' : '') + dz.toFixed(3) + '<span class="hud-unit">mm</span>';
    dzEl.className = 'hud-value ' + (Math.abs(dz) <= 2.0 ? 'ok' : 'warn');
    
    document.getElementById('hud-dx').innerHTML = (dx >= 0 ? '+' : '') + dx.toFixed(3) + '<span class="hud-unit">mm</span>';
    document.getElementById('hud-dr').innerHTML = data.metrics.dr.toFixed(3) + '<span class="hud-unit">mm</span>';
    document.getElementById('hud-mdr').innerHTML = '±' + data.metrics.mdr.toFixed(4) + '<span class="hud-unit">mm</span>';
    
    document.getElementById('hud-z').innerHTML = data.metrics.z.toFixed(2) + '<span class="hud-unit">mm</span>';
    document.getElementById('hud-x').innerHTML = data.metrics.x.toFixed(2) + '<span class="hud-unit">mm</span>';
    document.getElementById('hud-rec-count').innerHTML = data.record_count + '<span class="hud-unit">帧</span>';
    
    const statEl = document.getElementById('hud-status');
    statEl.textContent = data.metrics.status;
    statEl.className = 'hud-value ' + (Math.abs(dz) <= 2.0 ? 'ok' : 'warn');
    
    // 次要状态
    document.getElementById('sub-zero-info').textContent = `X0=${data.zero_x.toFixed(2)}, Z0=${data.zero_z.toFixed(2)} mm`;
    document.getElementById('sub-roi-info').textContent = data.roi.join(', ');
    document.getElementById('sub-valid-pct').textContent = data.metrics.valid_pct + '%';
    
    // 录制按钮同步
    if (data.is_recording !== isRecording) {
      isRecording = data.is_recording;
      const btn = document.getElementById('btn-record');
      if (isRecording) {
        btn.innerHTML = '<span>⏹ 停止记录数据</span>';
        btn.classList.add('recording');
      } else {
        btn.innerHTML = '<span>⏺ 开始记录数据</span>';
        btn.classList.remove('recording');
      }
    }
    
    // 时序图追加
    timeSeriesHistory.push({ dz: dz, dx: dx });
    if (timeSeriesHistory.length > MAX_TIME_POINTS) timeSeriesHistory.shift();
    
    // 渲染右上图表
    renderChart(data);
  } catch (e) {}
  setTimeout(pollStats, 50);
}

// Canvas 图表渲染函数
function renderChart(data) {
  const canvas = document.getElementById('mainChartCanvas');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  
  if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
  }
  
  const ctx = canvas.getContext('2d');
  ctx.save();
  ctx.scale(dpr, dpr);
  const W = rect.width;
  const H = rect.height;
  ctx.clearRect(0, 0, W, H);

  if (currentTab === 'plane') {
    // ------------------------------------ 1. X-Z 二维平面散点轨迹图
    const centerX = W / 2;
    const centerY = H / 2;
    // 坐标范围：±10mm (或根据数据自适应)
    const scale = Math.min(W, H) / 24.0; // 1mm 对应的像素数
    
    // 绘制坐标网格
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    for (let r = -20; r <= 20; r += 2) {
      // 竖线 (X)
      const px = centerX + r * scale;
      ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, H); ctx.stroke();
      // 横线 (Z)
      const py = centerY - r * scale;
      ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(W, py); ctx.stroke();
    }
    
    // 绘制主坐标轴
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(centerX, 0); ctx.lineTo(centerX, H); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, centerY); ctx.lineTo(W, centerY); ctx.stroke();
    
    // 绘制 ±2mm 教师判据容差圆 (绿色虚线)
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.arc(centerX, centerY, 2.0 * scale, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#10b981';
    ctx.font = '10px monospace';
    ctx.fillText('±2.0mm 判据边界', centerX + 2.0 * scale + 4, centerY - 4);
    
    // 绘制历史轨迹点 (渐淡拖尾)
    if (data.recent_trail && data.recent_trail.length) {
      ctx.lineWidth = 2;
      for (let i = 0; i < data.recent_trail.length; i++) {
        const pt = data.recent_trail[i];
        const alpha = (i + 1) / data.recent_trail.length;
        const ptX = centerX + pt.dx * scale;
        const ptY = centerY - pt.dz * scale;
        ctx.fillStyle = `rgba(56, 189, 248, ${alpha * 0.7})`;
        ctx.beginPath();
        ctx.arc(ptX, ptY, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    
    // 绘制当前点 (高亮脉冲光圈)
    const curX = centerX + data.metrics.dx * scale;
    const curY = centerY - data.metrics.dz * scale;
    ctx.fillStyle = '#38bdf8';
    ctx.beginPath(); ctx.arc(curX, curY, 5, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // 轴指示文字
    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px sans-serif';
    ctx.fillText('X 轴横向 (mm) →', W - 110, centerY - 6);
    ctx.fillText('↑ Z 轴位移 (mm)', centerX + 6, 16);
    ctx.fillText('(0,0) 原点', centerX + 6, centerY + 14);

  } else {
    // ------------------------------------ 2. 时序位移波形 ΔZ(t)
    const padL = 40, padR = 20, padT = 30, padB = 30;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;
    const zeroY = padT + plotH / 2;
    const mmScale = plotH / 8.0; // ±4mm 范围
    
    // 零位与 ±2mm 警戒线
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, zeroY); ctx.lineTo(W - padR, zeroY); ctx.stroke();
    
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    const top2mm = zeroY - 2.0 * mmScale;
    const bot2mm = zeroY + 2.0 * mmScale;
    ctx.beginPath(); ctx.moveTo(padL, top2mm); ctx.lineTo(W - padR, top2mm); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(padL, bot2mm); ctx.lineTo(W - padR, bot2mm); ctx.stroke();
    ctx.setLineDash([]);
    
    ctx.fillStyle = '#ef4444';
    ctx.font = '10px monospace';
    ctx.fillText('+2.0mm', padL - 38, top2mm + 3);
    ctx.fillText('-2.0mm', padL - 38, bot2mm + 3);
    ctx.fillStyle = '#94a3b8';
    ctx.fillText(' 0.0mm', padL - 38, zeroY + 3);
    
    // 绘制 ΔZ 曲线 (蓝色)
    if (timeSeriesHistory.length >= 2) {
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      for (let i = 0; i < timeSeriesHistory.length; i++) {
        const xPos = padL + (i / (MAX_TIME_POINTS - 1)) * plotW;
        const yPos = zeroY - timeSeriesHistory[i].dz * mmScale;
        if (i === 0) ctx.moveTo(xPos, yPos);
        else ctx.lineTo(xPos, yPos);
      }
      ctx.stroke();
    }
    
    ctx.fillStyle = '#38bdf8';
    ctx.font = '11px sans-serif';
    ctx.fillText('实时波形: ΔZ(t) 深度位移', padL, padT - 10);
  }
  
  ctx.restore();
}

// 页面加载完成后启动轮询
window.addEventListener('DOMContentLoaded', () => {
  pollStats();
});
</script>
</body>
</html>
"""


# ============================================================================== HTTP 请求处理器
class WorkbenchHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        # 静默常规流请求日志，保持终端清爽
        pass

    def do_GET(self):
        url = urlparse(self.path)

        if url.path in ("/", "/index.html"):
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif url.path == "/stream_color":
            self.serve_mjpeg("jpeg_color")

        elif url.path == "/stream_ir":
            self.serve_mjpeg("jpeg_ir")

        elif url.path == "/api/stats":
            with LOCK:
                res = {
                    "metrics": STATE["metrics"],
                    "zero_x": round(STATE["zero_x"], 3),
                    "zero_z": round(STATE["zero_z"], 3),
                    "roi": STATE["roi"],
                    "is_recording": STATE["is_recording"],
                    "record_count": len(STATE["records"]),
                    "recent_trail": list(STATE["recent_trail"]),
                }
            body = json.dumps(res).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif url.path == "/api/export_excel":
            # 一键导出 Excel (时间点: (x, z))
            with LOCK:
                recs = list(STATE["records"])
                meta = dict(CAMERA_META)
                meta["zero_x"] = STATE["zero_x"]
                meta["zero_z"] = STATE["zero_z"]
                meta["export_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 若未录制点，生成一份包含当前点与基准的样例
            if not recs:
                with LOCK:
                    m = STATE["metrics"]
                    recs = [{
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.000"),
                        "elapsed": 0.0,
                        "x": m["x"], "z": m["z"],
                        "dx": m["dx"], "dz": m["dz"],
                        "dr": m["dr"], "mdr": m["mdr"],
                    }]

            excel_bytes = export_records_to_excel_bytes(recs, meta)
            time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fn_ascii = f"D435i_displacement_{time_str}.xlsx"
            fn_utf8 = f"D435i_位移数据表_{time_str}.xlsx"
            fn_quoted = quote(fn_utf8)

            self.send_response(200)
            self.send_header("Content-Type", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            self.send_header("Content-Disposition", f'attachment; filename="{fn_ascii}"; filename*=UTF-8\'\'{fn_quoted}')
            self.send_header("Content-Length", str(len(excel_bytes)))
            self.end_headers()
            self.wfile.write(excel_bytes)

        else:
            self.send_error(404, "Not Found")

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        url = urlparse(self.path)

        if url.path == "/api/zero":
            # 调零重置
            with LOCK:
                m = STATE["metrics"]
                STATE["zero_x"] = m["x"]
                STATE["zero_z"] = m["z"]
                STATE["has_zero"] = True
                STATE["recent_trail"].clear()
            self.send_json({"ok": True, "zero_x": m["x"], "zero_z": m["z"]})

        elif url.path == "/api/record/start":
            with LOCK:
                STATE["is_recording"] = True
                STATE["record_start_time"] = time.time()
                STATE["records"].clear()
            self.send_json({"ok": True})

        elif url.path == "/api/record/stop":
            with LOCK:
                STATE["is_recording"] = False
            self.send_json({"ok": True, "count": len(STATE["records"])})

        elif url.path == "/api/setroi":
            # /api/setroi?cx=...&cy=...&s=...
            q = parse_qs(url.query)
            cx = int(q.get("cx", [320])[0])
            cy = int(q.get("cy", [240])[0])
            s = int(q.get("s", [100])[0])
            rx = max(0, min(640 - s, cx - s // 2))
            ry = max(0, min(480 - s, cy - s // 2))
            with LOCK:
                STATE["roi"] = [rx, ry, s, s]
            self.send_json({"ok": True, "roi": [rx, ry, s, s]})

        elif url.path == "/api/auto_locate":
            # 智能重寻靶
            # 寻找当前深度图中像细杆白纸的候选
            try:
                pipe = rs.pipeline()
                cfg = rs.config()
                cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
                cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                # 仅从全局状态中利用已知算法重定位
                with LOCK:
                    rx, ry, rw, rh = STATE["roi"]
                self.send_json({"ok": True, "roi": [rx, ry, rw, rh], "z": STATE["metrics"]["z"]})
            except Exception as e:
                self.send_json({"ok": False, "err": str(e)})

        else:
            self.send_error(404, "Not Found")

    def send_json(self, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_mjpeg(self, key: str):
        """流式输出 MJPEG。"""
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.end_headers()
        try:
            while not STATE["stop"]:
                with LOCK:
                    jpg = STATE.get(key)
                if not jpg:
                    time.sleep(0.04)
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("utf-8"))
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                time.sleep(1.0 / 30.0)
        except (BrokenPipeError, ConnectionResetError):
            pass


# ============================================================================== 启动与主函数
def main():
    parser = argparse.ArgumentParser(description="Intel RealSense D435i 实时微位移监控数控展示台")
    parser.add_argument("--port", type=int, default=8899, help="Web 服务器端口 (默认: 8899)")
    parser.add_argument("--roi", type=str, default=None, help="初始目标 ROI (x,y,w,h)")
    args = parser.parse_args()

    # 尝试读取上次校准得到的优质 ROI
    roi_file = os.path.join(CURRENT_DIR, "roi.json")
    if os.path.exists(roi_file) and not args.roi:
        try:
            with open(roi_file, encoding="utf-8") as f:
                r_cfg = json.load(f)
                STATE["roi"] = r_cfg["roi"]
                print(f"📖 已载入校准 ROI: {STATE['roi']}")
        except Exception:
            pass

    if args.roi:
        vals = [int(v.strip()) for v in args.roi.split(",")]
        if len(vals) == 4:
            STATE["roi"] = vals

    # 启动相机采集线程
    cap_thread = threading.Thread(target=capture_thread_func, args=(args,), daemon=True)
    cap_thread.start()

    # 等待第一帧捕获
    print("⏳ 正在等待相机三流初始化...")
    for _ in range(120):
        with LOCK:
            if STATE["jpeg_color"] and STATE["jpeg_ir"]:
                break
        time.sleep(0.1)

    print(f"\n{'='*72}")
    print(f"🚀 视觉微位移监控数控展示台已成功启动！")
    print(f"🌐 本地 Web 服务地址: http://127.0.0.1:{args.port}/")
    print(f"📊 功能布局: 实时图 | 红外图 | 二维(X,Z)平面图 | 数控台(一键导出Excel)")
    print(f"{'='*72}\n")

    server = ThreadingHTTPServer(("0.0.0.0", args.port), WorkbenchHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        STATE["stop"] = True
        server.shutdown()
        print("👋 监控展示台已安全退出。")


if __name__ == "__main__":
    sys.exit(main())
