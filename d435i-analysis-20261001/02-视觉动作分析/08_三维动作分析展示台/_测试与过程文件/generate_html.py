# coding: utf-8
import os

html_content = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>腕-肘-肩三关节运动学建模与协同代偿三维可视化分析系统 (Three.js 3D Visualizer)</title>
  <style>
    :root {
      --bg-dark: #070a12;
      --panel-bg: rgba(13, 19, 33, 0.88);
      --panel-border: rgba(56, 189, 248, 0.22);
      --accent-cyan: #38bdf8;
      --accent-blue: #3b82f6;
      --accent-emerald: #10b981;
      --accent-amber: #f59e0b;
      --accent-magenta: #f43f5e;
      --accent-purple: #a855f7;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      user-select: none;
    }

    body {
      background-color: var(--bg-dark);
      color: var(--text-main);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
      overflow: hidden;
      width: 100vw;
      height: 100vh;
    }

    #canvas-container {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      z-index: 1;
    }

    /* Top Navigation Header */
    header {
      position: absolute;
      top: 16px;
      left: 20px;
      right: 20px;
      z-index: 10;
      display: flex;
      justify-content: space-between;
      align-items: center;
      pointer-events: none;
    }

    .header-left, .header-right {
      pointer-events: auto;
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .brand-title {
      font-size: 17px;
      font-weight: 700;
      letter-spacing: 0.5px;
      background: linear-gradient(135deg, #38bdf8 0%, #10b981 50%, #f43f5e 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .brand-sub {
      font-size: 11px;
      color: var(--text-muted);
      font-weight: 400;
      margin-top: 1px;
    }

    .status-pill {
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.35);
      color: #34d399;
      font-size: 11px;
      padding: 4px 10px;
      border-radius: 20px;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .status-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background-color: #34d399;
      box-shadow: 0 0 8px #34d399;
      animation: pulse 2s infinite;
    }

    @keyframes pulse {
      0% { opacity: 0.4; }
      50% { opacity: 1; }
      100% { opacity: 0.4; }
    }

    button {
      background: rgba(30, 41, 59, 0.75);
      color: var(--text-main);
      border: 1px solid var(--panel-border);
      border-radius: 8px;
      padding: 6px 14px;
      font-size: 12px;
      font-weight: 500;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
      backdrop-filter: blur(8px);
    }

    button:hover {
      background: rgba(56, 189, 248, 0.18);
      border-color: rgba(56, 189, 248, 0.5);
      color: #fff;
      transform: translateY(-1px);
    }

    button:active {
      transform: translateY(0);
    }

    button.primary {
      background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
      border-color: #38bdf8;
      color: #ffffff;
      box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3);
    }

    button.active {
      background: rgba(56, 189, 248, 0.25);
      border-color: var(--accent-cyan);
      color: var(--accent-cyan);
      font-weight: 600;
    }

    /* Left Telemetry HUD Panel */
    #telemetry-panel {
      position: absolute;
      top: 80px;
      left: 20px;
      width: 330px;
      z-index: 10;
      background: var(--panel-bg);
      backdrop-filter: blur(16px);
      border: 1px solid var(--panel-border);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 16px 36px rgba(0,0,0,0.6);
      display: flex;
      flex-direction: column;
      gap: 13px;
      max-height: calc(100vh - 190px);
      overflow-y: auto;
    }

    #telemetry-panel::-webkit-scrollbar {
      width: 4px;
    }
    #telemetry-panel::-webkit-scrollbar-thumb {
      background: rgba(56, 189, 248, 0.25);
      border-radius: 2px;
    }

    .panel-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid rgba(255,255,255,0.08);
      padding-bottom: 8px;
    }

    .panel-title {
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.8px;
      color: var(--accent-cyan);
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .phase-badge {
      padding: 3px 10px;
      border-radius: 12px;
      font-size: 11px;
      font-weight: 600;
      background: rgba(244, 63, 94, 0.2);
      color: #fb7185;
      border: 1px solid rgba(244, 63, 94, 0.35);
      transition: all 0.3s;
      white-space: nowrap;
    }

    .metric-grid-3 {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 8px;
    }

    .metric-grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }

    .metric-card {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 8px;
      padding: 7px 10px;
      position: relative;
    }

    .metric-card.wrist { border-left: 3px solid var(--accent-magenta); }
    .metric-card.elbow { border-left: 3px solid var(--accent-cyan); }
    .metric-card.shoulder { border-left: 3px solid var(--accent-emerald); }

    .metric-label {
      font-size: 10px;
      color: var(--text-muted);
      margin-bottom: 2px;
      display: flex;
      align-items: center;
      gap: 4px;
    }

    .metric-value {
      font-size: 15px;
      font-weight: 700;
      font-family: "SF Mono", Monaco, "Courier New", monospace;
      color: #fff;
    }

    .metric-unit {
      font-size: 10px;
      font-weight: 400;
      color: var(--text-dim);
      margin-left: 2px;
    }

    /* Compensation Severity Banner */
    .compensation-alert {
      background: linear-gradient(135deg, rgba(244, 63, 94, 0.15), rgba(245, 158, 11, 0.15));
      border: 1px solid rgba(244, 63, 94, 0.35);
      border-radius: 8px;
      padding: 10px;
    }

    .comp-title {
      font-size: 12px;
      font-weight: 700;
      color: #fb7185;
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 4px;
    }

    .comp-desc {
      font-size: 11px;
      color: #cbd5e1;
      line-height: 1.45;
    }

    .comp-ratio-row {
      display: flex;
      justify-content: space-between;
      margin-top: 6px;
      font-size: 11px;
      padding-top: 6px;
      border-top: 1px dashed rgba(244, 63, 94, 0.25);
    }

    /* Right Charts & Legend HUD */
    #chart-panel {
      position: absolute;
      top: 80px;
      right: 20px;
      width: 360px;
      z-index: 10;
      background: var(--panel-bg);
      backdrop-filter: blur(16px);
      border: 1px solid var(--panel-border);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 16px 36px rgba(0,0,0,0.6);
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .chart-container {
      width: 100%;
      height: 130px;
      background: rgba(10, 15, 28, 0.7);
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 8px;
      position: relative;
    }

    canvas.sync-chart {
      width: 100%;
      height: 100%;
      display: block;
    }

    .chart-legend {
      display: flex;
      justify-content: space-between;
      font-size: 11px;
      padding: 0 4px;
    }

    .legend-item {
      display: flex;
      align-items: center;
      gap: 6px;
      color: var(--text-muted);
    }

    .legend-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
    }

    /* Synchronized Video Window */
    #video-sync-box {
      position: absolute;
      bottom: 105px;
      right: 20px;
      width: 200px;
      height: 290px;
      background: #000;
      border: 1px solid var(--panel-border);
      border-radius: 12px;
      overflow: hidden;
      z-index: 10;
      box-shadow: 0 12px 30px rgba(0,0,0,0.7);
      display: flex;
      flex-direction: column;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    #video-sync-box.minimized {
      height: 36px;
      width: 160px;
    }

    .video-header {
      height: 32px;
      background: rgba(15, 23, 42, 0.95);
      padding: 0 10px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 11px;
      font-weight: 600;
      color: var(--accent-cyan);
      border-bottom: 1px solid rgba(255,255,255,0.08);
      cursor: pointer;
    }

    .video-wrapper {
      flex: 1;
      position: relative;
      background: #000;
    }

    #source-video {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }

    #video-overlay-canvas {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
    }

    /* Bottom Control Bar */
    #bottom-bar {
      position: absolute;
      bottom: 20px;
      left: 20px;
      right: 20px;
      z-index: 10;
      background: var(--panel-bg);
      backdrop-filter: blur(16px);
      border: 1px solid var(--panel-border);
      border-radius: 14px;
      padding: 12px 20px;
      box-shadow: 0 16px 36px rgba(0,0,0,0.6);
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .timeline-container {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .timeline-time {
      font-family: "SF Mono", Monaco, monospace;
      font-size: 12px;
      font-weight: 600;
      color: var(--accent-cyan);
      min-width: 80px;
    }

    .slider-track {
      flex: 1;
      position: relative;
      display: flex;
      align-items: center;
    }

    input[type=range] {
      -webkit-appearance: none;
      width: 100%;
      background: transparent;
    }

    input[type=range]:focus {
      outline: none;
    }

    input[type=range]::-webkit-slider-runnable-track {
      width: 100%;
      height: 6px;
      cursor: pointer;
      background: rgba(255, 255, 255, 0.12);
      border-radius: 3px;
      border: none;
    }

    input[type=range]::-webkit-slider-thumb {
      height: 16px;
      width: 16px;
      border-radius: 50%;
      background: var(--accent-cyan);
      cursor: pointer;
      -webkit-appearance: none;
      margin-top: -5px;
      box-shadow: 0 0 10px rgba(56, 189, 248, 0.8);
      border: 2px solid #fff;
      transition: transform 0.1s;
    }

    input[type=range]::-webkit-slider-thumb:hover {
      transform: scale(1.2);
    }

    .controls-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .btn-group {
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .view-group {
      display: flex;
      align-items: center;
      gap: 4px;
      background: rgba(15, 23, 42, 0.6);
      padding: 3px;
      border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .view-btn {
      padding: 4px 10px;
      font-size: 11px;
      border: none;
      background: transparent;
      color: var(--text-muted);
      border-radius: 6px;
    }

    .view-btn.active {
      background: var(--accent-blue);
      color: #fff;
      font-weight: 600;
    }

    /* Modal / Drawer for Scientific & Mathematical Modeling */
    #theory-modal {
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%) scale(0.95);
      width: 820px;
      max-width: 90vw;
      max-height: 85vh;
      background: #0b1120;
      border: 1px solid var(--panel-border);
      border-radius: 16px;
      box-shadow: 0 25px 60px rgba(0,0,0,0.85);
      z-index: 100;
      padding: 24px;
      overflow-y: auto;
      opacity: 0;
      pointer-events: none;
      transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }

    #theory-modal.visible {
      opacity: 1;
      pointer-events: auto;
      transform: translate(-50%, -50%) scale(1);
    }

    #modal-backdrop {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      background: rgba(0,0,0,0.7);
      backdrop-filter: blur(4px);
      z-index: 90;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.25s;
    }

    #modal-backdrop.visible {
      opacity: 1;
      pointer-events: auto;
    }

    .formula-block {
      background: rgba(15, 23, 42, 0.7);
      border-left: 3px solid var(--accent-cyan);
      padding: 12px 16px;
      border-radius: 6px;
      font-family: "SF Mono", Monaco, "Courier New", monospace;
      font-size: 13px;
      color: #38bdf8;
      margin: 10px 0;
      overflow-x: auto;
    }

    .modal-close {
      position: absolute;
      top: 18px;
      right: 18px;
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-size: 18px;
      cursor: pointer;
    }

    .modal-close:hover {
      color: #fff;
    }
  </style>
</head>
<body>

  <!-- 3D WebGL Canvas Container -->
  <div id="canvas-container"></div>

  <!-- Top Navigation Header -->
  <header>
    <div class="header-left">
      <div>
        <div class="brand-title">
          <span>🦾 上肢腕-肘-肩三关节运动学与协同代偿判断系统</span>
        </div>
        <div class="brand-sub">Biomechanical Kinematics & Multi-Joint Compensatory Synergy (Three.js 3D Visualizer)</div>
      </div>
      <div class="status-pill">
        <div class="status-dot"></div>
        <span>视觉轨迹提取与模型重建已就绪</span>
      </div>
    </div>

    <div class="header-right">
      <button onclick="toggleTheoryModal(true)">
        <span>📐 运动学建模数学描述</span>
      </button>
      <button onclick="downloadJSON()">
        <span>💾 导出轨迹数据 (JSON)</span>
      </button>
    </div>
  </header>

  <!-- Left Telemetry HUD Panel -->
  <div id="telemetry-panel">
    <div class="panel-header">
      <div class="panel-title">
        <span>📊 运动学实时遥测</span>
      </div>
      <div class="phase-badge" id="hud-phase">旋后代偿极大值</div>
    </div>

    <!-- Compensation Alert Card -->
    <div class="compensation-alert">
      <div class="comp-title">
        <span>⚠️ 运动学代偿判定: 显著代偿</span>
        <span style="font-size: 10px; background: rgba(244,63,94,0.3); padding: 1px 6px; border-radius: 4px;">Grade Ⅲ</span>
      </div>
      <div class="comp-desc">
        目标动作本为前臂单纯旋前-旋后，但由于远端关节隔离受限，诱发显著的<b>肩关节代偿抬升(耸肩/外展)</b>与<b>肘关节向外扑动(屈肘代偿)</b>，呈现显著的整体多关节协同联带运动。
      </div>
      <div class="comp-ratio-row">
        <span>肩代偿率 (SCR): <b id="hud-scr" style="color:var(--accent-emerald);">42.7%</b></span>
        <span>肘代偿率 (ECR): <b id="hud-ecr" style="color:var(--accent-cyan);">82.2%</b></span>
      </div>
    </div>

    <!-- Joint Positions -->
    <div style="font-size: 11px; font-weight: 600; color: var(--text-muted); margin-top: 4px;">当前三关节空间坐标 (mm)</div>
    <div class="metric-grid-3">
      <div class="metric-card shoulder">
        <div class="metric-label">肩关节 (X,Y,Z)</div>
        <div class="metric-value" id="hud-sh-pos" style="font-size:12px;">--</div>
      </div>
      <div class="metric-card elbow">
        <div class="metric-label">肘关节 (X,Y,Z)</div>
        <div class="metric-value" id="hud-el-pos" style="font-size:12px;">--</div>
      </div>
      <div class="metric-card wrist">
        <div class="metric-label">腕关节 (X,Y,Z)</div>
        <div class="metric-value" id="hud-wr-pos" style="font-size:12px;">--</div>
      </div>
    </div>

    <!-- Joint Angles -->
    <div style="font-size: 11px; font-weight: 600; color: var(--text-muted); margin-top: 4px;">生物力学关节角度 (Degrees)</div>
    <div class="metric-grid-2">
      <div class="metric-card elbow">
        <div class="metric-label">肘关节弯曲角 (θ_elbow)</div>
        <div class="metric-value"><span id="hud-elbow-angle">--</span><span class="metric-unit">°</span></div>
      </div>
      <div class="metric-card shoulder">
        <div class="metric-label">肩关节抬升角 (θ_elev)</div>
        <div class="metric-value"><span id="hud-shoulder-angle">--</span><span class="metric-unit">°</span></div>
      </div>
    </div>

    <!-- Speeds -->
    <div style="font-size: 11px; font-weight: 600; color: var(--text-muted); margin-top: 4px;">瞬时线速度 (mm/s)</div>
    <div class="metric-grid-3">
      <div class="metric-card shoulder">
        <div class="metric-label">肩部速度</div>
        <div class="metric-value" style="font-size:13px;"><span id="hud-sh-speed">--</span></div>
      </div>
      <div class="metric-card elbow">
        <div class="metric-label">肘部速度</div>
        <div class="metric-value" style="font-size:13px;"><span id="hud-el-speed">--</span></div>
      </div>
      <div class="metric-card wrist">
        <div class="metric-label">腕部速度</div>
        <div class="metric-value" style="font-size:13px;"><span id="hud-wr-speed">--</span></div>
      </div>
    </div>

    <!-- Toggle Layer Checkboxes -->
    <div style="font-size: 11px; font-weight: 600; color: var(--text-muted); margin-top: 4px;">视图图层控制</div>
    <div style="display:flex; flex-direction:column; gap:6px; font-size:11px;">
      <label style="display:flex; align-items:center; gap:8px; cursor:pointer;">
        <input type="checkbox" id="chk-skeleton" checked onchange="toggleLayer('skeleton', this.checked)">
        <span>显示完整上肢骨骼联动 (Upper Limb Rig)</span>
      </label>
      <label style="display:flex; align-items:center; gap:8px; cursor:pointer;">
        <input type="checkbox" id="chk-trajectories" checked onchange="toggleLayer('trajectories', this.checked)">
        <span>显示三关节3D时空运动轨迹 (3-Joint Ribbons)</span>
      </label>
      <label style="display:flex; align-items:center; gap:8px; cursor:pointer;">
        <input type="checkbox" id="chk-vectors" checked onchange="toggleLayer('vectors', this.checked)">
        <span>显示代偿偏离向量与速度箭头 (Compensation Vectors)</span>
      </label>
      <label style="display:flex; align-items:center; gap:8px; cursor:pointer;">
        <input type="checkbox" id="chk-projections" checked onchange="toggleLayer('projections', this.checked)">
        <span>显示桌面/背板正交运动投影 (Orthogonal Projections)</span>
      </label>
    </div>
  </div>

  <!-- Right Dynamic Curves HUD -->
  <div id="chart-panel">
    <div class="panel-header">
      <div class="panel-title">
        <span>📈 多关节时序曲线分析</span>
      </div>
      <div style="font-size: 11px; color: var(--text-muted);">
        基频: <span id="hud-freq" style="color:var(--accent-cyan); font-weight:700;">1.21 Hz</span>
      </div>
    </div>

    <!-- Chart 1: Y Displacements (Wrist, Elbow, Shoulder) -->
    <div style="font-size: 11px; color: var(--text-muted); display:flex; justify-content:space-between;">
      <span>三关节垂直向位移时序 Y(t) (mm)</span>
      <span style="font-size:10px; color:#38bdf8;">强协同耦合 r=0.86</span>
    </div>
    <div class="chart-container">
      <canvas id="chart-disp" class="sync-chart"></canvas>
    </div>
    <div class="chart-legend">
      <div class="legend-item"><div class="legend-dot" style="background:var(--accent-magenta);"></div>腕部 Y</div>
      <div class="legend-item"><div class="legend-dot" style="background:var(--accent-cyan);"></div>肘部 Y</div>
      <div class="legend-item"><div class="legend-dot" style="background:var(--accent-emerald);"></div>肩部 Y</div>
    </div>

    <!-- Chart 2: Joint Angles (Elbow Flexion, Shoulder Elevation) -->
    <div style="font-size: 11px; color: var(--text-muted); display:flex; justify-content:space-between; margin-top:4px;">
      <span>肘屈曲角 θ_elbow 与 肩抬升角 (deg)</span>
      <span style="font-size:10px; color:#fb7185;">肘波动 Δ=23.9°</span>
    </div>
    <div class="chart-container">
      <canvas id="chart-angle" class="sync-chart"></canvas>
    </div>
    <div class="chart-legend">
      <div class="legend-item"><div class="legend-dot" style="background:#06b6d4;"></div>肘关节屈曲角</div>
      <div class="legend-item"><div class="legend-dot" style="background:#10b981;"></div>肩关节抬升角</div>
    </div>
  </div>

  <!-- Synchronized Raw Video Window -->
  <div id="video-sync-box">
    <div class="video-header" onclick="toggleVideoBox()">
      <span>📹 原始视频同步核验</span>
      <span id="video-minimize-btn" style="cursor:pointer; font-size:12px;">▼</span>
    </div>
    <div class="video-wrapper">
      <video id="source-video" src="07_资料库/实验2-腕肘肩代偿判断-视频3-腕肘肩_h264.mp4" muted playsinline preload="auto"></video>
      <canvas id="video-overlay-canvas"></canvas>
    </div>
  </div>

  <!-- Bottom Control Bar -->
  <div id="bottom-bar">
    <div class="timeline-container">
      <div class="timeline-time">
        <span id="time-current">0.00</span> / <span id="time-total">7.10</span> s
      </div>
      <div class="slider-track">
        <input type="range" id="timeline-slider" min="0" max="212" value="0" step="1">
      </div>
      <div style="font-size:11px; color:var(--text-muted); min-width:60px; text-align:right;">
        帧: <span id="frame-current" style="color:#fff; font-weight:700;">0</span> / 212
      </div>
    </div>

    <div class="controls-row">
      <!-- Playback buttons -->
      <div class="btn-group">
        <button id="btn-play" class="primary" onclick="togglePlay()">
          <span id="play-icon">▶</span> <span id="play-text">播放</span>
        </button>
        <button onclick="stepFrame(-1)">⏮ 上一帧</button>
        <button onclick="stepFrame(1)">下一帧 ⏭</button>
        <button id="btn-loop" class="active" onclick="toggleLoop()">🔁 循环</button>
        <div style="display:flex; align-items:center; gap:4px; margin-left:8px;">
          <span style="font-size:11px; color:var(--text-muted);">倍速:</span>
          <button class="speed-btn" onclick="setSpeed(0.25, this)">0.25x</button>
          <button class="speed-btn active" onclick="setSpeed(0.5, this)">0.5x</button>
          <button class="speed-btn" onclick="setSpeed(1.0, this)">1.0x</button>
        </div>
      </div>

      <!-- Camera Views -->
      <div class="view-group">
        <span style="font-size:11px; color:var(--text-muted); padding-left:6px;">视角:</span>
        <button class="view-btn active" onclick="setCameraView('perspective', this)">3D透视</button>
        <button class="view-btn" onclick="setCameraView('coronal', this)">冠状面(正视)</button>
        <button class="view-btn" onclick="setCameraView('sagittal', this)">矢状面(侧视)</button>
        <button class="view-btn" onclick="setCameraView('transverse', this)">水平面(俯视)</button>
        <button class="view-btn" onclick="setCameraView('closeup', this)">腕关节特写</button>
      </div>
    </div>
  </div>

  <!-- Backdrop -->
  <div id="modal-backdrop" onclick="toggleTheoryModal(false)"></div>

  <!-- Theory & Mathematical Modeling Modal -->
  <div id="theory-modal">
    <button class="modal-close" onclick="toggleTheoryModal(false)">✕</button>
    <h2 style="color:var(--accent-cyan); font-size:18px; margin-bottom:12px; display:flex; align-items:center; gap:8px;">
      <span>📐 腕-肘-肩三关节运动学建模与代偿判断数学理论</span>
    </h2>

    <div style="font-size:13px; line-height:1.6; color:#cbd5e1;">
      <h3 style="color:#fff; font-size:14px; margin:14px 0 6px 0;">一、生物力学与临床代偿机制定义</h3>
      <p>
        在临床康复评估（如脑卒中 Fugl-Meyer 评测、运动协调性测试 RAMT）中，前臂的旋前-旋后（Pronation-Supination）应主要由<b>桡尺近端与远端关节（Radio-ulnar Joints）</b>沿前臂纵轴孤立旋转完成。
        当受试者远端肌群控制不充分或出现联带痉挛时，会借用近端强有力的肌群进行运动补偿（代偿）：
      </p>
      <ul style="margin-left: 20px; margin-top: 6px;">
        <li><b>肩关节代偿 (Shoulder Compensation)</b>：通过斜方肌/三角肌收缩引发<b>肩胛骨抬升（耸肩）</b>与<b>肩关节外展</b>，借由躯干及大臂摇摆带动腕部旋转。</li>
        <li><b>肘关节代偿 (Elbow Compensation)</b>：通过肱二头肌/肱肌的周期性屈伸，产生<b>肘关节外展摇晃（Flaring）</b>与屈曲代偿。</li>
      </ul>

      <h3 style="color:#fff; font-size:14px; margin:14px 0 6px 0;">二、三维运动学轨迹数学建模</h3>
      <p>设各关节在时间 $t$ 的三维空间位移向量分别为 $\mathbf{P}_{\text{wrist}}(t), \mathbf{P}_{\text{elbow}}(t), \mathbf{P}_{\text{shoulder}}(t) \in \mathbb{R}^3$：</p>
      <div class="formula-block">
        \mathbf{P}_{\text{wrist}}(t) = [X_w(t), Y_w(t), Z_w(t)]^T \\
        \mathbf{P}_{\text{elbow}}(t) = [X_e(t), Y_e(t), Z_e(t)]^T \\
        \mathbf{P}_{\text{shoulder}}(t) = [X_s(t), Y_s(t), Z_s(t)]^T
      </div>

      <p>各关节瞬时运动速度 $\mathbf{v}(t)$ 与瞬时加速度 $\mathbf{a}(t)$ 满足导数连续方程：</p>
      <div class="formula-block">
        \mathbf{v}(t) = \frac{d\mathbf{P}(t)}{dt}, \quad v(t) = \|\mathbf{v}(t)\|_2 \\
        \mathbf{a}(t) = \frac{d^2\mathbf{P}(t)}{dt^2}, \quad a(t) = \|\mathbf{a}(t)\|_2
      </div>

      <h3 style="color:#fff; font-size:14px; margin:14px 0 6px 0;">三、空间关节角度解算 (Joint Angles)</h3>
      <p>由肩-肘-腕构成的肘关节屈曲角 $\theta_{\text{elbow}}(t)$ 可通过大臂向量 $\mathbf{u}(t) = \mathbf{P}_s - \mathbf{P}_e$ 与小臂向量 $\mathbf{f}(t) = \mathbf{P}_w - \mathbf{P}_e$ 的内积确定：</p>
      <div class="formula-block">
        \theta_{\text{elbow}}(t) = \arccos\left( \frac{\mathbf{u}(t) \cdot \mathbf{f}(t)}{\|\mathbf{u}(t)\| \|\mathbf{f}(t)\|} \right)
      </div>

      <h3 style="color:#fff; font-size:14px; margin:14px 0 6px 0;">四、代偿量化指数 (Compensatory Indices)</h3>
      <p>定义肩部代偿率（Shoulder Compensation Ratio, SCR）与肘部代偿率（Elbow Compensation Ratio, ECR）：</p>
      <div class="formula-block">
        \text{SCR}_Y = \frac{\Delta Y_{\text{shoulder}}}{\Delta Y_{\text{wrist}}} = \frac{\max(Y_s) - \min(Y_s)}{\max(Y_w) - \min(Y_w)} = \frac{44.9\text{ mm}}{105.2\text{ mm}} = 0.427 \ (42.7\%) \\
        \text{ECR}_Y = \frac{\Delta Y_{\text{elbow}}}{\Delta Y_{\text{wrist}}} = \frac{\max(Y_e) - \min(Y_e)}{\max(Y_w) - \min(Y_w)} = \frac{86.5\text{ mm}}{105.2\text{ mm}} = 0.822 \ (82.2\%)
      </div>
      <p>定义运动学协同耦合系数（Kinematic Coupling Coefficient, KCC）：</p>
      <div class="formula-block">
        r(\text{Elbow}_Y, \text{Wrist}_Y) = 0.856, \quad r(\text{Shoulder}_Y, \text{Wrist}_Y) = 0.819
      </div>
      <p>高皮尔逊相关系数（$r > 0.8$）定量证实了肩、肘、腕垂直运动之间的高度协同性，确凿证明了代偿模式的存在。</p>

      <h3 style="color:#fff; font-size:14px; margin:14px 0 6px 0;">五、周期性傅里叶级数展开参数</h3>
      <p>以动作基频 $f_0 = 1.21\text{ Hz}$（周期 $T_0 = 0.83\text{ s}$）对腕关节垂直位移 $Y(t)$ 进行 3 阶谐波分解：</p>
      <div class="formula-block">
        Y_w(t) = 66.37 - 0.37 t + [5.14\cos(\omega_0 t) + 10.53\sin(\omega_0 t)] + [0.30\cos(2\omega_0 t) + 0.02\sin(2\omega_0 t)] + [0.93\cos(3\omega_0 t) - 0.39\sin(3\omega_0 t)]
      </div>
    </div>

    <div style="margin-top:20px; display:flex; justify-content:flex-end; gap:10px;">
      <button onclick="downloadJSON()">下载轨迹原始数据 (JSON)</button>
      <button class="primary" onclick="toggleTheoryModal(false)">返回三维视图</button>
    </div>
  </div>

  <!-- Local Script Dependencies -->
  <script src="libs/three.min.js"></script>
  <script src="libs/OrbitControls.js"></script>
  <script src="arm_compensation_data.js"></script>

  <script>
    // --- Application State ---
    const data = window.ARM_COMPENSATION_DATA;
    const frames = data.frames;
    const waypoints = data.waypoints;
    const totalFrames = frames.length;
    const duration = data.metadata.duration_s;

    let currentIndex = 0;
    let isPlaying = false;
    let isLooping = true;
    let playbackSpeed = 0.5;
    let lastAnimTime = null;

    let layerStates = {
      skeleton: true,
      trajectories: true,
      vectors: true,
      projections: true
    };

    // --- Three.js Scene Setup ---
    const container = document.getElementById('canvas-container');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x070a12);
    scene.fog = new THREE.FogExp2(0x070a12, 0.001);

    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 4000);
    // Position to frame the whole arm (Shoulder ~460mm high, elbow ~200mm, wrist ~65mm)
    camera.position.set(120, 360, 680);

    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.target.set(-140, 240, 90);
    controls.maxPolarAngle = Math.PI / 2 + 0.1;
    controls.minDistance = 80;
    controls.maxDistance = 1800;

    // --- Lighting ---
    const ambientLight = new THREE.AmbientLight(0xdbeafe, 0.7);
    scene.add(ambientLight);

    const mainLight = new THREE.DirectionalLight(0xffffff, 1.2);
    mainLight.position.set(200, 600, 300);
    mainLight.castShadow = true;
    mainLight.shadow.mapSize.width = 2048;
    mainLight.shadow.mapSize.height = 2048;
    mainLight.shadow.camera.near = 50;
    mainLight.shadow.camera.far = 1500;
    mainLight.shadow.camera.left = -400;
    mainLight.shadow.camera.right = 400;
    mainLight.shadow.camera.top = 400;
    mainLight.shadow.camera.bottom = -400;
    scene.add(mainLight);

    const cyanRim = new THREE.DirectionalLight(0x38bdf8, 0.8);
    cyanRim.position.set(-400, 200, -200);
    scene.add(cyanRim);

    const magentaRim = new THREE.DirectionalLight(0xf43f5e, 0.6);
    magentaRim.position.set(300, 100, -300);
    scene.add(magentaRim);

    // Coordinate conversion
    function toThree(pos) {
      return new THREE.Vector3(pos[0], pos[1], pos[2]);
    }

    // --- Ground Grid & Anatomical Coordinates ---
    const envGroup = new THREE.Group();

    // Table Plane at Y=0
    const groundGeo = new THREE.PlaneGeometry(1200, 1200);
    const groundMat = new THREE.MeshStandardMaterial({
      color: 0x0a101d,
      roughness: 0.8,
      metalness: 0.1
    });
    const groundMesh = new THREE.Mesh(groundGeo, groundMat);
    groundMesh.rotation.x = -Math.PI / 2;
    groundMesh.receiveShadow = true;
    envGroup.add(groundMesh);

    // Millimeter Grid on Floor
    const gridHelper = new THREE.GridHelper(1000, 50, 0x1e293b, 0x0f172a);
    gridHelper.position.y = 0.1;
    envGroup.add(gridHelper);

    // 3D Coordinate Arrows at Origin
    const axesGroup = new THREE.Group();
    const axisLen = 100;
    axesGroup.add(new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0.5, 0), axisLen, 0xef4444, 12, 6)); // X Medial-Lateral
    axesGroup.add(new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0.5, 0), axisLen, 0x10b981, 12, 6)); // Y Superior-Inferior
    axesGroup.add(new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0.5, 0), axisLen, 0x3b82f6, 12, 6)); // Z Anterior-Posterior
    axesGroup.position.set(-250, 0, -50);
    envGroup.add(axesGroup);

    scene.add(envGroup);

    // --- Trajectories Generation ---
    const trajGroup = new THREE.Group();
    const projGroup = new THREE.Group();

    function createJointTube(points, colorHex, radius = 2.0, isGradientSpeed = false) {
      const curve = new THREE.CatmullRomCurve3(points);
      curve.curveType = 'centripetal';
      const segments = 400;
      const geo = new THREE.TubeGeometry(curve, segments, radius, 10, false);

      let mat;
      if (isGradientSpeed) {
        const colors = [];
        const radialSegments = 10;
        for (let i = 0; i <= segments; i++) {
          const u = i / segments;
          const fIdx = Math.min(Math.floor(u * (totalFrames - 1)), totalFrames - 1);
          const spd = frames[fIdx].wr_speed;
          const norm = Math.min(spd / 600, 1.0);
          const c = new THREE.Color();
          if (norm < 0.5) {
            c.lerpColors(new THREE.Color(0xf43f5e), new THREE.Color(0xfacc15), norm * 2.0);
          } else {
            c.lerpColors(new THREE.Color(0xfacc15), new THREE.Color(0x38bdf8), (norm - 0.5) * 2.0);
          }
          for (let r = 0; r <= radialSegments; r++) {
            colors.push(c.r, c.g, c.b);
          }
        }
        geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        mat = new THREE.MeshStandardMaterial({
          vertexColors: true,
          roughness: 0.25,
          metalness: 0.5,
          emissive: 0x330010,
          emissiveIntensity: 0.3
        });
      } else {
        mat = new THREE.MeshStandardMaterial({
          color: colorHex,
          roughness: 0.3,
          metalness: 0.4,
          emissive: colorHex,
          emissiveIntensity: 0.4
        });
      }

      const mesh = new THREE.Mesh(geo, mat);
      mesh.castShadow = true;
      return mesh;
    }

    const shPoints = frames.map(f => toThree(f.sh_3d));
    const elPoints = frames.map(f => toThree(f.el_3d));
    const wrPoints = frames.map(f => toThree(f.wr_3d));

    const shTube = createJointTube(shPoints, 0x10b981, 1.8, false); // Emerald
    const elTube = createJointTube(elPoints, 0x06b6d4, 2.0, false); // Cyan
    const wrTube = createJointTube(wrPoints, 0xf43f5e, 2.5, true);  // Magenta/Speed

    trajGroup.add(shTube);
    trajGroup.add(elTube);
    trajGroup.add(wrTube);
    scene.add(trajGroup);

    // Ground Projections (Floor Shadow Ribbons)
    function createProjectionLine(points, colorHex) {
      const pPts = points.map(p => new THREE.Vector3(p.x, 0.4, p.z));
      const geo = new THREE.BufferGeometry().setFromPoints(pPts);
      const mat = new THREE.LineBasicMaterial({
        color: colorHex,
        transparent: true,
        opacity: 0.35,
        linewidth: 2
      });
      return new THREE.Line(geo, mat);
    }

    projGroup.add(createProjectionLine(shPoints, 0x10b981));
    projGroup.add(createProjectionLine(elPoints, 0x06b6d4));
    projGroup.add(createProjectionLine(wrPoints, 0xf43f5e));
    scene.add(projGroup);

    // --- Dynamic Arm Skeletal Rig ---
    const skeletonGroup = new THREE.Group();

    // 1. Joint Spheres
    function createJointMesh(colorHex, size, glowHex) {
      const geo = new THREE.SphereGeometry(size, 24, 24);
      const mat = new THREE.MeshStandardMaterial({
        color: colorHex,
        emissive: glowHex || colorHex,
        emissiveIntensity: 0.7,
        roughness: 0.2,
        metalness: 0.6
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.castShadow = true;
      return mesh;
    }

    const shJointMesh = createJointMesh(0x10b981, 7.5, 0x059669);
    const elJointMesh = createJointMesh(0x06b6d4, 7.0, 0x0891b2);
    const wrJointMesh = createJointMesh(0xf43f5e, 6.5, 0xe11d48);
    const lShJointMesh = createJointMesh(0x64748b, 5.0, 0x475569);

    skeletonGroup.add(shJointMesh);
    skeletonGroup.add(elJointMesh);
    skeletonGroup.add(wrJointMesh);
    skeletonGroup.add(lShJointMesh);

    // 2. Bone Cylinders (Connecting Joints dynamically)
    function createBoneMesh(radius, colorHex) {
      const geo = new THREE.CylinderGeometry(radius, radius, 1, 16);
      geo.translate(0, 0.5, 0); // pivot at base
      geo.rotateX(Math.PI / 2); // align with Z forward
      const mat = new THREE.MeshStandardMaterial({
        color: colorHex,
        roughness: 0.4,
        metalness: 0.5
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.castShadow = true;
      return mesh;
    }

    const clavicleBone = createBoneMesh(3.5, 0x475569); // Left Shoulder to Right Shoulder
    const humerusBone = createBoneMesh(6.5, 0x334155);  // Shoulder to Elbow
    const radiusBone = createBoneMesh(3.8, 0x1e293b);   // Elbow to Wrist (Radius)
    const ulnaBone = createBoneMesh(3.8, 0x1e293b);     // Elbow to Wrist (Ulna)

    skeletonGroup.add(clavicleBone);
    skeletonGroup.add(humerusBone);
    skeletonGroup.add(radiusBone);
    skeletonGroup.add(ulnaBone);

    // 3. Hand / Palm Mesh & Normal Vector
    const handGroup = new THREE.Group();
    const palmGeo = new THREE.BoxGeometry(32, 8, 48);
    const palmMat = new THREE.MeshStandardMaterial({
      color: 0xf43f5e,
      roughness: 0.3,
      metalness: 0.4,
      emissive: 0x9f1239,
      emissiveIntensity: 0.3
    });
    const palmMesh = new THREE.Mesh(palmGeo, palmMat);
    palmMesh.position.set(0, 0, 24);
    handGroup.add(palmMesh);

    // Palm Normal Vector Arrow
    const palmNormalArrow = new THREE.ArrowHelper(
      new THREE.Vector3(0, 1, 0),
      new THREE.Vector3(0, 0, 20),
      40,
      0xfacc15,
      10,
      5
    );
    handGroup.add(palmNormalArrow);

    skeletonGroup.add(handGroup);
    scene.add(skeletonGroup);

    // --- Dynamic Compensation Indicators (Vector Arrows & Deviation Lines) ---
    const vectorGroup = new THREE.Group();

    // Neutral reference markers
    const neutralShPos = toThree(frames[0].sh_3d);
    const neutralElPos = toThree(frames[0].el_3d);

    const shRefSphere = new THREE.Mesh(
      new THREE.SphereGeometry(4, 16, 16),
      new THREE.MeshBasicMaterial({ color: 0x10b981, wireframe: true, transparent: true, opacity: 0.4 })
    );
    shRefSphere.position.copy(neutralShPos);
    vectorGroup.add(shRefSphere);

    const elRefSphere = new THREE.Mesh(
      new THREE.SphereGeometry(4, 16, 16),
      new THREE.MeshBasicMaterial({ color: 0x06b6d4, wireframe: true, transparent: true, opacity: 0.4 })
    );
    elRefSphere.position.copy(neutralElPos);
    vectorGroup.add(elRefSphere);

    // Deviation lines
    const shDevGeo = new THREE.BufferGeometry().setFromPoints([neutralShPos, neutralShPos]);
    const shDevLine = new THREE.Line(shDevGeo, new THREE.LineDashedMaterial({ color: 0x10b981, dashSize: 4, gapSize: 2 }));
    vectorGroup.add(shDevLine);

    const elDevGeo = new THREE.BufferGeometry().setFromPoints([neutralElPos, neutralElPos]);
    const elDevLine = new THREE.Line(elDevGeo, new THREE.LineDashedMaterial({ color: 0x06b6d4, dashSize: 4, gapSize: 2 }));
    vectorGroup.add(elDevLine);

    // Wrist Velocity Arrow
    const wrVelArrow = new THREE.ArrowHelper(
      new THREE.Vector3(1, 0, 0),
      new THREE.Vector3(0, 0, 0),
      30,
      0xf43f5e,
      8,
      4
    );
    vectorGroup.add(wrVelArrow);

    scene.add(vectorGroup);

    // Helper to position and orient bone cylinder between two points
    function updateBone(mesh, pA, pB, offsetDir = null, offsetDist = 0) {
      const vA = pA.clone();
      const vB = pB.clone();
      if (offsetDir) {
        vA.addScaledVector(offsetDir, offsetDist);
        vB.addScaledVector(offsetDir, offsetDist);
      }
      mesh.position.copy(vA);
      mesh.lookAt(vB);
      const dist = vA.distanceTo(vB);
      mesh.scale.set(1, 1, Math.max(dist, 0.01));
    }

    // --- Update Dynamic 3D Scene to Frame Index ---
    function updateSceneToFrame(idx) {
      const f = frames[idx];
      const pSh = toThree(f.sh_3d);
      const pEl = toThree(f.el_3d);
      const pWr = toThree(f.wr_3d);
      const pLSh = toThree(f.l_sh_3d);

      // Update Joint Positions
      shJointMesh.position.copy(pSh);
      elJointMesh.position.copy(pEl);
      wrJointMesh.position.copy(pWr);
      lShJointMesh.position.copy(pLSh);

      // Update Bones
      updateBone(clavicleBone, pLSh, pSh);
      updateBone(humerusBone, pSh, pEl);

      // Dual Forearm Bones (Radius & Ulna)
      const fArmVec = new THREE.Vector3().subVectors(pWr, pEl).normalize();
      const sideVec = new THREE.Vector3().crossVectors(fArmVec, new THREE.Vector3(0, 1, 0)).normalize();
      updateBone(radiusBone, pEl, pWr, sideVec, 7);
      updateBone(ulnaBone, pEl, pWr, sideVec, -7);

      // Update Hand / Palm
      handGroup.position.copy(pWr);
      handGroup.lookAt(new THREE.Vector3().addVectors(pWr, fArmVec));
      // Hand rotation around longitudinal axis by pronation_deg
      handGroup.rotation.z = THREE.MathUtils.degToRad(-f.pronation_deg);

      // Update Palm Normal Arrow
      const normVec = new THREE.Vector3(f.normal_vec[0], f.normal_vec[1], f.normal_vec[2]).normalize();
      palmNormalArrow.setDirection(normVec);

      // Update Deviation lines
      shDevGeo.setFromPoints([neutralShPos, pSh]);
      shDevLine.computeLineDistances();

      elDevGeo.setFromPoints([neutralElPos, pEl]);
      elDevLine.computeLineDistances();

      // Wrist Velocity Arrow
      wrVelArrow.position.copy(pWr);
      const velVec = new THREE.Vector3(f.wr_vel[0], f.wr_vel[1], f.wr_vel[2]);
      const speed = velVec.length();
      if (speed > 10) {
        wrVelArrow.visible = true;
        wrVelArrow.setDirection(velVec.clone().normalize());
        wrVelArrow.setLength(Math.min(speed * 0.12, 60), 8, 4);
      } else {
        wrVelArrow.visible = false;
      }

      // Update HUD Numbers
      document.getElementById('hud-phase').innerText = f.phase;
      document.getElementById('hud-sh-pos').innerText = `${f.sh_3d[0].toFixed(0)}, ${f.sh_3d[1].toFixed(0)}, ${f.sh_3d[2].toFixed(0)}`;
      document.getElementById('hud-el-pos').innerText = `${f.el_3d[0].toFixed(0)}, ${f.el_3d[1].toFixed(0)}, ${f.el_3d[2].toFixed(0)}`;
      document.getElementById('hud-wr-pos').innerText = `${f.wr_3d[0].toFixed(0)}, ${f.wr_3d[1].toFixed(0)}, ${f.wr_3d[2].toFixed(0)}`;

      document.getElementById('hud-elbow-angle').innerText = f.elbow_angle.toFixed(1);
      document.getElementById('hud-shoulder-angle').innerText = f.shoulder_elev.toFixed(1);

      document.getElementById('hud-sh-speed').innerText = f.sh_speed.toFixed(0);
      document.getElementById('hud-el-speed').innerText = f.el_speed.toFixed(0);
      document.getElementById('hud-wr-speed').innerText = f.wr_speed.toFixed(0);

      document.getElementById('time-current').innerText = f.time.toFixed(2);
      document.getElementById('frame-current').innerText = idx;
      document.getElementById('timeline-slider').value = idx;

      // Sync Video
      syncVideo(f.time);
      drawVideoOverlay(f);

      // Update Charts Cursor Line
      drawCharts(idx);
    }

    // --- Synchronized Canvas Charts ---
    const dispCanvas = document.getElementById('chart-disp');
    const angleCanvas = document.getElementById('chart-angle');
    const dCtx = dispCanvas.getContext('2d');
    const aCtx = angleCanvas.getContext('2d');

    function resizeCanvases() {
      const rect1 = dispCanvas.parentElement.getBoundingClientRect();
      dispCanvas.width = rect1.width * window.devicePixelRatio;
      dispCanvas.height = rect1.height * window.devicePixelRatio;
      dCtx.scale(window.devicePixelRatio, window.devicePixelRatio);

      const rect2 = angleCanvas.parentElement.getBoundingClientRect();
      angleCanvas.width = rect2.width * window.devicePixelRatio;
      angleCanvas.height = rect2.height * window.devicePixelRatio;
      aCtx.scale(window.devicePixelRatio, window.devicePixelRatio);
    }

    function drawCharts(currentIdx) {
      const w = dispCanvas.width / window.devicePixelRatio;
      const h = dispCanvas.height / window.devicePixelRatio;

      // --- Draw Chart 1: Y-Displacement ---
      dCtx.clearRect(0, 0, w, h);

      // Grid lines
      dCtx.strokeStyle = 'rgba(255,255,255,0.06)';
      dCtx.lineWidth = 1;
      for (let y = 20; y < h; y += 30) {
        dCtx.beginPath();
        dCtx.moveTo(0, y);
        dCtx.lineTo(w, y);
        dCtx.stroke();
      }

      // Curves bounds
      // Normalize Y between min ~ 0 and max ~ 500
      const minY = 0;
      const maxY = 500;
      const mapY = (val) => h - 10 - ((val - minY) / (maxY - minY)) * (h - 20);
      const mapX = (i) => (i / (totalFrames - 1)) * w;

      function drawCurve(ctx, key, color, lineWidth = 2) {
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        for (let i = 0; i < totalFrames; i++) {
          const x = mapX(i);
          const y = mapY(frames[i][key][1]);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      drawCurve(dCtx, 'wr_3d', '#f43f5e', 2.5); // Wrist Magenta
      drawCurve(dCtx, 'el_3d', '#06b6d4', 2.0); // Elbow Cyan
      drawCurve(dCtx, 'sh_3d', '#10b981', 1.8); // Shoulder Emerald

      // Time cursor line
      const curX = mapX(currentIdx);
      dCtx.strokeStyle = '#ffffff';
      dCtx.lineWidth = 1.5;
      dCtx.setLineDash([4, 2]);
      dCtx.beginPath();
      dCtx.moveTo(curX, 0);
      dCtx.lineTo(curX, h);
      dCtx.stroke();
      dCtx.setLineDash([]);

      // Current point dots
      [
        { key: 'wr_3d', col: '#f43f5e' },
        { key: 'el_3d', col: '#06b6d4' },
        { key: 'sh_3d', col: '#10b981' }
      ].forEach(item => {
        const py = mapY(frames[currentIdx][item.key][1]);
        dCtx.fillStyle = item.col;
        dCtx.beginPath();
        dCtx.arc(curX, py, 4, 0, Math.PI * 2);
        dCtx.fill();
      });

      // --- Draw Chart 2: Angles ---
      const w2 = angleCanvas.width / window.devicePixelRatio;
      const h2 = angleCanvas.height / window.devicePixelRatio;
      aCtx.clearRect(0, 0, w2, h2);

      // Grid lines
      aCtx.strokeStyle = 'rgba(255,255,255,0.06)';
      aCtx.lineWidth = 1;
      for (let y = 20; y < h2; y += 30) {
        aCtx.beginPath();
        aCtx.moveTo(0, y);
        aCtx.lineTo(w2, y);
        aCtx.stroke();
      }

      const minDeg = 0;
      const maxDeg = 180;
      const mapDegY = (val) => h2 - 10 - ((val - minDeg) / (maxDeg - minDeg)) * (h2 - 20);

      // Elbow Flexion Angle
      aCtx.beginPath();
      aCtx.strokeStyle = '#06b6d4';
      aCtx.lineWidth = 2.0;
      for (let i = 0; i < totalFrames; i++) {
        const x = mapX(i);
        const y = mapDegY(frames[i].elbow_angle);
        if (i === 0) aCtx.moveTo(x, y);
        else aCtx.lineTo(x, y);
      }
      aCtx.stroke();

      // Shoulder Elevation Angle
      aCtx.beginPath();
      aCtx.strokeStyle = '#10b981';
      aCtx.lineWidth = 1.8;
      for (let i = 0; i < totalFrames; i++) {
        const x = mapX(i);
        const y = mapDegY(frames[i].shoulder_elev);
        if (i === 0) aCtx.moveTo(x, y);
        else aCtx.lineTo(x, y);
      }
      aCtx.stroke();

      // Cursor line on Chart 2
      aCtx.strokeStyle = '#ffffff';
      aCtx.lineWidth = 1.5;
      aCtx.setLineDash([4, 2]);
      aCtx.beginPath();
      aCtx.moveTo(curX, 0);
      aCtx.lineTo(curX, h2);
      aCtx.stroke();
      aCtx.setLineDash([]);
    }

    // --- Synchronized Video Element & Keypoint Overlay ---
    const video = document.getElementById('source-video');
    const vOverlay = document.getElementById('video-overlay-canvas');
    const voCtx = vOverlay.getContext('2d');

    function syncVideo(targetTime) {
      if (Math.abs(video.currentTime - targetTime) > 0.08) {
        video.currentTime = targetTime;
      }
    }

    function drawVideoOverlay(f) {
      const vw = vOverlay.parentElement.clientWidth;
      const vh = vOverlay.parentElement.clientHeight;
      if (vOverlay.width !== vw || vOverlay.height !== vh) {
        vOverlay.width = vw;
        vOverlay.height = vh;
      }
      voCtx.clearRect(0, 0, vw, vh);

      // Original video is 720 x 1280
      const sx = vw / 720;
      const sy = vh / 1280;

      const pSh = [f.sh_2d[0] * sx, f.sh_2d[1] * sy];
      const pEl = [f.el_2d[0] * sx, f.el_2d[1] * sy];
      const pWr = [f.wr_2d[0] * sx, f.wr_2d[1] * sy];

      // Draw skeleton lines on video
      voCtx.lineWidth = 3;
      voCtx.strokeStyle = '#10b981';
      voCtx.beginPath();
      voCtx.moveTo(pSh[0], pSh[1]);
      voCtx.lineTo(pEl[0], pEl[1]);
      voCtx.stroke();

      voCtx.strokeStyle = '#06b6d4';
      voCtx.beginPath();
      voCtx.moveTo(pEl[0], pEl[1]);
      voCtx.lineTo(pWr[0], pWr[1]);
      voCtx.stroke();

      // Joint circles
      function drawPt(pt, col, r=5) {
        voCtx.fillStyle = col;
        voCtx.beginPath();
        voCtx.arc(pt[0], pt[1], r, 0, Math.PI * 2);
        voCtx.fill();
        voCtx.strokeStyle = '#fff';
        voCtx.lineWidth = 1.5;
        voCtx.stroke();
      }

      drawPt(pSh, '#10b981', 6);
      drawPt(pEl, '#06b6d4', 6);
      drawPt(pWr, '#f43f5e', 6);
    }

    function toggleVideoBox() {
      const box = document.getElementById('video-sync-box');
      const btn = document.getElementById('video-minimize-btn');
      box.classList.toggle('minimized');
      btn.innerText = box.classList.contains('minimized') ? '▲' : '▼';
    }

    // --- Animation & Playback Loop ---
    function animate(time) {
      requestAnimationFrame(animate);

      if (isPlaying) {
        if (!lastAnimTime) lastAnimTime = time;
        const delta = (time - lastAnimTime) / 1000;
        lastAnimTime = time;

        const frameInc = delta * data.metadata.fps * playbackSpeed;
        currentIndex += frameInc;

        if (currentIndex >= totalFrames - 1) {
          if (isLooping) {
            currentIndex = 0;
          } else {
            currentIndex = totalFrames - 1;
            pause();
          }
        }
        updateSceneToFrame(Math.floor(currentIndex));
      } else {
        lastAnimTime = null;
      }

      controls.update();
      renderer.render(scene, camera);
    }

    // --- Control Handlers ---
    function play() {
      isPlaying = true;
      video.play().catch(() => {});
      document.getElementById('play-icon').innerText = '⏸';
      document.getElementById('play-text').innerText = '暂停';
      document.getElementById('btn-play').classList.add('active');
    }

    function pause() {
      isPlaying = false;
      video.pause();
      document.getElementById('play-icon').innerText = '▶';
      document.getElementById('play-text').innerText = '播放';
      document.getElementById('btn-play').classList.remove('active');
    }

    function togglePlay() {
      if (isPlaying) pause();
      else play();
    }

    function stepFrame(step) {
      pause();
      currentIndex = Math.max(0, Math.min(totalFrames - 1, currentIndex + step));
      updateSceneToFrame(Math.floor(currentIndex));
    }

    function toggleLoop() {
      isLooping = !isLooping;
      document.getElementById('btn-loop').classList.toggle('active', isLooping);
    }

    function setSpeed(spd, btn) {
      playbackSpeed = spd;
      document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    }

    // Slider scrub
    document.getElementById('timeline-slider').addEventListener('input', (e) => {
      pause();
      currentIndex = parseInt(e.target.value);
      updateSceneToFrame(currentIndex);
    });

    // Layer Toggles
    function toggleLayer(layer, visible) {
      layerStates[layer] = visible;
      if (layer === 'skeleton') skeletonGroup.visible = visible;
      if (layer === 'trajectories') trajGroup.visible = visible;
      if (layer === 'vectors') vectorGroup.visible = visible;
      if (layer === 'projections') projGroup.visible = visible;
    }

    // Camera Presets
    function setCameraView(view, btn) {
      document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const target = new THREE.Vector3(-140, 240, 90);
      controls.target.copy(target);

      if (view === 'perspective') {
        camera.position.set(120, 360, 680);
      } else if (view === 'coronal') {
        // Frontal view: look along Z from front to back
        camera.position.set(-140, 240, 750);
      } else if (view === 'sagittal') {
        // Side view: look along X from side
        camera.position.set(650, 240, 90);
      } else if (view === 'transverse') {
        // Top-down view: look from above Y
        camera.position.set(-140, 900, 100);
      } else if (view === 'closeup') {
        // Close-up on wrist
        const pWr = toThree(frames[Math.floor(currentIndex)].wr_3d);
        controls.target.copy(pWr);
        camera.position.set(pWr.x + 100, pWr.y + 120, pWr.z + 180);
      }
      controls.update();
    }

    // Theory Modal
    function toggleTheoryModal(show) {
      const m = document.getElementById('theory-modal');
      const b = document.getElementById('modal-backdrop');
      if (show) {
        m.classList.add('visible');
        b.classList.add('visible');
      } else {
        m.classList.remove('visible');
        b.classList.remove('visible');
      }
    }

    // Download JSON
    function downloadJSON() {
      const str = JSON.stringify(data, null, 2);
      const blob = new Blob([str], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'arm_compensation_data.json';
      a.click();
      URL.revokeObjectURL(url);
    }

    // Spacebar to play/pause
    window.addEventListener('keydown', (e) => {
      if (e.code === 'Space') {
        e.preventDefault();
        togglePlay();
      } else if (e.code === 'ArrowLeft') {
        stepFrame(-1);
      } else if (e.code === 'ArrowRight') {
        stepFrame(1);
      }
    });

    // Window Resize
    window.addEventListener('resize', () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
      resizeCanvases();
      drawCharts(Math.floor(currentIndex));
    });

    // Initialize
    window.addEventListener('DOMContentLoaded', () => {
      resizeCanvases();
      updateSceneToFrame(0);
      requestAnimationFrame(animate);
    });
  </script>
</body>
</html>
'''

with open('arm_compensation_visualization.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("arm_compensation_visualization.html generated successfully!")
