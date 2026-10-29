#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
web_biofeedback_app.py - D435i 实时闭环视觉生物反馈 Web 前端系统
彻底规避 Windows Session 隔离导致的桌面 OpenCV 窗口不可见问题：
1. 本地轻量 Web 服务器 (基于 Python 原生 http.server，无需外部依赖)
2. 30 FPS 极低延迟 MJPEG 实时视频流 (/video_feed)
3. 现代化医疗级暗黑风前端界面 (HTML5 + CSS3 + Web Audio API 提示音)
4. 毫秒级遥测与自主导演指令同步 (/telemetry)
5. 启动后自动调起用户默认浏览器 (Edge / Chrome) 打开 http://127.0.0.1:8088
"""

import os
import sys
import time
import json
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import numpy as np
import cv2

# 确保本脚本所在目录位于 sys.path 首位
curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

import pyrealsense2 as rs
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from biomechanics_engine import BiomechanicsEngine
from autonomous_director import AutonomousRehabDirector
from realtime_biofeedback_app import ChineseHudRenderer


# 全局状态单例
class BiofeedbackSystemState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.latest_telemetry = {
            "stage_id": "WAIT_USER",
            "stage_name": "【步骤 0/4：测试者就位】",
            "stage_progress_pct": 0,
            "director_headline": "系统正在连接摄像头，请坐好准备...",
            "director_subtext": "请端正自然坐姿，露出右手手臂。",
            "accent_color_hex": "#00e650",
            "fe_angle": 0.0,
            "target_angle": 40.0,
            "shrug_displacement_mm": 0.0,
            "is_shrugging": False,
            "abduction_displacement_mm": 0.0,
            "is_flying_elbow": False,
            "standard_reps": 0,
            "compensated_reps": 0,
            "history_peak_angle": 0.0,
            "fps": 30.0,
            "is_task_complete": False
        }
        self.running = True
        self.reset_calib_requested = False
        self.reset_counters_requested = False

state = BiofeedbackSystemState()


# HTML5 前端页面源码
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>D435i 腕关节康复感知与代偿识别系统</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0d1117;
      color: #c9d1d9;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Microsoft YaHei", sans-serif;
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }
    header {
      background: #161b22;
      border-bottom: 1px solid #30363d;
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    header h1 {
      font-size: 20px;
      color: #58a6ff;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .badge-stage {
      background: #238636;
      color: #ffffff;
      padding: 4px 12px;
      border-radius: 20px;
      font-size: 14px;
      font-weight: 600;
    }
    .main-container {
      flex: 1;
      display: flex;
      padding: 16px;
      gap: 16px;
      min-height: 0;
    }
    .video-panel {
      flex: 1.6;
      background: #000000;
      border-radius: 12px;
      border: 1px solid #30363d;
      display: flex;
      flex-direction: column;
      position: relative;
      overflow: hidden;
      box-shadow: 0 8px 24px rgba(0,0,0,0.6);
    }
    .video-panel img {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    .sidebar {
      flex: 1;
      max-width: 380px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .card {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 10px;
      padding: 16px;
    }
    .card-title {
      font-size: 14px;
      color: #8b949e;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 10px;
    }
    .metric-gauge {
      display: flex;
      align-items: center;
      justify-content: space-around;
      padding: 8px 0;
    }
    .gauge-circle {
      position: relative;
      width: 110px;
      height: 110px;
    }
    .gauge-circle svg {
      transform: rotate(-90deg);
      width: 110px;
      height: 110px;
    }
    .gauge-circle circle {
      fill: none;
      stroke-width: 10;
      stroke-linecap: round;
    }
    .gauge-bg { stroke: #21262d; }
    .gauge-val {
      stroke: #58a6ff;
      transition: stroke-dashoffset 0.1s ease, stroke 0.2s ease;
    }
    .gauge-text {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      font-size: 26px;
      font-weight: 700;
      color: #ffffff;
    }
    .metric-row {
      display: flex;
      justify-content: space-between;
      padding: 8px 0;
      border-bottom: 1px solid #21262d;
      font-size: 15px;
    }
    .metric-row:last-child { border-bottom: none; }
    .metric-val { font-weight: 600; color: #ffffff; }
    .val-normal { color: #3fb950 !important; }
    .val-danger { color: #f85149 !important; font-weight: 700; }
    .val-warning { color: #d29922 !important; }
    
    .btn-group {
      display: flex;
      gap: 10px;
      margin-top: 10px;
    }
    button {
      flex: 1;
      padding: 10px;
      border-radius: 6px;
      border: 1px solid #30363d;
      background: #21262d;
      color: #c9d1d9;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }
    button:hover {
      background: #30363d;
      color: #ffffff;
      border-color: #8b949e;
    }
    button.btn-primary {
      background: #1f6feb;
      border-color: #388bfd;
      color: #ffffff;
    }
    button.btn-primary:hover { background: #388bfd; }

    footer.director-banner {
      background: #161b22;
      border-top: 2px solid #58a6ff;
      padding: 16px 24px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      box-shadow: 0 -4px 20px rgba(0,0,0,0.4);
      transition: border-color 0.3s;
    }
    .director-headline {
      font-size: 22px;
      font-weight: 700;
      color: #ffffff;
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .director-subtext {
      font-size: 16px;
      color: #8b949e;
    }
  </style>
</head>
<body>
  <header>
    <h1>
      <span>📷</span>
      <span>D435i 腕关节康复感知与代偿识别系统</span>
    </h1>
    <div style="display: flex; gap: 14px; align-items: center;">
      <div id="stage-badge" class="badge-stage">【步骤 0/4：测试者就位】</div>
      <div style="font-size: 13px; color: #8b949e;" id="fps-readout">FPS: 30</div>
    </div>
  </header>

  <div class="main-container">
    <div class="video-panel">
      <img src="/video_feed" alt="RealSense D435i 实时画面">
    </div>

    <div class="sidebar">
      <div class="card">
        <div class="card-title">腕关节背伸角度 (FE Angle)</div>
        <div class="metric-gauge">
          <div class="gauge-circle">
            <svg>
              <circle class="gauge-bg" cx="55" cy="55" r="45"></circle>
              <circle id="gauge-arc" class="gauge-val" cx="55" cy="55" r="45" stroke-dasharray="283" stroke-dashoffset="283"></circle>
            </svg>
            <div id="angle-display" class="gauge-text">0°</div>
          </div>
          <div style="display: flex; flex-direction: column; gap: 8px;">
            <div style="font-size: 13px; color: #8b949e;">目标角度: <span style="color:#ffffff; font-weight:600;">40°</span></div>
            <div style="font-size: 13px; color: #8b949e;">本组峰值: <span id="peak-angle" style="color:#e3b341; font-weight:600;">0°</span></div>
            <div style="font-size: 13px; color: #8b949e;">考核进度: <span id="reps-display" style="color:#3fb950; font-weight:700; font-size:16px;">0 / 5 次</span></div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-title">解剖代偿实时量化监控</div>
        <div class="metric-row">
          <span>双肩耸肩差分:</span>
          <span id="shrug-val" class="metric-val val-normal">+0 mm</span>
        </div>
        <div class="metric-row">
          <span>手肘外展偏移:</span>
          <span id="abduct-val" class="metric-val val-normal">+0 mm</span>
        </div>
        <div class="metric-row">
          <span>静息抗误报状态:</span>
          <span id="comp-status" class="metric-val val-normal">静息平衡 (正常)</span>
        </div>
      </div>

      <div class="card">
        <div class="card-title">测试中枢手动指令</div>
        <div class="btn-group">
          <button class="btn-primary" onclick="recalibrate()">重新标定零位 (C)</button>
          <button onclick="resetCounters()">重置计数 (R)</button>
        </div>
      </div>
    </div>
  </div>

  <footer id="director-banner" class="director-banner">
    <div id="director-headline" class="director-headline">系统正在连接摄像头，请稍候...</div>
    <div id="director-subtext" class="director-subtext">请端正坐姿，双肩自然平沉，手臂平放在桌面上。</div>
  </footer>

  <script>
    // Web Audio API 声音合成反馈
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    function playBeep(freq, duration, type='sine') {
      try {
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = type;
        osc.frequency.value = freq;
        gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + duration);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + duration);
      } catch(e) {}
    }

    let prevShrugAlert = false;
    let prevSuccess = false;

    // 轮询更新遥测数据 (每 150ms 一次)
    setInterval(async () => {
      try {
        const res = await fetch('/telemetry');
        const data = await res.json();

        // 阶段 Badge
        document.getElementById('stage-badge').innerText = data.stage_name;
        document.getElementById('fps-readout').innerText = `FPS: ${data.fps}`;

        // 角度仪表盘
        const angle = Math.round(data.fe_angle);
        document.getElementById('angle-display').innerText = `${angle}°`;
        document.getElementById('peak-angle').innerText = `${Math.round(data.history_peak_angle)}°`;
        document.getElementById('reps-display').innerText = `${data.standard_reps} / 5 次`;

        // 弧度百分比 (283 为周长)
        const ratio = Math.max(0, Math.min(1.2, angle / 40.0));
        const offset = 283 - (283 * Math.min(1.0, ratio));
        const arc = document.getElementById('gauge-arc');
        arc.style.strokeDashoffset = offset;

        if (data.is_shrugging || data.is_flying_elbow) {
          arc.style.stroke = '#f85149';
        } else if (ratio >= 1.0) {
          arc.style.stroke = '#3fb950';
        } else {
          arc.style.stroke = '#58a6ff';
        }

        // 代偿读数
        const shrugEl = document.getElementById('shrug-val');
        shrugEl.innerText = `${data.shrug_displacement_mm >= 0 ? '+' : ''}${Math.round(data.shrug_displacement_mm)} mm`;
        shrugEl.className = data.is_shrugging ? 'metric-val val-danger' : 'metric-val val-normal';

        const abductEl = document.getElementById('abduct-val');
        abductEl.innerText = `${data.abduction_displacement_mm >= 0 ? '+' : ''}${Math.round(data.abduction_displacement_mm)} mm`;
        abductEl.className = data.is_flying_elbow ? 'metric-val val-danger' : 'metric-val val-normal';

        // 导演命令横幅
        const banner = document.getElementById('director-banner');
        const headline = document.getElementById('director-headline');
        const subtext = document.getElementById('director-subtext');

        headline.innerText = data.director_headline;
        subtext.innerText = data.director_subtext;

        if (data.is_shrugging || data.is_flying_elbow) {
          banner.style.borderTopColor = '#f85149';
          headline.style.color = '#ff7b72';
          if (!prevShrugAlert) {
            playBeep(420, 0.25, 'sawtooth'); // 代偿报警声
            prevShrugAlert = true;
          }
        } else if (ratio >= 1.0) {
          banner.style.borderTopColor = '#3fb950';
          headline.style.color = '#56d364';
          prevShrugAlert = false;
          if (!prevSuccess) {
            playBeep(1200, 0.18, 'sine'); // 达标成功声
            prevSuccess = true;
          }
        } else {
          banner.style.borderTopColor = '#58a6ff';
          headline.style.color = '#ffffff';
          prevShrugAlert = false;
          prevSuccess = false;
        }

      } catch(e) {}
    }, 150);

    // 交互按键监听
    window.addEventListener('keydown', (e) => {
      if (e.key === 'c' || e.key === 'C') recalibrate();
      if (e.key === 'r' || e.key === 'R') resetCounters();
    });

    async function recalibrate() {
      await fetch('/calibrate', { method: 'POST' });
    }

    async function resetCounters() {
      await fetch('/reset', { method: 'POST' });
    }
  </script>
</body>
</html>
"""


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class WebBiofeedbackHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # 屏蔽常规 HTTP 请求输出，保持终端清爽
        return

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))

        elif self.path == "/telemetry":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with state.lock:
                telemetry_bytes = json.dumps(state.latest_telemetry).encode("utf-8")
            self.wfile.write(telemetry_bytes)

        elif self.path == "/video_feed":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()

            try:
                while state.running:
                    with state.lock:
                        jpeg_bytes = state.latest_jpeg

                    if jpeg_bytes is not None:
                        self.wfile.write(b"--FRAME\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(jpeg_bytes)))
                        self.end_headers()
                        self.wfile.write(jpeg_bytes)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.033)  # ~30 FPS
            except (ConnectionResetError, BrokenPipeError):
                pass

        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path == "/calibrate":
            state.reset_calib_requested = True
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        elif self.path == "/reset":
            state.reset_counters_requested = True
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        else:
            self.send_error(404, "Not Found")


class VisionProcessingWorker(threading.Thread):
    def __init__(self, width: int = 848, height: int = 480, fps: int = 30):
        super().__init__(daemon=True)
        self.width = width
        self.height = height
        self.fps = fps

        self.biomechanics = BiomechanicsEngine(target_action="wrist_FE", target_angle_deg=40.0)
        self.director = AutonomousRehabDirector(target_angle_deg=40.0, required_clean_reps=5, save_dir=curr_dir)
        self.hud = ChineseHudRenderer(width=self.width, height=self.height)

        # MediaPipe
        model_path = os.path.join(curr_dir, "pose_landmarker_lite.task")
        with open(model_path, "rb") as f:
            model_bytes = f.read()

        base_options = python.BaseOptions(model_asset_buffer=model_bytes)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.35,
            min_tracking_confidence=0.35
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

        self.joints_3d_filtered = {}
        self.filter_alpha = 0.35

    def extract_patch_depth_m(self, depth_raw: np.ndarray, u_px: int, v_px: int, scale: float) -> float:
        h, w = depth_raw.shape
        u_px = max(0, min(w - 1, u_px))
        v_px = max(0, min(h - 1, v_px))
        r = 3
        patch = depth_raw[max(0, v_px - r):min(h, v_px + r + 1), max(0, u_px - r):min(w, u_px + r + 1)]
        valid = patch > 0
        if np.count_nonzero(valid) > 0:
            d_m = float(np.median(patch[valid])) * scale
            if 0.4 <= d_m <= 2.2:
                return d_m
        return 0.0

    def run(self):
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        profile = pipe.start(cfg)
        align = rs.align(rs.stream.color)

        depth_sensor = profile.get_device().first_depth_sensor()
        scale = depth_sensor.get_depth_scale()

        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

        joint_id_map = {
            "l_shoulder": 11,
            "r_shoulder": 12,
            "r_elbow": 14,
            "r_wrist": 16,
            "r_pinky": 18,
            "r_index": 20
        }

        prev_time = time.time()
        fps_calc = 30.0

        try:
            while state.running:
                # 检查按键指令
                if state.reset_calib_requested:
                    self.biomechanics.reset_calibration()
                    self.director.stage = "AUTO_CALIB"
                    self.director.stage_start_time = time.time()
                    state.reset_calib_requested = False

                if state.reset_counters_requested:
                    self.biomechanics.reset_counters()
                    state.reset_counters_requested = False

                fs = pipe.wait_for_frames()
                aligned = align.process(fs)

                cf = aligned.get_color_frame()
                df = aligned.get_depth_frame()
                if not cf or not df:
                    continue

                color_bgr = np.asanyarray(cf.get_data())
                depth_raw = np.asanyarray(df.get_data())

                now = time.time()
                dt = now - prev_time
                prev_time = now
                if dt > 0:
                    fps_calc = 0.9 * fps_calc + 0.1 * (1.0 / dt)

                # Pose 检测
                rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                det = self.detector.detect(mp_img)

                joints_3d = {}
                joints_disp_mirrored = {}
                landmarks = det.pose_landmarks[0] if det.pose_landmarks else None

                if landmarks:
                    for name, lm_id in joint_id_map.items():
                        if lm_id < len(landmarks):
                            lm = landmarks[lm_id]
                            if (lm.visibility or 1.0) >= 0.25:
                                u_px = lm.x * self.width
                                v_px = lm.y * self.height
                                joints_disp_mirrored[name] = (int(round(self.width - 1 - u_px)), int(round(v_px)))

                                d_m = self.extract_patch_depth_m(depth_raw, int(round(u_px)), int(round(v_px)), scale)
                                if d_m > 0:
                                    pt_3d = np.array([
                                        (u_px - cx) * d_m / fx,
                                        (v_px - cy) * d_m / fy,
                                        d_m
                                    ], dtype=np.float64)

                                    if name not in self.joints_3d_filtered:
                                        self.joints_3d_filtered[name] = pt_3d
                                    else:
                                        self.joints_3d_filtered[name] = (
                                            self.filter_alpha * pt_3d +
                                            (1.0 - self.filter_alpha) * self.joints_3d_filtered[name]
                                        )
                                    joints_3d[name] = self.joints_3d_filtered[name].copy()

                # 生物力学分析
                eval_res = self.biomechanics.evaluate_frame(joints_3d)

                # 自主主控导演推进
                director_cmd = self.director.update(eval_res, joints_3d)

                # 镜像翻转并绘制 HUD
                mirror_bgr = cv2.flip(color_bgr, 1)
                hud_bgr = self.hud.draw_hud(mirror_bgr, eval_res, joints_disp_mirrored, fps_calc, director_cmd=director_cmd)

                # 压缩为 JPEG 供 Web 流输出
                ret, jpg = cv2.imencode(".jpg", hud_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    with state.lock:
                        state.latest_jpeg = jpg.tobytes()
                        state.latest_telemetry = {
                            "stage_id": director_cmd["stage_id"],
                            "stage_name": director_cmd["stage_name"],
                            "stage_progress_pct": director_cmd["stage_progress_pct"],
                            "director_headline": director_cmd["director_headline"],
                            "director_subtext": director_cmd["director_subtext"],
                            "fe_angle": eval_res["fe_angle"],
                            "target_angle": eval_res["target_angle"],
                            "shrug_displacement_mm": eval_res["shrug_displacement_mm"],
                            "is_shrugging": eval_res["is_shrugging"],
                            "abduction_displacement_mm": eval_res["abduction_displacement_mm"],
                            "is_flying_elbow": eval_res["is_flying_elbow"],
                            "standard_reps": eval_res["standard_reps"],
                            "compensated_reps": eval_res["compensated_reps"],
                            "history_peak_angle": eval_res["history_peak_angle"],
                            "fps": round(fps_calc, 1),
                            "is_task_complete": director_cmd["is_task_complete"]
                        }

        finally:
            pipe.stop()
            self.detector.close()


def main():
    port = 8088
    server_address = ("127.0.0.1", port)

    # 启动视觉后台处理线程
    worker = VisionProcessingWorker(width=848, height=480, fps=30)
    worker.start()

    # 启动多线程 Web 服务器
    httpd = ThreadedHTTPServer(server_address, WebBiofeedbackHandler)
    url = f"http://127.0.0.1:{port}"
    print(f"\n========================================================")
    print(f">>> [Web服务就绪] D435i 康复前端已启动: {url}")
    print(f"正在调起默认浏览器访问...")
    print(f"========================================================\n")

    # 自动打开浏览器
    webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        httpd.server_close()
        print("\n[退出] Web 服务与相机已安全关闭。")


if __name__ == "__main__":
    main()
