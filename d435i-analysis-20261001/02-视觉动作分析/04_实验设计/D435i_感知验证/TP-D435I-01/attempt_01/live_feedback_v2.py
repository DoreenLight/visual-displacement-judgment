"""Local D435i streaming feedback. Run using the project's .venv Python."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import queue
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
from PIL import Image, ImageDraw, ImageFont
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from feedback_engine_v2 import FeedbackEngine

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
JOINTS = {'l_shoulder': 11, 'r_shoulder': 12, 'r_elbow': 14,
          'r_wrist': 16, 'r_index': 20, 'r_pinky': 18}
LOG = logging.getLogger('feedback')


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.stop = threading.Event()
        self.commands = queue.Queue(maxsize=20)
        self.jpeg = None
        self.sequence = 0
        self.telemetry = {'status': 'connecting', 'headline': '正在连接 D435i 摄像头',
                          'detail': '模型在本机运行，画面不会上传。', 'output_mono_s': time.monotonic()}
        self.camera = {}
        self.latest_frame = None
        self.recording = False
        self.record_error = None
        self.record_run = None

    def publish(self, image, telemetry):
        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            with self.condition:
                self.jpeg = encoded.tobytes()
                self.sequence += 1
                self.telemetry = telemetry
                self.condition.notify_all()

    def snapshot(self):
        with self.lock:
            data = dict(self.telemetry)
            frame = self.latest_frame
            data['capture_age_ms'] = round((time.monotonic() - frame['received']) * 1000) if frame else None
            data['recording'] = self.recording
            data['record_error'] = self.record_error
            data['record_run'] = self.record_run
            data['camera'] = self.camera
        age = (time.monotonic() - data['output_mono_s']) * 1000
        data['result_age_ms'] = round(age)
        if age > 1000:
            data.update(status='unknown', headline='分析暂时中断，请暂停动作',
                        detail='等待新画面；过期角度和提示已隐藏。', bend_deg=None,
                        shoulder_mm=None, elbow_deg=None, shoulder_alert=False, elbow_alert=False)
        return data


class Recorder(threading.Thread):
    def __init__(self, state):
        super().__init__(daemon=True)
        self.state = state
        self.items = queue.Queue(maxsize=90)
        self.directory = None
        self.handle = None
        self.meta = None
        self.saved = 0
        self.dropped = 0

    def start_run(self, camera, engine):
        self.directory = ROOT / '07_资料库' / 'D435i_感知验证' / datetime.now().strftime('preview_%Y%m%d_%H%M%S_%f')
        self.directory.mkdir(parents=True)
        self.meta = {'schema_version': 'feedback-preview/v2', 'run_id': self.directory.name,
                     'participant_id': 'local_preview', 'training_side': 'right',
                     'data_source': 'real_rgbd', 'camera': camera, 'started_at': datetime.now().isoformat(),
                     'reference_system': None, 'accuracy_verdict': 'BLOCKED',
                     'calibration_id': engine['calibration_id'], 'target_deg': engine['target_deg'],
                     'angle_kind': 'unsigned_change_of_forearm_hand_included_angle',
                     'notices': 'Aligned depth surface samples; not validated joint centers or FE/RU.',
                     'model_sha256': hashlib.sha256((HERE / 'pose_landmarker_lite.task').read_bytes()).hexdigest()}
        self.handle = (self.directory / 'frames.jsonl').open('w', encoding='utf-8')
        self.saved = self.dropped = 0
        self.write_meta()

    def write_meta(self):
        (self.directory / 'run.json').write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding='utf-8')

    def close_run(self):
        if self.handle:
            self.handle.close()
            self.handle = None
            self.meta.update(ended_at=datetime.now().isoformat(), saved_frames=self.saved, dropped_frames=self.dropped)
            self.write_meta()
            self.directory = None

    def submit(self, frame, telemetry):
        if not self.state.recording:
            return
        try:
            self.items.put_nowait(('frame', frame, telemetry))
        except queue.Full:
            self.dropped += 1
            with self.state.lock:
                self.state.record_error = '存储跟不上，部分帧未保存；丢帧数会写入记录。'

    def run(self):
        try:
            while True:
                item = self.items.get()
                try:
                    if item[0] == 'quit':
                        break
                    if item[0] == 'start':
                        self.close_run()
                        self.start_run(item[1], item[2])
                        with self.state.lock:
                            self.state.record_error = None
                            self.state.record_run = self.directory.name
                    elif item[0] == 'stop':
                        self.close_run()
                    elif item[0] == 'frame' and self.handle:
                        frame, telemetry = item[1:]
                        stem = f'{self.saved:07d}'
                        ok, rgb = cv2.imencode('.jpg', frame['color'], [cv2.IMWRITE_JPEG_QUALITY, 92])
                        if not ok:
                            raise IOError('RGB encoding failed')
                        (self.directory / f'{stem}_rgb.jpg').write_bytes(rgb.tobytes())
                        np.save(self.directory / f'{stem}_depth.npy', frame['depth'])
                        entry = {'frame_id': frame['id'], 'rgb_device_s': frame['rgb_s'],
                                 'depth_device_s': frame['depth_s'], 'rgb_clock_domain': frame['rgb_clock'],
                                 'depth_clock_domain': frame['depth_clock'], 'host_mono_s': frame['received'],
                                 'rgb_file': f'{stem}_rgb.jpg', 'depth_file': f'{stem}_depth.npy',
                                 'telemetry': telemetry}
                        self.handle.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + '\n')
                        self.saved += 1
                except Exception as error:
                    LOG.exception('Recording error')
                    with self.state.lock:
                        self.state.recording = False
                        self.state.record_error = str(error)
                    self.close_run()
                finally:
                    self.items.task_done()
        finally:
            self.close_run()


def capture(state):
    sequence = 0
    while not state.stop.is_set():
        pipe = rs.pipeline()
        started = False
        try:
            cfg = rs.config()
            cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
            cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
            profile = pipe.start(cfg)
            started = True
            device = profile.get_device()
            scale = device.first_depth_sensor().get_depth_scale()
            exposures = {}
            for sensor in device.query_sensors():
                if sensor.supports(rs.option.exposure):
                    exposures[sensor.get_info(rs.camera_info.name)] = {
                        'exposure_value': sensor.get_option(rs.option.exposure),
                        'auto_exposure': sensor.get_option(rs.option.enable_auto_exposure) if sensor.supports(rs.option.enable_auto_exposure) else None}
            align = rs.align(rs.stream.color)
            with state.lock:
                state.camera = {'name': device.get_info(rs.camera_info.name),
                                'serial': device.get_info(rs.camera_info.serial_number),
                                'firmware': device.get_info(rs.camera_info.firmware_version), 'depth_scale_m': scale,
                                'rgb_mode': '848x480 bgr8 30', 'depth_mode': '848x480 z16 30','exposure':exposures}
            while not state.stop.is_set():
                frames = align.process(pipe.wait_for_frames(2000))
                color, depth = frames.get_color_frame(), frames.get_depth_frame()
                if not color or not depth:
                    continue
                sequence += 1
                intr = depth.profile.as_video_stream_profile().get_intrinsics()
                frame = {'id': sequence, 'received': time.monotonic(),
                         'color': np.asanyarray(color.get_data()).copy(),
                         'depth': np.asanyarray(depth.get_data()).copy(), 'intr': intr, 'scale': scale,
                         'rgb_s': color.get_timestamp() / 1000, 'depth_s': depth.get_timestamp() / 1000,
                         'rgb_clock': str(color.get_frame_timestamp_domain()),
                         'depth_clock': str(depth.get_frame_timestamp_domain())}
                with state.lock:
                    state.latest_frame = frame
                    state.camera['intrinsics'] = {'width':intr.width,'height':intr.height,'fx': intr.fx, 'fy': intr.fy, 'ppx': intr.ppx, 'ppy': intr.ppy,
                                                   'model': str(intr.model), 'coeffs': list(intr.coeffs)}
                if hasattr(state, 'record_sink'):
                    state.record_sink.submit_raw(frame)
        except Exception as error:
            LOG.exception('Camera connection failed')
            with state.lock:
                state.latest_frame = None
            state.publish(placeholder('等待 D435i 连接'), {'status': 'error', 'headline': '摄像头未就绪，正在重试',
                          'detail': str(error), 'output_mono_s': time.monotonic()})
        finally:
            if started:
                pipe.stop()
        state.stop.wait(2)


FONT = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 22)
SMALL = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 16)


def placeholder(text):
    image = Image.new('RGB', (848, 480), '#10202b')
    ImageDraw.Draw(image).text((55, 215), text, font=FONT, fill='white')
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def render(frame, pixels, output):
    picture = cv2.flip(frame['color'], 1)
    color = (60, 190, 250) if output['status'] in ('unknown', 'calibrating') else (110, 230, 80)
    for a, b in [('r_shoulder', 'r_elbow'), ('r_elbow', 'r_wrist'), ('r_wrist', 'r_index'), ('r_wrist', 'r_pinky')]:
        if a in pixels and b in pixels:
            cv2.line(picture, pixels[a], pixels[b], color, 3, cv2.LINE_AA)
    for name, point in pixels.items():
        alert = (name == 'r_shoulder' and output['shoulder_alert']) or (name == 'r_elbow' and output['elbow_alert'])
        cv2.circle(picture, point, 7, (70, 70, 255) if alert else color, -1, cv2.LINE_AA)
    image = Image.fromarray(cv2.cvtColor(picture, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 848, 42), fill='#10202b')
    draw.text((14, 8), '右手实时反馈 · 镜像画面', font=SMALL, fill='white')
    draw.rectangle((0, 394, 848, 480), fill='#10202b')
    draw.text((14, 405), output['headline'], font=FONT, fill='#ffce75' if output['status'] == 'adjust' else 'white')
    draw.text((14, 446), output['detail'][:47], font=SMALL, fill='#b3c6d7')
    for name, label in [('r_shoulder', '右肩'), ('r_elbow', '右肘'), ('r_wrist', '右腕')]:
        if name in pixels:
            x, y = pixels[name]
            draw.text((max(0, min(790, x + 10)), max(45, min(365, y - 20))), label, font=SMALL, fill='#ffffff', stroke_width=1, stroke_fill='#10202b')
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def measure(result, frame):
    points, pixels = {}, {}
    if not result.pose_landmarks:
        return points, pixels
    h, w = frame['depth'].shape
    for name, index in JOINTS.items():
        lm = result.pose_landmarks[0][index]
        visibility = lm.visibility if lm.visibility is not None else 0.0
        presence = lm.presence if lm.presence is not None else 0.0
        if min(visibility, presence) < 0.5 or not (0 <= lm.x < 1 and 0 <= lm.y < 1):
            continue
        u, v = int(lm.x * w), int(lm.y * h)
        patch = frame['depth'][max(0, v-3):min(h, v+4), max(0, u-3):min(w, u+4)]
        values = patch[patch > 0].astype(float) * frame['scale']
        if len(values) < patch.size * 0.65:
            continue
        depth = float(np.median(values))
        if not 0.35 < depth < 2.5 or np.percentile(values, 90) - np.percentile(values, 10) > 0.07:
            continue
        points[name] = np.array(rs.rs2_deproject_pixel_to_point(frame['intr'], [float(u), float(v)], depth))
        pixels[name] = (w - 1 - u, v)
    return points, pixels


def process(state, recorder):
    engine = FeedbackEngine()
    model = (HERE / 'pose_landmarker_lite.task').read_bytes()
    options = vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_buffer=model),
              running_mode=vision.RunningMode.VIDEO, min_pose_detection_confidence=0.5,
              min_pose_presence_confidence=0.5, min_tracking_confidence=0.5)
    last_id = None
    last_timestamp = 0
    last_received = None
    last_camera_serial = None
    previous_done = None
    fps = 0.0
    try:
        with vision.PoseLandmarker.create_from_options(options) as detector:
            while not state.stop.is_set():
                while not state.commands.empty():
                    action, payload = state.commands.get_nowait()
                    if action == 'calibrate':
                        engine.recalibrate()
                    elif action == 'train':
                        if engine.baseline is not None:
                            engine.interrupt()
                            engine.training = True
                    elif action == 'pause':
                        engine.training = False
                        engine.interrupt()
                    elif action == 'target':
                        engine.target = float(payload['value'])
                        engine.interrupt()
                    elif action == 'record':
                        if not state.recording:
                            recorder.items.put(('start', dict(state.camera),
                                                {'calibration_id': engine.calibration_id, 'target_deg': engine.target}))
                            with state.lock:
                                state.recording = True
                    elif action == 'stop_record':
                        with state.lock:
                            state.recording = False
                        recorder.items.put(('stop',))
                    elif action == 'stop':
                        state.stop.set()
                    state.commands.task_done()
                with state.lock:
                    frame = state.latest_frame
                    serial = state.camera.get('serial')
                if frame is None or time.monotonic() - frame['received'] > 0.4:
                    engine.interrupt()
                    engine.recalibrate()
                    state.stop.wait(0.05)
                    continue
                if frame['received'] == last_received:
                    state.stop.wait(0.003)
                    continue
                if last_id is not None and (frame['id'] <= last_id or serial != last_camera_serial):
                    engine.recalibrate()
                last_id, last_received, last_camera_serial = frame['id'], frame['received'], serial
                began = time.monotonic()
                timestamp = max(last_timestamp + 1, int(frame['received'] * 1000))
                last_timestamp = timestamp
                rgb = cv2.cvtColor(frame['color'], cv2.COLOR_BGR2RGB)
                result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)
                points, pixels = measure(result, frame)
                output = engine.update(points, frame['received'])
                done = time.monotonic()
                if done - frame['received'] > 0.4:
                    output = engine.update({}, done)
                    output['headline'] = '分析延迟较高，请暂停动作'
                    output['detail'] = '正在等待及时的新观测，暂不计数。'
                if previous_done is not None:
                    fps = 0.8 * fps + 0.2 / max(0.001, done - previous_done)
                previous_done = done
                output.update(frame_id=frame['id'], output_mono_s=done, received_mono_s=frame['received'],
                              inference_ms=round((done-began)*1000, 1),
                              processing_latency_ms=round((done-frame['received'])*1000, 1), fps=round(fps, 1),
                              valid_joints=list(points), joints_3d_m={k: p.tolist() for k, p in points.items()})
                state.publish(render(frame, pixels, output), output)
                recorder.submit(frame, output)
    except Exception as error:
        LOG.exception('Processing stopped')
        state.publish(placeholder('本地分析暂时中断'), {'status': 'error', 'headline': '模型分析暂时中断',
                     'detail': str(error), 'output_mono_s': time.monotonic()})


def handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, data, mime='application/json; charset=utf-8'):
            body = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                if self.path == '/':
                    self.reply(200, (HERE / 'live_feedback_v2.html').read_bytes(), 'text/html; charset=utf-8')
                elif self.path == '/telemetry':
                    self.reply(200, state.snapshot())
                elif self.path == '/snapshot.jpg':
                    with state.lock:
                        jpeg = state.jpeg
                    self.reply(200, jpeg, 'image/jpeg') if jpeg else self.reply(503, {'error': 'no frame'})
                elif self.path == '/video_feed':
                    self.send_response(200)
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers()
                    previous = -1
                    while not state.stop.is_set():
                        with state.condition:
                            state.condition.wait_for(lambda: state.sequence != previous or state.stop.is_set(), timeout=1)
                            sequence, jpeg = state.sequence, state.jpeg
                        if jpeg and sequence != previous:
                            self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(jpeg)).encode() + b'\r\n\r\n' + jpeg + b'\r\n')
                            previous = sequence
                else:
                    self.reply(404, {'error': 'not found'})
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            # Reject browser requests from other sites even though the service is loopback-only.
            origin = self.headers.get('Origin')
            if origin and origin != 'http://' + self.headers.get('Host', ''):
                self.reply(403, {'error': 'origin rejected'})
                return
            action = self.path.lstrip('/')
            if action not in ('calibrate', 'train', 'pause', 'target', 'record', 'stop_record', 'stop'):
                self.reply(404, {'error': 'unknown command'})
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 <= size <= 2048:
                    raise ValueError('body too large')
                payload = json.loads(self.rfile.read(size) or b'{}')
                if action == 'target' and not 10 <= float(payload['value']) <= 40:
                    raise ValueError('target must be 10..40')
                if action == 'train' and not state.snapshot().get('calibrated'):
                    self.reply(409, {'error': '请先完成稳定零位标定'})
                    return
                if action == 'stop':
                    self.reply(200, {'accepted': action})
                    state.stop.set()
                    return
                state.commands.put_nowait((action, payload))
                self.reply(200, {'accepted': action})
            except (ValueError, TypeError, KeyError, queue.Full) as error:
                self.reply(400, {'error': str(error)})
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8090)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    state = State()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(state))
    server.daemon_threads = True
    recorder = Recorder(state)
    recorder.start()
    workers = [threading.Thread(target=capture, args=(state,), daemon=True),
               threading.Thread(target=process, args=(state, recorder), daemon=True)]
    for worker in workers:
        worker.start()
    LOG.info('Open http://127.0.0.1:%s', args.port)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        while not state.stop.wait(0.25):
            pass
    except KeyboardInterrupt:
        state.stop.set()
    finally:
        with state.condition:
            state.condition.notify_all()
        for worker in workers:
            worker.join(timeout=5)
        state.recording = False
        recorder.items.put(('quit',))
        recorder.join(timeout=15)
        server.shutdown()
        server.server_close()
        LOG.info('Camera and service stopped')


if __name__ == '__main__':
    main()
