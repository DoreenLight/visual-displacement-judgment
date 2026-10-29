"""Causal engineering feedback; this is not validated anatomical FE/RU measurement."""
from collections import deque
import math
import numpy as np

REQUIRED = ('l_shoulder', 'r_shoulder', 'r_elbow', 'r_wrist', 'r_index', 'r_pinky')


def angle(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator < 1e-6:
        raise ValueError('degenerate segment')
    return math.degrees(math.acos(float(np.clip(np.dot(a, b) / denominator, -1, 1))))


class FeedbackEngine:
    def __init__(self):
        self.samples = deque()
        self.baseline = None
        self.training = False
        self.target = 20.0
        self.reps = 0
        self.flagged_reps = 0
        self.interrupted_reps = 0
        self.active = False
        self.reached = False
        self.dirty = False
        self.peak = 0.0
        self.alert_since = {}
        self.alerts = {'shoulder': False, 'elbow': False}
        self.last_time = None
        self.last_value = None
        self.wrist_value = None
        self.calibration_id = None
        self.rest_since = None

    def interrupt(self):
        if self.active:
            self.interrupted_reps += 1
        self.active = self.reached = self.dirty = False
        self.peak = 0.0
        self.alert_since.clear()
        self.alerts = {'shoulder': False, 'elbow': False}
        self.rest_since = None

    def recalibrate(self):
        self.interrupt()
        self.training = False
        self.baseline = None
        self.samples.clear()
        self.last_value = self.wrist_value = None
        self.calibration_id = None

    def features(self, p):
        forearm = p['r_wrist'] - p['r_elbow']
        hand = (p['r_index'] + p['r_pinky']) / 2 - p['r_wrist']
        upper = p['r_elbow'] - p['r_shoulder']
        if not 0.10 < np.linalg.norm(forearm) < 0.55:
            raise ValueError('forearm length')
        if not 0.10 < np.linalg.norm(upper) < 0.60:
            raise ValueError('upper arm length')
        if not 0.035 < np.linalg.norm(hand) < 0.23:
            raise ValueError('hand length')
        return {'bend': angle(forearm, hand), 'elbow': 180 - angle(-upper, forearm),
                'shoulder_height': p['l_shoulder'][1] - p['r_shoulder'][1],
                'elbow_relative': p['r_elbow'] - p['r_shoulder'],
                'upper': upper / np.linalg.norm(upper)}

    def update(self, points, now):
        out = {'status': 'unknown', 'headline': '请露出双肩、右肘、右腕和右手',
               'detail': '观测不足时暂停评价；不把缺失点当成正常动作。',
               'bend_deg': None, 'shoulder_mm': None, 'elbow_deg': None,
               'elbow_mm': None, 'shoulder_alert': False, 'elbow_alert': False,
               'calibration_progress': 0.0, 'calibrated': self.baseline is not None}
        gap = self.last_time is not None and now - self.last_time > 0.35
        self.last_time = now
        if gap:
            self.interrupt()
            self.samples.clear()
            self.last_value = self.wrist_value = None
        if any(k not in points or not np.isfinite(points[k]).all() for k in REQUIRED):
            self.interrupt()
            self.samples.clear()
            self.last_value = self.wrist_value = None
            labels = {'l_shoulder': '左肩', 'r_shoulder': '右肩', 'r_elbow': '右肘',
                      'r_wrist': '右腕', 'r_index': '右手食指侧', 'r_pinky': '右手小指侧'}
            missing = [labels[k] for k in REQUIRED if k not in points or not np.isfinite(points[k]).all()]
            out['detail'] = '尚未可靠观测：' + '、'.join(missing) + '。请让上半身更正对镜头，露出手背。'
            return self.finish(out)
        try:
            f = self.features(points)
        except ValueError:
            self.interrupt()
            self.samples.clear()
            out['headline'] = '手臂测量不稳定，请调整到无遮挡的位置'
            return self.finish(out)
        if self.baseline is None:
            sample = np.concatenate([points[k] for k in REQUIRED])
            self.samples.append((now, sample, f))
            while self.samples and now - self.samples[0][0] > 3.1:
                self.samples.popleft()
            stack = np.array([s[1] for s in self.samples])
            duration = now - self.samples[0][0]
            # Stability is measured over physical elapsed time, not a fixed frame count.
            if len(stack) >= 5:
                spread = np.percentile(np.linalg.norm((stack - np.median(stack, axis=0)).reshape(-1, 6, 3), axis=2), 95)
                if spread > 0.018 or np.std([s[2]['bend'] for s in self.samples]) > 4:
                    self.samples.clear()
                    out.update(status='calibrating', headline='请放松双肩，前臂支撑，手腕自然放平',
                               detail='检测到移动或抖动，正在重新等待稳定基线。')
                    return self.finish(out)
            out.update(status='calibrating', headline='保持自然坐姿，正在建立零位',
                       detail='请让双肩、右臂和手背保持可见，稳定约 3 秒。',
                       calibration_progress=min(1.0, duration / 3))
            if duration >= 3 and len(stack) >= 15:
                self.baseline = {k: np.median([s[2][k] for s in self.samples], axis=0) for k in f}
                self.calibration_id = f'baseline-{now:.3f}'
                self.samples.clear()
                self.last_value = 0.0
                self.wrist_value = 0.0
                out.update(calibrated=True, calibration_progress=1.0, status='ready',
                           headline='零位已建立，可点击“开始训练”', detail='先做小幅、缓慢的腕部屈伸。')
            return self.finish(out)
        b = self.baseline
        shoulder = (f['shoulder_height'] - b['shoulder_height']) * 1000
        elbow_delta = abs(f['elbow'] - b['elbow'])
        elbow_mm = float(np.linalg.norm(f['elbow_relative'] - b['elbow_relative']) * 1000)
        upper_delta = angle(f['upper'], b['upper'])
        value = abs(float(f['bend'] - b['bend']))
        # A large one-frame angle jump is invalid, rather than smoothed into a valid rep.
        if self.last_value is not None and abs(value - self.last_value) > 25:
            self.interrupt()
            self.last_value = value
            self.wrist_value = None
            out['headline'] = '手部关键点跳变，请放平手腕并保持可见'
            return self.finish(out)
        self.last_value = value
        self.wrist_value = value if self.wrist_value is None else 0.55 * value + 0.45 * self.wrist_value
        value = self.wrist_value
        out.update(bend_deg=round(value, 1), shoulder_mm=round(shoulder, 1),
                   elbow_deg=round(elbow_delta, 1), elbow_mm=round(elbow_mm, 1),
                   status='ready', headline='已就位，点击“开始训练”',
                   detail='当前显示手腕相对弯曲变化估计，不是已验证的 FE/RU 角度。')
        if not self.training:
            self.interrupt()
            return self.finish(out)
        if value < 6:
            if self.rest_since is None:
                self.rest_since = now
            if self.active and now - self.rest_since >= 0.25:
                if self.reached:
                    if self.dirty:
                        self.flagged_reps += 1
                    else:
                        self.reps += 1
                self.interrupt()
            out.update(status='rest', headline='缓慢弯曲手腕，保持肩肘自然稳定',
                       detail=f'到达设定参考幅度后缓慢回位；参考目标 {self.target:.0f}°，不必勉强。')
        else:
            self.rest_since = None
            if value >= 9:
                self.active = True
            if self.active:
                self.peak = max(self.peak, value)
                self.reached = self.reached or value >= self.target
                signals = {'shoulder': max(shoulder / 30, upper_delta / 15), 'elbow': elbow_delta / 15}
                for name, signal in signals.items():
                    if signal >= 1:
                        self.alert_since.setdefault(name, now)
                        if now - self.alert_since[name] >= 0.18:
                            self.alerts[name] = True
                    elif signal < 0.7:
                        self.alert_since.pop(name, None)
                        self.alerts[name] = False
                self.dirty |= any(self.alerts.values())
                out.update(status='moving', headline='保持肩肘稳定，缓慢完成动作',
                           detail=f'当前弯曲变化约 {value:.0f}°；回到零位后计入一次完整动作。',
                           shoulder_alert=self.alerts['shoulder'], elbow_alert=self.alerts['elbow'])
                if self.reached:
                    out.update(headline='已到参考幅度，请缓慢回到放平位置')
                if any(self.alerts.values()):
                    out.update(status='adjust', headline='检测到肩肘额外参与，请减小幅度并放松')
                    if self.alerts['shoulder']:
                        out['detail'] = '右肩高度或上臂方向发生变化，请放松肩部并保持支撑。'
                    else:
                        out['detail'] = '肘屈伸变化较大，请保持前臂支撑，缓慢回位。'
        return self.finish(out)

    def finish(self, out):
        out.update(training=self.training, target_deg=self.target, reps=self.reps,
                   flagged_reps=self.flagged_reps, interrupted_reps=self.interrupted_reps,
                   calibration_id=self.calibration_id, measurement_source='real_rgbd_surface',
                   angle_kind='unsigned_change_of_forearm_hand_included_angle',
                   validation='engineering_preview_not_clinically_validated')
        return out
