import os
import cv2
import mediapipe as mp
import numpy as np
import json
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
video_path = os.path.join(base_dir, '07_资料库/实验2-腕肘肩代偿判断-视频3-腕肘肩_h264.mp4')
if not os.path.exists(video_path):
    video_path = os.path.join(base_dir, '07_资料库/实验2-腕肘肩代偿判断-视频3-腕肘肩.mp4')
if not os.path.exists(video_path):
    video_path = '07_资料库/实验2-腕肘肩代偿判断-视频3-腕肘肩_h264.mp4'

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
crop_x_start = int(w * 0.15)
cw = w - crop_x_start

print(f"Opening video: {total_frames} frames, {w}x{h}, {fps} fps")

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.2
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.2
)

raw_data = []
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    time_s = frame_idx / fps
    crop = frame[:, crop_x_start:]
    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    res_pose = pose.process(rgb_crop)
    
    # Process hands on full frame
    rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res_hand = hands.process(rgb_full)
    
    # Pose landmarks
    if res_pose.pose_landmarks and res_pose.pose_world_landmarks:
        lms = res_pose.pose_landmarks.landmark
        wlms = res_pose.pose_world_landmarks.landmark
        
        # Right shoulder, elbow, wrist, left shoulder
        r_sh = lms[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        r_el = lms[mp_pose.PoseLandmark.RIGHT_ELBOW]
        r_wr = lms[mp_pose.PoseLandmark.RIGHT_WRIST]
        l_sh = lms[mp_pose.PoseLandmark.LEFT_SHOULDER]
        
        r_sh_w = wlms[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        r_el_w = wlms[mp_pose.PoseLandmark.RIGHT_ELBOW]
        r_wr_w = wlms[mp_pose.PoseLandmark.RIGHT_WRIST]
        l_sh_w = wlms[mp_pose.PoseLandmark.LEFT_SHOULDER]
        
        sh_2d = [float(r_sh.x * cw + crop_x_start), float(r_sh.y * h)]
        el_2d = [float(r_el.x * cw + crop_x_start), float(r_el.y * h)]
        wr_2d = [float(r_wr.x * cw + crop_x_start), float(r_wr.y * h)]
        l_sh_2d = [float(l_sh.x * cw + crop_x_start), float(l_sh.y * h)]
        
        # World coordinates in meters: convert to millimeters, with Three.js friendly coordinates
        # MediaPipe World: X right(+), Y down(+), Z forward(-)
        # In biomechanics/Three.js:
        # X: Medial-Lateral (Arm right/lateral is +X)
        # Y: Inferior-Superior (Up is +Y) -> inverted from MP Y
        # Z: Posterior-Anterior (Forward towards camera is +Z) -> inverted from MP Z
        sh_3d = [float(r_sh_w.x * 1000), float(-r_sh_w.y * 1000), float(-r_sh_w.z * 1000)]
        el_3d = [float(r_el_w.x * 1000), float(-r_el_w.y * 1000), float(-r_el_w.z * 1000)]
        wr_3d = [float(r_wr_w.x * 1000), float(-r_wr_w.y * 1000), float(-r_wr_w.z * 1000)]
        l_sh_3d = [float(l_sh_w.x * 1000), float(-l_sh_w.y * 1000), float(-l_sh_w.z * 1000)]
    else:
        # Fallback to previous frame if any
        sh_2d = raw_data[-1]['sh_2d'] if raw_data else [0, 0]
        el_2d = raw_data[-1]['el_2d'] if raw_data else [0, 0]
        wr_2d = raw_data[-1]['wr_2d'] if raw_data else [0, 0]
        l_sh_2d = raw_data[-1]['l_sh_2d'] if raw_data else [0, 0]
        sh_3d = raw_data[-1]['sh_3d'] if raw_data else [0, 0, 0]
        el_3d = raw_data[-1]['el_3d'] if raw_data else [0, 0, 0]
        wr_3d = raw_data[-1]['wr_3d'] if raw_data else [0, 0, 0]
        l_sh_3d = raw_data[-1]['l_sh_3d'] if raw_data else [0, 0, 0]

    # Hand landmarks
    hand_lms_3d = []
    normal_vec = [0.0, 1.0, 0.0]
    pronation_deg = 0.0
    
    if res_hand.multi_hand_landmarks:
        hlm = res_hand.multi_hand_landmarks[0]
        # 21 landmarks relative to wrist in mm
        wrist_pt = hlm.landmark[0]
        for lm in hlm.landmark:
            # Scale hand to realistic hand size ~180mm
            dx = (lm.x - wrist_pt.x) * w * 0.4
            dy = -(lm.y - wrist_pt.y) * h * 0.4
            dz = -lm.z * 180.0
            hand_lms_3d.append([float(dx), float(dy), float(dz)])
        
        # Calculate palm normal from wrist(0), index_mcp(5), pinky_mcp(17)
        v1 = np.array(hand_lms_3d[5]) - np.array(hand_lms_3d[0])
        v2 = np.array(hand_lms_3d[17]) - np.array(hand_lms_3d[0])
        n = np.cross(v1, v2)
        norm_len = np.linalg.norm(n)
        if norm_len > 1e-4:
            n = n / norm_len
            normal_vec = [float(n[0]), float(n[1]), float(n[2])]
            # Pronation angle around Z axis or relative to vertical:
            angle = float(np.degrees(np.arctan2(n[0], n[1])))
            pronation_deg = (angle + 360.0) % 360.0
    elif raw_data:
        hand_lms_3d = raw_data[-1]['hand_lms_3d']
        normal_vec = raw_data[-1]['normal_vec']
        pronation_deg = raw_data[-1]['pronation_deg']
    else:
        hand_lms_3d = [[0,0,0]] * 21

    raw_data.append({
        'frame': frame_idx,
        'time': round(time_s, 4),
        'sh_2d': sh_2d,
        'el_2d': el_2d,
        'wr_2d': wr_2d,
        'l_sh_2d': l_sh_2d,
        'sh_3d': sh_3d,
        'el_3d': el_3d,
        'wr_3d': wr_3d,
        'l_sh_3d': l_sh_3d,
        'hand_lms_3d': hand_lms_3d,
        'normal_vec': normal_vec,
        'pronation_deg': pronation_deg
    })
    frame_idx += 1

cap.release()
pose.close()
hands.close()

print(f"Extracted {len(raw_data)} raw frames. Processing smoothing & kinematics...")

# Convert to numpy arrays for filtering
sh_arr = np.array([d['sh_3d'] for d in raw_data])
el_arr = np.array([d['el_3d'] for d in raw_data])
wr_arr = np.array([d['wr_3d'] for d in raw_data])
l_sh_arr = np.array([d['l_sh_3d'] for d in raw_data])

# Apply Savitzky-Golay filter to smooth 3D trajectories
window = 9
poly = 3
sh_arr_smooth = np.copy(sh_arr)
el_arr_smooth = np.copy(el_arr)
wr_arr_smooth = np.copy(wr_arr)
l_sh_arr_smooth = np.copy(l_sh_arr)

for dim in range(3):
    sh_arr_smooth[:, dim] = savgol_filter(sh_arr[:, dim], window, poly)
    el_arr_smooth[:, dim] = savgol_filter(el_arr[:, dim], window, poly)
    wr_arr_smooth[:, dim] = savgol_filter(wr_arr[:, dim], window, poly)
    l_sh_arr_smooth[:, dim] = savgol_filter(l_sh_arr[:, dim], window, poly)

dt = 1.0 / fps

# Calculate velocities (mm/s)
sh_vel = np.gradient(sh_arr_smooth, dt, axis=0)
el_vel = np.gradient(el_arr_smooth, dt, axis=0)
wr_vel = np.gradient(wr_arr_smooth, dt, axis=0)

sh_speed = np.linalg.norm(sh_vel, axis=1)
el_speed = np.linalg.norm(el_vel, axis=1)
wr_speed = np.linalg.norm(wr_vel, axis=1)

# Calculate accelerations (mm/s^2)
sh_acc = np.gradient(sh_vel, dt, axis=0)
el_acc = np.gradient(el_vel, dt, axis=0)
wr_acc = np.gradient(wr_vel, dt, axis=0)

sh_acc_mag = np.linalg.norm(sh_acc, axis=1)
el_acc_mag = np.linalg.norm(el_acc, axis=1)
wr_acc_mag = np.linalg.norm(wr_acc, axis=1)

# Calculate joint angles
# 1. Elbow angle: angle between Upper Arm (Elbow -> Shoulder) and Forearm (Elbow -> Wrist)
u_arm = sh_arr_smooth - el_arr_smooth
f_arm = wr_arr_smooth - el_arr_smooth
u_norm = np.linalg.norm(u_arm, axis=1, keepdims=True) + 1e-6
f_norm = np.linalg.norm(f_arm, axis=1, keepdims=True) + 1e-6
dot_ef = np.sum((u_arm / u_norm) * (f_arm / f_norm), axis=1)
dot_ef = np.clip(dot_ef, -1.0, 1.0)
elbow_angle_deg = np.degrees(np.arccos(dot_ef))

# 2. Shoulder elevation angle (angle of upper arm vector relative to downward vertical -Y)
down_vec = np.array([0.0, -1.0, 0.0])
dot_sh = np.sum((-(u_arm / u_norm)) * down_vec, axis=1)
dot_sh = np.clip(dot_sh, -1.0, 1.0)
shoulder_elev_deg = np.degrees(np.arccos(dot_sh))

# Frequency & Period Analysis based on wrist and elbow Y-displacement
wr_y = wr_arr_smooth[:, 1]
peaks, _ = find_peaks(wr_y, distance=15, prominence=15)
valleys, _ = find_peaks(-wr_y, distance=15, prominence=15)

peak_times = [p / fps for p in peaks]
if len(peak_times) >= 2:
    periods = np.diff(peak_times)
    fundamental_period = float(np.mean(periods))
    fundamental_freq = float(1.0 / fundamental_period)
else:
    fundamental_period = 1.0
    fundamental_freq = 1.0

print(f"Cycle analysis: {len(peaks)} peaks, {len(valleys)} valleys, f0 = {fundamental_freq:.2f} Hz, T0 = {fundamental_period:.2f} s")

# Fourier Series 3-Harmonics Fit for Wrist Y(t)
t_series = np.array([d['time'] for d in raw_data])
omega = 2 * np.pi * fundamental_freq

def fourier_func(t, c0, c1, a1, b1, a2, b2, a3, b3):
    return (c0 + c1*t + 
            a1*np.cos(1*omega*t) + b1*np.sin(1*omega*t) +
            a2*np.cos(2*omega*t) + b2*np.sin(2*omega*t) +
            a3*np.cos(3*omega*t) + b3*np.sin(3*omega*t))

try:
    popt, _ = curve_fit(fourier_func, t_series, wr_y)
    fourier_params = {
        'c0': round(float(popt[0]), 2),
        'c1': round(float(popt[1]), 2),
        'a1': round(float(popt[2]), 2), 'b1': round(float(popt[3]), 2),
        'a2': round(float(popt[4]), 2), 'b2': round(float(popt[5]), 2),
        'a3': round(float(popt[6]), 2), 'b3': round(float(popt[7]), 2)
    }
except Exception as e:
    print("Fourier fit fallback:", e)
    fourier_params = {'c0': 0, 'c1': 0, 'a1': 0, 'b1': 0, 'a2': 0, 'b2': 0, 'a3': 0, 'b3': 0}

# Calculate Compensatory Metrics
wr_span_x = float(np.ptp(wr_arr_smooth[:, 0]))
wr_span_y = float(np.ptp(wr_arr_smooth[:, 1]))
wr_span_z = float(np.ptp(wr_arr_smooth[:, 2]))
wr_span_3d = float(np.max(np.linalg.norm(wr_arr_smooth - np.mean(wr_arr_smooth, axis=0), axis=1)) * 2)

el_span_x = float(np.ptp(el_arr_smooth[:, 0]))
el_span_y = float(np.ptp(el_arr_smooth[:, 1]))
el_span_z = float(np.ptp(el_arr_smooth[:, 2]))
el_span_3d = float(np.max(np.linalg.norm(el_arr_smooth - np.mean(el_arr_smooth, axis=0), axis=1)) * 2)

sh_span_x = float(np.ptp(sh_arr_smooth[:, 0]))
sh_span_y = float(np.ptp(sh_arr_smooth[:, 1]))
sh_span_z = float(np.ptp(sh_arr_smooth[:, 2]))
sh_span_3d = float(np.max(np.linalg.norm(sh_arr_smooth - np.mean(sh_arr_smooth, axis=0), axis=1)) * 2)

# Compensation ratios
sci_y = float(sh_span_y / wr_span_y) if wr_span_y > 0 else 0
sci_3d = float(sh_span_3d / wr_span_3d) if wr_span_3d > 0 else 0

eci_y = float(el_span_y / wr_span_y) if wr_span_y > 0 else 0
eci_3d = float(el_span_3d / wr_span_3d) if wr_span_3d > 0 else 0

# Kinematic Cross-Correlations
r_el_wr = float(np.corrcoef(el_arr_smooth[:, 1], wr_arr_smooth[:, 1])[0, 1])
r_sh_wr = float(np.corrcoef(sh_arr_smooth[:, 1], wr_arr_smooth[:, 1])[0, 1])
r_sh_el = float(np.corrcoef(sh_arr_smooth[:, 1], el_arr_smooth[:, 1])[0, 1])

# Waypoints
waypoints = []
waypoints.append({
    'type': 'start',
    'time': 0.0,
    'frame': 0,
    'label': '起始动作 (t=0.0s)',
    'sh_pos': [round(float(v), 2) for v in sh_arr_smooth[0]],
    'el_pos': [round(float(v), 2) for v in el_arr_smooth[0]],
    'wr_pos': [round(float(v), 2) for v in wr_arr_smooth[0]],
    'elbow_angle': round(float(elbow_angle_deg[0]), 1)
})

cycle_num = 1
for p_idx in sorted(list(peaks) + list(valleys)):
    is_p = p_idx in peaks
    t_val = round(p_idx / fps, 3)
    lb = f"周期{cycle_num} " + ("波峰/旋后最大" if is_p else "波谷/旋前最大")
    if is_p:
        cycle_num += 1
    waypoints.append({
        'type': 'peak' if is_p else 'valley',
        'time': t_val,
        'frame': int(p_idx),
        'label': f"{lb} (t={t_val}s)",
        'sh_pos': [round(float(v), 2) for v in sh_arr_smooth[p_idx]],
        'el_pos': [round(float(v), 2) for v in el_arr_smooth[p_idx]],
        'wr_pos': [round(float(v), 2) for v in wr_arr_smooth[p_idx]],
        'elbow_angle': round(float(elbow_angle_deg[p_idx]), 1)
    })

# Compile per-frame data
processed_frames = []
for i in range(len(raw_data)):
    # Phase determination
    t = raw_data[i]['time']
    # Find closest previous and next waypoint
    phase = "旋前/旋后代偿运动过渡"
    if i in peaks:
        phase = "旋后代偿极大值 (Max Supination & Elevation)"
    elif i in valleys:
        phase = "旋前代偿极大值 (Max Pronation & Flaring)"
    elif wr_vel[i, 1] > 20:
        phase = "旋后抬升相 (Upward Supination Phase)"
    elif wr_vel[i, 1] < -20:
        phase = "旋前下落相 (Downward Pronation Phase)"
        
    processed_frames.append({
        'frame': i,
        'time': round(t, 3),
        'sh_2d': [round(v, 1) for v in raw_data[i]['sh_2d']],
        'el_2d': [round(v, 1) for v in raw_data[i]['el_2d']],
        'wr_2d': [round(v, 1) for v in raw_data[i]['wr_2d']],
        'sh_3d': [round(float(v), 2) for v in sh_arr_smooth[i]],
        'el_3d': [round(float(v), 2) for v in el_arr_smooth[i]],
        'wr_3d': [round(float(v), 2) for v in wr_arr_smooth[i]],
        'l_sh_3d': [round(float(v), 2) for v in l_sh_arr_smooth[i]],
        'sh_vel': [round(float(v), 1) for v in sh_vel[i]],
        'el_vel': [round(float(v), 1) for v in el_vel[i]],
        'wr_vel': [round(float(v), 1) for v in wr_vel[i]],
        'sh_speed': round(float(sh_speed[i]), 1),
        'el_speed': round(float(el_speed[i]), 1),
        'wr_speed': round(float(wr_speed[i]), 1),
        'sh_accel': round(float(sh_acc_mag[i]), 1),
        'el_accel': round(float(el_acc_mag[i]), 1),
        'wr_accel': round(float(wr_acc_mag[i]), 1),
        'elbow_angle': round(float(elbow_angle_deg[i]), 1),
        'shoulder_elev': round(float(shoulder_elev_deg[i]), 1),
        'pronation_deg': round(float(raw_data[i]['pronation_deg']), 1),
        'normal_vec': [round(float(v), 3) for v in raw_data[i]['normal_vec']],
        'hand_lms_3d': [[round(float(c), 2) for c in pt] for pt in raw_data[i]['hand_lms_3d']],
        'phase': phase
    })

# Build complete output object
output_data = {
    'metadata': {
        'video_name': '实验2-腕肘肩代偿判断-视频3-腕肘肩.mp4',
        'fps': fps,
        'total_frames': total_frames,
        'duration_s': round(total_frames / fps, 2),
        'resolution': {'width': w, 'height': h}
    },
    'model_summary': {
        'task_name': '腕-肘-肩三关节运动学与协同代偿分析 (Multi-Joint Compensatory Kinematics)',
        'fundamental_frequency_hz': round(fundamental_freq, 2),
        'fundamental_period_s': round(fundamental_period, 2),
        'fourier_formula': "P(t) = c_0 + c_1*t + sum_{k=1}^3 [ A_k*cos(2*pi*k*f_0*t) + B_k*sin(2*pi*k*f_0*t) ]",
        'fourier_params_wr_y': fourier_params,
        'spans_mm': {
            'wrist': {'x': round(wr_span_x, 1), 'y': round(wr_span_y, 1), 'z': round(wr_span_z, 1), 'span_3d': round(wr_span_3d, 1)},
            'elbow': {'x': round(el_span_x, 1), 'y': round(el_span_y, 1), 'z': round(el_span_z, 1), 'span_3d': round(el_span_3d, 1)},
            'shoulder': {'x': round(sh_span_x, 1), 'y': round(sh_span_y, 1), 'z': round(sh_span_z, 1), 'span_3d': round(sh_span_3d, 1)}
        },
        'compensation_indices': {
            'shoulder_compensation_ratio_y': round(sci_y, 3),
            'shoulder_compensation_ratio_3d': round(sci_3d, 3),
            'elbow_compensation_ratio_y': round(eci_y, 3),
            'elbow_compensation_ratio_3d': round(eci_3d, 3),
            'elbow_angle_range_deg': [round(float(np.min(elbow_angle_deg)), 1), round(float(np.max(elbow_angle_deg)), 1)],
            'elbow_angle_variation_deg': round(float(np.ptp(elbow_angle_deg)), 1),
            'shoulder_elevation_range_deg': [round(float(np.min(shoulder_elev_deg)), 1), round(float(np.max(shoulder_elev_deg)), 1)],
            'shoulder_elevation_variation_deg': round(float(np.ptp(shoulder_elev_deg)), 1),
            'coupling_r_elbow_wrist_y': round(r_el_wr, 3),
            'coupling_r_shoulder_wrist_y': round(r_sh_wr, 3),
            'coupling_r_shoulder_elbow_y': round(r_sh_el, 3),
            'clinical_severity': '显著多关节协同代偿 (Severe Multi-Joint Compensatory Synergy)'
        },
        'speed_stats_mm_s': {
            'wrist_max_speed': round(float(np.max(wr_speed)), 1),
            'wrist_mean_speed': round(float(np.mean(wr_speed)), 1),
            'elbow_max_speed': round(float(np.max(el_speed)), 1),
            'elbow_mean_speed': round(float(np.mean(el_speed)), 1),
            'shoulder_max_speed': round(float(np.max(sh_speed)), 1),
            'shoulder_mean_speed': round(float(np.mean(sh_speed)), 1)
        }
    },
    'waypoints': waypoints,
    'frames': processed_frames
}

# Save to JSON
output_dir = os.path.join(base_dir, "04_腕肘肩协同代偿")
os.makedirs(output_dir, exist_ok=True)
json_path = os.path.join(output_dir, 'arm_compensation_data.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

# Save to JS
js_path = os.path.join(output_dir, 'arm_compensation_data.js')
with open(js_path, 'w', encoding='utf-8') as f:
    f.write('window.ARM_COMPENSATION_DATA = ')
    json.dump(output_data, f, ensure_ascii=False)
    f.write(';')

print(f"Saved {json_path} and {js_path} successfully!")
