"""Read-only telemetry sampling. Does not start training or open a camera."""
import argparse
import csv
import hashlib
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np

JOINTS = ('r_shoulder', 'r_elbow', 'r_wrist', 'hand_wrist', 'hand_middle_mcp')
ROOT = Path(__file__).resolve().parents[4]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--phase', choices=('instructed_static', 'free_motion'), default='instructed_static')
    args = parser.parse_args()
    if not 1 <= args.seconds <= 120:
        parser.error('duration must be 1..120 seconds')
    run = 'trajectory_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    raw = ROOT / '07_资料库' / 'D435i_感知验证' / run
    out = Path(__file__).parent / run
    raw.mkdir(parents=True, exist_ok=False)
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    packets, seen, failures, duplicates = [], set(), [], 0
    camera = None
    with (raw / 'telemetry.jsonl').open('x', encoding='utf-8') as stream:
        while time.monotonic() - started < args.seconds:
            try:
                with urllib.request.urlopen('http://127.0.0.1:8090/telemetry', timeout=2) as response:
                    source = json.load(response)
                key = (source.get('frame_id'), source.get('received_mono_s'))
                if key in seen:
                    duplicates += 1
                elif key[0] is not None and key[1] is not None:
                    seen.add(key)
                    camera = source.get('camera', camera)
                    p = {k: source.get(k) for k in ('frame_id', 'received_mono_s', 'output_mono_s', 'body_valid', 'joints_3d_m', 'measurements', 'live_version')}
                    p['collector_mono_s'] = time.monotonic()
                    packets.append(p)
                    stream.write(json.dumps(p, ensure_ascii=False, allow_nan=False) + '\n')
            except (OSError, ValueError) as exc:
                failures.append({'collector_mono_s': time.monotonic(), 'error': str(exc)})
            time.sleep(.035)
    elapsed = time.monotonic() - started
    metrics = []
    with (raw / 'keypoints.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['frame_id', 'host_receive_mono_s', 'output_mono_s', 'collector_mono_s', 'joint_id', 'measurement_valid', 'x_m', 'y_m', 'z_m', 'invalid_reason', 'body_valid'])
        for joint in JOINTS:
            valid_points = []
            gap_start, gap_max = None, 0.
            for p in packets:
                m = (p.get('measurements') or {}).get(joint, {})
                xyz = (p.get('joints_3d_m') or {}).get(joint)
                valid = bool(m.get('measurement_valid')) and xyz is not None and len(xyz) == 3 and bool(np.isfinite(xyz).all())
                t = p['received_mono_s']
                if valid:
                    valid_points.append(xyz)
                    if gap_start is not None:
                        gap_max = max(gap_max, t - gap_start)
                        gap_start = None
                elif gap_start is None:
                    gap_start = t
                writer.writerow([p['frame_id'], t, p.get('output_mono_s'), p['collector_mono_s'], joint, valid, *(xyz if valid else ['', '', '']), m.get('invalid_reason') or ('' if valid else 'point_missing'), p.get('body_valid')])
            if gap_start is not None:
                gap_max = max(gap_max, packets[-1]['received_mono_s'] - gap_start)
            a = np.array(valid_points, dtype=float)
            radial = np.linalg.norm(a - np.median(a, axis=0), axis=1) * 1000 if len(a) else []
            metrics.append({'joint_id': joint, 'valid_count': len(a), 'polled_unique_outputs': len(packets), 'valid_rate_among_polled_outputs': len(a)/len(packets) if packets else None, 'axis_sd_mm': (np.std(a, axis=0, ddof=1)*1000).tolist() if len(a)>1 else None, 'median_distance_p95_mm': float(np.percentile(radial, 95)) if len(a) else None, 'median_distance_max_mm': float(max(radial)) if len(a) else None, 'observed_invalid_gap_max_s': gap_max, 'accuracy_status': 'BLOCKED_NO_INDEPENDENT_REFERENCE'})
    source_times = [p['received_mono_s'] for p in packets]
    step_max = max(np.diff(source_times), default=0.)
    frame_span = packets[-1]['frame_id'] - packets[0]['frame_id'] + 1 if packets else None
    metadata = {'run_id': run, 'phase_instruction': args.phase, 'static_compliance': 'NOT_INDEPENDENTLY_CONFIRMED', 'reference': None, 'camera': camera, 'seconds': elapsed, 'requested_seconds': args.seconds, 'collector_start_mono_s': started, 'capture_type': 'sparse_existing_telemetry_no_rgbd', 'source': 'http://127.0.0.1:8090/telemetry', 'coordinate_frame': 'camera_color_optical', 'units': 'm', 'collector_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'http_failures': failures}
    (raw / 'run.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    result = {'run_id': run, 'phase': args.phase, 'static_compliance': metadata['static_compliance'], 'seconds': elapsed, 'unique_outputs': len(packets), 'duplicate_polls_ignored': duplicates, 'http_failure_count': len(failures), 'captured_frame_id_span': frame_span, 'max_polled_source_time_step_s': float(step_max), 'denominator_note': 'Unique polled analyzed outputs only; not all camera frames. Missing source frames are unobserved, not classified valid.', 'joints': metrics, 'accuracy_status': 'BLOCKED_NO_INDEPENDENT_REFERENCE', 'raw_directory': str(raw)}
    (out / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    for folder in (raw, out):
        lines = [hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + p.name for p in sorted(folder.iterdir()) if p.is_file() and p.name != 'files.sha256']
        (folder / 'files.sha256').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
