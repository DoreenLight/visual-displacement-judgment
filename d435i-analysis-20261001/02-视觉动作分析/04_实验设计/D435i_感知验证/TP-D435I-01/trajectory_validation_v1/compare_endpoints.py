"""Compare median endpoint displacement; ruler nominal distance is not 3D truth."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def compare(base, a_run, b_run, nominal_mm):
    data = []
    for run in (a_run, b_run):
        summary = json.loads((base/run/'summary.json').read_text(encoding='utf-8'))
        with (Path(summary['raw_directory'])/'keypoints.csv').open(encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        keys = [(r['frame_id'], r['host_receive_mono_s'], r['joint_id']) for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError('duplicate coordinate keys')
        data.append(rows)
    results = []
    for joint in ('r_wrist', 'hand_wrist'):
        medians = []
        counts = []
        for rows in data:
            points = np.array([[float(r[k]) for k in ('x_m','y_m','z_m')] for r in rows if r['joint_id']==joint and r['measurement_valid']=='True'])
            if len(points)<2 or not np.isfinite(points).all():
                raise ValueError('insufficient finite endpoint observations')
            counts.append(len(points))
            medians.append(np.median(points, axis=0))
        delta = (medians[1]-medians[0])*1000
        distance = float(np.linalg.norm(delta))
        results.append({'joint_id':joint,'valid_counts_a_b':counts,'median_positions_m':[x.tolist() for x in medians],'delta_xyz_mm':delta.tolist(),'median_endpoint_distance_mm':distance,'nominal_ruler_distance_mm':nominal_mm,'difference_from_nominal_mm':distance-nominal_mm})
    return {'a_run_id':a_run,'b_run_id':b_run,'results':results,'comparison_scope':'single_pair_median_endpoint_distance_only','reference_source':'operator_confirmed_ruler_distance','reference_uncertainty_mm':None,'white_marker_tracking':False,'anatomical_point_mapping':None,'continuous_dynamic_accuracy':'BLOCKED','absolute_3d_accuracy':'BLOCKED'}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('a_run');p.add_argument('b_run');p.add_argument('--nominal-mm',type=float,required=True)
    args=p.parse_args()
    if not np.isfinite(args.nominal_mm) or args.nominal_mm<=0:
        p.error('nominal distance must be finite and positive')
    print(json.dumps(compare(Path(__file__).parent,args.a_run,args.b_run,args.nominal_mm),ensure_ascii=True,indent=2))
