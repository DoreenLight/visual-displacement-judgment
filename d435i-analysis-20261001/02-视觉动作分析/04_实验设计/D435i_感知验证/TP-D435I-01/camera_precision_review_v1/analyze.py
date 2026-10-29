"""Read-only historical camera evidence review; never opens a camera."""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
CAMERA = ROOT.parent / '01-摄像头'
TRAJECTORY = BASE.parent / 'trajectory_validation_v1'


def chord_angle(d_mm, length_mm):
    if not math.isfinite(d_mm) or not math.isfinite(length_mm) or length_mm <= 0 or not 0 <= d_mm <= 2*length_mm:
        raise ValueError('invalid chord/radius')
    return math.degrees(2*math.asin(d_mm/(2*length_mm)))


def main():
    paths = sorted(p for p in CAMERA.rglob('*') if p.is_file() and '.git' not in p.parts)
    before = {str(p.relative_to(CAMERA)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    data = json.loads((CAMERA/'07_本机实机验证/blind_calib.json').read_text(encoding='utf-8'))
    x = np.array([p['physical'] for p in data['points']], float)
    y = np.array([p['z'] for p in data['points']], float)
    a,b = np.linalg.lstsq(np.column_stack((x,np.ones(len(x)))), y, rcond=None)[0]
    residuals = y-(a*x+b)
    fit = {'a':float(a),'b_mm':float(b),'in_sample_rmse_mm':float(np.sqrt(np.mean(residuals**2))),
           'r2':float(1-np.sum(residuals**2)/np.sum((y-y.mean())**2)),
           'residuals_mm':residuals.tolist(),'physical_distance_span_mm':[float(x.min()),float(x.max())],
           'read_distance_span_mm':[float(y.min()),float(y.max())],
           'slope_only_2mm_projection_mm':float(abs(a-1)*2),
           'step_nominal_mm':np.diff(x).tolist(),'step_read_mm':np.diff(y).tolist(),
           'step_difference_mm':(np.diff(y)-np.diff(x)).tolist(),
           'stored_mdr_range_mm':[min(p['mdr'] for p in data['points']),max(p['mdr'] for p in data['points'])],
           'reference':'operator_steel_ruler; uncertainty not calibrated; historical report estimates +/-0.5mm',
           'heldout_validation':False,'raw_temporal_sequences_available':False}
    assert np.isclose(a,data['fit']['a']) and np.isclose(fit['in_sample_rmse_mm'],data['fit']['rmse'])
    (BASE/'historical_fit_recomputed.json').write_text(json.dumps(fit,ensure_ascii=False,indent=2),encoding='utf-8')
    with (BASE/'historical_steps.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['from_point','to_point','nominal_step_mm','observed_step_mm','difference_mm'])
        for i,(dx,dy) in enumerate(zip(np.diff(x),np.diff(y))):writer.writerow([i+1,i+2,float(dx),float(dy),float(dy-dx)])
    subject=json.loads((BASE/'subject_lengths.json').read_text(encoding='utf-8'))
    lengths=sorted(set([50,60,80,100,240,250,300]+[subject[k] for k in ('upper_arm_mm','forearm_mm','wrist_to_middle_mcp_mm')]))
    displacement=[.0146,.045,.3066,.710,1,2,3,4.066481862029925,4.919157714865316,5,6.66843264628094,8.002842114149509,10]
    with (BASE/'angle_sensitivity_examples.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['assumed_length_mm','displacement_mm','equivalent_chord_angle_deg','status'])
        for length in lengths:
            for d in displacement:writer.writerow([length,d,chord_angle(d,length),'conditional_geometry_not_validated_angle_accuracy_or_counting_threshold'])
    with (BASE/'personal_angle_conversion.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['segment','operator_length_mm','displacement_mm','equivalent_chord_angle_deg','interpretation'])
        for segment,key in [('upper_arm','upper_arm_mm'),('forearm','forearm_mm'),('hand_marker','wrist_to_middle_mcp_mm')]:
            for d in [.3066,1,2,3,5,10]:writer.writerow([segment,subject[key],d,chord_angle(d,subject[key]),'fixed_pivot_planar_rotation_assumption_only'])
    lu,lf,lh=[subject[k] for k in ('upper_arm_mm','forearm_mm','wrist_to_middle_mcp_mm')]
    propagated=[]
    for error in [2,5,10]:
        upper=math.degrees(math.asin(2*error/lu))
        fore=math.degrees(math.asin(2*error/lf))
        hand=math.degrees(math.asin(2*error/lh))
        propagated.append({'assumed_per_point_error_bound_mm':error,'upper_arm_direction_bound_deg':upper,'elbow_included_angle_bound_deg':upper+fore,'hand_forearm_included_angle_bound_deg':hand+fore,'status':'hypothetical_error_bound_not_measured_FE_RU_accuracy'})
    (BASE/'conditional_error_propagation.json').write_text(json.dumps(propagated,indent=2),encoding='utf-8')
    counting=[]
    for threshold in [5,10,15]:
        counting.append({'single_side_angle_deg':threshold,'peak_to_peak_angle_deg':2*threshold,'neutral_to_extreme_chord_mm_at_user_hand_length':2*lh*math.sin(math.radians(threshold)/2),'opposite_extremes_chord_mm_at_user_hand_length':2*lh*math.sin(math.radians(threshold)),'status':'software_protocol_geometry_not_empirical_detection_threshold'})
    (BASE/'counting_geometry.json').write_text(json.dumps(counting,indent=2),encoding='utf-8')
    packets=[json.loads(line) for line in (ROOT/'07_资料库/D435i_感知验证/trajectory_20261001_223257_586667/telemetry.jsonl').read_text(encoding='utf-8').splitlines()]
    model_lengths=[]
    for start,end in [('r_shoulder','r_elbow'),('r_elbow','r_wrist'),('hand_wrist','hand_middle_mcp')]:
        values=[np.linalg.norm(np.array(p['joints_3d_m'][start])-p['joints_3d_m'][end])*1000 for p in packets if start in p['joints_3d_m'] and end in p['joints_3d_m']]
        model_lengths.append({'start':start,'end':end,'model_median_length_mm':float(np.median(values)),'status':'model_estimate_not_independent_subject_anthropometry'})
    (BASE/'model_lengths_for_context.json').write_text(json.dumps(model_lengths,indent=2),encoding='utf-8')
    pair_rows=[]
    for n in range(1,5):
        p=json.loads((TRAJECTORY/f'ruler_pair_{n:02d}_result.json').read_text(encoding='utf-8'))
        for metric in p['results']:
            pair_rows.append({'pair':n,'condition':'original_endpoints' if n<=3 else 'changed_endpoints_still_50mm_operator_confirmed',
                             'joint_id':metric['joint_id'],'distance_mm':metric['median_endpoint_distance_mm'],
                             'nominal_mm':50.,'difference_mm':metric['median_endpoint_distance_mm']-50.,
                             'reference_uncertainty_mm':None,'a_run_id':p['a_run_id'],'b_run_id':p['b_run_id']})
    (BASE/'endpoint_pairs_reviewed.json').write_text(json.dumps(pair_rows,ensure_ascii=False,indent=2),encoding='utf-8')
    after={str(p.relative_to(CAMERA)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert before==after, 'historical source mutated'
    (BASE/'source_manifest.json').write_text(json.dumps({'source_root':str(CAMERA),'file_count':len(paths),'files':before},ensure_ascii=False,indent=2),encoding='utf-8')
    checks={'historical_source_files_unchanged':len(paths),'stored_fit_recomputed':True,'nominal_2mm_step_difference_mm':float(np.diff(y)[2]-np.diff(x)[2]),
            'zero_chord_angle':chord_angle(0,80),'diameter_chord_angle':chord_angle(160,80),
            'small_angle_check_deg':chord_angle(2,80),'human_accuracy':'BLOCKED_NO_INDEPENDENT_DYNAMIC_REFERENCE',
            'personal_lengths':'OPERATOR_REPORTED_300_290_112_MM_NOT_INDEPENDENTLY_VALIDATED','measurement_session':'ENDED_BY_USER','pair_5_acquired':False}
    assert checks['zero_chord_angle']==0 and np.isclose(checks['diameter_chord_angle'],180)
    assert np.isclose(2*80*math.sin(math.radians(checks['small_angle_check_deg'])/2),2)
    (BASE/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps({'source_files':len(paths),'fit':fit,'checks':checks},ensure_ascii=True))


if __name__=='__main__':main()
