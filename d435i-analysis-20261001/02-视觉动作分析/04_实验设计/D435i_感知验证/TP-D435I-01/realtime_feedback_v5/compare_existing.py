from pathlib import Path
import sys,json,csv,hashlib
import numpy as np,pyrealsense2 as rs
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];ASSETS=HERE.parent/'attempt_01'
sys.path.insert(0,str(ASSETS));sys.path.insert(0,str(HERE))
from training_protocol_v3 import ProtocolEngine as Old,HAND
from training_protocol_v5 import ProtocolEngine as New
from wrist_depth import hand_depth
runs=['eight_FE_20261001_132450_246061a9','eight_RU_20261001_133347_d19a29ac']
# Resolve FE ID from the retained list rather than guessing a run identifier.
retained=json.loads((ASSETS/'eight_task_sessions.json').read_text(encoding='utf-8'));runs[0]=next(r['run_id'] for r in retained if r['task_id']==3)
result=[];export=[]
for run in runs:
 raw=ROOT/'07_资料库/D435i_感知验证'/run;meta=json.loads((raw/'run.json').read_text(encoding='utf-8'))
 source=HERE.parent/'offline_reanalysis_v1/dense_20261001_v1/runs'/run/'measurements.jsonl'
 frames={x['frame_id']:x for x in map(json.loads,(raw/'frames.jsonl').read_text(encoding='utf-8').splitlines())}
 records=[json.loads(x) for x in source.read_text(encoding='utf-8').splitlines()][:300]
 engines=[Old(),New()]
 for e in engines:
  e.select(meta['task_id']);e.baseline={k:np.array(v) for k,v in meta.get('baseline_m',{}).items()} or None;e.hand_basis={k:np.array(v) if isinstance(v,list) else v for k,v in (meta.get('hand_basis') or {}).items()} or None;e.training=True;e.calibration_id=meta.get('calibration_id')
 cfg=meta['camera']['intrinsics'];intr=rs.intrinsics()
 for k in ('width','height','fx','fy','ppx','ppy'):setattr(intr,k,cfg[k])
 intr.coeffs=cfg['coeffs'];intr.model=getattr(rs.distortion,cfg['model'].split('.')[-1]);scale=meta['camera']['depth_scale_m']
 counts=[0,0];jump=[[],[]];prev=[None,None]
 for rec in records:
  fr=frames[rec['frame_id']];now=fr['host_mono_s'];depth=np.load(raw/fr['depth_file'],allow_pickle=False)
  points={k:np.array(x['point_m']) for k,x in rec['measurements'].items() if x.get('measurement_valid') and x.get('point_m') is not None};newpoints=dict(points)
  for name in HAND:
   newpoints.pop(name,None);m=rec['measurements'][name];u,v=m.get('u_px'),m.get('v_px')
   if u is None or v is None or not (0<=u<depth.shape[1] and 0<=v<depth.shape[0]):continue
   z,reason,details=hand_depth(depth,int(u),int(v),scale)
   if z is not None:newpoints[name]=np.array(rs.rs2_deproject_pixel_to_point(intr,[float(int(u)),float(int(v))],z))
  outputs=[engines[0].update(points,now),engines[1].update(newpoints,now)]
  for i,o in enumerate(outputs):
   counts[i]+=bool(o['estimate_valid'])
   v=o['theta_fe_est_deg'] if meta['task_id']==3 else o['theta_ru_est_deg']
   if v is not None and prev[i] is not None:jump[i].append(abs(v-prev[i]))
   prev[i]=v
  export.append({'run_id':run,'frame_id':rec['frame_id'],'old_valid':outputs[0]['estimate_valid'],'new_valid':outputs[1]['estimate_valid'],'old_fe':outputs[0]['theta_fe_est_deg'],'new_fe':outputs[1]['theta_fe_est_deg'],'old_ru':outputs[0]['theta_ru_est_deg'],'new_ru':outputs[1]['theta_ru_est_deg'],'old_reason':outputs[0]['estimate_invalid_reason'],'new_reason':outputs[1]['estimate_invalid_reason']})
 result.append({'run_id':run,'frames':len(records),'old_valid':counts[0],'new_valid':counts[1],'old_reps':engines[0].reps,'new_reps':engines[1].reps,'old_step_p95_deg':float(np.percentile(jump[0],95)) if jump[0] else None,'new_step_p95_deg':float(np.percentile(jump[1],95)) if jump[1] else None,'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
with (HERE/'existing_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(export[0]));w.writeheader();w.writerows(export)
report={'results':result,'scope':'first 300 captured frames each retained FE1/RU2; previous pose/hand pixel detections reused; hand depth resampled; stored original neutral references reused to isolate changes; rounded pixels may differ at a pixel boundary; no independent angular reference','accuracy_verified':False,'neutral_recalibration_evaluated':False,'counter_new_thresholds':'original FE15 RU10','camera_opened':False}
(HERE/'existing_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=True))
