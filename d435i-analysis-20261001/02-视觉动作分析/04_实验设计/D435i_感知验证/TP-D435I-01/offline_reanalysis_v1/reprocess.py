import sys,json,time,argparse,hashlib
from pathlib import Path
from collections import Counter
import numpy as np,cv2,mediapipe as mp,pyrealsense2 as rs
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
ROOT=Path(__file__).resolve().parents[4]
SRC=ROOT/'04_实验设计/D435i_感知验证/TP-D435I-01/attempt_01'
sys.path.insert(0,str(SRC))
from live_feedback_v3 import measure_all
from training_protocol_v3 import ProtocolEngine,BODY,HAND

def write(p,d):
 p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(x) for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def process(run,out,limit=0):
 raw=ROOT/'07_资料库/D435i_感知验证'/run;m=load(raw/'run.json');fs=lines(raw/'frames.jsonl');events=lines(raw/'events.jsonl')
 if limit:fs=fs[:limit]
 d=out/'runs'/run;d.mkdir(parents=True,exist_ok=False)
 i=rs.intrinsics();cfg=m['camera']['intrinsics']
 for k in ['width','height','fx','fy','ppx','ppy']:setattr(i,k,cfg[k])
 i.coeffs=cfg['coeffs'];i.model=getattr(rs.distortion,cfg['model'].split('.')[-1]);scale=m['camera']['depth_scale_m']
 e=ProtocolEngine();e.select(m['task_id'],m['participant_id']);e.baseline={k:np.array(v) for k,v in (m.get('baseline_m') or {}).items()} or None
 e.hand_basis={k:np.array(v) if isinstance(v,list) else v for k,v in (m.get('hand_basis') or {}).items()} or None
 e.calibration_id=m.get('calibration_id');e.training=True;e.run_id=run;e.last_clock=fs[0]['host_mono_s']
 po=vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_buffer=(SRC/'pose_landmarker_lite.task').read_bytes()),running_mode=vision.RunningMode.VIDEO,min_pose_detection_confidence=.5,min_pose_presence_confidence=.5,min_tracking_confidence=.5)
 ho=vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_buffer=(SRC/'hand_landmarker.task').read_bytes()),running_mode=vision.RunningMode.VIDEO,num_hands=2,min_hand_detection_confidence=.5,min_hand_presence_confidence=.5,min_tracking_confidence=.5)
 stats={'run_id':run,'task_id':m['task_id'],'frames':len(fs),'body_valid':0,'angle_valid':0,'states':Counter(),'reasons':Counter()};ei=0;ts=-1;start=time.perf_counter()
 with vision.PoseLandmarker.create_from_options(po) as pose,vision.HandLandmarker.create_from_options(ho) as hand,(d/'measurements.jsonl').open('w',encoding='utf-8') as f:
  for idx,fr in enumerate(fs):
   now=fr['host_mono_s'];t=now-fs[0]['host_mono_s']
   while ei<len(events) and events[ei]['host_mono_s']<=now:
    ev=events[ei]['event'];ei+=1
    if ev=='pause':e.pause(now)
    elif ev=='resume':e.paused=False;e.last_clock=now
    elif ev=='switch_direction' and e.task_id==5 and e.direction=='clockwise':e.switch_direction()
   begin=time.perf_counter();color=cv2.imdecode(np.fromfile(raw/fr['rgb_file'],np.uint8),cv2.IMREAD_COLOR);depth=np.load(raw/fr['depth_file'],allow_pickle=False)
   if color is None or color.shape[:2]!=depth.shape:raise ValueError('invalid RGB-depth pair')
   ts=max(ts+1,int(round(t*1000)));image=mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(color,cv2.COLOR_BGR2RGB))
   points,pixels,meas=measure_all(pose.detect_for_video(image,ts),hand.detect_for_video(image,ts),{'depth':depth,'scale':scale,'intr':i});pred=e.update(points,now)
   for name,z in meas.items():
    u,v=z.get('u_px'),z.get('v_px');z.update(patch_coverage=None,patch_spread_m=None,patch_median_m=None)
    if u is not None and v is not None and 0<=u<depth.shape[1] and 0<=v<depth.shape[0]:
     u,v=int(u),int(v);patch=depth[max(0,v-3):v+4,max(0,u-3):u+4];vals=patch[patch>0].astype(float)*scale;z['patch_coverage']=len(vals)/patch.size
     if len(vals):z['patch_spread_m']=float(np.percentile(vals,90)-np.percentile(vals,10));z['patch_median_m']=float(np.median(vals))
   keep=['body_valid','estimate_valid','estimate_invalid_reason','theta_fe_est_deg','theta_ru_est_deg','compensation_valid','compensation_state','shoulder_alert','elbow_alert','shoulder_abs_mm','elbow_deg','upper_arm_swing_deg','reps','paused','direction']
   pred={k:pred.get(k) for k in keep};stats['body_valid']+=bool(pred['body_valid']);stats['angle_valid']+=bool(pred['estimate_valid']);stats['states'][pred['compensation_state']]+=1
   if not pred['estimate_valid']:stats['reasons'][pred['estimate_invalid_reason']]+=1
   rec={'index':idx,'frame_id':fr['frame_id'],'t_s':t,'rgb_file':fr['rgb_file'],'depth_file':fr['depth_file'],'measurements':meas,'prediction':pred,'processing_ms':(time.perf_counter()-begin)*1000}
   f.write(json.dumps(rec,ensure_ascii=False,allow_nan=False)+'\n')
   if idx%250==0 or idx==len(fs)-1:
    progress={'run':run,'frame':idx+1,'of':len(fs),'fps':round((idx+1)/(time.perf_counter()-start),2)};write(out/'progress.json',progress);print(json.dumps(progress),flush=True)
 stats.update(final_reps=pred['reps'],seconds=time.perf_counter()-start);write(d/'summary.json',stats)
 write(d/'input_manifest.json',{'run_id':run,'metadata':[{'name':x,'sha256':sha(raw/x)} for x in ['run.json','frames.jsonl','analysis.jsonl','events.jsonl','files.sha256']],'image_scope':'all selected frames; paired files read, no thinning; previous raw manifest SHA audit retained'})
 return stats

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--limit',type=int,default=0);ap.add_argument('--runs',nargs='*');args=ap.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
 selected=load(SRC/'eight_task_sessions.json');allowed={x['run_id'] for x in selected};runs=args.runs
 if not runs:
  runs=[next(x['run_id'] for x in selected if x['task_id']==t) for t in range(1,9)];runs+=sorted(allowed-set(runs))
 if not set(runs)<=allowed:raise ValueError('excluded or unknown run')
 files=[SRC/x for x in ['pose_landmarker_lite.task','hand_landmarker.task','training_protocol_v3.py','live_feedback_v3.py','live_feedback_v2.py','feedback_engine_v2.py','eight_task_sessions.json','excluded_trials.json']]+[Path(__file__)]
 receipt={'command':sys.argv,'python':sys.version,'sources':[{'path':str(x),'sha256':sha(x)} for x in files],'runs':runs,'limit':args.limit,'offline_method':'fresh VIDEO tracker per trial; original gates and stored baseline; captured time; no live deadline or interpolation','camera_opened':False,'exit_code':None};write(out/'execution.json',receipt)
 sums=[]
 for run in runs:sums.append(process(run,out,args.limit));write(out/'trial_summaries.json',sums)
 receipt['exit_code']=0;write(out/'execution.json',receipt);write(out/'progress.json',{'stage':'complete','trials':len(sums),'frames':sum(x['frames'] for x in sums)});print('REINFERENCE_COMPLETE',flush=True)
if __name__=='__main__':main()
