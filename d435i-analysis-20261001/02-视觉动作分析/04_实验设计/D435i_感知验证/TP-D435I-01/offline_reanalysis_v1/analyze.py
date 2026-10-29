import json,csv,sys,math,hashlib
from pathlib import Path
from collections import Counter
import numpy as np,cv2
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];BASE=Path(__file__).parent;SRC=ROOT/'04_实验设计/D435i_感知验证/TP-D435I-01/attempt_01'
BODY=['r_shoulder','r_elbow','r_wrist'];HAND=['hand_wrist','hand_index_mcp','hand_middle_mcp','hand_pinky_mcp']
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(x) for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csvsave(p,rows):
 if not rows:return
 with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def gap(t,valid):
 if len(t)<2:return 0.
 delta=np.median(np.diff(t));edges=np.r_[t,t[-1]+delta];start=None;best=0.
 for i,v in enumerate(valid):
  if not v and start is None:start=i
  if v and start is not None:best=max(best,edges[i]-edges[start]);start=None
 if start is not None:best=max(best,edges[-1]-edges[start])
 return float(best)
def elbow(s,e,w):
 a=s-e;b=w-e;return 180-np.degrees(np.arccos(np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1)))
def alerts(values,t,valid):
 flags=np.zeros((len(t),2),bool);since=[None,None];on=[False,False]
 for i in range(len(t)):
  if not valid[i]:since=[None,None];on=[False,False];continue
  for j,x in enumerate(values[i]):
   if x>=1:
    if since[j] is None:since[j]=t[i]
    if t[i]-since[j]>=.18:on[j]=True
   elif x<.7:since[j]=None;on[j]=False
  flags[i]=on
 return flags

def analyze(runfolder,out):
 run=runfolder.name;records=lines(runfolder/'measurements.jsonl');raw=ROOT/'07_资料库/D435i_感知验证'/run;meta=load(raw/'run.json');task=meta['task_id'];t=np.array([x['t_s'] for x in records]);n=len(t)
 pred=[x['prediction'] for x in records];body=np.array([x['body_valid'] for x in pred],bool);cand=np.array([x['estimate_valid'] for x in pred],bool);active=np.array([not x['paused'] for x in pred],bool)
 points={name:np.array([x['measurements'][name]['point_m'] or [np.nan]*3 for x in records],float) for name in BODY+HAND};joint=[]
 for name,pts in points.items():
  mm=[x['measurements'][name] for x in records];valid=np.array([x['measurement_valid'] for x in mm],bool);vs=pts[valid];med=np.median(vs,axis=0) if len(vs) else None;uv=np.array([[x['u_px'] if x['u_px'] is not None else np.nan,x['v_px'] if x['v_px'] is not None else np.nan] for x in mm])[valid]
  sd=np.std(vs,axis=0)*1000 if len(vs) else [None]*3
  j=dict(run_id=run,task_id=task,joint=name,frames=n,valid_frames=int(valid.sum()),valid_rate=float(valid.mean()),gap_s=gap(t,valid),x_sd_mm=sd[0],y_sd_mm=sd[1],z_sd_mm=sd[2],position_p95_mm=float(np.percentile(np.linalg.norm(vs-med,axis=1),95)*1000) if len(vs) else None,median_z_m=float(med[2]) if med is not None else None,z_p05_m=float(np.percentile(vs[:,2],5)) if len(vs) else None,z_p95_m=float(np.percentile(vs[:,2],95)) if len(vs) else None,u_sd_px=float(np.std(uv[:,0])) if len(vs) else None,v_sd_px=float(np.std(uv[:,1])) if len(vs) else None,mm_per_pixel=float(med[2]/meta['camera']['intrinsics']['fx']*1000) if med is not None else None,invalid_reasons=json.dumps(dict(Counter(x['invalid_reason'] for x in mm if not x['measurement_valid']))))
  early=pts[(t<2)&valid]
  b=(meta.get('baseline_m') or {}).get(name)
  j['original_to_first2s_offset_mm']=float(np.linalg.norm(np.median(early,axis=0)-b)*1000) if b is not None and len(early) else None
  if task in [1,2] and len(vs):
   z=vs[:,2];zm=float(np.median(z));fx=meta['camera']['intrinsics']['fx'];fy=meta['camera']['intrinsics']['fy']
   j['xy_pixel_only_sd_mm']=float(np.sqrt((np.std(uv[:,0])*zm/fx)**2+(np.std(uv[:,1])*zm/fy)**2)*1000)
   # Best/worst fixed-duration windows are exploratory extrema, never trial pass.
   windows=[]
   for start in np.arange(0,max(0,t[-1]-5)+.1,1):
    win=(t>=start)&(t<start+5);vv=pts[win&valid]
    if win.sum() and len(vv)>=win.sum()*.9:windows.append(float(np.percentile(np.linalg.norm(vv-np.median(vv,axis=0),axis=1),95)*1000))
   j['exploratory_5s_best_p95_mm']=min(windows) if windows else None;j['exploratory_5s_worst_p95_mm']=max(windows) if windows else None
  else:j.update(xy_pixel_only_sd_mm=None,exploratory_5s_best_p95_mm=None,exploratory_5s_worst_p95_mm=None)
  joint.append(j)
 # Recentered first-2s diagnostic, which may contain motion, not neutral calibration.
 valid_initial=(t<2)&body;local={k:np.median(v[valid_initial],axis=0) for k,v in points.items() if len(v[valid_initial]) and np.isfinite(v[valid_initial]).all()}
 signals=np.full((n,2),np.nan);shoulder=np.full(n,np.nan);ed=np.full(n,np.nan);swing=np.full(n,np.nan)
 if all(k in local for k in BODY):
  s0,e0,w0=[local[k] for k in BODY];eb=elbow(s0,e0,w0);upper0=e0-s0
  for i in np.flatnonzero(body):
   s,e,w=[points[k][i] for k in BODY];shoulder[i]=np.linalg.norm(s-s0)*1000;ed[i]=abs(elbow(s,e,w)-eb);upper=e-s;swing[i]=np.degrees(np.arccos(np.clip(np.dot(upper,upper0)/np.linalg.norm(upper)/np.linalg.norm(upper0),-1,1)));signals[i]=[max(shoulder[i]/50,swing[i]/15),ed[i]/15]
 local_flags=alerts(signals,t,body);old_alert=np.array([bool(x['shoulder_alert']) or bool(x['elbow_alert']) for x in pred]);local_alert=np.any(local_flags,axis=1)
 fe=np.array([x['theta_fe_est_deg'] if x['theta_fe_est_deg'] is not None else np.nan for x in pred]);ru=np.array([x['theta_ru_est_deg'] if x['theta_ru_est_deg'] is not None else np.nan for x in pred])
 summary=dict(run_id=run,task_id=task,frames=n,body_valid_frames=int(body.sum()),body_valid_rate=float(body.mean()),body_gap_s=gap(t,body),candidate_angle_frames=int(cand.sum()),candidate_angle_rate=float(cand.mean()),candidate_angle_gap_s=gap(t,cand),final_reps=pred[-1]['reps'],original_alert_share_of_observable=float(old_alert[body].mean()) if body.any() else None,recentered_alert_share_of_observable=float(local_alert[body].mean()) if body.any() and local else None,angle_invalid_reasons=json.dumps(dict(Counter(x['estimate_invalid_reason'] for x in pred if not x['estimate_valid']))),fe_p05_p95_span_deg=float(np.nanpercentile(fe,95)-np.nanpercentile(fe,5)) if np.isfinite(fe).any() else None,ru_p05_p95_span_deg=float(np.nanpercentile(ru,95)-np.nanpercentile(ru,5)) if np.isfinite(ru).any() else None)
 summary['has_stored_hand_basis']=bool(meta.get('hand_basis'));summary['has_stored_body_baseline']=bool(meta.get('baseline_m'))
 allhand=np.array([all(x['measurements'][k]['measurement_valid'] for k in HAND) for x in records],bool)
 summary['all_hand_points_valid_rate']=float(allhand.mean());summary['body_valid_but_hand_missing_rate']=float((body & ~allhand).mean())
 # Largest valid 10s window, followed by visual inspection; not independent validation.
 best=None
 for start in np.arange(2,max(2,t[-1]-10)+.1,1):
  mask=(t>=start)&(t<start+10)&active
  if mask.sum()<30:continue
  score=float(body[mask].mean())+(.6*float(cand[mask].mean()) if task in [3,4,5] else 0)
  if best is None or score>best[0]:best=(score,float(start))
 begin=best[1] if best else 0.;end=min(begin+10,float(t[-1]));summary['selected_clip_start_s']=begin;summary['selected_clip_end_s']=end
 case=out/'cases'/run;case.mkdir(parents=True,exist_ok=True)
 idx=np.flatnonzero((t>=begin)&(t<=end));picks=np.linspace(idx[0],idx[-1],6).astype(int);canvas=np.zeros((3*370,2*600,3),np.uint8)
 for q,i in enumerate(picks):
  image=overlay(raw,records[i]);image=cv2.resize(image,(600,340));canvas[(q//2)*370:(q//2)*370+340,(q%2)*600:(q%2+1)*600]=image
  cv2.putText(canvas,'t={:.2f}s  {}'.format(t[i],pred[i]['compensation_state']),((q%2)*600+8,(q//2)*370+363),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
 cv2.imencode('.jpg',canvas)[1].tofile(case/'contact_sheet.jpg')
 fig,ax=plt.subplots(3,1,figsize=(10,8),sharex=True)
 for k,col in zip(BODY,['C0','C1','C2']):
  v=points[k];reference=np.nanmedian(v,axis=0);dist=np.linalg.norm(v-reference,axis=1)*1000;dist[~body]=np.nan;ax[0].plot(t,dist,label=k,linewidth=.7)
 ax[0].set_ylabel('Position fluctuation (mm)');ax[0].legend(loc='upper right');ax[0].set_title(run+' / raw depth surface points')
 ax[1].plot(t,fe,label='candidate FE');ax[1].plot(t,ru,label='candidate RU');ax[1].set_ylabel('Candidate angle (deg)');ax[1].legend();ax[1].axhline(0,color='gray',lw=.6)
 vals=np.array([[x['shoulder_abs_mm'] if x['shoulder_abs_mm'] is not None else np.nan,(x['upper_arm_swing_deg'] or 0)*50/15 if x['body_valid'] else np.nan,(x['elbow_deg'] or 0)*50/15 if x['body_valid'] else np.nan] for x in pred])
 for k,label in enumerate(['shoulder displacement / 50mm','upper arm swing / 15deg','elbow change / 15deg']):ax[2].plot(t,vals[:,k]/50,label=label,lw=.7)
 ax[2].axhline(1,color='red',ls='--',label='source threshold');ax[2].set_ylabel('Threshold ratio');ax[2].set_xlabel('Capture time from trial start (s)');ax[2].legend(fontsize=7)
 for a in ax:a.axvspan(begin,end,color='green',alpha=.09);a.grid(alpha=.2)
 fig.tight_layout();fig.savefig(case/'signals.png',dpi=130);plt.close(fig)
 csvsave(case/'selected_telemetry.csv',[dict(frame_id=records[i]['frame_id'],t_s=t[i],body_valid=bool(body[i]),candidate_fe_deg=float(fe[i]) if np.isfinite(fe[i]) else None,candidate_ru_deg=float(ru[i]) if np.isfinite(ru[i]) else None,state=pred[i]['compensation_state']) for i in idx])
 save(case/'selection.json',dict(run_id=run,start_s=begin,end_s=end,selection='highest dense-body/candidate availability among fixed 10s windows; exploratory; visual review required',review_status='not_yet_reviewed',baseline_counterfactual_note='first2s recentered baseline may contain motion; diagnostic only'))
 return summary,joint

def overlay(raw,rec):
 image=cv2.imdecode(np.fromfile(raw/rec['rgb_file'],np.uint8),cv2.IMREAD_COLOR)
 colors={'r_shoulder':(255,150,0),'r_elbow':(0,180,255),'r_wrist':(0,255,0)}
 for name,m in rec['measurements'].items():
  if m['u_px'] is None or m['v_px'] is None:continue
  u,v=int(m['u_px']),int(m['v_px']);color=colors.get(name,(255,255,0)) if m['measurement_valid'] else (0,0,255)
  cv2.circle(image,(u,v),5,color,2)
 for x,y in [('r_shoulder','r_elbow'),('r_elbow','r_wrist')]:
  a,b=rec['measurements'][x],rec['measurements'][y]
  if a['measurement_valid'] and b['measurement_valid']:cv2.line(image,(int(a['u_px']),int(a['v_px'])),(int(b['u_px']),int(b['v_px'])),(0,255,255),2)
 pr=rec['prediction'];cv2.rectangle(image,(0,0),(848,49),(0,0,0),-1);cv2.putText(image,'t={:.2f}s body={} FE={} RU={}'.format(rec['t_s'],pr['body_valid'],pr['theta_fe_est_deg'],pr['theta_ru_est_deg']),(6,19),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1);cv2.putText(image,'observed: '+pr['compensation_state']+'  / candidate only',(6,40),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
 return image

def main():
 source=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True);sums=[];joints=[]
 for run in sorted((source/'runs').iterdir()):
  if not (run/'summary.json').exists():continue
  s,j=analyze(run,out);sums.append(s);joints+=j
 csvsave(out/'trial_metrics.csv',sums);csvsave(out/'joint_metrics.csv',joints);save(out/'analysis_summary.json',{'completed_trials':len(sums),'frames':sum(x['frames'] for x in sums),'body_valid_frames':sum(x['body_valid_frames'] for x in sums),'candidate_angle_frames':sum(x['candidate_angle_frames'] for x in sums),'summaries':sums,'analysis_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()});print(json.dumps({'analyzed_trials':len(sums),'frames':sum(x['frames'] for x in sums)}))
if __name__=='__main__':main()
