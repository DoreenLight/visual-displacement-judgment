"""Independent closeout audit. Reads immutable inputs; emits a new version of results."""
import argparse,csv,hashlib,json,math,os,sys,time
from collections import Counter
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'04_实验设计'/'D435i_感知验证'
SOURCE=BASE/'TP-D435I-01'/'attempt_01'
sys.path.insert(0,str(SOURCE))
from trial_selection_v3 import active_history
COLS='run_id,frame_id,joint_id,t_rgb_device_s,t_depth_device_s,t_host_mono_s,clock_domain,u_px,v_px,x_raw_m,y_raw_m,z_raw_m,x_filt_m,y_filt_m,z_filt_m,keypoint_score,visibility,depth_valid,measurement_valid,imputed,valid_for_analysis,invalid_reason,coord_frame'.split(',')
JOINTS=('shoulder','elbow','wrist')
THRESHOLDS={'point_valid_rate_min':.95,'invalid_gap_s_max':.2,'static_axis_sd_mm_max':5.,'static_position_p95_mm_max':10.,
            'reference_3d_rmse_mm_max':10.,'reference_3d_error_p95_mm_max':20.,'angle_rmse_deg_max':5.,'angle_error_p95_deg_max':10.,
            'rom_error_deg_max':5.,'count_error_per_10_max':1,'channel_sensitivity_min':.90,'channel_specificity_min':.90,
            'detection_latency_p95_ms_max':300,'quality_end_to_end_latency_p95_ms_max':200,'independent_auc_min':.85}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def jsonl(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value
def save_json(path,value):Path(path).write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def save_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
        for row in rows:writer.writerow({k:float('nan') if v is None else v for k,v in row.items()})
def flag(series):return series.fillna(False).astype(str).str.lower().isin(('true','1'))
def gap(times,valid,start=None,end=None):
    t=np.asarray(times,dtype=float);mask=np.asarray(valid,dtype=bool)
    if not len(t):return max(0.,end-start) if start is not None and end is not None else None
    period=float(np.median(np.diff(t))) if len(t)>1 else 0.
    longest=max(0.,t[0]-start) if start is not None else 0.
    pending=start if start is not None and not mask[0] else None
    for i,(now,ok) in enumerate(zip(t,mask)):
        if i and now-t[i-1]>max(.2,period*2.5):longest=max(longest,now-t[i-1])
        if not ok and pending is None:pending=now
        elif ok and pending is not None:longest=max(longest,now-pending);pending=None
    if pending is not None:longest=max(longest,(end if end is not None else t[-1]+period)-pending)
    if end is not None:longest=max(longest,end-t[-1])
    return float(longest)
def verdict(ok,available=True):return 'BLOCKED' if not available else 'PASS' if ok else 'FAIL'
def scalar(v,digits=3):return '不可计算' if v is None else f'{v:.{digits}f}'
def scoring_state(task,manual,qgate,angles_valid=False,labels_valid=False):
    applicable=task in (3,4,5)
    return {'A_value':None,'C_value':None,'S_value':None,'A_valid':False,'C_valid':False,'S_valid':False,
            'A_status':'not_applicable' if not applicable else 'not_completed' if manual<10 else 'not_scorable',
            'C_status':'independent_labels_missing' if not labels_valid else 'unvalidated',
            'S_status':'not_applicable' if task not in (3,4) else 'not_scorable_unvalidated_angles',
            'overall_status':'not_completed' if not applicable or manual<10 else 'not_scorable',
            'supervisor_display':'data_unavailable_no_score','actuator_command_count':0}
def candidate_cycles(t,q):
    """Offline exploratory complete peak-trough-peak cycles, never bridged across gaps."""
    t=np.asarray(t);q=np.asarray(q)
    if len(t)<5:return 0
    splits=np.split(np.arange(len(t)),np.flatnonzero(np.diff(t)>.2)+1);count=0
    for ix in splits:
        if len(ix)<8:continue
        tt,qq=t[ix],q[ix];dt=np.median(np.diff(tt))
        if dt<=0:continue
        u=np.arange(tt[0],tt[-1]+dt/2,dt);signal=np.interp(u,tt,qq)
        distance=max(1,int(.6/dt))
        peaks=find_peaks(signal,prominence=6,distance=distance)[0]
        troughs=find_peaks(-signal,prominence=6,distance=distance)[0]
        count+=sum(np.any((troughs>a)&(troughs<b)) for a,b in zip(peaks[:-1],peaks[1:]))
    return int(count)

def verify_manifest(manifest,root,full=True):
    checked=0;issues=[]
    if not manifest.exists():return {'checked_files':0,'issues':['manifest_missing']}
    for line in manifest.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():continue
        expected,name=line.split(maxsplit=1);path=(root/name.lstrip('*')).resolve()
        # Raw manifests must not escape the declared run root.
        if root.resolve() not in path.parents:issues.append('unsafe_path:'+name);continue
        if not path.is_file():issues.append('missing:'+name);continue
        if full or path.suffix in ('.json','.jsonl'):
            checked+=1
            if digest(path)!=expected:issues.append('hash_mismatch:'+name)
    return {'checked_files':checked,'issues':issues,'scope':'all_manifest_files' if full else 'json/jsonl only'}

def run_audit(review_id,full_hash=True):
    index_hash=digest(SOURCE/'eight_task_sessions.json');selected=active_history(SOURCE)
    excluded=read(SOURCE/'excluded_trials.json');excluded_ids={r['run_id'] for r in excluded}
    assert len({r['run_id'] for r in selected})==len(selected),'duplicate selected run IDs'
    assert not excluded_ids&{r['run_id'] for r in selected},'excluded data reintroduced'
    out={n:BASE/f'TP-D435I-0{n}'/'closeout_review_v1'/review_id for n in range(1,5)}
    for p in out.values():p.mkdir(parents=True,exist_ok=False)
    inputs=[];inventory=[];issues=[];kp_rows=[];kin_rows=[];comp_rows=[];q_rows=[];latencies=[]
    formal_angles=0;total_points=0;capture_count=0;analysis_count=0;hash_count=0
    for s in selected:
        rid=s['run_id'];tid=s['task_id'];raw=Path(s['raw_directory']);m=read(raw/'run.json')
        frames=jsonl(raw/'frames.jsonl');analysis=jsonl(raw/'analysis.jsonl');events=jsonl(raw/'events.jsonl')
        n=len(frames);capture_count+=n;analysis_count+=len(analysis)
        frame_ids=[f['frame_id'] for f in frames];times=np.array([f['host_mono_s'] for f in frames]);start=m.get('started_host_mono_s');end=m.get('ended_host_mono_s')
        raw_issues=[]
        if len(set(frame_ids))!=n:raw_issues.append('duplicate_capture_frame_id')
        if n!=m.get('saved_frames') or n!=s['captured_frames']:raw_issues.append('capture_count_disagreement')
        if len(times)>1 and np.any(np.diff(times)<=0):raw_issues.append('nonmonotonic_capture_time')
        analysed_ids=[a['frame_id'] for a in analysis]
        if len(set(analysed_ids))!=len(analysed_ids):raw_issues.append('duplicate_analysis_frame_id')
        orphan_analysis=len(set(analysed_ids)-set(frame_ids))
        for f in frames:
            for name in ('rgb_file','depth_file'):
                path=raw/f[name]
                if not path.is_file() or path.stat().st_size==0:raw_issues.append('missing_or_empty:'+f[name])
        hashes=verify_manifest(raw/'files.sha256',raw,full_hash);hash_count+=hashes['checked_files'];raw_issues+=hashes['issues']
        paths={i:BASE/f'TP-D435I-0{i}'/'eight_task_v3'/rid for i in range(1,5)}
        kp=pd.read_csv(paths[1]/'keypoints.csv');kin=pd.read_csv(paths[2]/'wrist_kinematics.csv');comp=pd.read_csv(paths[3]/'compensation.csv')
        if list(kp.columns)!=COLS:raw_issues.append('keypoint_schema_not_23_columns')
        if kp.duplicated(['run_id','frame_id','joint_id']).any():raw_issues.append('duplicate_keypoint_grain')
        if set(kp.frame_id)!=set(frame_ids):raw_issues.append('keypoint_frame_join_mismatch')
        for name,df in [('kinematics',kin),('compensation',comp)]:
            if df.frame_id.duplicated().any() or set(df.frame_id)!=set(frame_ids):raw_issues.append(name+'_frame_join_mismatch')
        if flag(kp.imputed).any():raw_issues.append('imputed_points_present')
        invalid=~flag(kp.measurement_valid)
        if kp.loc[invalid,['x_raw_m','y_raw_m','z_raw_m']].notna().any().any():raw_issues.append('invalid_coordinates_not_nan')
        if set(kp.coord_frame.dropna())!={'camera_color_optical'}:raw_issues.append('coordinate_frame_mismatch')
        total_points+=len(kp)
        masks={};joint_stats={}
        for joint in JOINTS:
            df=kp[kp.joint_id==joint].set_index('frame_id').reindex(frame_ids)
            xyz=df[['x_raw_m','y_raw_m','z_raw_m']].to_numpy(float)
            valid=flag(df.valid_for_analysis).to_numpy()&np.isfinite(xyz).all(axis=1)&~flag(df.imputed).to_numpy()
            masks[joint]=valid;points=xyz[valid]
            sd=points.std(axis=0)*1000 if len(points)>1 else None
            p95=float(np.percentile(np.linalg.norm(points-np.median(points,axis=0),axis=1)*1000,95)) if len(points) else None
            longest=gap(times,valid,start,end);rate=float(valid.mean()) if n else None
            row={'run_id':rid,'participant_id':m['participant_id'],'task_id':tid,'joint_id':joint,'capture_frames':n,
                 'valid_frames':int(valid.sum()),'valid_rate':rate,'invalid_gap_max_s':longest,
                 'raw_sd_x_mm':sd[0] if sd is not None else None,'raw_sd_y_mm':sd[1] if sd is not None else None,'raw_sd_z_mm':sd[2] if sd is not None else None,
                 'position_p95_mm':p95,'availability_verdict':verdict(rate is not None and rate>=.95 and longest<=.2,bool(n)),
                 'static_stability_verdict':verdict(sd is not None and max(sd)<=5 and p95<=10,len(points)>1) if tid in (1,2) else 'NOT_APPLICABLE',
                 'accuracy_verdict':'BLOCKED','coordinate_source':'depth_surface_sample_not_validated_joint_center'}
            kp_rows.append(row);joint_stats[joint]=row
        kin=kin.set_index('frame_id').reindex(frame_ids);comp=comp.set_index('frame_id').reindex(frame_ids)
        angle_ok=flag(kin.angle_valid).to_numpy()&kin[['theta_fe_deg','theta_ru_deg']].notna().all(axis=1).to_numpy()
        formal_angles+=int(angle_ok.sum())
        collecting=(kin.trial_phase=='collecting').to_numpy()
        estimated=flag(kin.estimate_valid).to_numpy()&kin[['theta_fe_est_deg','theta_ru_est_deg']].notna().all(axis=1).to_numpy()
        est_active=estimated&collecting
        vals=kin.loc[est_active,['theta_fe_est_deg','theta_ru_est_deg']]
        kin_row={'run_id':rid,'task_id':tid,'capture_frames':n,'formal_angle_valid_frames':int(angle_ok.sum()),
                 'formal_angle_valid_rate':float(angle_ok.mean()) if n else None,'formal_angle_verdict':'BLOCKED',
                 'candidate_valid_frames':int(estimated.sum()),'candidate_valid_rate':float(estimated.mean()) if n else None,
                 'candidate_gap_max_s':gap(times,estimated,start,end),'candidate_active_samples':int(est_active.sum()),
                 'candidate_fe_range_deg':float(vals.theta_fe_est_deg.max()-vals.theta_fe_est_deg.min()) if len(vals) else None,
                 'candidate_ru_range_deg':float(vals.theta_ru_est_deg.max()-vals.theta_ru_est_deg.min()) if len(vals) else None,
                 'operator_reported_count':m.get('manual_count'),'online_candidate_count':m.get('estimated_reps'),
                 'offline_exploratory_cycles':candidate_cycles(times[est_active],vals.theta_fe_est_deg.to_numpy() if tid==3 else vals.theta_ru_est_deg.to_numpy()) if tid in (3,4) else None,
                 'independent_count_verdict':'BLOCKED','rotation_direction_verdict':'BLOCKED','reference_position_accuracy_verdict':'BLOCKED'}
        kin_rows.append(kin_row)
        cv=flag(comp.valid).to_numpy()&comp.state.notna().to_numpy()&(comp.state!='unknown').to_numpy()
        state_counts=Counter(comp.loc[cv,'state'].tolist());mode=sorted(state_counts,key=lambda x:(-state_counts[x],x))[0] if state_counts else 'unknown'
        alarm=cv&comp.state.isin(('shoulder_compensation','elbow_compensation','combined')).to_numpy()
        first=comp.loc[alarm,'output_mono_s'].dropna() if 'output_mono_s' in comp else pd.Series(dtype=float)
        comp_row={'run_id':rid,'task_id':tid,'instruction_label':m['expected_label'],'label_source':'instruction_not_ground_truth',
                  'capture_frames':n,'observable_frames':int(cv.sum()),'observable_rate':float(cv.mean()) if n else None,
                  'unknown_frames':int((~cv).sum()),'modal_observation':mode,'normal_frames':state_counts['normal'],
                  'shoulder_frames':state_counts['shoulder_compensation'],'elbow_frames':state_counts['elbow_compensation'],'combined_frames':state_counts['combined'],
                  'alarm_frames':int(alarm.sum()),'alarm_rate_among_observable':float(alarm.sum()/cv.sum()) if cv.sum() else None,
                  'trial_start_to_first_alert_s':float(first.iloc[0]-start) if len(first) and start is not None else None,
                  'independent_sensitivity':None,'independent_specificity':None,'event_detection_latency_p95_ms':None,
                  'classification_verdict':'BLOCKED','coverage_verdict':verdict(n and cv.mean()>=.95,bool(n))}
        comp_rows.append(comp_row)
        body=np.logical_and.reduce(list(masks.values()));body_rate=float(body.mean()) if n else None;body_gap=gap(times,body,start,end)
        gate=bool(n and body_rate>=.95 and body_gap<=.2)
        state=scoring_state(tid,m.get('manual_count',0),gate)
        delays=[a['processing_latency_ms'] for a in analysis if isinstance(a.get('processing_latency_ms'),(int,float))]
        latencies+=delays
        q_rows.append({'run_id':rid,'task_id':tid,'window_id':'whole_saved_capture_trial',**state,'Q_value':body_rate,'Q_valid':bool(n),
                       'Q_gap_max_s':body_gap,'score_data_gate':gate,'angle_input_valid':bool(n and angle_ok.mean()>=.95),
                       'Q_gate_verdict':verdict(gate,bool(n)),'host_receive_output_p95_ms':float(np.percentile(delays,95)) if delays else None,
                       'end_to_end_latency_verdict':'BLOCKED','parameters_version':'closeout-review/v1'})
        inv={'run_id':rid,'participant_id':m['participant_id'],'task_id':tid,'task_name':m['task_name'],'captured_frames':n,
             'analysis_records':len(analysis),'orphan_analysis_records':orphan_analysis,'queue_drops':m.get('dropped_frames'),
             'active_duration_s':m.get('active_duration_s'),'capture_wall_s':end-start if start is not None and end is not None else None,
             'pause_events':sum(e['event']=='pause' for e in events),'structural_issues':raw_issues,'sha256_checked_files':hashes['checked_files'],
             'missing_reference':m.get('reference_system') is None,'missing_camera_placement':m.get('camera_placement') is None,
             'missing_clock_mapping':m.get('device_host_clock_mapping') is None}
        inventory.append(inv)
        if raw_issues:issues.append({'run_id':rid,'issues':raw_issues})
        for p in [raw/'run.json',raw/'frames.jsonl',raw/'analysis.jsonl',raw/'events.jsonl',raw/'files.sha256',
                  paths[1]/'keypoints.csv',paths[2]/'wrist_kinematics.csv',paths[3]/'compensation.csv',paths[4]/'quality.csv']:
            inputs.append({'path':str(p),'sha256':digest(p),'run_id':rid})
        print(f'Checked {rid}: {n} capture frames, {len(analysis)} analysis records, {len(raw_issues)} structural issues',flush=True)
    assert digest(SOURCE/'eight_task_sessions.json')==index_hash,'selection changed during review'
    summary={'revision':'closeout-review/v1','review_id':review_id,'checked_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'participants':sorted({s['participant_id'] for s in selected}),'retained_trials':len(selected),'excluded_trials':len(excluded),
        'task_trial_counts':dict(sorted(Counter(s['task_id'] for s in selected).items())),'captured_rgb_depth_pairs':capture_count,
        'analysis_records':analysis_count,'keypoint_rows':total_points,'formal_angle_valid_frames':formal_angles,
        'raw_hash_checked_files':hash_count,'hash_scope':'all raw manifest files' if full_hash else 'json/jsonl only',
        'structural_issues':issues,'all_original_thresholds':THRESHOLDS,'selection_index_sha256':index_hash,
        'exclusion_manifest_sha256':digest(SOURCE/'excluded_trials.json'),
        'pooled_host_receive_to_output_p95_ms':float(np.percentile(latencies,95)) if latencies else None,
        'trial_inventory':inventory,'package_verdicts':{'TP-D435I-01':'FAIL','TP-D435I-02':'BLOCKED','TP-D435I-03':'BLOCKED','TP-D435I-04':'BLOCKED'},
        'qualification':'retained trial does not mean valid measurement; single participant/session; instruction labels are not independent labels; no robot commands'}
    avail=Counter(r['availability_verdict'] for r in kp_rows)
    static=Counter(r['static_stability_verdict'] for r in kp_rows if r['task_id'] in (1,2))
    summary.update(point_availability_checks=dict(avail),static_stability_checks=dict(static),
                   quality_gated_trials=sum(r['score_data_gate'] for r in q_rows),
                   tp03_independent_label_count=0,tp03_formal_event_onset_count=0)
    if not any(r['availability_verdict']=='FAIL' or r['static_stability_verdict']=='FAIL' for r in kp_rows):
        summary['package_verdicts']['TP-D435I-01']='BLOCKED'
    for num,rows,name in [(1,kp_rows,'keypoint_statistics.csv'),(2,kin_rows,'candidate_kinematics_summary.csv'),(3,comp_rows,'compensation_observations.csv'),(4,q_rows,'quality.csv')]:
        save_csv(out[num]/name,rows);save_json(out[num]/'summary.json',summary)
        save_json(out[num]/'input_manifest.json',{'source_files':inputs,'selection_sha256':index_hash,'exclusions':excluded,
                                               'script_sha256':digest(__file__),'thresholds':THRESHOLDS})
        save_json(out[num]/'metric_dictionary.json',{
            'positions':'aligned depth surface samples, camera_color_optical, meters; no validated anatomical centers',
            'static_sd':'per-axis population SD of finite original valid nonimputed positions; mm',
            'position_p95':'P95 distance to within-trial median on original valid positions; mm; not reference error',
            'rates':'valid original observations / all saved captured frames; skipped inference and unknown retained',
            'gaps':'maximum invalid duration including trial boundaries and capture gaps; seconds; full stored trial including pauses',
            'candidate_angles':'fixed neutral palm reference estimates, not validated FE/RU; no formal ROM or direction inference',
            'candidate_cycles':'offline complete peak-trough-peak per contiguous valid collecting segment; gaps >.2s split; prominence 6deg, distance .6s; exploratory parameters not evaluated as accuracy',
            'A_C_S':'formal values unavailable; NaN in CSV, null in JSON; no fabricated CV substitute or default reward',
            'Q':'all-three-body-points true validity fraction; observation reliability separate from clinical quality; gate .95 and gap .2s',
            'delay':'host-receive-to-output only; trial-start-to-first-alert is NOT event detection latency',
            'labels':'instruction metadata is not independent annotation; unknown not discarded; static labels not called normal training',
            'actuators':'none; display-only unavailable state'} )
    write_reports(summary,out,kp_rows,kin_rows,comp_rows,q_rows)
    make_plots(out[1],kp_rows)
    save_json(out[1]/'supervisor_replay.json',{'mode':'display_only','actuator_commands':[],
                                              'windows':[{'run_id':r['run_id'],'display':r['supervisor_display'],'score_valid':False} for r in q_rows]})
    legacy=read(BASE/'TP-D435I-04'/'tp04_synthesis_summary.json')
    stat=legacy['discriminant_validity_test']
    save_json(out[1]/'legacy_claim_reconciliation.json',{'legacy_tp04_p_value':stat['p_value'],'is_p_less_than_001':stat['p_value']<.001,
        'interpretation':'legacy within-P001 instruction-group exploration, not independent clinical discriminant validity',
        'legacy_tp03_computed':{k:v for k,v in read(BASE/'TP-D435I-03'/'tp03_synthesis_summary.json').items() if k in ('overall_classification_accuracy_pct','negative_control_specificity_pct','mean_detection_latency_s')},
        'legacy_frame_claim':30774,'independently_counted_retained_frames':capture_count})
    for folder in out.values():
        files=[f'{digest(p)}  {p.name}' for p in sorted(folder.iterdir()) if p.is_file() and p.name!='files.sha256']
        (folder/'files.sha256').write_text('\n'.join(files)+'\n',encoding='utf-8')
    print(json.dumps(clean({k:v for k,v in summary.items() if k!='trial_inventory'}),ensure_ascii=False,indent=2),flush=True)
    return summary,out

def write_reports(s,out,kp,kin,comp,q):
    for num,folder in out.items():
        title=['三维关键点','腕关节轨迹','肩肘代偿','运动质量'][num-1]
        lines=[f'# TP-D435I-0{num} {title}：收尾复核结果',
            f'\n结论：**{s["package_verdicts"][f"TP-D435I-0{num}"]}**。基于原任务包阈值，未降低要求。',
            f'\n保留试次 {s["retained_trials"]}，参与者 {", ".join(s["participants"])}，原始 RGB/对齐深度 {s["captured_rgb_depth_pairs"]:,} 对，分析记录 {s["analysis_records"]:,} 条。排除 4 次依据用户选择，不称已证实异常。',
            '\n选择索引、逐文件输入哈希、配置、脚本版本见 `input_manifest.json`。全部保留试次纳入分母，不仅报告最好片段。',
            '\n## 验收项与证据']
        if num==1:
            lines+=['\n|项目|原目标|复核|结论|','|---|---|---|---|',
                    f'|每试次每关节可用性|有效率≥95%，最大失效≤0.2s|{s["point_availability_checks"]}|'+s['package_verdicts']['TP-D435I-01']+'|',
                    f'|S1/S2原始稳定性|各轴SD≤5mm、位置P95≤10mm|{s["static_stability_checks"]}|见逐项PASS/FAIL/BLOCKED|',
                    '|独立3D准确性|RMSE≤10mm、误差P95≤20mm|无独立参考|BLOCKED|',
                    '\n|试次|关节|有效率|最长无效(s)|SD X/Y/Z(mm)|位置P95(mm)|可用性|静态波动|',
                    '|---|---|---:|---:|---|---:|---|---|']
            for r in kp:lines.append(f'|{r["run_id"]}|{r["joint_id"]}|{scalar(r["valid_rate"])}|{scalar(r["invalid_gap_max_s"])}|'+ '/'.join(scalar(r[f'raw_sd_{x}_mm']) for x in 'xyz')+f'|{scalar(r["position_p95_mm"])}|{r["availability_verdict"]}|{r["static_stability_verdict"]}|')
            lines+=['\n表面深度点不等于已验证解剖关节中心。S1/S2 是单人重复条件，抖动变化不能归因于机械支撑或相机精度；不能据此解除下游准确度依赖。']
        elif num==2:
            lines+=['\n|项目|原目标|现有证据|结论|','|---|---|---|---|',
                    '|位置误差|独立参考3D RMSE≤10mm|无参考|BLOCKED|',
                    '|FE/RU角度/ROM|RMSE≤5°、P95≤10°、ROM误差≤5°|正式有效角度帧 '+str(s['formal_angle_valid_frames'])+'；正式角度列NaN|BLOCKED|',
                    '|角度连续性|≥95%，失效≤0.2s|候选估计覆盖不能替代正式角度覆盖|BLOCKED|',
                    '|计数与方向|每10次误差≤1，不混正反向|操作员点击/候选数不构成独立视频计数；方向参考未标定|BLOCKED|',
                    '|同步|参考时钟映射残差≤半帧|无独立参考/时钟映射|BLOCKED|',
                    '\n|试次|任务|候选覆盖|正式角度覆盖|候选FE范围(°)|候选RU范围(°)|操作员次数|在线参考数|离线探索数|',
                    '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
            for r in kin:
                if r['task_id'] in (3,4,5):lines.append(f'|{r["run_id"]}|{r["task_id"]}|{scalar(r["candidate_valid_rate"])}|{scalar(r["formal_angle_valid_rate"])}|{scalar(r["candidate_fe_range_deg"])}|{scalar(r["candidate_ru_range_deg"])}|{r["operator_reported_count"]}|{r["online_candidate_count"]}|{r["offline_exploratory_cycles"]}|')
            lines+=['\n候选范围仅为有效活动样本的最大值减最小值，不称解剖ROM。未校准零位/符号/前臂轴向参考；不从手/腕位移比推断动作已正确解耦。所有试次保留，失效不跳过。']
        elif num==3:
            lines+=['\n|项目|原目标|现有证据|结论|','|---|---|---|---|',
                    '|肩、肘检测性能|独立测试集各敏感度/特异度≥0.90|没有双人独立标签及隔离测试集|BLOCKED|',
                    '|组合混淆/一致性|独立标签、计数、置信区间|只有指令条件及观测分布|BLOCKED|',
                    '|事件检测延迟|可观测事件P95≤300ms|没有独立事件起点与同步映射|BLOCKED|',
                    '|可判读覆盖|候选≥95%，unknown保留|见每次observable_rate|逐试次见CSV|',
                    '\n|试次|指令条件（非真值）|可判读覆盖|unknown帧|众数观测|报警占可判读比例|试次开始到首警(s)|',
                    '|---|---|---:|---:|---|---:|---:|']
            for r in comp:lines.append(f'|{r["run_id"]}|{r["instruction_label"]}|{scalar(r["observable_rate"])}|{r["unknown_frames"]}|{r["modal_observation"]}|{scalar(r["alarm_rate_among_observable"])}|{scalar(r["trial_start_to_first_alert_s"])}|')
            lines+=['\n首警时间以实际主机输出时间减试次开始，仅是流程观测，不是独立事件起点的检测延迟。静止不视作完成腕训练的normal真值。不能用试次众数为normal证明整次零虚警。没有独立标签时不发布灵敏度、特异度或准确率。']
        else:
            lines+=['\n|项目|原目标|复核|结论|','|---|---|---|---|',
                    '|A/C/S解释与输入|经验证角度、代偿及任务定义|值保留NaN，各自有效性/原因明确|正式评分BLOCKED|',
                    f'|数据门控|真实有效率≥95%、最大失效≤0.2s|{s["quality_gated_trials"]}/{s["retained_trials"]} 窗满足观测门控|'+verdict(s['quality_gated_trials']==s['retained_trials'])+'|',
                    '|独立判别力|AUC≥0.85、独立标签与置信区间|单人同次指令分组不满足|BLOCKED|',
                    '|延迟|采集到输出P95≤200ms|仅主机接收到输出；无设备时钟映射|BLOCKED|',
                    '|重复性|跨次重复性及置信区间|单人重复数据可描述，不能替代跨人/跨session验证|BLOCKED|',
                    '|安全监督|无执行器命令、缺失不增加助力|display-only replay，actuator_commands=[]|代码/回放边界可检查；硬件联调BLOCKED|',
                    '\n|试次|Q观测比例|最大无效(s)|评分门控|A状态|总体状态|',
                    '|---|---:|---:|---|---|---|']
            for r in q:lines.append(f'|{r["run_id"]}|{scalar(r["Q_value"])}|{scalar(r["Q_gap_max_s"])}|{r["Q_gate_verdict"]}|{r["A_status"]}|{r["overall_status"]}|')
            lines+=['\nQ是测量覆盖，不是临床运动质量。不能给静止/诱导 A=1；缺失SPARC不以周期CV或0.5补分；不设置未经验证的机器人辅助/脱开动作。旧Welch p=0.001345 不满足 p<0.001，且试次来自同一P001，不能外推独立群体临床效度。']
        lines+=['\n## 复现与范围',f'\n统一命令：项目根目录 `.venv/Scripts/python.exe 04_实验设计/D435i_感知验证/TP-D435I-01/closeout_review_v1/review_four_packages.py --review-id <新的ID>`；实际命令/退出码另见 `execution.json`。',
                '\n原始运行、完成的逐试次结果及冻结任务书未覆盖。本版本另存，旧综合初稿不能作为验收依据。单人单session、摆位/参考/授权类型等未提供元信息保留未知，不补造。',
                '\n下一步先修正实际可用性与失效，再按原任务补参考、独立视频事件标注、跨session数据隔离及采集到输出时钟映射；质量验证和控制联调继续受各自准入约束。']
        (folder/'result.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    master=['# D435i 四问任务包：收尾复核与修订结论',
            f'\n保留 **{s["retained_trials"]} 个试次、{s["captured_rgb_depth_pairs"]:,} 对 RGB/对齐深度、{s["analysis_records"]:,} 条分析记录**；{len(s["participants"])} 名参与者，四次按用户要求排除。保留不等于测量有效。',
            '\n原结题报告的“四包全PASS、正式验收通过”不受当前证据支持，本修订版撤回该结论。',
            '\n|任务包|复核裁决|已有工作|未达标／阻塞|','|---|---|---|---|']
    descriptions=[('三维关键点','真实采集、契约表、原始波动与有效性统计','原阈值可用性/稳定性存在FAIL；独立准确性BLOCKED'),
                  ('腕关节轨迹','腕/手点轨迹、候选角及操作记录','解剖角、独立参考/计数/方向验证BLOCKED'),
                  ('肩肘代偿','双通道观测、未知覆盖和首警时刻','独立标签、测试划分、真正事件延迟BLOCKED'),
                  ('运动质量','A/C/S/Q有效性、观测门控、只读监督回放','数据门控FAIL；评分和效度/端到端延迟BLOCKED')]
    for n,(title,done,blocked) in enumerate(descriptions,1):
        rel=os.path.relpath(out[n]/'result.md',out[1]).replace('\\','/')
        master.append(f'|[TP-D435I-0{n} {title}]({rel})|**{s["package_verdicts"][f"TP-D435I-0{n}"]}**|{done}|{blocked}|')
    master+=['\n## 纠正的关键问题',
             '\n1. 原始保留帧由29,938对独立复数确认，替换30,774硬编码；四个排除编号不重新纳入。',
             '\n2. 恢复原目标：各轴SD≤5mm、位置P95≤10mm、有效率≥95%、最长失效≤0.2s；不以更松RMS/失效目标宣布PASS。',
             '\n3. 指令条件不是真值，试次多数类别不是零误报证据；灵敏度/特异度/事件延迟待独立标注。',
             '\n4. 候选角不能当解剖FE/RU或ROM；缺失不补默认角度、平滑分或可靠性0.6。',
             '\n5. p=0.001345 >0.001；单人反复试次不是独立受试者群体，撤回临床分离效度和机器人控制安全证明。',
             '\n## 图表与复核范围', '\n![逐试次观测覆盖及原95%目标](observation_coverage.png)',
             '\n![六次静态试次位置波动及原10mm目标](static_position_p95.png)',
             f'\n原始清单 SHA-256 校验范围：{s["hash_scope"]}；已校验文件 {s["raw_hash_checked_files"]:,}。契约/键/关联检查覆盖25个保留试次，结构问题见 `summary.json`，不将文件完整性当运动性能。',
             '\n本版本结论可用于真实工程阶段汇报，不能作为四包正式通过或临床/机器人闭环验收。实际执行和交付缺口见 `execution.json`、`deliverable_checklist.md`。']
    (out[1]/'MASTER_REVIEW.md').write_text('\n'.join(master)+'\n',encoding='utf-8')

def make_plots(folder,kp):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(dict.fromkeys(r['run_id'] for r in kp))
    fig,ax=plt.subplots(figsize=(12,5));x=np.arange(len(names))
    for j in JOINTS:
        values=[next(r['valid_rate'] for r in kp if r['run_id']==rid and r['joint_id']==j) for rid in names]
        ax.plot(x,np.array(values)*100,'o-',label=j,linewidth=1)
    ax.axhline(95,color='red',linestyle='--',label='original 95% target');ax.set_ylim(0,103)
    ax.set_xticks(x,[str(i+1) for i in x]);ax.set_ylabel('Valid observations / all saved frames (%)');ax.set_xlabel('Retained trial order (see inventory)');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(folder/'observation_coverage.png',dpi=160);plt.close(fig)
    static=[r for r in kp if r['task_id'] in (1,2)];fig,ax=plt.subplots(figsize=(10,4))
    for i,j in enumerate(JOINTS):
        rr=[r for r in static if r['joint_id']==j];ax.scatter(np.arange(6)+i*.12,[r['position_p95_mm'] for r in rr],label=j)
    ax.axhline(10,color='red',linestyle='--',label='original 10 mm target');ax.set_xticks(range(6),['S1-1','S1-2','S1-3','S2-1','S2-2','S2-3']);ax.set_ylim(bottom=0)
    ax.set_ylabel('Position fluctuation P95 (mm), not reference error');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(folder/'static_position_p95.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--review-id',required=True);parser.add_argument('--metadata-hash-only',action='store_true')
    args=parser.parse_args()
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',args.review_id):parser.error('invalid review ID')
    run_audit(args.review_id,not args.metadata_hash_only)
