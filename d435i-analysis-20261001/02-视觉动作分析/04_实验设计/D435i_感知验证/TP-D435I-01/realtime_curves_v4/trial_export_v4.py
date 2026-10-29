"""Traceable per-run exports for TP-01..04; absent references remain BLOCKED."""
import csv
import hashlib
import json
from pathlib import Path
import math
import numpy as np

KEYPOINT_COLUMNS='run_id,frame_id,joint_id,t_rgb_device_s,t_depth_device_s,t_host_mono_s,clock_domain,u_px,v_px,x_raw_m,y_raw_m,z_raw_m,x_filt_m,y_filt_m,z_filt_m,keypoint_score,visibility,depth_valid,measurement_valid,imputed,valid_for_analysis,invalid_reason,coord_frame'.split(',')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_csv(path,rows,columns):
    with Path(path).open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def numeric(value):return float('nan') if value is None else value


def max_gap(times,mask,start=None,end=None):
    if not times:return max(0.,end-start) if start is not None and end is not None else None
    times=np.asarray(times);mask=np.asarray(mask,dtype=bool)
    period=float(np.median(np.diff(times))) if len(times)>1 else 0.
    starts=start if start is not None and not mask[0] else None;longest=0.
    for i,(time,valid) in enumerate(zip(times,mask)):
        if i and time-times[i-1]>max(.2,period*2.5):longest=max(longest,time-times[i-1])
        if not valid and starts is None:starts=time
        if valid and starts is not None:longest=max(longest,time-starts);starts=None
    if starts is not None:longest=max(longest,(end if end is not None else times[-1]+period)-starts)
    if start is not None:longest=max(longest,times[0]-start)
    if end is not None:longest=max(longest,end-times[-1])
    return float(longest)


def sparc(speed,fs):
    if len(speed)<12 or np.max(speed)<.001:return None
    fft=np.abs(np.fft.rfft(speed,n=2**(int(np.ceil(np.log2(len(speed))))+3)))
    if fft[0]<=1e-9:return None
    fft/=fft[0];freq=np.fft.rfftfreq((len(fft)-1)*2,1/fs)
    indexes=np.flatnonzero((freq<=10)&(fft>=.05))
    if not len(indexes) or indexes[-1]<2:return None
    end=indexes[-1];x=freq[:end+1]/freq[end];y=fft[:end+1]
    return float(-np.sum(np.sqrt(np.diff(x)**2+np.diff(y)**2)))


def exploratory_smoothness(analyses,task_id):
    if task_id not in (3,4):return {'status':'not_applicable','values':[]}
    from scipy.signal import find_peaks,savgol_filter
    channel='theta_fe_est_deg' if task_id==3 else 'theta_ru_est_deg'
    rows=[a for a in analyses if a.get('trial_phase')=='collecting']
    good=[a for a in rows if a.get('estimate_valid') and a.get(channel) is not None]
    if len(good)<30 or len(good)/max(1,len(rows))<.95:return {'status':'insufficient_valid_estimates','values':[]}
    t=np.array([a['received_mono_s'] for a in good]);q=np.array([a[channel] for a in good])
    if np.max(np.diff(t))>.2 or t[-1]-t[0]<1 or np.ptp(q)<10:return {'status':'static_or_gapped','values':[]}
    dt=float(np.median(np.diff(t)));fs=1/dt;u=np.arange(t[0],t[-1],dt)
    q=np.interp(u,t,q)
    window=min(9,len(q)//2*2-1)
    q=savgol_filter(q,window,2) if window>=5 else q
    distance=max(1,int(.45*fs))
    extrema=np.unique(np.concatenate((find_peaks(q,prominence=5,distance=distance)[0],find_peaks(-q,prominence=5,distance=distance)[0])))
    values=[]
    for start,end in zip(extrema[:-1],extrema[1:]):
        if end-start>=12 and abs(q[end]-q[start])>=10:
            value=sparc(np.abs(np.gradient(q[start:end+1],dt)),fs)
            if value is not None:values.append(value)
    return {'status':'exploratory_unvalidated_angle' if values else 'no_valid_submovements',
            'values':values,'mean':float(np.mean(values)) if values else None,
            'signal':'estimated angular speed deg/s','offline_filter':'uniform resampling + Savitzky-Golay 9/2',
            'formal_valid':False}


def export_trial(raw,root):
    raw=Path(raw);root=Path(root)
    meta=json.loads((raw/'run.json').read_text(encoding='utf-8'))
    frames=[json.loads(line) for line in (raw/'frames.jsonl').read_text(encoding='utf-8').splitlines() if line]
    analyses=[json.loads(line) for line in (raw/'analysis.jsonl').read_text(encoding='utf-8').splitlines() if line]
    by_frame={a['frame_id']:a for a in analyses}
    trial_id=meta['run_id'];task_id=meta['task_id'];times=[f['host_mono_s'] for f in frames]
    start_time=meta.get('started_host_mono_s');end_time=meta.get('ended_host_mono_s')
    keypoints=[];kinematics=[];compensation=[]
    # All captured frames are the denominator, including frames skipped by inference.
    masks={k:[] for k in ('shoulder','elbow','wrist')};positions={k:[] for k in masks}
    estimate_mask=[];comp_mask=[];latencies=[];active_times=[];frame_ids=[]
    source_names=('r_shoulder','r_elbow','r_wrist','l_shoulder','r_index','r_pinky',
                  'hand_wrist','hand_index_mcp','hand_middle_mcp','hand_pinky_mcp')
    names={'r_shoulder':'shoulder','r_elbow':'elbow','r_wrist':'wrist','l_shoulder':'contralateral_shoulder'}
    for f in frames:
        a=by_frame.get(f['frame_id'],{});measurements=a.get('measurements',{})
        active=a.get('trial_phase')=='collecting'
        frame_ids.append(f['frame_id'])
        for name in source_names:
            joint=names.get(name,name);m=measurements.get(name,{})
            point=m.get('point_m');valid=bool(m.get('measurement_valid'))
            analysis_valid=valid and bool(a.get('body_valid',False)) if name in ('r_shoulder','r_elbow','r_wrist') else valid
            reason=m.get('invalid_reason','unanalysed_frame' if not a else 'missing_keypoint')
            if valid and not analysis_valid:reason='body_geometry_invalid'
            row=dict(zip(KEYPOINT_COLUMNS,[None]*len(KEYPOINT_COLUMNS)))
            row.update(run_id=trial_id,frame_id=f['frame_id'],joint_id=joint,
                       t_rgb_device_s=f['rgb_device_s'],t_depth_device_s=f['depth_device_s'],t_host_mono_s=f['host_mono_s'],
                       clock_domain=f['rgb_clock_domain']+' / '+f['depth_clock_domain'],
                       u_px=numeric(m.get('u_px')),v_px=numeric(m.get('v_px')),
                       keypoint_score=numeric(m.get('keypoint_score')),visibility=numeric(m.get('visibility')),
                       depth_valid=bool(m.get('depth_valid')),measurement_valid=valid,imputed=False,
                       valid_for_analysis=analysis_valid,invalid_reason=reason,coord_frame='camera_color_optical')
            # No point filter is applied: the same valid coordinate is explicitly copied.
            for i,axis in enumerate('xyz'):
                row[f'{axis}_raw_m']=point[i] if valid else float('nan')
                row[f'{axis}_filt_m']=point[i] if valid else float('nan')
            keypoints.append(row)
            if joint in masks:
                masks[joint].append(analysis_valid)
                if analysis_valid:positions[joint].append(point)
        estimated=bool(a.get('estimate_valid'));estimate_mask.append(estimated)
        comp_mask.append(bool(a.get('compensation_valid')))
        if 'processing_latency_ms' in a:latencies.append(a['processing_latency_ms'])
        active_times.append(active)
        wrist=measurements.get('r_wrist',{}).get('point_m');hand=measurements.get('hand_middle_mcp',{}).get('point_m')
        kr={'run_id':trial_id,'frame_id':f['frame_id'],'t_host_mono_s':f['host_mono_s'],
            'task_id':task_id,'trial_phase':a.get('trial_phase','unanalysed'),
            'theta_fe_deg':float('nan'),'theta_ru_deg':float('nan'),'angle_valid':False,
            'theta_fe_est_deg':numeric(a.get('theta_fe_est_deg')),'theta_ru_est_deg':numeric(a.get('theta_ru_est_deg')),
            'estimate_valid':estimated,'angle_invalid_reason':'forearm_axial_reference_and_independent_validation_missing',
            'wrist_position_valid':wrist is not None,'hand_position_valid':hand is not None,
            'calibration_id':a.get('calibration_id'),'coord_frame':'camera_color_optical'}
        for i,axis in enumerate('xyz'):
            kr[f'wrist_{axis}_m']=wrist[i] if wrist is not None else float('nan')
            kr[f'hand_{axis}_m']=hand[i] if hand is not None else float('nan')
        kinematics.append(kr)
        compensation.append({'run_id':trial_id,'frame_id':f['frame_id'],'t_host_mono_s':f['host_mono_s'],
            'task_id':task_id,'shoulder_displacement_mm':a.get('shoulder_abs_mm'),'elbow_displacement_mm':a.get('elbow_mm'),
            'elbow_angle_change_deg':a.get('elbow_deg'),'shoulder_prediction':a.get('shoulder_alert') if a.get('compensation_valid') else None,
            'upper_arm_swing_deg':a.get('upper_arm_swing_deg'),'bilateral_shoulder_height_change_mm':a.get('shoulder_mm'),
            'elbow_prediction':a.get('elbow_alert') if a.get('compensation_valid') else None,
            'state':a.get('compensation_state','unknown'),'valid':a.get('compensation_valid',False),
            'prediction_confidence':float('nan'),'confidence_kind':'heuristic_threshold_no_calibrated_probability',
            'output_mono_s':numeric(a.get('output_mono_s')),'event_time_mono_s':numeric(a.get('output_mono_s')),
            'unknown_reason':'' if a.get('compensation_valid') else 'missing_or_unanalysed_body_reference',
            'instruction_label':meta['expected_label'],'independent_label':None,'label_source':'instruction_only_not_ground_truth'})
    n=len(frames)
    stats={}
    for name,mask in masks.items():
        points=np.array(positions[name]);rate=float(np.mean(mask)) if mask else 0.
        sd=points.std(axis=0)*1000 if len(points)>1 else None
        drift=np.linalg.norm(points-np.median(points,axis=0),axis=1)*1000 if len(points) else []
        gap=max_gap(times,mask,start_time,end_time)
        static=task_id in (1,2)
        stats[name]={'valid_rate':rate,'gap_max_s':gap,'raw_axis_sd_mm':sd.tolist() if sd is not None else None,
            'position_fluctuation_p95_mm':float(np.percentile(drift,95)) if len(drift) else None,
            'availability_verdict':'PASS' if n and rate>=.95 and gap is not None and gap<=.2 else 'FAIL' if n else 'BLOCKED',
            'static_fluctuation_verdict':('PASS' if sd is not None and max(sd)<=5 and len(drift) and np.percentile(drift,95)<=10 else 'FAIL') if static else 'NOT_APPLICABLE',
            'accuracy_verdict':'BLOCKED'}
    smooth=exploratory_smoothness(analyses,task_id)
    bodymask=[all(masks[j][i] for j in masks) for i in range(n)]
    reliability=float(np.mean(bodymask)) if bodymask else 0.
    complete=(meta.get('manual_clockwise',0)>=5 and meta.get('manual_counterclockwise',0)>=5) if task_id==5 else meta.get('manual_count',0)>=10
    quality={'run_id':trial_id,'task_id':task_id,'window_id':'whole_trial','start_mono_s':times[0] if times else None,
      'end_mono_s':times[-1] if times else None,'A_value':float('nan'),'C_value':float('nan'),'S_value':float('nan'),'Q_value':reliability,
      'A_valid':False,'A_status':'not_applicable' if task_id in (1,2,6,7,8) else 'not_completed' if not complete else 'not_scorable_angle_unvalidated',
      'reported_cycles':meta.get('manual_count',0),'estimated_cycles':meta.get('estimated_reps',0),
      'C_valid':False,'C_status':'independent_labels_missing','S_valid':False,'S_status':smooth['status'],
      'Q_valid':bool(n),'Q_body_valid_rate':reliability,'Q_max_gap_s':max_gap(times,bodymask,start_time,end_time),
      'angle_estimate_coverage':float(np.mean(estimate_mask)) if n else 0.,
      'compensation_coverage':float(np.mean(comp_mask)) if n else 0.,
      'host_receive_to_output_p95_ms':float(np.percentile(latencies,95)) if latencies else None,
      'end_to_end_latency_verdict':'BLOCKED','reason':'no_independent_reference_or_labels; no_device_host_clock_mapping',
      'parameters_version':'eight-task/v3','overall_status':'not_scorable'}
    # JSON uses null for unavailable values; CSV preserves NaN explicitly.
    quality_json={k:None if isinstance(v,float) and math.isnan(v) else v for k,v in quality.items()}
    candidate_rom={}
    for channel in ('theta_fe_est_deg','theta_ru_est_deg'):
        values=[a[channel] for a in analyses if a.get('trial_phase')=='collecting' and a.get('estimate_valid') and a.get(channel) is not None]
        candidate_rom[channel]={'valid_samples':len(values),'min_deg':min(values) if values else None,
                               'max_deg':max(values) if values else None,'range_deg':max(values)-min(values) if values else None,
                               'formal_rom_valid':False,'source':'constrained_reference_estimate'}
    summary={'run_id':trial_id,'participant_id':meta['participant_id'],'task_id':task_id,'task_name':meta['task_name'],
      'captured_frames':n,'analysed_frames':len(analyses),'saved_frames':meta.get('saved_frames'),
      'capture_queue_drops':meta.get('dropped_frames',0),'active_duration_s':meta.get('active_duration_s'),
      'finish_reason':meta.get('finish_reason'),'estimated_reps':meta.get('estimated_reps',0),'manual_count':meta.get('manual_count',0),
      'static_duration_verdict':('PASS' if meta.get('active_duration_s',0)>=29.9 else 'FAIL') if task_id in (1,2) else 'NOT_APPLICABLE',
      'keypoint_statistics':stats,'quality':quality_json,'smoothness_exploratory':smooth,'candidate_angle_ranges':candidate_rom,
      'operator_reported_completion':complete if task_id in (3,4,5) else None,
      'manual_clockwise':meta.get('manual_clockwise',0),'manual_counterclockwise':meta.get('manual_counterclockwise',0),
      'formal_verdicts':{'TP-01':'BLOCKED_ACCURACY','TP-02':'BLOCKED_ANATOMICAL_ANGLES_AND_REFERENCE',
                         'TP-03':'BLOCKED_INDEPENDENT_LABELS','TP-04':'BLOCKED_INPUT_VALIDATION'},
      'observed_source_frame_skips':sum(max(0,b-a-1) for a,b in zip(frame_ids[:-1],frame_ids[1:]))}
    summary['capture_wall_duration_s']=end_time-start_time if start_time is not None and end_time is not None else None
    summary['capture_hz_observed']=n/summary['capture_wall_duration_s'] if summary['capture_wall_duration_s'] else None
    source_hash=digest(raw/'run.json');analysis_hash=digest(raw/'analysis.jsonl')
    summary['source_hashes']={'run.json':source_hash,'analysis.jsonl':analysis_hash,'frames.jsonl':digest(raw/'frames.jsonl')}
    outdirs=[]
    for number in range(1,5):
        directory=root/'04_实验设计'/'D435i_感知验证'/f'TP-D435I-0{number}'/'realtime_curves_v4'/trial_id
        directory.mkdir(parents=True,exist_ok=False);outdirs.append(directory)
        if number==1:write_csv(directory/'keypoints.csv',keypoints,KEYPOINT_COLUMNS)
        elif number==2:write_csv(directory/'wrist_kinematics.csv',kinematics,list(kinematics[0]) if kinematics else ['run_id','frame_id','theta_fe_deg','theta_ru_deg','angle_valid'])
        elif number==3:write_csv(directory/'compensation.csv',compensation,list(compensation[0]) if compensation else ['run_id','frame_id','state','valid','independent_label'])
        else:write_csv(directory/'quality.csv',[quality],list(quality))
        (directory/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        dictionary={
          'A':{'input':'independently validated FE/RU, target and cycles','formula':None,'unit':None,'valid':False,'invalid_reason':'angles and reference unvalidated'},
          'C':{'input':'shoulder/elbow regional movement features','formula':None,'unit':None,'valid':False,'invalid_reason':'independent labels and causal validation missing; no composite index frozen'},
          'S':{'input':'candidate angular speed per FE/RU submovement','formula':'spectral arc length, normalized FFT magnitude >=0.05, <=10 Hz','unit':'dimensionless','direction':'higher is smoother',
               'formal_valid':False,'invalid_reason':'candidate angles unvalidated; static/incomplete/gapped segments excluded','normalization':'frequency/max included frequency; FFT magnitude/DC'},
          'Q':{'input':'all saved capture frames, including unanalysed frames','formula':'frames with all three body points valid / saved capture frames','unit':'fraction','direction':'higher has more observed frames',
               'supplements':'largest invalid gap includes capture gaps and trial boundaries; queue drop counts separate','normalization':'0..1; no clinical score'},
          'positions':{'source':'aligned depth surface deprojection','unit':'m','coord_frame':'camera_color_optical','joint_center_validated':False,'filter':'none'},
          'angles':{'formal_columns':'theta_fe_deg/theta_ru_deg = NaN','candidate_columns':'theta_fe_est_deg/theta_ru_est_deg','unit':'deg',
                    'assumption':'fixed neutral palm normal; unobserved forearm axial rotation','filter':'causal EMA alpha .55'},
          'timing':{'host_receive_to_output_ms':'host monotonic output minus host frame receipt; does not establish device end-to-end latency','device_host_mapping':None}}
        (directory/'metric_dictionary.json').write_text(json.dumps(dictionary,ensure_ascii=False,indent=2),encoding='utf-8')
        verdict=summary['formal_verdicts'][f'TP-0{number}']
        lines=[f'# TP-D435I-0{number}：{meta["task_name"]} 工程记录',
          f'\n运行：`{trial_id}`；参与者代号：`{meta["participant_id"]}`。',
          f'\n正式验收状态：**{verdict}**。采集成功不等于全部验收通过。',
          f'\n真实输入：{raw}；记录 SHA-256：`{source_hash}`。',
          f'\n采集帧 {n}，分析帧 {len(analyses)}；活动时长 {meta.get("active_duration_s",0):.2f} 秒；保存队列丢帧 {meta.get("dropped_frames",0)}。',
          '\n## 工程实测', '\n关节 | 有效率 | 最长失效秒 | 位置波动P95毫米 | 可用性 | 准确性',
          '---|---:|---:|---:|---|---']
        for name,s in stats.items():lines.append(f'{name}|{s["valid_rate"]:.3f}|{s["gap_max_s"]}|{s["position_fluctuation_p95_mm"]}|{s["availability_verdict"]}|BLOCKED')
        lines+=['\n详细指标、各轴原始 SD、失败条件、数据覆盖与处理延迟见 `summary.json`。',
          '\n## 阻塞和适用范围',
          '\n没有独立位置／角度参考或独立人工标签。指令名称仅为诱导条件，不充当真值。FE/RU 候选值依赖固定中立手掌参考，无法观察前臂轴向旋转；正式角度列为 NaN。记录角度估计与独立有效性，不混充准确测量。SPARC 若可计算仅为离线候选角速度子动作探索结果。',
          '\n程序、环境、真实命令和软件测试入口见 TP-01/attempt_01/EIGHT_TASKS_V3.md。后续需至少重复试次、重新摆位、独立参考与双人事件标注，再冻结测试划分并重新验收。']
        (directory/'result.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        manifest=[]
        for file in directory.iterdir():
            if file.is_file():manifest.append(f'{digest(file)}  {file.name}')
        (directory/'files.sha256').write_text('\n'.join(manifest)+'\n',encoding='utf-8')
    summary['output_directories']=[str(d) for d in outdirs]
    return summary
