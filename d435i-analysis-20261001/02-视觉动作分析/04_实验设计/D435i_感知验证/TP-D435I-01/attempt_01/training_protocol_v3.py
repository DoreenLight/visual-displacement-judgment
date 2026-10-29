"""Eight-task engineering protocol. Candidate angles are distinct from validated FE/RU."""
from collections import deque
import math
import numpy as np
from feedback_engine_v2 import angle

TASKS = [
 {'id':1,'code':'S1','name':'自然坐姿静止','group':'静态稳定性','duration_s':30,'target_reps':0,'expected_label':'not_applicable','instruction':'手臂自然放在桌面，肩、肘、腕可见，放松静止 30 秒。','package':'TP-01'},
 {'id':2,'code':'S2','name':'握柄姿态静止','group':'静态稳定性','duration_s':30,'target_reps':0,'expected_label':'not_applicable','instruction':'握住实际手柄、握力器或水杯，保持准备姿势静止 30 秒；保留真实遮挡。','package':'TP-01'},
 {'id':3,'code':'FE','name':'标准腕屈伸','group':'正常腕动作','duration_s':None,'target_reps':10,'expected_label':'normal','instruction':'肩肘保持支撑，只做腕背屈／掌屈，候选幅度 −30°～+30°，每往返约 2～3 秒，共 10 次。','package':'TP-02'},
 {'id':4,'code':'RU','name':'标准桡偏／尺偏','group':'正常腕动作','duration_s':None,'target_reps':10,'expected_label':'normal','instruction':'保持手心向下或对侧，仅腕左右偏摆，候选幅度 −20°～+20°，每往返约 2～3 秒，共 10 次。','package':'TP-02'},
 {'id':5,'code':'CIRCLE','name':'腕部复合回转','group':'正常腕动作','duration_s':None,'target_reps':10,'expected_label':'normal','instruction':'手腕顺时针 5 圈，再逆时针 5 圈；完成顺时针后点击“切换逆时针”。','package':'TP-02'},
 {'id':6,'code':'SHOULDER','name':'肩部诱导代偿','group':'诱导样本','duration_s':None,'target_reps':0,'expected_label':'shoulder_compensation','instruction':'做腕运动时故意耸肩、上提或前倾／侧倾，记录明显肩部运动（候选 >5 cm）。结束后手动保存。','package':'TP-03'},
 {'id':7,'code':'ELBOW','name':'肘部诱导代偿','group':'诱导样本','duration_s':None,'target_reps':0,'expected_label':'elbow_compensation','instruction':'手腕尽量不动，故意大臂外展、手肘屈伸来晃动手腕。结束后手动保存。','package':'TP-03'},
 {'id':8,'code':'COMBINED','name':'肩肘协同代偿','group':'诱导样本','duration_s':None,'target_reps':0,'expected_label':'combined','instruction':'耸肩同时肘关节外翻，肩肘共同晃动。结束后手动保存。','package':'TP-03'},
]
BODY = ('r_shoulder','r_elbow','r_wrist')
HAND = ('hand_wrist','hand_index_mcp','hand_middle_mcp','hand_pinky_mcp')


def unit(v):
    n = np.linalg.norm(v)
    if n < 1e-6: raise ValueError('degenerate direction')
    return v/n


def hand_frame(p):
    forward = unit(p['hand_middle_mcp']-p['hand_wrist'])
    lateral = unit(p['hand_index_mcp']-p['hand_pinky_mcp'])
    normal = unit(np.cross(lateral,forward))
    if not .03 < np.linalg.norm(p['hand_middle_mcp']-p['hand_wrist']) < .18:
        raise ValueError('hand geometry')
    if np.linalg.norm(p['hand_wrist']-p['r_wrist']) > .08:
        raise ValueError('hand/pose wrist mismatch')
    return forward,normal


class SweepCounter:
    def __init__(self, threshold):
        self.threshold=threshold
        self.phase=0
        self.count=0
    def reset_partial(self): self.phase=0
    def update(self,value):
        if value is None:
            self.reset_partial(); return self.count
        if self.phase==0 and value>=self.threshold:self.phase=1
        elif self.phase==1 and value<=-self.threshold:self.phase=2
        elif self.phase==2 and value>=self.threshold:
            self.count+=1;self.phase=1
        return self.count


class CircleCounter:
    def __init__(self):
        self.previous=None;self.total=0.;self.count=0
    def reset_partial(self):self.previous=None;self.total=0.
    def update(self,fe,ru):
        if fe is None or ru is None or math.hypot(fe,ru)<6:
            self.reset_partial();return self.count
        phase=math.atan2(fe,ru)
        if self.previous is not None:
            delta=(phase-self.previous+math.pi)%(2*math.pi)-math.pi
            if abs(delta)>math.pi/2:
                self.reset_partial()
            else:
                self.total+=delta
                if abs(self.total)>=2*math.pi-.05:
                    self.count+=1;self.total-=math.copysign(2*math.pi,self.total)
        self.previous=phase
        return self.count


class ProtocolEngine:
    def __init__(self):
        self.task_id=1;self.participant_id='P001';self.training=False;self.paused=False
        self.samples=deque();self.baseline=None;self.hand_basis=None;self.calibration_id=None
        self.run_id=None;self.elapsed=0.;self.last_clock=None;self.finish_requested=False
        self.finish_reason=None;self.manual_count=0;self.direction='clockwise'
        self.sweep=SweepCounter(15);self.circle=CircleCounter();self.clockwise_count=0
        self.reps=0;self.flagged_reps=0;self.interrupted_reps=0;self.target=30.
        self.last_estimate=None;self.angle_filtered=None;self.last_measurement_time=None
        self.alert_since={};self.alerts={'shoulder':False,'elbow':False}

    @property
    def task(self):return TASKS[self.task_id-1]

    def select(self,task_id,participant_id=None):
        if self.training:raise ValueError('请先结束并保存当前试次，再切换训练')
        if task_id not in range(1,9):raise ValueError('训练编号应为 1～8')
        self.task_id=task_id
        if participant_id:self.participant_id=participant_id
        self.target=20. if task_id==4 else 30.
        self.recalibrate()
        self.reset_trial()

    def reset_trial(self):
        self.elapsed=0.;self.last_clock=None;self.finish_requested=False;self.finish_reason=None
        self.manual_count=0;self.direction='clockwise';self.clockwise_count=0
        self.manual_clockwise=0;self.manual_counterclockwise=0
        self.sweep=SweepCounter(10 if self.task_id==4 else 15);self.circle=CircleCounter()
        self.reps=0;self.run_id=None;self.paused=False

    def recalibrate(self):
        self.baseline=None;self.hand_basis=None;self.calibration_id=None;self.samples.clear()
        self.last_estimate=self.angle_filtered=None;self.last_measurement_time=None

    def start(self,run_id,now):
        if self.training:raise ValueError('当前已有进行中的试次')
        if self.task_id>=3 and self.baseline is None:raise ValueError('请先保持自然姿势完成上肢基线标定')
        self.reset_trial();self.run_id=run_id;self.training=True;self.last_clock=now

    def finish(self,reason='operator_finish'):
        self.training=False;self.paused=False;self.finish_reason=reason;self.finish_requested=False

    def pause(self,now):
        self.advance(now);self.paused=True
        self.sweep.reset_partial();self.circle.reset_partial()

    def resume(self,now):self.paused=False;self.last_clock=now

    def advance(self,now):
        if self.training and not self.paused and self.last_clock is not None:
            self.elapsed+=max(0.,now-self.last_clock)
        self.last_clock=now
        if self.training and self.task['duration_s'] and self.elapsed>=self.task['duration_s']:
            self.finish_requested=True;self.finish_reason='duration_complete'

    def switch_direction(self):
        if self.task_id!=5:raise ValueError('方向切换仅用于复合回转')
        if self.direction=='counterclockwise':raise ValueError('已经处于逆时针阶段')
        self.clockwise_count=self.circle.count;self.circle=CircleCounter();self.direction='counterclockwise'

    def estimate_angles(self,p):
        if self.hand_basis is None or not all(k in p for k in BODY+HAND):return None,None,'hand_or_basis_missing'
        try:
            hand,_=hand_frame(p);forward=unit(p['r_wrist']-p['r_elbow'])
            normal=unit(self.hand_basis['normal']-np.dot(self.hand_basis['normal'],forward)*forward)
            lateral=unit(np.cross(forward,normal))
            fe=math.degrees(math.atan2(np.dot(hand,normal),np.dot(hand,forward)))-self.hand_basis['fe_zero']
            ru=math.degrees(math.atan2(np.dot(hand,lateral),np.dot(hand,forward)))-self.hand_basis['ru_zero']
            if max(abs(fe),abs(ru))>85:return None,None,'angle_out_of_observable_range'
            return fe,ru,''
        except ValueError:return None,None,'hand_geometry_or_reference_invalid'

    def update(self,p,now):
        self.advance(now)
        out={'status':'ready','headline':'选择训练项目，按提示就位', 'detail':self.task['instruction'],
             'bend_deg':None,'theta_fe_est_deg':None,'theta_ru_est_deg':None,
             'theta_fe_deg':None,'theta_ru_deg':None,'angle_valid':False,'estimate_valid':False,
             'angle_invalid_reason':'anatomical_forearm_axial_reference_unobserved',
             'shoulder_mm':None,'shoulder_abs_mm':None,'elbow_deg':None,'elbow_mm':None,
             'shoulder_alert':False,'elbow_alert':False,'compensation_state':'unknown',
             'compensation_valid':False,'calibration_progress':0.,'calibrated':self.baseline is not None,
             'body_valid':all(k in p for k in BODY),'trial_phase':'paused' if self.paused else 'collecting' if self.training else 'ready'}
        valid=out['body_valid']
        if valid:
            try:
                if not .10<np.linalg.norm(p['r_elbow']-p['r_shoulder'])<.60 or not .10<np.linalg.norm(p['r_wrist']-p['r_elbow'])<.55:
                    valid=False;out['body_valid']=False
            except ValueError:valid=False;out['body_valid']=False
        if not self.training and self.baseline is None and valid:
            self.samples.append((now,{k:v.copy() for k,v in p.items()}))
            while self.samples and now-self.samples[0][0]>3.1:self.samples.popleft()
            data=np.array([np.concatenate([s[1][k] for k in BODY]) for s in self.samples])
            duration=now-self.samples[0][0]
            spread=np.percentile(np.linalg.norm((data-np.median(data,axis=0)).reshape(-1,3,3),axis=2),95)
            if spread>.018:self.samples.clear();duration=0.
            out.update(status='calibrating',headline='请保持自然坐姿，正在建立上肢基线',calibration_progress=min(1.,duration/3))
            if duration>=3 and len(data)>=15:
                names=set.intersection(*(set(s[1]) for s in self.samples))
                self.baseline={k:np.median([s[1][k] for s in self.samples],axis=0) for k in names}
                self.calibration_id=f'baseline-{now:.3f}'
                self.samples.clear();out.update(calibrated=True,calibration_progress=1.)
                self.try_hand_basis(p)
        elif not valid and not self.training:
            self.samples.clear();out.update(status='unknown',headline='请露出右肩、右肘、右腕后就位')
        if self.baseline is not None and self.hand_basis is None and not self.training:
            self.try_hand_basis(p)
        if valid and self.baseline is not None:
            b=self.baseline
            out['shoulder_abs_mm']=round(float(np.linalg.norm(p['r_shoulder']-b['r_shoulder'])*1000),2)
            out['elbow_mm']=round(float(np.linalg.norm(p['r_elbow']-b['r_elbow'])*1000),2)
            if 'l_shoulder' in p and 'l_shoulder' in b:
                out['shoulder_mm']=round(float((p['l_shoulder'][1]-p['r_shoulder'][1]-b['l_shoulder'][1]+b['r_shoulder'][1])*1000),2)
            current_elbow=180-angle(p['r_shoulder']-p['r_elbow'],p['r_wrist']-p['r_elbow'])
            base_elbow=180-angle(b['r_shoulder']-b['r_elbow'],b['r_wrist']-b['r_elbow'])
            out['elbow_deg']=round(abs(current_elbow-base_elbow),2)
            upper_delta=angle(p['r_elbow']-p['r_shoulder'],b['r_elbow']-b['r_shoulder'])
            out['upper_arm_swing_deg']=round(upper_delta,2)
            signal={'shoulder':max(out['shoulder_abs_mm']/50,upper_delta/15),
                    'elbow':out['elbow_deg']/15}
            # No wrist-motion gate: induced elbow samples explicitly keep the wrist still.
            for name,value in signal.items():
                if value>=1:
                    self.alert_since.setdefault(name,now)
                    if now-self.alert_since[name]>=.18:self.alerts[name]=True
                elif value<.7:self.alert_since.pop(name,None);self.alerts[name]=False
            out.update(shoulder_alert=self.alerts['shoulder'],elbow_alert=self.alerts['elbow'],
                       compensation_valid=True,compensation_state='combined' if all(self.alerts.values()) else
                       'shoulder_compensation' if self.alerts['shoulder'] else 'elbow_compensation' if self.alerts['elbow'] else 'normal')
        else:
            self.alert_since.clear();self.alerts={'shoulder':False,'elbow':False}
        fe,ru,reason=self.estimate_angles(p) if valid else (None,None,'body_geometry_or_keypoints_invalid')
        gap=self.last_measurement_time is not None and now-self.last_measurement_time>.25
        if gap:self.sweep.reset_partial();self.circle.reset_partial();self.last_estimate=self.angle_filtered=None
        self.last_measurement_time=now
        if fe is not None:
            candidate=np.array([fe,ru])
            if self.last_estimate is not None and not gap and max(abs(candidate-self.last_estimate))>25:
                fe=ru=None;reason='angle_jump'
            self.last_estimate=candidate
            if fe is not None:
                self.angle_filtered=candidate if self.angle_filtered is None or gap else .55*candidate+.45*self.angle_filtered
                fe,ru=map(float,self.angle_filtered)
        else:self.last_estimate=self.angle_filtered=None
        out.update(theta_fe_est_deg=round(fe,2) if fe is not None else None,
                   theta_ru_est_deg=round(ru,2) if ru is not None else None,
                   estimate_valid=fe is not None,estimate_invalid_reason=reason)
        out['bend_deg']=out['theta_ru_est_deg'] if self.task_id==4 else out['theta_fe_est_deg']
        if self.training:
            if not self.paused:
                if self.task_id in (3,4):self.reps=self.sweep.update(fe if self.task_id==3 else ru)
                elif self.task_id==5:
                    if gap:self.circle.reset_partial()
                    self.circle.update(fe,ru);self.reps=self.clockwise_count+self.circle.count
            out.update(status='collecting',headline=self.task['name']+'：采集中',detail=self.task['instruction'])
            if self.task_id in (1,2):
                out['headline']=f'{self.task["name"]}：还需保持 {max(0,30-self.elapsed):.1f} 秒'
                out['detail']='保持当前姿势；有效率与遮挡会如实记录，不用补点掩盖失效。'
            elif self.task_id in (3,4):
                out['headline']=f'{self.task["name"]}：自动估计 {self.reps}/10 次，人工确认 {self.manual_count}/10 次'
                if any(self.alerts.values()):out['detail']='检测到肩肘额外参与，请减小幅度并保持前臂支撑。'
            elif self.task_id==5:
                direction='顺时针' if self.direction=='clockwise' else '逆时针'
                out['headline']=f'{direction}采集：本方向自动估计 {self.circle.count}/5 圈'
            else:
                out['headline']='诱导样本采集：'+self.task['name']
                label={'normal':'未见明显额外参与','shoulder_compensation':'肩部额外参与','elbow_compensation':'肘部额外参与','combined':'肩肘共同参与','unknown':'暂不可判读'}[out['compensation_state']]
                out['detail']='当前观测：'+label+'；完成后点击结束并保存。'
            if not valid:out.update(status='unknown',headline='仍在采集，但上肢观测不足',detail='请保持或调整至可见位置；失效数据与时长会保留。')
            elif self.task_id in (3,4,5) and fe is None:
                out.update(status='unknown',headline='正在采集，手部角度暂不可判读',detail='请露出右手背及指根；可人工确认次数，缺失角度不会填成零。')
            if self.paused:out.update(status='paused',headline='本项已暂停，点击继续采集',detail='暂停时长不计入 30 秒任务；暂停片段仍明确标记。')
        elif self.baseline is not None:
            out.update(status='ready',headline='已就位，点击“开始本项并采集”')
        out.update(task_id=self.task_id,task_code=self.task['code'],task_name=self.task['name'],
                   task_catalog=TASKS,expected_label=self.task['expected_label'],
                   training=self.training,paused=self.paused,elapsed_s=round(self.elapsed,2),
                   target_duration_s=self.task['duration_s'],target_reps=self.task['target_reps'],
                   reps=self.reps,manual_count=self.manual_count,direction=self.direction,
                   clockwise_est_count=self.clockwise_count if self.direction=='counterclockwise' else self.circle.count,
                   counterclockwise_est_count=self.circle.count if self.direction=='counterclockwise' else 0,
                   target_deg=self.target,flagged_reps=0,interrupted_reps=0,
                   participant_id=self.participant_id,run_id=self.run_id,calibration_id=self.calibration_id,
                   angle_kind='constrained_neutral_hand_plane_estimate',validation='engineering_only',
                   angle_reference='fixed_neutral_palm_normal_forearm_axial_rotation_unobserved')
        return out

    def try_hand_basis(self,p):
        if not all(k in p for k in BODY+HAND):return
        try:
            hand,normal=hand_frame(p);forward=unit(p['r_wrist']-p['r_elbow'])
            normal=unit(normal-np.dot(normal,forward)*forward)
            lateral=unit(np.cross(forward,normal))
            self.hand_basis={'normal':normal,
                'fe_zero':math.degrees(math.atan2(np.dot(hand,normal),np.dot(hand,forward))),
                'ru_zero':math.degrees(math.atan2(np.dot(hand,lateral),np.dot(hand,forward)))}
        except ValueError:pass
