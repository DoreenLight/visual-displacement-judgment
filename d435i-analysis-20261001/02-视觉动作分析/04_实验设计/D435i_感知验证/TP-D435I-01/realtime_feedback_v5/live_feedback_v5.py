"""Eight selectable D435i collection protocols with local pose + hand inference."""
from pathlib import Path
import base64
from collections import deque
import argparse
import json
import logging
import queue
import re
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse,parse_qs
from uuid import uuid4
import sys
from importlib.metadata import version
import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
ASSETS=Path(__file__).resolve().parent.parent/'attempt_01'
sys.path.insert(0,str(ASSETS))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from live_feedback_v2 import State as BaseState,render,placeholder,ROOT,JOINTS
HERE=Path(__file__).resolve().parent
from curves import Curves
from wrist_depth import hand_depth
from trial_management import TrialManagement
from training_protocol_v5 import ProtocolEngine,TASKS,HAND
from trial_recorder_v5 import TrialRecorder
from trial_export_v3 import digest

LOG=logging.getLogger('eight_tasks')


class State(BaseState):
    def __init__(self):
        super().__init__();self.finalizing=False;self.completed_runs=[];self.command_error=None
        self.trials=TrialManagement(HERE);self.curves=Curves();self.curve_history=deque(maxlen=2400)
        self.engine=ProtocolEngine();self.engine.reset_trial()
        self.telemetry.update(self.engine.update({},time.monotonic()))
    def publish(self,image,telemetry):
        telemetry['live_version']='feedback/v5'
        points={k:np.array(v) for k,v in telemetry.get('joints_3d_m',{}).items()}
        if 'curve_sample' not in telemetry: telemetry['curve_sample']=self.curves.update(telemetry,points,self.engine.baseline)
        ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,82])
        if ok:
            with self.condition:
                self.sequence+=1
                telemetry['display_sequence']=self.sequence
                self.jpeg=encoded.tobytes();self.telemetry=telemetry
                sample=dict(telemetry['curve_sample']);sample['sequence']=self.sequence
                self.curve_history.append(sample)
                self.condition.notify_all()
    def packet(self,after):
        data=self.snapshot()
        with self.lock:
            # Take metadata, image and history together; the displayed frame and curve share one sequence.
            data.update(self.telemetry)
            now=time.monotonic();data['result_age_ms']=round((now-data['output_mono_s'])*1000)
            history=[dict(x) for x in self.curve_history if x['sequence']>after and now-x['capture_mono_s']<=60]
            return {'telemetry':data,'sequence':self.sequence,'jpeg_base64':base64.b64encode(self.jpeg).decode() if self.jpeg else None,
                    'history':history,'server_mono_s':now}
    def snapshot(self):
        data=super().snapshot()
        with self.lock:
            data.update(finalizing=self.finalizing,command_error=self.command_error,deleted_runs=self.trials.deleted(),
                        completed_runs=[{k:r.get(k) for k in ('run_id','participant_id','task_id','task_name','captured_frames',
                         'active_duration_s','manual_count','static_duration_verdict','finish_reason','raw_directory','quality','formal_verdicts')} for r in self.completed_runs])
        if data['result_age_ms']>1000:
            data.update(theta_fe_est_deg=None,theta_ru_est_deg=None,estimate_valid=False,
                        compensation_valid=False,compensation_state='unknown')
        return data


def measure_all(pose,hand,frame):
    points,pixels,records={},{},{}
    h,w=frame['depth'].shape
    def sample(name,lm,score,visibility,reason=''):
        m={'point_m':None,'measurement_valid':False,'depth_valid':False,'keypoint_score':score,
           'visibility':visibility,'u_px':None,'v_px':None,'invalid_reason':reason or 'missing_keypoint'}
        records[name]=m
        if lm is None:return
        m['u_px']=round(lm.x*w,2);m['v_px']=round(lm.y*h,2)
        if score is not None and score<.5:m['invalid_reason']='keypoint_confidence_low';return
        if not (0<=lm.x<1 and 0<=lm.y<1):m['invalid_reason']='outside_image';return
        u,v=int(lm.x*w),int(lm.y*h)
        if name.startswith('hand_'):
            z,reason,details=hand_depth(frame['depth'],u,v,frame['scale']);m.update(details)
            if z is None:m['invalid_reason']=reason;return
            p=np.array(rs.rs2_deproject_pixel_to_point(frame['intr'],[float(u),float(v)],z))
            if not np.all(np.isfinite(p)):m['invalid_reason']='nonfinite_depth_point';return
            points[name]=p;pixels[name]=(w-1-u,v)
            m.update(point_m=p.tolist(),measurement_valid=True,depth_valid=True,invalid_reason='');return
        patch=frame['depth'][max(0,v-3):min(h,v+4),max(0,u-3):min(w,u+4)]
        values=patch[patch>0].astype(float)*frame['scale']
        if len(values)<patch.size*.65:m['invalid_reason']='depth_coverage_low';return
        z=float(np.median(values))
        if not .35<z<2.5:m['invalid_reason']='depth_range';return
        if np.percentile(values,90)-np.percentile(values,10)>.07:m['invalid_reason']='depth_edge_spread';return
        p=np.array(rs.rs2_deproject_pixel_to_point(frame['intr'],[float(u),float(v)],z))
        if not np.all(np.isfinite(p)):m['invalid_reason']='nonfinite_depth_point';return
        points[name]=p;pixels[name]=(w-1-u,v)
        m.update(point_m=p.tolist(),measurement_valid=True,depth_valid=True,invalid_reason='')
    landmarks=pose.pose_landmarks[0] if pose.pose_landmarks else None
    for name,index in JOINTS.items():
        lm=landmarks[index] if landmarks else None
        vis=float(lm.visibility or 0.) if lm else None
        score=min(vis,float(lm.presence or 0.)) if lm else None
        sample(name,lm,score,vis)
    match=None
    if landmarks and hand.hand_landmarks:
        wrist=landmarks[16]
        distances=[np.hypot(l[0].x-wrist.x,l[0].y-wrist.y) for l in hand.hand_landmarks]
        index=int(np.argmin(distances))
        if distances[index]<.12 and (wrist.visibility or 0.)>=.5:match=hand.hand_landmarks[index]
    # Hand API has no per-landmark confidence. Handedness score is NOT reused as localization confidence.
    for name,index in zip(HAND,(0,5,9,17)):
        sample(name,match[index] if match else None,None,None,'right_hand_association_missing' if match is None else '')
    return points,pixels,records


def finish_info(e,reason):
    return {'finish_reason':reason,'active_duration_s':round(e.elapsed,3),'manual_count':e.manual_count,
            'ended_host_mono_s':time.monotonic(),
            'manual_clockwise':e.manual_clockwise,'manual_counterclockwise':e.manual_counterclockwise,
            'estimated_reps':e.reps,'clockwise_est_count':e.clockwise_count if e.direction=='counterclockwise' else e.circle.count,
            'counterclockwise_est_count':e.circle.count if e.direction=='counterclockwise' else 0}


def process(state,recorder):
    e=state.engine;last_received=None;last_timestamp=0;previous_done=None;fps=0.
    pose_options=vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_buffer=(ASSETS/'pose_landmarker_lite.task').read_bytes()),
        running_mode=vision.RunningMode.VIDEO,min_pose_detection_confidence=.5,min_pose_presence_confidence=.5,min_tracking_confidence=.5)
    hand_options=vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_buffer=(ASSETS/'hand_landmarker.task').read_bytes()),
        running_mode=vision.RunningMode.VIDEO,num_hands=2,min_hand_detection_confidence=.5,min_hand_presence_confidence=.5,min_tracking_confidence=.5)
    hashes={name:digest(ASSETS/name) for name in ('pose_landmarker_lite.task','hand_landmarker.task','training_protocol_v3.py','live_feedback_v3.py','trial_recorder_v3.py','trial_export_v3.py')}
    task_hashes={p.name:digest(p) for p in (ROOT/'04_实验设计'/'D435i_四问任务包').glob('*.md')}
    last_device=None;last_rgb_time=None
    def finish(reason):
        e.advance(time.monotonic());info=finish_info(e,reason);e.finish(reason);recorder.finish(info);e.recalibrate()
    try:
        with vision.PoseLandmarker.create_from_options(pose_options) as detector,vision.HandLandmarker.create_from_options(hand_options) as hand_detector:
            while not state.stop.is_set():
                while not state.commands.empty():
                    action,payload=state.commands.get_nowait()
                    try:
                        now=time.monotonic()
                        if action in ('delete_trial','restore_trial'):
                            if e.training or state.recording or state.finalizing:raise ValueError('请先结束并保存当前试次，再删除或恢复数据')
                            runs=state.trials.apply(payload.get('run_id'),restore=action=='restore_trial')
                            with state.lock:state.completed_runs=runs
                        elif action=='set_count_threshold':
                            if e.training or state.finalizing:raise ValueError('请在开始前调整计数门槛')
                            value=float(payload['degrees'])
                            if not 5<=value<=30:raise ValueError('参考计数单侧门槛范围5到30度')
                            if e.task_id not in (3,4):raise ValueError('仅FE/RU可调整往返门槛')
                            e.count_thresholds[e.task_id]=value;e.reset_trial()
                        elif action=='select_task':
                            if state.finalizing:raise ValueError('请等待保存完成')
                            e.select(payload['task_id'],payload.get('participant_id'))
                        elif action=='calibrate':
                            if e.training:raise ValueError('请先结束当前试次')
                            e.recalibrate()
                        elif action in ('train','record'):
                            if state.finalizing:raise ValueError('请等待保存完成')
                            with state.lock:frame=state.latest_frame;camera=dict(state.camera)
                            if frame is None or now-frame['received']>.4:raise ValueError('摄像头未就绪')
                            run_id=datetime.now().strftime(f'feedback5_{e.task["code"]}_%Y%m%d_%H%M%S_')+uuid4().hex[:8]
                            e.start(run_id,now)
                            meta={'schema_version':'wrist-rehab/v1','protocol_version':'eight-task/feedback-v5','run_id':run_id,
                                'attempt_id':'realtime_feedback_v5','session_id':datetime.now().strftime('%Y%m%d')+'_'+e.participant_id,
                                'trial':1+sum(r['participant_id']==e.participant_id and r['task_id']==e.task_id for r in state.completed_runs),
                                'task_package_ids':['TP-D435I-01','TP-D435I-02','TP-D435I-03','TP-D435I-04'],
                                'participant_id':e.participant_id,'task_id':e.task_id,'task_name':e.task['name'],
                                'expected_label':e.task['expected_label'],'instruction':e.task['instruction'],'instruction_is_ground_truth':False,
                                'training_side':'right','data_source':'real_rgbd','started_at':datetime.now().isoformat(),
                                'started_host_mono_s':now,
                                'camera':camera,'reference_system':None,'accuracy_verdict':'BLOCKED',
                                'exposure':camera.get('exposure'),'camera_placement':None,'rgb_depth_extrinsics':None,
                                'depth_alignment':'SDK aligned to color optical frame','sync_method':None,'device_host_clock_mapping':None,
                                'executed_command':[sys.executable]+sys.argv,'command_exit_code':None,
                                'unavailable_metadata_note':'Placement not supplied; independent reference and synchronization absent. Process is ongoing; exit code null. Aligned depth samples retained, unaligned depth not retained.',
                                'calibration_id':e.calibration_id,'baseline_m':{k:p.tolist() for k,p in (e.baseline or {}).items()},
                                'hand_basis':{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in (e.hand_basis or {}).items()},
                                'candidate_angle_assumption':'fixed_neutral_palm_normal; forearm_axial_rotation_unobserved',
                                'coordinate_source':'aligned_depth_surface_sample_not_anatomical_joint_center','keypoint_filter':'none',
                                'angle_filter':'causal EMA alpha 0.55','coord_frame':'camera_color_optical','model_and_code_sha256':{**hashes,'live_feedback_v5.py':digest(HERE/'live_feedback_v5.py'),'curves.py':digest(HERE/'curves.py'),'trial_recorder_v5.py':digest(HERE/'trial_recorder_v5.py'),'trial_export_v5.py':digest(HERE/'trial_export_v5.py'),'training_protocol_v5.py':digest(HERE/'training_protocol_v5.py'),'wrist_depth.py':digest(HERE/'wrist_depth.py'),'trial_management.py':digest(HERE/'trial_management.py')},
                                'task_document_sha256':task_hashes,'upstream_validated_results_sha256':None,
                                'software':{'mediapipe':mp.__version__,'numpy':np.__version__,'opencv':cv2.__version__,'pyrealsense2':version('pyrealsense2')},
                                'target_duration_s':e.task['duration_s'],'target_reps':e.task['target_reps'],
                                'thresholds':{'wrist_count_one_side_deg':e.sweep.threshold,'missing_count_grace_s':.25,'hand_depth_sampling':'center_connected_current_frame_3_or_5px','hand_neutral_duration_s':1.,'depth_patch':7,'depth_coverage':.65,'depth_range_m':[.35,2.5],
                                   'depth_spread_max_m':.07,'shoulder_displacement_mm':50,'upper_arm_swing_deg':15,
                                   'elbow_angle_change_deg':15,'alert_persistence_s':.18,'alert_release_ratio':.7}}
                            try:recorder.begin(meta)
                            except Exception:e.finish('start_failed');raise
                        elif action=='pause' and e.training:e.pause(now);recorder.event('pause',{})
                        elif action=='resume' and e.training:e.resume(now);recorder.event('resume',{})
                        elif action=='discard_trial':
                            if not e.training:raise ValueError('当前没有进行中的试次')
                            finish('operator_discarded')
                        elif action in ('finish','stop_record') and e.training:finish('operator_finish')
                        elif action=='mark_rep':
                            if not e.training or e.paused or e.task_id not in (3,4,5):raise ValueError('仅在腕动作采集中确认次数')
                            e.manual_count+=1
                            if e.task_id==5:
                                if e.direction=='clockwise':e.manual_clockwise+=1
                                else:e.manual_counterclockwise+=1
                            recorder.event('manual_rep',{'count':e.manual_count,'direction':e.direction})
                        elif action=='switch_direction':
                            if not e.training or e.paused:raise ValueError('请在回转采集中切换方向')
                            e.switch_direction();recorder.event('direction',{'direction':e.direction})
                        with state.lock:state.command_error=None
                    except Exception as error:
                        with state.lock:state.command_error=str(error)
                    finally:state.commands.task_done()
                now=time.monotonic();e.advance(now)
                if e.training and not state.recording and not state.finalizing:
                    e.finish('storage_error')
                if e.finish_requested:finish(e.finish_reason)
                with state.lock:frame=state.latest_frame
                if frame is None or now-frame['received']>.4:
                    out=e.update({},now);out.update(output_mono_s=now,headline='等待及时的摄像头画面；采集时失效会保留')
                    state.publish(placeholder(out['headline']),out);state.stop.wait(.05);continue
                if frame['received']==last_received:state.stop.wait(.003);continue
                device=state.camera.get('serial')
                if last_device is not None and (device!=last_device or frame['rgb_s']<last_rgb_time):
                    if e.training:finish('camera_reconnected')
                    e.recalibrate()
                last_device,last_rgb_time=device,frame['rgb_s']
                last_received=frame['received'];began=time.monotonic()
                timestamp=max(last_timestamp+1,int(frame['received']*1000));last_timestamp=timestamp
                image=mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frame['color'],cv2.COLOR_BGR2RGB))
                pose=detector.detect_for_video(image,timestamp);hand=hand_detector.detect_for_video(image,timestamp)
                points,pixels,measurements=measure_all(pose,hand,frame)
                done=time.monotonic()
                if done-frame['received']>.4:
                    points={}
                    pixels={}
                    for m in measurements.values():m.update(point_m=None,measurement_valid=False,invalid_reason='processing_result_stale')
                out=e.update(points,done)
                if previous_done is not None:fps=.8*fps+.2/max(.001,done-previous_done)
                previous_done=done
                out.update(frame_id=frame['id'],output_mono_s=done,received_mono_s=frame['received'],
                           inference_ms=round((done-began)*1000,1),processing_latency_ms=round((done-frame['received'])*1000,1),fps=round(fps,1),
                           measurements=measurements,joints_3d_m={k:p.tolist() for k,p in points.items()},
                           manual_clockwise=e.manual_clockwise,manual_counterclockwise=e.manual_counterclockwise)
                out['curve_sample']=state.curves.update(out,points,e.baseline)
                recorder.submit_analysis(out);state.publish(render(frame,pixels,out),out)
                if e.finish_requested:finish(e.finish_reason)
    except Exception as error:
        LOG.exception('Processing stopped')
        state.publish(placeholder('本地模型中断'),{'status':'error','headline':'本地模型分析中断','detail':str(error),'output_mono_s':time.monotonic()})
    finally:
        if e.training:finish('service_shutdown' if state.stop.is_set() else 'model_error')


def handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,code,data,mime='application/json; charset=utf-8'):
            body=data if isinstance(data,bytes) else json.dumps(data,ensure_ascii=False,allow_nan=False).encode('utf-8')
            self.send_response(code);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
        def do_GET(self):
            try:
                url=urlparse(self.path)
                if url.path=='/':self.reply(200,(HERE/'live_feedback_v5.html').read_bytes(),'text/html; charset=utf-8')
                elif url.path=='/clock':self.reply(200,{'server_mono_s':time.monotonic()})
                elif url.path=='/display_packet':
                    after=int(parse_qs(url.query).get('after',['0'])[0]);self.reply(200,state.packet(after))
                elif url.path=='/curves.csv':
                    import csv,io
                    with state.lock: rows=list(state.curve_history)
                    text=io.StringIO();writer=csv.writer(text)
                    writer.writerow(['sequence','frame_id','host_receive_mono_s','run_id','shoulder_proxy_deg','elbow_flex_est_deg','wrist_fe_candidate_deg','wrist_ru_candidate_deg','shoulder_raw_deg','elbow_raw_deg','wrist_fe_raw_deg','wrist_ru_raw_deg'])
                    for x in rows:writer.writerow([x['sequence'],x['frame_id'],x['capture_mono_s'],x['run_id']]+[x['values_deg'][k] for k in ('shoulder','elbow','wrist_fe','wrist_ru')]+[x['raw_values_deg'][k] for k in ('shoulder','elbow','wrist_fe','wrist_ru')])
                    self.reply(200,text.getvalue().encode('utf-8-sig'),'text/csv; charset=utf-8')
                elif url.path=='/telemetry':self.reply(200,state.snapshot())
                elif url.path=='/results':
                    with state.lock:results=list(state.completed_runs)
                    self.reply(200,results)
                elif url.path=='/download':
                    query=parse_qs(url.query);run=query.get('run',[''])[0];package=query.get('package',[''])[0]
                    filename=query.get('file',[''])[0]
                    with state.lock:known={r['run_id'] for r in state.completed_runs}
                    allowed={'1':('keypoints.csv','summary.json','result.md','files.sha256'),
                             '2':('wrist_kinematics.csv','summary.json','result.md','files.sha256'),
                             '3':('compensation.csv','summary.json','result.md','files.sha256'),
                             '4':('quality.csv','summary.json','result.md','files.sha256')}
                    if run not in known or package not in allowed or filename not in allowed[package]:self.reply(404,{'error':'unknown result'});return
                    path=ROOT/'04_实验设计'/'D435i_感知验证'/f'TP-D435I-0{package}'/('realtime_feedback_v5' if run.startswith('feedback5_') else 'realtime_curves_v4' if run.startswith('curves4_') else 'eight_task_v3')/run/filename
                    self.reply(200,path.read_bytes(),'text/plain; charset=utf-8')
                elif url.path=='/snapshot.jpg':
                    with state.lock:jpeg=state.jpeg
                    self.reply(200,jpeg,'image/jpeg') if jpeg else self.reply(503,{'error':'no frame'})
                elif url.path=='/video_feed':
                    self.send_response(200);self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame');self.end_headers();previous=-1
                    while not state.stop.is_set():
                        with state.condition:
                            state.condition.wait_for(lambda:state.sequence!=previous or state.stop.is_set(),timeout=1)
                            seq,jpeg=state.sequence,state.jpeg
                        if jpeg and seq!=previous:
                            self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpeg)).encode()+b'\r\n\r\n'+jpeg+b'\r\n');previous=seq
                else:self.reply(404,{'error':'not found'})
            except (BrokenPipeError,ConnectionResetError):pass
        def do_POST(self):
            origin=self.headers.get('Origin')
            if origin and origin!='http://'+self.headers.get('Host',''):self.reply(403,{'error':'origin rejected'});return
            action=self.path.lstrip('/')
            if action not in ('discard_trial','delete_trial','restore_trial','set_count_threshold','select_task','train','record','pause','resume','finish','stop_record','calibrate','mark_rep','switch_direction','stop'):
                self.reply(404,{'error':'unknown command'});return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<=size<=2048:raise ValueError('body too large')
                payload=json.loads(self.rfile.read(size) or b'{}')
                if not isinstance(payload,dict):raise ValueError('expected object')
                if action=='select_task':
                    if type(payload.get('task_id')) is not int or payload['task_id'] not in range(1,9):raise ValueError('训练编号应为 1～8')
                    if not re.fullmatch(r'[A-Za-z0-9_-]{1,32}',payload.get('participant_id','P001')):raise ValueError('参与者代号请用字母、数字、下划线或短横线')
                if action=='stop':self.reply(200,{'accepted':action});state.stop.set();return
                state.commands.put_nowait((action,payload));self.reply(200,{'accepted':action})
            except (ValueError,TypeError,KeyError,queue.Full) as error:self.reply(400,{'error':str(error)})
    return Handler


def capture(state):
    sequence = 0
    while not state.stop.is_set():
        pipe = rs.pipeline()
        started = False
        try:
            cfg = rs.config()
            cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
            cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
            profile = pipe.start(cfg)
            started = True
            device = profile.get_device()
            scale = device.first_depth_sensor().get_depth_scale()
            exposures = {}
            for sensor in device.query_sensors():
                if sensor.supports(rs.option.exposure):
                    exposures[sensor.get_info(rs.camera_info.name)] = {
                        'exposure_value': sensor.get_option(rs.option.exposure),
                        'auto_exposure': sensor.get_option(rs.option.enable_auto_exposure) if sensor.supports(rs.option.enable_auto_exposure) else None}
            align = rs.align(rs.stream.color)
            with state.lock:
                state.camera = {'name': device.get_info(rs.camera_info.name),
                                'serial': device.get_info(rs.camera_info.serial_number),
                                'firmware': device.get_info(rs.camera_info.firmware_version), 'depth_scale_m': scale,
                                'rgb_mode': '848x480 bgr8 30', 'depth_mode': '848x480 z16 30','exposure':exposures}
            while not state.stop.is_set():
                native_frames = pipe.wait_for_frames(2000)
                received_mono = time.monotonic()
                frames = align.process(native_frames)
                color, depth = frames.get_color_frame(), frames.get_depth_frame()
                if not color or not depth:
                    continue
                sequence += 1
                intr = depth.profile.as_video_stream_profile().get_intrinsics()
                frame = {'id': sequence, 'received': received_mono,
                         'color': np.asanyarray(color.get_data()).copy(),
                         'depth': np.asanyarray(depth.get_data()).copy(), 'intr': intr, 'scale': scale,
                         'rgb_s': color.get_timestamp() / 1000, 'depth_s': depth.get_timestamp() / 1000,
                         'rgb_clock': str(color.get_frame_timestamp_domain()),
                         'depth_clock': str(depth.get_frame_timestamp_domain())}
                with state.lock:
                    state.latest_frame = frame
                    state.camera['intrinsics'] = {'width':intr.width,'height':intr.height,'fx': intr.fx, 'fy': intr.fy, 'ppx': intr.ppx, 'ppy': intr.ppy,
                                                   'model': str(intr.model), 'coeffs': list(intr.coeffs)}
                if hasattr(state, 'record_sink'):
                    state.record_sink.submit_raw(frame)
        except Exception as error:
            LOG.exception('Camera connection failed')
            with state.lock:
                state.latest_frame = None
            state.publish(placeholder('等待 D435i 连接'), {'status': 'error', 'headline': '摄像头未就绪，正在重试',
                          'detail': str(error), 'output_mono_s': time.monotonic()})
        finally:
            if started:
                pipe.stop()
        state.stop.wait(2)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8090);args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    state=State();server=ThreadingHTTPServer(('127.0.0.1',args.port),handler(state));server.daemon_threads=True
    recorder=TrialRecorder(state,ROOT,HERE);state.record_sink=recorder;recorder.start()
    workers=[threading.Thread(target=capture,args=(state,),daemon=True),threading.Thread(target=process,args=(state,recorder),daemon=True)]
    for worker in workers:worker.start()
    threading.Thread(target=server.serve_forever,daemon=True).start();LOG.info('Eight tasks: http://localhost:%s',args.port)
    try:
        while not state.stop.wait(.25):pass
    except KeyboardInterrupt:state.stop.set()
    finally:
        with state.condition:state.condition.notify_all()
        for worker in workers:worker.join(timeout=10)
        with state.lock:state.recording=False
        recorder.items.put(('quit',));recorder.join()
        server.shutdown();server.server_close();LOG.info('Service stopped; queued files finalized')


if __name__=='__main__':main()
