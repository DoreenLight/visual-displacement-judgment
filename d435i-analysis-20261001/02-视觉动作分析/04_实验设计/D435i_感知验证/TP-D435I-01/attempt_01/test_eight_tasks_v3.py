import csv
import json
import math
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from training_protocol_v3 import ProtocolEngine,TASKS,SweepCounter,CircleCounter
from trial_export_v3 import KEYPOINT_COLUMNS,export_trial,exploratory_smoothness,max_gap
from trial_recorder_v3 import TrialRecorder
from live_feedback_v3 import State


def body():
    return {k:np.array(v,dtype=float) for k,v in {'r_shoulder':[0,0,1], 'r_elbow':[0,.25,1],
      'r_wrist':[.25,.25,1],'l_shoulder':[-.3,0,1],'hand_wrist':[.25,.25,1],
      'hand_middle_mcp':[.33,.25,1],'hand_index_mcp':[.32,.225,1],'hand_pinky_mcp':[.32,.275,1]}.items()}


def calibrated(task):
    e=ProtocolEngine();e.select(task)
    for t in np.arange(0,3.2,.05):e.update(body(),float(t))
    assert e.baseline is not None
    return e


class ProtocolTests(unittest.TestCase):
    def test_catalog_and_static_missing_observations(self):
        self.assertEqual(len(TASKS),8)
        for task in (1,2):
            e=ProtocolEngine();e.select(task);e.start('synthetic',10)
            e.update({},39.9);self.assertFalse(e.finish_requested)
            out=e.update({},40.1);self.assertTrue(e.finish_requested)
            self.assertFalse(out['body_valid']);self.assertIsNone(out['theta_fe_deg'])

    def test_pause_excluded_and_switch_rejected_during_trial(self):
        e=calibrated(1);e.start('synthetic',10);e.pause(20);e.update({},40);e.resume(40);e.update({},59)
        self.assertAlmostEqual(e.elapsed,29);self.assertFalse(e.finish_requested)
        with self.assertRaises(ValueError):e.select(2)

    def test_all_eight_start_finish(self):
        for task in range(1,9):
            e=calibrated(task);e.start('synthetic',4);out=e.update(body(),4.05)
            self.assertEqual(out['task_id'],task);self.assertTrue(out['training']);e.finish()
            self.assertFalse(e.training)

    def test_induced_shoulder_without_wrist_motion(self):
        e=calibrated(6);e.start('synthetic',4);p=body();p['r_shoulder'][1]-=.065
        for t in np.arange(4,4.5,.05):out=e.update(p,float(t))
        self.assertTrue(out['shoulder_alert']);self.assertTrue(out['training'])

    def test_induced_elbow_without_wrist_angle_motion(self):
        e=calibrated(7);e.start('synthetic',4);p=body();p['r_elbow'][0]+=.13
        for t in np.arange(4,4.5,.05):out=e.update(p,float(t))
        self.assertTrue(out['elbow_alert']);self.assertTrue(out['training'])

    def test_separate_estimates_and_invalid_formal_angles(self):
        e=calibrated(3);p=body();forward=np.array([math.cos(.4),0,math.sin(.4)])
        p['hand_middle_mcp']=p['hand_wrist']+.08*forward
        fe,ru,reason=e.estimate_angles(p)
        self.assertGreater(abs(fe),20);self.assertAlmostEqual(ru,0)
        out=e.update(p,4);self.assertIsNone(out['theta_fe_deg']);self.assertFalse(out['angle_valid'])
        p.pop('hand_middle_mcp');out=e.update(p,4.1);self.assertFalse(out['estimate_valid'])

    def test_counters_interrupt_and_direction(self):
        c=SweepCounter(15)
        for q in (20,-20,None,20):c.update(q)
        self.assertEqual(c.count,0)
        for q in (-20,20):c.update(q)
        self.assertEqual(c.count,1)
        c=CircleCounter()
        for q in np.linspace(0,2*math.pi,90):c.update(20*math.sin(q),20*math.cos(q))
        self.assertEqual(c.count,1)
        e=calibrated(5);e.start('synthetic',4);e.circle.count=5;e.switch_direction()
        self.assertEqual(e.clockwise_count,5)
        with self.assertRaises(ValueError):e.switch_direction()

    def test_static_does_not_receive_smoothness_reward(self):
        self.assertEqual(exploratory_smoothness([],1)['status'],'not_applicable')
        rows=[{'trial_phase':'collecting','estimate_valid':True,'theta_fe_est_deg':0,'received_mono_s':i/30} for i in range(120)]
        self.assertEqual(exploratory_smoothness(rows,3)['status'],'static_or_gapped')

    def test_camera_disconnect_at_end_is_not_hidden(self):
        self.assertAlmostEqual(max_gap([10,10.03,10.06],[True]*3,10,40),29.94)
        self.assertEqual(max_gap([],[],10,40),30)

    def test_recorder_and_four_exports_keep_unanalysed_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);here=root/'code';here.mkdir();state=State();rec=TrialRecorder(state,root,here);rec.start()
            rec.begin({'run_id':'synthetic_fixture','task_id':1,'task_name':'fixture','participant_id':'TEST',
                'expected_label':'not_applicable','data_source':'synthetic_test_not_experiment'})
            for i in range(4):
                rec.submit_raw({'id':i,'received':i/30,'color':np.zeros((8,8,3),np.uint8),'depth':np.ones((8,8),np.uint16),
                   'rgb_s':i/30,'depth_s':i/30,'rgb_clock':'test','depth_clock':'test'})
            p=body();measurements={k:{'point_m':v.tolist(),'measurement_valid':True,'depth_valid':True,'invalid_reason':''} for k,v in p.items()}
            rec.submit_analysis({'frame_id':0,'body_valid':True,'measurements':measurements,'trial_phase':'collecting',
                                 'received_mono_s':0,'compensation_valid':True,'compensation_state':'normal'})
            rec.finish({'finish_reason':'test_only','active_duration_s':.1,'manual_count':0,'estimated_reps':0})
            rec.items.join();rec.items.put(('quit',));rec.join(10)
            self.assertFalse(rec.is_alive());self.assertIsNone(state.record_error)
            result=state.completed_runs[0];self.assertEqual(result['captured_frames'],4)
            self.assertEqual(result['keypoint_statistics']['wrist']['valid_rate'],.25)
            self.assertEqual(len(result['output_directories']),4)
            kp=Path(result['output_directories'][0])/'keypoints.csv'
            with kp.open(encoding='utf-8-sig') as handle:
                reader=csv.DictReader(handle);self.assertEqual(reader.fieldnames,KEYPOINT_COLUMNS);rows=list(reader)
            missing=next(r for r in rows if r['frame_id']=='1' and r['joint_id']=='wrist')
            self.assertEqual(missing['x_raw_m'],'nan');self.assertEqual(missing['imputed'],'False')
            with (Path(result['output_directories'][1])/'wrist_kinematics.csv').open(encoding='utf-8-sig') as handle:
                self.assertEqual(next(csv.DictReader(handle))['theta_fe_deg'],'nan')
            raw=Path(result['raw_directory']);self.assertEqual(len(list(raw.glob('*_rgb.jpg'))),4)
            self.assertEqual(len(list(raw.glob('*_depth.npy'))),4);self.assertTrue((raw/'files.sha256').exists())
            with (Path(result['output_directories'][2])/'compensation.csv').open(encoding='utf-8-sig') as handle:
                row=next(csv.DictReader(handle));self.assertEqual(row['independent_label'],'')
                self.assertEqual(row['label_source'],'instruction_only_not_ground_truth')

    def test_export_failure_does_not_hang_shutdown_or_claim_success(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);here=root/'code';here.mkdir();state=State();rec=TrialRecorder(state,root,here)
            with patch('trial_recorder_v3.export_trial',side_effect=OSError('synthetic failure')),self.assertLogs(level='ERROR'):
                rec.start();rec.begin({'run_id':'synthetic_failure','task_id':1,'participant_id':'TEST'})
                rec.items.put(('quit',));rec.join(10)
            self.assertFalse(rec.is_alive());self.assertFalse(state.recording);self.assertFalse(state.finalizing)
            self.assertIn('synthetic failure',state.record_error);self.assertEqual(state.completed_runs,[])


if __name__=='__main__':unittest.main(verbosity=2)
