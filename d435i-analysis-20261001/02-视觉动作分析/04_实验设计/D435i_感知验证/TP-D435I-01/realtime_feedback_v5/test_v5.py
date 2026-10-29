import unittest,tempfile,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'attempt_01'))
from training_protocol_v5 import ProtocolEngine,SweepCounter,CircleCounter,HAND
from wrist_depth import hand_depth
from trial_management import TrialManagement
class TestV5(unittest.TestCase):
 def test_short_gap(self):
  c=SweepCounter(10);c.update(12,0);c.update(None,.05);c.update(-12,.1);c.update(None,.15);self.assertEqual(c.update(12,.2),1)
 def test_long_gap(self):
  c=SweepCounter(10);c.update(12,0);c.update(None,.3);c.update(-12,.4);self.assertEqual(c.update(12,.5),0)
 def test_start_negative(self):
  c=SweepCounter(10);c.update(-12,0);c.update(12,.1);self.assertEqual(c.update(-12,.2),1)
 def test_jitter_not_counted(self):
  c=SweepCounter(10)
  for i in range(100):c.update(3*np.sin(i),i*.03)
  self.assertEqual(c.count,0)
 def test_jump_not_counted(self):
  c=SweepCounter(10);c.update(20,0);c.update(-30,.05);c.update(20,.1);self.assertEqual(c.count,0)
 def points(self):return {'r_elbow':np.array([0.,0.,1.]),'r_wrist':np.array([.25,0.,1.]),'hand_wrist':np.array([.25,0.,1.]),'hand_middle_mcp':np.array([.33,0.,1.]),'hand_index_mcp':np.array([.32,-.03,1.]),'hand_pinky_mcp':np.array([.32,.03,1.])}
 def engine(self):
  e=ProtocolEngine();e.select(3);e.baseline={'r_shoulder':np.array([0.,-.3,1.]),'r_elbow':np.array([0.,0.,1.]),'r_wrist':np.array([.25,0.,1.])};e.hand_basis={'normal':np.array([0.,0.,1.]),'fe_zero':0.,'ru_zero':0.};return e
 def test_wrist_without_shoulder(self):
  e=self.engine();o=e.update(self.points(),1);self.assertTrue(o['estimate_valid']);self.assertFalse(o['body_valid']);self.assertFalse(o['compensation_valid'])
 def test_valid_wrist_telemetry_json(self):
  e=self.engine();out=e.update(self.points(),1)
  self.assertIs(type(out['wrist_observation_valid']),bool)
  json.dumps(out,allow_nan=False)
 def test_missing_wrist_telemetry_json(self):
  e=self.engine();out=e.update({},1)
  self.assertIs(type(out['wrist_observation_valid']),bool)
  json.dumps(out,allow_nan=False)
 def test_forearm_invalid(self):
  e=self.engine();p=self.points();p['r_elbow']=p['r_wrist'].copy();self.assertFalse(e.update(p,1)['estimate_valid'])
 def test_hand_calibration_stable(self):
  e=self.engine();e.hand_basis=None
  for i in range(40):e.update(self.points(),i*.035)
  self.assertIsNotNone(e.hand_basis)
 def test_start_requires_hand_zero(self):
  e=self.engine();e.hand_basis=None
  with self.assertRaises(ValueError):e.start('test',1)
 def test_alarm_not_count_gate(self):
  e=self.engine();e.training=True;values=iter([(v,0,'') for v in [18,18,18,0,-18,-18,-18,0,18,18,18]]);e.sweep=SweepCounter(10);e.estimate_angles=lambda p:next(values)
  p=self.points();p['r_shoulder']=np.array([.2,-.3,1.])
  for i in range(11):o=e.update(p,i*.06)
  self.assertGreaterEqual(e.reps,1);self.assertTrue(o['shoulder_alert'])
 def test_hand_depth_edge(self):
  d=np.full((9,9),1500,dtype=np.uint16);d[3:6,3:6]=1000;z,reason,info=hand_depth(d,4,4,.001);self.assertEqual(z,1.);self.assertEqual(reason,'');self.assertEqual(info['depth_patch'],3)
 def test_missing_depth_no_fill(self):
  d=np.full((9,9),1000,dtype=np.uint16);d[4,4]=0;self.assertIsNone(hand_depth(d,4,4,.001)[0])
 def test_circle_short_gap(self):
  c=CircleCounter()
  for i in range(38):
   t=i*.04;a=i*2*np.pi/36
   if i==15:c.update(None,None,t)
   else:c.update(20*np.sin(a),20*np.cos(a),t)
  self.assertEqual(c.count,1)
 def test_circle_long_gap(self):
  c=CircleCounter();c.update(0,20,0);c.update(20,0,.1);c.update(None,None,.5)
  self.assertEqual(c.total,0);self.assertIsNone(c.previous)
 def test_delete_restore_keeps_raw(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);old=root/'attempt_01';old.mkdir();here=root/'realtime_feedback_v5';here.mkdir();raw=root/'original.txt';raw.write_text('unaltered');(old/'eight_task_sessions.json').write_text(json.dumps([{'run_id':'eight_TEST_123','participant_id':'P001'}]));m=TrialManagement(here)
   self.assertEqual(len(m.active()),1);m.apply('eight_TEST_123');self.assertEqual(len(m.active()),0);self.assertEqual(raw.read_text(),'unaltered');m.apply('eight_TEST_123',True);self.assertEqual(len(m.active()),1)
 def test_discard_finalizes_then_marks_deleted(self):
  from types import SimpleNamespace
  import threading
  from unittest.mock import patch
  from trial_recorder_v5 import TrialRecorder
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);here=root/'realtime_feedback_v5';here.mkdir();state=SimpleNamespace(lock=threading.Lock(),finalizing=True)
   recorder=TrialRecorder(state,root,here);recorder.start_run({'run_id':'feedback5_DISCARD_123'})
   with patch('trial_recorder_v5.export_trial',return_value={'run_id':'feedback5_DISCARD_123','participant_id':'P001','task_id':3}):recorder.close_run({'finish_reason':'operator_discarded'})
   manager=TrialManagement(here);self.assertEqual(manager.active(),[]);self.assertEqual(len(manager.deleted()),1);self.assertFalse(state.finalizing)
   self.assertTrue((root/'07_资料库/D435i_感知验证/feedback5_DISCARD_123/run.json').exists())
 def test_unknown_delete_rejected(self):
  with tempfile.TemporaryDirectory() as directory:
   here=Path(directory)/'realtime_feedback_v5';here.mkdir();m=TrialManagement(here)
   with self.assertRaises(ValueError):m.apply('../../data')
   with self.assertRaises(ValueError):m.apply('unknown')
if __name__=='__main__':unittest.main()
