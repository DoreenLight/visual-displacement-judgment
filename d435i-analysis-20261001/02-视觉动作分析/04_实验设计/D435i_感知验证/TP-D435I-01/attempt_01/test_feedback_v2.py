import unittest
import numpy as np
from feedback_engine_v2 import FeedbackEngine


def pose(bend=0, translation=None):
    p = {'l_shoulder': np.array([-.2,0.,1.]), 'r_shoulder': np.array([.2,0.,1.]),
         'r_elbow': np.array([.2,.25,1.]), 'r_wrist': np.array([.2,.25,.75])}
    a = np.radians(bend)
    tip = p['r_wrist'] + np.array([0,-.10*np.sin(a),-.10*np.cos(a)])
    p['r_index'] = tip + np.array([-.015,0,0])
    p['r_pinky'] = tip + np.array([.015,0,0])
    if translation is not None:
        p = {k:v+translation for k,v in p.items()}
    return p


class FeedbackTests(unittest.TestCase):
    def calibrated(self):
        e = FeedbackEngine()
        for t in np.arange(0,3.2,.05):
            e.update(pose(), float(t))
        self.assertIsNotNone(e.baseline)
        return e

    def test_missing_hand_has_no_angle_and_breaks_cycle(self):
        e = self.calibrated()
        e.training = True
        for i in range(12):
            e.update(pose(25),3.2+i*.05)
        p = pose()
        del p['r_index']
        out = e.update(p,3.85)
        self.assertEqual(out['status'],'unknown')
        self.assertIsNone(out['bend_deg'])
        self.assertFalse(e.active)
        for i in range(10):
            e.update(pose(),3.9+i*.05)
        self.assertEqual(e.reps,0)
        self.assertEqual(e.interrupted_reps,1)

    def test_translation_does_not_create_shoulder_alert(self):
        e = self.calibrated()
        e.training = True
        for i in range(20):
            out=e.update(pose(25,np.array([.03,.03,0])),3.2+i*.05)
        self.assertAlmostEqual(out['shoulder_mm'],0)
        self.assertFalse(out['shoulder_alert'])

    def test_only_full_return_counts(self):
        e=self.calibrated()
        e.training=True
        t=3.2
        for bend in [10]*6+[25]*10+[12]*6+[0]*15:
            e.update(pose(bend),t)
            t+=.05
        self.assertEqual(e.reps,1)
        for i in range(10):
            e.update(pose(),t+i*.05)
        self.assertEqual(e.reps,1)

    def test_unstable_calibration_does_not_pass_on_timeout(self):
        e=FeedbackEngine()
        for i in range(200):
            e.update(pose(30 if i%2 else 0),i*.05)
        self.assertIsNone(e.baseline)

    def test_zero_visibility_is_not_a_valid_measurement(self):
        # Actual perception gate is tested with mocked MediaPipe output and depth.
        from types import SimpleNamespace
        from live_feedback_v2 import measure
        landmarks=[SimpleNamespace(x=.5,y=.5,visibility=0.,presence=1.) for _ in range(33)]
        points,pixels=measure(SimpleNamespace(pose_landmarks=[landmarks]),{'depth':np.full((20,20),1000,dtype=np.uint16),'scale':.001})
        self.assertEqual(points,{})
        self.assertEqual(pixels,{})

    def test_pause_preserves_recorded_counts_but_cancels_cycle(self):
        e=self.calibrated()
        e.training=True
        for i in range(10):e.update(pose(25),3.2+i*.05)
        e.training=False
        e.interrupt()
        self.assertEqual(e.reps,0)
        self.assertFalse(e.active)

    def test_shoulder_alert_has_persistence_and_recovery(self):
        e=self.calibrated()
        e.training=True
        p=pose(25)
        p['r_shoulder'][1]-=.045
        first=e.update(p,3.2)
        self.assertFalse(first['shoulder_alert'])
        for i in range(1,8):out=e.update(p,3.2+i*.05)
        self.assertTrue(out['shoulder_alert'])
        out=e.update(pose(25),3.6)
        self.assertFalse(out['shoulder_alert'])

    def test_recorded_frame_pairs_and_stop_metadata(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import patch
        import live_feedback_v2 as app
        with tempfile.TemporaryDirectory() as temp, patch.object(app,'ROOT',Path(temp)):
            state=app.State()
            writer=app.Recorder(state)
            writer.start()
            writer.items.put(('start',{'name':'synthetic_test'}, {'calibration_id':None,'target_deg':20}))
            writer.items.join()
            state.recording=True
            frame={'id':3,'color':np.zeros((16,16,3),dtype=np.uint8),
                   'depth':np.full((16,16),777,dtype=np.uint16),'rgb_s':1.,'depth_s':1.001,
                   'rgb_clock':'test','depth_clock':'test','received':2.}
            writer.submit(frame,{'status':'unknown','bend_deg':None})
            writer.items.put(('stop',))
            writer.items.join()
            writer.items.put(('quit',))
            writer.join(5)
            self.assertFalse(writer.is_alive())
            files=list(Path(temp).rglob('run.json'))
            self.assertEqual(len(files),1)
            meta=json.loads(files[0].read_text(encoding='utf-8'))
            self.assertEqual(meta['saved_frames'],1)
            self.assertEqual(meta['accuracy_verdict'],'BLOCKED')
            data=json.loads((files[0].parent/'frames.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(data['frame_id'],3)
            self.assertTrue((files[0].parent/data['rgb_file']).is_file())
            np.testing.assert_array_equal(np.load(files[0].parent/data['depth_file']),frame['depth'])

    def test_stale_results_are_hidden(self):
        import time
        from live_feedback_v2 import State
        state=State()
        state.telemetry.update(output_mono_s=time.monotonic()-2, bend_deg=20, shoulder_alert=True)
        result=state.snapshot()
        self.assertEqual(result['status'],'unknown')
        self.assertIsNone(result['bend_deg'])
        self.assertFalse(result['shoulder_alert'])


if __name__=='__main__':
    unittest.main()
