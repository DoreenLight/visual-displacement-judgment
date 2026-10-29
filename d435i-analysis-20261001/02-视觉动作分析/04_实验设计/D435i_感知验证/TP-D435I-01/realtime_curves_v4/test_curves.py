import unittest
import numpy as np
from curves import Curves
class Tests(unittest.TestCase):
 def points(self):return {'r_shoulder':np.array([0.,0.,0.]),'r_elbow':np.array([0.,1.,0.]),'r_wrist':np.array([1.,1.,0.])}
 def out(self,t,**kw):return dict(output_mono_s=t,received_mono_s=t,body_valid=True,estimate_valid=True,upper_arm_swing_deg=0.,theta_fe_est_deg=20.,theta_ru_est_deg=10.,calibration_id='a',run_id='b',task_id=3,**kw)
 def test_elbow_and_units(self):
  x=Curves().update(self.out(1),self.points(),{});self.assertAlmostEqual(x['values_deg']['elbow'],90);self.assertFalse(x['standard_biomechanical_validated'])
 def test_independent_wrist_missing(self):
  o=self.out(1);o['estimate_valid']=False;x=Curves().update(o,self.points(),{});self.assertIsNone(x['values_deg']['wrist_fe']);self.assertIsNotNone(x['values_deg']['elbow'])
 def test_causal_smoothing(self):
  c=Curves();c.update(self.out(1),self.points(),{});o=self.out(1.1);o['theta_fe_est_deg']=40;x=c.update(o,self.points(),{});self.assertGreater(x['values_deg']['wrist_fe'],20);self.assertLess(x['values_deg']['wrist_fe'],40)
 def test_invalid_break(self):
  c=Curves();c.update(self.out(1),self.points(),{});o=self.out(1.1);o['estimate_valid']=False;c.update(o,self.points(),{});o=self.out(1.2);o['theta_fe_est_deg']=40;self.assertEqual(c.update(o,self.points(),{})['values_deg']['wrist_fe'],40)
 def test_long_gap_resets(self):
  c=Curves();c.update(self.out(1),self.points(),{});o=self.out(1.4);o['theta_fe_est_deg']=40;self.assertEqual(c.update(o,self.points(),{})['values_deg']['wrist_fe'],40)
 def test_baseline_change_resets(self):
  c=Curves();c.update(self.out(1),self.points(),{});o=self.out(1.1);o['calibration_id']='new';o['theta_fe_est_deg']=40;self.assertEqual(c.update(o,self.points(),{})['values_deg']['wrist_fe'],40)
if __name__=='__main__':unittest.main()
