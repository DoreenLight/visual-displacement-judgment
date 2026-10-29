import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from analyze import gap,elbow,alerts,SRC
sys.path.insert(0,str(SRC))
from training_protocol_v3 import SweepCounter
class AnalysisTests(unittest.TestCase):
 def test_gap_middle(self):
  self.assertAlmostEqual(gap(np.array([0,.1,.2,.3]),np.array([True,False,False,True])),.2)
 def test_gap_terminal(self):
  self.assertAlmostEqual(gap(np.array([0,.1,.2,.3]),np.array([True,True,False,False])),.2)
 def test_elbow_translation_invariant(self):
  s,e,w=np.array([0.,0,0]),np.array([1.,0,0]),np.array([1.,1,0]);delta=np.array([.4,-.7,2.])
  self.assertAlmostEqual(elbow(s,e,w),90);self.assertAlmostEqual(elbow(s+delta,e+delta,w+delta),90)
 def test_alert_persistence_and_invalid_reset(self):
  a=alerts(np.ones((4,2))*1.2,np.array([0,.1,.2,.3]),np.array([True,True,True,False]))
  np.testing.assert_array_equal(a,[[False,False],[False,False],[True,True],[False,False]])
 def test_counter_requires_signed_crossing(self):
  c=SweepCounter(15)
  for x in [40,25,5,25,40]:c.update(x)
  self.assertEqual(c.count,0)
  for x in [-20,20]:c.update(x)
  self.assertEqual(c.count,1)
 def test_missing_angle_resets_partial_cycle(self):
  c=SweepCounter(15)
  for x in [20,None,-20,20]:c.update(x)
  self.assertEqual(c.count,0)
if __name__=='__main__':unittest.main()
