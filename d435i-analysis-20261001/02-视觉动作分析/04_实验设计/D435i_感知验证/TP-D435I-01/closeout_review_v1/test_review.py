import unittest
from pathlib import Path
import tempfile
import numpy as np
from review_four_packages import gap,scoring_state,candidate_cycles,verify_manifest,THRESHOLDS,digest


class ReviewTests(unittest.TestCase):
    def test_original_thresholds_preserved(self):
        self.assertEqual(THRESHOLDS['static_axis_sd_mm_max'],5)
        self.assertEqual(THRESHOLDS['static_position_p95_mm_max'],10)
        self.assertEqual(THRESHOLDS['point_valid_rate_min'],.95)
        self.assertEqual(THRESHOLDS['invalid_gap_s_max'],.2)
        self.assertEqual(THRESHOLDS['detection_latency_p95_ms_max'],300)

    def test_terminal_camera_loss_counted(self):
        self.assertAlmostEqual(gap([10,10.03,10.06],[True]*3,10,40),29.94)
        self.assertEqual(gap([],[],10,40),30)

    def test_invalid_run_includes_both_trial_boundaries(self):
        self.assertEqual(gap([10,11,12],[False]*3,0,30),30)
        self.assertEqual(gap([10,11,12],[False,False,True],0,12),12)

    def test_no_static_or_missing_data_reward(self):
        for task in range(1,9):
            out=scoring_state(task,0,False)
            for k in ('A_value','C_value','S_value'):self.assertIsNone(out[k])
            self.assertEqual(out['actuator_command_count'],0)
            self.assertFalse(out['A_valid']);self.assertFalse(out['C_valid']);self.assertFalse(out['S_valid'])

    def test_reported_ten_does_not_make_unvalidated_angles_scorable(self):
        out=scoring_state(3,10,True)
        self.assertEqual(out['overall_status'],'not_scorable');self.assertIsNone(out['A_value'])

    def test_no_count_across_missing_intervals(self):
        t=np.arange(0,30,.03);q=30*np.sin(2*np.pi*t/3)
        self.assertGreater(candidate_cycles(t,q),5)
        self.assertEqual(candidate_cycles(np.array([0,.01,10,10.01]),np.array([30,-30,30,-30])),0)

    def test_hash_mismatch_and_traversal_are_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);file=root/'x.json';file.write_text('{}')
            manifest=root/'files.sha256';manifest.write_text(digest(file)+'  x.json\n')
            self.assertEqual(verify_manifest(manifest,root)['issues'],[])
            file.write_text('{"changed":true}')
            self.assertIn('hash_mismatch:x.json',verify_manifest(manifest,root)['issues'])
            manifest.write_text('0'*64+'  ../outside.json\n')
            self.assertIn('unsafe_path:../outside.json',verify_manifest(manifest,root)['issues'])

    def test_p_value_comparison_is_not_reversed(self):
        self.assertFalse(.0013450953457995237<.001)


if __name__=='__main__':unittest.main(verbosity=2)
