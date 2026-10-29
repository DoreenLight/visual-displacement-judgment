import json
from pathlib import Path
import tempfile
import unittest
from trial_selection_v3 import active_history,exclude_trials


class SelectionTests(unittest.TestCase):
    def test_exact_selection_is_persistent_and_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            here=Path(folder);records=[{'run_id':'A'},{'run_id':'B'}]
            path=here/'eight_task_sessions.json';path.write_text(json.dumps(records),encoding='utf-8')
            self.assertEqual(exclude_trials(here,['B'])['active_trials'],1)
            self.assertEqual(active_history(here),[{'run_id':'A'}])
            self.assertEqual(exclude_trials(here,['B'])['excluded_trials'],1)
            self.assertEqual(json.loads((here/'excluded_trials.json').read_text()) [0]['record'],records[1])
            # An interrupted second write cannot accidentally reactivate an excluded run.
            path.write_text(json.dumps(records),encoding='utf-8')
            self.assertEqual(active_history(here),[records[0]])

    def test_unknown_id_does_not_change_anything(self):
        with tempfile.TemporaryDirectory() as folder:
            here=Path(folder);path=here/'eight_task_sessions.json'
            path.write_text('[{"run_id":"A"}]',encoding='utf-8');before=path.read_bytes()
            with self.assertRaises(ValueError):exclude_trials(here,['A','wrong'])
            self.assertEqual(path.read_bytes(),before);self.assertFalse((here/'excluded_trials.json').exists())


if __name__=='__main__':unittest.main(verbosity=2)
