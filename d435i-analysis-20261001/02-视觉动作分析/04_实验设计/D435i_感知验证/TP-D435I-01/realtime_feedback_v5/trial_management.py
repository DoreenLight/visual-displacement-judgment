"""Recoverable deletion overlay; frozen raw data and earlier indices remain untouched."""
import json,re
from datetime import datetime
from pathlib import Path
from trial_selection_v3 import active_history,atomic_json,read_json
class TrialManagement:
 def __init__(self,here):self.here=Path(here);self.path=self.here/'deleted_trials.json'
 def all_runs(self):
  by_id={}
  for directory in ('attempt_01','realtime_curves_v4','realtime_feedback_v5'):
   for record in active_history(self.here.parent/directory):by_id[record['run_id']]=record
  return list(by_id.values())
 def deleted(self):return read_json(self.path,[])
 def active(self):
  removed={r['run_id'] for r in self.deleted()};return [r for r in self.all_runs() if r['run_id'] not in removed]
 def apply(self,run_id,restore=False):
  if not isinstance(run_id,str) or not re.fullmatch(r'[A-Za-z0-9_]+',run_id):raise ValueError('无效试次编号')
  known={r['run_id']:r for r in self.all_runs()}
  if run_id not in known:raise ValueError('试次不存在或已经被原始选择规则排除')
  deleted=self.deleted()
  if restore:deleted=[r for r in deleted if r['run_id']!=run_id]
  elif run_id not in {r['run_id'] for r in deleted}:
   deleted.append({'run_id':run_id,'deleted_at':datetime.now().astimezone().isoformat(),'reason':'operator_deleted_in_ui','record':known[run_id],'physical_files_retained':True})
  atomic_json(self.path,deleted);return self.active()
