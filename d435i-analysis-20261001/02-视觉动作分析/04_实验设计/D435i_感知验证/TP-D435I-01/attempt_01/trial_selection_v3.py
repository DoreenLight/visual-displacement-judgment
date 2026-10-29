"""Explicit trial selection; original RGB-D and per-run outputs remain immutable."""
import json
from pathlib import Path
from datetime import datetime


def read_json(path,default):
    path=Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def active_history(here):
    here=Path(here)
    excluded={r['run_id'] for r in read_json(here/'excluded_trials.json',[])}
    return [r for r in read_json(here/'eight_task_sessions.json',[]) if r['run_id'] not in excluded]


def atomic_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def exclude_trials(here,run_ids,reason='user_marked_unneeded'):
    here=Path(here);requested=set(run_ids)
    history=read_json(here/'eight_task_sessions.json',[])
    excluded=read_json(here/'excluded_trials.json',[])
    known={r['run_id'] for r in history}|{r['run_id'] for r in excluded}
    if requested-known:raise ValueError('Unknown trial IDs: '+', '.join(sorted(requested-known)))
    already={r['run_id'] for r in excluded}
    for r in history:
        if r['run_id'] in requested and r['run_id'] not in already:
            excluded.append({'run_id':r['run_id'],'excluded_at':datetime.now().astimezone().isoformat(),
                             'reason':reason,'record':r})
    # The exclusion manifest is authoritative if interrupted between the two writes.
    remaining=[r for r in history if r['run_id'] not in requested|already]
    atomic_json(here/'excluded_trials.json',excluded)
    atomic_json(here/'eight_task_sessions.json',remaining)
    return {'active_trials':len(remaining),'excluded_trials':len(excluded),'requested':sorted(requested)}
