import tempfile,threading,json,time,sys
from pathlib import Path
import live_feedback_v5 as app
original_here=app.HERE
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);here=root/'realtime_feedback_v5';here.mkdir();(here/'live_feedback_v5.html').write_bytes((original_here/'live_feedback_v5.html').read_bytes());(here/'eight_task_sessions.json').write_text(json.dumps([{'run_id':'feedback5_TEST_123','participant_id':'P001','task_id':3,'task_name':'Fixture FE','captured_frames':30,'active_duration_s':1,'finish_reason':'fixture'}]))
 app.HERE=here;app.ROOT=root;state=app.State();recorder=app.TrialRecorder(state,root,here);state.record_sink=recorder;recorder.start();state.engine.select(3)
 server=app.ThreadingHTTPServer(('127.0.0.1',8091),app.handler(state));server.daemon_threads=True
 workers=[threading.Thread(target=server.serve_forever,daemon=True),threading.Thread(target=app.process,args=(state,recorder),daemon=True)]
 for worker in workers:worker.start()
 print('FIXTURE_READY',flush=True)
 while not state.stop.wait(.25):pass
 workers[1].join(timeout=10);recorder.items.put(('quit',));recorder.join();server.shutdown();server.server_close()
