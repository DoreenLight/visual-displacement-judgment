"""Capture and analysis streams are separate; exports never invent missing observations."""
import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from trial_export_v3 import digest, export_trial
from trial_selection_v3 import active_history


class TrialRecorder(threading.Thread):
    def __init__(self,state,root,here):
        super().__init__(daemon=True)
        self.state=state;self.root=Path(root);self.here=Path(here)
        self.items=queue.Queue(maxsize=180);self.directory=None;self.files={}
        self.meta=None;self.saved=0;self.dropped=0;self.analysis_drops=0
        self.history_path=self.here/'eight_task_sessions.json'
        self.history=active_history(self.here)
        state.completed_runs=list(self.history)

    def begin(self,meta):
        with self.state.lock:
            if self.state.recording or self.state.finalizing:raise ValueError('请等待当前试次保存完成')
            self.dropped=self.analysis_drops=0
            self.items.put_nowait(('start',meta))
            self.state.record_run=meta['run_id'];self.state.record_error=None;self.state.recording=True

    def submit_raw(self,frame):
        with self.state.lock:
            if not self.state.recording:return
            try:self.items.put_nowait(('raw',frame,self.state.record_run))
            except queue.Full:
                self.dropped+=1;self.state.record_error='保存队列丢帧，报告会如实列出。'

    def submit_analysis(self,output):
        with self.state.lock:
            if not self.state.recording:return
            try:self.items.put_nowait(('analysis',output,self.state.record_run))
            except queue.Full:
                self.analysis_drops+=1;self.state.record_error='部分分析未保存，报告会计入失效。'

    def event(self,action,detail):
        # Commands run on the processing thread; their ordering is preserved.
        self.items.put(('event',{'event':action,'host_mono_s':time.monotonic(),'detail':detail}))

    def finish(self,info):
        with self.state.lock:
            self.state.recording=False;self.state.finalizing=True
        self.items.put(('finish',info))

    def write_meta(self):
        (self.directory/'run.json').write_text(json.dumps(self.meta,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

    def start_run(self,meta):
        if self.directory:raise RuntimeError('previous run not finalized')
        self.meta=dict(meta);self.saved=0
        self.directory=self.root/'07_资料库'/'D435i_感知验证'/meta['run_id']
        self.directory.mkdir(parents=True,exist_ok=False)
        for name in ('frames','analysis','events'):
            self.files[name]=(self.directory/f'{name}.jsonl').open('w',encoding='utf-8')
        self.write_meta()

    def close_run(self,info):
        if not self.directory:
            with self.state.lock:self.state.finalizing=False
            return
        raw=self.directory
        for handle in self.files.values():handle.close()
        self.files={}
        self.meta.update(info,ended_at=datetime.now().isoformat(),saved_frames=self.saved,
                         dropped_frames=self.dropped,analysis_queue_drops=self.analysis_drops)
        self.write_meta();self.directory=None
        with (raw/'files.sha256').open('w',encoding='utf-8') as handle:
            for file in sorted(raw.iterdir()):
                if file.is_file() and file.name!='files.sha256':handle.write(f'{digest(file)}  {file.name}\n')
        summary=export_trial(raw,self.root)
        summary['raw_directory']=str(raw)
        self.history.append(summary)
        temporary=self.history_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.history,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        temporary.replace(self.history_path)
        with self.state.lock:
            self.state.completed_runs=list(self.history);self.state.finalizing=False

    def run(self):
        while True:
            item=self.items.get()
            try:
                kind=item[0]
                if kind=='quit':
                    if self.directory:self.close_run({'finish_reason':'service_shutdown'})
                    break
                if kind=='start':self.start_run(item[1])
                elif kind=='finish':self.close_run(item[1])
                elif kind=='event' and self.files:
                    self.files['events'].write(json.dumps(item[1],ensure_ascii=False,allow_nan=False)+'\n')
                elif kind in ('raw','analysis') and self.directory and item[2]==self.meta['run_id']:
                    if kind=='analysis':
                        self.files['analysis'].write(json.dumps(item[1],ensure_ascii=False,allow_nan=False)+'\n')
                    else:
                        f=item[1];stem=f'{self.saved:07d}'
                        ok,rgb=cv2.imencode('.jpg',f['color'],[cv2.IMWRITE_JPEG_QUALITY,92])
                        if not ok:raise IOError('RGB encoding failed')
                        (self.directory/f'{stem}_rgb.jpg').write_bytes(rgb.tobytes())
                        np.save(self.directory/f'{stem}_depth.npy',f['depth'])
                        entry={'frame_id':f['id'],'rgb_device_s':f['rgb_s'],'depth_device_s':f['depth_s'],
                               'rgb_clock_domain':f['rgb_clock'],'depth_clock_domain':f['depth_clock'],
                               'host_mono_s':f['received'],'rgb_file':f'{stem}_rgb.jpg','depth_file':f'{stem}_depth.npy'}
                        self.files['frames'].write(json.dumps(entry,allow_nan=False)+'\n');self.saved+=1
            except Exception as error:
                import logging
                logging.exception('Trial save/export failed')
                with self.state.lock:
                    self.state.recording=False;self.state.finalizing=False;self.state.record_error='保存或导出失败：'+str(error)
                try:
                    for handle in self.files.values():handle.close()
                    if self.directory:
                        self.meta.update(finish_reason='storage_error',storage_error=str(error),saved_frames=self.saved,
                                         dropped_frames=self.dropped,ended_at=datetime.now().isoformat())
                        self.write_meta()
                except Exception:logging.exception('Unable to persist failure metadata')
                finally:self.files={};self.directory=None
            finally:
                self.items.task_done()
            if item[0]=='quit':return
