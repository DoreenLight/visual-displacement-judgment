"""Engineering angle traces; raw observations and valid gaps retained."""
import math
import numpy as np

class Curves:
    def __init__(self, tau=.10):
        self.tau=tau; self.previous={}; self.context=None
    def update(self, out, points, baseline):
        now=out.get('received_mono_s',out['output_mono_s'])
        context=(out.get('calibration_id'),out.get('run_id'),out.get('task_id'),out.get('paused'))
        if context!=self.context: self.previous.clear(); self.context=context
        body=bool(out.get('body_valid'))
        elbow=None
        if body and all(k in points for k in ('r_shoulder','r_elbow','r_wrist')):
            a=points['r_shoulder']-points['r_elbow']; b=points['r_wrist']-points['r_elbow']
            n=float(np.linalg.norm(a)*np.linalg.norm(b))
            if n>1e-9: elbow=180-math.degrees(math.acos(float(np.clip(np.dot(a,b)/n,-1,1))))
        raw={'shoulder':out.get('upper_arm_swing_deg') if body and baseline is not None else None,
             'elbow':elbow,'wrist_fe':out.get('theta_fe_est_deg') if out.get('estimate_valid') else None,
             'wrist_ru':out.get('theta_ru_est_deg') if out.get('estimate_valid') else None}
        values={}
        for key,v in raw.items():
            if v is None or not math.isfinite(v): self.previous.pop(key,None); values[key]=None; continue
            prev=self.previous.get(key); dt=now-prev[0] if prev else None
            filtered=v if not prev or dt<=0 or dt>.25 else prev[1]+(-math.expm1(-dt/self.tau))*(v-prev[1])
            self.previous[key]=(now,filtered);values[key]=round(filtered,3)
        return {'capture_mono_s':now,'frame_id':out.get('frame_id'),'run_id':out.get('run_id'),
                'calibration_id':out.get('calibration_id'),'task_id':out.get('task_id'),'paused':out.get('paused',False),
                'values_deg':values,'raw_values_deg':raw,'tau_s':self.tau,
                'standard_biomechanical_validated':False,'gap_break_s':.25}
