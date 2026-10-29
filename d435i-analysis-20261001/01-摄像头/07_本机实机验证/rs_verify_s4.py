#!/usr/bin/env python3
"""S4 方法端到端验证 —— 用喂帧线程避免 filter 状态中断，验证 S4 的调零与 MDR。

背景：边界测试中发现「自动定位可能选错目标」，
曾把 421px 的小块误当作杆（报 MDR=0.020mm），
实际杆是 1800px、MDR=0.307mm。本脚本用于确认 ROI 身份。

用法
----
sudo -E /opt/anaconda3/envs/realsense/bin/python3 rs_verify_s4.py
"""
"""最终验证：用「永久喂帧线程」避免 filter 状态中断。

诊断结论：问题出在 find_target_block 期间停止向 pipeline 取帧，
打断了 temporal filter 的连续输入。解法是所有处理都放进同一个取帧循环，
不中断。
"""
import sys, numpy as np, time, threading
sys.path.insert(0,"/Users/xbpd/Projects/视觉位移判断/摄像头/07_本机实机验证")
import macos_usb_fix; macos_usb_fix.ensure_ready(verbose=False)
time.sleep(1)
import pyrealsense2 as rs
import rs_measure as M

p=rs.pipeline(); cfg=rs.config()
cfg.enable_stream(rs.stream.depth,848,480,rs.format.z16,30)
cfg.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)
prof=p.start(cfg); al=rs.align(rs.stream.color)
tf=rs.temporal_filter(); tf.set_option(rs.option.filter_smooth_alpha,0.4)
tf.set_option(rs.option.filter_smooth_delta,20)

latest={'d':None}; stop={'v':False}
def feeder():
    while not stop['v']:
        fs=al.process(p.wait_for_frames(timeout_ms=8000))
        d=np.asanyarray(tf.process(fs.get_depth_frame()).get_data())
        latest['d']=d

th=threading.Thread(target=feeder, daemon=True); th.start()
time.sleep(2)
for _ in range(40):
    if latest['d'] is not None: break
    time.sleep(0.1)
print("喂帧线程已启动")

# 20 帧参考（不停止取帧）
st=[]
for _ in range(20):
    d=latest['d']
    while d is None or (st and d is st[-1]): time.sleep(0.01); d=latest['d']
    st.append(d.astype(np.float32).copy())
med=np.median(np.stack(st),axis=0)

# 定位（取帧线程继续跑，不中断）
tgt=M.find_target_block(med, min_px=300, rng=(150,350))
if tgt is None: stop['v']=True; p.stop(); sys.exit(1)
print(f"定位: roi={tgt['roi']}  z={tgt['z']:.0f}mm  {tgt['npx']}px  σ={tgt['sigma']:.1f}mm")

m=M.Measurement(roi=tgt['roi'], trim_mm=20.0, physical_mm=200.0)

base=[]
prev=None
for _ in range(60):
    while latest['d'] is prev: time.sleep(0.005)
    prev=latest['d']; base.append(m.read(latest['d']))
stt=m.calibrate(base)
print(f"调零: {m.summary()}")

N=400; ser=[]; prev=None
for _ in range(N):
    while latest['d'] is prev: time.sleep(0.005)
    prev=latest['d']; ser.append(m.read(latest['d']))
ser=np.array(ser)
st2=M.series_mdr(ser)

stop['v']=True
time.sleep(0.5)
p.stop()

print(f"\n=== {N} 帧稳定性（S4 + temporal）===")
print(f"  z_ref      = {st2['z_ref']:.2f} mm")
print(f"  MDR(3σ)    = {st2['mdr']:.3f} mm")
print(f"  σ 散布     = {st2['scatter']:.3f} mm")
print(f"  漂移       = {st2['drift']:.3f} mm")
print(f"  离群帧     = {st2['glitch']}")
print(f"  取值个数   = {st2['uniq']}")
print(f"\n  要求 2.000mm → {'✅ 达标' if st2['mdr']<=2 else '❌ 未达标'}  余量 {2.0/max(st2['mdr'],1e-9):.1f}×")
