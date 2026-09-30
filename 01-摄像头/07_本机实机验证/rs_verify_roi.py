#!/usr/bin/env python3
"""ROI 身份验证 —— 列出场景中所有深度一致块，确认自动定位选中的确实是杆。

背景：边界测试中发现「自动定位可能选错目标」，
曾把 421px 的小块误当作杆（报 MDR=0.020mm），
实际杆是 1800px、MDR=0.307mm。本脚本用于确认 ROI 身份。

用法
----
sudo -E /opt/anaconda3/envs/realsense/bin/python3 rs_verify_roi.py
"""
"""关键验证：自动定位的 ROI 到底是什么？对比杆 ROI。"""
import sys, numpy as np, time, threading
sys.path.insert(0,"/Users/xbpd/Projects/视觉位移判断/摄像头/07_本机实机验证")
import macos_usb_fix; macos_usb_fix.ensure_ready(verbose=False)
time.sleep(1)
import pyrealsense2 as rs
import rs_measure as M
from PIL import Image

p=rs.pipeline(); cfg=rs.config()
cfg.enable_stream(rs.stream.depth,848,480,rs.format.z16,30)
cfg.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)
prof=p.start(cfg); al=rs.align(rs.stream.color)
tf=rs.temporal_filter(); tf.set_option(rs.option.filter_smooth_alpha,0.4)
latest={'d':None,'c':None}; stop={'v':False}
def feeder():
    while not stop['v']:
        fs=al.process(p.wait_for_frames(timeout_ms=8000))
        latest['d']=np.asanyarray(tf.process(fs.get_depth_frame()).get_data())
        latest['c']=np.asanyarray(fs.get_color_frame().get_data()).copy()
threading.Thread(target=feeder,daemon=True).start()
time.sleep(3)
while latest['d'] is None: time.sleep(0.1)

st=[]
for _ in range(20):
    st.append(latest['d'].astype(np.float32).copy()); time.sleep(0.04)
med=np.median(np.stack(st),axis=0)
col=latest['c'].copy()

tgt=M.find_target_block(med, min_px=300, rng=(150,350))
print(f"自动定位 ROI: {tgt['roi']}  z={tgt['z']:.0f}mm")

# 列出 150-350mm 所有块
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
valid=(med>0)&(med<65535); n=int(valid.sum()); h,w=med.shape
di=med.astype(np.int32); idx=np.full((h,w),-1,np.int64); idx[valid]=np.arange(n)
mh=valid[:,:-1]&valid[:,1:]&(np.abs(di[:,:-1]-di[:,1:])<30)
mv=valid[:-1,:]&valid[1:,:]&(np.abs(di[:-1,:]-di[1:,:])<30)
rh,ch=np.where(mh); rv,cv=np.where(mv)
rows=np.concatenate([idx[rh,ch],idx[rv,cv]]); cols=np.concatenate([idx[rh,ch+1],idx[rv+1,cv]])
g=coo_matrix((np.ones(rows.size,np.int8),(rows,cols)),shape=(n,n))
ncomp,labels=connected_components(g,directed=False)
cnt=np.bincount(labels,minlength=ncomp); vy,vx=np.where(valid)
print(f"\n{'中位Z':>7} {'像素':>6} {'宽':>4} {'高':>4} {'σ':>6}  {'位置(y)':>8}  形状")
blocks=[]
for k in range(ncomp):
    if cnt[k]<150: continue
    sel=labels==k; ys,xs=vy[sel],vx[sel]
    z=float(np.median(med[ys,xs]))
    if not(150<z<350): continue
    sig=float(np.std(med[ys,xs]))
    bw=int(xs.max()-xs.min()+1); bh=int(ys.max()-ys.min()+1)
    shape = "细高(杆)" if bh>bw*1.5 else ("横条" if bw>bh*1.5 else "方块")
    blocks.append((z,len(ys),int(xs.min()),int(ys.min()),bw,bh,sig,shape))
    print(f"{z:>7.0f} {len(ys):>6} {bw:>4} {bh:>4} {sig:>6.1f}  y={ys.min():>3}-{ys.max():<3} {shape}")

# 画图：标出自动 ROI 和所有块
img=col.copy()
for z,npx,x0,y0,bw,bh,sig,shape in blocks:
    _c=(0,255,0) if shape=="细高(杆)" else (255,255,0)
    img[max(0,y0-1):y0+bh+1, max(0,x0-1):x0+bw+1] = _c
x,y,rw,rh2 = tgt['roi']
img[max(0,y-2):y+rh2+2, max(0,x-2):x+rw+2] = (0,0,255)   # 自动 ROI 红框
Image.fromarray(img[:,:,::-1]).save("/tmp/roi_verify.png")
print(f"\n已存 /tmp/roi_verify.png（红=自动选中，绿=细高块，黄=其他）")

# 对比两种 ROI 的 MDR
rod = max(blocks, key=lambda b: b[1] if b[7]=="细高(杆)" else -1)
print(f"\n=== 对比测试 ===")
for name, roi in (("自动选中", tgt['roi']),
                  ("最大细高块", (rod[2],rod[3],rod[4],rod[5])) if rod[7]=="细高(杆)" else None):
    if roi is None: continue
    m=M.Measurement(roi=roi, trim_mm=20.0)
    ser=[]; prev=None
    for _ in range(300):
        while latest['d'] is prev: time.sleep(0.005)
        prev=latest['d']; ser.append(m.read(latest['d']))
    s=np.array(ser); stt=M.series_mdr(s)
    print(f"  {name} roi={roi}")
    print(f"    z_ref={stt['z_ref']:.2f}  MDR_diff={stt['mdr']:.4f}  "
          f"3σ散布={stt['mdr_scatter']:.4f}  σ={stt['scatter']:.4f}  "
          f"比值={stt['mdr_scatter']/max(stt['mdr'],1e-9):.2f}")
stop['v']=True; time.sleep(0.5)
try: p.stop()
except: pass
