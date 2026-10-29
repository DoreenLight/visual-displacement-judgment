"""Small hand landmark patches: local current-frame cluster, no time filling."""
import numpy as np

def hand_depth(depth,u,v,scale):
 h,w=depth.shape
 # Require the actual center sample to support the target; no nearest/background fill.
 center=float(depth[v,u])*scale
 if not .35<center<2.5:return None,'hand_center_depth_missing',{}
 for radius in (1,2):
  patch=depth[max(0,v-radius):min(h,v+radius+1),max(0,u-radius):min(w,u+radius+1)].astype(float)*scale
  mask=(patch>.35)&(patch<2.5)&(np.abs(patch-center)<=.020)
  y,x=np.nonzero(mask);seen=set();seed=(min(radius,v),min(radius,u));todo=[seed]
  while todo:
   py,px=todo.pop()
   if (py,px) in seen or not (0<=py<patch.shape[0] and 0<=px<patch.shape[1]) or not mask[py,px]:continue
   seen.add((py,px));todo.extend([(py+1,px),(py-1,px),(py,px+1),(py,px-1)])
  vals=np.array([patch[y,x] for y,x in seen]);fraction=len(vals)/patch.size
  if len(vals)>=max(4,patch.size*.5):
   spread=float(np.percentile(vals,90)-np.percentile(vals,10))
   if spread<=.025:return float(np.median(vals)),'',{'depth_sampling':'center_connected_current_frame','depth_patch':2*radius+1,'depth_cluster_fraction':fraction,'depth_cluster_spread_m':spread}
 return None,'hand_depth_cluster_insufficient',{}
