from pathlib import Path
import json,csv,shutil,os,sys,io,contextlib,hashlib
import numpy as np,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];BASE=Path(__file__).parent;DENSE=BASE/'dense_20261001_v1';OUT=BASE/'analysis_20261001_v1'
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def num(x):return float(x) if x not in ['',None] else None

def main():
 assert load(DENSE/'execution.json')['exit_code']==0
 a=load(OUT/'analysis_summary.json');assert a['completed_trials']==25 and a['frames']==29938
 tr=rows(OUT/'trial_metrics.csv');joints=rows(OUT/'joint_metrics.csv');old=load(ROOT/'04_实验设计/D435i_感知验证/TP-D435I-01/attempt_01/eight_task_sessions.json')
 assert {x['run_id'] for x in tr}=={x['run_id'] for x in old}
 ov=round(sum(x['keypoint_statistics']['wrist']['valid_rate']*x['captured_frames'] for x in old));dv=sum(int(x['body_valid_frames']) for x in tr);n=29938
 groups=[]
 for t in range(1,9):
  ts=[x for x in tr if int(x['task_id'])==t];rs=[x for x in old if x['task_id']==t];frames=sum(int(x['frames']) for x in ts)
  groups.append(dict(task_id=t,trials=len(ts),frames=frames,online_rate=sum(x['keypoint_statistics']['wrist']['valid_rate']*x['captured_frames'] for x in rs)/frames,dense_rate=sum(int(x['body_valid_frames']) for x in ts)/frames,candidate_rate=sum(int(x['candidate_angle_frames']) for x in ts)/frames))
 save(OUT/'group_summary.json',groups)
 fig,ax=plt.subplots(figsize=(10,4.5));x=np.arange(8);w=.25
 for shift,key,label in [(-w,'online_rate','Original online body'),(0,'dense_rate','Dense offline body'),(w,'candidate_rate','Dense candidate angles')]:ax.bar(x+shift,[g[key]*100 for g in groups],w,label=label)
 ax.set_xticks(x,['S1','S2','FE','RU','CIRCLE','SHOULDER','ELBOW','COMBINED']);ax.set_ylim(0,110);ax.set_ylabel('Valid / all captured frames (%)');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(OUT/'coverage_comparison.png',dpi=150);plt.close(fig)
 static=[j for j in joints if int(j['task_id']) in [1,2] and j['joint'] in ['r_shoulder','r_elbow','r_wrist']]
 fig,ax=plt.subplots(figsize=(9,4.3))
 for joint in ['r_shoulder','r_elbow','r_wrist']:ax.plot(range(6),[num(j['position_p95_mm']) for j in static if j['joint']==joint],marker='o',label=joint)
 ax.set_xticks(range(6),['S1-1','S1-2','S1-3','S2-1','S2-2','S2-3']);ax.set_ylabel('Raw position fluctuation P95 (mm)');ax.set_title('Static repetition; not independent accuracy');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'static_stability.png',dpi=150);plt.close(fig)
 notes={
 'eight_S1_20261001_131959_59d5ec91':'静止且关键点位置基本贴合画面，未触发额外参与提示合理；部分手点深度无效。',
 'eight_S1_20261001_132059_bc0d3f66':'采样画面姿态基本稳定却持续combined；旧基线与重新摆位不匹配的证据强，不能视为持续真实代偿。',
 'eight_S1_20261001_132148_b0ac2f4c':'采样画面基本静止，旧基线持续combined；试次局部基线反事实不再报警，但尚未改实时服务。',
 'eight_S2_20261001_132239_ea3010e2':'握杯时大关节点仍可见；原存储身体基线/手掌基准缺失导致unknown/角无效，不是三维点未采到。',
 'eight_FE_20261001_133658_73162b96':'手姿态变化且前臂较稳定，点位追踪大致合理；FE/RU候选都明显变化与偏零，不能证明轴分离或计数正确。',
 'eight_RU_20261001_133347_d19a29ac':'该试次候选角可用96.96%，周期变化清楚，但RU候选一直偏负、主周期也出现在另一候选通道；只能证明存在可提取运动信号，不能以指令名判定轴准确。',
 'eight_RU_20261001_133130_b225807d':'手方向变化可见，指根深度间歇无效；候选RU偏零且缺失，不能证明可靠计数。',
 'eight_CIRCLE_20261001_134353_ece4c4e8':'肘支撑时手朝向变化可见；未触发额外参与大致合理；六帧不能独立确认圈数或真实方向。',
 'eight_SHOULDER_20261001_134728_5cedac08':'肩/上臂姿态变化与肩通道提示有可观测联系；归因仍受基线影响，不能证明临床原因。',
 'eight_ELBOW_20261001_135236_ac256f5f':'明显外展和肘运动可见；原肩通道包含上臂摆动，因此combined可解释，但不证明同时耸肩。',
 'eight_COMBINED_20261001_135607_5bb6b817':'手臂与躯干/肩姿态变化可见，混合提示有迹象；部分帧因观测/几何失效进入unknown。'}
 reviews=[]
 sys.path.insert(0,str(BASE));from analyze import lines,overlay
 for run,note in notes.items():
  c=OUT/'cases'/run;sel=load(c/'selection.json');records=lines(DENSE/'runs'/run/'measurements.jsonl');times=np.array([x['t_s'] for x in records]);path=c/'overlay_clip.avi'
  if not path.exists():
   writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),15,(848,480));assert writer.isOpened()
   for t in np.arange(sel['start_s'],sel['end_s'],1/15):writer.write(overlay(ROOT/'07_资料库/D435i_感知验证'/run,records[min(int(np.searchsorted(times,t)),len(times)-1)]))
   writer.release()
  cap=cv2.VideoCapture(str(path));ok,_=cap.read();count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release();assert ok
  sel.update(review_status='sampled_images_and_signals_checked',review_scope='六张叠加采样帧与信号合理性；无独立完整逐事件标注',note=note,video_frames=count,video_decoded=True,video_encoding='MJPEG AVI 15fps nearest capture visualization only');save(c/'selection.json',sel);reviews.append(sel)
 save(OUT/'visual_review.json',reviews)
 shutil.copy2(BASE/'preview/failure_boundary_examples.jpg',OUT/'failure_boundary_examples.jpg');shutil.copy2(BASE/'preview/failure_boundary_examples.json',OUT/'failure_boundary_examples.json')
 text='''# 现有25试次离线重分析：三维点位、腕轨迹与肩肘参与

## 结论

- **三维点位具备初步能力**：逐帧连续性明显改善，但各关节毫米级稳定性并不一致。
- **腕/手位置轨迹与候选角度可以获得**：解剖FE/RU、ROM及可靠计数仍未建立。
- **明显肩肘额外运动可以产生提示**：旧基线和通道语义影响归因，尚不支持可靠临床代偿分类。

本轮覆盖25个保留试次、29938组RGB/对齐深度，单一P001。结论适用于当前录像和布置，不等于D435i普遍硬件极限、正式准确性或临床效度。本轮不修改实时服务，不覆盖原片和旧输出。

## 方法

逐试次新建VIDEO跟踪器，所有捕获帧按原时间顺序推理，不跳帧、不补点、不套实时超时。复用原模型、深度窗口/门控、候选角及规则，存储基线为主结果；暂停/方向切换按事件回放。全部帧上下文与原实时序列不同，覆盖改善不能全部归因于跳帧单一因素，也不证明实时性能已达标。

独立点位测量与肩肘腕组合几何门控分别报告。深度表面点不等于已验证解剖中心；处理耗时不是同步端到端延迟。

'''
 text+='原在线组合有效 **{}/29938（{:.1%}）**，离线组合有效 **{}/29938（{:.1%}）**。\n\n'.format(ov,ov/n,dv,dv/n)
 text+='|任务|次数|捕获帧|在线组合可用|离线组合可用|离线候选角可用|\n|---|---:|---:|---:|---:|---:|\n'
 for g in groups:text+='|{}|{}|{}|{:.1%}|{:.1%}|{:.1%}|\n'.format(['S1','S2','FE','RU','回转','肩诱导','肘诱导','共同诱导'][g['task_id']-1],g['trials'],g['frames'],g['online_rate'],g['dense_rate'],g['candidate_rate'])
 text+='\n![覆盖率同分母比较](analysis_20261001_v1/coverage_comparison.png)\n\n## 静止稳定性\n\n全试次原始坐标，无滤波、无独立参考。动态波动不能解释为传感器噪声。\n\n|试次|关节|独立点可用率|X/Y/Z标准差毫米|位置波动P95毫米|\n|---|---|---:|---|---:|\n'
 for j in static:text+='|{}|{}|{:.1%}|{:.1f}/{:.1f}/{:.1f}|{:.1f}|\n'.format(j['run_id'],j['joint'],num(j['valid_rate']),num(j['x_sd_mm']),num(j['y_sd_mm']),num(j['z_sd_mm']),num(j['position_p95_mm']))
 text+='''
![静止重复试次](analysis_20261001_v1/static_stability.png)

首个S1肘点X标准差12.6mm、Z标准差3.1mm、U像素SD8.6px；固定深度的XY敏感度SD约12.3mm。腕点Z标准差约1.5mm，XY像素变化对应约6.5mm。这是代数敏感度对照，不是独立方差分解；支持定位漂移/人体微动影响横向坐标，不能称为纯相机噪声。

## 清晰片段合理性检查

按10秒窗口的身体/候选角可用率选片，检查六帧叠加画面和信号。该选择偏向清晰案例，不能替代全体性能。11段AVI是15fps捕获时间可视化重采样，所有统计仍来自全量原始帧。未建立独立完整逐事件标签。

'''
 for v in reviews:
  path='analysis_20261001_v1/cases/'+v['run_id'];text+='- **{}，{:.1f}–{:.1f}s**：{} [叠加回放]({}/overlay_clip.avi)，[六帧]({}/contact_sheet.jpg)，[信号]({}/signals.png)。\n'.format(v['run_id'],v['start_s'],v['end_s'],v['note'],path,path,path)
 text+='''
## 主要诊断

**旧基线**：S1第二/第三试次沿用第一个试次之前的基线；本轮原基线报警均约99.3%，首2秒试次局部基线诊断为0%，采样画面稳定。支持基线不匹配，不能解释为持续真实代偿。局部重置只作反事实，动态试次开头可能已运动，不能一概用首2秒自动纠正。

**腕角与计数**：固定中立手掌法向且缺前臂轴向参考，候选角会偏零/串扰。指根深度窗口易碰到手缘、背景及空洞；缺任一点使角度无效。原计数需连续跨正负阈值，缺角重置半周期；逐帧后计数仍为0，问题不只实时跳帧。

**肩肘语义**：肩通道包含上臂摆动，飞肘会同时触发肩通道。combined是两条规则同时触发，不能直接说看到了耸肩。normal仅表示未触发当前运动规则，不证明腕训练完成。

**点位正确性**：一致的背景深度可能通过局部窗口门控；需要RGB/深度对照及几何检查。小手点和大关节点的可用性不能混为一谈。

![三个实际边界案例](analysis_20261001_v1/failure_boundary_examples.jpg)

边界图依次为FE手缘深度分散、飞肘时候选角超软件可观测范围、共同动作中身体几何/点位失效。最后一种需结合同帧坐标进一步归因，不能只凭unknown称硬件损坏。伪彩深度仅供位置对照，颜色不是误差真值。

## 设备边界与下一步

当前支持大关节点的连续表面点位、部分腕轨迹及明显额外运动筛查；手部边缘/遮挡、前臂轴向旋转、重新摆位和大幅动作是已发现的系统边界。0.35–2.5m范围、7×7窗口、50mm/15deg阈值都是软件参数，不是硬件极限。

没有多距离/多视角和独立参考，无法估计D435i普遍最近/最远可靠距离、解剖中心误差或全部条件保证。应先改进检测/深度目标所属/基线/角度轴，再判断是否需要改变硬件。

[原理与官方依据](PRINCIPLES_AND_BOUNDARIES.md)；[改进顺序](NEXT_ACTIONS.md)。

## 复现与证据

- [输入、模型和代码哈希](dense_20261001_v1/execution.json)
- [25试次指标](analysis_20261001_v1/trial_metrics.csv)、[175条关节诊断](analysis_20261001_v1/joint_metrics.csv)
- [逐帧记录目录](dense_20261001_v1/runs/)
- [分析伴随笔记本](analysis_companion.ipynb)、[核验记录](verification.json)

原正式FAIL/BLOCKED复核保持历史口径；本轮回答初步能力与当前布置边界，没有改变验收阈值。
'''
 (BASE/'REPORT.md').write_text(text,encoding='utf-8')
 save(OUT/'capability_summary.json',{'scope':'25 retained trials, one participant, current scene; exploratory','frames':n,'online_body_valid_frames':ov,'dense_body_valid_frames':dv,'question1':'surface points observable, stability varies; localization contributes','question2':'positions and candidate angles observable; anatomical angle/count not validated','question3':'extra movement responses; baseline/semantics limit attribution','hardware_extreme_limit':'not identifiable from current data'})
 print(json.dumps({'report_saved':True,'dense_rate':dv/n,'reviewed_clips':len(reviews),'static_rows':len(static)}))
if __name__=='__main__':main()
