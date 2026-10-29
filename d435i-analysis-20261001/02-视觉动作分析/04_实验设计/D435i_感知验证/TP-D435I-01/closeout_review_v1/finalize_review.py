"""Complete versioned evidence, supersession notices and project navigation."""
import csv,hashlib,json,os,re,shutil,subprocess,sys
from pathlib import Path
import pandas as pd
from review_four_packages import ROOT,BASE,SOURCE,read,save_json,digest,verify_manifest

RID='review_20261001_v1'
OUT={n:BASE/f'TP-D435I-0{n}'/'closeout_review_v1'/RID for n in range(1,5)}
PRIMARY=OUT[1]
REL=f'04_实验设计/D435i_感知验证/TP-D435I-01/closeout_review_v1/{RID}/MASTER_REVIEW.md'

def relative(target,parent):return os.path.relpath(target,parent).replace('\\','/')
def save_snapshot(path):
    target=PRIMARY/'prior_closeout'/path.relative_to(ROOT)
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():shutil.copy2(path,target)
    return target

def main():
    s=read(PRIMARY/'summary.json')
    assert s['retained_trials']==25 and s['captured_rgb_depth_pairs']==29938
    prior=[]
    for name in ('README.md','INDEX.md','HANDOFF.md','STATUS.md'):prior.append(ROOT/name)
    master=ROOT/'04_实验设计'/'D435i_四问任务包'/'D435I_四问任务包全套工程验证与科学总结报告.md'
    prior.append(master)
    for n in range(1,5):
        prior.extend([BASE/f'TP-D435I-0{n}'/f'TP0{n}_SYNTHESIS_REPORT.md',BASE/f'TP-D435I-0{n}'/f'tp0{n}_synthesis_summary.json'])
    prior.extend((BASE/'comprehensive_analysis').glob('*.py'))
    for p in prior:save_snapshot(p)
    snapshots=[p for p in (PRIMARY/'prior_closeout').rglob('*') if p.is_file()]
    save_json(PRIMARY/'prior_snapshot_manifest.json',[{'path':str(p.relative_to(PRIMARY)),'sha256':digest(p)} for p in snapshots])
    docs=ROOT/'04_实验设计'/'D435i_四问任务包';frozen=[]
    for manifest in [docs/'文件清单.sha256',docs/'历史归档'/'原件哈希.sha256']:
        for line in manifest.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():continue
            expected,name=line.split(maxsplit=1);p=(manifest.parent/name.lstrip('*')).resolve()
            actual=digest(p);assert actual==expected,str(p)
            frozen.append({'path':str(p),'sha256':actual,'verdict':'PASS'})
    save_json(PRIMARY/'frozen_contract_verification.json',frozen)
    derived=[]
    for run in read(SOURCE/'eight_task_sessions.json'):
        for n in range(1,5):
            folder=BASE/f'TP-D435I-0{n}'/'eight_task_v3'/run['run_id']
            result=verify_manifest(folder/'files.sha256',folder,True)
            assert not result['issues'],(folder,result)
            derived.append({'run_id':run['run_id'],'package':n,**result})
    save_json(PRIMARY/'derived_manifest_verification.json',derived)
    # Run exactly the existing checker's embedded Python when the native shell is unavailable.
    kb=ROOT/'06_附录'/'自检脚本'/'verify_kb.sh';text=kb.read_text(encoding='utf-8')
    embedded=text.split("<<'PYEOF'",1)[1].split('\nPYEOF',1)[0].lstrip('\n')
    env=dict(os.environ,PYTHONIOENCODING='utf-8')
    proc=subprocess.run([sys.executable,'-c',embedded,str(ROOT)],capture_output=True,text=True,encoding='utf-8',env=env,timeout=60)
    (PRIMARY/'kb_check.log').write_text('Native Git Bash launch returned 49 with no output; existing embedded Python executed unchanged.\n'+proc.stdout+'\n'+proc.stderr,encoding='utf-8')
    save_json(PRIMARY/'kb_execution.json',{'native_shell_exit_code':49,'embedded_python_exit_code':proc.returncode,'checker_sha256':digest(kb),'source_modified':False})
    # Superseded reports keep their original body and byte-preserved snapshots.
    for n in range(1,5):
        p=BASE/f'TP-D435I-0{n}'/f'TP0{n}_SYNTHESIS_REPORT.md'
        body=p.read_text(encoding='utf-8')
        if not body.startswith('> **已被收尾复核版取代'):
            notice=f'> **已被收尾复核版取代，以下为原初稿留证，不能作为现行验收结论。** 初稿有硬编码指标及未经独立参考支持的PASS；现行结论见[修订版]({relative(OUT[n]/"result.md",p.parent)})。初稿原字节已另存校验，不修改原始采集或完成的逐试次结果。\n\n'
            p.write_text(notice+body,encoding='utf-8')
    body=master.read_text(encoding='utf-8')
    if not body.startswith('> **原全PASS结论已撤回'):
        master.write_text(f'> **原全PASS结论已撤回；本文件作为初稿差异留证，不是现行验收报告。** 当前结论及真实指标见[收尾复核总报告]({relative(PRIMARY/"MASTER_REVIEW.md",master.parent)})。初稿原字节和SHA-256见修订版本的 `prior_closeout` 与快照清单。\n\n'+body,encoding='utf-8')
    # Prevent rerunning the legacy generators from reinstating unsupported current reports.
    for p in (BASE/'comprehensive_analysis').glob('*.py'):
        text=p.read_text(encoding='utf-8')
        if 'LEGACY_CLOSEOUT_DISABLED' not in text:
            text=text.replace('if __name__ == "__main__":','if __name__ == "__main__":\n    # LEGACY_CLOSEOUT_DISABLED: prior source preserved in review snapshot.\n    raise SystemExit("Legacy closeout generator disabled: use TP-D435I-01/closeout_review_v1/review_four_packages.py with a new review ID.")')
            p.write_text(text,encoding='utf-8')
    (BASE/'comprehensive_analysis'/'README.md').write_text(f'# 原综合生成脚本：已停用\n\n原脚本包含硬编码PASS、放宽阈值和将指令条件当真值的问题，不能重新生成现行验收报告。原源码按字节留证。请使用[收尾复核]({relative(PRIMARY/"MASTER_REVIEW.md",BASE/"comprehensive_analysis")})及其中复现命令。\n',encoding='utf-8')
    status=ROOT/'STATUS.md';text=status.read_text(encoding='utf-8')
    section=f'''## 0. D435i四问分支状态（采集完整性通过，性能验收未通过）

当前依据：[2026-10-01 收尾复核总报告]({REL})。保留25试次（P001；FE 4次，其余各3次），真实RGB/对齐深度 **29,938对**，分析记录18,310条；4次按用户选择排除。原始及已完成逐试次结果保留。

|任务包|当前复核结论|依据|
|---|---|---|
|TP-D435I-01|**FAIL**；独立准确度BLOCKED|75项可用性检查均FAIL，18项静态波动检查7 PASS/11 FAIL；真实表面深度点不等于解剖中心|
|TP-D435I-02|**BLOCKED**|正式角度有效帧为0，候选角及操作计数不等于独立验证|
|TP-D435I-03|**BLOCKED**|无独立双人标签/事件起点/隔离测试集；不发布指令对照的临床性能|
|TP-D435I-04|**BLOCKED**，观测门控FAIL|0/25窗口满足原95%和0.2s门控；A/C/S正式不评分；仅只读监督回放|

原结题初稿的“四包全PASS”、30,774帧、100%特异度及p<0.001等结论已撤回，旧版留证。59,976个原始清单文件SHA-256通过；输入契约和关联检查通过不等于性能通过。共享任务书与阈值、现行9项和历史3项文件哈希保持原版本；未晋升正式验收状态。

下一步：先处理模型跳帧与观测失效，新增版本离线重处理/实时优化并复测；补独立位置和角度参考、双人事件标注、跨session测试划分及设备时钟映射。不能用插值或调松阈值制造通过。知识库自检执行与未解决的无关项见修订报告，2026-09-30的18/18是历史记录，不代表本次自检。

'''
    text=re.sub(r'## 0\. D435i四问分支状态.*?(?=## 1\.)',lambda m:section,text,flags=re.S)
    text=text.replace('**最后更新**：2026-09-30（D435i四问分包、目录规范、型号修正、归档与入口同步）','**最后更新**：2026-10-01（四问收尾复核与验收结论修正）')
    status.write_text(text,encoding='utf-8')
    handoff=ROOT/'HANDOFF.md';text=handoff.read_text(encoding='utf-8')
    block=f'''## 当前收尾复核：采集证据已核实，四包正式验收未通过（2026-10-01）

- 当前唯一收尾裁决入口：[修订总报告]({REL})；每包另存版本化 `result.md`、CSV、输入/配置/原始文件哈希及真实执行记录。
- 用户保留25试次（单一P001），排除4次已留证；29,938对RGB/深度，18,310条分析记录。59,976个原始清单文件哈希通过，原始资料未改写。
- TP-01 **FAIL**（75项可用性检查均未达原目标，静態波动7 PASS/11 FAIL；准确性BLOCKED）；TP-02/03/04 **BLOCKED**（缺独立角度参考、标签和上游效度；质量观测门控0/25）。
- 原“四包工程指标全部达标、正式验收通过”结论已撤回；旧汇总与源码留证，旧生成入口停用，不能混用。候选角不是解剖ROM，指令不是标签真值，试次开始到首警不是检测延迟；p=0.001345不满足p<0.001。
- 下一步先改善实际观测覆盖，再补参考、独立事件标注、跨session划分、同步及正式质量评价；没有执行器命令，不批准机器人闭环。

'''
    text=re.sub(r'## 最新阶段突破.*?(?=## 实时反馈新增交接)',lambda m:block,text,flags=re.S)
    text=text.replace('status: task-docs-ready-experiments-pending','status: engineering-reviewed-formal-validation-pending').replace('updated: 2026-09-30','updated: 2026-10-01')
    text=text.replace('workspace_path: "/Users/xbpd/Projects/视觉位移判断/02-视觉动作分析"',f'workspace_path: "{ROOT.as_posix()}"')
    handoff.write_text(text,encoding='utf-8')
    intro=ROOT/'README.md';text=intro.read_text(encoding='utf-8')
    text=text.replace('> **状态**：知识库就位；D435i四问任务文档就绪，程序与人体实验未验收',f'> **状态**：D435i真实采集已完成并做收尾复核；性能未验收通过。当前结论见[修订总报告]({REL})。')
    text=text.replace('> **最后更新**：2026-09-30（四问分包、型号修正与相关文件归档）','> **最后更新**：2026-10-01（四问收尾复核，研究主线保留）');intro.write_text(text,encoding='utf-8')
    index=ROOT/'INDEX.md';text=index.read_text(encoding='utf-8')
    text=text.replace('统一数据契约、依赖、版本与交接；文档就绪，实验未执行','冻结的数据契约、原阈值与依赖；不作为当前实验结论')
    marker='### 腕关节康复感知（D435i）— 2026-09-30 新增'
    text=text.replace(marker,marker+f'\n\n当前实验收尾与未达标项：[2026-10-01修订总报告]({REL})。任务文档保持冻结版本，正式状态未晋升。')
    index.write_text(text,encoding='utf-8')
    candidate=SOURCE/'README.md';text=candidate.read_text(encoding='utf-8')
    text=f'> **当前数据收尾裁决**：[修订总报告]({relative(PRIMARY/"MASTER_REVIEW.md",SOURCE)})。原始采集完整性已核对；TP-01性能FAIL，TP-02/03/04正式验证BLOCKED。旧离线夹具或初稿的PASS不能替代真人验收。\n\n'+text
    candidate.write_text(text,encoding='utf-8')
    # Within-participant spread is descriptive; do not claim independent population inference.
    df=pd.read_csv(PRIMARY/'keypoint_statistics.csv');static=df[df.task_id.isin((1,2))]
    repeats=static.groupby(['task_id','joint_id']).agg(trials=('run_id','count'),p95_mean_mm=('position_p95_mm','mean'),p95_sd_between_trials_mm=('position_p95_mm','std'),p95_min_mm=('position_p95_mm','min'),p95_max_mm=('position_p95_mm','max'))
    repeats.to_csv(PRIMARY/'within_P001_repeat_description.csv',encoding='utf-8-sig')
    annotation=OUT[3]/'annotation_template.csv'
    with annotation.open('w',newline='',encoding='utf-8-sig') as f:
        cols=['run_id','annotator_id','event_id','t_onset_host_mono_s','t_offset_host_mono_s','shoulder_label','elbow_label','combined_label','wrist_action_completed','label_reason','reference_video','time_mapping_id','adjudicator_id']
        writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader()
        writer.writerows({'run_id':r['run_id']} for r in read(SOURCE/'eight_task_sessions.json'))
    (OUT[3]/'ANNOTATION_PROTOCOL.md').write_text('# 独立标注补证要求\n\n此模板没有真值，仅给出保留run_id。两名标注者分别观看受限原视频／帧序列，不查看算法报警与诱导条件；标肩/肘自身参与、共同参与、正常腕动作及unknown，记录事件实际起止及时间映射。先冻结定义和裁决规则，再计算一致性（任务包候选Kappa），分歧由第三方裁定；按参与者/session隔离调参与测试。试次名称、操作员点击次数、首个可判读帧均不能自动作为事件真值。当前单P001单session不能伪造独立测试集。\n',encoding='utf-8')
    for n in range(1,5):
        checks={1:[('D1 程序/环境/命令','PARTIAL','候选程序及收尾命令齐；历史采集源码版本恢复范围有限'),('D2 原始/参考/哈希','PARTIAL','原始完整哈希PASS；独立参考BLOCKED，摆位等缺失不补造'),('D3 契约表','PASS','25试次23列、唯一键和帧关联检查'),('D4 稳定性/误差/丢点','PARTIAL','原目标有FAIL；误差无参考BLOCKED'),('D5 测试/对照/结论','PARTIAL','数学/契约边界测试，真实骨架与独立参考对照待补')],
                2:[('D1 有符号角标定','BLOCKED','候选参考保留，前臂轴向姿态未观察'),('D2 位置/角度表','PARTIAL','点位独立有效；正式角度NaN'),('D3 原始/参考/循环','PARTIAL','三类动作都有数据；独立参考/视频计数未提供'),('D4 误差/连续性','BLOCKED','不以候选范围当ROM误差'),('D5 几何/侧别/时间测试','PARTIAL','现有夹具不含完整左右侧与临床几何参考')],
                3:[('D1 定义/基线/冻结阈值','PARTIAL','候选原阈值留证；标注规范已补'),('D2 检测程序','PASS','实际观测及双通道输出可追溯'),('D3 compensation表','PASS','25试次、unknown保留、分通道字段'),('D4 独立标签/划分','BLOCKED','模板不是标签，独立session未提供'),('D5 性能/事件延迟','BLOCKED','不以指令条件或试次首警替代正式指标')],
                4:[('D1 指标字典','PASS','NaN/valid/status及门控定义明确'),('D2 离线/只读回放','PASS','supervisor_replay无执行器命令'),('D3 quality/哈希','PASS','25窗口含not_completed/not_scorable'),('D4 效度/重复性/延迟','PARTIAL','仅P001重复描述，群体推断及端到端延迟BLOCKED'),('D5 正反测试/准入','PARTIAL','8项审查边界测试；机器人接口物理验收未执行')]}[n]
        rows=['# 交付完整性与下一步','\nPARTIAL 是交付覆盖说明，不替代验收PASS/FAIL/BLOCKED。','\n|交付项|覆盖|证据／缺口|','|---|---|---|']+[f'|{a}|{b}|{c}|' for a,b,c in checks]
        (OUT[n]/'deliverable_checklist.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    master_new=PRIMARY/'MASTER_REVIEW.md';text=master_new.read_text(encoding='utf-8')
    text+='\n观测覆盖图三条曲线重合：采集程序的 `valid_for_analysis` 对肩肘腕使用共同上肢几何门控；75项是25×3的检查项，不是75个独立样本。未分析的保存帧是观测缺失，不能等同于原始摄像头文件丢帧。\n\n重复试次的P95均值/离散见 `within_P001_repeat_description.csv`，这是单人重复描述，不是独立群体置信区间或测量真值误差。摆位、参考、健康/患者类型等未确认元信息保持未知；不能追认固定45°布局或健康受试者。\n'
    master_new.write_text(text,encoding='utf-8')
    print(json.dumps({'frozen_files':len(frozen),'derived_manifests':len(derived),'derived_files_checked':sum(d['checked_files'] for d in derived),'kb_exit_code':proc.returncode,'prior_snapshots':len(snapshots)},ensure_ascii=False))

if __name__=='__main__':main()
