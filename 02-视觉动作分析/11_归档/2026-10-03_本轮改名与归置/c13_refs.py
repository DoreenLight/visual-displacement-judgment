import re,glob,sys
SKIP_DIR=('11_归档','_translation_outputs','_research_run','01_参考论文','.pi','历史归档')
# 目标文件标识 -> (前缀, 本地 ID)
OWN={
 'Anatomy_aware_Region_Encoder.md':('ARE',['F1','F2','F3','F4','F5','F6','F7','R1','R2','R3','R4']),
 'Spatio_Temporal_Encoder.md':('STE',['F1','F2','F3','F4','F5','F6','F7','R1','R2','R3']),
 '01_上肢关键点三维坐标任务包.md':('TP01',['D1','D2','D3','D4','D5','K1','K2','K3','K4']),
 '02_腕关节运动轨迹任务包.md':('TP02',['D1','D2','D3','D4','D5']),
 '03_肩肘代偿识别任务包.md':('TP03',['D1','D2','D3','D4','D5','C1','C2','C3','C4']),
 '04_运动质量评价任务包.md':('TP04',['D1','D2','D3','D4','D5','Q1','Q2','Q3','Q4']),
 'D435i_实验方案与风险清单.md':('D4Q',['R01','R02','R03','R04','R05','R06','R07','R08','R09','R10']),
}
# ID 命名约定与缺口台账必须同时呈现新旧写法，不能改
SKIP_FILE=('06_附录/ID命名约定.md','06_附录/缺口台账.md')
HIST=('编号更正','直接撞号','变更记录','ID 命名空间冲突','C13')
def sub(l,pre,ids):
    for x in ids:
        l=re.compile(r'(?<![A-Za-z0-9_-])'+re.escape(x)+r'(?![0-9A-Za-z_-])').sub('%s-%s'%(pre,x),l)
    return l
APPLY='--apply' in sys.argv
total=0
for f in sorted(glob.glob('**/*.md',recursive=True)):
    if any(s in f for s in SKIP_DIR) or f in SKIP_FILE: continue
    txt=open(f,encoding='utf-8').read().split('\n'); c=0
    for i,l in enumerate(txt):
        if any(h in l for h in HIST): continue
        new=l
        for key,(pre,ids) in OWN.items():
            if key in l: new=sub(new,pre,ids)
        if new!=l:
            txt[i]=new; c+=1
            print('  %s:%d  %s' % (f,i+1,new.strip()[:118]))
    if c:
        total+=c
        if APPLY: open(f,'w',encoding='utf-8').write('\n'.join(txt))
print('跨文件引用改动 %d 行（模式：%s）'%(total,'APPLY' if APPLY else 'DRY-RUN'))
