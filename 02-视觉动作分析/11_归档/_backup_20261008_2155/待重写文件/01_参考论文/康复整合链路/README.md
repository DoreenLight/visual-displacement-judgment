# 康复整合链路：本轮新取得的全文

这里保存原文，不是用户下载清单。用户只需查看[仍需补取的论文](../../06_附录/待下载论文总清单.md)。本研究的任务、筛选规则及分析入口在[链路研究专题](../../10_研究方向/主动参与与训练强度_链路研究/README.md)。

## 原件与版本

当前20份原件：12份完整JATS XML、1份机构作者接受稿PDF、7份用户提供IEEE出版PDF；详见[index.json](index.json)的DOI/版本/SHA。XML有正文，不是摘要或翻译；TFS为接受稿，其余7PDF为出版排版。用户仅授权这7篇从01_参考论文根目录移动一次，保留文件名和字节；旧原件保持位置，不复制。55条纳入均已有本地原件，当前无待下载论文；新7篇关键机制与64条原件定位已核，不等于创新定案、独立复现或临床验证完成。

| 原件 | 主要用途 |
|---|---|
| [2021患者协作控制综述](PMC8688994.xml) | 控制分类、适配及slacking |
| [2009机器人运动训练控制综述](PMC2710333.xml) | 辅助、抵抗和控制策略基础 |
| [2025参与感知ALP控制](PMC11885312.xml) | 多模态参与估计→执行时间调整的整合反例 |
| [宋嵘团队2022目标/误差阻抗控制](PMC8927629.xml) | 表现驱动辅助与主动努力的区别 |
| [宋嵘团队2013肌电控制腕康复](PMC3685570.xml) | 连续EMG辅助、抵抗与患者训练先例 |
| [宋嵘团队2025负荷/学习脑活动](PMC11971689.xml) | 健康人机制证据及强度解释边界 |
| [2025参与表现评估综述](PMC12204900.xml) | 测量分类与局限 |
| [2024患者参与评估临床研究](PMC11021326.xml) | 参与评估的患者证据及适用条件 |
| [2026训练时长剂量综述](PMC12989500.xml) | 低确定性剂量收益与重复次数试验的差别 |
| [2026儿科步态多模态参与估计](PMC13523423.xml) | 下肢/儿科迁移方法，不能当成人上肢有效性证据 |
| [2014竞争/合作游戏与IMI心理评价](PMC4021830.xml) | 30健常/8慢性卒中；动机、偏好、自报努力与物理贡献区分 |
| [2022游戏化机器人训练/偏好与运动表现](PMC9708787.xml) | 10慢性卒中；精度/平滑与速度、偏好、学习/迁移分别解释 |
| [2025神经模糊肌骨模型驱动AAN](TFS2025_NeuroFuzzy_AAN.pdf) | 力矩估计→可变阻抗的最近整合反例；[证据卡](TFS2025_NeuroFuzzy_AAN_Evidence_Card.md) |

## 2026-10-04用户补取原件

| ID / 原件 | 入库状态 |
|---|---|
| [Song_TRO](A_Cable-Driven_Upper_Limb_Rehabilitation_Robot_With_Muscle-Synergy-Based_Myoelectric_Controller.pdf) | 13页；运动学参考力/肌协同、8健常及阴性项已定位 |
| [Song_TIE](Deep-LSTM-Based_Human-Robot_Collaboration_Control_With_Static-Dynamic_Calibration.pdf) | 11页；静动态标签/重标定及3参与者边界已定位 |
| [Song_VAAN](Voluntary_Assist-as-Needed_Controller_for_an_Ankle_Power-Assist_Rehabilitation_Robot.pdf) | 9页；传感器MVC、5卒中踝研究及比例/比较边界已定位 |
| [Song_TBME](Performance-Based_Hybrid_Control_of_a_Cable-Driven_Upper-Limb_Rehabilitation_Robot.pdf) | 9页；模糊/限制模式、张力代理及10健常边界已定位 |
| [Song_TBCAS](A_Hybrid_Arm-Hand_Rehabilitation_Robot_With_EMG-Based_Admittance_Controller.pdf) | 11页；单人、反算力标签及手FZCR不显著已定位 |
| [RL2026](Personalized_Adaptive_Assistance_With_Reinforcement_Learning_Control_Enhances_Engagement_Performance_and_Retention_in_Robot-Assisted_Arm-Reaching_Exercises.pdf) | 11页；原图/公式/表已核，符号歧义和旧派生历史保留 |
| [Shoulder2025](Assist-as-Needed_Control_of_a_Shoulder_Rehabilitation_Robot_Using_a_Virtual_Biomechanical_Model_and_Online_Stiffness_Adaptation.pdf) | 10页；期望模型力矩/刚度、10健常及稳定/安全边界已定位 |

[移动/回滚清单](../../05_进展日志/2026-10-02_康复链路研究/七篇原件入库/原件移动与回滚清单.json)；每份出版PDF有同名`_Evidence_Card.md`，记录身份及关键方法/边界选择核验，不伪装独立复现、患者安全或创新定案；[最新64条原件与候选裁剪](../../10_研究方向/主动参与与训练强度_链路研究/补取七篇_原件机制与候选裁剪.md)。

证据卡明确不是全文翻译；新原文入库后更新索引和独立计数，并运行 `bash 06_附录/自检脚本/verify_kb.sh`。不按文件存在就宣布临床证据充分，不把未知人群、未报告安全性或摘要描述补成实验结果。
