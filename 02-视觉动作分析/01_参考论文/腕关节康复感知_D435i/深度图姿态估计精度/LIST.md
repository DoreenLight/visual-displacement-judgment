# 深度图姿态估计精度

> 子题：RGB-D 深度相机（D435i 等）做人体上肢 3D 关键点 / 运动学的**精度实测**证据
> 归属：`01_参考论文/腕关节康复感知_D435i/`
> 关联回答：任务书问题 1（关键点稳定性）、问题 2（腕轨迹）

## 已入库论文

| # | 文件 | 标题 | 年份 | 来源 | 优先级 | 对应任务书问题 |
|---|---|---|---|---|---|---|
| 1 | `RealSense-Kinematics：单 RGBD 相机实时开源上肢肌肉骨骼分析/RealSense-Kinematics：单 RGBD 相机实时开源上肢肌肉骨骼分析.pdf` | Real-time, accurate, and open source upper-limb musculoskeletal analysis using a single RGBD camera | 2024 | arXiv 2406.10007 | ⭐⭐⭐⭐⭐ | Q1 / Q2 |

**为什么必读**：这是本子题唯一一篇**用同一台 RGB-D 相机（D435i）实测上肢 marker-assisted 运动学精度**的开放论文，报告 3.3±3.9 mm @1 m、60 fps。它提供了「加标记点能到 mm 级」这一关键对照基准，用来标定我们 D435i 实测的 cm 级 markerless 差距究竟来自相机还是模型。

## 关键锚点（供回答文档引用）

- marker-assisted 上肢运动学：**3.3 ± 3.9 mm** @1 m，60 fps
- 对照：markerless 上肢 3D 关键点误差 7–12 cm（本子题另有文献，见主文档证据表）
- 核心结论：**重复性（SD<3 mm）与绝对精度（cm 级）是两件事**

## 未收录进本库的文献（超出本子题范围，仅作背景引用，见主文档「参考来源」）

- 上肢 markerless 3D 关键点误差与重复性实测（OpenPose + RealSense）
- 上肢 markerless 运动学可靠性 ICC 0.25–0.70（D435 派生运动学量）
