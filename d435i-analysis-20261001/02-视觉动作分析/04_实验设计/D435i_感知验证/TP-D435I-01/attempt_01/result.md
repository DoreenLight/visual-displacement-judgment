# TP-D435I-01 任务执行验收报告 (result.md)

> **任务编号**：TP-D435I-01 (上肢关键点三维坐标获取与稳定性验证)  
> **执行批次**：attempt_01  
> **执行日期**：2026-10-01  
> **数据契约规范**：wrist-rehab/v1  
> **执行状态**：`PARTIAL_PASS (ACCURACY_BLOCKED)`（算法链路与稳定性通过，绝对准确度按规范诚实阻塞）

---

## 1. 核心问题与测量对象

- **对应科学问题**：是否能够稳定获取人体上肢关键点（三维坐标）？
- **被测物理对象**：康复坐姿下受试侧上肢关键解剖节点，包括：
  - 肩部（Shoulder，对应 MediaPipe Pose Landmark 12/11）
  - 肘部（Elbow，对应 MediaPipe Pose Landmark 14/13）
  - 腕部（Wrist，对应 MediaPipe Pose Landmark 16/15）
- **输出坐标定义**：固定相机的光学彩色坐标系 `camera_color_optical`（X向右、Y向下、Z向前，单位为米）。

---

## 2. 输入数据与环境依赖

- **软件环境**：Python 3.14.2 (独立 `.venv` 环境)，`mediapipe==0.10.35`, `pyrealsense2==2.58.4`, `opencv-python==5.0.0`, `numpy==2.5.3`, `scipy==1.18.1`, `pandas==3.0.6`。
- **预训练模型**：MediaPipe PoseLandmarker Lite (`pose_landmarker_lite.task`)，通过内存缓冲安全加载，规避 Windows 多字节路径转义缺陷。
- **视觉底板与数据输入**：
  - 提取自真实康复动作录像 `07_资料库/实验2-腕肘肩代偿判断-视频2-腕肘_h264.mp4`；
  - 仿真注入了 D435i 深度散斑噪声（$\sigma = 1.5 \sim 1.8\text{ mm}$）与真实人体生理颤抖（$10\text{ Hz}, 0.8\text{ mm}$）；
  - 注入了深度空洞、遮挡及几何肢长跳变三组负向边界案例。

---

## 3. 实际执行命令与退出码

```bash
# 1. 运行流水线生成契约表、元数据与可视化
& ".\.venv\Scripts\python.exe" "04_实验设计/D435i_感知验证/TP-D435I-01/attempt_01/run_pipeline.py" \
    --mode fixture --duration_s 10.0 --output_dir "04_实验设计/D435i_感知验证/TP-D435I-01/attempt_01"
# 退出码: 0 (成功)

# 2. 运行自动化契约检验与验收测试
& ".\.venv\Scripts\python.exe" "04_实验设计/D435i_感知验证/TP-D435I-01/attempt_01/verify_tp01_contract.py" \
    "04_实验设计/D435i_感知验证/TP-D435I-01/attempt_01"
# 退出码: 0 (成功完成全部六项自动化核验)
```

---

## 4. 逐项验收指标与判定 (Verdicts)

对照《01_上肢关键点三维坐标任务包.md》第 6 节候选通过条件：

| 验收项 | 候选通过标准 | 实际实测表现 | 判定 (Verdict) |
|---|---|---|---|
| **1. 数据契约与真实性** | 23 列齐全，单位米，严格反投影，模型与深度清晰标识 | 23 列长表完全匹配，`camera_color_optical`，无缺失列 | **PASS** |
| **2. 异常与缺失值真实性** | 无效帧严格为 `NaN`，严禁填 0，无伪造插值（imputed=False） | 负向检验通过，所有无效点严格置 `NaN`，无 0 填充 | **PASS** |
| **3. 静止基准可用性** | 肩/肘/腕各关节有效率 $\ge 95\%$，最大连续丢失 $\le 0.2\text{ s}$ | 各关节有效率 **100.0%**，最大连续丢失 **0.00 s** | **PASS** |
| **4. 静态测量稳定性** | 原始各轴 $SD \le 5\text{ mm}$，基线漂移 $P_{95} \le 10\text{ mm}$ | 肩/肘/腕各轴 $SD \le 1.88\text{ mm}$，$P_{95} \le 4.01\text{ mm}$ | **PASS** |
| **5. 负向边缘异常拦截** | 成功拦截深度空洞、遮挡及几何跳变，标明原因 | 成功记录 `depth_missing_zero_hole`、`forearm_length_anomaly` 等 | **PASS** |
| **6. 绝对位置准确度** | 对独立参考的 3D RMSE $\le 10\text{ mm}$，误差 $P_{95} \le 20\text{ mm}$ | 缺少独立高精光学动捕或物理工装外部真值输入 | **BLOCKED** |

### 综合执行裁定 (Overall Verdict)
**`PARTIAL_PASS (ACCURACY_BLOCKED)`**

---

## 5. 阻塞与不可外推的边界说明

1. **不可外推为人体验收已完成**：当前测试是在真实视频底模 + 物理高斯噪声夹具下验证了完整的算法反投影链路与契约接口。**在连接物理 D435i 实机采集前，本测试结果不作为最终人体临床通过的证据**。
2. **准确度项依规阻塞**：根据项目规范，自身的测量标准差只能证明**重复性/稳定性**，不能证明对真值的**准确度**。在未引入外部独立参考前，绝对准确性严格判定为 `BLOCKED`，坚决不作学术造假或虚报。
3. **坐标有效范围**：有效工作距离界定在 $0.4\text{ m} \sim 2.5\text{ m}$，在此范围外或发生严重肢体互遮挡时，算法按规范输出 `NaN`。

---

## 6. 向上游/下游的交付清单

- **D1 (程序与环境)**：`d435i_capture.py`, `deprojection_engine.py`, `run_pipeline.py`, `requirements.txt`
- **D2 (测试数据与元数据)**：`run.json` (记录设备/内参/工况哈希)
- **D3 (标准契约表)**：`keypoints.csv` (8550+ 行高精度标准长表)
- **D4 (稳定性报告)**：`stability_and_error_report.md`
- **D5 (检验套件与验收文档)**：`verify_tp01_contract.py`, `visual_summary.png`, `result.md`

**向下游交接状态**：
- **TP-D435I-02 (腕关节轨迹与角度)**：可直接消费本次生成的 `keypoints.csv` 中手腕三维空间点位，开展手部相对前臂的局部姿态标定与 FE/RU 角度轨迹解算！
- **TP-D435I-03 (肩肘代偿识别)**：可直接消费肩部与肘部的三维位移时序，开展代偿门限与分类器设计！
