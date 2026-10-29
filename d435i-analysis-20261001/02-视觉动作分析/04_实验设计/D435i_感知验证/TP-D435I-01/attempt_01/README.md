> **当前数据收尾裁决**：[修订总报告](../closeout_review_v1/review_20261001_v1/MASTER_REVIEW.md)。原始采集完整性已核对；TP-01性能FAIL，TP-02/03/04正式验证BLOCKED。旧离线夹具或初稿的PASS不能替代真人验收。

# TP-D435I-01 上肢关键点三维坐标感知验证工作区 (attempt_01)

> **当前实时入口（2026-10-01）**：[八项训练采集与四任务包工程输出 v3](EIGHT_TASKS_V3.md)。双击 `start_live_feedback_v3.cmd`；原 v2 快捷入口也转到八项版本。右侧选择训练后开始，S1/S2 自动计时 30 秒，其余手动保存，每试次自动生成四包对应数据及工程报告。正式准确度验收仍需独立参考。

> **实时预览新入口（2026-10-01）**：[本地实时动作反馈 v2](LIVE_FEEDBACK_V2.md)，双击 `start_live_feedback_v2.cmd`，浏览器使用 `http://127.0.0.1:8090`。支持持续采集、本地分析、画面内提示、稳定标定及手动录制。腕角为工程估计，不能以本程序完成流程替代四问正式验收。下文 TP-01 既有离线报告保留其原始适用范围。

> **对应任务包**：[01_上肢关键点三维坐标任务包.md](../../../D435i_四问任务包/01_上肢关键点三维坐标任务包.md)  
> **数据契约标准**：`wrist-rehab/v1`  
> **当前验证状态**：`PARTIAL_PASS (ACCURACY_BLOCKED)`（离线算法链路、数据契约与稳定性全通过；实机驱动代码就绪；准确度项依规阻塞）

---

## 1. 目录交付物清单 (Deliverables D1 ~ D5)

```
attempt_01/
├── requirements.txt                 # [D1] Python 依赖版本锁定清单
├── d435i_capture.py                 # [D1] 真实 Intel RealSense D435i 相机流与对齐驱动
├── deprojection_engine.py           # [D1] MediaPipe Pose + 深度中值滤波 + 针孔空间几何反投影引擎
├── synthetic_fixture.py             # [D2] 高保真仿真测试夹具 (模拟真实底模、生理颤抖与测距噪声)
├── run_pipeline.py                  # [D1] 统一执行主程序 (支持 --mode live / fixture / replay)
├── pose_landmarker_lite.task        # [D1] MediaPipe Pose 神经网络权重模型
│
├── run.json                         # [D2] 本次运行的元数据、内参、硬件配置与工况信息
├── keypoints.csv                    # [D3] 严格符合 wrist-rehab/v1 契约的 23 列三维关键点长表
├── visual_summary.png               # [D5] 三维空间散点分布、各轴时序图与滤波平滑对照图
├── verification_summary.json        # [D5] 机器自检生成的逐项验收数据 JSON
│
├── stability_and_error_report.md    # [D4] 详细的分层稳定性、抖动、丢点与边界工况分析报告
├── result.md                        # [D5] 严格按规范编制的任务执行验收报告
└── README.md                        # [D1] 本说明文档
```

---

## 2. 离线复测与验证命令 (当前状态)

在未连接物理相机时，可通过以下命令一键复现全部测试与自动化核验：

```powershell
# 1. 运行流水线生成契约表与可视化
& ".\.venv\Scripts\python.exe" "04_实验设计/D435i_感知验证/TP-D435I-01/attempt_01/run_pipeline.py" `
    --mode fixture --duration_s 10.0 --output_dir "04_实验设计/D435i_感知验证/TP-D435I-01/attempt_01"

# 2. 运行自动化契约检验
& ".\.venv\Scripts\python.exe" "04_实验设计/D435i_感知验证/TP-D435I-01/attempt_01/verify_tp01_contract.py" `
    "04_实验设计/D435i_感知验证/TP-D435I-01/attempt_01"
```

---

## 3. 当你插上 D435i 相机后的实机运行操作指南

一旦将 RealSense D435i 相机通过 **USB 3.0 数据线** 插入电脑，只需以下三步即可直接进行实机人体采集：

### 第一步：检查相机识别
打开 PowerShell 运行：
```powershell
& ".\.venv\Scripts\python.exe" -c "
import pyrealsense2 as rs
ctx = rs.context()
devs = ctx.query_devices()
print(f'已连接设备数量: {len(devs)}')
for d in devs:
    print('设备名称:', d.get_info(rs.camera_info.name), '序列号:', d.get_info(rs.camera_info.serial_number))
"
```
若显示 `Intel RealSense D435i` 及序列号，说明硬件连接完全正常。

### 第二步：执行实机实时采集
运行：
```powershell
& ".\.venv\Scripts\python.exe" "04_实验设计/D435i_感知验证/TP-D435I-01/attempt_01/run_pipeline.py" `
    --mode live --duration_s 30.0 --training_side right `
    --output_dir "04_实验设计/D435i_感知验证/TP-D435I-01/attempt_01_live"
```
程序将自动完成：
1. 启动 `848x480@30fps` 深度流与彩色流并完成硬件对齐；
2. 自动丢弃前 30 帧完成激光散斑与自动曝光预热；
3. 实时提取肩、肘、腕三维空间米制坐标并因果滤波平滑；
4. 输出实机 `keypoints.csv` 与 `run.json`。

### 第三步：运行实机验收检查
```powershell
& ".\.venv\Scripts\python.exe" "04_实验设计/D435i_感知验证/TP-D435I-01/attempt_01/verify_tp01_contract.py" `
    "04_实验设计/D435i_感知验证/TP-D435I-01/attempt_01_live"
```
即可查看在真实环境下的有效率、静止抖动及各项契约达标情况！
