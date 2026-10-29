---
type: handoff
handoff_id: HO-20260923-004
status: active
sender: "pi-agent (deepseek-v4.1-flash)"
intended_receiver: "next-agent / human-owner"
created: 2026-09-22
updated: 2026-09-23
state_pointer: "07_本机实机验证/尺度标定报告.md"
workspace_path: "/Users/xbpd/Projects/视觉位移判断/摄像头"
---

# 当前全局交接：首次实验结束（2mm 判据通过）

## 更新摘要（HO-002 → HO-003 → HO-004）

| 里程碑 | 结果 |
|---|---|
| **首次实验结束** | ✅ 2026-09-23 |
| **尺度标定完成** | ✅ 4 点盲测，尺度误差 0.045% |
| **2mm 判据** | ✅ **通过**，2mm 位移误差仅 0.0009mm（余量 30–2222×） |
| 汇报图 | `07_本机实机验证/图表/report.png`（4 子图，已嵌入尺度标定报告） |
| 关键实验突破 | 黑背景（消除红外污染）；白纸展平（消除深度空洞）；S4 算法 |
| 外部审阅 | Sol 提出 7 条纠正，逐条验证：4 成立、1 否证、2 待测 |
| 当前对象 | 3mm 金属杆 + 平整白纸 + 黑背景，距相机 208–221mm |

### 已确认的完整能力

| 能力 | 实测 | 余量 |
|---|---|---|
| 分辨力 (MDR, 3σ) | 0.015–0.061 mm | 33–137× |
| 可复现性（3 次极差） | 0.024 mm | 83× |
| **尺度准确度** | **0.045%** | 2mm 误差 0.0009mm |
| 线性度 (RMSE) | 0.068 mm | 29× |
| R² | 0.99992 | |

### 下一步（已制定计划，未执行）

📋 详见 [`07_本机实机验证/后续实验计划.md`](07_本机实机验证/后续实验计划.md)

**十个未验证缺口，按优先级**：

| # | 缺口 | 级别 | 说明 |
|---|---|---|---|
| 1 | **俯仰串扰** | **P0 最高** | ⚠️ 几何效应：`Z_read = Z_true/cos θ`，**10° 倾斜造成 3.4mm 偏大，超判据 170%** |
| 2 | 横向串扰 | P0 | 颈部不会严格沿 Z 移动 |
| 3 | 跨度上限 | P0 | 目前仅 13mm，牵引行程需 50–100mm |
| 4 | 长时漂移 | P1 | 仅测秒级 |
| 5 | 工作距离上限 | P1 | 仅 221mm |
| 6 | 真实人体 | P1 | 3mm 金属杆 vs 颈部曲面 |
| 7–10 | 温度/运动/多目标/近距离 | P2 | |

**建议最小可行路径**（只做三件事）：
```
第1件 A3 俯仰串扰  ← 最高优先，垫片即可，1-2h
第2件 A2 横向串扰  ← 同装置，1-2h
第3件 A1 跨度扩展  ← 2-3h
```

**⚠️ 一个必须先解决的阻塞问题**：
俯仰串扰是**几何效应**（非噪声，无法靠平均消除）。
若实测确认 10° → 3.4mm 的量级，则必须三选一：
- (a) 机械限位约束倾角 <3°
- (b) 在线估计倾角并做 `Z × cos θ` 修正
- (c) 用 ≥3 个标记点解算平面法线

**工具缺口**：钢尺 ±0.5mm 限制了标定精度。
建议购**塞尺**（约 50 元，可到 ±0.05mm），能把标定精度从 ±2.4% 提到 ±0.2%。

---

## 1. 项目背景与当前阶段

| 项 | 值 |
|---|---|
| 项目 | 视觉位移判断（Intel RealSense D435i Z 轴位移测量） |
| 最终用途 | 人体颈部受牵引力时的移动距离测量（**离线分析**） |
| 当前实验对象 | 直径 3mm、长 122mm 金属杆，**已贴白纸**，位于 20cm |
| 资料库根 | `/Users/xbpd/Projects/视觉位移判断/摄像头/` |
| 当前阶段 | **P1 边界测试完成** —— 静置精度已确认达标，**待做位移标定** |
| 版本控制 | ⚠️ **无 git**。交接依赖本文档的 SHA256 表检测漂移 |

### 环境基线

```
OS            macOS 27.0 (Build 26A428)
CPU           Apple M5 (arm64)
Python        3.12.14  (/opt/anaconda3/envs/realsense/bin/python3)
librealsense  2.58.4   (Homebrew, arm64 native bottle)
pyrealsense2  2.58.3   (conda-forge, osx-arm64)
相机          Intel RealSense D435i
  SDK 序列号  244222076984      ← 标定用这个
  USB 序列号  254843063163      ← ioreg 里显示的是这个，两者不同属正常
  固件        5.15.1.55
  USB 链路    3.2 (5 Gb/s)
  物理端口    1-2-2  (bus=1 addr=2，直插机身雷雳口)

⚠️ macOS 硬限制：IMU 被官方禁用（无法做 VIO/SLAM）；无 GUI viewer；必须 sudo
✅ 已解决：UVC 驱动抢占 interface 0 → 用 macos_usb_fix.ensure_ready()
```

---

## 2. 已完成工作

### 2.1 环境安装（implemented + tested）

1. `brew install librealsense` → 2.58.4，arm64 原生 bottle，含 37 个 `rs-*` 命令行工具。
2. conda 环境 `realsense`：pyrealsense2 2.58.3 + opencv 5.0.0 + pillow + scipy + matplotlib。

### 2.2 攻克 macOS 连通问题（implemented + tested）★核心成果

**问题**：相机能被系统识别（`ioreg` 可见、链路 5 Gb/s），但 librealsense 报
`failed to claim usb interface: 0, error: RS2_USB_STATUS_ACCESS` /
`failed to set power state`，且**间歇性** —— 第 1 次成功，之后持续失败，
并伴随 `mutex lock failed` 崩溃。

**根因**（用 `rs_iface_probe.py` 实测确认）：
macOS 12+ 的 CoreMediaIO 体系（`cameracaptured` / `VDCAssistant`）将 D435i 的
**interface 0**（UVC 视频类，`bInterfaceClass=14`）当作系统摄像头绑定，
AVFoundation 中可见 `RealSense 435i RGB` 被注册为普通摄像头。
libusb 因此无法 claim interface 0，导致整个设备创建失败。

**解法**：在 `import pyrealsense2` **之前**调用 `libusb_detach_kernel_driver()`
解绑系统驱动。解绑后 6 个接口全部可 claim。

### 2.3 实测**无效**的方案（已排除，勿重复尝试）

| 尝试 | 结果 |
|---|---|
| 反复重试 10 轮（间隔 1.5s） | ❌ 全部失败 |
| `libusb_reset_device()` USB 端口复位 | ⚠️ 短暂恢复，`pipeline.start` 随即再失败 |
| 仅加 `sudo` | ❌ 必要但不充分 |
| 换直插雷雳口 | ❌ 本就是直插（独立 XHCI），仍失败 |
| 反复 start/stop pipeline | ❌ 深度跑完 30 帧后彩色报 `No device connected` |

### 2.4 计算并修正了一个可视化 bug

初版深度伪彩色图「看起来全黑」，但统计显示 33.6% 有效、中位数 339mm。
原因是 `min-max(174~20139)` 归一化时，339mm 被压缩到 **0.6/255**，主内容
（200~400mm 的桌面）被 >10m 的离群值拉伸淹没。
已改为 **分位数定区间 + 3m 固定上限**。

### 2.5 ★ 统计方法突破：S4 中值锚定截尾均值

**这是本轮最重要的技术成果。**

#### 问题

3mm 细杆在 20cm 处的实测出现一个巨大矛盾：

| 目标 | 单像素时间 σ | 块级 MDR |
|---|---|---|
| 3mm 杆 @ 228mm | **0.20 mm** ✅ | **25 mm** ❌ |
| 桌面 @ 863mm | 4.14 mm | 1.29 mm |

单像素读数很稳，但块级均值极差 —— 差 125 倍。

#### 两个叠加的假噪声机制

用 `rs_rod_boundary.py` 对比 6 种统计策略定位：

| 机制 | 表现 | 实验证据 |
|---|---|---|
| **一：集合漂移** | 细杆深度逐帧闪烁，被平均的像素集合每帧不同。200 个背景像素（4000mm）足以把 3200 像素的均值拉动 20mm+ | S1 裸有效均值 = **199.7mm** |
| **二：背景复活** | 用「时间中值」填补固定像素集合，会把少量背景像素固化为常驻参与者，它们「复活」时注入 4000mm 的值 | S3 固定像素均值 = **133.0mm**（修了机制一却几乎无改善） |

#### 解法

```python
def measure_s4(d, roi, trim_mm=20.0):
    sub = d[y:y+h, x:x+w].astype(np.float32)
    vv  = sub[(sub > 0) & (sub < 65535)]      # 该帧所有有效像素
    med = np.median(vv)                        # 中值做锚点（抗离群）
    win = vv[np.abs(vv - med) <= trim_mm]      # 排除背景/离群
    return float(win.mean())                   # 对干净窗口求均值
```

**效果：MDR 199.7mm → 0.307mm，改善 651 倍。**

#### 六种策略对比（400 帧，无滤波）

| 策略 | MDR(3σ) | 判定 |
|---|---|---|
| S1 裸有效均值（**此前使用的**） | 199.694 | ❌ |
| S2 裸有效中值 | 2.759 | ⚠️ |
| S3 固定像素均值 | 133.033 | ❌ |
| **S4 中值锚定均值** | **0.710** | ✅ |
| S5 逐像素差分 | 134.231 | ❌ |
| S6 分位中点 | 18.804 | ❌ |

加 temporal α=0.4 后：S4 → **0.334mm**

#### 实现

`rs_measure.py` 已封装为可复用库：

```python
import rs_measure as M
m = M.Measurement(roi=(385, 104, 39, 83), trim_mm=20.0, physical_mm=200.0)
m.calibrate(depth_frames)   # 调零（≥40 帧）
z = m.read(depth_frame)     # 单次读数 (mm)
d = m.displacement(z)       # 相对零点位移
```

### 2.6 另一个真实 bug：`get_supported_options()` 阻塞帧流

在 pipeline 运行期间调用 `sensor.get_supported_options()` 会**阻塞帧流**
（无论是否读取值）：

```
枚举 25 项（不读值） → ❌ Frame didn't arrive within 8000
只读取 exposure/gain/laser → ✅ 仍能出帧
```

**全局约定：pipeline 运行期间禁止调用 `get_supported_options()`。**
需要了解 sensor 能力时用硬编码常量，或先 `stop()` 再枚举。

另：`filter` 实例（temporal_filter）必须**创建一次并复用**，
每帧新建会耗尽资源。

---

## 3. 验证证据（tested，非 human-accepted）

### 3.1 ★ 边界测试最终结果（3mm 杆 @ 20cm）

```
ROI            = (385, 104, 39, 83)   形状=细高（杆），y=104-186
z_ref          = 224.60 mm
像素数          = 1800
MDR(3σ 帧差口径) = 0.3066 mm      ← 主指标
MDR(3σ 散布口径) = 0.3238 mm
σ               = 0.1079 mm
两口径比值       = 1.06             ← 一致，噪声近似独立

要求 2.000 mm → ✅ 达标，余量 6.2×
```

> ⚠️ **一次必须记录的误判**：中途自动定位选中了画面下方的 421px 小块
> （非杆），报出 MDR = 0.020mm，看起来极好但是错的。
> 用 `rs_verify_roi.py` 列块 + 画框标注后才确认杆是 `(385,104,39,83)`。
> **自动定位必须验证 ROI 身份，不能只看数字漂亮。**

### 3.2 三流并发协商结果

```
depth  848x480 @ 30  fx=428.0 fy=428.0 ppx=425.8 ppy=236.8  brown_conrady
color  640x480 @ 30  fx=606.7 fy=606.5 ppx=327.3 ppy=259.0  inverse_brown_conrady
ir     848x480 @ 30  fx=428.0 fy=428.0 ppx=425.8 ppy=236.8  brown_conrady
```

### 3.3 关键参数

```
depth_scale = 0.0010000000474974513   →  1 原始单位 = 1.0000 mm  ★
Emitter Enabled = 1.0    Laser Power = 150.0
曝光 = 8500μs   增益 = 16（采集中稳定）
深度→彩色 外参平移 = 14.76 mm（注意：单位是米，文档中勿再写 0.015mm）
```

### 3.4 稳定运行验证

```
连续采集 15.0s → 480 帧集 / 32.0 fps
  ✅ depth 450/450 (100%)   ✅ color 450/450 (100%)

边界测试 400 帧 → 13.3s（30 fps）
  ✅ 无丢帧，曝光/增益/帧号全程稳定
```

### 3.5 未验证 / 不可用项

| 项 | 状态 |
|---|---|
| **位移准确度（关键）** | ❌ **未验证** —— 0.307mm 是重复性，非准确度 |
| 大位移下的 S4 窗口行为 | ❌ 未验证（±20mm 边界） |
| 距离扫描（150–1000mm） | ❌ 未做（旧数据部分作废） |
| 表面材质对比 | ❌ 未做（仅发现线索） |
| 多相机硬件同步 | ❌ 未测试 |
| 相机内参标定 | ❌ 未做（仅用出厂内参） |
| IMU | ❌ macOS 官方禁用 |

## 4. 关键文件与状态对照表

SHA256 取前 16 位，用于检测文件漂移。

### 4.1 本机实机验证目录（完整清单）

#### 文档

| 文件 | SHA256(16) | 状态 | 说明 |
|---|---|---|---|
| `07_本机实机验证/尺度标定报告.md` | `c890f3ef22dd018c` | tested | ⭐ **2mm 判据通过（余量 30–2222×）** |
| `07_本机实机验证/后续实验计划.md` | `314fe93a0767fbae` | tested | ⭐ 十个未验证缺口 + 俯仰串扰几何分析 |
| `07_本机实机验证/杆边界测试报告.md` | `48efbe33a3797698` | tested | S4 算法与 6 策略对比；MDR 0.307mm |
| `07_本机实机验证/Sol咨询与验证记录.md` | `74ced52bf26c878f` | tested | 外部审阅 7 条纠正逐条验证 |
| `07_本机实机验证/16倍异常根因报告.md` | `2ed01a6e7b66489b` | tested | 异常排查全过程（API 阻塞帧流） |
| `07_本机实机验证/macOS环境与验证.md` | `aabb498451c5671c` | tested | brew/conda 安装实测 |
| `07_本机实机验证/macOS连通问题与修复.md` | `47b7778153ac069a` | tested | UVC 驱动抢占 interface 0 |
| `07_本机实机验证/Z轴位移测量方案_v1.md` | `f36ed2f7abe3afcb` | ⚠️ deprecated | ⚠️ 已作废（ROI 方法错误） |
| `07_本机实机验证/扰动极限测试计划_v3.md` | `c7f0e9214d9c47fa` | ⚠️ deprecated | ⚠️ 已被后续实验计划取代 |
| `07_本机实机验证/颈部位移测量方案_v2.md` | `269c7c86d8dad07e` | ⚠️ deprecated | ⚠️ 需求已变更（对象改为金属杆） |
| `07_本机实机验证/实测数据汇总.md` | `e7960c2bc11ca0ac` | ⚠️ deprecated | ⚠️ 部分作废（见文首声明） |

#### 08 实时数控展示台 (作业汇报版)

| 文件 | SHA256(16) | 状态 | 说明 |
|---|---|---|---|
| `08_实时数控展示台/app.py` | `9b5a9c0971ae4793` | tested | **展示台主程序** (2x2布局/三流采集/S4解算/API) |
| `08_实时数控展示台/export_excel.py` | `1c6ec7b1760041a5` | tested | **Excel 报表生成器** (含 (x,z) 坐标与嵌入折线图) |
| `08_实时数控展示台/启动展示台.command` | `7525d154918ce318` | tested | **Finder 双击一键启动** (自动提权+唤起浏览器) |
| `08_实时数控展示台/README.md` | `ac8bbe0a6f7fd36e` | tested | **展示台操作说明与实验报告** |
| `08_实时数控展示台/界面展示效果.png` | `f52cf37c1d2fd349` | tested | **2x2 实测界面全景截图** (符合草图原型) |
| `08_实时数控展示台/示例导出数据.xlsx` | — | tested | 实测录制并导出的标准 Excel 样例 |

#### 核心库与常用脚本

| 文件 | SHA256(16) | 说明 |
|---|---|---|
| `07_本机实机验证/rs_measure.py` | `0f05781aea13012c` | ⭐ **核心库**：S4 + 调零 + 坐标校验 |
| `07_本机实机验证/rs_locate.py` | `7abff359d9220737` | 定位杆 + 白纸质量评估 |
| `07_本机实机验证/rs_blind.py` | `e94bfdaf6b2fa513` | 盲测记录 + 拟合 |
| `07_本机实机验证/macos_usb_fix.py` | `37f508b72d195dcd` | USB 修复（UVC 抢占，必须先于 pyrealsense2） |
| `07_本机实机验证/图表/report.png` | `8a2df2dfa93305c0` | ⭐ 汇报图 |

#### 数据文件

| 文件 | SHA256(16) | 说明 |
|---|---|---|
| `07_本机实机验证/roi.json` | `f3bbd350bfb319ad` | 当前有效 ROI |
| `07_本机实机验证/blind_calib.json` | `cdb33c35f03d66ca` | 标定数据与拟合结果 |
| `07_本机实机验证/disp_cal.json` | `0c210fc451113dd0` | 旧产物（ROI 已失效） |
| `07_本机实机验证/zero_20cm.json` | `2863db9038398b96` | 旧产物（ROI 已失效） |

#### 其余脚本（历史探索，按需参考）

| 文件 | 用途 |
|---|---|
| `rs_calib_step.py` | 步进式标定 |
| `rs_scale.py` | 尺度标定（真值直接输入） |
| `rs_rod_boundary.py` | 6 策略对比 |
| `rs_verify_roi.py` | ROI 身份验证 |
| `rs_verify_s4.py` | S4 端到端验证 |
| `rs_live_view.py` | 实时预览 |
| `rs_zero.py` | 调零工具 |
| `rs_displace.py` | 双目标测量 |
| `rs_calib_small.py` | 交互式标定 |
| `rs_calibrate.py` | 位移标定（旧） |
| `rs_block_precision.py` | 深度一致块精度扫描 |
| `rs_mdr_validity.py` | MDR 估计有效性 |
| `rs_anomaly_probe.py` | 异常诊断（三套逻辑交叉验证） |
| `rs_mdr_probe.py` | MDR/共模/相关性探测 |
| `rs_pollution_ab.py` | A/B 对照实验 |
| `rs_precision_probe.py` | 精度基线探测 |
| `rs_resolution_curve.py` | 细粒度分辨力曲线 |
| `rs_rod_test.py` | 细杆可用性专项 |
| `rs_neck_probe.py` | 颈部探测（未用） |
| `rs_filter_compare.py` | 14 种 SDK 配置对比 |
| `rs_iface_probe.py` | USB 接口占用探测 |
| `rs_stream_test.py` | 多流并发测试 |
| `rs_capture.py` | 单帧采集 |
| `rs_check.py` | 兼容性自检 |
| `rs_usb_reset.py` | USB 复位 |

> 注：其余脚本均为实验过程产出的探测/验证工具，SHA 见文件本身，未逐一登记。

### 4.2 知识库主体（其他会话产出）

| 文件 | 状态 | 本次改动 |
|---|---|---|
| `INDEX.md` | active | ✅ 新增最新结果与核心库用法 |
| `README.md` | active | — |
| `02_SDK与编程/SDK安装.md` | active | ✅ 修正 macOS 从「源码编译」为 brew |
| `05_校准/IMU校准.md` | ⚠️ | **本机不可用**（macOS 禁用 IMU） |
| （其余 00~06 共 19 个 md） | active | 未改动 |

### 4.3 未改动文件清单

```
00_产品概述/应用场景.md            e715a17b34b20e9b
00_产品概述/坐标系与外参.md         6fa08d0de41af9f7
00_产品概述/产品定位与规格.md       dc617ea66a7a4904
01_硬件详解/相机模组.md             6ad736f0ce36bcc8
01_硬件详解/多相机同步.md           862767916c9fa175
01_硬件详解/IMU_BMI055.md           2ae36fc533f5bd6d
02_SDK与编程/示例代码.md            7e1ea3a62cdbf3b4
02_SDK与编程/C++_API.md             4aa6f1e52ab3551f
02_SDK与编程/Python_API.md          a6d1924ad8d3c9dd
03_ROS集成/常用配置.md              978e1e3f06a07618
03_ROS集成/话题与参数.md            e6570513459ff245
03_ROS集成/ROS2_Wrapper.md          10ae5c5f6dc71e2a
04_应用场景/物体检测.md             5dd78e973a8ba4e1
04_应用场景/点云与测距.md           b43160bccba0f6e4
04_应用场景/多相机阵列.md           d9c4f3a27cf9e2cb
04_应用场景/SLAM建图.md             02be648ea6e49ba8
05_校准/内外参标定.md               2c5704fdb4f23928
05_校准/IMU校准.md                  bec16b1ad628dc5d
06_参考资料/社区资源.md             afec4e1e290bc413
06_参考资料/官方文档汇总.md         d7e0d8642a835e88
```

### 4.4 样例数据

```
07_本机实机验证/captures/frame01_{color.png,depth.npy,depth.png,ir.png}
07_本机实机验证/captures/frame02_{color.png,depth.npy,depth.png,ir.png}
```
深度为 align 到彩色后的 480x640 uint16（毫米）。

---

## 5. 接手者操作指南

### 5.1 环境自检（第一条命令）

```bash
cd /Users/xbpd/Projects/视觉位移判断/摄像头/07_本机实机验证

# 检查环境是否还在
/opt/anaconda3/envs/realsense/bin/python3 -c "import pyrealsense2 as rs; print('OK', rs.__version__ if hasattr(rs,'__version__') else '')"
brew list --versions librealsense
```

### 5.2 标准运行流程

```bash
cd /Users/xbpd/Projects/视觉位移判断/摄像头/07_本机实机验证

# ① 主测试（内置 USB 修复，三流并发）
sudo -E /opt/anaconda3/envs/realsense/bin/python3 rs_stream_test.py

# ② 稳定性测试
sudo -E /opt/anaconda3/envs/realsense/bin/python3 rs_stream_test.py \
    --seconds 15 --streams depth,color

# ③ 采集一帧落盘
sudo -E /opt/anaconda3/envs/realsense/bin/python3 rs_capture.py --out captures/frame03

# ④ 遇到 USB 报错时，先诊断
sudo -E /opt/anaconda3/envs/realsense/bin/python3 rs_iface_probe.py --detach
```

> **非交互环境**（自动化脚本、Agent 调用）下 `sudo` 无法输密码，改用系统授权弹窗：
> ```bash
> osascript -e 'do shell script "<命令> > /tmp/out.txt 2>&1; echo done" \
>   with administrator privileges'
> ```

### 5.3 在自己的代码里集成（★必读）

```python
import sys
sys.path.insert(0, "/Users/xbpd/Projects/视觉位移判断/摄像头/07_本机实机验证")

import macos_usb_fix
macos_usb_fix.ensure_ready()          # ★ 必须在 import pyrealsense2 之前

import pyrealsense2 as rs
import rs_measure as M                 # ★ 必须用这个库，不要自己写统计

# 测量器（roi 从 rs_live_view 点击获取，或 rs_measure.find_target_block 自动定位）
m = M.Measurement(roi=(385, 104, 39, 83), trim_mm=20.0, physical_mm=200.0)
m.calibrate(depth_frames)             # 调零，需 ≥40 帧静置数据
z = m.read(depth_frame)               # 单次读数 (mm, float)
d = m.displacement(z)                 # 相对零点位移
```

### 5.4 四条硬约束（违反会浪费时间）

1. **必须 `sudo -E`**。解绑驱动后非 root 依然看不到设备 —— macOS 硬限制。

2. **绝不能用「按深度值筛选像素」建 ROI。** 这会拿到散布在画面各处的
   深度碎片（实测空间 σ 达 58mm），逐帧平均产生假噪声，
   可把 MDR 从 0.3mm 虚高到 **199.7mm**。
   → **必须用 `rs_measure.measure_s4()`**。

3. **必须用单 pipeline 多流并发**。macOS 上反复 start/stop 会让 SDK
   上下文失效，报 `No device connected`。

4. **filter 实例必须创建一次并复用**，每帧新建会耗尽资源。
   **且 pipeline 运行期间禁止调用 `get_supported_options()`**（会阻塞帧流）。

---

## 6. 已知限制与风险

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| R1 | **准确度从未标定** | 0.307mm 是**重复性**，不是准确度 | 需精密平移台做位移标定（下一步） |
| R2 | **39% 像素不稳定** | S4 只是「绕过」闪烁，未解决根因 | 位移量接近 ±20mm 窗口时需重新验证 |
| R3 | **无 git 仓库** | 无法回溯历史 | 建议尽快 `git init` |
| R4 | macOS 禁用 IMU | 无法做 VIO/SLAM | 如需换 Ubuntu |
| R5 | 解绑驱动后其他 App 无法当摄像头用 | 影响 FaceTime/Zoom | 物理拔插相机即恢复；或 `--reattach` |
| R6 | 人体不自主运动（1–3mm） | 颈部实测时可能与目标信号同量级 | Sol 建议：刚性 marker + event-locked averaging |
| R7 | 尺度系数 k=0.943 偏离 1 | 可能是真实系统偏差，也可能是钢尺量测误差 | 需用平移台交叉验证 |
| R8 | 中值锚定窗口（±20mm）可能排除真实位移 | 大位移时失效 | 需自适应窗口或改用参考物差分 |

---

## 7. 待完成工作与当前唯一下一步

### 现状

**静置精度已确认达标**（MDR 0.307mm vs 要求 2mm，余量 6.2×），
但**尚未验证「真实位移能否被正确测量」**。

### 唯一下一步

**做位移标定：验证「棍移动 2mm，读数是否也变化 2mm」。**

| 有精密平移台 | 无平移台（降级） |
|---|---|
| 步进 0.2/0.5/1/2/5/10/20mm，每点重复 20 次 | 用塞尺/已知厚度垫片叠加 |
| 可测绝对下限 | 最小步进约 0.5–1mm |
| 可测迟滞（正反行程） | 无法测迟滞 |

**必答指标**：
- 线性度（拟合残差）
- 尺度误差（真实 50mm 时读数是多少）
- 迟滞（正反向同位置差）
- **P_detect 曲线**（能可靠检出 2mm 的概率）

### 后续队列（部分依赖 Sol 的建议）

1. **位移标定**（P0，下一步）
2. **验证 ±20mm 窗口边界**（P0）—— 位移接近窗口宽时 S4 是否失效
3. **距离扫描** 150–1000mm + **表面材质对比**（P1）
   （本次发现材质影响可能大于距离：杆 0.20mm vs 桌面 4.14mm 单像素σ）
4. **人体测量载体设计**（P1，Sol 建议）—— 刚性 marker vs 皮肤 ROI
5. **人体无牵引基线**（P1，Sol 建议）—— 定量自然运动
6. **牵引同步实验**（P2，Sol 建议）—— event-locked averaging

### 阶段重排建议（采纳 Sol 的意见）

Sol 明确要求：
> 「**阶段 3 绝对不要真人做『标定』。人不能提供 ground truth。**
> 刚性替代靶标应成为**相机子系统的正式验收工具**。」
> 「**下一轮实验应从『测噪声』转向『输入已知位移，看输出是否正确』。**」

原 `扰动极限测试计划_v3.md` 的阶段顺序需按此重排。

## 8. 知识复利状态

| 项 | 状态 |
|---|---|
| 项目知识库（`05_研究资产/wiki/`） | ❌ 未初始化（本库为资料库结构，非 R-ICL 科研档案结构） |
| 个人知识库（`~/.llm-wiki/`） | ❌ 不存在 |
| 任务复盘（retro） | ❌ 未沉淀 |

> 本次关键洞察已就地写入
> [`07_本机实机验证/macOS连通问题与修复.md`](07_本机实机验证/macOS连通问题与修复.md)，
> 内容包括：UVC 驱动抢占接口的根因分析、无效方案清单、深度可视化归一化陷阱。
> 若后续启用 `~/.agents/skills/project-init-handoff` 的 Wiki 层，
> 可将该文提炼为 `retro` 与 `skill` 页面（个人知识库「macOS USB 设备被系统驱动抢占」是可跨工程复用的通用范式）。

---

## 9. 验收与确认

按六级状态词分离协议，当前状态为：

- `macos_usb_fix.py` 及配套脚本：**tested**（自动化运行通过，实测数据见 §3）
- 环境安装：**tested**
- 文档：**implemented**
- 整机可用性：**implemented + tested**，⏳ **尚未 human-accepted**
- 位移测量能力：**未开始**

> ⚠️ **警示**：脚本退出码为 0、文件存在、测试通过，
> **均不等于 human-accepted（人类验收）**。
> 需由 human-owner 亲自插上相机、看到实时画面、确认符合预期后方可升级状态。

### 交接确认

- receiver status：待确认
- received at：—
- rejection reason：—
