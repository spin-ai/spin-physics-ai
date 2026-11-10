# MicroDuck 视觉感知 —— 需求与技术方案（v0.1）

> 目标：在 `microduck-docker` 容器内，把「基于几张照片的三维重建」接成真实视觉管线，
> 替换镜像自带的**几何替身检测器**，并用仿真真值量化误差。
>
> 本文只覆盖**感知**。不涉及真机、不涉及策略训练（见 §1.3 非目标）。

---

## 0. 结论摘要

| 问题 | 结论 |
|---|---|
| 需要训练模型吗 | **不需要**。几何/语义全用预训练或外部 API；控制策略镜像自带 ONNX |
| 容器里能拿到真实照片吗 | **能**。`apt install libosmesa6` + `MUJOCO_GL=osmesa`，实测 320×240 ≈ 24 ms/帧 |
| 重建要产出 mesh 吗 | **不要**。目标 obs 只需要 6 个数（见 §4.2） |
| 工作量 | 单人约 5～7 人日，分 4 个里程碑（§8） |
| 最大价值 | 仿真 GT 可用 → 能算出毫米级误差，这是真机拿不到的 |

---

## 1. 背景与目标

### 1.1 现状

镜像 `srayuth089/microduck-docker:latest`（Debian 12 / aarch64 / Python 3.12）内已具备：

- MuJoCo 3.10 仿真 + 多个机器人本体：**microduck**（四足，头相机 + jaw）、
  **MOSS**（28 cm 履带车 + SO-101 臂 + 平行夹爪 + 腕部相机 + 背篓）、
  **MARS**（双镜头人形）、Asimov、Unitree G1
- 控制策略 ONNX 热插拔，50 Hz；duck 为 61 obs / 14 action，MOSS 为 32 obs / 8 action
- 训练栈（CPU PPO，~6.4k env-steps/s）、评测、viewer（:63317）、lab API（:8788）

**但视觉不是像素，是几何替身**：

| 文件 | 替身行为 |
|---|---|
| `sensors/detector.py` | 不渲染，直接算可见性/遮挡/角宽，输出 `{cls, bearing, elevation, width, conf}` |
| `robots/moss_wrist.py` (`MossTargetSensors`) | 直接从 MuJoCo 状态读，输出 `{att, range, top, z, grip, yaw}` |

这两个文件把**真机视觉栈的接口形状定死了**。本项目的全部工作，就是产出同一组字段，
只不过中间真的经过「渲染成照片 → 外部模型 → 三维重建」。

### 1.2 本项目目标

1. 打通**离屏渲染 → 图片落盘 → 外部模型调用 → 结构化感知 → 写回 obs** 的完整链路
2. 建立**重建精度 vs 仿真真值**的评测方法，输出可复现的误差表
3. 用现成 ONNX 策略跑闭环，量化「换视觉后策略退化多少」

### 1.3 非目标

- ❌ 训练或微调任何视觉模型
- ❌ 训练新控制策略（除非 §7 R1 触发，且届时另立方案）
- ❌ 真机部署、手眼标定、尺度标定（仿真内世界坐标已知，尺度问题不存在）
- ❌ 输出可视化 mesh / 神经渲染
- ❌ 实时性优化（明确不要求低延迟，预算 0.2～2 s/次感知）

---

## 2. 运行环境与实测基线

| 项 | 实测值 | 影响 |
|---|---|---|
| 架构 | linux/aarch64，**原生**（非 qemu） | pip 需 aarch64 wheel |
| CPU | Apple Silicon，无 CUDA | torch 2.9.1+cpu，onnxruntime 1.31 CPU |
| 离屏渲染（原始状态） | **失败**：`no OpenGL context`（glfw 要 DISPLAY，无 EGL） | 必须补 GL |
| 离屏渲染（装 `libosmesa6` 后） | `MUJOCO_GL=osmesa` 通过；320×240 = **24 ms/帧**（含 `mj_forward`）；首次建 context ≈ 0.5 s | 视觉线程可跑 5～10 Hz |
| 容器出网 | `pypi.org` HTTP 200，但首包 ≈ 8 s（慢，可用） | 调外部 API 建议宿主侧代理 |
| 持久化 | `/data` 是 named volume `microduck-data`（`/data/captures`、`/data/runs` 已存在） | 产物写这里 |
| 代码位置 | `/app/microduck_local`（镜像内，只读心态，不改） | 新代码放 `/data/vision` + `PYTHONPATH` |

> **环境副作用（已发生）**：为验证渲染，已在运行中的容器内以 root 执行
> `apt-get install -y libosmesa6`。容器重建后失效，需固化（§4.6）。

---

## 3. 需求

### 3.1 功能需求

| ID | 需求 | 优先级 |
|---|---|---|
| FR-1 | 能以任意相机（`head_camera` / MOSS `wrist` / `front_camera`）在任意 `qpos` 下离屏渲染 RGB（+可选 depth），尺寸可配 | P0 |
| FR-2 | 能在一个 episode 内按脚本移动底盘/臂/头，采集 **≥4 个视角**，每视角记录 `time, qpos, 相机外参, GT 物体位姿` | P0 |
| FR-3 | 数据导出为标准 episode 目录：`rgb/*.png`、`meta.json`（内参/外参/GT/指令） | P0 |
| FR-4 | 几何重建：从 N 张图恢复每个目标物体的 **基座系位置 xyz + 主轴方向 + upright + 顶端高度** | P0 |
| FR-5 | 语义识别：给定文本指令（`"the red can"`），返回目标实例 ID；支持零样本 | P1 |
| FR-6 | 适配层：把感知结果映射到 `moss.OBS_TARGET_BASE(24:27)`、`OBS_TARGET_SEEN(27)`、`OBS_TARGET_AXIS(28:30)`、`OBS_TARGET_UPRIGHT(30)`、`OBS_SPARE(31)` | P0 |
| FR-7 | 外部模型可插拔：同一接口背后可切换 本地模型 / HTTP API / 纯几何基线 | P1 |
| FR-8 | 误差评测：批量跑 N 个 episode（随机物体/位姿/光照），输出误差分布与失败样本可视化 | P0 |

### 3.2 非功能需求

| ID | 需求 | 指标 |
|---|---|---|
| NFR-1 | 单次感知时延 | ≤ 2 s（4 视角 + 1 次模型调用） |
| NFR-2 | 控制环不受影响 | 感知在独立线程/进程，异步写共享状态；50 Hz 控制环 jitter < 1 ms |
| NFR-3 | 可复现 | 固定 seed；同一 episode 重跑误差 < 1e-6 |
| NFR-4 | 零侵入 | **不修改** `/app/microduck_local` 内任何文件，只 import 其 API |
| NFR-5 | 零训练 | 只用预训练权重与现成 ONNX |

### 3.3 验收标准（这是本项目的核心）

以 MOSS pick 场景、腕部相机、无遮挡为标定基线：

| 指标 | 定义 | M1 目标 | M2 目标 |
|---|---|---|---|
| 位置误差 | 重建质心 vs GT 质心，基座系欧氏距离 | 中位数 ≤ 10 mm，P95 ≤ 25 mm | 中位数 ≤ 5 mm |
| 主轴误差 | 重建轴 vs GT 轴，取 `min(θ, π−θ)`（圆柱对称） | 中位数 ≤ 10° | ≤ 5° |
| Upright 误差 | \|cos(tilt)\| 绝对差 | ≤ 0.10 | ≤ 0.05 |
| 顶高误差 | 物体顶端 z vs GT | ≤ 5 mm | ≤ 3 mm |
| 漏检率 | FOV 内无遮挡但 `seen=0` 的比例 | ≤ 5% | ≤ 2% |
| 误检率 | 无目标时 `seen=1` 的比例 | ≤ 5% | ≤ 2% |
| 时效 | 4 视角端到端 | ≤ 2 s | ≤ 2 s |
| 策略退化 | 现成 pick 策略成功率下降 | ≤ 15 个百分点 | ≤ 5 个百分点 |

> 数值依据：MOSS 夹爪为 NormaCore 平行夹爪，抓取容差约数毫米，10 mm 是与内置
> `DETECTOR` 噪声同量级的量级线。M1 先「对齐替身」，M2 才谈超越。

---

## 4. 技术方案

### 4.1 架构

```
                    ┌─────────────────────────── 50 Hz 控制环（不动） ─┐
                    │  policy.onnx (61/14 或 32/8) ← obs ← 环境 step    │
                    └───────────────▲───────────────────────────────────┘
                                    │ obs slots (目标 xyz / axis / upright)
                    ┌───────────────┴───────────┐
                    │  adapters.py              │  把感知 dict 映射到 obs
                    └───────────────▲───────────┘
                                    │ 最新一帧感知（带时间戳 + 新鲜度门限）
┌────────────────────────── 视觉线程（5 Hz，异步）───────────────────────────┐
│  capture.py      采集调度：按脚本移动 → 渲染 N 视角 → 保存 png+meta         │
│  geometry.py     重建：深度/点云 → 分割 → 目标 6 元组                        │
│  semantics.py    语义：检测/分割/VLM（本地或 HTTP）                          │
│  backends/       模型后端：local / http / analytic 三种实现                  │
└────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 感知输出契约（唯一需要对齐的东西）

**MOSS 32 维 obs 中与视觉相关的 7 个 slot**（来自 `robots/moss.py`，逐字对照）：

| slot | 名称 | 含义 | 单位 |
|---|---|---|---|
| 24:27 | `OBS_TARGET_BASE` | 目标在**基座系**的 xyz | m |
| 27 | `OBS_TARGET_SEEN` | 该定位新鲜的标志 | 0/1 |
| 28:30 | `OBS_TARGET_AXIS` | `cos(2ψ), sin(2ψ)`，ψ = 主轴 yaw（**倍角**，避免 θ/θ+π 歧义） | — |
| 30 | `OBS_TARGET_UPRIGHT` | `|cos(tilt)|`，1 直立、0 侧躺 | — |
| 31 | `OBS_SPARE` | 空闲；方案内用于「顶端高度」 | m |

> 注意槽位语义：`OBS_RESERVED(28:32)` 是「任务自己填」的保留区，
> `approach` / `stow` 两个 task 会把它置零。因此**回填必须按 task 开关**，
> 否则会破坏已训练策略的输入分布。

统一内部数据结构（`adapters.py` 的输入/输出）：

```python
@dataclass
class Perception:
    t: float                  # 仿真时刻
    frame: str                # "base" | "tool"
    target_id: int | None
    label: str
    pos: np.ndarray | None    # (3,) 基座系
    axis_yaw_double: tuple | None  # (cos2ψ, sin2ψ)
    upright: float | None
    top_z: float | None
    conf: float
    stale: bool               # True → 下游把 seen 置 0
```

**microduck（四足）侧的契约不同**：`sensors/detector.py` 的 `Detection.as_payload()`
输出 `{cls, bearing, elevation, width, conf}`，异常角，无 3D。
本期以 **MOSS 为主**（有臂、有腕相机、obs 里本就有 3D 槽位），四足侧作为 FR-1 的验证载体。

### 4.3 阶段拆分

**S0 采集与渲染（FR-1/2/3）**

- 渲染入口：`mujoco.Renderer(model, h, w)` + `update_scene(data, camera=...)`
- 四视角策略（MOSS）：`base` 前视 3 个底盘位姿（±y 平移 15 cm）+ `wrist` 下视 1 张
- 每视角记录：`png`、`depth (可选)`、`cam_xpos/cam_xmat`、`qpos`、`data.time`、GT（`xpos/quat` of target body）
- 输出目录：
  ```
  /data/captures/<ep>/view_00/{rgb.png,depth.npy,meta.json}
  /data/captures/<ep>/scene.json      # 相机内外参、GT、指令、seed
  ```

**S1 几何重建（FR-4，核心）**

三档实现，同一接口，先跑通再到精度：

| 档 | 方法 | 用途 |
|---|---|---|
| G0 | **analysis-by-synthesis 基线**：已知 GT 几何 + 单视图反投影 + 平面/圆柱拟合 | 验证坐标系与误差计算是否正确（不对就一定是坐标系错了） |
| G1 | 深度图 → 点云 → RANSAC 平面（地面/桌面）→ 欧式聚类 → PCA 主轴 / 拟合圆柱 | **主力方案**，纯 numpy，无依赖 |
| G2 | 外部模型：VGGT / MoGe-2 / DUSt3R → 点图；SAM 3 / GroundingDINO 出 mask | 只用 RGB 时启用 |

关键实现点（纯 numpy，~200 行，**不要引入 open3d**，aarch64 wheel 不稳）：

1. 深度 → 点云：`p_cam = depth * K^-1 @ [u,v,1]`，再 `T_base_cam @ p_cam`
2. 平面拟合：RANSAC 迭代 200 次，取最大内点集；用于裁剪工作区与重力对齐
3. 聚类：欧式距离 + 并查集（体素 5 mm）
4. 主轴：协方差特征分解 → 最大特征值方向；圆柱对称 → 角度取 `mod π`
5. 顶端高度：内点云 `z` 的 99 分位（对噪声比 max 稳）

**S2 语义（FR-5）**

- 检测：`grounding-dino` 或 VLM 直出（"which object is the red can, return bbox"）
- 分割：SAM 2 / SAM 3，或 G1 的几何聚类直接当 mask（仿真里背景干净，通常够）
- 消歧：把 4 视角里最大的 3 个候选各裁一张图，交给 VLM 选 ID —— 输出**索引而非坐标**（避免幻觉坐标）

**S3 回填与闭环（FR-6/7/8）**

- `adapters.py` 提供 `Perception → obs dict`，带 `publish_axis` / `publish_top` 开关
- 新鲜度门限复用仓库自己的语义：`ARM_STALE_S` / `STALE_S`，超时则 `seen=0`
- 评测脚本：跑 N=50 episode（随机物体形状/位姿/光照 3 档），输出 §3.3 全表 + 失败案例网格图

### 4.4 外部模型清单

| 用途 | 候选 | 输入 | 输出 | 调用 |
|---|---|---|---|---|
| 多视图几何 | **VGGT** / Fast3R / MASt3R-SfM | 4× RGB | 相机位姿 + 深度 + 点图 | 本地 aarch64 跑不动 → HTTP（fal/Replicate/自建 4090） |
| 单目度量深度 | **MoGe-2** / Depth Anything V2 (metric) / Metric3Dv2 | 1× RGB + 焦距 | 度量深度图 | 小模型可本地 CPU（320×240 约 0.5～2 s） |
| 开放词检测 | Grounding DINO 1.5 / DINO-X，或 VLM | RGB + 文本 | bbox + score | HTTP |
| 概念分割 | SAM 3 / SAM 2 | RGB + prompt | mask | HTTP |
| 语义消歧/校验 | 豆包视觉（火山方舟）/ GPT-4o / Qwen2.5-VL | RGB + 指令 | JSON | HTTP（已有 ark 通道） |
| 单物体 mesh（**可选**） | SAM 3D Objects / TRELLIS | RGB | mesh | 仅在需要精确碰撞时 |

> 训练阶段（M1）**默认不接任何外部 API**，只跑 G0/G1，避免网络不确定性干扰误差归因。
> M3 才接 G2 并对比。

### 4.5 调度与时间预算

| 环节 | 预算 | 实测/估计 |
|---|---|---|
| 渲染 4 视角 @320×240 | 100 ms | 24 ms/帧（实测） |
| G1 几何重建（~30k 点） | 100 ms | numpy + RANSAC，估计 |
| 外部模型调用（可选） | 500～1500 ms | 取决于链路 |
| 回填 + 校验 | < 5 ms | — |
| **合计** | **≤ 2 s** | 满足 NFR-1 |

控制环 50 Hz（20 ms/step）**不参与**上述任何环节：视觉线程独立进程，通过
`multiprocessing.Manager` 或 `mmap` 共享最新 `Perception`，控制侧只读。

### 4.6 环境固化（必须做，一次）

三选一，推荐 (b)：

- (a) `docker commit` 现有容器 → 快但不可复现
- (b) **`Dockerfile` FROM 该镜像 + `RUN apt-get install -y libosmesa6`**，并通过
  `-v $(pwd)/microduck-vision:/data/vision` 挂载代码 → 可复现，代码在 git 里
- (c) 渲染放宿主（macOS 原生 `mujoco.cgl`，仓库注释明确支持）→ 零容器改动，但要维护两套环境

镜像内运行约定：

```bash
MUJOCO_GL=osmesa PYTHONPATH=/data/vision:/app/microduck_local/src \
  /opt/venv/bin/python -m mvision.capture --scenario moss-demo --episodes 20
```

---

## 5. 关键决策（ADR）

| # | 决策 | 理由 | 放弃的方案 |
|---|---|---|---|
| D1 | 产出 2.5D 六元组，不做 mesh | 目标 obs 只需要 6 个数；mesh 只在精细碰撞时需要 | 神经重建 mesh |
| D2 | 渲染用容器内 osmesa | 与仿真同进程，能拿到 `qpos`/GT；改一行 apt 就通 | 宿主 CGL（需双环境）、viewer 截屏（浏览器侧，不可控） |
| D3 | 视觉独立异步线程，5 Hz | 24 ms/帧 × 多视角 + 模型调用会吃掉 20 ms 控制预算 | 同步嵌入 step |
| D4 | 先对 GT 后接 API | 误差必须能归因到算法，不能被网络抖动混淆 | 一上来就调外部模型 |
| D5 | 零训练 | 三层都有现成权重/策略；训练会把一周变成一个月 | 训感知网络 / 训 VLA |
| D6 | 回填按 task 开关 | `approach`/`stow` 的 `OBS_RESERVED` 恒为 0，强行填会破坏已训策略 | 无条件全填 |
| D7 | 不 import open3d | aarch64 wheel 不可靠；本尺度纯 numpy 足够 | open3d TSDF |

---

## 6. 评测方法

**对齐规则**（否则误差无意义）：

1. 同一目标：用 GT 的 `body_id` 关联，不靠外观匹配
2. 同一时刻：`perception.t` 与 `data.time` 差 > 1 帧则丢弃（不是插值凑）
3. 坐标系：全部转到 `base` 系再比，禁止在相机系比较
4. 主轴：圆柱/方盒对称 → 角度按 `mod π` 取小；球体 → 该指标记为 N/A
5. 每次结果附 `seed` + 场景 JSON 哈希

**场景矩阵**（每格 ≥ 10 episode）：

| 维度 | 取值 |
|---|---|
| 物体形状 | 圆柱（罐）× 方盒 × 不规则 |
| 目标姿态 | 直立 / 侧躺 / 任意滚转 |
| 光照 | 默认 / 变方向 / 变强度 / 变色温（改 MuJoCo `light`） |
| 遮挡 | 无 / 部分（前向 1 个干扰物）/ 严重（仅 30% 可见） |
| 视角数 | 1 / 2 / 4 |
| 相机 | 腕部（近距）/ 前视（远距） |

输出：§3.3 表格（每格一行）+ `failures/` 网格图（重建点云叠加 GT 线框）。

---

## 7. 风险与对策

| # | 风险 | 概率 | 影响 | 对策 |
|---|---|---|---|---|
| R1 | 换成真实视觉后现成 pick 策略成功率大跌 | 中 | 高 | 先做 §6 度量退化幅度；对策顺序：① 把 `Detector` 噪声参数校准到实测误差后复测 → ② 增大视角数/加先验把误差压回策略容忍带 → ③ 才考虑重训（CPU PPO，分钟级），并另立方案 |
| R2 | aarch64 缺 wheel（scipy/opencv/torch 扩展） | 中 | 中 | 只用 Debian apt + pure-python numpy 路径；深度模型全走 HTTP |
| R3 | 容器出网慢/抖动 | 高 | 中 | M1 不依赖网络；HTTP backend 设 3 s 超时 + 失败回落 G1 |
| R4 | 渲染污染控制环时序 | 中 | 中 | 独立进程 + 固定 5 Hz 节流 + 共享内存；用 `perf_counter` 打点验证 jitter |
| R5 | 仿真太干净 → 结论不可迁移 | 高 | 中 | §6 场景矩阵强制加光照/遮挡/噪声；G1 用足够大的参数扰动测试鲁棒性 |
| R6 | 回填 slot 破坏已训策略 | 中 | 高 | 默认全关，逐 task 打开；每次改动跑仓库自带回归（`tests/`） |
| R7 | 改了镜像内文件导致不可复现 | 中 | 中 | NFR-4 硬约束：`/app` 只读，新代码只在 `/data/vision` |

---

## 8. 里程碑与交付物

| 里程碑 | 内容 | 交付物 | 验收 | 预估 |
|---|---|---|---|---|
| **M0** | 环境固化 + 渲染通路 | `Dockerfile`、`mvision/capture.py`、20 个 episode 数据集 | 4 视角 png + `scene.json`，GT 与渲染对齐可肉眼验证 | 1 人日 |
| **M1** | G0 + G1 重建 + 误差表 | `mvision/geometry.py`、`mvision/evaluate.py`、`reports/M1.md` | 达成 §3.3 M1 全部列 | 2 人日 |
| **M2** | 回填 obs + 闭环 | `mvision/adapters.py`、成功率对比报告 | 策略退化 ≤ 15 pt | 2 人日 |
| **M3** | 外部模型后端 + 端到端 | `mvision/backends/http.py`、端到端 demo 录像 | 4 视角 ≤ 2 s，M2 列精度 | 2 人日 |

**第一个可判定结论点**：M1 结束。届时若「重建误差 ≲ 内置检测器噪声」，则
**训练问题可以永久关闭**（见 §1.3）。

---

## 9. 未决问题

1. **主载体是 MOSS 还是 microduck？** 本文按 MOSS（有臂 + 腕相机 + obs 里本就有 3D 槽位）
   设计；若只想在四足上做「看到球→走过去→踢」，则契约退化为 detector 的
   `{bearing, elevation, width}`，工作量减半。
2. **是否允许 4 视角主动采集？** 若只允许单张固定视角，§3.3 位置误差目标需放宽到 ~25 mm。
3. **外部 API 归属**：自建 GPU 机 vs 现成 SaaS vs 火山方舟（已有通道）——影响 M3 的具体实现。
4. **是否需要 depth buffer**：MuJoCo 可出深度，有深度则 G1 精度大幅提升；若坚持「只用 RGB 照片」，
   G1 要换成 §4.3 的 G2 路径，误差会明显变差。**建议允许仿真内取深度，但把 RGB 通路并行验证。**
