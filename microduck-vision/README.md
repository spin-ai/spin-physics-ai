# microduck-vision（`mvision`）—— 状态：M0/M1 已实现，M2/M3 部分

把「几张照片 → 三维重建 → 结构化感知 → 写回 obs」接成 MicroDuck 仿真里的真实视觉管线，
替换镜像自带的**几何替身**检测器，并用仿真真值量化误差。

**方案全文：[`../docs/MICRODUCK_VISION_RD.md`](../docs/MICRODUCK_VISION_RD.md)**
（需求 / 契约 / 里程碑 / 验收表都在那儿；§8.1 是交付状态，§10 是实测结果）

**实验报告（HTML，单文件、图片与 3D 数据全部内嵌、可直接打印）：`reports/index.html`**
（`reports/` 是**本地产物，不入库**；克隆后按下面「跑起来」一节生成）

报告按四段讲：**① 背景要做什么 → ② 照片是什么 → ③ 重建结果是什么样 → ④ 怎么重建的**，
外加第五段 **⑤ 换成网上的真人照片**（真实数据，结论见下）。
里面有可拖动旋转的 3D 重建过程（① 四视角点云 → ② 去地面 → ③ 掩码+聚类 → ④ 圆柱拟合），
另附俯视布局图、深度图与失败案例。图与数据都可复现：

```bash
./tools/sync.sh python tools/export_figures.py --seed 1 --width 480   # 照片/深度/俯视图
./tools/sync.sh python tools/export_3d.py --seed 1 --limit 3000 \
    --out reports/scene3d.json                                        # 3D 控件的数据
```

截图/定格的 URL 参数：`?stage=0..3&spin=0&yaw=-1&pitch=.55&dist=1.1&hud=0`。

## 它替换的是什么

上游的视觉**不是像素**，是几何替身：

| 文件 | 替身的做法 |
|---|---|
| `sensors/detector.py` | 不渲染，直接算可见性/遮挡/角宽，输出 `{cls, bearing, elevation, width, conf}` |
| `robots/moss_wrist.py` | 直接从 MuJoCo 状态读，输出 `{att, range, top, grip, yaw}` |

这两个文件把真机视觉栈的**接口形状钉死了**。本包让中间真的经过「渲染成照片 →
三维重建 → 感知」，产出同一组字段。控制环、策略、评测全部不动。

## 实测结果（20 episode，4 视角，真实 osmesa 渲染）

| 后端 | 位置 median / p95 | 主轴 median | 顶高 median | 检出率 | 端到端 median |
|---|---|---|---|---|---|
| `oracle`（真值掩码，上限） | 2.1 / 5.6 mm | 3.0° | 1.7 mm | 100% | 1.07 s |
| `color`（品红阈值，零训练零联网） | **0.5 / 5.6 mm** | 3.4° | 1.0 mm | 100% | 1.01 s |
| `none`（无语义，取最大簇） | 3.7 / 10.2 mm | 3.0° | 1.2 mm | 100% | 1.17 s |

验收线是 median 10 mm / p95 25 mm（RD §3.3）—— 全部通过。
`color ≈ oracle` 说明**瓶颈在几何不在语义**；把干扰物加到 4 个后无语义的 `none`
崩到 p95 **114.8 mm**，`color` 稳在 0.4 / 5.4 mm —— 这是"为什么需要语义层"的定量证据。

## 三条不可破的约束（RD §3.2）

1. **零侵入**：不改 `/app/microduck_local` 里的任何文件，只 import。上游源码在
   [`../robots/microduck/`](../robots/microduck/README.md)，只读。
2. **零训练**：只用几何算法 + 可选的现成模型/HTTP API，不训任何权重。
3. **零大依赖**：纯 numpy 几何（不引入 open3d，aarch64 wheel 不可靠），深度模型一律走 HTTP。

## 跑起来

### A. 在现有容器里（最快，已验证）

现有 `microduck` 容器里已经有 osmesa、MOSS 资产（`/data/moss`）和本项目代码副本，
所以不必重建镜像：

```bash
cd microduck-vision
./tools/sync.sh                                   # 同步代码 + 跑 selftest（≈10 s）
./tools/sync.sh eval --episodes 20 --backends oracle,color,none
./tools/sync.sh closedloop
./tools/sync.sh show --ep /data/captures/episodes/ep0000
open reports/index.html                           # 实验报告（含可旋转 3D）
```

3D 控件的数据是单独导出的（`reports/scene3d.json`，65 KB）：

```bash
docker exec microduck sh -c 'cd /data/vision && export MUJOCO_GL=osmesa \
  PYTHONPATH=/data/vision:/app/microduck_local/src && \
  python tools/export_3d.py --seed 1 --limit 3000 --out reports/scene3d.json'
```

等价的手工写法：

```bash
docker cp . microduck:/data/vision
docker exec microduck sh -c 'cd /data/vision && export MUJOCO_GL=osmesa \
  MICRODUCK_MOSS_DIR=/data/moss PYTHONPATH=/data/vision:/app/microduck_local/src && \
  /opt/venv/bin/python -m mvision selftest'
```

### B. 从镜像跑（干净环境 / CI）

```bash
docker compose build          # 基础层复用本地 srayuth089/microduck-docker，不重下 648 MB
docker run --rm -v microduck-data:/data microduck-vision:local \
    python -m mvision selftest
```

镜像里烘了三件事：`libosmesa6`、`git`、以及 MOSS 资产（`/opt/moss`）。
构建时 `fetch-robot moss` 走 raw.githubusercontent.com，逐文件拉 119 次 —— **会抖**，
所以构建脚本重试三次并在装不全时大声警告；运行期的回落顺序是
`$MICRODUCK_MOSS_DIR → /opt/moss → /data/moss`（后者是 named volume，
本机跑过 `fetch-robot moss` 就已经有 121 个文件了）。
网络不好时直接跳过构建期下载：`docker compose build --build-arg FETCH_MOSS=0`。
**资产不能放 `/data`**：`/data` 是 named volume，会把镜像内容整个盖掉。

代码在镜像里备份于 `/opt/mvision-src`（不是 `/data/vision`，同样因为卷会盖掉它），
`PYTHONPATH=/data/vision:/opt/mvision-src:/app/microduck_local/src` —— 挂载的宿主代码优先。

### C. 切换到 compose 常驻（会重建容器）

```bash
docker rm -f microduck                                              # 会丢掉手装的 libosmesa6 —— 正是要换掉它的原因
docker compose up -d --build                                        # container_name 仍是 microduck，旧命令不变
curl -sI localhost:63317 | head -1                                  # 期望 200
```

### 常用子命令

```bash
python -m mvision selftest                     # 投影约定 / 场景 / 一条完整链路
python -m mvision capture --seed 0 --backend color   # 单个 episode，落盘 rgb/depth/mask/overlay
python -m mvision eval --episodes 20 --backends oracle,color,none [--clutter 4] [--pose lying]
python -m mvision closedloop --steps 30        # seam + 时延
python -m mvision show --ep /data/captures/episodes/ep0000
python /data/vision/tools/verify_projection.py # 单独验证 -Z 光轴 / 米制深度
cd /data/vision && python -m unittest discover -s tests -v   # 不需要渲染
```

## 管线

```
  scene.py         MOSS 静态场景：品红目标 + N 个干扰物 + 两个 <camera>
       │           前视 D455(87°×62°) / 腕部 D405(70°×55°)，从 moss.py 的常量生成
       ▼
  camera.py        内参 (fx=fy=W/2/tan(hfov/2))、投影、深度反投影、Renderer 池
       │           MuJoCo 约定已实测：看向 -Z，+X 右、+Y 上，深度是**米**
       ▼
  4 视角渲染       底盘退到 ~0.5 m 拍 3 张（D455 实测最小深度 0.52 m）
       │           + 腕部 home 拍 1 张（真正的抓取视角）
       ▼
  semantics.py     oracle / color / http 三种后端，统一成 (H,W) 掩码
       ▼
  geometry.py      点云 → RANSAC 平面 → 体素并查集聚类 → 圆柱代数拟合
       │           → 共识投票（剔除被毁掉的单帧）→ 六元组
       ▼
  adapters.py      → moss.OBS_TARGET_BASE(24:27) / SEEN(27) / AXIS(28:30)
       │             UPRIGHT(30) / SPARE(31)；或 duck 的 detector payload
       ▼
  evaluate.py      与 GT 比，出 markdown + JSON + 失败样本拼图
```

### 三个"不这样就会被骗"的细节

1. **目标颜色必须是品红**。一开始用橙色，结果前视画面被 MOSS 自己的橙红机械臂占满，
   橙色分割把机器人当成了目标（掩码命中 2/4、位置误差 20 mm）。品红是场景里唯一没有的颜色。
2. **圆柱中心不能取点云质心**，也不能用"点 − r·径向"的迭代：只看到一段窄弧时
   迭代里的 r 被低估，自洽陷阱让直立罐子稳定偏 15 mm（半径 21.6 vs 33.0）。
   改成在其法平面里做代数圆拟合（Kåsa + 剔除），c 和 r 一起解出来，误差掉到 0.4 mm。
3. **顶高由模型给，不由点云分位给**：躺着看时顶面是掠射，`percentile(z,99)`
   比真值低 10 mm。用 `c_z + h·|u_z| + r·(1−|u_z|)` —— 与 `GroundTruth.top_z` 同一个公式。

## 感知契约（唯一需要对齐的东西）

MOSS 32 维 obs（`robots/moss.py` 的 contract v0，**下标从那里读，不在这里抄**）：

| slot | 名字 | 我们写的值 |
|---|---|---|
| 24:27 | `OBS_TARGET_BASE` | 目标质心，rover 系，米 |
| 27 | `OBS_TARGET_SEEN` | 定位新鲜（超过 `MV_STALE_S`=0.25 s 归零） |
| 28:30 | `OBS_TARGET_AXIS` | `cos 2ψ, sin 2ψ`（倍角；直立时置零） |
| 30 | `OBS_TARGET_UPRIGHT` | `|cos tilt|` |
| 31 | `OBS_SPARE` | 顶端高度，米 |

`approach` / `stow` 两个 task 的保留区恒为零 —— 所以 `moss_slots()` 有
`publish_axis / publish_top / publish_upright` 三个开关（默认开、按 task 关）。

microduck（61 维）没有 3D 槽位，视觉走 `detector.Detection.as_payload()`：
`adapters.duck_payload()` 产出同形的 `{cls, bearing, elevation, width, conf}`
（安装帧 x 前 / y 左 / z 上，bearing 左为正）。

## 环境固定了什么（每条都是踩过的坑）

| 内容 | 为什么 |
|---|---|
| `apt install libosmesa6` | 镜像没有 GL context（glfw 要 DISPLAY、无 EGL），不装就 `no OpenGL context` |
| `apt install git` | 镜像里**没有 git**，而代码在宿主版本化、挂载进 `/data/vision` |
| `ENV MUJOCO_GL=osmesa` `PYOPENGL_PLATFORM=osmesa` | 容器内唯一可行的离屏后端（320×240 ≈ 24 ms/帧） |
| `ENV MICRODUCK_RUNS_DIR=/data/runs` | 默认值落在 `/opt/venv/lib/python3.12/runs`，直接 `train-walk` 会 `PermissionError` |
| `ENV MICRODUCK_MOSS_DIR=/opt/moss` | MOSS 119 个资产不在镜像里；**不能放 `/data`**，那是 named volume，会盖掉镜像内容 |
| `volumes: microduck-data { name: microduck-data }` | 不钉名字，compose 会新建 `microduck-vision_microduck-data`，现有训练数据就"看不见"了 |
| `RendererPool` 复用 + `close()` | 每次新建 Renderer 而不 close，第 2 个之后返回**未初始化内存**（随机色块）；同一 Renderer 在 RGB/深度间切换是安全的，但 `update_scene` 必须重调 |

## 时延实测

| 环节 | median | p95 |
|---|---|---|
| 单视角渲染 (320×228, osmesa) | 91 ms | 127 ms |
| 4 视角 + 重建 + 语义（整个感知） | **535 ms** | 736 ms |
| 环境 step（`MossPickEnv`，25 Hz） | 1.10 ms | 1.53 ms |

全部落进 RD 的 2 s 预算（NFR-1），控制环完全不受影响（NFR-2）。

## 真实照片那一路（`mvision/real.py`）

同一套几何，换成 Wikimedia Commons 上的真人照片（单目深度 = Depth Anything V2 Small，CPU 0.6 s/张）：

| | 仿真 | 真实照片 |
|---|---|---|
| 深度 | MuJoCo 米制深度，无噪声 | 单目模型，**只有相对深度** |
| 相机内参 | 视场角精确给定 | 照片不给，只能假定 65° |
| 相机位姿 | 关节角直接给（免费） | 要自己解：本次 901 匹配 → **45 内点（5%）、转角 135° ⇒ 伪解** |
| 尺度 | 世界坐标，米 | 无解，必须外部锚（本次假定身高 ⇒ 身高自证） |
| 目标模型 | 圆柱，6 个数够用 | 人不是圆柱，要 SMPL 类模型 |

点云要看清，得做六件事（都已实现）：**DA-V2 Base**（比 Small 轮廓干净）、视差**双边滤波**
（保边，中值会把腿并成一片）、**手工多边形**掩码（人/树丛/地面在单目深度上是连通的）、
**随机抽样**（等步长会在斜面上形成摩尔纹）、渲染时**法向明暗 + 深度着色 + 自动取景**、
黑白照片**按亮度补增益**。

**上限也要说清楚**：单帧单目只能得到一张 2.5D 浮雕 —— 背面没有，深度的高频噪声会变成条纹。
再往上要走多帧融合、深度相机或人像专用模型（SMPL），不是调渲染能解决的。

控件有两个开关（`深度着色` / `明暗`），也支持 URL 定格：
`?rcloud=toddler1&ryaw=0.3&rdist=3.2&rdepth=1&spin=0`。

结论：**仿真那 0.5 mm 的精度，一半功劳属于"尺度/内参/位姿全都白送"**。
真实照片要能用，缺的是 ① 人像/物体分割 ② SfM 或前馈多视图重建 ③ 尺度锚定。

```bash
./tools/sync.sh python tools/export_real.py --out reports/real   # 4 张照片的深度 + 点云 + 双视角
```

依赖与三个实测坑（torchvision 必须卸、transformers 钉 4.46.3、权重走 hf-mirror）见
[`requirements-real.txt`](requirements-real.txt)。

## 没做什么（诚实清单）

- **没测策略成功率**。镜像里没有一个训练好的 MOSS pick 策略（`/app/policies` 是 duck 的），
  所以 `closedloop` 证明的是 **seam 与实延**，不是"接上新视觉策略还灵"。
  RD §7 R1 的退化评估需要一条训练过的腿。
- **没走纯 RGB（单目）路径**。`semantics.Http` 留了口子，但默认管线吃的是仿真深度；
  只用照片要换成 VGGT/MoGe 那类前馈重建，精度会明显变差（RD §4.4 G2）。
- **没做真机**：手眼标定、尺度锚定、反光/透明物体都不在范围内。
- **仿真比真机干净**：单个方向光、无运动模糊、无卷帘、深度无噪声。
  `color` 后端在这种条件下是"简单模式"，真机上必须换成检测器 + VLM。

## 文件

```
mvision/
  __main__.py    `python -m mvision <cmd>`
  config.py      路径/分辨率/ROI/门限，全部走环境变量
  camera.py      内参、投影、反投影、Renderer 池（含两个 osmesa 坑的注释）
  scene.py       MOSS 场景、视角计划、真值（多视角并到统一参考帧）
  geometry.py    平面/聚类/圆柱拟合/共识重建
  semantics.py   oracle / color / http 后端
  real.py        真实照片适配：单目深度 / 分割 / 测量 / 双视角位姿
  adapters.py    → obs 槽位 / detector payload
  capture.py     episode 采集与落盘
  evaluate.py    指标、汇总、报告、失败拼图
  closedloop.py  gym Wrapper + seam 检查 + 时延
  cli.py         selftest / capture / eval / closedloop / show
tools/
  verify_projection.py   单独验证 MuJoCo 相机约定（-Z 光轴、米制深度）
  export_3d.py           导出 3D 控件的数据（四段点云 + 真值 + 相机 + 拟合）
  export_figures.py      报告里的照片 / 深度图 / 俯视布局图（SVG）
  export_real.py         真实照片那一路：深度 → 点云 → 量身高 → 双视角位姿
  sync.sh                宿主代码同步进运行中的容器
tests/
  test_geometry.py       合成圆柱/平面/聚类（纯 numpy）
  test_adapters.py       契约：槽位、倍角、门限、duck payload
```
