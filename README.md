# spin-physics-ai

物理仿真 + 机器人学习的工作区：上游源码、自己的项目代码、大文件数据和跨项目文档**分开放**。

```
spin-physics-ai/
├── README.md              ← 你在这里（工作区地图 / 约定 / 速查）
├── AGENTS.md              ← 给 AI 助手和协作者的硬规则（路径、禁改目录、网络兜底）
├── benchmarks/            ← 第三方基准（不入库，fetch.py 复原）
│   └── LIBERO-plus/           VLA 鲁棒性基准（MuJoCo + robosuite，conda env `liberoplus`）
├── robots/                ← 第三方机器人源码（不入库，fetch.py 复原）
│   └── microduck/             双足机器鸭（Pollen Robotics）
│       ├── microduck-lab/         CPU 训练 + 浏览器 viewer（主工作区，含两个上游 checkout）
│       └── docker/                镜像 `srayuth089/microduck-docker` 的构建上下文
├── microduck-vision/      ← 我们自己的项目代码（MicroDuck 视觉感知）
│   ├── Dockerfile            FROM 社区镜像 + libosmesa6/git，固定 env（已 lint，未切换）
│   └── docker-compose.yml    挂载 ./:/data/vision，卷名前缀已钉住
├── scripts/               ← 复原 + 跑基准的工具（入库）
│   ├── fetch.py              第三方源码 / 权重 / 数据 / 镜像（钉 commit + sha256）
│   ├── run_liberoplus.sh     进 conda `liberoplus` 环境跑任意命令（render 环境变量已设好）
│   ├── smoke_test.py         单任务自检：建环境 → reset → step → 渲染
│   ├── smoke_test_categories.py  7 个扰动维度各渲一张
│   └── demo_video.py         脚本化 pick-and-place（录 MP4 / 开真窗口）
├── docs/                  ← 跨项目文档（RD、安装记录）
└── data/                  ← 大文件 / 数据集（不入库，fetch.py 复原）
    └── libero-plus/          LIBERO-plus 原始 assets.zip（6.4 GB，校验与 HF LFS oid 一致）
```

## 四类目录，四条规矩

| 目录 | 是什么 | 入库？ | 规矩 |
|---|---|---|---|
| `benchmarks/` | 别人的**基准仓库**，要能 `git pull` 跟上上游 | ❌ `/benchmarks/LIBERO-plus/` | 本地只做兼容性改动（`fetch.py` 里逐条「原文→新文」精确替换，幂等），不做无关重构 |
| `robots/` | 别人的**机器人源码**，只读参考 + 被我们 import | ❌ 整个目录 | 固定 commit，由 `fetch.py` 复原；要改就写在 `microduck-vision/` 里覆盖 |
| 顶层项目目录<br>（`microduck-vision/`） | **我们自己的代码** | ✅ | 自己的仓库内容；产物写 `/data` 或 `data/` |
| `data/` | 数据集、压缩包、采集结果 | ❌ | 能重新下载的写明命令（见下） |

新项目就照这个模式加一行：顶层一个项目目录，第三方依赖进 `robots/` 或 `benchmarks/`，数据进 `data/`。

## 复原：一条命令把第三方和大文件拉回来

本仓库只存**我们自己的东西**（文档、`microduck-vision/`、`scripts/`）。克隆到一个新机器后：

```bash
git clone <this-repo> spin-physics-ai && cd spin-physics-ai
./scripts/fetch.py              # 全部：第三方源码 + 权重 + 基准 + 4 GB 数据 + 镜像
./scripts/fetch.py verify       # 只校验（默认只查 size，6.4 GB 重算加 --deep）
```

| 子命令 | 拉什么 | 钉什么 |
|---|---|---|
| `upstream` | 4 个 MicroDuck 上游 checkout | commit SHA（`git clone` 优先，github 不通自动降为 codeload 快照） |
| `policies` | 10 个预训练 ONNX | HF revision + **每个文件的 sha256** |
| `libero` | LIBERO-plus 仓库 + 兼容性改动 + `assets.zip` + 解压 | commit + **assets sha256**（与 HF LFS oid 一致）+ 4 处源码精确替换 |
| `docker` | `srayuth089/microduck-docker` 镜像 | 镜像源 `docker.1panel.live` + retag，大层 short read 自动重试 |

已实测：空目录跑 `--no-git upstream` → 4 个快照落地、sentinel 文件哈希全对（与现树字节一致）。

> 为什么不用 git submodule：这台机器 `github.com:443` 常年不通，submodule 会直接卡死整个 clone；
> 而且上游有 4 个仓库叠加 6.4 GB 数据，子模块反而更难控。

## 环境速查

| 用途 | 怎么进 |
|---|---|
| LIBERO-plus | `./scripts/run_liberoplus.sh <cmd>`（自带 conda `liberoplus` + `MUJOCO_GL=glfw` + `PYTHONPATH`）<br>自检：`./scripts/run_liberoplus.sh python scripts/smoke_test.py`<br>看它动：`./scripts/run_liberoplus.sh python scripts/demo_video.py --live` |
| MicroDuck 容器 | 已在跑：`http://localhost:63317`（viewer）/ `:8788`（lab API）；`docker start microduck` 起，`docker logs -f microduck` 看日志 |
| MicroDuck 本地栈 | `cd robots/microduck/microduck-lab/microduck_local && uv run train-walk …`（需 `uv sync` 一次） |

### 这台机器的网络状况（重要，会浪费很多时间）

| 目标 | 状态 | 兜底 |
|---|---|---|
| `github.com:443`（git clone） | ❌ 基本连不上 | `https://codeload.github.com/<repo>/tar.gz/refs/heads/<branch>`（实测 66 MB / 7 s） |
| `api.github.com`、`raw.githubusercontent.com` | ⚠️ 时通时断 | 加重试，或走 codeload |
| `registry-1.docker.io`（Docker Hub） | ❌ 超时 | daemon 已配镜像源，用 `docker.1panel.live/<image>` 拉完再 `docker tag` |
| `huggingface.co` | ❌ 超时 | `https://hf-mirror.com/...`（同路径替换域名） |
| `pypi.org` / `registry.npmjs.org` | ✅ 通，但首包慢 | — |

细节与踩坑记录见 [`AGENTS.md`](AGENTS.md) 和 [`robots/microduck/README.md`](robots/microduck/README.md)。

## 许可

本仓库自身代码与文档：[`Apache-2.0`](LICENSE)。
第三方组件**不入库**，由 `scripts/fetch.py` 按固定 commit / sha256 复原，各自许可见 [LICENSE](LICENSE) 末尾的汇总：
microduck 系列与 policies 为 Apache-2.0；LIBERO-plus 上游未声明许可证（本仓库不复制其代码，兼容性改动以运行时精确替换实现）；LIBERO-plus assets 为 MIT。

## 测试

```bash
cd microduck-vision && pip install -e ".[dev]" && pytest      # 裸克隆即可，9 passed / 3 skipped
```
需要上游 `microduck_local` 的 3 个用例（MOSS 32 维槽位）在容器外自动 skip。

## 当前状态

| 项目 | 状态 |
|---|---|
| LIBERO-plus | 已装好（conda `liberoplus`，assets 已解压进 `benchmarks/LIBERO-plus/libero/libero/assets`） |
| MicroDuck 容器 | 运行中，9 个预训练策略，容器内 CPU 训练已验证 |
| MicroDuck 源码 | 已 vendored 到 `robots/microduck/`（SHA 见该目录 README） |
| microduck-vision | 环境已就绪（`Dockerfile` + `docker-compose.yml`，lint 通过），**代码待 M0**；切换容器时机见其 README |
