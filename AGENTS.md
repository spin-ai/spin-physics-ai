# AGENTS.md —— 在本工作区动手前的硬规则

给 AI 助手 / 协作者。目标：不重复踩坑、不污染上游、不把 20 GB 数据塞进 git。

## 1. 目录归属（先判断你要改的东西属于哪类）

> 本仓库（git）只装**我们自己的东西**：`README/AGENTS/docs/scripts/microduck-vision`。
> `robots/`、`benchmarks/LIBERO-plus/`、`data/` 都在 `.gitignore` 里，由 `./scripts/fetch.py` 复原。
> **不要把第三方仓库或大文件 `git add -f` 进来，也不要改成 submodule**（github 直连不通会卡死 clone）。

| 路径 | 归属 | 能不能改 |
|---|---|---|
| `benchmarks/LIBERO-plus/` | 上游 benchmark（有 `.git`） | 上游**未声明许可证**：不要往本仓库里复制它的代码或 .patch（diff 的上下文行也算）。兼容性改动只能写成 `fetch.py` 的 `COMPAT_EDITS`（原文→新文，幂等），并记进 `docs/LIBERO_PLUS_SETUP.md`；不要重构、不要格式化全文件 |
| `robots/microduck/microduck-lab/microduck/` | Pollen 上游（vendored 快照） | ❌ 只读。要改行为就写 `microduck-vision/` 里的覆盖层 |
| `robots/microduck/microduck-lab/microduck_rl/` | Pollen 上游 MJCF + 官方 CUDA 训练栈 | ❌ 只读（它是仿真模型的唯一来源） |
| `robots/microduck/microduck-lab/` 其余 | 上游 microduck-lab（训练 + viewer） | ⚠️ 默认只读；确需改先在这里开分支并记录 diff 理由 |
| `robots/microduck/docker/` | 镜像构建上下文（vendored） | ❌ 只读；我们要的改动写 `microduck-vision/Dockerfile`（FROM 该镜像） |
| `microduck-vision/` | **我们的代码** | ✅ 随便改 |
| `data/` | 大文件 | ✅ 可删可重下，但**永不 git add** |

## 2. 路径约定（改了就全挂）

- `microduck_rl` 必须是 `microduck-lab/microduck_rl`：`contract.py` 默认
  `Path(__file__).parents[3] / "microduck_rl"`。放到别处就得设 `MICRODUCK_RL_DIR`。
- 预训练策略必须在 `microduck-lab/microduck/policies/*.onnx`（10 个，HF
  `pollen-robotics/microduck-policies` @ `088524a6…`，`velstand` 取自 `main`）。
- 容器里跑训练必须给 `MICRODUCK_RUNS_DIR=/data/runs`，否则 `RUNS_DIR` 落在
  `/opt/venv/lib/python3.12/runs` 并报 `PermissionError`。
- 容器只绑 `127.0.0.1`：viewer `63317`，lab API `8788`（两个都要开，缺 API 页面连不上物理）。

## 3. 网络兜底（这台机器 github/dockerhub/HF 直连都不通）

**优先用现成的：`./scripts/fetch.py [upstream|policies|libero|docker|verify]`** —— 里面已经把下面的兜底都写好了。
只有需要拉清单外的东西时才手打：

```bash
# 下载 GitHub 源码（不要用 git clone，大概率 75 s 超时）
curl -sL --retry 15 --retry-all-errors \
  -o x.tgz "https://codeload.github.com/<owner>/<repo>/tar.gz/refs/heads/<branch>"

# 取上游最新 SHA（api.github.com 时通时断，失败就重试）
curl -s "https://api.github.com/repos/<owner>/<repo>/commits/<branch>"

# HuggingFace 文件（把 huggingface.co 换成 hf-mirror.com）
curl -sL "https://hf-mirror.com/<repo>/resolve/<rev>/<file>"

# Docker 镜像（Hub 直连超时；拉完 retag 回标准名）
docker pull docker.1panel.live/<image>:<tag> && docker tag docker.1panel.live/<image>:<tag> <image>:<tag>
# 大层可能报 short read / unexpected EOF：直接重跑 pull（已完成的层会命中缓存），
# 实测 410 MB 那层重试到第 4 次才过。
```

## 4. 大文件纪律

- `data/libero-plus/assets.zip` 6.4 GB：sha256 已钉在 `scripts/fetch.py`（与 HF LFS oid 一致），已解压到
  `benchmarks/LIBERO-plus/libero/libero/assets`，**验证后可删**（`./scripts/fetch.py libero` 会重下）。
- 训练产物（`runs/`、`captures/`）留在容器 `/data`（volume `microduck-data`）或 `data/`，不进 git。
- 报告目录 `microduck-vision/reports/`（HTML / 图 / 3D 数据 / GIF）**整目录不入库**，
  由 `tools/export_figures.py`、`tools/export_3d.py`、`tools/export_real.py` 生成；
  文档里引用 `reports/...` 的路径只是本地相对路径，克隆后需要先跑生成命令。
- `~/.libero/config.yaml` 指向 `benchmarks/LIBERO-plus/...`。**再移动这个目录必须同步改它。**

## 5. 改了版本怎么同步

1. 改 `scripts/fetch.py` 顶部的清单（`REPOS` / `POLICIES` / `LIERO_ASSETS` 的 commit、sha256）
2. 同步 `robots/microduck/README.md` 的版本表
3. `./scripts/fetch.py verify && ./scripts/fetch.py`

## 6. 提交前自查

```bash
git status --short                         # robots/ benchmarks/ data/ reports/ 不应出现
find . -type f -size +5M -not -path './.git/*' -not -path './data/*' \
  -not -path './robots/*' -not -path './benchmarks/*'   # 应为空
./scripts/fetch.py verify                  # 复原链路 + LIBERO 兼容改动自检
du -sh .                                  # 入库部分应 < 200 KB
(cd microduck-vision && pytest -q)         # 裸克隆也要绿（需要环境的用例自动 skip）
```
