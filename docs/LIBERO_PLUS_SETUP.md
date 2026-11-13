# LIBERO-Plus 本地安装 & 冒烟测试记录（macOS arm64）

## 目录

| 路径 | 说明 |
|---|---|
| `spin-physics-ai/benchmarks/LIBERO-plus` | 仓库（已 `pip install -e .`） |
| `benchmarks/LIBERO-plus/libero/libero/assets` | 官方 assets（9.4 GB，来自 HF `Sylvest/LIBERO-plus` 的 `assets.zip`） |
| `spin-physics-ai/data/libero-plus/assets.zip` | 下载的原始压缩包（6.0 GB，验证无误后可删） |
| `~/.libero/config.yaml` | libero 路径配置（已指向 `benchmarks/LIBERO-plus/...`，再移动目录要同步改它） |

## conda 环境

```bash
conda activate liberoplus        # python 3.10.22
```

入口脚本（入库，路径无关）：

```bash
./scripts/run_liberoplus.sh <cmd>        # 自带 MAGICK_HOME / MUJOCO_GL / PYTHONPATH
./scripts/run_liberoplus.sh bash         # 进一个准备好的 shell
source scripts/run_liberoplus.sh         # 或者 source 进当前 shell（只设环境变量）
```

> 上游是 src 布局（`libero/libero/...`），`pip install -e .` 在这里不会产生 import 映射，
> 所以 `run_liberoplus.sh` 直接把 `$LIBERO_PLUS_DIR` 加进 `PYTHONPATH`；`scripts/*.py`
> 也自己 `sys.path.insert`。换目录位置只需 `LIBERO_PLUS_DIR=/path/to/LIBERO-plus`。

版本：mujoco 2.3.7 · robosuite 1.4.0 · numpy 1.22.4 · torch 2.14.1 · robomimic 0.2.0 · bddl 1.0.1

## 安装过程中对上游做的小改动（仅兼容性问题）

**不再用 `.patch` 文件**：上游仓库没有声明许可证，本仓库不复制、不分发它的源码（diff 的上下文行也算）。
改动改为 `scripts/fetch.py` 里的 `COMPAT_EDITS`（「一行原文 → 一行新文」，幂等），
由 `apply_libero_compat()` 在本地对固定 commit 的文件做精确替换；`./scripts/fetch.py verify` 会检查是否在位。

1. `torch.load(..., weights_only=False)`：torch ≥ 2.6 默认 `weights_only=True`，无法加载
   `*.pruned_init` / `*.init`。涉及 `libero/libero/benchmark/__init__.py`（2 处）、`libero/lifelong/{metric,evaluate}.py`。
2. `Benchmark.get_task_init_states()` 落不到分支会 `UnboundLocalError`（影响 `libero_90`），
   已补 plain 任务（无 `_table_/_tb_/_light_/_add_/_level` 后缀）的路径兜底。
3. robosuite/robomimic 用 `--no-deps` 安装（`egl_probe` 需要 CMake 且只在 Linux 有意义）。
4. `transformers==4.21.1` 与 `usd-core` 未安装（前者在 py3.10/arm64 需编译 Rust，且仅在
   VLA 策略代码里用到；后者仅用于资产转换，benchmark 环境不需要）。
5. `wand`（SegmentationRenderEnv 依赖）需要 ImageMagick：`brew install imagemagick` +
   `export MAGICK_HOME=/opt/homebrew/opt/imagemagick`。

## 试跑

```bash
./scripts/run_liberoplus.sh python scripts/smoke_test.py             # 单任务：建环境/重置/step/渲染
./scripts/run_liberoplus.sh python scripts/smoke_test_categories.py  # 7 个扰动维度各渲一张，拼成 grid
./scripts/run_liberoplus.sh python scripts/demo_video.py --live      # 脚本化 pick-and-place，开真窗口
```

输出（`benchmarks/LIBERO-plus/outputs/`，该目录不入库）：`smoke_agentview.png`、`smoke_wrist.png`、
`categories_grid.png`、`demo_pick.mp4`。

已验证可用：

- benchmark 套件：`libero_spatial`(2402) / `libero_object`(2518) / `libero_goal`(2591) /
  `libero_10`(2519) / `libero_90`(90)
- 7 个扰动维度全部能建环境并渲染（Background Textures / Robot Initial States /
  Camera Viewpoints / Language Instructions / Sensor Noise / Objects Layout / Light Conditions）
- `_view_..._initstate_N` 任务本身没有独立 bddl 文件，属正常：`ControlEnv` 会截断到基础 bddl
  并据此设置相机视角与机器人初始位姿。
- `libero_mix` / `libero_100` 在当前上游版本会 `KeyError`（`task_maps` 只注册了 5 个套件），
  这是仓库自身的 bug，与本机环境无关。

## 接下来（真正评测 VLA 模型）

评测脚本与 LIBERO 一致：把套件名作为 `benchmark_name` 传入你现有的 eval 脚本即可，唯一改动是
`num_trials_per_task` 用 **1**（LIBERO-Plus 每个任务只跑 1 次）。任务 ID ↔ 扰动类别 ↔ 难度
的映射在 `libero/libero/benchmark/task_classification.json`。

注意：这些任务需要 GPU 才能跑 OpenVLA / π₀ 等模型；本机（macOS arm64，无 CUDA）只能跑仿真环境本身。
