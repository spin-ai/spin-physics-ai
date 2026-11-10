# microduck-vision（`mvision`）—— 状态：环境已就绪（Dockerfile 备好，未切换容器），代码未开工

把「几张照片 → 三维重建 → 结构化感知 → 写回 obs」接成 MicroDuck 仿真里的真实视觉管线，
替换镜像自带的几何替身检测器，并用仿真真值量化误差。

**方案全文：[`../docs/MICRODUCK_VISION_RD.md`](../docs/MICRODUCK_VISION_RD.md)**（需求 / 契约 / 里程碑 / 验收表都在那儿）

## 三条不可破的约束（摘自 RD §3.2）

1. **零侵入**：不改 `/app/microduck_local` 里的任何文件，只 import。上游源码在
   [`../robots/microduck/`](../robots/microduck/README.md)，只读。
2. **零训练**：只用预训练权重 + 现成 ONNX。
3. **零大依赖**：纯 numpy 几何（不引入 open3d），深度模型一律走 HTTP。

## 目标目录（M0 起往里填）

```
microduck-vision/
├── Dockerfile            ✅ 已备好  FROM srayuth089/microduck-docker + libosmesa6 + git
├── docker-compose.yml    ✅ 已备好  挂载 ./:/data/vision，端口同镜像（63317 / 8788）
├── .dockerignore         ✅
├── src/mvision/          ⬜ 待写
│   ├── capture.py        S0 离屏渲染 + 四视角采集（FR-1/2/3）
│   ├── geometry.py       S1 G0/G1 重建：深度→点云→RANSAC 平面→聚类→PCA（FR-4）
│   ├── adapters.py       S2/S3 Perception → obs dict（FR-6）
│   ├── backends/         外部模型可插拔：`http.py` / 本地 / 纯几何（FR-7）
│   └── evaluate.py       误差评测 + 失败样本可视化（FR-8）
├── tests/                ⬜
├── reports/              ⬜ M1.md 误差表、成功率对比
└── data/                 采集产物（git-ignored，对应容器 /data/captures）
```

## 运行约定（RD §4.6）

```bash
# 宿主侧（工作区根目录）
docker compose -f microduck-vision/docker-compose.yml up -d --build

# 容器内
MUJOCO_GL=osmesa PYTHONPATH=/data/vision:/app/microduck_local/src \
  /opt/venv/bin/python -m mvision.capture --scenario moss-demo --episodes 20
```

## 环境固定了什么（四条都是踩过的坑）

| 内容 | 为什么 |
|---|---|
| `apt install libosmesa6` | 镜像没有 GL context（glfw 要 DISPLAY，无 EGL），不装就 `no OpenGL context` |
| `apt install git` | 镜像里**没有 git**，而代码在宿主版本化、挂载进 `/data/vision` |
| `ENV MUJOCO_GL=osmesa` `PYOPENGL_PLATFORM=osmesa` | 容器内唯一可行的离屏后端（320×240 ≈ 24 ms/帧） |
| `ENV MICRODUCK_RUNS_DIR=/data/runs` | 默认值落在 `/opt/venv/lib/python3.12/runs`，直接 `train-walk` 会 `PermissionError` |
| `volumes: microduck-data { name: microduck-data }` | 不钉名字，compose 会新建 `microduck-vision_microduck-data`，现有训练数据就“看不见”了 |

## 切换到 compose（M0 第一步做，现在别做）

```bash
docker rm -f microduck                                              # 会丢掉手装的 libosmesa6 —— 正是要换掉它的原因
docker compose -f microduck-vision/docker-compose.yml up -d --build  # 基础层复用本地 648 MB，不重下
curl -sI localhost:63317 | head -1                                  # 期望 200
```

切换后 `container_name` 仍是 `microduck`，`docker logs -f microduck` 等旧命令不变。

## 待办

- [x] 环境固化文件（`Dockerfile` + `docker-compose.yml`，已 `docker build --check` 通过：no warnings）
- [ ] M0 切换容器 + `mvision/capture.py` + 20 episode 数据集
- [ ] M1 G0/G1 重建 + 误差表（**第一个可判定结论点**）
- [ ] M2 回填 obs + 策略退化对比
- [ ] M3 外部模型后端 + 端到端 ≤ 2 s
- [ ] 未决：主载体 MOSS（有臂+腕相机）还是四足 microduck
