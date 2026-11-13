# syntax=docker/dockerfile:1.7
#
# microduck-vision —— 在社区镜像之上只加三件事：离屏渲染依赖、MOSS 资产、代码约定。
#
# 为什么需要这个文件（而不是 docker commit 一下了事）：
#   1. 镜像自带 MuJoCo 但**没有 OpenGL context**（glfw 要 DISPLAY，无 EGL），
#      离屏渲染必须装 libosmesa6；之前是在运行中的容器里手装的，容器一删就没了。
#   2. 镜像里**没有 git**，代码是烤死在 /app 的副本；我们的代码必须在宿主版本化，
#      靠 compose 把它挂到 /data/vision。
#   3. MOSS 的 119 个资产（20 MB）不在镜像里，`moss.scene_spec()` 会直接
#      FileNotFoundError —— 而容器出网首包要 8 s，不该让 selftest 被网络决定成败。
# 详见 ../docs/MICRODUCK_VISION_RD.md §2 / §4.6。
#
# 上游镜像是社区构建（Apache-2.0 AND CC-BY-NC-SA-4.0，模型权重非商用）；
# 我们只叠加 Debian 包与 MOSS 的 CC BY 4.0 几何资产，不再分发任何权重。

FROM srayuth089/microduck-docker:latest

# 镜像默认 USER duck；装系统包要临时提权，装完立刻降回去。
USER root

# libosmesa6 : MuJoCo 的 osmesa 后端（320x240 ≈ 24 ms/帧，已验证）
# git        : 容器里没有，挂载进来的 /data/vision 需要一个能查 diff/污染的 git
RUN apt-get update \
 && apt-get install -y --no-install-recommends libosmesa6 git \
 && rm -rf /var/lib/apt/lists/*

# 渲染后端：容器内无 DISPLAY，只能走 osmesa（PyOpenGL 也一起对齐）
ENV MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa

# 代码覆盖路径：挂载的 /data/vision 优先，其次镜像内备份 /opt/mvision-src，
# 最后才是镜像自带（非 editable）的 microduck_local 源码。
# 为什么备份放 /opt 而不是 /data/vision：/data 会被 named volume 整个盖掉 ——
# 把代码烤在 /data/vision 等于"有卷就看不见"，而卷里那份很容易是旧的。
ENV PYTHONPATH=/data/vision:/opt/mvision-src:/app/microduck_local/src

# 镜像的 RUNS_DIR 默认落在 /opt/venv/lib/python3.12/runs 且不可写，
# 直接跑 train-walk 会 PermissionError。这里一次性钉死。
ENV MICRODUCK_RUNS_DIR=/data/runs \
    MVISION_DATA=/data/captures \
    MV_DATA=/data/captures

# MOSS 的资产（119 文件 / 20 MB，CC BY 4.0，sha 由上游钉死）。
# 放在 /opt/moss 而不是 /data/moss —— /data 是 named volume，会把镜像里的
# 内容整个盖掉，装在那里等于没装。
# 上游逐文件拉 119 次，raw.githubusercontent.com 会抖 —— 重试三次，每次都
# 会补齐上次缺的文件。**不吞错误**：装不全就大声说出来，别留一个"看起来成功"
# 的镜像让别人在 selftest 里看 FileNotFoundError。
ENV MICRODUCK_MOSS_DIR=/opt/moss
# 网络不好时：docker compose build --build-arg FETCH_MOSS=0
# （运行期会回落到 /data/moss，那个卷里通常已经有资产了）
ARG FETCH_MOSS=1
RUN mkdir -p /opt/moss \
 && ( [ "$FETCH_MOSS" = "1" ] || { echo "FETCH_MOSS=0，跳过 MOSS 资产下载"; exit 0; } ) \
 && ( for i in 1 2 3; do \
        HOME=/root fetch-robot moss && break \
        || echo "== fetch-robot moss 第 $i 次失败，重试 =="; \
      done ) \
 && chown -R duck:duck /opt/moss \
 && N=$(find /opt/moss/model -type f 2>/dev/null | wc -l) \
 && if [ "$N" -lt 100 ]; then \
      echo "!! 构建警告：MOSS 资产只有 $N/119（网络）。microduck 四足不受影响。"; \
      echo "!! MOSS 场景的运行期回落顺序：\$MICRODUCK_MOSS_DIR → /opt/moss → /data/moss"; \
      echo "!! 已经跑过 fetch-robot moss 的容器把资产留在 /data/moss 卷里，compose"; \
      echo "!! 挂载后即可用；否则跑一次：docker exec -u root -e HOME=/root \\"; \
      echo "!!   -e MICRODUCK_MOSS_DIR=/opt/moss microduck fetch-robot moss"; \
    else \
      echo "MOSS 资产就位：$N 个文件"; \
    fi

# 真实照片那一路（可选）：单目深度 + 特征匹配。
#   torchvision 必须卸掉（它是给 torch 2.15 编的，torch 2.9.1 下 import 就崩）
#   transformers 钉 4.46.3（再高会强升 torch，破坏 microduck_local 的 pin）
#   权重走 hf-mirror，HF 直连在这台机器上超时
RUN /opt/venv/bin/pip install -q --no-input -r /opt/mvision-requirements-real.txt \
 && /opt/venv/bin/pip uninstall -y -q torchvision || true

# 代码也进镜像一份，这样 `docker run <img> python -m mvision ...` 不挂载也能跑；
# compose 里 `./:/data/vision` 会盖掉它，开发时改宿主代码即时生效。
COPY requirements-real.txt /opt/mvision-requirements-real.txt
COPY . /opt/mvision-src
RUN chown -R duck:duck /opt/mvision-src

USER duck
WORKDIR /app

# 沿用上游的 entrypoint/CMD（serve）：启动 lab API(8788) + viewer(63317)
