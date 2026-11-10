# syntax=docker/dockerfile:1.7
#
# microduck-vision —— 在社区镜像之上只加两件事：离屏渲染依赖 + 可挂载代码的约定。
#
# 为什么需要这个文件（而不是 docker commit 一下了事）：
#   1. 镜像自带 MuJoCo 但**没有 OpenGL context**（glfw 要 DISPLAY，无 EGL），
#      离屏渲染必须装 libosmesa6；之前是在运行中的容器里手装的，容器一删就没了。
#   2. 镜像里**没有 git**，代码是烤死在 /app 的副本；我们的代码必须在宿主版本化，
#      靠 compose 把它挂到 /data/vision。
# 详见 ../docs/MICRODUCK_VISION_RD.md §2 / §4.6。
#
# 上游镜像是社区构建（Apache-2.0 AND CC-BY-NC-SA-4.0，模型权重非商用）；
# 我们只叠加 Debian 包，不再分发任何权重。

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

# 代码覆盖路径：挂载的 mvision 优先，其次用镜像内的 microduck_local 源码
# （镜像里 /opt/venv 装的是非 editable 副本，PYTHONPATH 会盖过它）
ENV PYTHONPATH=/data/vision:/app/microduck_local/src

# 镜像的 RUNS_DIR 默认落在 /opt/venv/lib/python3.12/runs 且不可写，
# 直接跑 train-walk 会 PermissionError。这里一次性钉死。
ENV MICRODUCK_RUNS_DIR=/data/runs \
    MVISION_DATA=/data/captures

USER duck
WORKDIR /app

# 沿用上游的 entrypoint/CMD（serve）：启动 lab API(8788) + viewer(63317)
