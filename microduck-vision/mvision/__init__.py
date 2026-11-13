"""mvision — 用真实像素替换 MicroDuck 仿真里的几何替身检测器。

分层：
    camera.py    相机内参、投影/反投影、把 <camera> 挂到 MOSS 的安装帧上
    scene.py     MOSS 场景构建、GT 读取、多视角渲染
    geometry.py  深度 → 点云 → 平面/聚类 → 六元组（位置/主轴/upright/顶高）
    semantics.py 语义后端：色块(analytic) / 外部 VLM(http) / 直通(oracle)
    adapters.py  Perception → MOSS 32 维 obs 的 24:32 / duck detector payload
    capture.py   多视角 episode 采集与落盘
    evaluate.py  与 GT 对比、出误差表、失败样本拼图
    closedloop.py 把感知写进环境 obs，验证 seam 与 50 Hz 控制环的 jitter
    cli.py       命令行入口
"""

__version__ = "0.1.0"
