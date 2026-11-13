"""评测：重建 vs 仿真真值，出表格 + 报告 + 失败样本拼图。

对齐规则（否则误差没有意义）：
  1. 同一目标：场景里只有一个橙色目标，`gt()` 与 `perception` 都指向它；
  2. 同一时刻：perception.t 取最后一帧的 `data.time`，与 GT 同一次 forward；
  3. 坐标系：全部在 rover 系比较，绝不在相机系比；
  4. 主轴：圆柱对称 → 角度按 mod π 取小；竖直目标的主轴方位角是噪声 → 记 N/A；
  5. 每个数字附带 seed，`perception.json` 与 `scene.json` 都在 episode 目录里。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config
from .capture import capture_episode

#: 验收线，抄自 MICRODUCK_VISION_RD.md §3.3
ACCEPT = {
    "pos_err_mm": {"median": 10.0, "p95": 25.0, "name": "位置误差 (mm)"},
    "axis_err_deg": {"median": 10.0, "p95": 25.0, "name": "主轴误差 (deg)"},
    "upright_err": {"median": 0.10, "p95": 0.20, "name": "upright 误差"},
    "top_err_mm": {"median": 5.0, "p95": 12.0, "name": "顶高误差 (mm)"},
}


@dataclass
class EpisodeMetrics:
    seed: int
    backend: str
    target_pose: str
    detected: bool
    pos_err_mm: float | None = None
    axis_err_deg: float | None = None
    upright_err: float | None = None
    top_err_mm: float | None = None
    n_points: int = 0
    timings: dict = field(default_factory=dict)
    dir: str = ""

    def to_dict(self) -> dict:
        return {**self.__dict__}


def _axis_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """圆柱/方盒都有 180° 对称，所以取 mod π 的小角。"""
    c = abs(float(np.dot(a, b))) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9)
    return math.degrees(math.acos(min(1.0, c)))


def evaluate_episode(ep: dict) -> EpisodeMetrics:
    gt = ep["gt"]
    perc = ep["perception"]
    m = EpisodeMetrics(seed=ep["seed"], backend=ep["backend"],
                       target_pose=ep.get("scene", {}).get("target_pose", "?"),
                       detected=perc.get("pos") is not None,
                       n_points=int(perc.get("n_points", 0)),
                       timings=ep.get("timings", {}), dir=ep.get("dir", ""))
    if not m.detected:
        return m
    est = np.array(perc["pos"], float)
    gt_pos = np.array(gt["pos"], float)
    m.pos_err_mm = float(np.linalg.norm(est - gt_pos) * 1000.0)
    if gt.get("axis_defined") and perc.get("axis") is not None:
        m.axis_err_deg = _axis_angle_deg(np.array(perc["axis"], float),
                                         np.array(gt["axis"], float))
    if perc.get("upright") is not None:
        m.upright_err = float(abs(float(perc["upright"]) - float(gt["upright"])))
    if perc.get("top_z") is not None:
        m.top_err_mm = float(abs(float(perc["top_z"]) - float(gt["top_z"])) * 1000.0)
    return m


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #


def _stats(values: list[float]) -> dict:
    v = np.asarray([x for x in values if x is not None], float)
    if len(v) == 0:
        return {"n": 0}
    return {"n": int(len(v)), "mean": float(v.mean()),
            "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)),
            "max": float(v.max())}


def aggregate(rows: list[EpisodeMetrics]) -> dict:
    out: dict = {"n_episodes": len(rows),
                 "detect_rate": float(np.mean([r.detected for r in rows])) if rows else 0.0}
    for key in ("pos_err_mm", "axis_err_deg", "upright_err", "top_err_mm"):
        out[key] = _stats([getattr(r, key) for r in rows])
    out["timings_s"] = {
        k: _stats([r.timings.get(k) for r in rows])
        for k in ("render", "semantics", "geometry", "total")
    }
    for pose in ("upright", "lying"):
        sub = [r for r in rows if r.target_pose == pose]
        if sub:
            out[f"pose_{pose}"] = {
                "n": len(sub),
                "pos_err_mm": _stats([r.pos_err_mm for r in sub]),
                "axis_err_deg": _stats([r.axis_err_deg for r in sub]),
            }
    return out


def _fmt(v: float | None, unit: str = "") -> str:
    if v is None:
        return "—"
    return f"{v:.1f}{unit}" if abs(v) < 1000 else f"{v:.0f}{unit}"


def markdown_report(agg: dict, *, title: str, config_used: dict,
                    failures: list[EpisodeMetrics]) -> str:
    L: list[str] = [f"# {title}", ""]
    L += ["## 配置", ""]
    for k, v in config_used.items():
        L.append(f"- `{k}`: {v}")
    L += ["", "## 结果", "",
          f"- episodes: **{agg['n_episodes']}**",
          f"- 检出率: **{agg['detect_rate'] * 100:.1f}%**", "",
          "| 指标 | n | mean | median | p95 | max | 验收线 (median / p95) | 判定 |",
          "|---|---|---|---|---|---|---|---|"]
    for key, spec in ACCEPT.items():
        s = agg[key]
        if s.get("n", 0) == 0:
            L.append(f"| {spec['name']} | 0 | — | — | — | — | — | — |")
            continue
        ok_m = s["median"] <= spec["median"]
        ok_p = s["p95"] <= spec["p95"]
        verdict = "✅" if (ok_m and ok_p) else ("⚠️" if ok_m else "❌")
        L.append(f"| {spec['name']} | {s['n']} | {_fmt(s['mean'])} | "
                 f"{_fmt(s['median'])} | {_fmt(s['p95'])} | {_fmt(s['max'])} | "
                 f"{spec['median']} / {spec['p95']} | {verdict} |")
    t = agg["timings_s"]
    L += ["", "## 时延", "",
          "| 环节 | median (s) | p95 (s) |", "|---|---|---|"]
    for k in ("render", "semantics", "geometry", "total"):
        s = t.get(k, {})
        if s.get("n"):
            L.append(f"| {k} | {s['median']:.3f} | {s['p95']:.3f} |")
    L += ["", "## 按目标姿态", ""]
    for pose in ("upright", "lying"):
        g = agg.get(f"pose_{pose}")
        if not g:
            continue
        L.append(f"- **{pose}** ({g['n']} ep): 位置 median "
                 f"{_fmt(g['pos_err_mm'].get('median'), ' mm')}, 主轴 median "
                 f"{_fmt(g['axis_err_deg'].get('median'), '°')}")
    if failures:
        L += ["", "## 失败样本", "",
              "| seed | 姿态 | 位置误差 (mm) | 主轴 (deg) | 目录 |", "|---|---|---|---|---|"]
        for f in failures[:20]:
            L.append(f"| {f.seed} | {f.target_pose} | {_fmt(f.pos_err_mm)} | "
                     f"{_fmt(f.axis_err_deg)} | `{Path(f.dir).name}` |")
    L.append("")
    return "\n".join(L)


def failure_grid(dirs: list[Path], out_path: Path, cols: int = 4,
                 thumb: tuple[int, int] = (320, 240)) -> Path | None:
    from PIL import Image
    imgs = []
    for d in dirs:
        p = Path(d) / "overlay_view0.png"
        if p.exists():
            imgs.append(Image.open(p).convert("RGB").resize(thumb))
    if not imgs:
        return None
    rows = (len(imgs) + cols - 1) // cols
    grid = Image.new("RGB", (cols * thumb[0], rows * thumb[1]), (20, 20, 24))
    for i, im in enumerate(imgs):
        grid.paste(im, ((i % cols) * thumb[0], (i // cols) * thumb[1]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    return out_path


# --------------------------------------------------------------------------- #
# 批量
# --------------------------------------------------------------------------- #


def sweep(n: int, backends: list[str] | None = None, seed0: int = 0,
          out_root: Path | str = config.CAPTURE_ROOT,
          scene_kw: dict | None = None, do_overlay: bool = True,
          verbose: bool = True) -> list[EpisodeMetrics]:
    backends = backends or ["none"]
    rows: list[EpisodeMetrics] = []
    for backend in backends:
        for i in range(n):
            seed = seed0 + i
            ep = capture_episode(seed, out_root=out_root, backend=backend,
                                 scene_kw=scene_kw, do_overlay=do_overlay)
            m = evaluate_episode(ep)
            rows.append(m)
            if verbose:
                pose = m.target_pose
                print(f"[{backend:6s}] seed={seed:04d} {pose:8s} "
                      f"pos={_fmt(m.pos_err_mm, 'mm'):>8s} "
                      f"axis={_fmt(m.axis_err_deg, '°'):>7s} "
                      f"top={_fmt(m.top_err_mm, 'mm'):>7s} "
                      f"n={m.n_points:5d} "
                      f"t={m.timings.get('total', 0):.2f}s", flush=True)
    return rows


def write_report(rows: list[EpisodeMetrics], name: str,
                 config_used: dict | None = None,
                 report_root: Path | str = config.REPORT_ROOT,
                 worst: int = 8) -> tuple[Path, Path]:
    report_root = Path(report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    # 按 backend 分组出表
    by_backend: dict[str, list[EpisodeMetrics]] = {}
    for r in rows:
        by_backend.setdefault(r.backend, []).append(r)

    sections: list[str] = []
    metrics_all: dict = {"config": config_used or {}, "backends": {}}
    for backend, sub in by_backend.items():
        agg = aggregate(sub)
        metrics_all["backends"][backend] = agg
        det = [r for r in sub if r.detected]
        fails = sorted([r for r in det if (r.pos_err_mm or 0) > 15],
                       key=lambda r: -(r.pos_err_mm or 0))[:worst]
        sections.append(markdown_report(
            agg, title=f"感知评测 — backend `{backend}`",
            config_used={"episodes": len(sub), **(config_used or {})},
            failures=fails))
        if fails:
            failure_grid([Path(r.dir) for r in fails],
                         report_root / f"failures_{backend}.png")

    md = report_root / f"{name}.md"
    js = report_root / f"{name}.json"
    md.write_text("\n\n---\n\n".join(sections) if sections else "# no data\n")
    js.write_text(json.dumps(metrics_all, indent=1, default=str))
    return md, js
