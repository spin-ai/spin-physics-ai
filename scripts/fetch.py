#!/usr/bin/env python3
"""把「不进 git 的第三方仓库 + 大文件」拉到位。

工作区里只有我们自己的东西入库；下面这些都不入库，靠本脚本按固定 commit / revision 复原：

  第三方 git 仓库  robots/microduck/**            （4 个上游 checkout）
  预训练权重      .../microduck/policies/*.onnx   （10 个 ONNX）
  基准仓库        benchmarks/LIBERO-plus           （含本地兼容补丁）
  大数据          data/libero-plus/assets.zip     （6.4 GB，解压进 LIBERO assets）
  容器镜像        srayuth089/microduck-docker     （650 MB）

用法：
    ./scripts/fetch.py                 # = all
    ./scripts/fetch.py upstream        # 4 个上游 checkout（含 policies 目录占位）
    ./scripts/fetch.py policies        # 10 个 ONNX
    ./scripts/fetch.py libero          # LIBERO-plus 仓库 + 兼容补丁 + assets.zip 解压
    ./scripts/fetch.py docker          # 镜像（Hub 直连不通，走镜像源 + retag）
    ./scripts/fetch.py verify          # 只校验，不下载

设计取舍：
  * 源码优先 `git clone` + `checkout <commit>`，这样能 `git log` / `git diff` 上游；
    本机 github.com:443 常年不通时自动退回 codeload tarball（快照，无 .git），
    并在最后打印提示。两条路都按 commit 钉死，内容等价。
  * 大文件用 curl（支持断点续传），下完校验 sha256；不匹配即删掉重来。
  * 只用标准库 + curl + tar + git。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(os.environ.get("FETCH_ROOT") or Path(__file__).resolve().parent.parent).resolve()
NO_GIT = False          # --no-git：强制走 codeload tarball（测试 / github 不通时省时间）
HF = "https://hf-mirror.com"          # huggingface.co 直连不通
GH = "https://github.com"
CODELOAD = "https://codeload.github.com"
DOCKER_MIRROR = "docker.1panel.live"  # registry-1.docker.io 直连超时

# --------------------------------------------------------------------------
# 清单：改版本就改这里，然后重跑 ./scripts/fetch.py
# --------------------------------------------------------------------------

REPOS = [
    dict(
        name="microduck-lab",
        repo="jonathanhawkins/microduck-lab",
        commit="8dad9218062f7b0ad29ae6d564305411f5a117c2",
        dest="robots/microduck/microduck-lab",
        sentinel={
            "microduck_local/pyproject.toml": "380f120423c64385e84df1bce324f35a9635b8ca58454d60ee83deda19a631ae",
            "AGENTS.md": "0f971a8654eff0e8d8098537ce4cf5e81dc58a5cff9c4218f354419100af0c03",
            "duck-viewer/package.json": "8f557fb988949d23a754bf397390aa8af41cb6b3908336cb1a84c3af85e6fd2b",
        },
    ),
    # 注意层级：contract.py 用 parents[3]/"microduck_rl" 找模型，这两个必须套在 lab 里
    dict(
        name="microduck",
        repo="pollen-robotics/microduck",
        commit="171a8e1a0c7e2220daa4d8d2a9059e6e4db710d5",
        dest="robots/microduck/microduck-lab/microduck",
        sentinel={"Cargo.toml": "719e7914bfd439def40f0a2acf6be2d4bc755d6917457b982a7db834a14e8fe6"},
    ),
    dict(
        name="microduck_rl",
        repo="pollen-robotics/microduck_rl",
        commit="0daab160d3fb1c2cbaba16a4a19052f36709dd9e",
        dest="robots/microduck/microduck-lab/microduck_rl",
        sentinel={
            "src/mjlab_microduck/robot/microduck/scene_walk.xml": "6e23146c1a5025473b55ec94cbe37b08104ca388b7eb005ea277f8230736c8be",
            "src/mjlab_microduck/robot/microduck/scene.xml": "9a85461121ba8273083c482c166f27e143ca386993b3ec8f8d97eb09aa257d31",
        },
    ),
    dict(
        name="microduck-docker",
        repo="srayuth089/microduck-docker",
        commit="178721fb9aec0f628723e88dc859bbd25c245b9f",
        dest="robots/microduck/docker",
        sentinel={"Dockerfile": "ea52a372914ca9b07a0f71d5ef17c04f4690c18b799e80ac7c3592a65ce216e1"},
    ),
]

LIBERO = dict(
    repo="sylvestf/LIBERO-plus",
    commit="4976dc30028e805ff8094b55501d532c48fec182",
    dest="benchmarks/LIBERO-plus",
    patch="scripts/patches/libero-plus-compat.patch",   # 本地兼容性补丁（torch/Py3.10）
)

POLICIES = dict(
    repo="pollen-robotics/microduck-policies",
    revision="088524a64e2557dc453256b6071dbb9d23888802",
    dest="robots/microduck/microduck-lab/microduck/policies",
    files={
        "alpha_walking.onnx":     ("e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c", 793705),
        "alpha_stand.onnx":       ("1569268713e40deea795dd2922dba50d3621e15a872855408b6b1b125b1c094b", 793705),
        "alpha_sitstand.onnx":    ("c6c40e35e726eabd803d633e090d112994f469921152448367953fbaf9799bc8", 793695),
        "alpha_ground_pick.onnx": ("ffbf5109982ff999b0ba53afe86b9ae731bbec679d67fb7f8ab4c52152c88872", 793685),
        "ball_kick_left.onnx":    ("d6928284dccd3dd61e08bf2f760effa74309fbefd97b2b31afb2a60f526d196a", 793685),
        "ball_kick_right.onnx":   ("147a32c388c6b19111b3ac3b550a9a6dc8b8bf267118af4d8c3712522eedb5af", 793685),
        "roller.onnx":            ("cf05651d2708a2f9364212e86b866c97a70ace8131c492500105e8f28bf99afd", 793685),
        "roller_crouch.onnx":     ("a1a084be240469c76ac9d3fa44d4792f16d4b1da60398b3ecd3cfc5e2244d990", 793685),
        "roulade.onnx":           ("3d60da08fc13f29c1b57f41977aa898132c0d60042100149d8e775affcbca32b", 793685),
        # velstand 不在钉住的 revision 里，只能取 main（上游后续新增）
        "velstand.onnx":          ("ce894d30b3343036d499bfac1102bdaa58d59a21c1b58401ecb70fb6a2f63e0a", 793705),
    },
    main_only={"velstand.onnx"},   # 走 /resolve/main/ 而不是钉住的 revision
)

LIERO_ASSETS = dict(
    url=f"{HF}/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip",
    dest="data/libero-plus/assets.zip",
    size=6395849578,
    # 与 HF LFS oid 一致（2026-11-10 实测下载 + 上游元数据双向对上）
    sha256="96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf",
    unzip_to="benchmarks/LIBERO-plus/libero/libero/assets",
)

IMAGE = dict(
    remote="srayuth089/microduck-docker:latest",
    local="srayuth089/microduck-docker:latest",
)


# --------------------------------------------------------------------------
# 基础设施
# --------------------------------------------------------------------------

def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def say(msg: str) -> None:
    print(f"\033[36m▸ {msg}\033[0m", flush=True)


def warn(msg: str) -> None:
    print(f"\033[33m⚠ {msg}\033[0m", flush=True)


def fail(msg: str) -> None:
    print(f"\033[31m✗ {msg}\033[0m", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def curl(url: str, dest: Path, *, tries: int = 15) -> bool:
    """带断点续传与重试的下载。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-L", "--fail", "-C", "-", "--retry", str(tries),
           "--retry-all-errors", "--retry-delay", "2", "--connect-timeout", "20",
           "-o", str(dest), url]
    return run(cmd).returncode == 0


def fetch_tarball(repo: str, commit: str, dest: Path) -> bool:
    tmp = Path(tempfile.mkdtemp(prefix="fetch-"))
    tgz = tmp / "src.tgz"
    if not curl(f"{CODELOAD}/{repo}/tar.gz/{commit}", tgz):
        shutil.rmtree(tmp, ignore_errors=True)
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as ex:
        if run(["tar", "xzf", str(tgz), "-C", ex]).returncode != 0:
            return False
        tops = list(Path(ex).iterdir())
        if len(tops) != 1:
            fail(f"{repo}: tarball 结构异常")
            return False
        shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(tops[0]), str(dest))
    shutil.rmtree(tmp, ignore_errors=True)
    return True


def fetch_repo(spec: dict) -> bool:
    dest: Path = ROOT / spec["dest"]
    want = spec["commit"]
    name = spec["name"]

    if (dest / ".git").exists():
        run(["git", "-C", str(dest), "fetch", "--quiet", "origin"])
        head = run(["git", "-C", str(dest), "rev-parse", "HEAD"]).stdout.strip()
        if head == want:
            say(f"{name}: 已在 {want[:8]}")
            return check_sentinel(spec)
        say(f"{name}: 切到 {want[:8]}")
        if run(["git", "-C", str(dest), "checkout", "--quiet", want]).returncode == 0:
            return check_sentinel(spec)

    say(f"{name}: git clone（github 直连，可能失败）")
    shutil.rmtree(dest, ignore_errors=True)
    cloned = False
    for attempt in range(1, 1 if NO_GIT else 3):
        r = run(["git", "clone", "--quiet", "--no-checkout", f"{GH}/{spec['repo']}.git", str(dest)],
                timeout=90)
        if r.returncode == 0:
            cloned = run(["git", "-C", str(dest), "checkout", "--quiet", want]).returncode == 0
            break
        warn(f"{name}: git clone 第 {attempt} 次失败，重试")
        shutil.rmtree(dest, ignore_errors=True)

    if not cloned:
        warn(f"{name}: 退回 codeload tarball（快照，无 .git）")
        shutil.rmtree(dest, ignore_errors=True)
        if not fetch_tarball(spec["repo"], want, dest):
            fail(f"{name}: 下载失败")
            return False
        (dest / ".FETCHED_FROM").write_text(
            f"{GH}/{spec['repo']}@{want}\n# codeload tarball 快照：本机 github:443 不可用时降级\n")
    return check_sentinel(spec)


def check_sentinel(spec: dict) -> bool:
    dest = ROOT / spec["dest"]
    ok = True
    for rel, want in spec.get("sentinel", {}).items():
        p = dest / rel
        if not p.exists():
            fail(f"{spec['name']}: 缺 {rel}")
            ok = False
        elif want and sha256(p) != want:
            fail(f"{spec['name']}: {rel} 校验不符（上游变过？）")
            ok = False
    if ok:
        say(f"{spec['name']}: ✓ {spec['commit'][:8]}")
    return ok


# --------------------------------------------------------------------------
# 各子任务
# --------------------------------------------------------------------------

def do_upstream() -> bool:
    return all([fetch_repo(s) for s in REPOS])


def do_policies() -> bool:
    dest = ROOT / POLICIES["dest"]
    dest.mkdir(parents=True, exist_ok=True)
    ok = True
    for fname, (want_sha, want_size) in POLICIES["files"].items():
        p = dest / fname
        repo = POLICIES["repo"]
        rev = "main" if fname in POLICIES["main_only"] else POLICIES["revision"]
        url = f"{HF}/{repo}/resolve/{rev}/{fname}"
        if p.exists() and sha256(p) == want_sha:
            continue
        say(f"policy {fname} ({want_size/1e6:.1f} MB)")
        for attempt in range(1, 4):
            if p.exists():
                p.unlink()
            if curl(url, p) and p.stat().st_size == want_size and sha256(p) == want_sha:
                break
            warn(f"{fname}: 第 {attempt} 次校验失败")
        else:
            fail(f"{fname}: 拉取失败")
            ok = False
    say(f"policies: ✓ {len(POLICIES['files'])} 个")
    return ok


def do_libero() -> bool:
    spec = dict(name="LIBERO-plus", repo=LIBERO["repo"], commit=LIBERO["commit"],
                dest=LIBERO["dest"], sentinel={})
    if not fetch_repo(spec):
        return False

    dest = ROOT / LIBERO["dest"]
    patch = ROOT / LIBERO["patch"]
    if patch.exists():
        if (dest / ".git").exists():
            if run(["git", "-C", str(dest), "apply", "--check", str(patch)]).returncode == 0:
                say("应用兼容补丁 (torch weights_only / Py3.10)")
                run(["git", "-C", str(dest), "apply", str(patch)])
            else:
                say("兼容补丁已在位（跳过）")
        elif run(["patch", "-p1", "--forward", "--dry-run", "-i", str(patch)],
                 cwd=str(dest)).returncode == 0:
            say("应用兼容补丁（tarball 快照，用 patch）")
            run(["patch", "-p1", "--forward", "-i", str(patch)], cwd=str(dest))
        else:
            warn("补丁状态未知，见 docs/LIBERO_PLUS_SETUP.md 手工核对")

    # 大文件 + 解压
    p = ROOT / LIERO_ASSETS["dest"]
    want_sha, want_size = LIERO_ASSETS["sha256"], LIERO_ASSETS["size"]
    if not p.exists() or p.stat().st_size != want_size:
        say(f"assets.zip ({want_size/1e9:.2f} GB) — 断点续传，慢是正常的")
        if not curl(LIERO_ASSETS["url"], p):
            fail("assets.zip 下载失败")
            return False
    if want_sha and sha256(p) != want_sha:
        fail("assets.zip sha256 不符")
        return False
    say("assets.zip ✓")

    unzip_to = ROOT / LIERO_ASSETS["unzip_to"]
    if unzip_to.exists() and any(unzip_to.iterdir()):
        say(f"assets 已解压（{len(list(unzip_to.iterdir()))} 项），跳过")
    else:
        say("解压到 benchmarks/LIBERO-plus/libero/libero/assets")
        unzip_to.parent.mkdir(parents=True, exist_ok=True)
        r = run(["unzip", "-q", "-o", str(p), "-d", str(unzip_to.parent)])
        if r.returncode != 0:
            fail(f"解压失败: {r.stderr.strip()[:200]}")
            return False
        # 若压缩包内自带顶层 assets/，就地修正层级
        nested = unzip_to / "assets"
        if nested.is_dir():
            for item in nested.iterdir():
                shutil.move(str(item), str(unzip_to / item.name))
            nested.rmdir()
    say(f"assets ✓ {len(list(unzip_to.iterdir()))} 项")
    warn("还要一步：pip install -e benchmarks/LIBERO-plus && 写 ~/.libero/config.yaml"
         "（见 docs/LIBERO_PLUS_SETUP.md）")
    return True


def do_docker() -> bool:
    src = f"{DOCKER_MIRROR}/{IMAGE['remote']}"
    say(f"pull {src}（Hub 直连超时，走镜像源）")
    for attempt in range(1, 6):
        r = run(["docker", "pull", src], timeout=3600)
        if r.returncode == 0:
            run(["docker", "tag", src, IMAGE["local"]])
            say(f"→ 已 retag 为 {IMAGE['local']}")
            return True
        tail = (r.stdout + r.stderr).strip().splitlines()[-1:]
        warn(f"第 {attempt} 次失败（{tail[0][:80] if tail else '?'}）—— 大层 short read 常见，已完成层会命中缓存")
    fail("镜像拉取失败")
    return False


def do_verify(deep: bool = False) -> bool:
    ok = True
    for spec in REPOS:
        ok &= check_sentinel(spec)
    dest = ROOT / POLICIES["dest"]
    for fname, (want_sha, _) in POLICIES["files"].items():
        p = dest / fname
        if not p.exists():
            fail(f"policy {fname}: 缺失")
            ok = False
        elif sha256(p) != want_sha:
            fail(f"policy {fname}: sha256 不符")
            ok = False
    if ok:
        say(f"policies: ✓ {len(POLICIES['files'])} 个")

    lr = ROOT / LIBERO["dest"]
    if not (lr / ".git").exists() and not (lr / ".FETCHED_FROM").exists():
        warn("LIBERO-plus: 未拉取（benchmarks/LIBERO-plus 不存在）")
    ap_zip = ROOT / LIERO_ASSETS["dest"]
    if ap_zip.exists():
        if ap_zip.stat().st_size != LIERO_ASSETS["size"]:
            fail(f"assets.zip 大小不符 ({ap_zip.stat().st_size})")
            ok = False
        elif deep:
            want = LIERO_ASSETS["sha256"]
            if want and sha256(ap_zip) != want:
                fail("assets.zip sha256 不符")
                ok = False
            else:
                say("assets.zip ✓ (sha256, 6.4 GB 已重算)")
        else:
            say("assets.zip ✓ (size；要重算 sha256 加 --deep)")
    unz = ROOT / LIERO_ASSETS["unzip_to"]
    if unz.is_dir() and any(unz.iterdir()):
        say(f"LIBERO assets ✓ {len(list(unz.iterdir()))} 项")
    else:
        warn("LIBERO assets: 未解压")
    if run(["docker", "image", "inspect", IMAGE["local"]]).returncode == 0:
        say(f"镜像 ✓ {IMAGE['local']}")
    else:
        warn(f"镜像: {IMAGE['local']} 不在本地")
    print("\n全部通过" if ok else "\n有问题，见上面 ✗")
    return ok


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="拉取不进 git 的第三方源码与大文件")
    ap.add_argument("what", nargs="?", default="all",
                    choices=["all", "upstream", "policies", "libero", "docker", "verify"])
    ap.add_argument("--no-git", action="store_true",
                    help="跳过 git clone，直接下 codeload tarball（github 不通时省 3 分钟）")
    ap.add_argument("--deep", action="store_true",
                    help="verify 时连 6.4 GB assets.zip 也重算 sha256（慢）")
    a = ap.parse_args()

    global NO_GIT
    NO_GIT = a.no_git

    if shutil.which("curl") is None:
        fail("需要 curl")
        return 2

    tasks = {
        "upstream": do_upstream,
        "policies": do_policies,
        "libero": do_libero,
        "docker": do_docker,
    }
    if a.what == "verify":
        return 0 if do_verify(a.deep) else 1
    if a.what == "all":
        ok = all([f() for f in tasks.values()])
        print()
        ok &= do_verify(a.deep)
        return 0 if ok else 1

    ok = tasks[a.what]()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
