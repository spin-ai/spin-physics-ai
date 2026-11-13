#!/usr/bin/env bash
# 把 reports/index.html 里的可拖拽 3D 控件录成循环旋转的 GIF。
#
# 为什么不用录屏：控件本身就吃 URL 参数，所以只要按角度逐帧截屏再拼起来，
# 就能得到一条无缝循环（首尾角度差正好 360°）。
#
# 报告里有两个控件，用 MODE 选：
#   viz  —— 仿真场景（§04），参数 ?stage=&yaw=&pitch=&dist=&spin=&hud=
#   rviz —— 真人照片点云（§05），参数 ?rcloud=&ryaw=&rpitch=&rdist=&rdepth=
#
# 用法：
#   ./tools/make_rotation_gif.sh                                   # viz, stage=3, 36 帧
#   ./tools/make_rotation_gif.sh viz  0  24 9  rot_cloud           # 四视角点云
#   ./tools/make_rotation_gif.sh rviz "" 24 9  rot_portrait        # 人像点云（toddler1）
#   ./tools/make_rotation_gif.sh rviz "" 24 9  rot_bust  portrait    # 半身像点云（最后一个参数选点云）
#
# 依赖：Google Chrome（无头截图）+ ImageMagick（拼 GIF）。都只在本机用，
# 运行期不需要，因此没写进 requirements。
set -euo pipefail

MODE="${1:-viz}"
STAGE="${2:-3}"          # 仅 viz 用；rviz 传空串
FRAMES="${3:-36}"
DELAY="${4:-7}"          # 单位 1/100 s
NAME="${5:-rot_fit}"
CLOUD="${6:-toddler1}"   # 仅 rviz 用：toddler1 | portrait
RDEPTH="${RDEPTH:-1}"    # 仅 rviz 用：1 = 按深度上色，0 = 按原图颜色

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # → microduck-vision/
REPORTS="$HERE/reports"
OUT="$REPORTS/img/$NAME.gif"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
[ -x "$CHROME" ] || { echo "找不到 Chrome：$CHROME" >&2; exit 1; }
command -v magick >/dev/null || { echo "需要 ImageMagick（magick）" >&2; exit 1; }

case "$MODE" in
  viz)  ELEM="viz";  CANVAS="viz-canvas"  ;;
  rviz) ELEM="rviz"; CANVAS="rviz-canvas" ;;
  *) echo "MODE 只能是 viz 或 rviz" >&2; exit 2 ;;
esac

# 画布在页面里的纵向偏移随内容变化，先量一次。
# 探针页把偏移写进 document.title，用 --dump-dom 读回来。
python3 - "$REPORTS/index.html" "$TMP/_probe.html" "$ELEM" "$CANVAS" <<'PY'
import sys
src, dst, elem, canvas = sys.argv[1:5]
s = open(src, encoding="utf-8").read()
probe = ("<script>window.addEventListener('load',()=>{const v=document.getElementById('%s');"
         "document.title='OFFSET:'+Math.round(v.getBoundingClientRect().top+window.scrollY)"
         "+':H:'+Math.round(v.getBoundingClientRect().height)"
         "+':CW:'+document.getElementById('%s').clientWidth});</script>" % (elem, canvas))
open(dst, "w", encoding="utf-8").write(s.replace("</body>", probe + "</body>"))
PY

GEOM=$("$CHROME" --headless --disable-gpu --use-gl=swiftshader --virtual-time-budget=4000 \
       --window-size=1200,900 --dump-dom "file://$TMP/_probe.html" 2>/dev/null \
       | grep -o "OFFSET:[0-9]*:H:[0-9]*:CW:[0-9]*" | head -1)
OFFSET=$(echo "$GEOM" | cut -d: -f2)
HEIGHT=$(echo "$GEOM" | cut -d: -f4)
WIDTH=$(echo "$GEOM" | cut -d: -f6)
echo "控件 #${ELEM}：宽 ${WIDTH}px 高 ${HEIGHT}px，页面偏移 ${OFFSET}px"

# 外面套一层 iframe，用负 margin 把画布顶到视口里，这样每帧只截 ${WIDTH}×${HEIGHT}。
# 关键：iframe 的高度必须超过 offset+height，否则画布落在 iframe 视口之外，截出来是空白。
CROP_X=120
CROP_Y=14
WIN_W=1200
WIN_H=$((HEIGHT + 28))
FRAME_H=$((OFFSET + HEIGHT + 200))

mkdir -p "$TMP/f"
for i in $(seq 0 $((FRAMES - 1))); do
  # 两个控件的默认朝向不同：viz 的 DEF.yaw=-1.05，rviz 的 DEF.yaw=0
  if [ "$MODE" = "viz" ]; then
    ANGLE=$(python3 -c "import math;print(round(-1.05 + 2*math.pi*$i/$FRAMES, 4))")
    QUERY="stage=${STAGE}&spin=0&hud=0&yaw=$ANGLE"
  else
    ANGLE=$(python3 -c "import math;print(round(2*math.pi*$i/$FRAMES, 4))")
    QUERY="rcloud=${CLOUD}&rdepth=${RDEPTH}&ryaw=$ANGLE"
  fi
  cat > "$TMP/w.html" <<HTML
<!doctype html><html><body style="margin:0;overflow:hidden;background:#0d1219">
<iframe src="$REPORTS/index.html?$QUERY"
        style="border:0;width:${WIN_W}px;height:${FRAME_H}px;margin-top:-$((OFFSET - CROP_Y))px;display:block"></iframe>
</body></html>
HTML
  "$CHROME" --headless --disable-gpu --use-gl=swiftshader --virtual-time-budget=2500 \
    --screenshot="$TMP/f/f$(printf %02d "$i").png" \
    --window-size="$WIN_W,$WIN_H" --hide-scrollbars "file://$TMP/w.html" >/dev/null 2>&1
  echo -ne "\r  帧 $((i + 1))/$FRAMES"
done
echo

magick -delay "$DELAY" -loop 0 -dispose none "$TMP"/f/*.png \
  -crop "${WIDTH}x${HEIGHT}+${CROP_X}+${CROP_Y}" +repage -layers Optimize "$OUT"

echo "写出 ${OUT}（$(du -h "$OUT" | cut -f1)）"
