#!/usr/bin/env bash
# 启动展示台脚本 (macOS / Linux)
cd "$(dirname "$0")/.." || exit 1

PORT=8089

# 检查 8089 端口是否已在运行
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "⚡ 本地 HTTP 服务已在端口 $PORT 运行中..."
else
    echo "🚀 启动本地 HTTP 服务 (端口 $PORT)..."
    python3 -m http.server $PORT > /dev/null 2>&1 &
    sleep 1
fi

URL="http://localhost:$PORT/08_三维动作分析展示台/"
echo "🌐 正在浏览器中打开展示台: $URL"

if command -v open > /dev/null; then
    open "$URL"
elif command -v xdg-open > /dev/null; then
    xdg-open "$URL"
else
    echo "请手动在浏览器中访问: $URL"
fi
