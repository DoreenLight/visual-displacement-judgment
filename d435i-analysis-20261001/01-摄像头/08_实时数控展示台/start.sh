#!/bin/bash
# ==============================================================================
# Intel RealSense D435i 视觉位移监控数控展示台 - 一键启动脚本
# ==============================================================================

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "================================================================================"
echo " 🚀 正在启动：Intel RealSense D435i 视觉微位移监控数控展示台"
echo "================================================================================"

PYTHON_BIN="/opt/anaconda3/envs/realsense/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    echo "❌ 未检测到 conda 环境 realsense，尝试使用系统 python3..."
    PYTHON_BIN="$(which python3)"
fi

echo "✔ Python 解释器: $PYTHON_BIN"

# 检查相机硬件连接
if ! ioreg -p IOUSB -w0 -l 2>/dev/null | grep -qi "RealSense"; then
    echo "⚠️ 警告：系统 USB 总线未检测到 RealSense 相机，请确保 USB-C 数据线已直插机身！"
fi

# 终止可能占用的旧进程
pkill -f "rs_live_view" 2>/dev/null || true
pkill -f "app.py --port 8899" 2>/dev/null || true
sleep 1

# 启动后台服务并延迟打开浏览器
(
    sleep 3
    echo "🌐 正在打开浏览器展示台..."
    open "http://127.0.0.1:8899/" 2>/dev/null || true
) &

echo "✔ 正在启动 Web 展示台服务 (端口 8899)..."
echo "✔ 提示：macOS 12+ 访问 USB 底层需管理员权限，若提示请输入密码。"
echo "--------------------------------------------------------------------------------"

# 如果临时凭据存在则直接使用，否则正常终端提权
if [ -f "/tmp/.sdsp" ]; then
    sudo -S -E "$PYTHON_BIN" -u "$DIR/app.py" --port 8899 < /tmp/.sdsp
else
    sudo -E "$PYTHON_BIN" -u "$DIR/app.py" --port 8899
fi
