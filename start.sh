#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PORT="${PORT:-8765}"
URL="http://127.0.0.1:${PORT}"

pick_python() {
  for cand in \
    /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3.10 /opt/homebrew/bin/python3 \
    /usr/local/bin/python3.13 /usr/local/bin/python3.12 /usr/local/bin/python3.11 /usr/local/bin/python3.10 /usr/local/bin/python3 \
    /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 \
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
    /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
    /Library/Frameworks/Python.framework/Versions/3.10/bin/python3 \
    "$(command -v python3 2>/dev/null || true)"; do
    if [ -n "${cand}" ] && [ -x "${cand}" ]; then
      if "${cand}" -c "import sys; assert sys.version_info >= (3, 9)" >/dev/null 2>&1; then
        echo "${cand}"
        return 0
      fi
    fi
  done
  return 1
}

open_browser() {
  if command -v open >/dev/null 2>&1; then
    open "$1" >/dev/null 2>&1 || true
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$1" >/dev/null 2>&1 || true
  fi
}

# Rebuild .venv if copied from another machine with broken symlinks
if [ -d ".venv" ] && ! .venv/bin/python -c "import sys" >/dev/null 2>&1; then
  echo "检测到旧的虚拟环境路径不匹配本机，正在重建 .venv..."
  rm -rf .venv
fi

if [ ! -d ".venv" ]; then
  PY_BIN="$(pick_python || true)"
  if [ -z "${PY_BIN}" ]; then
    echo "错误：本机未找到可用的 Python 3.9+。请先安装 Python（例如 brew install python 或从 python.org 下载安装）。"
    exit 1
  fi
  echo "使用 ${PY_BIN} 创建虚拟环境 .venv..."
  "${PY_BIN}" -m venv .venv
fi

if ! .venv/bin/python -c "import sys, fastapi, uvicorn, pymupdf, faster_whisper, numpy; assert sys.version_info >= (3, 10) or __import__('eval_type_backport')" >/dev/null 2>&1; then
  echo "正在安装所需依赖（fastapi / pymupdf / faster-whisper）..."
  .venv/bin/python -m pip install --upgrade pip --timeout 15 -i https://pypi.tuna.tsinghua.edu.cn/simple \
    || .venv/bin/python -m pip install --upgrade pip
  .venv/bin/pip install --timeout 15 -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt \
    || .venv/bin/pip install -r requirements.txt
fi

echo "正在启动 ListenSync 听力歌词同步器: ${URL}"
(
  for _ in $(seq 1 40); do
    if .venv/bin/python -c "import socket; s=socket.create_connection(('127.0.0.1', ${PORT}), 0.5); s.close()" >/dev/null 2>&1; then
      open_browser "${URL}"
      exit 0
    fi
    sleep 0.5
  done
) &

exec .venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port "${PORT}"
