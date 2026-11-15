#!/bin/bash
set -euo pipefail

exec 1>&2

echo "=== start.sh running as: $(whoami) ==="

export DISPLAY=:99
export HOME=/tmp
mkdir -p /tmp
chmod 1777 /tmp

# ---- clear stale X locks from a previous (killed) run ----
echo "=== Clearing stale X locks ==="
rm -f /tmp/.X99-lock /tmp/.X11-unix/X99 2>/dev/null || true

# ---- Xvfb ----
echo "=== Starting Xvfb ==="
Xvfb :99 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!

# wait for the X socket, not a magic sleep
for i in $(seq 1 40); do
    [ -S /tmp/.X11-unix/X99 ] && break
    sleep 0.25
done
if ! kill -0 "$XVFB_PID" 2>/dev/null || [ ! -S /tmp/.X11-unix/X99 ]; then
    echo "ERROR: Xvfb failed to start (pid $XVFB_PID)"
    exit 1
fi
echo "Xvfb started (pid $XVFB_PID)"

# ---- fluxbox ----
echo "=== Starting fluxbox ==="
fluxbox > /tmp/fluxbox.log 2>&1 &
sleep 1

# ---- x11vnc ----
echo "=== Starting x11vnc ==="
x11vnc -display :99 -forever -shared -rfbport 5900 -nopw -bg -o /tmp/x11vnc.log
sleep 2

# ---- noVNC ----
echo "=== Starting websockify / noVNC ==="
WEBROOT=/usr/share/novnc
websockify --web="$WEBROOT" 6080 localhost:5900 > /tmp/novnc.log 2>&1 &
sleep 2

echo "=== noVNC ready → http://localhost:6080/vnc.html ==="
echo "=== Starting Playwright sandbox API ==="

exec uvicorn server:app --host 0.0.0.0 --port 8080 --log-level info
