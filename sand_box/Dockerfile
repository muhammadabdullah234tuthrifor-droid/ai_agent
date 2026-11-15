FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    wget \
    git \
    fonts-liberation \
    xvfb \
    x11vnc \
    fluxbox \
    novnc \
    websockify \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libpango-1.0-0 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libxshmfence1 \
    xdg-utils \
    procps \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

# Verify novnc was installed
RUN ls -la /usr/share/novnc || (echo "ERROR: novnc not found" && exit 1)

RUN pip install --no-cache-dir playwright fastapi "uvicorn[standard]"
RUN playwright install chromium && chmod -R 755 /ms-playwright

RUN useradd --create-home --shell /bin/bash sandbox

WORKDIR /workspace

COPY sandbox_runner.py /workspace/sandbox_runner.py
COPY server.py          /workspace/server.py
COPY start.sh           /workspace/start.sh

RUN chmod +x /workspace/start.sh

# Keep root for the VNC stack (simpler & more reliable)
# USER sandbox   ← comment this out for now

ENTRYPOINT ["/workspace/start.sh"]
