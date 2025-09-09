FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ca-certificates libglib2.0-0 libgl1 git \
    && rm -rf /var/lib/apt/lists/*

# Make sure pip is new enough to handle PyTorch indexes
RUN python3 -m pip install --upgrade pip setuptools wheel

# PyTorch (CUDA 12.1 wheels)
RUN pip3 install --no-cache-dir \
    torch==2.3.1+cu121 torchvision==0.18.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# App deps
COPY server2/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Entrypoint + app
COPY server2/serve /usr/local/bin/serve
RUN chmod +x /usr/local/bin/serve

WORKDIR /opt/program
COPY server2/app.py .
COPY server2/worker.py .

CMD ["serve"]

