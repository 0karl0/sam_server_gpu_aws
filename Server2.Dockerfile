FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (torch compiled for CUDA 12.1)
RUN pip3 install --no-cache-dir torch==2.3.1+cu121 torchvision==0.18.1+cu121 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Application requirements
COPY server2/requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Application code
WORKDIR /opt/program
COPY server2/app.py .
COPY server2/worker.py .

CMD ["gunicorn", "--workers", "1", "--timeout", "3600", "--bind", "0.0.0.0:8080", "app:app"]
