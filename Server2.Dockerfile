FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    git \ 
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY server2/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY server2/ .

CMD ["python3", "worker.py"]
