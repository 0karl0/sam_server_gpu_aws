# Server1.Dockerfile
FROM python:3.10-slim

# Ensure Python output is sent straight to the console so Docker logs
# are emitted in real time.  Without this the application appeared to
# hang because stdout was block-buffered until the buffer filled.
ENV PYTHONUNBUFFERED=1
# Disable model downloads; SageMaker provides required weights
ENV ENABLE_MODEL_DOWNLOADS=0

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*




# Install deps
COPY server1/requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY server1/ .

# Flask runs on port 8080
EXPOSE 8080

# Run the app with unbuffered output so log messages appear immediately
CMD ["python", "-u", "app.py"]
