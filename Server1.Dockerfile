# Server1.Dockerfile
FROM python:3.10-slim

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

CMD ["python", "app.py"]
