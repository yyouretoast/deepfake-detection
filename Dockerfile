FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and GL
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application modules, config, and tests
COPY app.py ./
COPY config ./config
COPY src ./src
COPY tests ./tests

# Run as non-root user for security
RUN useradd -m -u 1000 appuser
USER appuser

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
