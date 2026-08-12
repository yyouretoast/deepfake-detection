FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for libglib (used by OpenCV headless)
# Note: libgl1, libsm6, libxext6, libxrender-dev are NOT needed because
# requirements.txt uses opencv-python-headless which excludes X11/GL dependencies.
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# Install Python requirements
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application modules, config, and tests
COPY app.py ./
COPY config ./config
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests

# Run as non-root user for security
RUN useradd -m -u 1000 appuser
USER appuser

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
