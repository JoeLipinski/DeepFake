# Base image: CUDA 12.1 + cuDNN + Ubuntu 22.04
# Matches DGX Spark CUDA toolkit. Use -devel for compilation of basicsr/realesrgan.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps: Python 3.11, Cairo (for cairosvg), curl (healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    libcairo2-dev \
    libpango1.0-dev \
    libgdk-pixbuf2.0-dev \
    libffi-dev \
    shared-mime-info \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && python -m pip install --upgrade pip

WORKDIR /app

# Install Python deps first (cache layer)
COPY requirements.txt .
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
    && pip install -r requirements.txt

# Copy application source
COPY . .

# Create jobs directory
RUN mkdir -p /app/jobs

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
