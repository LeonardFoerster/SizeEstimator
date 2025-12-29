# Base image with ROCm 6.2 and PyTorch pre-installed
FROM rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch2.3.0

# Set the architecture override for RX 7800 XT (and other RDNA3 cards)
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for OpenCV and general utility
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Note: We rely on the base image's PyTorch installation.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Ensure the run_gpu.sh script is executable
RUN chmod +x run_gpu.sh

# Default command
ENTRYPOINT ["./run_gpu.sh"]
