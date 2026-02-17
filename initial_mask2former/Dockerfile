# Start from a base image with CUDA and PyTorch already inside
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy your requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Install mmsegmentation directly from the source into the container
# This way, you don't have it in your repo, but the container HAS it.
RUN pip install openmim
RUN mim install "mmengine>=0.7.4" "mmcv>=2.0.0" "mmdet>=3.0.0" "mmsegmentation>=1.2.2"


ENV TORCH_HOME=/mmsegmentation/.cache
ENV MMENGINE_CACHE_DIR=/mmsegmentation/.cache

# Copy your specific project files into the container
COPY . .

# Command to keep the container running
CMD ["bash"]