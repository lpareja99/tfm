# Start from a base image with CUDA and PyTorch already inside
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

# Install system dependencies for OpenCV
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    build-essential \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip setuptools wheel
RUN pip install "setuptools<70.0" "cmake>=3.25.0"
RUN pip install --only-binary=:all: pyarrow mlflow azureml-mlflow


RUN cmake --version

# Set the working directory inside the container
WORKDIR /app

# Install mmsegmentation directly from the source into the container
# This way, you don't have it in your repo, but the container HAS it.
RUN pip install openmim
RUN mim install "mmengine>=0.7.4" "mmcv==2.1.0" "mmdet>=3.0.0" "mmsegmentation>=1.2.2"

ENV TORCH_HOME=/mmsegmentation/.cache
ENV MMENGINE_CACHE_DIR=/mmsegmentation/.cache

# Copy your requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Command to keep the container running
CMD ["bash"]