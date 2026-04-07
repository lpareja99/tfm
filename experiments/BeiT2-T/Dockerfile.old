#FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime 

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends\
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir "setuptools<70.0" "cmake>=3.25.0"
RUN pip install --no-cache-dir --only-binary=:all: pyarrow mlflow azureml-mlflow

RUN cmake --version

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Install mmsegmentation directly from the source into the container
# This way, you don't have it in your repo, but the container HAS it.
RUN pip install --no-cache-dir openmim
RUN mim install "mmengine>=0.7.4" "mmcv==2.1.0" "mmdet>=3.0.0" "mmsegmentation>=1.2.2"
RUN rm -rf /root/.cache/mim

ENV TORCH_HOME=/mmsegmentation/.cache
ENV MMENGINE_CACHE_DIR=/mmsegmentation/.cache

# Copy your specific project files into the container
COPY . .

# Command to keep the container running
CMD ["bash"]