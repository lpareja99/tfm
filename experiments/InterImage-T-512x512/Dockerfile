# docker build -t interimage-t -f ./experiments/InterImage-T/Dockerfile .

# To run the container locally (run at root level):
# $ docker compose run --rm interimage-t bash

FROM crflowityartifacts.azurecr.io/roadai/laura_tfm:base_img

RUN apt-get update && apt-get install -y --no-install-recommends unzip wget \
    && rm -rf /var/lib/apt/lists/* 

WORKDIR /app

# 1. Compile DCNv3 FIRST so Docker caches this heavy step
RUN mkdir -p /app/custom_modules && \
    wget https://github.com/OpenGVLab/InternImage/archive/refs/heads/master.zip -O master.zip && \
    unzip master.zip "InternImage-master/segmentation/ops_dcnv3/*" -d /app/custom_modules/ && \
    mv /app/custom_modules/InternImage-master/segmentation/ops_dcnv3 /app/custom_modules/ops_dcnv3 && \
    rm -rf /app/custom_modules/InternImage-master master.zip

WORKDIR /app/custom_modules/ops_dcnv3
ENV CUDA_HOME=/usr/local/cuda
ENV CPATH=/usr/local/cuda/include:${CPATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN echo "import torch\n\
def mock_is_available(): return True\n\
torch.cuda.is_available = mock_is_available" > mock_cuda.py && \
    sed -i '1i import mock_cuda' setup.py

ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
ENV MAX_JOBS=4
RUN sh make.sh

# 2. Go back to main directory
WORKDIR /app

# 3. Copy specific experimental files LAST
# This guarantees that tweaking your scripts won't trigger a C++ compilation
COPY Makefile .
COPY runs/ ./runs/
COPY experiments/InterImage-T-512x512/ .


CMD ["bash"]