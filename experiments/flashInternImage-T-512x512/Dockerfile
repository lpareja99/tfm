# docker build -t flashinternimage-experiment -f ./experiments/flashInternImage-T-512x512/Dockerfile .

# To run the container locally (run at root level):
# $ docker compose run --rm flashinternimage-t-hijo bash

FROM crflowityartifacts.azurecr.io/roadai/laura_tfm:base_img

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Makefile .
COPY runs/ ./runs/
COPY experiments/flashInternImage-T-512x512/ .

WORKDIR /app/custom_modules/ops_dcnv4

ENV CUDA_HOME=/usr/local/cuda
ENV CPATH=/usr/local/cuda/include
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN echo "import torch\n\
def mock_is_available(): return True\n\
torch.cuda.is_available = mock_is_available" > mock_cuda.py && \
    sed -i '1i import mock_cuda' setup.py

ENV FORCE_CUDA="1"

# Local RTX 4070 = 8.6
# Azure A100 GPU = 8.0
# Azure V100 = 7.0
# Other used ones = 7.5
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
ENV MAX_JOBS=4
RUN sh make.sh 

WORKDIR /app


CMD ["bash"]