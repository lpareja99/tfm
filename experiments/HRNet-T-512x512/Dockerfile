# docker build -t hrnet-experiment -f ./experiments/HRNet-T-512x512/Dockerfile .

# To run the container locally (run at root level):
# $ docker compose run --rm hrnet-t-hijo bash

ARG ACR_REGISTRY=your-registry.azurecr.io
FROM ${ACR_REGISTRY}/roadai/laura_tfm:base_img

WORKDIR /app

COPY Makefile .
COPY runs/ ./runs/
COPY experiments/HRNet-T-512x512/ .

COPY experiments/HRNet-T-512x512/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["bash"]