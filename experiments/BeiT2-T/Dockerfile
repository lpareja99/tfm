# docker build -t beit2 -f ./experiments/BeiT2-T/Dockerfile .

# To run the container locally (run at root level):
# $ docker compose run --rm beit2 bash

ARG ACR_REGISTRY=your-registry.azurecr.io
FROM ${ACR_REGISTRY}/roadai/laura_tfm:base_img

WORKDIR /app

COPY Makefile .
COPY runs/ ./runs/
COPY experiments/BeiT2-T/ .

COPY experiments/BeiT2-T/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install "scipy<1.14.0"

CMD ["bash"]