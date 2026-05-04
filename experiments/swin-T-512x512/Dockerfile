# docker build -t swin-experiment -f ./experiments/swin-T-512x512/Dockerfile .

# To run the container locally (run at root level):
# $ docker compose run --rm swin-t-hijo bash

FROM crflowityartifacts.azurecr.io/roadai/laura_tfm:base_img

WORKDIR /app

COPY Makefile .
COPY runs/ ./runs/
COPY experiments/swin-T-512x512/ .

CMD ["bash"]