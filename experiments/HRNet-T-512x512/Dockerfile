# docker build -t hrnet-experiment -f ./experiments/HRNet-T-512x512/Dockerfile .

# To run the container locally (run at root level):
# $ docker compose run --rm hrnet-t-hijo bash

FROM crflowityartifacts.azurecr.io/roadai/laura_tfm:base_img

WORKDIR /app

COPY Makefile .
COPY runs/ ./runs/
COPY experiments/HRNet-T-512x512/ .

COPY experiments/HRNet-T-512x512/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["bash"]