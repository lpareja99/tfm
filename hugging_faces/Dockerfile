# Upgrade to 2.4.0 to satisfy the new Transformers requirement
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt .
# Pin NumPy to avoid the 2.x crash we saw earlier
RUN pip uninstall -y numpy && pip install --no-cache-dir "numpy<2.0.0" && pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["bash"]