# Setup

Environment and Docker image builds for the road-defect segmentation repo.

## Prerequisites

- **Docker** (with the **NVIDIA Container Toolkit** for GPU). On WSL, install the toolkit and
  wire it into Docker:
  ```bash
  # NVIDIA Container Toolkit
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
       | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
       | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  # verify:
  docker run --rm --gpus all nvidia/cuda:12.0.1-base-ubuntu22.04 nvidia-smi
  ```
- **Azure CLI** (`az`) only if you submit Azure ML jobs or download checkpoints.
- `cp .env.example .env` and fill in your Azure/ACR coordinates (gitignored).

## Docker images

| Image | Built from | Command | Used by |
|---|---|---|---|
| `road_defect_base` | root `Dockerfile` (`pytorch 2.1.2 / CUDA 11.8` + mmengine/mmcv 2.1.0/mmdet/mmseg) | `make build-base` | swin / hrnet / beit; parent of the two CUDA images |
| `road_defect_flash` | `experiments/flashInternImage-T-512x512/Dockerfile.local` | `make build-flash` | FlashInternImage — compiles **DCNv4** for sm_50 |
| `road_defect_intern` | `experiments/InterImage-T-512x512/Dockerfile.local` | `make build-intern` | InterImage — compiles **DCNv3** for sm_50 |

The `docker-compose.yml` also defines a `jupyter` service and the per-experiment **Azure
child** build services (`swin_t_child`, …) that inherit `${ACR_REGISTRY}/roadai/laura_tfm:base_img`.
Each experiment additionally has a self-contained `Dockerfile.monolith` (installs mmseg and
compiles ops from scratch) as a reproducible standalone alternative.

## Local GPU builds (custom CUDA ops)

InterImage (**DCNv3**) and FlashInternImage (**DCNv4**) need their op compiled for your GPU's
compute capability. The `Dockerfile.local` files do this on top of `road_defect_base`; the
thesis machine is a **Quadro M1200 (sm_50)**, so they compile with `TORCH_CUDA_ARCH_LIST="5.0"`.

```bash
make build-intern   # InterImage / DCNv3
make build-flash    # FlashInternImage / DCNv4
```

The op is validated at **runtime** (the build host has no GPU; a `mock_cuda` shim is used at
build time). At run time the config's `custom_imports` is resolved by adding the experiment dir
to `PYTHONPATH` — the runners in `scripts/run/` do this automatically.

For reference, compiling an op by hand inside a container follows the upstream steps:
```bash
pip install ninja timm
export CUDA_HOME=/usr/local/cuda           # (or /opt/conda for the older base)
export CPATH=$CUDA_HOME/include:$CPATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
sh make.sh                                 # inside custom_modules/ops_dcnv3 (or ops_dcnv4)
```
Sources: DCNv3 → [OpenGVLab/InternImage](https://github.com/OpenGVLab/InternImage) ·
DCNv4 → [OpenGVLab/DCNv4](https://github.com/OpenGVLab/DCNv4).

## Sanity check

```bash
make weather MODEL=swin MODE=smoke DEVICE=cpu
```
Runs Mask2Former+Swin on 3 weather images inside `road_defect_base` and should end with
`Testing finished successfully`.
