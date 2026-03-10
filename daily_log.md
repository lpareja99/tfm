### Feb 02
- Start initiall developing, trying to get to run a basic mask2former from mmsegmentation
- Steps
```
Install Docker Desktop

Install NVIDIA Container Toolkit in WSL
# 1. Setup the package repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 2. Install the toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 3. Configure the Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker

# 4. Restart WSL - powershell
wsl --shutdown
Restart-Service *docker*

# 5. verify GPU active (ubuntu)
docker run --rm --gpus all nvidia/cuda:12.0.1-base-ubuntu22.04 nvidia-smi
```

```

Clone repo
$ git clone https://github.com/open-mmlab/mmsegmentation.git
$ cd mmsegmentation

Build docker img
$ docker build -t mmseg-mask2former .

Run container
$ docker run --gpus all --shm-size=8g -it --name mmseg_tfm -u $(id -u):$(id -g) -v $(pwd):/mmsegmentation -v $(pwd)/data:/mmsegmentation/data mmseg-mask2former

Start container 
$ docker start mmseg_tfm

Inside the container
$ pip install -v -e .
$ python -c "import mmseg; print(mmseg.__version__)"
$ mim install mmdet

Download tiny-checkpoint of ado20k
$ mim download mmsegmentation --config mask2former_swin-t_8xb2-160k_ade20k-512x512 --dest .

Try an img demo
$ python demo/image_demo.py demo/demo.png mask2former_swin-t_8xb2-160k_ade20k-512x512.py mask2former_swin-t_8xb2-160k_ade20k-512x512_20221203_234230-7d64e5dd.pth --device cuda:0 --out-file ./work_dirs/mask2former_experiment_1/result.jpg
```

Traning sample with ade20k dataset
```
1. Run container

2. Download dataset
mim download mmsegmentation --dataset ade20k

3. dowload test bed Carl sent
4. Create config -py 


5. train the model

rm -rf ./work_dirs/company_first_run/2026*

Train data
$ python tools/train.py configs/tfm/initial_test_flowity.py --amp --work-dir ./work_dirs/company_first_run

Run on best iter
python tools/test.py configs/tfm/initial_test_flowity.py work_dirs/company_first_run/best_mIoU_iter_4480.pth --show-dir work_dirs/company_first_run/results

```