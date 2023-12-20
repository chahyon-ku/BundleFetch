```bash
mamba create -n bundle_fetch
mamba activate bundle_fetch
mamba install python=3.8
mamba install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
mamba install -c nvidia cuda=11.7
mamba install -c fvcore -c iopath -c conda-forge fvcore iopath
mamba install pytorch-scatter -c pyg
mamba install pytorch3d -c pytorch3d
mamba install matplotlib einops yacs hydra-core imageio plotly dash scikit-learn addict pandas tqdm opencv protobuf grpcio deprecated pyjwt
pip install --upgrade --no-deps bosdyn-client bosdyn-mission bosdyn-choreography-client bosdyn-api bosdyn-core
pip install --no-deps pyrealsense2 kornia open3d dearpygui
pip install --no-deps git+https://github.com/chahyon-ku/XMem.git
pip install --no-deps git+https://github.com/facebookresearch/segment-anything.git
pip install --no-deps git+https://github.com/princeton-vl/lietorch.git
pip install --no-deps -e .

mkdir checkpoints
cd checkpoints
wget https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth 
```