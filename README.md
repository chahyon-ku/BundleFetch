```
mamba create -n bundle_fetch
mamba activate bundle_fetch
mamba install python=3.8
mamba install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
mamba install matplotlib einops yacs hydra-core imageio plotly dash scikit-learn addict pandas tqdm dearpygui opencv libjpeg libpng
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
mamba install pytorch3d -c pytorch3d
pip install --no-deps kornia open3d dearpygui
pip install --no-deps git+https://github.com/facebookresearch/segment-anything.git
pip install --no-deps pyrealsense2

pip install --upgrade --no-deps bosdyn-client bosdyn-mission bosdyn-choreography-client bosdyn-api bosdyn-core
mamba install protobuf grpcio deprecated pyjwt

pip install --no-deps git+https://github.com/princeton-vl/lietorch.git
mamba install pytorch-scatter -c pyg
```