import time
from matplotlib import pyplot as plt
import numpy as np
import torch
import bundle_fetch.bf_dataset
from torchvision.transforms.functional import rgb_to_grayscale
import open3d as o3d
from bundle_fetch.sam_gui import Segmenter


def frame_thread_target(frame_stop, track_queue):
    print('frame_thread_target')
    cuda_stream = torch.cuda.Stream()
    dataset = bundle_fetch.bf_dataset.BfDataset('data/test', schema=['rgb', 'depth', 'masks'])
    i_frame = 0
    uv1 = torch.stack(torch.meshgrid(torch.arange(0, 640), torch.arange(0, 480))).float()
    uv1 = torch.cat((uv1, torch.ones_like(uv1[:1])), dim=0)

    start = time.time()
    while not frame_stop.is_set():
        frame = dataset[i_frame]
        frame['gray'] = rgb_to_grayscale(frame['rgb'])
        frame['xyz'] = torch.inverse(frame['cam_K']) @ (frame['depth'].permute(0, 2, 1) * uv1).reshape(3, -1)
        frame['xyz'] = frame['xyz'].reshape(3, 640, 480).permute(0, 2, 1) / 1000
        frame['wh'] = torch.tensor([640, 480])

        # if i_frame == 0:
        #     frame['mask'] = Segmenter((frame['rgb'].permute(1, 2, 0) * 255).numpy().astype(np.uint8)).mask
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(frame['xyz'].permute(1, 0).numpy())
        # pcd.colors = o3d.utility.Vector3dVector(frame['rgb'].reshape(3, -1).permute(1, 0).numpy())
        # o3d.visualization.draw_geometries([pcd])

        with torch.cuda.stream(cuda_stream):
            frame = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in frame.items()}
            event = torch.cuda.Event()
            track_queue.put((frame, event))
        
        i_frame += 1
        if i_frame == len(dataset):
            frame_stop.set()