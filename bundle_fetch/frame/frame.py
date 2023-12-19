import torch
from bundle_fetch.frame.frame_dataset import FrameDataset
from torchvision.transforms.functional import rgb_to_grayscale
import open3d as o3d
import numpy as np
from bundle_fetch.frame.frame_realsense import FrameRealsense
from bundle_fetch.frame.frame_spot import FrameSpot


class Frame(object):
    def __init__(self, frame_stop, track_queue) -> None:
        self.frame_stop = frame_stop
        self.track_queue = track_queue
        self.source = None
        W = 640
        H = 480
        self.uv1 = torch.stack(torch.meshgrid(torch.arange(0, W), torch.arange(0, H))).float()
        self.uv1 = torch.cat((self.uv1, torch.ones_like(self.uv1[:1])), dim=0)

    def __call__(self):
        """
        Get frames
        """
        if self.source is None:
            # self.source = BfDataset('/media/rpm/Data/imitation_learning/BundleFetch/data/test_multiobj')
            # self.source = FrameRealsense()
            self.source = FrameSpot()
        i_frame = 0
        while not self.frame_stop.is_set():
            frame = self.source.get_frame()
            if frame is None:
                break
            frame = process_frame(frame, self.uv1)

            self.track_queue.put(frame)
            
            i_frame += 1


def process_frame(frame, uv1):
    C, H, W = frame['rgb'].shape
    frame['gray'] = rgb_to_grayscale(frame['rgb']) # (1, H, W)
    frame['wh'] = torch.tensor([W, H]) # (2)

    # add xyz
    frame['xyz'] = torch.inverse(frame['cam_K']) @ (frame['depth'].permute(0, 2, 1) * uv1).reshape(3, -1) # (3, W*H)
    frame['xyz'] = frame['xyz'].reshape(3, W, H).permute(0, 2, 1) / 1000 # (3, H, W)

    # calculate normal
    frame['nxyz'] = torch.zeros_like(frame['xyz']) # (3, H, W)
    dz_thresh = 0.1
    dzdx_p = frame['xyz'][:, 1:-1, 1:-1] - frame['xyz'][:, 1:-1, 2:] # (3, H-2, W-2)
    dzdx_n = frame['xyz'][:, 1:-1, 1:-1] - frame['xyz'][:, 1:-1, :-2] # (3, H-2, W-2)
    dzdy_p = frame['xyz'][:, 1:-1, 1:-1] - frame['xyz'][:, 2:, 1:-1] # (3, H-2, W-2)
    dzdy_n = frame['xyz'][:, 1:-1, 1:-1] - frame['xyz'][:, :-2, 1:-1] # (3, H-2, W-2)
    x_dir = (
        (abs(dzdx_p) < dz_thresh) * (abs(dzdx_n) < dz_thresh) * (dzdx_p + dzdx_n) / 2 +
        (abs(dzdx_p) < dz_thresh) * (abs(dzdx_n) >= dz_thresh) * dzdx_p +
        (abs(dzdx_p) >= dz_thresh) * (abs(dzdx_n) < dz_thresh) * dzdx_n
    ) # (3, H-2, W-2)
    y_dir = (
        (abs(dzdy_p) < dz_thresh) * (abs(dzdy_n) < dz_thresh) * (dzdy_p + dzdy_n) / 2 +
        (abs(dzdy_p) < dz_thresh) * (abs(dzdy_n) >= dz_thresh) * dzdy_p +
        
        (abs(dzdy_p) >= dz_thresh) * (abs(dzdy_n) < dz_thresh) * dzdy_n
    ) # (3, H-2, W-2)
    frame['nxyz'][:, 1:-1, 1:-1] = torch.cross(x_dir, y_dir, dim=0) # (3, H-2, W-2)
    frame['nxyz'][:, 1:-1, 1:-1] /= torch.norm(frame['nxyz'][:, 1:-1, 1:-1], dim=0, keepdim=True) * 10 # (3, H-2, W-2)
    frame['nxyz'] *= -torch.sign(frame['nxyz'][[2]] + 1e-8) # (3, H, W)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(frame['xyz'].reshape(3, -1).permute(1, 0).numpy())
    # pcd.colors = o3d.utility.Vector3dVector(frame['rgb'].reshape(3, -1).permute(1, 0).numpy())
    # pcd.estimate_normals(fast_normal_computation=True)
    # normals = np.asarray(pcd.normals)
    # pcd.normals = o3d.utility.Vector3dVector(normals / 10)
    # # pcd.normals = o3d.utility.Vector3dVector(frame['nxyz'].reshape(3, -1).permute(1, 0).numpy())
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(frame['xyz'].permute(1, 0).numpy())
    # pcd.colors = o3d.utility.Vector3dVector(frame['rgb'].reshape(3, -1).permute(1, 0).numpy())
    # o3d.visualization.draw_geometries([pcd])
    return frame