from bundle_fetch.frame.bf_dataset import BfDataset
from torchvision.transforms.functional import rgb_to_grayscale
import torch


def dataset_get_frame():
    def get_frame():
        if get_frame.i_frame >= len(get_frame.dataset):
            return None
        frame = get_frame.dataset[get_frame.i_frame]
        frame['w_T_c'] = torch.eye(4)
        get_frame.i_frame += 1
        return frame
    get_frame.dataset = BfDataset('/media/rpm/Data/imitation_learning/BundleFetch/data/test')
    get_frame.i_frame = 0
    return get_frame


def get_process_frame():
    def process_frame(frame):
        frame['gray'] = rgb_to_grayscale(frame['rgb']) # (1, 480, 640)
        frame['wh'] = torch.tensor([640, 480]) # (2)

        # add xyz
        frame['xyz'] = torch.inverse(frame['cam_K']) @ (frame['depth'].permute(0, 2, 1) * process_frame.uv1).reshape(3, -1) # (3, 307200)
        frame['xyz'] = frame['xyz'].reshape(3, 640, 480).permute(0, 2, 1) / 1000 # (3, 480, 640)
        return frame
    process_frame.uv1 = torch.stack(torch.meshgrid(torch.arange(0, 640), torch.arange(0, 480))).float()
    process_frame.uv1 = torch.cat((process_frame.uv1, torch.ones_like(process_frame.uv1[:1])), dim=0)
    return process_frame

    # # calculate normal
    # frame['nxyz'] = torch.zeros_like(frame['xyz'])
    # dz_thresh = 0.1
    # dzdx_p = frame['xyz'][:, 1:-1, 1:-1] - frame['xyz'][:, 1:-1, 2:]
    # dzdx_n = frame['xyz'][:, 1:-1, 1:-1] - frame['xyz'][:, 1:-1, :-2]
    # dzdy_p = frame['xyz'][:, 1:-1, 1:-1] - frame['xyz'][:, 2:, 1:-1]
    # dzdy_n = frame['xyz'][:, 1:-1, 1:-1] - frame['xyz'][:, :-2, 1:-1]
    # x_dir = (
    #     (abs(dzdx_p) < dz_thresh) * (abs(dzdx_n) < dz_thresh) * (dzdx_p + dzdx_n) / 2 +
    #     (abs(dzdx_p) < dz_thresh) * (abs(dzdx_n) >= dz_thresh) * dzdx_p +
    #     (abs(dzdx_p) >= dz_thresh) * (abs(dzdx_n) < dz_thresh) * dzdx_n
    # )
    # y_dir = (
    #     (abs(dzdy_p) < dz_thresh) * (abs(dzdy_n) < dz_thresh) * (dzdy_p + dzdy_n) / 2 +
    #     (abs(dzdy_p) < dz_thresh) * (abs(dzdy_n) >= dz_thresh) * dzdy_p +
    #     (abs(dzdy_p) >= dz_thresh) * (abs(dzdy_n) < dz_thresh) * dzdy_n
    # )
    # frame['nxyz'][:, 1:-1, 1:-1] = torch.cross(x_dir, y_dir, dim=0)
    # frame['nxyz'][:, 1:-1, 1:-1] /= torch.norm(frame['nxyz'][:, 1:-1, 1:-1], dim=0, keepdim=True) * 10
    # frame['nxyz'] *= -torch.sign(frame['nxyz'][[2]] + 1e-8)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(frame['xyz'].reshape(3, -1).permute(1, 0).numpy())
    # pcd.colors = o3d.utility.Vector3dVector(frame['rgb'].reshape(3, -1).permute(1, 0).numpy())
    # pcd.estimate_normals(fast_normal_computation=True)
    # normals = np.asarray(pcd.normals)
    # pcd.normals = o3d.utility.Vector3dVector(normals / 10)
    # # pcd.normals = o3d.utility.Vector3dVector(frame['nxyz'].reshape(3, -1).permute(1, 0).numpy())
    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    # if i_frame == 0:
    #     frame['mask'] = Segmenter((frame['rgb'].permute(1, 2, 0) * 255).numpy().astype(np.uint8)).mask
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(frame['xyz'].permute(1, 0).numpy())
    # pcd.colors = o3d.utility.Vector3dVector(frame['rgb'].reshape(3, -1).permute(1, 0).numpy())
    # o3d.visualization.draw_geometries([pcd])