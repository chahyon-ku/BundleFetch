

from matplotlib import pyplot as plt
import torch
from bundle_fetch.utils import nvtx_range
from bundle_fetch.track.solver_pytorch2 import solve
from lietorch import SE3


def get_pose_model():
    def pose_model(corr, frame):
        if corr is None:
            max_masked_xyz = (frame['xyz'] * frame['mask'])
            min_masked_xyz = (frame['xyz'] * frame['mask'] + 999 * (1 - frame['mask']))
            o_T_c_a = torch.sum(max_masked_xyz, dim=(1, 2)) / torch.sum(frame['mask']) # (3)
            o_T_c_a = torch.concat([
                o_T_c_a,
                torch.zeros(3, device=o_T_c_a.device),
                torch.ones(1, device=o_T_c_a.device)],
                dim=-1
            ).requires_grad_() # attach unit quaternion
            o_T_c_a = SE3.InitFromVec(o_T_c_a[None])
            print(o_T_c_a.shape, o_T_c_a.data.shape, frame['xyz'].shape, frame['xyz'].data.shape)
            return o_T_c_a, None
        else:
            o_T_c_a, o_T_c_b = solve(**corr)

            return o_T_c_a, o_T_c_b
                
    return pose_model
