

from matplotlib import pyplot as plt
import torch
from bundle_fetch.utils import nvtx_range
from bundle_fetch.track.solver_pytorch import solve


def get_pose_model():
    def pose_model(corr, frame):
        if True:#corr is None:
            max_masked_xyz = (frame['xyz'] * frame['mask'])
            min_masked_xyz = (frame['xyz'] * frame['mask'] + 999 * (1 - frame['mask']))
            avg = torch.sum(max_masked_xyz, dim=(1, 2)) / torch.sum(frame['mask'])
            o_T_c_a = torch.eye(4, device=frame['xyz'].device)
            o_T_c_a[:3, 3] = avg
            return o_T_c_a[None], None
        else:
            o_T_c_a, o_T_c_b = solve(
                corr['o_T_c_a'],
                corr['o_T_c_b'],
                corr['xyz1_a'],
                corr['xyz1_b']
            )

            return o_T_c_a, o_T_c_b
                
    
    return pose_model
