from bundle_fetch.utils import nvtx_range
import torch
from bundle_fetch.track.se3 import residual, jacobian
from pytorch3d.transforms import se3_exp_map, se3_log_map


def solve(o_T_c_a, o_T_c_b, xyz1_a, xyz1_b):
    # o_T_c_a: (N, 4, 4)
    # o_T_c_b: (N, 4, 4)
    # xyz1_a: (N, M, 4, 1)
    # xyz1_b: (N, M, 4, 1)
    o_T_c = torch.cat([o_T_c_a, o_T_c_b], dim=0).detach().requires_grad_() # (2N, 4, 4)
    o_se3_c = se3_log_map(o_T_c.permute(0, 2, 1)).detach().requires_grad_() # (2N, 6)
    xyz1 = torch.cat([xyz1_a, -xyz1_b], dim=0).requires_grad_() # (2N, M, 4, 1)
    N = o_T_c_a.shape[0]
    for i_nonlin in range(10):
        o_T_c = se3_exp_map(o_se3_c).permute(0, 2, 1) # (2N, 4, 4)
        xyz1_o = torch.einsum('nij,nmjk->nmik', o_T_c, xyz1) # (2N, M, 4, 1)
        residual = torch.pow(xyz1_o[:N, :3] - xyz1_o[N:, :3], 2).mean()
        print(f'nonlin {i_nonlin} residual {residual.item()}')
        residual.backward()
        o_se3_c.data -= o_se3_c.grad
        o_se3_c.grad.zero_()

    o_T_c = se3_exp_map(o_se3_c).permute(0, 2, 1) # (2N, 4, 4)
    print(o_T_c[0])
    # log -> transform
    return o_T_c[:N].detach(), o_T_c[N:].detach()