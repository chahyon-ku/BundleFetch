from bundle_fetch.utils import nvtx_range
import torch
from bundle_fetch.track.se3 import residual, jacobian
from pytorch3d.transforms import se3_exp_map, se3_log_map


def solve(o_T_c_a, o_T_c_b, xyz1_a, xyz1_b):
    # o_T_c_a: (N, 4, 4)
    # o_T_c_b: (N, 4, 4)
    # xyz1_a: (N, M, 4, 1)
    # xyz1_b: (N, M, 4, 1)
    # o_T_c = torch.cat([o_T_c_a, o_T_c_b], dim=0).detach().requires_grad_() # (2N, 4, 4)
    o_se3_c_a = se3_log_map(o_T_c_a[[0]].permute(0, 2, 1)).detach().requires_grad_() # (6)
    o_se3_c_b = se3_log_map(o_T_c_b.permute(0, 2, 1)).detach().requires_grad_() # (2N, 6)
    xyz1_a = xyz1_a.clone().detach().requires_grad_()
    xyz1_b = xyz1_b.clone().detach().requires_grad_()
    # xyz1 = torch.cat([xyz1_a, -xyz1_b], dim=0).requires_grad_() # (2N, M, 4, 1)
    N = o_T_c_a.shape[0]
    for i_nonlin in range(10):
        o_T_c_a = se3_exp_map(o_se3_c_a).permute(0, 2, 1)[0] # (4, 4)
        o_T_c_b = se3_exp_map(o_se3_c_b).permute(0, 2, 1)
        xyz1_a_o = torch.einsum('ij,nmjk->nmik', o_T_c_a, xyz1_a)
        xyz1_b_o = torch.einsum('nij,nmjk->nmik', o_T_c_b, xyz1_b)
        residual = xyz1_a_o[..., :3, 0] - xyz1_b_o[..., :3, 0]
        residual = (residual * residual).sum()
        print(o_se3_c_a, residual.item())
        # print(f'nonlin {i_nonlin} residual {residual.item()}')
        residual.backward()
        o_se3_c_a.data -= 0.001 * o_se3_c_a.grad
        o_se3_c_a.grad.zero_()
        # o_se3_c_b.data -= 10 * o_se3_c_b.grad
        # o_se3_c_b.grad.zero_()

    o_T_c_a = se3_exp_map(o_se3_c_a).permute(0, 2, 1)[0] # (4, 4)
    o_T_c_b = se3_exp_map(o_se3_c_b).permute(0, 2, 1) # (N, 4, 4)
    # log -> transform
    return o_T_c_a.detach()[None], o_T_c_b.detach()