from bundle_fetch.utils import nvtx_range
import torch
from bundle_fetch.track.se3 import residual, jacobian
from pytorch3d.transforms import se3_exp_map


def solve(o_T_c_a, o_T_c_b, xyz1_a, xyz1_b):
    with torch.inference_mode():
        # o_T_c_a: (N, 4, 4)
        # o_T_c_b: (N, 4, 4)
        # xyz1_a: (N, M, 4, 1)
        # xyz1_b: (N, M, 4, 1)

        o_T_c = torch.cat([o_T_c_a, o_T_c_b], dim=0) # (2N, 4, 4)
        xyz1 = torch.cat([xyz1_a, -xyz1_b], dim=0) # (2N, M, 4, 1)
        N = o_T_c_a.shape[0]
        for i_nonlin in range(1):
            with nvtx_range('nonlin'):
                J_var = -jacobian(o_T_c, xyz1) # (2N, M, 3, 6)
                J_var_T = J_var.transpose(2, 3) # (2N, M, 6, 3)
                r_var = residual(o_T_c, xyz1) # (2N, M, 3, 1)

                A = torch.einsum('nmij,nmjk->nik', J_var_T,  J_var) # (2N, 6, 6)
                b = torch.einsum('nmij,nmjk->nik', J_var_T,  r_var) # (2N, 6, 1)
                x = torch.zeros_like(b) # (2N, 6, 1)
                r = b# - A @ x # (2N, 6, 1)
                M_inv = torch.diagonal(A, offset=0, dim1=1, dim2=2)[..., None] * r # (2N, 6, 1)
                M_inv = torch.where(M_inv < 1e-8, torch.ones_like(M_inv), 1 / M_inv)
                z = M_inv * r # (2N, 6, 1)
                p = z # (2N, 6, 1)
                
                for i_lin in range(1):
                    Ap = A @ p # (2N, 6, 1)
                    alpha_num = r.transpose(1, 2) @ z # (2N, 1, 1)
                    alpha_den = p.transpose(1, 2) @ Ap # (2N, 1, 1)
                    alpha = torch.where(alpha_den < 1e-8, torch.zeros_like(alpha_num), alpha_num / alpha_den)
                    x = x + alpha * p
                    r = r - alpha * Ap
                    z = M_inv * r
                    beta_num = r.transpose(1, 2) @ z
                    beta = torch.where(alpha_num < 1e-8, torch.zeros_like(beta_num), beta_num / alpha_num)
                    p = z + beta * p
                
                d_o_T_c_a = se3_exp_map(torch.sum(x[:N, ..., 0], dim=0, keepdim=True))
                d_o_T_c_a = d_o_T_c_a.permute(0, 2, 1)
                d_o_T_c_b = se3_exp_map(x[N:, ..., 0])
                d_o_T_c_b = d_o_T_c_b.permute(0, 2, 1)
                print(d_o_T_c_a)
                print(d_o_T_c_b)
                o_T_c[:N] = o_T_c[:N] @ d_o_T_c_a # (N, 4, 4)
                o_T_c[N:] = o_T_c[N:] @ d_o_T_c_b # (N, 4, 4)
            
        # log -> transform
        return o_T_c[:N], o_T_c[N:]