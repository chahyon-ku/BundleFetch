# https://github.com/princeton-vl/DROID-SLAM/blob/main/droid_slam/geom/ba.py


import lietorch
import torch
import torch.nn.functional as F

from .chol import block_solve, schur_solve
import geom.projective_ops as pops

from torch_scatter import scatter_sum


# utility functions for scattering ops
def safe_scatter_add_mat(A, ii, jj, n, m):
    v = (ii >= 0) & (jj >= 0) & (ii < n) & (jj < m)
    return scatter_sum(A[:,v], ii[v]*m + jj[v], dim=1, dim_size=n*m)

def safe_scatter_add_vec(b, ii, n):
    v = (ii >= 0) & (ii < n)
    return scatter_sum(b[:,v], ii[v], dim=1, dim_size=n)

# apply retraction operator to inv-depth maps
def disp_retr(disps, dz, ii):
    ii = ii.to(device=dz.device)
    return disps + scatter_sum(dz, ii, dim=1, dim_size=disps.shape[1])

# apply retraction operator to poses
def pose_retr(poses, dx, ii):
    ii = ii.to(device=dx.device)
    return poses.retr(scatter_sum(dx, ii, dim=1, dim_size=poses.shape[1]))


def MoBA(target, weight, eta, poses, disps, intrinsics, ii, jj, fixedp=1, rig=1):
    """ Motion only bundle adjustment """

    B, P, ht, wd = disps.shape
    N = ii.shape[0]
    D = poses.manifold_dim

    ### 1: commpute jacobians and residuals ###
    coords, valid, (Ji, Jj, Jz) = pops.projective_transform(
        poses, disps, intrinsics, ii, jj, jacobian=True)

    r = (target - coords).view(B, N, -1, 1) # (B, N, HW2, 1)
    w = .001 * (valid * weight).view(B, N, -1, 1) # (B, N, HW2, 1)

    ### 2: construct linear system ###
    Ji = Ji.reshape(B, N, -1, D) # (B, N, HW2, 6)
    Jj = Jj.reshape(B, N, -1, D) # (B, N, HW2, 6)
    wJiT = (w * Ji).transpose(2,3) # (B, N, 6, HW2)
    wJjT = (w * Jj).transpose(2,3) # (B, N, 6, HW2)

    Hii = torch.matmul(wJiT, Ji) # (B, N, 6, 6)
    Hij = torch.matmul(wJiT, Jj) # (B, N, 6, 6)
    Hji = torch.matmul(wJjT, Ji) # (B, N, 6, 6)
    Hjj = torch.matmul(wJjT, Jj) # (B, N, 6, 6)

    vi = torch.matmul(wJiT, r).squeeze(-1) # (B, N, 6)
    vj = torch.matmul(wJjT, r).squeeze(-1) # (B, N, 6)

    # only optimize keyframe poses
    P = P // rig - fixedp
    ii = ii // rig - fixedp
    jj = jj // rig - fixedp

    H = safe_scatter_add_mat(Hii, ii, ii, P, P) + \
        safe_scatter_add_mat(Hij, ii, jj, P, P) + \
        safe_scatter_add_mat(Hji, jj, ii, P, P) + \
        safe_scatter_add_mat(Hjj, jj, jj, P, P) # (B, P, P, 6, 6)

    v = safe_scatter_add_vec(vi, ii, P) + \
        safe_scatter_add_vec(vj, jj, P) # (B, P, 6)
    
    H = H.view(B, P, P, D, D)

    ### 3: solve the system ###
    dx = block_solve(H, v)

    ### 4: apply retraction ###
    poses = pose_retr(poses, dx, torch.arange(P) + fixedp)
    return poses

