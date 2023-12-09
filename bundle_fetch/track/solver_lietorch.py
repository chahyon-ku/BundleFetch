from bundle_fetch.utils import nvtx_range
import torch
import lietorch
import torch_scatter
from bundle_fetch.track.projective_ops import proj, actp
from torch.nn.functional import grid_sample

def solver(o_T_v, Kv, ii, jj, i_si, j_sj, v_dv, v_nv, v_mv):
    """
    Bundle Adjustment Solver

    V: number of vertices (frames)
    E: number of edges (pairs of frames)
    S: max number of condences per edge
    H: height of image
    W: width of image

    o_T_v: (V, 6) poses of vertices, converting points in camera n to object
    K_v: (V, 3, 3) intrinsics of vertices
    ii: (E,) src vertices of edges in [0, V)
    jj: (E,) dst vertices of edges in [0, V)
    i_si: (E, S, 3) sparse point correspondences in and from camera_i
    j_sj: (E, S, 3) sparse point correspondences in and from camera_j
    v_dv: (V, H, W, 3) dense points in and from camera_v
    v_nv: (V, H, W, 3) dense normals in and from camera_v
    v_mv: (V, H, W, 3) dense masks in and from camera_v
    """
    dtype = o_T_v.dtype
    device = o_T_v.device
    V = o_T_v.shape[0]
    E = ii.shape[0]
    S = i_si.shape[1]
    _, H, W = v_dv.shape[:3]

    o_T_i = o_T_v[ii] # (E, 6)
    o_T_j = o_T_v[jj] # (E, 6)

    # Sparse: ||o_T_i * i_si - o_T_j * j_sj||^2
    o_si, Jsi = actp(o_T_i, i_si, True) # (E, S, 3), (E, S, 3, 6)
    o_sj, Jsj = actp(o_T_j, j_sj, True) # (E, S, 3), (E, S, 3, 6)
    Jsj = -Jsj # (E, S, 3, 6)
    s = o_si - o_sj # (E, S, 3)
    S = torch.einsum('esi, esi -> es', s, s) # (E, S)

    # Jsv = torch_scatter.scatter_sum(Jsi, ii, dim=0, dim_size=V) + \
    #       - torch_scatter.scatter_sum(Jsj, jj, dim=0, dim_size=V) # (E, 3, 6)

    # Dense: (i_ni * (i_di - o_T_i.inv() * o_T_j * j_di)^2
    i_ni = v_nv[ii] # (E, H, W, 3)
    i_di = v_dv[ii] # (E, H, W, 3)
    j_dj = v_dv[jj] # (E, H, W, 3)
    j_T_i = o_T_j.inv() * o_T_i # (E, 6)
    i_T_j = j_T_i.inv() # (E, 6)

    j_di = j_T_i[:, None, None] * i_di # (E, H, W, 3)
    j_dj = grid_sample(j_dj, proj(j_di, Kv[jj])) # (E, H, W, 3)
    i_dj, Jdij = actp(i_T_j[:, None, None], j_dj, True) # (E, H, W, 3), (E, H, W, 3, 6)
    d = i_di - i_dj # (E, H, W, 3)
    D = torch.dot(i_ni, d).pow(2).sum(dim=-1) # (B, N)

    # (exp(eps_i) * o_T_i).inv() * exp(eps_j) * o_T_j * j_di
    # o_T_i.inv() * exp(-eps_i) * exp(eps_j) * o_T_j * j_di
    # exp(-Adj(o_T_i.inv()) * eps_i) * o_T_i.inv() * exp(eps_j) * o_T_j * j_di
    # exp(-Adj(o_T_i.inv()) * eps_i) * exp(Adj(o_T_i.inv()) * eps_j) * o_T_i.inv() * o_T_j * j_di
    Jdi = torch.einsum('ehwi, ehwij -> e(hw)j', i_ni, -o_T_i.inv().AdjT(Jdij)) # (E, HW, 3, 6)
    Jdj = torch.einsum('ehwi, ehwij -> e(hw)j', i_ni, o_T_i.inv().AdjT(Jdij)) # (E, HW, 3, 6)
    # Jdv = torch_scatter.scatter_sum(Jdi, ii, dim=0, dim_size=V) + \
    #     torch_scatter.scatter_sum(Jdj, jj, dim=0, dim_size=V) # (V, 6)
    
    Hii = torch.einsum('esij, esik -> esjk', Jsi, Jsi) # (E, S, 6, 6)
    Hij = torch.einsum('esij, esik -> esjk', Jsi, Jsj) # (E, S, 6, 6)
    Hji = torch.einsum('esij, esik -> esjk', Jsj, Jsi) # (E, S, 6, 6)
    Hjj = torch.einsum('esij, esik -> esjk', Jsj, Jsj) # (E, S, 6, 6)

    vi = torch.einsum('esij, esi -> esj', Jsi, s) # (E, S, 6)
    vj = torch.einsum('esij, esi -> esj', Jsj, s) # (E, S, 6)

    # only optimize keyframe poses
    P = P // rig - fixedp
    ii = ii // rig - fixedp
    jj = jj // rig - fixedp

    H = safe_scatter_add_mat(Hii, ii, ii, P, P) + \
        safe_scatter_add_mat(Hij, ii, jj, P, P) + \
        safe_scatter_add_mat(Hji, jj, ii, P, P) + \
        safe_scatter_add_mat(Hjj, jj, jj, P, P)

    v = safe_scatter_add_vec(vi, ii, P) + \
        safe_scatter_add_vec(vj, jj, P)
    
    H = H.view(B, P, P, D, D)

    ### 3: solve the system ###
    dx = block_solve(H, v)

    ### 4: apply retraction ###
    poses = pose_retr(poses, dx, torch.arange(P) + fixedp)
    
    # deriv for o_T_i: (A * e^e * D)^{-1} * p; A = Tj^{-1}; D = Ti
    # x, y, z, d = i_dj.unbind(dim=-1) # (E, H, W)
    # o = torch.zeros_like(d) # (E, H, W)
    # j_R_i = j_T_i.matrix()[..., :3, :3] # (E, 3, 3)
    # j_t_i = j_T_i.matrix()[..., 3, :3] # (E, 3)
    # x0 = x - j_t_i[:, None, None, 0] # (E, H, W)
    # y0 = y - j_t_i[:, None, None, 1] # (E, H, W)
    # z0 = z - j_t_i[:, None, None, 2] # (E, H, W)
    # r00, r01, r02 = j_R_i[:, 0, 0], j_R_i[:, 0, 1], j_R_i[:, 0, 2] # (E,)
    # r10, r11, r12 = j_R_i[:, 1, 0], j_R_i[:, 1, 1], j_R_i[:, 1, 2] # (E,)
    # r20, r21, r22 = j_R_i[:, 2, 0], j_R_i[:, 2, 1], j_R_i[:, 2, 2] # (E,)
    # Jdi0 = torch.stack([
    #     x, y, z, o, o, o, o, o, o, -r00, -r10, -r20,
    #     o, o, o, x, y, z, o, o, o, -r01, -r11, -r21,
    #     o, o, o, o, o, o, x, y, z, -r02, -r12, -r22,
    # ]).reshape(E, H, W, 3, 12) # (E, H, W, 3, 12)
    # jdi1 = torch.stack([
    #     o, o, o,
    # ])

    # # deriv for o_T_j: (A * e^e * D) * p; A = Ti^{-1}; D = Tj
    # x, y, z, d = i_dj.unbind(dim=-1) # (V, H, W)
    # o = torch.zeros_like(d) # (V, H, W)
    # o_R_i = o_T_i.matrix()[..., :3, :3] # (V, 3, 3)
    # Jdj = o_R_i * torch.stack([
    #     d,  o,  o,  o,  z, -y,
    #     o,  d,  o, -z,  o,  x, 
    #     o,  o,  d,  y, -x,  o,
    # ], dim=-1).view(V, H, W, 3, 6) # (V, H, W, 3, 6)