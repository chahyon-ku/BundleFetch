from bundle_fetch.utils import nvtx_range
import torch
import lietorch
import torch_scatter
from bundle_fetch.track.projective_ops import proj, actp
from torch.nn.functional import grid_sample
from torch.func import jacrev


def sparse(o_T_v, Kv, ii, jj, i_si, j_sj, v_dv, v_nv, v_mv):
    dtype = o_T_v.dtype
    device = o_T_v.device
    V = o_T_v.shape[0]
    E = ii.shape[0]
    S = i_si.shape[1]
    _, H, W = v_dv.shape[:3]

    o_T_i = o_T_v[ii] # (E, 6)
    o_T_j = o_T_v[jj] # (E, 6)

    # Sparse: ||o_T_i * i_si - o_T_j * j_sj||^2
    s = o_T_i * i_si - o_T_j * j_sj # (E, S, 3)
    S = torch.einsum('esi, esi -> es', s, s) # (E, S)
    return S

def dense(o_T_v, Kv, ii, jj, i_si, j_sj, v_dv, v_nv, v_mv):
    dtype = o_T_v.dtype
    device = o_T_v.device
    V = o_T_v.shape[0]
    E = ii.shape[0]
    S = i_si.shape[1]
    _, H, W = v_dv.shape[:3]

    o_T_i = o_T_v[ii] # (E, 6)
    o_T_j = o_T_v[jj] # (E, 6)

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
    return D

def solve(o_T_v, Kv, ii, jj, i_si, j_sj, v_dv, v_nv, v_mv):
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
    
    get_JS = jacrev(sparse, argnums=0)
    get_JD = jacrev(dense, argnums=0)

    S = sparse(o_T_v, Kv, ii, jj, i_si, j_sj, v_dv, v_nv, v_mv)
    D = dense(o_T_v, Kv, ii, jj, i_si, j_sj, v_dv, v_nv, v_mv)
    JS = get_JS(o_T_v, Kv, ii, jj, i_si, j_sj, v_dv, v_nv, v_mv)
    JD = get_JD(o_T_v, Kv, ii, jj, i_si, j_sj, v_dv, v_nv, v_mv)

    print('S', S.shape)
    print('D', D.shape)
    print('JS', JS.shape)
    print('JD', JD.shape)
