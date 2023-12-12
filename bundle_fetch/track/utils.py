import lietorch
from torchvision.transforms.functional import rgb_to_grayscale
import torch
from torch.nn.functional import grid_sample
from pytorch3d.transforms import matrix_to_quaternion
import open3d as o3d
from bundle_fetch.utils import nvtx_range


def get_mask(frame, xmem, n_objs):
    """
    Get mask from vertex
    """
    with nvtx_range('get_mask'):
        rgb = frame['rgb'].cuda()
        mask = frame.get('mask')
        if mask is None:
            if n_objs == 0:
                return None
            labels = None
        else:
            # mask = mask.cuda()
            # labels = torch.unique(mask)
            # labels = labels[labels!=0]
            # labels = torch.concatenate((prev_labels, labels))
            mask = mask + n_objs
            labels = torch.arange(1, n_objs+2, device=labels.device, dtype=labels.dtype, out=labels)
        prob = xmem.step(rgb, mask, labels)
        out_mask = torch.max(prob, dim=0).indices
        return out_mask


def get_features(frame, loftr):
    """
    Get features from frame
    """
    with nvtx_range('get_features'):
        data = {}
        data['image0'] = rgb_to_grayscale(frame['rgb'])[None].cuda()
        loftr.forward_backbone(data)
        feat_c = data['feat_c0']
        feat_f = data['feat_f0']
        hw_i = data['hw0_i']
        hw_c = data['hw0_c']
        hw_f = data['hw0_f']
        return feat_c, feat_f, hw_i, hw_c, hw_f


def get_covisibility(v, u):
    """
    Get covisibility between vertices
    """
    with nvtx_range('get_covisibility'):
        v_pv = v['frame']['xyz'] # (H, W, 3)
        v_nv = v['frame']['nxyz'] # (H, W, 3)
        v_mv = v['mask'] # (H, W)
        u_pv = u['o_T_c'].inv() * v['o_T_c'] * v_pv # (H, W, 3)
        u_nv = v['o_T_c'].inv() * u['o_T_c'] * v_nv # (H, W, 3)
        u_pv = u_pv / torch.norm(u_pv, dim=-1, keepdim=True) # (H, W, 3)
        u_nv = u_nv / torch.norm(u_nv, dim=-1, keepdim=True) # (H, W, 3)
        visible = torch.einsum('hwt, hwt -> hw', u_pv, u_nv) > 0.6 # (H, W)
        prop_visible = torch.sum(visible * v_mv) / torch.sum(v_mv)
        return prop_visible


def filter_edges(vertices, edges):
    """
    Filter edges
    """
    with nvtx_range('filter_edges'):
        edge_key, edge = list(edges.items())[0]
        vi = vertices[edge_key[0]]
        vj = vertices[edge_key[1]]
        o_T_i = vi['o_T_c'] # (6)

        i_si = edge['i_si'] # (S, 3)
        j_sj = edge['j_sj'] # (S, 3)
        conf = edge['conf'] # (S)
        S = i_si.shape[0]
        max_iter = 2000
        num_sample = 3

        samples = torch.multinomial((conf > 0).float().expand(S, -1), num_sample) # (max_iter, num_sample)
        i_pi = i_si[samples] # (max_iter, num_sample, 3)
        j_pj = j_sj[samples] # (max_iter, num_sample, 3)
        i_qi = i_pi - torch.mean(i_pi, dim=1, keepdim=True) # (max_iter, num_sample, 3)
        j_qj = j_pj - torch.mean(j_pj, dim=1, keepdim=True) # (max_iter, num_sample, 3)
        # print('i_qi', i_qi)
        # print('j_qj', j_qj)
        # input()

        # H = torch.einsum('msi, msj -> mij', i_qi, j_qj) # (max_iter, 3, 3)
        H = i_qi[..., None] * j_qj[..., None, :] # (max_iter, num_sample, 3, 3)
        H = torch.sum(H, dim=1) # (max_iter, 3, 3)
        U, _, Vh = torch.svd(H) # (max_iter, 3, 3), (max_iter, 3), (max_iter, 3, 3)
        R = Vh.transpose(1, 2) @ U.transpose(1, 2) # (max_iter, 3, 3)
        # R = torch.einsum('mij, mki -> mjk', Vh, U) # (max_iter, 3, 3)
        # print('Rj - i', R[:, None] @ j_qj[..., None] - i_qi[..., None])
        # print('Ri - j', R[:, None] @ i_qi[..., None] - j_qj[..., None])
        # input()
        t = torch.mean(j_pj, dim=1) - torch.einsum('mij, mj -> mi', R, torch.mean(i_pi, dim=1)) # (max_iter, 3)
        print('t', t)
        q_R = matrix_to_quaternion(R) # (max_iter, 4)
        j_T_i = lietorch.SE3.InitFromVec(torch.cat([t, q_R], dim=-1)) # (max_iter, 6)
        o_T_j = o_T_i[None] * j_T_i.inv() # (max_iter, 6)
        print('o_T_j', o_T_j.matrix())

        o_si = o_T_i[None, None] * i_si[None] # (1, S, 3)
        o_sj = o_T_j[:, None] * j_sj[None] # (max_iter, S, 3)

        print('o_si', o_si[:, conf > 0])
        print('j_sj', j_sj[conf > 0])
        print('o_sj', o_sj[:, conf > 0])

        o_d = torch.norm(o_si - o_sj, dim=-1) # (max_iter, S)
        o_d = torch.where(conf > 0, o_d, torch.ones_like(o_d)) # (max_iter, S)
        val_S = torch.sum(conf > 0) # ()
        print('o_d', o_d[conf > 0])
        n_inliers = torch.sum(o_d < 0.01, dim=-1) # (max_iter)
        best_n_inliers, best_idx  = torch.max(n_inliers, dim=-1) # (), ()
        print('best_prop_inliers', best_n_inliers.item() / val_S)
        vj['o_T_c'] = o_T_j[best_idx] # (6)
        print('best_o_T_j', vj['o_T_c'].data)
    

def sparse(o_T_v, Kv, ii, jj, i_si, j_sj, v_dv, v_nv, v_mv):
    """
    Compute sparse residual: o_T_i * i_si - o_T_j * j_sj
    """
    with nvtx_range('sparse'):
        # i_si: (E, S, 3)
        # j_sj: (E, S, 3)
        dtype = o_T_v.dtype
        device = o_T_v.device
        # o_T_v = lietorch.SE3.exp(o_T_v) # (V, 6)
        V = o_T_v.shape[0]
        E = ii.shape[0]
        S = i_si.shape[1]
        _, H, W = v_dv.shape[:3]

        o_T_i = o_T_v[ii] # (E, 6)
        o_T_j = o_T_v[jj] # (E, 6)

        o_si = o_T_i[:, None] * i_si # (E, S, 3)
        x, y, z = o_si.unbind(-1) # (E, S)
        w = torch.ones_like(x) # (E, S)
        o = torch.zeros_like(x) # (E, S)
        Ji = torch.stack([
            w,  o,  o,  o,  z, -y,
            o,  w,  o, -z,  o,  x, 
            o,  o,  w,  y, -x,  o,
        ]).view(E, S, 3, 6) # (E, S, 3, 6)

        o_sj = o_T_j[:, None] * j_sj # (E, S, 3)
        x, y, z = o_sj.unbind(-1) # (E, S)
        w = torch.ones_like(x) # (E, S)
        o = torch.zeros_like(x) # (E, S)
        Jj = -torch.stack([
            w,  o,  o,  o,  z, -y,
            o,  w,  o, -z,  o,  x,
            o,  o,  w,  y, -x,  o,
        ]).view(E, S, 3, 6) # (E, S, 3, 6)

        s = o_si - o_sj # (E, S, 3)
        return s, Ji, Jj


    def proj(K, x):
        """
        Project points x to image plane using intrinsics K
        """
        # K: (..., 3, 3)
        # x: (..., 3)
        p = torch.einsum('eij, e...i -> e...j', K, x)
        p = p[..., :2] / p[..., 2]
        return p


def dense(o_T_v, Kv, ii, jj, i_si, j_sj, v_dv, v_nv, v_mv):
    """
    Compute dense residual: i_ni * (i_di - i_dj)
    """
    with nvtx_range('dense'):
        # i_si: (E, S, 3)
        # j_sj: (E, S, 3)
        dtype = o_T_v.dtype
        device = o_T_v.device
        V = o_T_v.shape[0]
        E = ii.shape[0]
        S = i_si.shape[1]
        _, H, W = v_dv.shape[:3]

        o_T_i = o_T_v[ii] # (E, 6)
        o_T_j = o_T_v[jj] # (E, 6)
        Kj = Kv[jj] # (E, 3, 3)

        i_ni = v_nv[ii] # (E, H, W, 3)
        i_di = v_dv[ii] # (E, H, W, 3)
        j_dj = v_dv[jj] # (E, H, W, 3)

        j_di = o_T_j.inv() * o_T_i * i_di # (E, H, W, 3)
        j_pi = proj(j_di, Kj) # (E, H, W, 2) pixel coordinates
        j_dj = grid_sample(j_dj, j_pi) # (E, H, W, 3)
        i_dj = o_T_i.inv() * o_T_j * j_dj # (E, H, W, 3), (E, H, W, 3, 6)
        d = torch.dot(i_ni, i_di - i_dj) # (E, H, W)
        return d


# https://jaxopt.github.io/stable/_modules/jaxopt/_src/gauss_newton.html
# https://jaxopt.github.io/stable/_modules/jaxopt/_src/linear_solve.html
# def get_hessian(fun, params, shape, dtype, *args, **kwargs):
#     """Gauss Newton approximation of the Hessian matrix J.T @ J."""
#     fun_with_args = lambda p: fun(p, *args, **kwargs)
    
#     matvec = lambda v: torch.func.vjp(fun_with_args, params)[1](
#         torch.func.jvp(fun_with_args, (params,), (v,))[1]
#     )[0]

#     x = torch.zeros(size=shape, dtype=dtype)
#     H = torch.func.jacfwd(matvec)(x)
#     return H


# def get_gradient(fun, params, residual, *args, **kwargs):
#     """Gauss Newton approximation of the gradient: J.T @ r."""
#     fun_with_args = lambda p: fun(p, *args, **kwargs)
#     g = torch.func.vjp(fun_with_args, params)[1](residual)[0]

#     return g