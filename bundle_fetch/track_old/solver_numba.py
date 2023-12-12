from bundle_fetch.utils import nvtx_range
import torch
from bundle_fetch.track.se3 import _se3_exp_map_numba, transform, jacobian, se3_exp_map, se3_exp_map_numba
from numba import cuda


@cuda.jit
def _transform(o_T_c, xyz1, xyz):
    i = cuda.grid(1)
    if i < N:
        for j in range(xyz1.shape[1]):
            cuda.atomic.add(xyz, (i, 0, 0), o_T_c[i, 0, 0] * xyz1[i, j, 0, 0] + o_T_c[i, 0, 1] * xyz1[i, j, 1, 0] + o_T_c[i, 0, 2] * xyz1[i, j, 2, 0] + o_T_c[i, 0, 3] * xyz1[i, j, 3, 0])
            cuda.atomic.add(xyz, (i, 1, 0), o_T_c[i, 1, 0] * xyz1[i, j, 0, 0] + o_T_c[i, 1, 1] * xyz1[i, j, 1, 0] + o_T_c[i, 1, 2] * xyz1[i, j, 2, 0] + o_T_c[i, 1, 3] * xyz1[i, j, 3, 0])
            cuda.atomic.add(xyz, (i, 2, 0), o_T_c[i, 2, 0] * xyz1[i, j, 0, 0] + o_T_c[i, 2, 1] * xyz1[i, j, 1, 0] + o_T_c[i, 2, 2] * xyz1[i, j, 2, 0] + o_T_c[i, 2, 3] * xyz1[i, j, 3, 0])

@cuda.jit
def _jacobian(T, xyz, J):
    # T: (N, 4, 4)
    # xyz: (N, 3, 1)
    # J: (N, 3, 6)
    i = cuda.grid(1)
    if i < T.shape[0]:
        J[i, 0, 0] = 1
        J[i, 0, 1] = 0
        J[i, 0, 2] = 0
        J[i, 0, 3] = 0
        J[i, 0, 4] = -xyz[i, 2, 0]
        J[i, 0, 5] = xyz[i, 1, 0]
        J[i, 1, 0] = 0
        J[i, 1, 1] = 1
        J[i, 1, 2] = 0
        J[i, 1, 3] = xyz[i, 2, 0]
        J[i, 1, 4] = 0
        J[i, 1, 5] = -xyz[i, 0, 0]
        J[i, 2, 0] = 0
        J[i, 2, 1] = 0
        J[i, 2, 2] = 1
        J[i, 2, 3] = -xyz[i, 1, 0]
        J[i, 2, 4] = xyz[i, 0, 0]
        J[i, 2, 5] = 0
    return J


@cuda.jit
def _transpose(M, M_T):
    i = cuda.grid(1)
    if i < M.shape[0]:
        for j in range(M.shape[1]):
            for k in range(M.shape[2]):
                M_T[i, k, j] = M[i, j, k]


@cuda.jit
def _matmul(M, N, MN):
    i = cuda.grid(1)
    if i < M.shape[0]:
        for j in range(M.shape[1]):
            for k in range(N.shape[2]):
                MN[i, j, k] = 0
                for l in range(M.shape[2]):
                    MN[i, j, k] += M[i, j, l] * N[i, l, k]

@cuda.jit
def _zero(M):
    i = cuda.grid(1)
    if i < M.shape[0]:
        for j in range(M.shape[1]):
            for k in range(M.shape[2]):
                M[i, j, k] = 0

@cuda.jit
def _assign(M, N):
    i = cuda.grid(1)
    if i < M.shape[0]:
        for j in range(M.shape[1]):
            for k in range(M.shape[2]):
                N[i, j, k] = M[i, j, k]


@cuda.jit
def _M_inv(M, M_diag):
    i = cuda.grid(1)
    if i < M.shape[0]:
        for j in range(M.shape[1]):
            if M[i, j, j] < 1e-8:
                M_diag[i, j, 0] = 0
            else:
                M_diag[i, j, 0] = 1 / M[i, j, j]

@cuda.jit
def _elemul(M, N, MN):
    i = cuda.grid(1)
    if i < M.shape[0]:
        for j in range(M.shape[1]):
            for k in range(M.shape[2]):
                MN[i, j, k] = M[i, j, k] * N[i, j, k]

@cuda.jit
def _elediv(numerator, denominator, result):
    i = cuda.grid(1)
    if i < numerator.shape[0]:
        if denominator[i, 0, 0] < 1e-8:
            result[i, 0, 0] = 0
        else:
            result[i, 0, 0] = numerator[i, 0, 0] / denominator[i, 0, 0]


@cuda.jit
def _elesum(M, N, MN):
    i = cuda.grid(1)
    if i < M.shape[0]:
        for j in range(M.shape[1]):
            for k in range(M.shape[2]):
                MN[i, j, k] = M[i, j, k] + N[i, j, k]

@cuda.jit
def _elesub(M, N, MN):
    i = cuda.grid(1)
    if i < M.shape[0]:
        for j in range(M.shape[1]):
            for k in range(M.shape[2]):
                MN[i, j, k] = M[i, j, k] - N[i, j, k]

@cuda.jit
def _reduce_half(M, M_reduced):
    i = cuda.grid(1)
    if i < M.shape[0] // 2:
        for j in range(M.shape[1]):
            for k in range(M.shape[2]):
                cuda.atomic.add(M_reduced, (0, j, k), M[i, j, k])

@cuda.jit
def _expand_half(M, M_expanded):
    i = cuda.grid(1)
    if i < M.shape[0] // 2:
        for j in range(M.shape[1]):
            for k in range(M.shape[2]):
                M_expanded[i, j, k] = M[0, j, k]


@cuda.jit
def _pre(o_T_c, xyz1, xyz, J, J_T, A, b, x_k, r_k, z_k, p_k, M_inv, Ap_k, alpha_num, alpha_den, alpha_k, alphap_k, alphaAp_k, rT_k, beta_num, beta_k, betap_k, x_k_a, o_T_c_d):
    _jacobian(o_T_c, xyz, J) # (N, 3, 6)
    _transpose(J, J_T) # (N, 6, 3)
    _matmul(J_T, J, A) # (N, 6, 6)
    _matmul(J_T, xyz, b) # (N, 6, 1)
    _zero(x_k) # (N, 6, 1)
    _assign(b, r_k) # (N, 6, 1)
    _M_inv(A, M_inv) # (N, 6, 1)
    _elemul(M_inv, r_k, z_k) # (N, 6, 1)
    _assign(z_k, p_k) # (N, 6, 1)

@cuda.jit
def _iter(o_T_c, xyz1, xyz, J, J_T, A, b, x_k, r_k, z_k, p_k, M_inv, Ap_k, alpha_num, alpha_den, alpha_k, alphap_k, alphaAp_k, rT_k, beta_num, beta_k, betap_k, x_k_a, o_T_c_d):
    _matmul(A, p_k, Ap_k)
    _matmul(r_k, z_k, alpha_num)
    _matmul(p_k, Ap_k, alpha_den)
    _elediv(alpha_num, alpha_den, alpha_k)
    _elemul(alpha_k, p_k, alphap_k)
    _elesum(x_k, alphap_k, x_k)
    _elemul(alpha_k, Ap_k, alphaAp_k)
    _elesub(r_k, alphaAp_k, r_k)
    _elemul(M_inv, r_k, z_k)
    _transpose(r_k, rT_k)
    _matmul(rT_k, z_k, beta_num)
    _elediv(beta_num, alpha_den, beta_k)
    _elemul(beta_k, p_k, betap_k)
    _elesum(z_k, betap_k, p_k)

@cuda.jit
def _post(o_T_c, xyz1, xyz, J, J_T, A, b, x_k, r_k, z_k, p_k, M_inv, Ap_k, alpha_num, alpha_den, alpha_k, alphap_k, alphaAp_k, rT_k, beta_num, beta_k, betap_k, x_k_a, o_T_c_d):
    _zero(x_k_a)
    _reduce_half(x_k[:N // 2], x_k_a)
    _expand_half(x_k_a, x_k[:N // 2])
    _se3_exp_map_numba(x_k, o_T_c_d)
    _matmul(o_T_c, o_T_c_d, o_T_c)


@cuda.jit
def _solve(o_T_c, xyz1, xyz, J, J_T, A, b, x_k, r_k, z_k, p_k, M_inv, Ap_k, alpha_num, alpha_den, alpha_k, alphap_k, alphaAp_k, rT_k, beta_num, beta_k, betap_k, x_k_a, o_T_c_d):
    for i_nonlin in range(7):
        _pre(o_T_c, xyz1, xyz, J, J_T, A, b, x_k, r_k, z_k, p_k, M_inv, Ap_k, alpha_num, alpha_den, alpha_k, alphap_k, alphaAp_k, rT_k, beta_num, beta_k, betap_k, x_k_a, o_T_c_d)
        for i_lin in range(3):
            _iter(o_T_c, xyz1, xyz, J, J_T, A, b, x_k, r_k, z_k, p_k, M_inv, Ap_k, alpha_num, alpha_den, alpha_k, alphap_k, alphaAp_k, rT_k, beta_num, beta_k, betap_k, x_k_a, o_T_c_d)
        _post(o_T_c, xyz1, xyz, J, J_T, A, b, x_k, r_k, z_k, p_k, M_inv, Ap_k, alpha_num, alpha_den, alpha_k, alphap_k, alphaAp_k, rT_k, beta_num, beta_k, betap_k, x_k_a, o_T_c_d)

N = 10
M = 100

def solve(o_T_c_a, o_T_c_b, xyz1_a, xyz1_b):
    # o_T_c: (N, 4, 4)
    # xyz1: (N, M, 4, 1)
    # xyz: (N, 3, 1)
    # J: (N, 3, 6)
    # J_T: (N, 6, 3)
    # A: (N, 6, 6)
    # b: (N, 6, 1)
    # x_k: (N, 6, 1)
    # r_k: (N, 6, 1)
    # z_k: (N, 6, 1)
    # p_k: (N, 6, 1)
    # M_inv: (N, 6, 1)
    # Ap_k: (N, 6, 1)
    # alpha_num: (N, 1, 1)
    # alpha_den: (N, 1, 1)
    # alpha_k: (N, 1, 1)
    o_T_c = torch.cat([o_T_c_a, o_T_c_b], dim=0) # (2N, 4, 4)
    xyz1 = torch.cat([xyz1_a, -xyz1_b], dim=0) # (2N, M, 4, 1)
    N = o_T_c.shape[0]
    xyz = torch.zeros((N, 3, 1), dtype=o_T_c.dtype, device=o_T_c.device)
    J = torch.zeros((N, 3, 6), dtype=o_T_c.dtype, device=o_T_c.device)
    J_T = torch.zeros((N, 6, 3), dtype=o_T_c.dtype, device=o_T_c.device)
    A = torch.zeros((N, 6, 6), dtype=o_T_c.dtype, device=o_T_c.device)
    b = torch.zeros((N, 6, 1), dtype=o_T_c.dtype, device=o_T_c.device)
    x_k = torch.zeros((N, 6, 1), dtype=o_T_c.dtype, device=o_T_c.device)
    r_k = torch.zeros((N, 6, 1), dtype=o_T_c.dtype, device=o_T_c.device)
    z_k = torch.zeros((N, 6, 1), dtype=o_T_c.dtype, device=o_T_c.device)
    p_k = torch.zeros((N, 6, 1), dtype=o_T_c.dtype, device=o_T_c.device)
    M_inv = torch.zeros((N, 6, 1), dtype=o_T_c.dtype, device=o_T_c.device)
    Ap_k = torch.zeros((N, 6, 1), dtype=o_T_c.dtype, device=o_T_c.device)
    alpha_num = torch.zeros((N, 1, 1), dtype=o_T_c.dtype, device=o_T_c.device)
    alpha_den = torch.zeros((N, 1, 1), dtype=o_T_c.dtype, device=o_T_c.device)
    alpha_k = torch.zeros((N, 1, 1), dtype=o_T_c.dtype, device=o_T_c.device)
    alphap_k = torch.zeros((N, 1, 1), dtype=o_T_c.dtype, device=o_T_c.device)
    alphaAp_k = torch.zeros((N, 1, 1), dtype=o_T_c.dtype, device=o_T_c.device)
    rT_k = torch.zeros((N, 1, 6), dtype=o_T_c.dtype, device=o_T_c.device)
    beta_num = torch.zeros((N, 1, 1), dtype=o_T_c.dtype, device=o_T_c.device)
    beta_k = torch.zeros((N, 1, 1), dtype=o_T_c.dtype, device=o_T_c.device)
    betap_k = torch.zeros((N, 1, 1), dtype=o_T_c.dtype, device=o_T_c.device)
    x_k_a = torch.zeros((1, 6, 1), dtype=o_T_c.dtype, device=o_T_c.device)
    o_T_c_d = torch.zeros((N, 4, 4), dtype=o_T_c.dtype, device=o_T_c.device)

    _solve[1, N](o_T_c, xyz1, xyz, J, J_T, A, b, x_k, r_k, z_k, p_k, M_inv, Ap_k, alpha_num, alpha_den, alpha_k, alphap_k, alphaAp_k, rT_k, beta_num, beta_k, betap_k, x_k_a, o_T_c_d)

    return o_T_c[:N], o_T_c[N:]
