import math
from bundle_fetch.utils import nvtx_range
import torch
from numba import cuda

def hat(v):
    # v: (..., 3, 1)
    # H: (..., 3, 3)
    H = torch.zeros((*v.shape[:-2], 3, 3), dtype=v.dtype, device=v.device)
    H[..., 2, 1] = v[..., 0, 0]
    H[..., 0, 2] = v[..., 1, 0]
    H[..., 1, 0] = v[..., 2, 0]
    H[..., 1, 2] = -v[..., 0, 0]
    H[..., 2, 0] = -v[..., 1, 0]
    H[..., 0, 1] = -v[..., 2, 0]

    return H


def jacobian(T, xyz1):
    # T: (2N, 4, 4)
    # xyz1: (2N, M, 4, 1)
    # J: (2N, M, 3, 6)
    twoN, M = xyz1.shape[:2]
    xyz1 *= (xyz1[:, 2:3, :] > 0) # mask out invalid points
    xyz = torch.einsum('nij,nmjk->nmik', T, xyz1)[:, :, :3] # (2N, M, 3, 1)
    J = torch.cat([
        torch.eye(3, dtype=T.dtype, device=T.device).expand(twoN, M, -1, -1), # (2N, M, 3, 3)
        -hat(xyz) # (2N, M, 3, 3)
    ], dim=-1)
    return J


def residual(T, xyz1):
    # T: (2N, 4, 4)
    # xyz1: (2N, M, 4, 1)
    # r: (2N, M, 3, 1)
    N = T.shape[0] // 2
    xyz1 *= (xyz1[:, 2:3, :] > 0) # mask out invalid points
    xyz = torch.einsum('nij,nmjk->nmik', T, xyz1)[:, :, :3] # (2N, M, 3, 1)
    r = torch.zeros_like(xyz) # (2N, M, 3, 1)
    r[:N] = xyz[:N] - xyz[N:]
    r[N:] = xyz[N:] - xyz[:N]
    return r


def se3_exp_map(x):
    with nvtx_range('se3_exp_map'):
        # x: (N, 6, 1)
        # T: (N, 4, 4)
        omega = x[:, 3:] # (N, 3, 1)
        theta_square = torch.sum(omega**2, dim=1, keepdim=True) # (N, 1, 1)
        theta = torch.sqrt(theta_square) # (N, 1, 1)


        # if theta_square < 1e-8:
        #     A = 1.0 - theta_square / 6
        #     B = 0.5
        #     t0 = ln_t0 + rt0 / 2
        #     t1 = ln_t1 + rt1 / 2
        #     t2 = ln_t2 + rt2 / 2
        # else:
        #     if theta_square < 1e-6:
        #         C = (1 - theta_square / 20) / 6
        #         A = 1 - theta_square * C
        #         B = 0.5 - theta_square / 24
        #     else:
        #         A = math.sin(theta) / theta
        #         B = (1 - math.cos(theta)) / theta_square
        #         C = (1 - A) / theta_square
        #     w0 = ln_r1 * rt2 - ln_r2 * rt1
        #     w1 = ln_r2 * rt0 - ln_r0 * rt2
        #     w2 = ln_r0 * rt1 - ln_r1 * rt0
        #     t0 = ln_t0 + B * rt0 + C * w0
        #     t1 = ln_t1 + B * rt1 + C * w1
        #     t2 = ln_t2 + B * rt2 + C * w2
        A = torch.where(
            theta_square < 1e-8,
            torch.ones_like(theta_square) - theta_square / 6,
            torch.where(
                theta_square < 1e-6,
                torch.ones_like(theta_square) - theta_square * (1 - theta_square / 20) / 6,
                torch.sin(theta) / theta
            )
        )
        B = torch.where(
            theta_square < 1e-8,
            torch.ones_like(theta_square) / 2,
            torch.where(
                theta_square < 1e-6,
                0.5 - theta_square / 24,
                (1 - torch.cos(theta)) / theta_square
            )
        )
        C = torch.where(
            theta_square < 1e-8,
            torch.zeros_like(theta_square),
            torch.where(
                theta_square < 1e-6,
                (1 - theta_square / 20) / 6,
                (1 - A) / theta_square
            )
        )
        
        omega_hat = hat(omega)
        omega_hat_square = omega_hat @ omega_hat
        with nvtx_range('R'):
            R = (
                torch.eye(3, dtype=x.dtype, device=x.device)[None]
                + A * omega_hat
                + B * omega_hat_square
            )
        with nvtx_range('V'):
            V = (
                torch.eye(3, dtype=x.dtype, device=x.device)[None]
                + B * omega_hat
                + C * omega_hat_square
            )
        with nvtx_range('T'):
            T = torch.cat([R, V @ x[:, :3]], dim=2) # (N, 3, 4)
            T = torch.cat([T, torch.zeros_like(T[:, :1])], dim=1) # (N, 4, 4)
            T[..., 3, 3] = 1

        return T

def se3_exp_map_numba(x):
    with nvtx_range('se3_exp_map_numba'):
        # x: (N, 6, 1)
        # T: (N, 4, 4)
        T = torch.zeros((x.shape[0], 4, 4), dtype=x.dtype, device=x.device)
        _se3_exp_map_numba[1, x.shape[0]](x, T)
        return T

@cuda.jit(inline=True)
def _se3_exp_map_numba(x, T):
    # x: (N, 6, 1)
    # T: (N, 4, 4)
    pos = cuda.grid(1)
    if pos < x.shape[0]:
        # float3 cr = cross(rot, trans);
        # if (theta_sq < 1e-8)
        # {
        #     A = 1.0f - ONE_SIXTH * theta_sq;
        #     B = 0.5f;
        #     translation = trans + 0.5f * cr;
        # }
        # else
        # {
        #     float C;
        #     if (theta_sq < 1e-6) {
        #         C = ONE_SIXTH*(1.0f - ONE_TWENTIETH * theta_sq);
        #         A = 1.0f - theta_sq * C;
        #         B = 0.5f - 0.25f * ONE_SIXTH * theta_sq;
        #     }
        #     else {
        #         const float inv_theta = 1.0f / theta;
        #         A = sin(theta) * inv_theta;
        #         B = (1 - cos(theta)) * (inv_theta * inv_theta);
        #         C = (1 - A) * (inv_theta * inv_theta);
        #     }
        #     float3 w_cross = cross(rot, cr);
        #     translation = trans + B * cr + C * w_cross;
        # }
        ln_t0 = x[pos, 0, 0]
        ln_t1 = x[pos, 1, 0]
        ln_t2 = x[pos, 2, 0]
        ln_r0 = x[pos, 3, 0]
        ln_r1 = x[pos, 4, 0]
        ln_r2 = x[pos, 5, 0]
        theta_square = ln_r0 * ln_r0 + ln_r1 * ln_r1 + ln_r2 * ln_r2
        theta = math.sqrt(theta_square)
        rt0 = ln_r1 * ln_t2 - ln_r2 * ln_t1
        rt1 = ln_r2 * ln_t0 - ln_r0 * ln_t2
        rt2 = ln_r0 * ln_t1 - ln_r1 * ln_t0

        if theta_square < 1e-8:
            A = 1.0 - theta_square / 6
            B = 0.5
            t0 = ln_t0 + rt0 / 2
            t1 = ln_t1 + rt1 / 2
            t2 = ln_t2 + rt2 / 2
        else:
            if theta_square < 1e-6:
                C = (1 - theta_square / 20) / 6
                A = 1 - theta_square * C
                B = 0.5 - theta_square / 24
            else:
                A = math.sin(theta) / theta
                B = (1 - math.cos(theta)) / theta_square
                C = (1 - A) / theta_square
            w0 = ln_r1 * rt2 - ln_r2 * rt1
            w1 = ln_r2 * rt0 - ln_r0 * rt2
            w2 = ln_r0 * rt1 - ln_r1 * rt0
            t0 = ln_t0 + B * rt0 + C * w0
            t1 = ln_t1 + B * rt1 + C * w1
            t2 = ln_t2 + B * rt2 + C * w2
        
        # {
        #     const float wx2 = w.x * w.x;
        #     const float wy2 = w.y * w.y;
        #     const float wz2 = w.z * w.z;
        #     R(0, 0) = 1.0f - B*(wy2 + wz2);
        #     R(1, 1) = 1.0f - B*(wx2 + wz2);
        #     R(2, 2) = 1.0f - B*(wx2 + wy2);
        # }
        # {
        #     const float a = A*w.z;
        #     const float b = B*(w.x * w.y);
        #     R(0, 1) = b - a;
        #     R(1, 0) = b + a;
        # }
        # {
        #     const float a = A*w.y;
        #     const float b = B*(w.x * w.z);
        #     R(0, 2) = b + a;
        #     R(2, 0) = b - a;
        # }
        # {
        #     const float a = A*w.x;
        #     const float b = B*(w.y * w.z);
        #     R(1, 2) = b - a;
        #     R(2, 1) = b + a;
        # }
        T[pos, 0, 0] = 1 - B * (ln_r1 * ln_r1 + ln_r2 * ln_r2)
        T[pos, 1, 1] = 1 - B * (ln_r0 * ln_r0 + ln_r2 * ln_r2)
        T[pos, 2, 2] = 1 - B * (ln_r0 * ln_r0 + ln_r1 * ln_r1)
        T[pos, 0, 1] = B * (ln_r0 * ln_r1) - A * ln_r2
        T[pos, 1, 0] = B * (ln_r0 * ln_r1) + A * ln_r2
        T[pos, 0, 2] = B * (ln_r0 * ln_r2) + A * ln_r1
        T[pos, 2, 0] = B * (ln_r0 * ln_r2) - A * ln_r1
        T[pos, 1, 2] = B * (ln_r1 * ln_r2) - A * ln_r0
        T[pos, 2, 1] = B * (ln_r1 * ln_r2) + A * ln_r0
        T[pos, 0, 3] = t0
        T[pos, 1, 3] = t1
        T[pos, 2, 3] = t2
        T[pos, 3, 0] = 0
        T[pos, 3, 1] = 0
        T[pos, 3, 2] = 0
        T[pos, 3, 3] = 1



