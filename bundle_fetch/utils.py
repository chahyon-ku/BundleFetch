
import numpy as np
import torch
import contextlib


@contextlib.contextmanager
def nvtx_range(msg):
    depth = torch.cuda.nvtx.range_push(msg)
    try:
        yield depth
    finally:
        torch.cuda.nvtx.range_pop()


def inv_transform(T):
    R = T[:3, :3]
    t = T[:3, 3]
    T = np.identity(4)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ t
    return T