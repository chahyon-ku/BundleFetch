


import numpy as np


def get_mask_vis(rgb, mask):
    vis_mask = rgb.copy()
    vis_mask[mask == 1] = np.clip(vis_mask[mask == 1].astype(int) - 64, 0, 255)
    vis_mask[mask == 1, 2] = 255
    return vis_mask