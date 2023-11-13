import random
from matplotlib import pyplot as plt
import torch
from torch.nn.functional import grid_sample
from torchvision.transforms.functional import rgb_to_grayscale, InterpolationMode
from loftr.loftr import LoFTR, default_cfg
from bundle_fetch.utils import nvtx_range


# https://github.com/chengzegang/TorchSIFT/blob/main/src/torchsift/ransac/ransac.py
def ransac(corr):
    r"""RANSAC algorithm to find the best model.

    Args:
        x (Tensor): The first set of features with shape :math:`(B, N, D)`.
        y (Tensor): The second set of features with shape :math:`(B, M, D)`.
        mask (Tensor): The mask with shape :math:`(B, N, M)`.
        solver (Callable): The solver function to find the model.
        evaluator (Callable): The evaluator function to evaluate the model.
        ransac_ratio (float, optional): The ratio of inliers to consider the model as the best. Defaults to 0.6.
        ransac_it (int, optional): The number of iterations. Defaults to 16.
        ransac_thr (float, optional): The threshold to consider a point as an inlier. Defaults to 0.75.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: The matching mask with shape :math:`(B, N, M)`.
    """

    x = corr['x']
    y = corr['y']
    m = corr['conf'] > 0
    ransac_it = 16
    B, N, _ = x.shape
    B, N, _ = y.shape

    sample_indices = torch.randint(0, N, (B * 3 * ransac_it), device=x.device)
    s_a = x[sample_indices].reshape(B, ransac_it, 3, 3)
    s_b = y[sample_indices]
    s_m = m[sample_indices]

    models = solver(s_a, s_b, s_m)  # (B * 3 * ransac_it, D, D)
    x = x.unsqueeze(1).repeat(1, ransac_it, 1, 1).flatten(0, 1)
    y = y.unsqueeze(1).repeat(1, ransac_it, 1, 1).flatten(0, 1)
    mask = mask.unsqueeze(1).repeat(1, ransac_it, 1, 1).flatten(0, 1)
    errors = evaluator(models, x, y, mask)  # (B * ransac_it, N, M)
    errors = errors.view(B, ransac_it, N, M)
    models = models.view(B, ransac_it, models.shape[-2], models.shape[-1])
    avg_errors = torch.nanmean(errors, dim=(-1, -2))
    best_model_idx = torch.argmin(avg_errors, dim=-1)

    best_model = torch.gather(
        models,
        dim=1,
        index=best_model_idx.view(-1, 1, 1, 1).repeat(
            1, 1, models.shape[-2], models.shape[-1]
        ),
    ).squeeze(1)

    best_errors = torch.gather(
        errors,
        dim=1,
        index=best_model_idx.view(-1, 1, 1, 1).repeat(
            1, 1, errors.shape[-2], errors.shape[-1]
        ),
    ).squeeze(1)
    thrs = torch.nanquantile(
        best_errors.flatten(-2), ransac_thr, dim=-1, keepdim=True
    ).unsqueeze(-1)
    inliers = best_errors <= thrs
    best_errors[~inliers] = torch.nan
    return best_model, inliers, best_errors


def get_corr_model():
    def corr_model(frame, prev_frame, keyframes):
        data = {}
        data['image0'] = rgb_to_grayscale(frame['rgb'])[None]
        corr_model.loftr.forward_backbone(data)
        frame['feat_c0'] = data['feat_c0']
        frame['feat_f0'] = data['feat_f0']

        if prev_frame is None:
            return None
        elif len(keyframes):
            i_keyframes = random.choices(range(len(keyframes)), k=min(5, len(keyframes)))
            frames = [prev_frame, *[keyframes[i_frame] for i_frame in i_keyframes]]
        else:
            i_keyframes = []
            frames = [prev_frame]

        N = len(frames)
        data.update({
            'bs': N,
            'feat_c0': data['feat_c0'].expand(N, -1, -1, -1),
            'feat_f0': data['feat_f0'].expand(N, -1, -1, -1),
            'hw1_i': data['hw0_i'], 
            'hw1_c': data['hw0_c'],
            'hw1_f': data['hw0_f'],
            'feat_c1': torch.cat([f['feat_c0'] for f in frames]),
            'feat_f1': torch.cat([f['feat_f0'] for f in frames]),
        })
        corr_model.loftr.forward_matching(data)
            
        conf = torch.reshape(data['mconf'], (N, -1))[:, :100] # (N, M)
        uv_a = torch.reshape(data['mkpts0_f'], (N, -1, 2))[:, :100] # (N, M, 2)
        uv_b = torch.reshape(data['mkpts1_f'], (N, -1, 2))[:, :100] # (N, M, 2)

        grid_a = uv_a / frame['wh'] # (N, M, 2)
        grid_a = grid_a * 2 - 1 # (N, M, 2)
        grid_a = grid_a[:, :, None] # (N, M, 1, 2)
        grid_b = uv_b / frame['wh'] # (N, M, 2)
        grid_b = grid_b * 2 - 1 # (N, M, 2)
        grid_b = grid_b[:, :, None] # (N, M, 1, 2)
        masked_ayz_a = frame['xyz'] * frame['mask'][None] # (3, 480, 640)
        masked_ayz_a = masked_ayz_a.expand(N, -1, -1, -1) # (N, 3, 480, 640)
        masked_ayz_b = torch.stack([f['xyz'] for f in frames]) * torch.stack([f['mask'][None] for f in frames]) # (N, 3, 480, 640)
        
        xyz1_a = grid_sample(masked_ayz_a, grid_a, align_corners=True) # (N, 3, M, 1)
        xyz1_a = xyz1_a.permute(0, 2, 1, 3) # (N, M, 3, 1)
        xyz1_a = torch.cat([xyz1_a, torch.ones_like(xyz1_a[:, :, :1])], dim=2) # (N, M, 4, 1)
        
        xyz1_b = grid_sample(masked_ayz_b, grid_b, align_corners=True) # (N, 3, M, 1)
        xyz1_b = xyz1_b.permute(0, 2, 1, 3) # (N, M, 3, 1)
        xyz1_b = torch.cat([xyz1_b, torch.ones_like(xyz1_b[:, :, :1])], dim=2) # (N, M, 4, 1)

        corr = {
            'o_T_c_a': prev_frame['o_T_c'].expand(N, -1, -1), # (N, 4, 4)
            'o_T_c_b': torch.stack([f['o_T_c'] for f in frames]), # (N, 4, 4)
            'xyz1_a': xyz1_a, # (N, M, 4, 1)
            'xyz1_b': xyz1_b, # (N, M, 4, 1)
            'conf': conf, # (N, M)
            'i_keyframes': i_keyframes, # (N)
        }

        # corr = ransac(corr)
        return corr
    corr_model.loftr = LoFTR(config=default_cfg)
    corr_model.loftr.load_state_dict(torch.load(f'checkpoints/outdoor_ds.ckpt')['state_dict'])
    corr_model.loftr = corr_model.loftr.eval().cuda()
    return corr_model

