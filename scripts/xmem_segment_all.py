import os
import hydra
import torch
from torch.utils.data.dataloader import DataLoader
from XMem.inference.data.mask_mapper import MaskMapper
from XMem.model.network import XMem
from XMem.inference.inference_core import InferenceCore
from omegaconf import open_dict
import matplotlib.pyplot as plt
import numpy as np
from BundleFetch.bf_dataset import BfDataset
from BundleFetch.image_utils import get_mask_vis
from PIL import Image


@hydra.main(config_path='../conf', config_name='xmem_segment_all')
def main(cfg):
    # enable cfg modification
    with open_dict(cfg):
        network = XMem(cfg.xmem, cfg.xmem.checkpoint).cuda().eval()
    model_weights = torch.load(cfg.xmem.checkpoint)
    network.load_weights(model_weights, init_as_zero_if_needed=True)
    mapper = MaskMapper()
    processor = InferenceCore(network, config=cfg.xmem)
    processor.set_all_labels(set([1]))

    dataset = BfDataset(cfg.data_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            for ti, data in enumerate(loader):
                rgb = data['rgb'].cuda()[0]
                mask = data.get('masks')
                if mask is None:
                    labels = None
                else:
                    labels = np.unique(mask)
                    labels = labels[labels!=0]
                    mask = mask.cuda()[0]
                prob = processor.step(rgb, mask, labels)
                out_mask = torch.max(prob, dim=0).indices
                out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)
                
                if mask is None:
                    rgb_path = data['rgb_path'][0]
                    mask_path = rgb_path.replace('rgb', 'masks')
                    mask_vis_path = rgb_path.replace('rgb', 'masks_vis')
                    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
                    os.makedirs(os.path.dirname(mask_vis_path), exist_ok=True)
                    os.remove(mask_path) if os.path.exists(mask_path) else None
                    os.remove(mask_vis_path) if os.path.exists(mask_vis_path) else None
                    rgb = Image.open(rgb_path).convert('RGB')
                    rgb = np.array(rgb)
                    mask_vis = get_mask_vis(rgb, out_mask)
                    Image.fromarray(out_mask * 255, mode='L').convert('1').save(mask_path)
                    Image.fromarray(mask_vis, mode='RGB').save(mask_vis_path)

if __name__ == '__main__':
    main()