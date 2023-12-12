
from omegaconf import open_dict, OmegaConf
from XMem.inference.data.mask_mapper import MaskMapper
from XMem.model.network import XMem
from XMem.inference.inference_core import InferenceCore
import torch


def get_mask_model():
    def mask_model(frame):
        rgb = frame['rgb']
        mask = frame.get('mask')
        if mask is None:
            labels = None
        else:
            labels = torch.unique(mask)
            labels = labels[labels!=0]
        with torch.inference_mode():
            prob = processor.step(rgb, mask, labels)
        out_mask = torch.max(prob, dim=0).indices
        return out_mask
    
    # load XMem
    cfg = OmegaConf.load('conf/xmem_segment_all.yaml')
    with open_dict(cfg):
        network = XMem(cfg.xmem, cfg.xmem.checkpoint).cuda().eval()
    model_weights = torch.load(cfg.xmem.checkpoint)
    network.load_weights(model_weights, init_as_zero_if_needed=True)
    processor = InferenceCore(network, config=cfg.xmem)
    processor.set_all_labels(set([1]))
    mask_model.xmem = processor
    return mask_model
