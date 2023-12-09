import torch
from bundle_fetch.track2.utils import filter_edges, get_features, get_mask
from loftr.loftr import LoFTR, default_cfg
from omegaconf import open_dict, OmegaConf
from XMem.inference.data.mask_mapper import MaskMapper
from XMem.model.network import XMem
from XMem.inference.inference_core import InferenceCore
import torch
from torchvision.transforms.functional import rgb_to_grayscale, InterpolationMode, resize


class Track:
    def __init__(self, track_stop, track_queue) -> None:
        self.track_stop = track_stop
        self.track_queue = track_queue

        self.new_vertex = None
        self.vertices = {}
        self.edges = {}

        self.loftr = load_loftr()
        self.xmem = load_xmem()

    def __call__(self):
        """
        Track objects
        """
        while not self.track_stop.is_set():
            frame, event = self.track_queue.get()
            event.synchronize()

            self.new_vertex = make_vertex(frame, self.new_vertex, len(self.vertices))
            subvertices = get_subvertices(self.new_vertex, self.vertices)
            subedges = get_subedges(subvertices, self.edges)
            optimize_graph(subvertices, subedges)
            
            if check_add_vertex(self.new_vertex, self.vertices):
                self.vertices[len(self.vertices)] = self.new_vertex
                for k_subedge, v_subedge in subedges.items():
                    if k_subedge not in self.edges:
                        self.edges[k_subedge] = v_subedge


def load_loftr():
    """
    Load LoFTR
    """
    loftr = LoFTR(config=default_cfg)
    loftr.load_state_dict(torch.load(f'checkpoints/outdoor_ds.ckpt')['state_dict'])
    loftr = loftr.eval().cuda()
    return loftr


def load_xmem():
    """
    Load XMem
    """
    cfg = OmegaConf.load('conf/xmem_segment_all.yaml')
    xmem = XMem(cfg.xmem, cfg.xmem.checkpoint).cuda().eval()
    xmem_weights = torch.load(cfg.xmem.checkpoint)
    xmem.load_weights(xmem_weights, init_as_zero_if_needed=True)
    xmem = InferenceCore(xmem, config=cfg.xmem)
    xmem.set_all_labels(set([1]))
    return xmem


def make_vertex(frame, prev_vertex, id, xmem, loftr):
    """
    Make vertex from frame
    """
    prev_vertex['id'] = id + 1

    mask = get_mask(frame, xmem)
    feat_c, feat_f = get_features(frame, loftr)
    vertex = {
        'id': id,
        'frame': frame,
        'mask': mask,
        'feat_c': feat_c,
        'feat_f': feat_f,
        'o_T_c': None,
    }
    subvertices = [vertex, prev_vertex]
    subedges = get_subedges(subvertices, {})
    filter_edges(subvertices, subedges)


def get_subvertices(vertex, vertices, n_subvertices):
    """
    Get subvertices
    """
    subvertices = []
    return subvertices


def get_subedges(vertices, edges):
    """
    Get edges between vertices
    """
    subedges = {}

    edges_to_add = []
    for i in range(len(vertices) - 1):
        for j in range(i + 1, len(vertices)):
            if (i, j) not in edges:
                edges_to_add.append((i, j))
            else:
                subedges[(i, j)] = edges[(i, j)]
    
    for i, j in edges_to_add:
        subedges[(i, j)] = {
            'i': i,
            'j': j,
            'i_si': vertices[i]['xyz'][vertices[i]['mask']],
            'j_sj': vertices[j]['xyz'][vertices[j]['mask']],
        }

    return edges


def optimize_graph(vertices, edges):
    """
    Bundle adjustment
    """
    # o_T_v, Kv, ii, jj, i_si, j_sj, v_dv, v_nv, v_mv
    o_T_v = torch.stack([v['o_T_c'] for v in vertices]) # (V, 4, 4)
    Kv = torch.stack([v['frame']['K'] for v in vertices]) # (V, 3, 3)
    ii = torch.stack([e['i'] for e in edges]) # (E)
    jj = torch.stack([e['j'] for e in edges]) # (E)
    i_si = torch.stack([e['i_si'] for e in edges]) # (E, S, 3)
    j_sj = torch.stack([e['j_sj'] for e in edges]) # (E, S, 3)
    v_dv = torch.stack([v['frame']['point'] for v in vertices]) # (V, H, W, 3)
    v_nv = torch.stack([v['frame']['normal'] for v in vertices]) # (V, H, W, 3)
    v_mv = torch.stack([v['frame']['mask'] for v in vertices]) # (V, H, W, 3)

    for i, v in enumerate(vertices):
        v['o_T_c'] = o_T_v[i]


def check_add_vertex(vertex, vertices):
    """
    Check if vertex should be added to vertices
    """
    return True