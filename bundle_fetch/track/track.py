from matplotlib import pyplot as plt
from bundle_fetch.track.utils import filter_edges, get_features, get_mask, sparse, dense, get_covisibility
from loftr.loftr import LoFTR, default_cfg
from omegaconf import open_dict, OmegaConf
from xmem.inference.data.mask_mapper import MaskMapper
from xmem.model.network import XMem
from xmem.inference.inference_core import InferenceCore
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale, InterpolationMode, resize
from torch.nn.functional import grid_sample
import lietorch
from torch.profiler import profile, ProfilerActivity
import open3d as o3d
from bundle_fetch.utils import nvtx_range


class Track(object):
    def __init__(self, track_stop, track_queue, gui_queue) -> None:
        self.track_stop = track_stop
        self.track_queue = track_queue
        self.gui_queue = gui_queue

        self.new_vertex = None
        self.vertices = {}
        self.edges = {}
        self.n_subvertices = 10

        self.loftr = load_loftr()
        self.xmem = load_xmem()

    def __call__(self):
        """
        Track objects
        """
        cuda_stream = torch.cuda.Stream()
        with torch.cuda.stream(cuda_stream):
            with torch.inference_mode():
                while not self.track_stop.is_set():
                    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    frame, event = self.track_queue.get()
                    event.synchronize()

                    self.new_vertex = make_vertex(frame, self.new_vertex, len(self.vertices), self.xmem, self.loftr)
                    if len(self.vertices) == 0:
                        self.vertices[0] = self.new_vertex
                    else:
                        subvertices = get_subvertices(self.new_vertex, self.vertices, self.n_subvertices)
                        subedges = get_subedges(subvertices, self.edges, self.loftr)
                        # optimize_graph(subvertices, subedges)
                        
                        if check_add_vertex(self.new_vertex, self.vertices):
                            self.vertices[self.new_vertex['id']] = self.new_vertex
                            for k_subedge, v_subedge in subedges.items():
                                if k_subedge not in self.edges:
                                    self.edges[k_subedge] = v_subedge
                        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

                    self.gui_queue.put(self.new_vertex)
                    print(self.new_vertex['o_T_c'].matrix())
                    input()


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
    with nvtx_range('make_vertex'):
        mask = get_mask(frame, xmem)
        # feat_c, feat_f, hw_i, hw_c, hw_f = get_features(frame, loftr)
        vertex = {
            'id': id,
            'frame': frame,
            'mask': mask,
            # 'feat_c': feat_c,
            # 'feat_f': feat_f,
            # 'hw_i': hw_i,
            # 'hw_c': hw_c,
            # 'hw_f': hw_f,
        }
        if id == 0:
            max_masked_xyz = (frame['xyz'].cuda() * mask.cuda())
            o_T_c = -torch.sum(max_masked_xyz, dim=(1, 2)) / torch.sum(mask.cuda()) # (3)
            o_T_c = torch.concat([o_T_c, torch.zeros_like(o_T_c)], dim=-1)
            vertex['o_T_c'] = lietorch.SE3.exp(o_T_c) # (6)
            return vertex

        prev_vertex['id'] = -1
        subvertices = {prev_vertex['id']: prev_vertex, vertex['id']: vertex}
        subedges = get_subedges(subvertices, {}, loftr)
        filter_edges(subvertices, subedges)

        return vertex


def get_subvertices(vertex, vertices, n_subvertices):
    """
    Get subvertices
    """
    with nvtx_range('get_subvertices'):
        if len(vertices) < n_subvertices:
            subvertices = {k: v for k, v in vertices.items()}
        else:
            covisibility = [get_covisibility(vertex, v) for k, v in vertices.items()]
            sorted_vertices = sorted(zip(covisibility, vertices.items()), key=lambda x: x[0], reverse=True)
            subvertices = {k: v for _, (k, v) in sorted_vertices[:n_subvertices - 1]}
        subvertices[vertex['id']] = vertex
        return subvertices


def get_subedges(vertices, edges, loftr):
    """
    Get edges between vertices
    """
    with nvtx_range('get_subedges'):
        N = len(vertices)
        subedges = {}
        if N < 2:
            return subedges

        edges_to_add = []
        keys = list(vertices.keys())
        keys_index = {k: i for i, k in enumerate(keys)}
        for i in range(N):
            for j in range(i + 1, N):
                if (keys[i], keys[j]):
                    edges_to_add.append((keys[i], keys[j]))

        v = list(vertices.values())[0]
        E = len(edges_to_add)
        # data = {
        #     'bs': len(edges_to_add),
        #     'feat_c0': torch.cat([vertices[i]['feat_c'] for i, j in edges_to_add]), # (E, C, H, W)
        #     'feat_f0': torch.cat([vertices[i]['feat_f'] for i, j in edges_to_add]), # (E, C, H, W)
        #     'mask0': torch.cat([
        #         F.max_pool2d(vertices[i]['mask'][None].float(), 8) > 0
        #         for i, _ in edges_to_add
        #     ]),
        #     'hw0_i': v['hw_i'],
        #     'hw0_c': v['hw_c'],
        #     'hw0_f': v['hw_f'],
        #     'hw1_i': v['hw_i'],
        #     'hw1_c': v['hw_c'],
        #     'hw1_f': v['hw_f'],
        #     'feat_c1': torch.cat([vertices[j]['feat_c'] for i, j in edges_to_add]), # (E, C, H, W)
        #     'feat_f1': torch.cat([vertices[j]['feat_f'] for i, j in edges_to_add]), # (E, C, H, W)
        #     'mask1': torch.cat([
        #         F.max_pool2d(vertices[j]['mask'][None].float(), 8) > 0
        #         for _, j in edges_to_add
        #     ]),
        # }
        # loftr.forward_matching(data)
        data = {
            'bs': len(edges_to_add),
            'image0': rgb_to_grayscale(torch.stack([vertices[i]['frame']['rgb'].cuda() for i, j in edges_to_add])), # (E, C, H, W)
            'image1': rgb_to_grayscale(torch.stack([vertices[j]['frame']['rgb'].cuda() for i, j in edges_to_add])), # (E, C, H, W)
            'mask0': torch.cat([
                F.max_pool2d(vertices[i]['mask'][None].cuda().float(), 8) > 0
                for i, _ in edges_to_add
            ]),
            'mask1': torch.cat([
                F.max_pool2d(vertices[j]['mask'][None].cuda().float(), 8) > 0
                for _, j in edges_to_add
            ]),
        }
        loftr(data)
        conf = torch.reshape(data['mconf'], (E, -1)) # (E, M)
        uv_a = torch.reshape(data['mkpts0_f'], (E, -1, 2)) # (E, M, 2)
        uv_b = torch.reshape(data['mkpts1_f'], (E, -1, 2)) # (E, M, 2)
        grid_a = uv_a / v['frame']['wh'].cuda() * 2 - 1 # (E, M, 2)
        grid_b = uv_b / v['frame']['wh'].cuda() * 2 - 1 # (E, M, 2)
        M = conf.shape[1]

        for i_edge, (i, j) in enumerate(edges_to_add):
            i_si = grid_sample(vertices[i]['frame']['xyz'][None].cuda(), grid_a[None, None, i_edge])[0, :, 0].permute(1, 0) # (S, 3)
            j_sj = grid_sample(vertices[j]['frame']['xyz'][None].cuda(), grid_b[None, None, i_edge])[0, :, 0].permute(1, 0)

            # visualize i_si with open3d
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(i_si[conf[i_edge] > 0].cpu().numpy())
            # o3d.visualization.draw_geometries([pcd])

            # visualize j_sj with open3d
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(j_sj[conf[i_edge] > 0].cpu().numpy())
            # o3d.visualization.draw_geometries([pcd])

            # visualize correspondences
            # fig, ax = plt.subplots()
            # ax.imshow(torch.cat([vertices[i]['frame']['rgb'], vertices[j]['frame']['rgb']], dim=2).permute(1, 2, 0).cpu().numpy())
            # xs = []
            # ys = []
            # for k in range(M):
            #     if conf[i_edge, k] == 0:
            #         continue
            #     xs.append(uv_a[i_edge, k, 0].item())
            #     ys.append(uv_a[i_edge, k, 1].item())
            #     xs.append(uv_b[i_edge, k, 0].item() + vertices[i]['frame']['wh'][0].item())
            #     ys.append(uv_b[i_edge, k, 1].item())
            # line, = ax.plot(xs, ys, 'b')
            # ax.set_title(f'edge {i} {j} {(conf > 0).sum()} matches')
            # plt.show()


            subedges[(i, j)] = {
                'i': i,
                'j': j,
                'i_si': i_si, # (S, 3)
                'j_sj': j_sj, # (S, 3)
                'conf': conf[i_edge], # (M)
            }

        return subedges


def optimize_graph(vertices, edges):
    """
    Bundle adjustment
    """
    with nvtx_range('optimize_graph'):
        o_T_v = lietorch.SE3.exp(torch.stack([v['o_T_c'].log() for v in vertices.values()])) # (V, 6)
        Kv = torch.stack([v['frame']['cam_K'] for v in vertices.values()]).cuda() # (V, 3, 3)
        vertex_keys_indices = {k: torch.tensor(i, dtype=torch.int, device=Kv.device) for i, k in enumerate(vertices.keys())}
        ii = torch.stack([vertex_keys_indices[e['i']] for e in edges.values()]).long() # (E)
        jj = torch.stack([vertex_keys_indices[e['j']] for e in edges.values()]).long() # (E)
        i_si = torch.stack([e['i_si'] for e in edges.values()]) # (E, S, 3)
        j_sj = torch.stack([e['j_sj'] for e in edges.values()]) # (E, S, 3)
        v_dv = torch.stack([v['frame']['xyz'].cuda() for v in vertices.values()]) # (V, H, W, 3)
        v_nv = torch.stack([v['frame']['nxyz'].cuda() for v in vertices.values()]) # (V, H, W, 3)
        v_mv = torch.stack([v['mask'].cuda() for v in vertices.values()]) # (V, H, W, 3)
        V = o_T_v.shape[0]
        E, S, _ = i_si.shape

        s, Ji, Jj = sparse(o_T_v, Kv, ii, jj, i_si, j_sj, v_dv, v_nv, v_mv)

        Hii = torch.einsum('esij, esik -> ejk', Ji, Ji) # (E, 6, 6)
        Hij = torch.einsum('esij, esik -> ejk', Ji, Jj) # (E, 6, 6)
        Hjj = torch.einsum('esij, esik -> ejk', Jj, Jj) # (E, 6, 6)
        Hji = torch.einsum('esij, esik -> ejk', Jj, Ji) # (E, 6, 6)
        gi = torch.einsum('esij, esi -> ej', Ji, s) # (E, 6)
        gj = torch.einsum('esij, esi -> ej', Jj, s) # (E, 6)
        H = torch.zeros((V * V, 6, 6), dtype=Hii.dtype, device=Hii.device) # (V * V, 6, 6)
        H.scatter_add_(0, (ii * V + ii)[:, None, None].expand(-1, 6, 6), Hii)
        H.scatter_add_(0, (jj * V + jj)[:, None, None].expand(-1, 6, 6), Hjj)
        H.scatter_add_(0, (ii * V + jj)[:, None, None].expand(-1, 6, 6), Hij)
        H.scatter_add_(0, (jj * V + ii)[:, None, None].expand(-1, 6, 6), Hji)
        g = torch.zeros((V, 6), dtype=gi.dtype, device=gi.device) # (V, 6)
        g.scatter_add_(0, ii[:, None].expand(-1, 6), gi)
        g.scatter_add_(0, jj[:, None].expand(-1, 6), gj)

        H = H.reshape(V, V, 6, 6).permute(0, 2, 1, 3).reshape(V * 6, V * 6) # (V * 6, V * 6)
        g = g.reshape(V * 6, 1) # (V * 6, 1)
        
        # D = dense(o_T_v, Kv, ii, jj, i_si, j_sj, v_dv, v_nv, v_mv)

        # H = get_hessian(sparse, o_T_v.data, S.shape, S.dtype, **{
        #     'Kv': Kv,
        #     'ii': ii,
        #     'jj': jj,
        #     'i_si'g: i_si,
        #     'j_sj': j_sj,
        #     'v_dv': v_dv,
        #     'v_nv': v_nv,
        #     'v_mv': v_mv,
        # })
        # g = get_gradient(sparse, o_T_v.data, S, **{
        #     'Kv': Kv,
        #     'ii': ii,
        #     'jj': jj,
        #     'i_si': i_si,
        #     'j_sj': j_sj,
        #     'v_dv': v_dv,
        #     'v_nv': v_nv,
        #     'v_mv': v_mv,
        # })
        
        L = torch.linalg.cholesky(H)
        dx = torch.cholesky_solve(-g, L) # (V * 6, 1)
        dx = dx.reshape(V, 6) # (V, 6)
        o_T_v = o_T_v.retr(dx)

        for i_v, v in enumerate(vertices.values()):
            v['o_T_c'] = o_T_v[i_v]


def check_add_vertex(vertex, vertices):
    """
    Check if vertex should be added to vertices
    """
    return True