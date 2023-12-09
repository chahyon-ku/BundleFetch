from torchvision.transforms.functional import rgb_to_grayscale


def filter_edges(vertices, edges):
    """
    Filter edges
    """
    for i, j in edges.keys():
        if i not in vertices or j not in vertices:
            del edges[(i, j)]


def get_mask(frame, xmem):
    """
    Get mask from vertex
    """
    rgb = frame['rgb']
    mask = frame.get('mask')
    if mask is None:
        labels = None
    else:
        labels = torch.unique(mask)
        labels = labels[labels!=0]
    prob = xmem.step(rgb, mask, labels)
    out_mask = torch.max(prob, dim=0).indices
    return out_mask


def get_features(frame, loftr):
    """
    Get features from frame
    """
    data = {}
    data['image0'] = rgb_to_grayscale(frame['rgb'])[None]
    loftr.forward_backbone(data)
    feat_c = data['feat_c0']
    feat_f = data['feat_f0']
    return feat_c, feat_f