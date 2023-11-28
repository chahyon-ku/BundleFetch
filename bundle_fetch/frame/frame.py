import torch
from bundle_fetch.frame.util import dataset_get_frame, get_process_frame


def frame_thread_target(frame_stop, track_queue):
    print('frame_thread_target')
    cuda_stream = torch.cuda.Stream()
    uv1 = torch.stack(torch.meshgrid(torch.arange(0, 640), torch.arange(0, 480))).float()
    uv1 = torch.cat((uv1, torch.ones_like(uv1[:1])), dim=0)
    get_frame = dataset_get_frame()
    process_frame = get_process_frame()

    while not frame_stop.is_set():
        frame = get_frame()
        if frame is None:
            break
        frame = process_frame(frame)

        with torch.cuda.stream(cuda_stream):
            frame = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in frame.items()}
            event = torch.cuda.Event()
            track_queue.put((frame, event))