import torch


def recon_thread_target(recon_stop, recon_queue):
    cuda_stream = torch.cuda.Stream()

    while not recon_stop.is_set():
        pass