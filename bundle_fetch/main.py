

import torch.multiprocessing as mp
from bundle_fetch.track.track import Track
from bundle_fetch.frame.frame import Frame
from bundle_fetch.recon.recon import recon_thread_target
from bundle_fetch.gui import gui_thread_target

def main():
    mp.set_start_method('spawn')
    frame_stop = mp.Event()
    track_stop = mp.Event()
    recon_stop = mp.Event()
    gui_stop = mp.Event()
    track_queue = mp.Queue(1)
    recon_queue = mp.Queue()
    gui_queue = mp.Queue(1)

    frame = Frame(frame_stop, track_queue)
    track = Track(track_stop, track_queue, gui_queue)

    frame_thread = mp.Process(target=frame)
    # track_thread = mp.Process(target=track)
    recon_thread = mp.Process(target=recon_thread_target, args=(recon_stop, recon_queue))
    gui_thread = mp.Process(target=gui_thread_target, args=(gui_stop, gui_queue,))

    frame_thread.start()
    # track_thread.start()
    recon_thread.start()
    gui_thread.start()

    # gui_thread_target(gui_queue)
    # frame_thread_target(frame_stop, track_queue)
    track()
    # mp.spawn(frame, nprocs=1, join=False)
    # mp.spawn(recon_thread_target, args=(recon_stop, recon_queue), nprocs=1, join=False)
    # mp.spawn(gui_thread_target, (gui_stop, gui_queue), nprocs=1, join=False)

    while True:
        if not frame_thread.is_alive() or not gui_thread.is_alive():
            frame_stop.set()
            track_stop.set()
            recon_stop.set()
            gui_stop.set()
            break

    # frame_thread.join()
    # track_thread.join()
    # recon_thread.join()