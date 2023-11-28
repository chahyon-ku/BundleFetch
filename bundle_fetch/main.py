

import threading
import queue
from bundle_fetch.track.corr import get_corr_model
from bundle_fetch.track.mask import get_mask_model
from bundle_fetch.track.pose import get_pose_model
from bundle_fetch.frame.frame import frame_thread_target
from bundle_fetch.track.track import track_thread_target
from bundle_fetch.recon.recon import recon_thread_target
from bundle_fetch.gui import gui_thread_target

def main():
    frame_stop = threading.Event()
    track_stop = threading.Event()
    recon_stop = threading.Event()
    gui_stop = threading.Event()
    track_queue = queue.Queue()
    recon_queue = queue.Queue()
    gui_queue = queue.Queue()
    mask_model = get_mask_model()
    corr_model = get_corr_model()
    pose_model = get_pose_model()

    frame_thread = threading.Thread(target=frame_thread_target, args=(frame_stop, track_queue,))
    # track_thread = threading.Thread(target=track_thread_target, args=(
    #     track_stop,
    #     track_queue,
    #     mask_model,
    #     corr_model,
    #     pose_model,
    #     gui_queue,
    #     recon_queue
    # ))
    recon_thread = threading.Thread(target=recon_thread_target, args=(recon_stop, recon_queue))
    gui_thread = threading.Thread(target=gui_thread_target, args=(gui_stop, gui_queue,))

    frame_thread.start()
    # track_thread.start()
    recon_thread.start()
    gui_thread.start()

    # gui_thread_target(gui_queue)
    # frame_thread_target(frame_stop, track_queue)
    track_thread_target(
        track_stop,
        track_queue,
        mask_model,
        corr_model,
        pose_model,
        gui_queue,
        recon_queue
    )

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