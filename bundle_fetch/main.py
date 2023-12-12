

import threading
import queue
from bundle_fetch.track.track import Track
from bundle_fetch.frame.frame import Frame
from bundle_fetch.recon.recon import recon_thread_target
from bundle_fetch.gui import gui_thread_target

def main():
    frame_stop = threading.Event()
    track_stop = threading.Event()
    recon_stop = threading.Event()
    gui_stop = threading.Event()
    track_queue = queue.Queue(5)
    recon_queue = queue.Queue()
    gui_queue = queue.Queue()

    frame = Frame(frame_stop, track_queue)
    track = Track(track_stop, track_queue, gui_queue)

    frame_thread = threading.Thread(target=frame)
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
    # recon_thread.start()
    gui_thread.start()

    # gui_thread_target(gui_queue)
    # frame_thread_target(frame_stop, track_queue)
    track()

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