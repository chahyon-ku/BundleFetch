

import torch.multiprocessing as mp
from bundle_fetch.track.track import Track
from bundle_fetch.frame.frame import Frame
from bundle_fetch.spot.spot import Spot
from bundle_fetch.gui import gui_process_target
import time

def main(frame_type):
    mp.set_start_method('spawn')
    frame_stop = mp.Event()
    track_stop = mp.Event()
    gui_stop = mp.Event()
    track_queue = mp.Queue(1)
    gui_queue = mp.Queue(1)
    spot_queue = mp.Queue(1)

    frame = Frame(frame_stop, track_queue, frame_type, spot_queue)
    track = Track(track_stop, track_queue, gui_queue, spot_queue)
    spot = Spot(spot_queue)

    frame_process = mp.Process(target=frame)
    track_process = mp.Process(target=track)
    gui_process = mp.Process(target=gui_process_target, args=(gui_stop, gui_queue))
    spot_process = mp.Process(target=spot)

    frame_process.start()
    track_process.start()
    gui_process.start()
    spot_process.start()

    while True:
        time.sleep(1)
    