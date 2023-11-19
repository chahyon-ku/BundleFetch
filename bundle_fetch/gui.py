import os
from bundle_fetch.utils import inv_transform
import dearpygui.dearpygui as dpg
from queue import Queue
import cv2
import numpy as np


def gui_thread_target(gui_stop, gui_queue: Queue):
    dpg.create_context()
    dpg.create_viewport(width=1280, height=1280)
    dpg.setup_dearpygui()

    with dpg.window(label="",tag="main"):
        dpg.add_group(horizontal=True, tag='row0')
        dpg.add_group(horizontal=True, tag='row1')
        dpg.add_group(horizontal=True, tag='row2')
        dpg.add_group(horizontal=True, tag='row3')
        dpg.add_group(horizontal=True, tag='row4')
    dpg.add_text("frame: 0",tag='frame_id',color=[0,255,0], parent='row3')
    dpg.add_text("keyframe_num: 0",tag='keyframe_num',color=[0,255,0], parent='row3')
    dpg.add_text("nerf_num_frames: X",tag='nerf_num_frames',color=[0,255,0], parent='row3')
    dpg.add_text("frame_num: 0",tag='frame_num',color=[0,255,0], parent='row4')
    dpg.add_text("time_elapsed: 0",tag='time_elapsed',color=[0,255,0], parent='row4')
    dpg.add_text("fps: 0",tag='fps',color=[0,255,0], parent='row4')

    dpg.set_primary_window("main", True)
    dpg.set_viewport_title('BundleFetch')
    dpg.show_viewport()

    IMG_HEIGHT = 480
    IMG_WIDTH = 640
    i_frame = 0

    while dpg.is_dearpygui_running() and not gui_stop.is_set():
        dpg.render_dearpygui_frame()

        frame = gui_queue.get()
        if 'mesh' in frame:
            pass
        else:
            # dpg.set_value('frame_id', f'frame: {frame["i_frame"]}')
            # dpg.set_value('keyframe_num', f'keyframe_num: {len(frame["keyframes"])}')
            # dpg.set_value('nerf_num_frames', f'nerf_num_frames: {frame["nerf_num_frames"]}')
            # dpg.set_value('frame_num', f'frame_num: {frame["frame_num"]}')
            # dpg.set_value('time_elapsed', f'time_elapsed: {frame["time_elapsed"]}')
            # dpg.set_value('fps', f'fps: {frame["fps"]}')

            rgba = frame['rgb'].cpu().numpy()
            rgba = np.concatenate([rgba, np.ones((1, rgba.shape[1], rgba.shape[2]))], axis=0)
            masked_rgba = rgba * frame['mask'].cpu().numpy()[None]
            rgba = rgba.transpose(1, 2, 0)
            masked_rgba = masked_rgba.transpose(1, 2, 0)

            o_T_c = frame['o_T_c'].cpu().numpy()
            c_T_o = inv_transform(o_T_c)
            axes = np.array([[0, 0.1, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0.1], [1, 1, 1, 1]])
            axes = c_T_o @ axes
            axes = frame['cam_K'].cpu().numpy() @ axes[:3]
            axes = axes[:2] / axes[2:3]
            axes = axes.round().astype(int)
            axes[0] = np.clip(axes[0], 0, 639)
            axes[1] = np.clip(axes[1], 0, 479)
            rgba = (rgba * 255).astype(np.uint8).copy()
            # print('c_T_o', c_T_o)
            # print(tuple(axes[:, 0]))
            rgba = cv2.arrowedLine(rgba, tuple(axes[:, 0]), tuple(axes[:, 1]), (0, 0, 255, 255), 2)
            rgba = cv2.arrowedLine(rgba, tuple(axes[:, 0]), tuple(axes[:, 2]), (0, 255, 0, 255), 2)
            rgba = cv2.arrowedLine(rgba, tuple(axes[:, 0]), tuple(axes[:, 3]), (255, 0, 0, 255), 2)
            rgba = (rgba / 255).astype(np.float32)

            if dpg.get_value("rgb") is None:
                with dpg.texture_registry(show=False):
                    dpg.add_dynamic_texture(IMG_WIDTH, IMG_HEIGHT, rgba.reshape(-1), tag="rgb_init")
                    dpg.add_dynamic_texture(IMG_WIDTH, IMG_HEIGHT, masked_rgba.reshape(-1), tag="masked_rgb_init")
                    dpg.add_dynamic_texture(IMG_WIDTH, IMG_HEIGHT, rgba.reshape(-1), tag="rgb")
                    dpg.add_dynamic_texture(IMG_WIDTH, IMG_HEIGHT, masked_rgba.reshape(-1), tag="masked_rgb")
                dpg.add_image("rgb_init", parent='row0')
                dpg.add_image("masked_rgb_init", parent='row0')
                dpg.add_image("rgb", parent='row1')
                dpg.add_image("masked_rgb", parent='row1')
            else:
                dpg.set_value("rgb", rgba.reshape(-1))
                dpg.set_value("masked_rgb", masked_rgba.reshape(-1))
            
            i_frame += 1

    dpg.destroy_context()
