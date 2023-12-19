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

    init_rgba = 255 * np.ones((IMG_HEIGHT, IMG_WIDTH * 2, 4), dtype=np.uint8)
    rgba = 255 * np.ones((IMG_HEIGHT * 2, IMG_WIDTH * 2, 4), dtype=np.uint8)

    while dpg.is_dearpygui_running() and not gui_stop.is_set():
        dpg.render_dearpygui_frame()

        try:
            frame, vertices = gui_queue.get(timeout=0.1)
        except:
            continue
        
        # dpg.set_value('frame_id', f'frame: {frame["i_frame"]}')
        # dpg.set_value('keyframe_num', f'keyframe_num: {len(frame["keyframes"])}')
        # dpg.set_value('nerf_num_frames', f'nerf_num_frames: {frame["nerf_num_frames"]}')
        # dpg.set_value('frame_num', f'frame_num: {frame["frame_num"]}')
        # dpg.set_value('time_elapsed', f'time_elapsed: {frame["time_elapsed"]}')
        # dpg.set_value('fps', f'fps: {frame["fps"]}')
        # print('c_T_o', c_T_o)
        # print(tuple(axes[:, 0]))

        if dpg.get_value("rgb") is None:
            init_rgba[:IMG_HEIGHT, :IMG_WIDTH, :3] = frame['rgb'].transpose(1, 2, 0) * 255
            init_rgba[:IMG_HEIGHT, IMG_WIDTH:, :3] = rgba[:IMG_HEIGHT, :IMG_WIDTH, :3]
            
            for vertex in vertices:
                init_rgba[:IMG_HEIGHT, IMG_WIDTH:, :3] *= vertex['mask'][None].transpose(1, 2, 0)

                c_T_o = vertex['c_T_o']
                axes = np.array([[0, 0.1, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0.1], [1, 1, 1, 1]])
                axes = c_T_o @ axes
                axes = frame['cam_K'] @ axes[:3]
                axes = axes[:2] / axes[2:3]
                axes = axes.round().astype(int)
                axes[0] = np.clip(axes[0], 0, 639)
                axes[1] = np.clip(axes[1], 0, 479)
                init_rgba = cv2.arrowedLine(init_rgba, tuple(axes[:, 0]), tuple(axes[:, 1]), (0, 0, 255, 255), 2)
                init_rgba = cv2.arrowedLine(init_rgba, tuple(axes[:, 0]), tuple(axes[:, 2]), (0, 255, 0, 255), 2)
                init_rgba = cv2.arrowedLine(init_rgba, tuple(axes[:, 0]), tuple(axes[:, 3]), (255, 0, 0, 255), 2)

            rgba[:IMG_HEIGHT] = init_rgba[:IMG_HEIGHT]

            with dpg.texture_registry(show=False):
                dpg.add_dynamic_texture(IMG_WIDTH * 2, IMG_HEIGHT * 2, (rgba / 255).astype(np.float32).reshape(-1), tag="rgb")
            dpg.add_image("rgb", parent='row0')
        else:
            rgba[:IMG_HEIGHT] = init_rgba[:IMG_HEIGHT]
            rgba[IMG_HEIGHT:, :IMG_WIDTH, :3] = frame['rgb'].transpose(1, 2, 0) * 255
            rgba[IMG_HEIGHT:, IMG_WIDTH:, :3] = frame['rgb'].transpose(1, 2, 0) * 255

            all_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
            for vertex in vertices:
                all_mask += vertex['mask'][None].transpose(1, 2, 0)
                c_T_o = vertex['c_T_o']
                axes = np.array([[0, 0.1, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0.1], [1, 1, 1, 1]])
                axes = c_T_o @ axes
                axes = frame['cam_K'] @ axes[:3]
                axes = axes[:2] / axes[2:3]
                axes = axes.round().astype(int)
                axes[0] = np.clip(axes[0], 0, 639)
                axes[1] = np.clip(axes[1], 0, 479)
                axes[1] += 480
                rgba = cv2.arrowedLine(rgba, tuple(axes[:, 0]), tuple(axes[:, 1]), (0, 0, 255, 255), 2)
                rgba = cv2.arrowedLine(rgba, tuple(axes[:, 0]), tuple(axes[:, 2]), (0, 255, 0, 255), 2)
                rgba = cv2.arrowedLine(rgba, tuple(axes[:, 0]), tuple(axes[:, 3]), (255, 0, 0, 255), 2)

                if 'uv_a' in vertex:
                    uv_a = vertex['uv_a'].cpu().numpy().round().astype(int)
                    uv_b = vertex['uv_b'].cpu().numpy().round().astype(int)
                    conf = vertex['conf'].cpu().numpy()

                    # draw correspondences
                    for i in range(uv_a.shape[0]):
                        if conf[i] > 0:
                            rgba = cv2.line(rgba, tuple(uv_a[i]), (uv_b[i, 0], uv_b[i, 1] + 480), (0, 255, 0, 255), 1)
                    
            rgba[IMG_HEIGHT:, IMG_WIDTH:, :3] *= (all_mask > 0)

            dpg.set_value("rgb", (rgba / 255).astype(np.float32).reshape(-1))
        
        # update frame number
        dpg.set_value('frame_id', f'frame: {i_frame}')
        i_frame += 1

    dpg.destroy_context()
