import os
import shutil
import pyrealsense2 as rs
import numpy as np
import cv2
import imageio
import hydra
from PIL import Image


@hydra.main(config_path='../conf', config_name='record_video')
def main(cfg):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, cfg.width, cfg.height, rs.format.z16, cfg.fps)
    config.enable_stream(rs.stream.color, cfg.width, cfg.height, rs.format.bgr8, cfg.fps)
    queue = rs.frame_queue(2, keep_frames=True)
    profile = pipeline.start(config, queue)
    print('width', cfg.width)
    print('height', cfg.height)
    print('fps', cfg.fps)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print('depth_scale', depth_scale)
    clipping_distance_in_meters = 1 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    align_to = rs.stream.color
    align = rs.align(align_to)

    i_frame = 0
    try:
        frames = queue.wait_for_frame()
        aligned_frames = align.process(rs.composite_frame(frames))
        intr = aligned_frames.get_profile().as_video_stream_profile().get_intrinsics()
        print(intr)
        intr_mat = np.array([
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1],
        ])
        with open(os.path.join(cfg.output_dir, 'cam_K.txt'), 'w') as f:
            f.write(f'{intr_mat[0, 0]} 0 {intr_mat[0, 2]}\n')
            f.write(f'0 {intr_mat[1, 1]} {intr_mat[1, 2]}\n')
            f.write(f'0 0 1\n')
    finally:
        pipeline.stop()


if __name__ == '__main__':
    main()