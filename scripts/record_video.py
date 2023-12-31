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

    shutil.rmtree(cfg.output_dir, ignore_errors=True)
    os.makedirs(os.path.join(cfg.output_dir, 'rgb'), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, 'depth'), exist_ok=True)

    i_frame = 0
    try:
        while True:
            frames = queue.wait_for_frame()
            aligned_frames = align.process(rs.composite_frame(frames))
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                raise RuntimeError('no depth or color frame')

            if i_frame == 0:
                intr = aligned_frames.get_profile().as_video_stream_profile().get_intrinsics()
                intr_mat = np.array([
                    [intr.fx, 0, intr.ppx],
                    [0, intr.fy, intr.ppy],
                    [0, 0, 1],
                ])
                with open(os.path.join(cfg.output_dir, 'cam_K.txt'), 'w') as f:
                    f.write(f'{intr_mat[0, 0]} 0 {intr_mat[0, 2]}\n')
                    f.write(f'0 {intr_mat[1, 1]} {intr_mat[1, 2]}\n')
                    f.write(f'0 0 1\n')

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())[..., ::-1]

            depth_scaled = (depth_image * depth_scale * 1000).astype(np.uint16)
            
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((color_image, depth_colormap))

            cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
            cv2.imshow('Align Example', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

            if i_frame % 3 == 0:
                imageio.imwrite(os.path.join(cfg.output_dir, f'rgb/{i_frame:06d}.png'), color_image)
                imageio.imwrite(os.path.join(cfg.output_dir, f'depth/{i_frame:06d}.png'), depth_scaled)
            i_frame += 1
    finally:
        pipeline.stop()


if __name__ == '__main__':
    main()