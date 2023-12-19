from matplotlib import pyplot as plt
import pyrealsense2 as rs
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import cv2

from bundle_fetch.utils import nvtx_range

class FrameRealsense:
    def __init__(self):
        self.config = rs.config()
        self.pipeline = rs.pipeline()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        self.align = rs.align(rs.stream.color)
        self.colorizer = rs.colorizer()
        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            # im_normalization,
        ])
        self.i_frame = 0
        
        self.prev_mask = None
        self.next_obj_timer = 0
        self.n_obj = 0

    def get_frame(self):
        with nvtx_range('get_frame'):
            # if self.i_frame == 30:
            #     return None
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            color_image = np.asanyarray(color_frame.get_data()) 
            depth_image = np.asanyarray(depth_frame.get_data())

            depth_scaled = (depth_image * self.depth_scale)
            intr = aligned_frames.get_profile().as_video_stream_profile().get_intrinsics()
            intr_mat = torch.tensor(np.array([
                [intr.fx, 0, intr.ppx],
                [0, intr.fy, intr.ppy],
                [0, 0, 1],
            ])).float()
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # images = np.hstack((color_image, depth_colormap))

            # cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
            # cv2.imshow('Align Example', images)
            # cv2.waitKey(1)

            color_image = Image.fromarray(color_image[..., ::-1]).convert('RGB')
            color_image = self.rgb_transform(color_image)
            depth = torch.from_numpy(depth_scaled).float()[None]
            self.i_frame += 1

            mask = ((0 < depth) & (depth < 0.6)).float()
            n_masked_pixels = mask.sum()

            cv2.imshow('mask', mask[0].numpy())
            cv2.waitKey(1)

            frame = {'rgb': color_image, 'depth': depth, 'cam_K': intr_mat}
            if self.next_obj_timer > 0:
                self.next_obj_timer -= 1
                # print('next obj counter', self.next_obj_timer)
            elif n_masked_pixels > 1000:
                if self.prev_mask is not None:
                    diff = (mask - self.prev_mask).abs().sum()
                    diff_thresh = n_masked_pixels / 25
                    if diff < diff_thresh:
                        frame['mask'] = mask
                        self.next_obj_timer = 30
                        self.n_obj += 1
                        print('diff', diff, diff_thresh)
                self.prev_mask = mask
            return frame