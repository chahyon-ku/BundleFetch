# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Simple image display example."""

import cv2
from matplotlib import pyplot as plt
import numpy as np

import bosdyn.client
import bosdyn.client.util
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.api import gripper_camera_param_pb2, header_pb2
from PIL import Image
import torch
from torchvision.transforms import ToTensor


class FrameSpot(object):
    def __init__(self):
        # Create robot
        self.sdk = bosdyn.client.create_standard_sdk('bundle_fetch')
        # self.robot = self.sdk.create_robot('192.168.80.3')
        self.robot = self.sdk.create_robot('10.0.0.3')
        bosdyn.client.util.authenticate(self.robot)
        self.robot.sync_with_directory()
        self.robot.time_sync.wait_for_sync()

        # Create clients
        self.image_client: ImageClient = self.robot.ensure_client(ImageClient.default_service_name)
        image_sources = ['hand_color_image', 'hand_depth_in_hand_color_frame']
        self.image_requests = [
            build_image_request(
                source,
                quality_percent=50,
                resize_ratio=1
            ) for source in image_sources
        ]
        
        # Object mask filtering
        self.prev_mask = None
        self.next_obj_timer = 0
        self.n_obj = 0

    
    def get_frame(self, spot_queue):
        images = self.image_client.get_image(self.image_requests)
        np_images = [image_to_opencv(image)[0] for image in images]
        
        rgb = torch.from_numpy(np_images[0].astype(np.float32)[..., ::-1] / 255).permute(2, 0, 1)
        depth = torch.from_numpy(np_images[1].astype(np.float32) / 1000).permute(2, 0, 1)
        cam_K = torch.from_numpy(np.array([
            [images[0].source.pinhole.intrinsics.focal_length.x, 0, images[0].source.pinhole.intrinsics.principal_point.x],
            [0, images[0].source.pinhole.intrinsics.focal_length.y, images[0].source.pinhole.intrinsics.principal_point.y],
            [0, 0, 1]]
        ).astype(np.float32))

        mask = ((0.5 < depth) & (depth < 0.6)).float()
        n_masked_pixels = mask.sum()

        cv2.imshow('mask', mask[0].numpy())
        cv2.waitKey(1)

        frame = {'rgb': rgb, 'depth': depth, 'cam_K': cam_K}
        if self.next_obj_timer > 0:
            self.next_obj_timer -= 1
            try:
                spot_queue.put(0, timeout=0.1)
            except:
                pass
            # print('next obj counter', self.next_obj_timer)
        elif n_masked_pixels > 1000:
            try:
                spot_queue.put(0.01, timeout=0.1)
            except:
                pass
            if self.prev_mask is not None:
                diff = (mask - self.prev_mask).abs().sum()
                diff_thresh = n_masked_pixels / 10
                if diff < diff_thresh:
                    frame['mask'] = mask
                    self.next_obj_timer = 10
                    self.n_obj += 1
                    print('diff', diff, diff_thresh)
            self.prev_mask = mask

        return frame
        

def image_to_opencv(image):
    """Convert an image proto message to an openCV image."""
    num_channels = 1  # Assume a default of 1 byte encodings.
    if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        dtype = np.uint16
        extension = '.png'
    else:
        dtype = np.uint8
        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
            num_channels = 3
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
            num_channels = 4
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
            num_channels = 1
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16:
            num_channels = 1
            dtype = np.uint16
        extension = '.jpg'

    img = np.frombuffer(image.shot.image.data, dtype=dtype)
    if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
        try:
            # Attempt to reshape array into a RGB rows X cols shape.
            img = img.reshape((image.shot.image.rows, image.shot.image.cols, num_channels))
        except ValueError:
            # Unable to reshape the image data, trying a regular decode.
            img = cv2.imdecode(img, -1)
    else:
        img = cv2.imdecode(img, -1)

    return np.ascontiguousarray(img), extension

