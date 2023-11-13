import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from glob import glob
import imageio
from PIL import Image
from torchvision import transforms
from XMem.dataset.range_transform import im_normalization


class BfDataset(Dataset):
    def __init__(self, data_dir, schema=['rgb', 'masks', 'depth']) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.schema = schema
        self.rgb_paths = list(sorted(glob(os.path.join(data_dir, 'rgb', '*.png'))))
        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            # im_normalization,
        ])
        self.cam_K = np.loadtxt(os.path.join(data_dir, 'cam_K.txt'))

    def __len__(self) -> int:
        return len(self.rgb_paths)

    def __getitem__(self, index: int):
        rgb_path = self.rgb_paths[index]
        data = {}
        data['rgb_path'] = rgb_path
        data['cam_K'] = torch.from_numpy(self.cam_K).float()
        
        for key in self.schema:
            image_path = rgb_path.replace('rgb', key)
            if not os.path.exists(image_path):
                continue
            if key == 'rgb':
                image = Image.open(image_path).convert('RGB')
                image = self.rgb_transform(image) # 1, 480, 640
            elif key == 'masks':
                if index > 0:
                    continue
                image = Image.open(image_path).convert('P')
                image = np.array(image, np.uint8)
                labels = np.unique(image)
                labels = labels[labels!=0]
                new_image = np.zeros((len(labels), *image.shape), dtype=np.uint8)
                for i_label, label in enumerate(labels):
                    new_image[i_label, image == label] = 1
                image = torch.from_numpy(new_image).float() # N, 480, 640
                key = 'mask'
            elif key == 'depth':
                image = imageio.imread(image_path)
                image = torch.from_numpy(image).float()[None] # 1, 480, 640
            else:
                raise NotImplementedError
            data[key] = image
        
        return data
        