import matplotlib.pyplot as plt
from bundle_fetch.sam_gui import Segmenter
import os
from glob import glob
import hydra
import numpy as np
from PIL import Image


@hydra.main(config_path='../conf', config_name='sam_segment_click', version_base=None)
def main(cfg):
    rgb_path = list(sorted(glob(os.path.join(cfg.data_dir, 'rgb', '*.png'))))[20]
    rgb_img = np.array(Image.open(rgb_path))
    mask_path = os.path.join(cfg.data_dir, 'masks', os.path.basename(rgb_path))
    
    segmenter = Segmenter(rgb_img, mask_path)
    plt.imshow(segmenter.mask)
    plt.show()
    

if __name__ == '__main__':
    main()