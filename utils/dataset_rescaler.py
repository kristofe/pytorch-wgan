import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import png
import os
import torch
from torchvision import utils
from torchvision import transforms
import numpy as np
from PIL import Image

'''
# mv command... for lots of files
echo *(*single_channel_shrunk_32.png) | xargs mv -t shrunk
'''

class Rescaler:
    def __init__(self, sizes=[4,8,16,32,64,128]):
        #self.IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
        self.IMG_EXTENSIONS = ['.png']
        self.sizes = sizes
    
    def ensure_dir_exists(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def is_image_file(self, filename):
        filename_lower = filename.lower()
        return any(filename_lower.endswith(ext) for ext in self.IMG_EXTENSIONS)


    def convert(self, root):

        images = []
        for base, _, filenames in sorted(os.walk(root)):
            for filename in filenames:
                if self.is_image_file(filename):
                    images.append((os.path.join(base,filename), base, filename))
        if len(images) == 0:
            raise(RuntimeError("No images with extensions {} found in {}".format(
                str(self.IMG_EXTENSIONS), root)))
        counter = 1

        count =  len(images)
        for size in self.sizes:
            self.ensure_dir_exists(f"{root}/{size}/")

        for path, base, filename in images:
            for size in self.sizes:
                #outpath = f"{path[:-4]}_RESCALED_{size}.png"
                outpath = f"{base}R{size}/{filename[:-4]}_R{size}.png"
                print(f"{counter}/{count} path {path} out {outpath}")
                #os.system(f"~/Applications/magick convert {path} -channel A -separate {outpath}")
                os.system(f"~/Applications/magick convert -resize {size}x{size} {path} {outpath}")
            counter += 1

if __name__ == "__main__":
    rescaler = Rescaler()
    rescaler.convert("datasets/erosion_targets50k/256/test/")
    rescaler.convert("datasets/erosion_targets50k/256/train/")
    