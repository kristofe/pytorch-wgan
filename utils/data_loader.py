import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from utils.fashion_mnist import MNIST, FashionMNIST
import png
import os
import torch
from torchvision import utils
from torchvision import transforms
import numpy as np
from PIL import Image

# Created custom image folder that can handle 16 bit pngs
class SyntheticImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform, crop_size=32, cache=False, device=torch.device('cpu')):
        #self.IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
        self.IMG_EXTENSIONS = ['.png']
        self.root = root
        self.crop_size = crop_size
        self.num_crops = 1024//self.crop_size
        self.cached = cache
        self.device = device
        self.force_from_disk = False
        self.img_paths = self.get_image_paths()
        self.transform = transform #transforms.Normalize([0.5],[0.5])
        if len(self.img_paths) == 0:
            raise(RuntimeError("No images with extensions {} found in {}".format(
                str(self.IMG_EXTENSIONS), self.root)))

        if self.cached:
          self.force_from_disk = True
          self.cache_images()
          self.force_from_disk = False
        
    
    def convert(self):
        counter = 1
        count =  len(self.img_paths)
        for path in self.img_paths:
            outpath = f"{path[:-4]}_shrunk_32.png"
            print(f"{count}/{counter} path {path} out {outpath}")
            #os.system(f"~/Applications/magick convert {path} -channel A -separate {outpath}")
            os.system(f"~/Applications/magick convert -resize 32x32 {path} {outpath}")
            counter += 1
        '''
        # mv command... for lots of files
        echo *(*single_channel_shrunk_32.png) | xargs mv -t shrunk
        '''

    
    def normalize01_with_minmax(self, data, data_min, data_max):
        diff = data_max - data_min
        if diff > 0.0:
            data = data.sub(data_min).div(data_max - data_min)
        return data

    def cache_images(self):
      cache_filename = f"{self.root}cache.pth"
      if os.path.exists(cache_filename):
        print(f"loading {cache_filename}")
        self.cached_images = torch.load(cache_filename, map_location=self.device)
        self.min = self.cached_images.min()
        self.max = self.cached_images.max()
        print(f"min {self.min}  max {self.max}")
        return

      count = len(self.img_paths)
      print(f"Caching {count} images")
      cache_inited = False
      for i in range(count):
        print(f"Caching {i}/{count}  ", end="\r")
        img = self.__getitem__(i)[0]
        if not cache_inited:
          print(f"image shape {img.shape}")
          c,h,w = img.shape
          self.cached_images= torch.zeros(count,c,h,w, dtype=torch.float32, device=self.device)
          cache_inited = True
        self.cached_images[i] = img
      print("done.........               ")
      self.min = self.cached_images.min()
      self.max = self.cached_images.max()
      print(f"min {self.min}  max {self.max}")
      #self.cached_images = self.normalize01_with_minmax(self.cached_images, self.min, self.max)
      self.min = self.cached_images.min()
      self.max = self.cached_images.max()
      print(f"min {self.min}  max {self.max}")
      torch.save(self.cached_images, cache_filename)
    
    def read_uint16_png(self, filepath):
        with open(filepath, 'rb') as f:
            reader = png.Reader(f)  # png.Reader(filepath)
            _, _, pngdata, _ = reader.asDirect()
            pixels = np.vstack(list(pngdata))#.astype(np.uint16)
            pixels_float = pixels.astype(np.float32)
            pixels_float =  torch.from_numpy(pixels_float)

            if pixels_float.ndim == 2:
              pixels_float = pixels_float.unsqueeze(0)
            return pixels_float
    

    def rand_grid_crop(self, data):
        assert(data.shape[2] == 256)
        assert(data.shape[0] == 1)

        #img = torch.nn.functional.interpolate(data.unsqueeze(0), size=32).squeeze()

        #i = torch.randint(low=0, high=self.num_crops, size=(1,))
        #j = torch.randint(low=0, high=self.num_crops, size=(1,))
        high = 256//32 - 1
        i = torch.randint(low=0, high=high, size=(1,))
        j = torch.randint(low=0, high=high, size=(1,))

        xs = i*self.crop_size
        xe = xs + self.crop_size
        ys = j*self.crop_size
        ye = ys + self.crop_size
        img = data[0:1,ys:ye,xs:xe]


        return img


    def image_loader(self, path):
        return self.read_uint16_png(path)

    def is_image_file(self, filename):
        filename_lower = filename.lower()
        return any(filename_lower.endswith(ext) for ext in self.IMG_EXTENSIONS)

    def get_image_paths(self):
        images = []
        for base, _, filenames in sorted(os.walk(self.root)):
            for file in filenames:
                if self.is_image_file(file):
                    images.append(os.path.join(base,file))
        return images

    def __getitem__(self, index):
        if self.cached and not self.force_from_disk:
            img = self.cached_images[index]
        else:
            path = self.img_paths[index]
            img = self.image_loader(path)
            #print(f"path {path}  img {img.shape}")
            if self.transform is not None:
                img = self.transform(img)
            #print(f"path {path}  img {img.shape}")
        #img = self.rand_grid_crop(img)
        return img,img
                

    def __len__(self):
        return len(self.img_paths)
    
    def size(self):
        return len(self.img_paths)

def get_data_loader(args):

    if args.dataset == 'mnist':
        trans = transforms.Compose([
            transforms.Scale(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train_dataset = MNIST(root=args.dataroot, train=True, download=args.download, transform=trans)
        test_dataset = MNIST(root=args.dataroot, train=False, download=args.download, transform=trans)

    elif args.dataset == 'fashion-mnist':
        trans = transforms.Compose([
            transforms.Scale(32),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Normalize([0.5], [0.5]),
        ])
        train_dataset = FashionMNIST(root=args.dataroot, train=True, download=args.download, transform=trans)
        test_dataset = FashionMNIST(root=args.dataroot, train=False, download=args.download, transform=trans)

    elif args.dataset == 'cifar':
        trans = transforms.Compose([
            transforms.Scale(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_dataset = dset.CIFAR10(root=args.dataroot, train=True, download=args.download, transform=trans)
        test_dataset = dset.CIFAR10(root=args.dataroot, train=False, download=args.download, transform=trans)

    elif args.dataset == 'stl10':
        trans = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])
        train_dataset = dset.STL10(root=args.dataroot, train=True, download=args.download, transform=trans)
        test_dataset = dset.STL10(root=args.dataroot, train=False, download=args.download, transform=trans)

    elif args.dataset == 'terrain':
        trans = transforms.Compose([
            transforms.Normalize([0.5], [0.5]),
        ])
        crop_size = 32
        train_dataset = SyntheticImageFolder(root=args.dataroot + "train/", transform=trans, crop_size=crop_size, cache=True)
        test_dataset = SyntheticImageFolder(root=args.dataroot + "test/", transform=trans, crop_size=crop_size, cache=True)

    # Check if everything is ok with loading datasets
    assert train_dataset
    assert test_dataset

    train_dataloader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_dataloader  = data_utils.DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=True, num_workers=2)

    return train_dataloader, test_dataloader
