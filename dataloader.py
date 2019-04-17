import os
import numpy as np
from glob import glob
from PIL import Image, ImageFilter

import torch
import torch.utils.data as data
import torchvision.transforms as transforms


def get_dataset(patch_size, upscale_factor, phase='train'):
    dir_list = glob(os.path.join('data', phase, '*'))
    files = []
    for directory in dir_list:
        files += [file for file in glob(os.path.join(directory,'*')) if 
                  is_image_file(file) and is_size_bigger(file,patch_size)]

    return DatasetFromFolder(files, patch_size, upscale_factor, phase)

class DatasetFromFolder(data.Dataset):
    def __init__(self, files, patch_size, upscale_factor, phase):
        super(DatasetFromFolder, self).__init__()
        
        self.files = files
        self.upscale_factor = upscale_factor
        
        self.target_transform = None
        self.input_transform = None
        self.totensor_transform = transforms.ToTensor()
        
        if phase == 'train':
            self.crop_size = self.valid_crop_size(patch_size)
            self.target_transform = self.train_target_transform()
            self.input_transform = self.train_input_transform()

        
    def __getitem__(self, index):
        target = self.load_img(self.files[index])
        if self.target_transform:
            target = self.target_transform(target)
        else:
            target = target.crop((0,0,
                                  self.valid_crop_size(target.size[0]),
                                  self.valid_crop_size(target.size[1])))
        
        input = target.copy()
        if self.input_transform:
            input = self.input_transform(input)
        else: 
            input = input.resize((input.size[0]//self.upscale_factor,
                                  input.size[1]//self.upscale_factor))
        
        input_blur = self.totensor_transform(self.filter_blur(input))
        input_sharpen = self.totensor_transform(self.filter_sharpen(input))        
        input = self.totensor_transform(input)
        input = torch.clamp(input + torch.randn_like(input)*torch.sqrt(torch.rand(1)*0.05),0.,1.)
        input = torch.cat((input,input_blur,input_sharpen), dim=0)
        
        target = self.totensor_transform(target)
        
        return input, target

    
    def __len__(self):
        return len(self.files)
    
    
    def train_target_transform(self):
        return transforms.Compose([
            transforms.RandomCrop(self.crop_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
        ])


    def train_input_transform(self):
        return transforms.Compose([
            transforms.Resize(self.crop_size // self.upscale_factor),
        ])
        
    
    def valid_crop_size(self, patch_size):
        return patch_size - (patch_size % self.upscale_factor)
    
    
    def load_img(self, filepath):
        img = Image.open(filepath).convert('YCbCr')
        y, _, _ = img.split()
        return y
    
    
    def filter_blur(self, y):
        im = y.filter(ImageFilter.GaussianBlur(2))
        return im


    def filter_sharpen(self, y):
        im = y.filter(ImageFilter.SMOOTH)
        im = im.filter(ImageFilter.DETAIL)
        im = im.filter(ImageFilter.EDGE_ENHANCE_MORE)
        return im
    
    
# ===========================================================
# ===========================================================

    
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def is_size_bigger(file,patch_size):
    return all(np.array(Image.open(file).size) > patch_size)




            
