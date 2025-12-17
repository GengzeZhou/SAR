import torch
import numpy as np
import os
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
from torchvision.transforms import InterpolationMode, transforms
from dataset.augmentation import random_crop_arr, center_crop_arr
import random


# Add a callable class to replace the lambda function
class RandomCropTransform:
    def __init__(self, final_reso):
        self.final_reso = final_reso
    
    def __call__(self, pil_image):
        return random_crop_arr(pil_image, self.final_reso)


class ImageNet(VisionDataset):
    def __init__(
        self,
        root: str,
        transform = None,
        target_transform = None,
        loader = default_loader,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)


        self.root = root
        self.loader = loader



        self.samples = self.make_dataset()


        print('total images:', len(self.samples))


    def make_dataset(self,):
        instances = []

        for split in ['train', ]:
            file_path = os.path.join(self.root, f'{split}.txt')
            with open(file_path, 'r') as file:
                lines = file.readlines()
            for line in lines:
                x = line.strip().split(' ')
                instances.append((os.path.join(self.root, split, x[0]), int(x[1])))

        random.shuffle(instances)

        return instances


    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)
    
def build_imagenet(args, ):
    train_aug = [
        transforms.RandomHorizontalFlip(),
        RandomCropTransform(args.final_reso),  # Replace lambda with callable class
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ] 
    train_aug = transforms.Compose(train_aug)

    return ImageNet(args.data_path, transform=train_aug)

