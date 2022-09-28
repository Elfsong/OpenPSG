# coding: utf-8

import io
import os
import json
import logging

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torchvision.transforms as trn

# To fix "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)

def get_transforms(stage: str, input_size: int):
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    if stage == 'train':
        return trn.Compose([
            Convert('RGB'),
            trn.Resize((input_size, input_size)),
            trn.RandomHorizontalFlip(),
            trn.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
            trn.RandomCrop((input_size, input_size), padding=4),
            trn.ToTensor(),
            trn.Normalize(mean, std),
        ])

    elif stage in ['val', 'test']:
        return trn.Compose([
            Convert('RGB'),
            trn.Resize((input_size, input_size)),
            trn.ToTensor(),
            trn.Normalize(mean, std),
        ])

class PSGClsDataset(Dataset):
    def __init__(self, stage, root='./data/coco/', label_path="./data/psg/psg_cls_basic.json", num_classes=56, input_size=224):
        super(PSGClsDataset, self).__init__()
        with open(label_path) as f:
            dataset = json.load(f)

        self.imglist = [
            d for d in dataset['data'] if d['image_id'] in dataset[f'{stage}_image_ids']
        ]

        self.root = root
        self.transform_image = get_transforms(stage, input_size)
        self.num_classes = num_classes
        self.input_size = input_size

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        sample = self.imglist[index]
        path = os.path.join(self.root, sample['file_name'])

        try:
            with open(path, 'rb') as f:
                content = f.read()
                filebytes = content
                buff = io.BytesIO(filebytes)
                image = Image.open(buff).convert('RGB')
                sample['data'] = self.transform_image(image)
        except Exception as e:
            logging.error('Error, cannot read [{}]'.format(path))
            raise e
            
        # Generate Soft Label
        soft_label = torch.Tensor(self.num_classes)
        soft_label.fill_(0)
        soft_label[sample['relations']] = 1
        sample['soft_label'] = soft_label
        del sample['relations']

        return sample



