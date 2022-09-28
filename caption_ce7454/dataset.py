# coding: utf-8

import io
import os
import json
import logging
from typing import Any

import torch
from pathlib import Path
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
    def __init__(self, stage: str, tokenizer: Any, root=Path('./data/'), num_classes=56, input_size=224):
        super(PSGClsDataset, self).__init__()

        self.stage = stage
        self.root = root
        self.transform_image = get_transforms(stage, input_size)
        self.num_classes = num_classes
        self.input_size = input_size
        self.tokenizer = tokenizer
        self.working_dir = self.root / self.stage
        self.image_text_pair_list = list()
        
        for image_path in self.working_dir.iterdir():
            name = image_path.name
            text = " ".join(name[:-4].split(" ")[1:])

            self.image_text_pair_list += [{
                "text": text,
                "path": image_path
            }]
            
    def __len__(self):
        return len(self.image_text_pair_list)

    def __getitem__(self, index):
        sample = self.image_text_pair_list[index]

        # Source (image)
        try:
            with open(sample["path"], 'rb') as f:
                content = f.read()
                filebytes = content
                buff = io.BytesIO(filebytes)
                image = Image.open(buff).convert('RGB')
                sample['source'] = self.transform_image(image)
        except Exception as e:
            logging.error('Error, cannot read [{}]'.format(sample["path"]))
            raise e
        del sample["path"]

        # Target (text)
        text_encode = self.tokenizer(sample["text"], return_tensors='pt', return_attention_mask=True, padding='max_length', max_length=16)
        sample["target"] = text_encode.input_ids
        sample["attention_mask"] = text_encode.attention_mask
            
        return sample



