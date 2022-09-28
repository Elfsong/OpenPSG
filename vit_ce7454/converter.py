# coding: utf-8

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from tqdm import tqdm
from PIL import Image
from torch import nn


import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import os


def ade_palette():
    return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]

feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model.to(device)


def generate(input_path, output_path):
    image = Image.open(input_path)

    pixel_values = feature_extractor(image, return_tensors="pt").pixel_values.to(device)
    outputs = model(pixel_values)
    logits = outputs.logits

    # First, rescale logits to original image size
    logits = nn.functional.interpolate(outputs.logits,
                    size=image.size[::-1], # (height, width)
                    mode='bilinear',
                    align_corners=False)

    # Second, apply argmax on the class dimension
    seg = logits.argmax(dim=1)[0].cpu()
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
    palette = np.array(ade_palette())
    objects = list()
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    color_seg = color_seg[..., ::-1]

    # Show image + mask
    img = np.array(image) * 0.3 + color_seg * 0.7
    img = img.astype(np.uint8)

    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.axis("off")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

    return objects


if __name__ == "__main__":
    input_directory = "./data/coco/val2017"
    output_directory = "./data/coco/val2017_sg_8"
    
    for filename in tqdm(os.listdir(input_directory)):
        input_file = os.path.join(input_directory, filename)
        output_file = os.path.join(output_directory, filename)
        if os.path.isfile(input_file):
            try:
                generate(input_file, output_file)
            except:
                os.popen(f'cp {input_file} {output_directory}') 
    
    input_directory = "./data/coco/train2017"
    output_directory = "./data/coco/train2017_sg_8"
    
    for filename in tqdm(os.listdir(input_directory)):
        input_file = os.path.join(input_directory, filename)
        output_file = os.path.join(output_directory, filename)
        if os.path.isfile(input_file):
            try:
                generate(input_file, output_file)
            except:
                os.popen(f'cp {input_file} {output_directory}') 