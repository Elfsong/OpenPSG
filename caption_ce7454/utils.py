# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2022 / 09 / 22

import csv
import cv2
import json
import torch
import numpy as np
import itertools as it
from typing import Any
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from panopticapi.utils import rgb2id
from detectron_viz import Visualizer
from detectron2.structures import Instances
from detectron2.utils.colormap import colormap
from detectron2.utils.visualizer import ColorMode


def get_path(path: str) -> Path:
    return Path(path)

def load_json(path: Path) -> Any:
    with path.open() as f:
        data = json.load(f)
    return data

def draw_img(img: np.array) -> None:
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def save_img(img: np.array, name: str, path: Path) -> None:
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(path / f"{name}.png", bbox_inches='tight', pad_inches=0.0)

def clean_plt() -> None:
    plt.clf()

def rgb2id(color: Any) -> int:
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

def get_seg_image_1(img: np.array, masks: list, index: list, colors: list) -> np.array:
    masks = np.array(masks)[index]
    masks = torch.tensor(masks)
    pred_instances = Instances((img.shape[0], img.shape[1]), pred_masks=masks, colors=colors)
    viz = Visualizer(img, instance_mode=ColorMode.IMAGE_BW)
    out = viz.draw_instance_predictions(pred_instances)
    viz_img = out.get_image()
    return viz_img

def get_seg_image_2(img: np.array, masks: list, index: list) -> np.array:
    # My stupid way, don't use it...
    mask = [masks[i] for i in index]
    mask = (1 * np.logical_or.reduce(mask)).astype('uint8')
    mask_inv = cv2.bitwise_not(mask)

    res = cv2.bitwise_and(img, img, mask=mask)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    background = cv2.bitwise_and(gray, gray, mask = mask_inv)
    background = np.stack((background,)*3, axis=-1)
    img_ca = cv2.add(res, background)
    return img_ca

def seg_ratio(img: np.array, mask: np.array) -> float:
    total = img.shape[0] * img.shape[1]
    seg = np.sum(1*mask)
    return seg / total


def generate_report(pred_list: list, gt_list: list) -> None:
    # Temporary function
    with open("results.csv", "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows([[p, g] for p, g in zip(pred_list, gt_list)])

def generate_all_possible_sequences(objects: list, relations: list) -> list:
    # CAUTION: Massive output. Don't print the return!
    return [e for e in it.product(objects, relations, objects)]


def batch_encode(tokenizer: Any, triplet_list: list) -> list:
    for triplet in triplet_list[:500]:
        text = " ".join(triplet)
        text_encode = tokenizer(text, return_tensors='pt', return_attention_mask=True, padding='max_length', max_length=16)
        print(text)
        print(text_encode)
        print("===========================")
