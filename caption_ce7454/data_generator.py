# coding: utf-8

# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2022 / 09 / 22

import utils
from tqdm import tqdm
from pathlib import Path
from detectron2.data.detection_utils import read_image
from transformers import AutoTokenizer


# Set working path
json_path = utils.get_path("/home/mingzhe/Projects/OpenPSG/ce7454/data/psg/psg_cls_advanced.json")
coco_path = utils.get_path("/home/mingzhe/Projects/OpenPSG/ce7454/data/coco")

psg_dataset_file = utils.load_json(json_path)
psg_rel_cats = psg_dataset_file['predicate_classes']
psg_thing_cats = psg_dataset_file['thing_classes']
psg_stuff_cats = psg_dataset_file['stuff_classes']
psg_obj_cats = psg_thing_cats + psg_stuff_cats

psg_dataset = {d["image_id"]: d for d in psg_dataset_file['data']}

print(f'Number of images: {len(psg_dataset)}')
print(f'# Object Classes: {len(psg_obj_cats)}')
print(f'# Relation Classes: {len(psg_rel_cats)}')

def generate_by_id(example_img_id: str, save_dir: Path):
    masks = []
    labels_coco = []
    seg_ratio_list= []

    data = psg_dataset[example_img_id]
    img = read_image(coco_path / data["file_name"], format="RGB")
    seg_img = read_image(coco_path / data["pan_seg_file_name"], format="RGB")
    seg_map = utils.rgb2id(seg_img)

    for i, s in enumerate(data["segments_info"]):
        label = psg_obj_cats[s["category_id"]]
        labels_coco.append(label)
        mask = (seg_map == s["id"])
        seg_ratio = utils.seg_ratio(seg_img, mask)
        masks.append(mask)
        seg_ratio_list.append(seg_ratio)

    for s_idx, o_idx, rel_id in data["relations"]:
        s_label = labels_coco[s_idx]
        o_label = labels_coco[o_idx]
        rel_label = psg_rel_cats[rel_id]

        s_img = utils.get_seg_image_1(img, masks, [s_idx, o_idx], ["blue", "red"])
        utils.save_img(s_img, f"[{example_img_id}] {s_label} {rel_label} {o_label}", save_dir)
    
    utils.clean_plt()


if __name__ == "__main__":
    # For testing
    # example_img_id = '2324911'
    # generate_by_id(example_img_id, Path("./data/"))

    # Train
    # for image_id in tqdm(psg_dataset_file["train_image_ids"]):
    #     generate_by_id(image_id, Path("./data/train"))
    
    # Validation
    # for image_id in tqdm(psg_dataset_file["val_image_ids"]):
    #     generate_by_id(image_id, Path("./data/val"))

    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    possible_sequences = utils.generate_all_possible_sequences(psg_obj_cats, psg_rel_cats)

    utils.batch_encode(tokenizer, possible_sequences)
