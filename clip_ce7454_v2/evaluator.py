import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Evaluator:
    def __init__(self, net: nn.Module, processor, k: int):
        self.net = net
        self.processor = processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.k = k
        self.text = [
                    "over",
                    "in front of",
                    "beside",
                    "on",
                    "in",
                    "attached to",
                    "hanging from",
                    "on back of",
                    "falling off",
                    "going down",
                    "painted on",
                    "walking on",
                    "running on",
                    "crossing",
                    "standing on",
                    "lying on",
                    "sitting on",
                    "flying over",
                    "jumping over",
                    "jumping from",
                    "wearing",
                    "holding",
                    "carrying",
                    "looking at",
                    "guiding",
                    "kissing",
                    "eating",
                    "drinking",
                    "feeding",
                    "biting",
                    "catching",
                    "picking",
                    "playing with",
                    "chasing",
                    "climbing",
                    "cleaning",
                    "playing",
                    "touching",
                    "pushing",
                    "pulling",
                    "opening",
                    "cooking",
                    "talking to",
                    "throwing",
                    "slicing",
                    "driving",
                    "riding",
                    "parked on",
                    "driving on",
                    "about to hit",
                    "kicking",
                    "swinging",
                    "entering",
                    "exiting",
                    "enclosing",
                    "leaning on"
                ]

    def eval_recall(self, data_loader: DataLoader):
        self.net.eval()
        loss_avg = 0.0
        pred_list, gt_list = [], []
        with torch.no_grad():
            for batch in data_loader:
                data = batch['data'].to(self.device)
                
                inputs = self.processor(text=self.text, return_tensors="pt", padding=True)
                inputs.to(self.device)

                outputs = self.net(input_ids=inputs.input_ids, pixel_values=data)

                prob = torch.sigmoid(outputs.logits_per_image)
                target = batch['soft_label'].to(self.device)
                
                loss_image = F.binary_cross_entropy_with_logits(
                    outputs.logits_per_image, target, reduction='sum')
                loss_text = F.binary_cross_entropy_with_logits(
                    outputs.logits_per_text.T, target, reduction='sum')
                loss = (loss_image + loss_text) / 2
                loss_avg += float(loss.data)

                # gather prediction and gt
                pred = torch.topk(prob.data, self.k)[1]
                pred = pred.cpu().detach().tolist()
                pred_list.extend(pred)
                for soft_label in batch['soft_label']:
                    gt_label = (soft_label == 1).nonzero(as_tuple=True)[0]\
                                .cpu().detach().tolist()
                    gt_list.append(gt_label)

        # compute mean recall
        score_list = np.zeros([56, 2], dtype=int)
        for gt, pred in zip(gt_list, pred_list):
            for gt_id in gt:
                # pos 0 for counting all existing relations
                score_list[gt_id][0] += 1
                if gt_id in pred:
                    # pos 1 for counting relations that is recalled
                    score_list[gt_id][1] += 1
        score_list = score_list[6:]

        # to avoid nan
        score_list[:, 0][score_list[:, 0] == 0] = 1
        meanrecall = np.mean(score_list[:, 1] / score_list[:, 0])

        metrics = {}
        metrics['test_loss'] = loss_avg / len(data_loader)
        metrics['mean_recall'] = meanrecall

        return metrics

    def submit(self, data_loader: DataLoader):
        self.net.eval()

        pred_list = []
        with torch.no_grad():
            for batch in data_loader:
                data = batch['data'].cuda()

                inputs = self.processor(text=self.text, return_tensors="pt", padding=True)
                inputs.to(self.device)
                outputs = self.net(input_ids=inputs.input_ids, pixel_values=data)

                prob = torch.sigmoid(outputs.logits_per_image)
                pred = torch.topk(prob.data, self.k)[1]
                pred = pred.cpu().detach().tolist()
                pred_list.extend(pred)
        return pred_list
