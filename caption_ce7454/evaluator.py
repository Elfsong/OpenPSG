
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

import utils

class Evaluator:
    def __init__(self, model: nn.Module, tokenizer: AutoTokenizer, k: int):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.k = k

    def eval_recall(self, data_loader: DataLoader):
        self.model.eval()
        loss_avg = 0.0
        pred_list, gt_list = [], []
        with torch.no_grad():
            for batch in data_loader:
                source = batch['source'].to(self.device)       
                target = batch['target'].to(self.device)

                output_ids = self.model.generate(source, max_length=16, num_beams=4)

                preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                preds = [pred.strip() for pred in preds]
                
                pred_list += preds
                gt_list += batch['text']

        utils.generate_report(pred_list, gt_list)

        return {}

    def submit(self, data_loader: DataLoader):
        self.model.eval()

        pred_list = []
        with torch.no_grad():
            for batch in data_loader:
                data = batch['data'].cuda()
                target = batch['soft_label'].to(self.device)
                
                outputs = self.model.forward(pixel_values=data, labels=target)
                prob = torch.sigmoid(outputs["logits"])
                pred = torch.topk(prob.data, self.k)[1]
                pred = pred.cpu().detach().tolist()
                pred_list.extend(pred)
        return pred_list
