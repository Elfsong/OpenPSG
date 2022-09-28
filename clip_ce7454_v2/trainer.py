import torch
import string
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


class BaseTrainer:
    def __init__(self,
                 model: nn.Module,
                 processor: CLIPProcessor,
                 train_loader: DataLoader,
                 learning_rate: float = 0.001,
                 momentum: float = 0.9,
                 weight_decay: float = 0.0005,
                 epochs: int = 100,
                 optimizer: string = "SGD") -> None:
        self.model = model
        self.processor = processor
        self.train_loader = train_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                model.parameters(),
                learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=True,
            )
        elif optimizer == "AdamW":
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise("Unknown optimizer!")

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                epochs * len(train_loader),
                1,  # since lr_lambda computes multiplicative factor
                1e-6 / learning_rate,
            ),
        )

    def train_epoch(self):
        self.model.train()  # enter train mode

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(range(1, len(train_dataiter) + 1)):
            batch = next(train_dataiter)

            data = batch['data'].to(self.device)
            text = [
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
            
            target = batch['soft_label'].to(self.device)

            inputs = self.processor(text=text, return_tensors="pt", padding=True)
            inputs.to(self.device)

            # forward
            outputs = self.model(input_ids=inputs.input_ids, pixel_values=data)

            loss_image = F.binary_cross_entropy_with_logits(outputs.logits_per_image, target, reduction='sum')
            # loss_text = F.binary_cross_entropy_with_logits(
            #     outputs.logits_per_text.T, target, reduction='sum')
            # loss = (loss_image + loss_text) / 
            loss = loss_image
            loss_avg += float(loss.data)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        metrics = {}
        metrics['train_loss'] = loss_avg

        return metrics
