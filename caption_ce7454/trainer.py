from cProfile import label
import torch
import string
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def get_cosine_annealing_schedule(optimiser, total_steps, lr_max, lr_min):
    def lr_lambda(current_step: int):
        return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(current_step / total_steps * np.pi))
    return torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

def get_linear_schedule_with_warmup(optimiser, warmup_steps, training_steps):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(training_steps - current_step) / float(max(1, training_steps - warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)


class BaseTrainer:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 learning_rate: float = 0.001,
                 momentum: float = 0.9,
                 weight_decay: float = 0.0005,
                 epochs: int = 100,
                 optimiser_name: string = "SGD",
                 warmup_steps: int = 5000,
                 ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.optimiser_name = optimiser_name
        self.warmup_steps = warmup_steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.optimiser_name == "SGD":
            self.optimiser = torch.optim.SGD(
                model.parameters(),
                learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=True,
            )
        elif self.optimiser_name == "AdamW":
            self.optimiser = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise("Unknown Optimiser!")

        self.scheduler = get_linear_schedule_with_warmup(self.optimiser, warmup_steps=self.warmup_steps, training_steps=self.epochs * len(self.train_loader))

    def train_epoch(self):
        self.model.train()  # enter train mode

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(range(1, len(train_dataiter) + 1)):
            batch = next(train_dataiter)

            source = batch['source'].to(self.device)       
            target = batch['target'].to(self.device)

            # forward
            outputs = self.model(pixel_values=source, labels=target)

            loss = outputs.loss
            loss_avg += float(loss.data)

            # backward
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            self.scheduler.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        metrics = {}
        metrics['train_loss'] = loss_avg

        return metrics
