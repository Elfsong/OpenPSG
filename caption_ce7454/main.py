import os
import time
import argparse

import wandb
import torch
from PIL import Image
from trainer import BaseTrainer
from evaluator import Evaluator
from dataset import PSGClsDataset
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer


# Arguments Parser
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='google/vit-large-patch16-384')
parser.add_argument('--optimiser', type=str, default='AdamW')
parser.add_argument('--epoch', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.05)
parser.add_argument('--warmup_steps', type=int, default=5000)
parser.add_argument('--h_dropout', type=float, default=0.2)
parser.add_argument('--a_dropout', type=float, default=0.2)
parser.add_argument('--num_labels', type=int, default=56)
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--input_size', type=int, default=384)
parser.add_argument('--save_checkpoint', type=bool, default=False)
parser.add_argument('--note', type=str, default="")
args = parser.parse_args()

# System Configuration
os.environ["CUDA_VISIBLE_DEVICES"]=f"{args.cuda}"
os.environ["TOKENIZERS_PARALLELISM"]=f"false"
savename = f'{args.model_name}_e{args.epoch}_lr{args.lr}_bs{args.batch_size}_m{args.momentum}_wd{args.weight_decay}'
os.makedirs('./checkpoints', exist_ok=True)
os.makedirs('./results', exist_ok=True)
print('[+] System Loaded', flush=True)

# Wandb Configuation
wandb.init(project="ce7454", entity="elfsong")
wandb.config.update(args)
print('[+] WandB Loaded', flush=True)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
model.to(device)
print('[+] Model Loaded', flush=True)

# Dataset
train_dataset = PSGClsDataset(stage='train', tokenizer=tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
val_dataset = PSGClsDataset(stage='val', tokenizer=tokenizer)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=1)
print('[+] Data Loaded', flush=True)

# Evaluator
evaluator = Evaluator(model, tokenizer, k=3)
print('[+] Evaluator Loaded', flush=True)

# Trainer
trainer = BaseTrainer(model,
                      train_dataloader,
                      learning_rate=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay,
                      epochs=args.epoch,
                      optimiser_name=args.optimiser,
                      warmup_steps=args.warmup_steps)
print('[+] Trainer Loaded', flush=True)

######################################################################

# Train!
print('[+] Training ...', flush=True)
begin_epoch = time.time()
best_val_recall = 0.0

for epoch in range(args.epoch):
    train_metrics = trainer.train_epoch()
    val_metrics = evaluator.eval_recall(val_dataloader)
    val_metrics = {"test_loss": 0, "mean_recall": 0}

    print(f"Epoch: {epoch + 1} | Time: {int(time.time() - begin_epoch)} | Train Loss: {train_metrics['train_loss']} | Test Loss: {val_metrics['test_loss']} | mR: {val_metrics['mean_recall']}", flush=True)
    
    # WandB Log
    wandb.log({
        "epoch": epoch + 1,
        "time": int(time.time() - begin_epoch),
        "train loss": train_metrics['train_loss'],
        "test loss": val_metrics['test_loss'],
        "mR": 100.0 * val_metrics['mean_recall'],
        "best_mR": 100.0 * best_val_recall,
    })

print('[+] Training Completed...', flush=True)

######################################################################