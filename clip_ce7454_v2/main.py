
import os
import time
import argparse

import wandb
import torch
from trainer import BaseTrainer
from evaluator import Evaluator
from dataset import PSGClsDataset
from torch.utils.data import DataLoader
from torchvision.models import resnet50, resnet101
from transformers import AutoModelForImageClassification
from transformers import ViTModel, ViTConfig
from transformers import CLIPProcessor, CLIPModel


# Arguments Parser
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='openai/clip-vit-base-patch16')
parser.add_argument('--optimiser', type=str, default='AdamW')
parser.add_argument('--epoch', type=int, default=64)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.005)
parser.add_argument('--h_dropout', type=float, default=0.1)
parser.add_argument('--a_dropout', type=float, default=0.1)
parser.add_argument('--num_labels', type=int, default=56)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--note', type=str, default="")
args = parser.parse_args()

# CUDA Configuration
os.environ["CUDA_VISIBLE_DEVICES"]=f"{args.cuda}"

# Wandb Configuation
wandb.init(project="ce7454", entity="elfsong")
wandb.config.update(args)

# Other Configuation
os.environ["TOKENIZERS_PARALLELISM"]=f"false"
savename = f'{args.model_name}_e{args.epoch}_lr{args.lr}_bs{args.batch_size}_m{args.momentum}_wd{args.weight_decay}'
os.makedirs('./checkpoints', exist_ok=True)
os.makedirs('./results', exist_ok=True)

# Datasets
train_dataset = PSGClsDataset(stage='train', input_size=args.input_size)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

val_dataset = PSGClsDataset(stage='val', input_size=args.input_size)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)

test_dataset = PSGClsDataset(stage='test', input_size=args.input_size)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)
print('[+] Data Loaded', flush=True)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model.to(device)
print('[+] Model Loaded', flush=True)

# Trainer
trainer = BaseTrainer(model,
                      processor,
                      train_dataloader,
                      learning_rate=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay,
                      epochs=args.epoch)
print('[+] Trainer Loaded', flush=True)

# Evaluator
evaluator = Evaluator(model, processor, k=3)
print('[+] Evaluator Loaded', flush=True)

# Train!
print('[+] Training ...', flush=True)
begin_epoch = time.time()
best_val_recall = 0.0

for epoch in range(args.epoch):
    train_metrics = trainer.train_epoch()
    val_metrics = evaluator.eval_recall(val_dataloader)

    # Show Log
    print(
        'Epoch {:3d} | Time {:5d}s | Train Loss {:.4f} | Test Loss {:.3f} | mR {:.2f}'
        .format((epoch + 1), 
                int(time.time() - begin_epoch),
                train_metrics['train_loss'], 
                val_metrics['test_loss'],
                100.0 * val_metrics['mean_recall']),
        flush=True)
    
    # WandB Log
    wandb.log({
        "epoch": epoch + 1,
        "time": int(time.time() - begin_epoch),
        "train loss": train_metrics['train_loss'],
        "test loss": val_metrics['test_loss'],
        "mR": 100.0 * val_metrics['mean_recall'],
        "best_mR": 100.0 * best_val_recall,
    })

    # Save Model
    if val_metrics['mean_recall'] >= best_val_recall:
        torch.save(model.state_dict(), f'./checkpoints/{savename}_best_{val_metrics["mean_recall"]}.ckpt')
        best_val_recall = val_metrics['mean_recall']
        print('[+] Model Saved!', flush=True)

    # Evaluation submission generation
    result = evaluator.submit(test_dataloader)
    with open(f'results/{savename}_{val_metrics["mean_recall"]}_epoch_result.txt', 'w') as writer:
        for label_list in result:
            a = [str(x) for x in label_list]
            save_str = ' '.join(a)
            writer.writelines(save_str + '\n')
    print('[+] Epoch Result Saved!', flush=True)

print('[+] Training Completed...', flush=True)

# Evaluation
result = evaluator.submit(test_dataloader)
with open(f'results/{savename}_{best_val_recall}.txt', 'w') as writer:
    for label_list in result:
        a = [str(x) for x in label_list]
        save_str = ' '.join(a)
        writer.writelines(save_str + '\n')
print('[+] Result Saved!', flush=True)