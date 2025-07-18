import sys
import os
from datetime import datetime

# Get script name
script_name = os.path.basename(__file__) 
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, f"{script_name}_{timestamp}.log")

class Logger(object):
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_file)

import urllib.request
import tarfile
import os
import requests
from tqdm import tqdm
import tarfile
import os
import numpy as np
url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
output_file = "imagenette2-160.tgz"
extract_folder = "imagenette2-160"

def download_with_progress(url, output_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB

    with open(output_path, 'wb') as file, tqdm(
        desc="Downloading",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))

# Step 1: Download
if not os.path.exists(output_file):
    print("Downloading dataset...")
    download_with_progress(url, output_file)
else:
    print("Dataset already downloaded.")

# Step 2: Extract
if not os.path.exists(extract_folder):
    print("Extracting dataset...")
    with tarfile.open(output_file) as tar:
        tar.extractall()
    print("Extraction complete.")
else:
    print("Dataset already extracted.")

from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class CIFAR10CDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])  # Convert NumPy -> PIL
        img = self.transform(img)
        label = self.labels[idx]
        return img, label

def get_cifar10c_loader(corruption='gaussian_noise', severity=1, batch_size=128, num_workers=2):
    path = f'cifar10_c/{corruption}.npy'
    labels_path = f'cifar10_c/labels.npy'

    data = np.load(path)
    labels = np.load(labels_path)

    # Extract severity level
    start = (severity - 1) * 10000
    end = start + 10000
    data = data[start:end]  # shape (10000, 32, 32, 3)
    labels = labels[start:end]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Needed for ViT
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    dataset = CIFAR10CDataset(data, labels, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import psutil
import os
import random
import timm
from collections import defaultdict
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import ImageFolder


# Setup: track total training start time and GPU memory usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def track_resources():
    import time
    import torch

    start_time = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()
    else:
        start_mem = 0

    return start_time, start_mem


def report_resources(start_time, start_mem):
    torch.cuda.synchronize()
    end_time = time.time()
    end_mem = torch.cuda.memory_allocated() / 1024**2
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    print(f"Time Taken: {end_time - start_time:.2f}s")
    print(f"Memory Used: {end_mem - start_mem:.2f} MB")
    print(f"Peak GPU Memory: {peak_mem:.2f} MB")


from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm

class SVPLayer(nn.Module):
    def __init__(self, in_dim, proj_dim, dropout_p=0.1):
        super().__init__()
        self.proj_dim = proj_dim
        self.dropout = nn.Dropout(p=dropout_p)
        self.register_buffer("R", torch.randn(proj_dim, in_dim) / math.sqrt(proj_dim))

    def forward(self, h):
        h_dropped = self.dropout(h)
        v = F.linear(h_dropped, self.R)
        return v

def cosine_similarity_matrix(v):
    v = F.normalize(v, dim=1)  # Normalize across feature dim
    return v @ v.T

def feature_alignment_loss(v1, v2):
    S1 = cosine_similarity_matrix(v1)
    S2 = cosine_similarity_matrix(v2.detach())  # stop-gradient on v2
    return F.mse_loss(S1, S2)

class SVPBlock(nn.Module):
    def __init__(self, blocks, d_model, num_classes):
        super().__init__()
        self.blocks = blocks
        self.svp = SVPLayer(in_dim=d_model, proj_dim=num_classes, dropout_p=0.1)

    def forward(self, x, prev_v=None, target=None):
        x = self.blocks(x)
        v = self.svp(x[:, 0])  # Fixed random projection of CLS token
  # Use CLS token
        pred_loss = F.cross_entropy(v, target) if target is not None else 0.0
        fa_loss = 0.0
        if prev_v is not None:
            S = lambda z: F.normalize(z, dim=1)
            fa_loss = F.mse_loss(S(v), S(prev_v.detach()))
        return x, v, pred_loss, fa_loss
    def forward_inference(self, x):
        x = self.blocks(x)
        return x


class SVP7Plus(nn.Module):
    def __init__(self, num_chunks=7, model_name='vit_tiny_patch16_224', embed_dim=192, num_classes=10, pretrained_path=None):
        super().__init__()
         # Load base model
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading model from {pretrained_path}")
            vit = timm.create_model(model_name, pretrained=False)

            original_weight = vit.patch_embed.proj.weight.clone()
            state_dict = torch.load(pretrained_path, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            missing_keys, unexpected_keys = vit.load_state_dict(new_state_dict, strict=False)

            if torch.equal(original_weight, vit.patch_embed.proj.weight):
                print(" Pretrained weights not loaded properly, falling back to timm pretrained")
                vit = timm.create_model(model_name, pretrained=True)
            else:
                print(" Pretrained weights loaded successfully")
        else:
            print("Using timm pretrained weights")
            vit= timm.create_model(model_name, pretrained=True)
        vit.head = nn.Identity()
        self.patch_embed = vit.patch_embed
        self.cls_token = vit.cls_token
        self.pos_embed = vit.pos_embed
        self.pos_drop = vit.pos_drop
        self.norm = vit.norm

        all_blocks = vit.blocks
          # add this import at the top

        chunks = np.array_split(all_blocks, num_chunks)
        self.chunked_blocks = nn.ModuleList([
            SVPBlock(nn.Sequential(*chunk), embed_dim, num_classes)
            for chunk in chunks if len(chunk) > 0  # safety check
        ])

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_chunks = num_chunks

    def forward(self, x, target=None):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        total_loss, ce_loss, fa_loss = 0.0, 0.0, 0.0
        prev_v = None
        for block in self.chunked_blocks:
            x, v, pred_loss, align_loss = block(x, prev_v, target)
            total_loss += pred_loss + align_loss
            ce_loss += pred_loss
            fa_loss += align_loss
            prev_v = v

        return total_loss, ce_loss, fa_loss
    def inference(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.chunked_blocks:
            x = block.forward_inference(x)

        cls_token_final = x[:, 0]  # Extract final [CLS] token
        logits = self.chunked_blocks[-1].svp(cls_token_final)

        return logits

def evaluate(model, dataloader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model.inference(inputs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0
    

def compute_confidence_and_entropy(model, dataloader, device):
    model.eval()
    all_confidences = []
    all_entropies = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            logits = model.inference(inputs) 
            probs = F.softmax(logits, dim=1)

            # Confidence: max probability
            max_conf, _ = probs.max(dim=1)
            all_confidences.extend(max_conf.cpu().numpy())

            # Entropy: -∑ p log p
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)
            all_entropies.extend(entropy.cpu().numpy())

    return np.array(all_confidences), np.array(all_entropies)

    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SVP7Plus(
    num_chunks=7,         # SVP+7 means 7 parts from vit model
    embed_dim=192,        # ViT Tiny default embedding dim
    num_classes=10
).to(device)

# Define optimizer

optimizers = [
    torch.optim.AdamW(block.parameters(), lr=1e-3)
    for block in model.chunked_blocks
]

from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms



transform_train = transforms.Compose([
    transforms.Resize((224, 224)), 
    
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
transform_test = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

imagenette_transform = transforms.Compose([
   
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform_train),
    batch_size=64, shuffle=True, num_workers=2
)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform_test),
    batch_size=64, shuffle=False, num_workers=2
)

cifar10c_loader = get_cifar10c_loader(corruption='gaussian_noise', severity=3) 
imagenette_val_set = ImageFolder(root="imagenette2-160/val", transform=imagenette_transform)
imagenette_val_loader = DataLoader(imagenette_val_set, batch_size=64, shuffle=False, num_workers=2)
num_epochs = 100


print("After model and dataloader init:")
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")



for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    start_time, start_mem = track_resources()

    model.train()
    train_loss = 0.0
    total_samples = 0

    loop = tqdm(train_loader, leave=True)
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        

                # Patch embedding + CLS token
        B = inputs.size(0)
        x = model.patch_embed(inputs)
        cls_tokens = model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + model.pos_embed
        x = model.pos_drop(x)

        total_loss = 0.0
        ce_loss = 0.0
        fa_loss = 0.0
        prev_v = None

        for i, block in enumerate(model.chunked_blocks):
            # Detach input to ensure no backward through earlier blocks
            x = x.detach()
            x.requires_grad = True  # Required for backward on this block

            # Forward pass with local loss
            x, v, pred_loss, align_loss = block(x, prev_v, labels)
            loss = pred_loss + align_loss

            # Local backward + optimizer step
            optimizers[i].zero_grad()
            loss.backward()
            optimizers[i].step()

            # Accumulate stats
            total_loss += loss.item()
            ce_loss += pred_loss.item()
            fa_loss += align_loss
            prev_v = v


        batch_size = labels.size(0)
        train_loss += total_loss * batch_size
        total_samples += batch_size

        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=total_loss, CE=ce_loss, FA=fa_loss)

    train_loss /= total_samples

    # === Evaluation ===
    test_acc = evaluate(model, test_loader, device)
    c10c_acc = evaluate(model, cifar10c_loader, device)
    imagenette_acc = evaluate(model, imagenette_val_loader, device)

    confidences_i, entropies_i = compute_confidence_and_entropy(model, imagenette_val_loader, device)
    confidences_c, entropies_c = compute_confidence_and_entropy(model, cifar10c_loader, device)
    confidences, entropies = compute_confidence_and_entropy(model, test_loader, device)

    # === Logging ===
    print("Robustness Evaluation on ImageNet Easy-10:")
    print(f"→ Mean Confidence: {confidences_i.mean():.4f}")
    print(f"→ Max Confidence: {confidences_i.max():.4f}")
    print(f"→ Mean Entropy: {entropies_i.mean():.4f}")
    print(f"→ Max Entropy: {entropies_i.max():.4f}")

    print("Robustness Evaluation on CIFAR-10-C:")
    print(f"→ Mean Confidence: {confidences_c.mean():.4f}")
    print(f"→ Max Confidence: {confidences_c.max():.4f}")
    print(f"→ Mean Entropy: {entropies_c.mean():.4f}")
    print(f"→ Max Entropy: {entropies_c.max():.4f}")

    print("Robustness Evaluation on CIFAR-10:")
    print(f"→ Mean Confidence: {confidences.mean():.4f}")
    print(f"→ Max Confidence: {confidences.max():.4f}")
    print(f"→ Mean Entropy: {entropies.mean():.4f}")
    print(f"→ Max Entropy: {entropies.max():.4f}")

    print(f"Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.4f} | CIFAR-10-C Acc: {c10c_acc:.4f} | Imagenette Acc: {imagenette_acc:.4f}")

    report_resources(start_time, start_mem)
