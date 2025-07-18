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
import torch, math, random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
import random
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import psutil
import os
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResidualDenoiseBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.shortcut = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.shortcut(x)

class NoPropDT(nn.Module):
    def __init__(self, embed_dim, num_classes, num_blocks=10, eta=1.0, use_noprop=False):
        super().__init__()
        self.use_noprop = use_noprop
        self.num_blocks = num_blocks
        self.eta = eta
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Label embedding
        self.embed = nn.Embedding(num_classes, embed_dim)
        self.WEmbed = nn.Parameter(F.normalize(torch.randn(num_classes, embed_dim), dim=1))

        # Image encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 96 -> 48
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),                  # 48 -> 48
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),# 48 -> 24
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool2d((8, 8))  # Final size: [B, 128, 8, 8]
        )

        self.fc_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),  # Assuming input size is 32x32
            nn.BatchNorm1d(256)
        )
        
        self.label_proj = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Add projection back to embed_dim for loss calculation
        self.proj_back = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )
        
        self.u_blocks = nn.ModuleList([
            ResidualDenoiseBlock(256, hidden_dim=256) for _ in range(num_blocks)
        ])

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def extract_features(self, x):
        h = self.encoder(x)                     # [B, 128, 8, 8]
        h = h.view(h.size(0), -1)               # Flatten
        return self.fc_proj(h)                  # [B, 256]

    def get_alpha_bar(self, t):
        s = 0.008
        frac = (t / self.num_blocks + s) / (1 + s)
        return math.cos(frac * math.pi / 2) ** 2

    def snr(self, alpha_bar):
        return alpha_bar / (1 - alpha_bar + 1e-8)

    def forward(self, x, y, t):
        h = self.extract_features(x)
        return self.forward_features(h, y, t)

    def forward_features(self, h, y, t):
        B = y.size(0)
        device = y.device

        u_y = self.embed(y)  # [B, embed_dim]
        noise = torch.randn_like(u_y)

        alpha_bar_t = self.get_alpha_bar(t)
        alpha_bar_prev = self.get_alpha_bar(t - 1) if t > 0 else 1.0

        sqrt_alpha_bar_t = math.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = math.sqrt(1 - alpha_bar_t)
        z_t = sqrt_alpha_bar_t * u_y + sqrt_one_minus_alpha_bar_t * noise

        label_feat = self.label_proj(z_t)  # [B, 256]

        # -- DIFFERENCE HAPPENS HERE --
        if self.use_noprop:
            with torch.no_grad():
                for i in range(t - 1):
                    _ = self.u_blocks[i](label_feat)
            u_t_hat = self.u_blocks[t - 1](label_feat)  # [B, 256]
        else:
            for i in range(t):
                label_feat = self.u_blocks[i](label_feat)
            u_t_hat = label_feat  # [B, 256]

        # Project back to embed_dim for loss calculation
        u_t_hat_proj = self.proj_back(u_t_hat)  # [B, embed_dim]

        snr_t = self.snr(alpha_bar_t)
        snr_t_prev = self.snr(alpha_bar_prev)
        snr_diff = snr_t - snr_t_prev

        # Now both tensors have the same dimension
        denoise_loss = (self.num_blocks / 2) * self.eta * abs(snr_diff) * F.mse_loss(u_t_hat_proj, u_y)

        # Always use z_T for classification
        alpha_bar_T = self.get_alpha_bar(self.num_blocks)
        sqrt_alpha_bar_T = math.sqrt(alpha_bar_T)
        sqrt_one_minus_alpha_bar_T = math.sqrt(1 - alpha_bar_T)
        noise_T = torch.randn_like(u_y)
        z_T = sqrt_alpha_bar_T * u_y + sqrt_one_minus_alpha_bar_T * noise_T

        label_feat_T = self.label_proj(z_T)  # [B, 256]
        u_T_hat = self.u_blocks[-1](label_feat_T)  # [B, 256]
        logits = self.classifier(torch.cat([h, u_T_hat], dim=1))
        ce_loss = F.cross_entropy(logits, y)

        z0 = u_y + torch.randn_like(u_y)
        kl_loss = 0.5 * torch.sum(z0 ** 2, dim=1).mean()

        total_loss = ce_loss + kl_loss + denoise_loss
        return total_loss, ce_loss, kl_loss, denoise_loss

    def inference_diffusion(self, x):
        self.eval()
        B = x.size(0)
        h = self.extract_features(x)
        z_t = torch.randn(B, self.embed_dim).to(x.device)

        with torch.no_grad():
            for t in reversed(range(1, self.num_blocks + 1)):
                label_feat = self.label_proj(z_t)  # [B, 256]
                z_hat_256 = self.u_blocks[t - 1](label_feat)  # [B, 256]
                z_hat = self.proj_back(z_hat_256)  # [B, embed_dim]

                alpha_bar_prev = self.get_alpha_bar(t - 1)
                noise = torch.randn_like(z_t)
                z_t = math.sqrt(alpha_bar_prev) * z_hat + math.sqrt(1 - alpha_bar_prev) * noise

            # Final classification - use 256-dim version for classifier
            z_t_256 = self.label_proj(z_t)  # [B, 256]
            final_combined = torch.cat([h, z_t_256], dim=1)
            logits = self.classifier(final_combined)
            return logits


# Training functions
import torch
import psutil
import os
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    peak_memory = 0
    
    # Reset peak memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    
    loop = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, (images, labels) in enumerate(loop):
        images, labels = images.to(device), labels.to(device)
        
        # Sample t uniformly from 1 to num_blocks
        t = torch.randint(1, model.num_blocks + 1, (1,)).item()
        loss, ce, kl, denoise = model(images, labels, t)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Monitor memory usage
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated(device) / 1024**3  # GB
            peak_memory = max(peak_memory, current_memory)
        
        loop.set_postfix(
            loss=loss.item(), 
            ce=ce.item(), 
            kl=kl.item(), 
            denoise=denoise.item(),
            mem_gb=f"{current_memory:.2f}" if torch.cuda.is_available() else "N/A"
        )
    
    # Get final memory stats
    if torch.cuda.is_available():
        peak_memory_pytorch = torch.cuda.max_memory_allocated(device) / 1024**3
        peak_memory_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
        print(f"Peak Memory Allocated: {peak_memory_pytorch:.2f} GB")
        print(f"Peak Memory Reserved: {peak_memory_reserved:.2f} GB")
    
    return total_loss / len(dataloader), peak_memory_pytorch if torch.cuda.is_available() else 0

def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model.inference_diffusion(images)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# Example usage (you'll need to define your data loaders)
# model = NoPropDT(embed_dim=512, num_classes=10, num_blocks=1, eta=1.0).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)


# === Data ===
transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_loader = DataLoader(
    datasets.CIFAR10(root='./data', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True)

test_loader = DataLoader(
    datasets.CIFAR10(root='./data', train=False, download=True, transform=transform),
    batch_size=64, shuffle=False)

transform = transforms.Compose([
    transforms.Resize((96, 96)),  # Resize to match your model
    transforms.ToTensor()
])

# Load dataset
train_path = 'imagenette2-160/train'
val_path = 'imagenette2-160/val'

train_data = datasets.ImageFolder(train_path, transform=transform)
val_data = datasets.ImageFolder(val_path, transform=transform)

# DataLoader
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=2)

num_classes = len(train_data.classes)
print("Classes:", train_data.classes)
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NoPropDT(embed_dim=512, num_classes=num_classes, num_blocks=10, eta=1.0,use_noprop=False).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

epochs = 100
for epoch in range(epochs):
    train_loss, peak_memory = train_epoch(model, train_loader, optimizer, device)
    val_acc = evaluate(model, val_loader)
    
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Acc = {val_acc*100:.2f}%, Peak Memory = {peak_memory:.2f} GB")
