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
        start_mem = torch.cuda.memory_allocated()/1024**2 
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
import time
import psutil
import os
import numpy as np
import random
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import ImageFolder

# Setup: track total training start time and GPU memory usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
    def __init__(self, embed_dim, num_classes, num_blocks=10, eta=1.0, use_noprop=True):
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
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout(0.2)
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
        h = self.encoder(x)                     # [B, 512, 4, 4]
        #h = F.adaptive_avg_pool2d(h, (1, 1))    # [B, 512, 1, 1]
        h = h.view(h.size(0), -1)               # Flatten
        return self.fc_proj(h)                  # [B, 512]

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

        u_y = self.embed(y)
        noise = torch.randn_like(u_y)

        alpha_bar_t = self.get_alpha_bar(t)
        alpha_bar_prev = self.get_alpha_bar(t - 1) if t > 0 else 1.0

        sqrt_alpha_bar_t = math.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = math.sqrt(1 - alpha_bar_t)
        z_t = sqrt_alpha_bar_t * u_y + sqrt_one_minus_alpha_bar_t * noise

        label_feat = self.label_proj(z_t)

        # -- DIFFERENCE HAPPENS HERE --
        if self.use_noprop:
            with torch.no_grad():
                for i in range(t - 1):
                    _ = self.u_blocks[i](label_feat)
            u_t_hat = self.u_blocks[t - 1](label_feat)  # only this gets grads
        else:
            for i in range(t):
                label_feat = self.u_blocks[i](label_feat)
            u_t_hat = label_feat

        snr_t = self.snr(alpha_bar_t)
        snr_t_prev = self.snr(alpha_bar_prev)
        snr_diff = snr_t - snr_t_prev

        at = math.sqrt(alpha_bar_t * (1 - alpha_bar_prev)) / (1 - alpha_bar_t + 1e-8)
        bt = math.sqrt(alpha_bar_prev * (1 - alpha_bar_t)) / (1 - alpha_bar_t + 1e-8)
        denoise_loss = (self.num_blocks / 2) * self.eta * abs(snr_diff) * F.mse_loss(u_t_hat, u_y)

        # Always use z_T for classification
        alpha_bar_T = self.get_alpha_bar(self.num_blocks)
        sqrt_alpha_bar_T = math.sqrt(alpha_bar_T)
        sqrt_one_minus_alpha_bar_T = math.sqrt(1 - alpha_bar_T)
        noise_T = torch.randn_like(u_y)
        z_T = sqrt_alpha_bar_T * u_y + sqrt_one_minus_alpha_bar_T * noise_T

        label_feat_T = self.label_proj(z_T)
        u_T_hat = self.u_blocks[-1](label_feat_T)
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
                
                label_feat = self.label_proj(z_t)
                z_hat = self.u_blocks[t - 1](label_feat)


                alpha_bar_prev = self.get_alpha_bar(t - 1)
                noise = torch.randn_like(z_t)
                z_t = math.sqrt(alpha_bar_prev) * z_hat + math.sqrt(1 - alpha_bar_prev) * noise

            # Final classification
            final_combined = torch.cat([h, z_t], dim=1)
            logits = self.classifier(final_combined)
            return logits


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Use model.inference_diffusion to get logits
            logits = model.inference_diffusion(inputs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0

import matplotlib.pyplot as plt
import numpy as np

def compute_confidence_and_entropy(model, dataloader, device):
    model.eval()
    all_confidences = []
    all_entropies = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            logits = model.inference_diffusion(inputs)
            probs = F.softmax(logits, dim=1)

            # Max confidence per sample
            max_conf, _ = probs.max(dim=1)
            all_confidences.extend(max_conf.cpu().numpy())

            # Entropy per sample
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)
            all_entropies.extend(entropy.cpu().numpy())

    return np.array(all_confidences), np.array(all_entropies)
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NoPropDT(
    embed_dim=256,         # or your desired embedding dimension
    num_classes=10,        # or your number of classes
    num_blocks=10,          # or your desired number of diffusion steps
    eta=0.1 ,
    use_noprop=False
).to(device)



optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=1e-3
)

from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
imagenette_transform = transforms.Compose([
    transforms.Resize((32,32)),  
   
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
    print(f"Epoch {epoch+1}/{num_epochs}")
        

    start_time, start_mem = track_resources()
    

    model.train()
    train_loss, train_acc = 0, 0
    total_samples = 0

    loop = tqdm(train_loader, leave=False)
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        t = random.randint(1, model.num_blocks)
        # Model returns total_loss, ce_loss, kl_loss, denoise_loss
        total_loss, ce_loss, kl_loss, denoise_loss = model(inputs, labels,t)

        total_loss.backward()
        optimizer.step()

    
        batch_size = labels.size(0)
        train_loss += total_loss.item() * batch_size
        # To get accuracy, you can compute predictions outside or modify model to return logits
        # Here, just omit train_acc or compute separately.

        total_samples += batch_size

        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=total_loss.item())

    train_loss /= total_samples
    # train_acc can be calculated by evaluating after epoch if you don't have logits now.

    test_acc = evaluate(model, test_loader, device)
    c10c_acc = evaluate(model, cifar10c_loader,device)
    confidences_i, entropies_i = compute_confidence_and_entropy(model,  imagenette_val_loader, device)
    confidences_c, entropies_c = compute_confidence_and_entropy(model,  cifar10c_loader, device)
    confidences, entropies = compute_confidence_and_entropy(model,  test_loader, device)
    
    imagenette_acc = evaluate(model, imagenette_val_loader, device)
    print("Robustness Evaluation on ImageNet Easy-10:")
    print(f"→ Mean Confidence: {confidences_i.mean():.4f}")
    print(f"→ Max Confidence: {confidences_i.max():.4f}")
    print(f"→ Mean Entropy: {entropies_i.mean():.4f}")
    print(f"→ Max Entropy: {entropies_i.max():.4f}")
    print("Robustness Evaluation on cifar_c:")
    print(f"→ Mean Confidence: {confidences_c.mean():.4f}")
    print(f"→ Max Confidence: {confidences_c.max():.4f}")
    print(f"→ Mean Entropy: {entropies_c.mean():.4f}")
    print(f"→ Max Entropy: {entropies_c.max():.4f}")
    print("Robustness Evaluation on cifar:")
    print(f"→ Mean Confidence: {confidences.mean():.4f}")
    print(f"→ Max Confidence: {confidences.max():.4f}")
    print(f"→ Mean Entropy: {entropies.mean():.4f}")
    print(f"→ Max Entropy: {entropies.max():.4f}")

    print(f"Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.4f} | CIFAR-10-C Acc: {c10c_acc:.4f} | Imagenette Acc: {imagenette_acc:.4f}")
    report_resources(start_time, start_mem)
