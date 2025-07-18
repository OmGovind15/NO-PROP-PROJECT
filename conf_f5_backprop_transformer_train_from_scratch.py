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

class ViTLocal(nn.Module):
    def __init__(self, model_name='vit_tiny_patch16_224', proj_dim=10):
        super().__init__()
        base = timm.create_model(model_name, pretrained=False)
        self.patch_embed = base.patch_embed
        self.cls_token = base.cls_token
        self.pos_embed = base.pos_embed
        self.pos_drop = base.pos_drop
        self.norm = base.norm
        
        # Group 12 blocks into 7 units
        block_groups = [
            base.blocks[0:1],    # unit 0
            base.blocks[1:3],    # unit 1
            base.blocks[3:5],    # unit 2
            base.blocks[5:7],    # unit 3
            base.blocks[7:8],    # unit 4
            base.blocks[8:10],   # unit 5
            base.blocks[10:12],  # unit 6
        ]

        # Wrap each group into nn.Sequential
        self.blocks = nn.ModuleList([nn.Sequential(*grp) for grp in block_groups])

        self.proj_dim = proj_dim
        # Make random projections learnable parameters
        self.random_projs = nn.ParameterList([
            nn.Parameter(torch.randn(proj_dim, base.embed_dim), requires_grad=True)
            for _ in self.blocks  # now length 7
        ])

        self.dropout = nn.Dropout(0.1)
        self.pred_head = nn.Linear(proj_dim, proj_dim)

        
    def forward(self, x, y, target_block_idx=None):
        B = x.size(0)
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        pred_loss = torch.tensor(0.0, device=x.device)
        align_loss = torch.tensor(0.0, device=x.device)
        v_prev = None

        for i, block in enumerate(self.blocks):
            # Always compute with gradients - no torch.no_grad()
            x = block(x)
            
            if i == target_block_idx:
                x_norm = F.layer_norm(x, x.shape[-1:])
                v = self.dropout(F.linear(x_norm, self.random_projs[i]))
                logits = self.pred_head(v[:, 0])
                pred_loss = F.cross_entropy(logits, y)

                if v_prev is not None:
                    S = lambda z: F.normalize(z, dim=-1) @ F.normalize(z, dim=-1).transpose(1, 2)
                    align_loss = F.mse_loss(S(v), S(v_prev))
                else:
                    align_loss = torch.tensor(0.0, device=x.device)

            # Always compute v_prev with gradients
            x_norm = F.layer_norm(x, x.shape[-1:])
            v_prev = self.dropout(F.linear(x_norm, self.random_projs[i]))

        # Final feature for downstream
        x = self.norm(x)
        return x[:, 0], pred_loss, align_loss


# === Residual Block ===
class ResidualDenoiseBlock(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim)
        )

    def forward(self, x):
        return self.net(x) + x


# === Main Model ===
class NoPropDT(nn.Module):
    def __init__(self, embed_dim=192, num_classes=10, num_blocks=10, eta=1.0):
        super().__init__()
        self.embed_dim, self.num_blocks, self.eta = embed_dim, num_blocks, eta
        self.embed = nn.Embedding(num_classes, embed_dim)

        self.vit_local = ViTLocal(model_name="vit_tiny_patch16_224", proj_dim=num_classes)
        
        self.fc_proj = nn.Sequential(nn.Flatten(), nn.Linear(192, 256), nn.BatchNorm1d(256))
        self.label_proj = nn.Sequential(nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU())
        self.z_proj = nn.Linear(embed_dim, 256)
        self.proj_back = nn.Linear(256, embed_dim)

        self.u_blocks = nn.ModuleList([ResidualDenoiseBlock(256) for _ in range(num_blocks)])
        self.classifier = nn.Sequential(
            nn.Linear(448, 256), nn.ReLU(), nn.Linear(256, num_classes)
        )

    def get_alpha_bar(self, t):
        s = 0.008
        frac = (t / self.num_blocks + s) / (1 + s)
        return math.cos(frac * math.pi / 2) ** 2

    def forward(self, x, y, t_vit, t_denoise):
        h, pred_loss, fa_loss = self.vit_local(x, y, target_block_idx=t_vit)
        u_y = self.embed(y)
        noise = torch.randn_like(u_y)
        alpha_bar = self.get_alpha_bar(t_denoise)
        z_t = (alpha_bar**0.5) * u_y + ((1 - alpha_bar)**0.5) * noise
        label_feat = self.label_proj(self.z_proj(z_t))
        u_t_hat = self.u_blocks[t_denoise - 1](label_feat)
        u_y_proj = self.label_proj(self.z_proj(u_y))
        denoise_loss = F.mse_loss(u_t_hat, u_y_proj)

        # Forward diffusion process - now with full gradients
        z_T = torch.randn_like(u_y)
        for step in reversed(range(1, self.num_blocks + 1)):
            feat = self.label_proj(self.z_proj(z_T))
            z_hat = self.u_blocks[step - 1](feat)
            z_hat = self.proj_back(z_hat)
            alpha_prev = self.get_alpha_bar(step - 1)
            noise = torch.randn_like(z_hat)
            z_T = (alpha_prev**0.5) * z_hat + ((1 - alpha_prev)**0.5) * noise

        final_z = self.z_proj(z_T)
        logits = self.classifier(torch.cat([h, final_z], dim=1))
        ce_loss = F.cross_entropy(logits, y)
        kl_loss = 0.5 * torch.sum((u_y + noise)**2, dim=1).mean()
        total_loss = ce_loss + self.eta * denoise_loss + kl_loss + pred_loss + fa_loss

        return total_loss, ce_loss, denoise_loss, kl_loss, pred_loss, fa_loss, ce_loss

    def predict(self, x):
        self.eval()
        with torch.no_grad():  # Only use no_grad during inference
            dummy_labels = torch.zeros(x.size(0), dtype=torch.long).to(x.device)
            h, _, _ = self.vit_local(x, dummy_labels, target_block_idx=6)
            z = torch.randn(x.size(0), self.embed_dim).to(x.device)
            for t in reversed(range(1, self.num_blocks + 1)):
                feat = self.label_proj(self.z_proj(z))
                z_hat = self.u_blocks[t - 1](feat)
                z_hat = self.proj_back(z_hat)
                alpha_prev = self.get_alpha_bar(t - 1)
                noise = torch.randn_like(z_hat)
                z = (alpha_prev**0.5) * z_hat + ((1 - alpha_prev)**0.5) * noise
            final_z = self.z_proj(z)
            return self.classifier(torch.cat([h, final_z], dim=1))


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


# === Train ===
model = NoPropDT(embed_dim=192, num_classes=10, num_blocks=10, eta=0.1).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

@torch.no_grad()
def visualize_label_denoising(model, num_samples=200):
    model.eval()
    print("Visualizing label embedding diffusion...")

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        u_y = model.embed(labels[:num_samples])
        break

    embed_dim = model.embed_dim
    num_blocks = model.num_blocks

    z_ts = []
    u_t_hats = []
    ts = []

    for t in range(1, num_blocks + 1):
        alpha_bar_t = model.get_alpha_bar(t)
        sqrt_alpha_bar_t = math.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = math.sqrt(1 - alpha_bar_t)

        noise = torch.randn_like(u_y)
        z_t = sqrt_alpha_bar_t * u_y + sqrt_one_minus_alpha_bar_t * noise
        label_feat = model.label_proj(model.z_proj(z_t))
        u_t_hat = model.u_blocks[t - 1](label_feat)

        z_ts.append(model.z_proj(z_t).cpu())
        u_t_hats.append(u_t_hat.cpu())
        ts.append(t)

    all_vectors = torch.cat(z_ts + u_t_hats, dim=0).numpy()
    labels_t = ["z_t"] * len(z_ts) + ["u_t_hat"] * len(u_t_hats)
    timesteps = [f"t={t}" for t in ts] * 2

    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto')
    proj_2d = tsne.fit_transform(all_vectors)

    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(ts)))

    offset = len(z_ts[0])
    for i, t in enumerate(ts):
        plt.scatter(proj_2d[i * offset:(i + 1) * offset, 0],
                    proj_2d[i * offset:(i + 1) * offset, 1],
                    color=colors[i], label=f"z_t t={t}", alpha=0.5, marker='o')

        plt.scatter(proj_2d[(i + len(ts)) * offset:(i + len(ts) + 1) * offset, 0],
                    proj_2d[(i + len(ts)) * offset:(i + len(ts) + 1) * offset, 1],
                    color=colors[i], label=f"u_t_hat t={t}", alpha=0.5, marker='x')

    plt.title("Label Embedding Diffusion vs Denoising Trajectories")
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1.0))
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("label_denoising_tsne.png", dpi=300)
    plt.show()

for epoch in range(100):
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        x, y = x.to(device), y.to(device)

        # === ViT: Curriculum scheduling for 7 blocks ===
        vit_progress = epoch / 100
        max_t_vit = int(vit_progress * 7)
        max_t_vit = max(0, min(max_t_vit, 6))
        t_vit = random.randint(0, max_t_vit)

        # === Denoise: Curriculum scheduling for 10 blocks ===
        denoise_progress = epoch / 100
        max_t_denoise = int(denoise_progress * model.num_blocks)
        max_t_denoise = max(1, min(max_t_denoise, model.num_blocks))
        t_denoise = random.randint(1, max_t_denoise)

        # === Forward ===
        loss, ce, d_loss, kl, pred_loss, fa_loss, cls_loss = model(x, y, t_vit, t_denoise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # === Evaluate ===
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model.predict(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f"Test Acc: {acc:.4f} | Train Loss: {total_loss / len(train_loader):.4f}")
    print(f"Epoch {epoch+1} Loss Breakdown → CE: {ce.item():.4f} | Denoise: {d_loss.item():.4f} | KL: {kl.item():.4f}")
    if (epoch + 1) % 10 == 0:
        visualize_label_denoising(model)
