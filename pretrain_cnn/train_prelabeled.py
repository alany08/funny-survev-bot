from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.optim import Adam
from network import ActorCritic

class BCDataset(Dataset):
    def __init__(self, path):
        with open(path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frames, state = self.data[idx]
        key_status, mousepos = state
        key_status = key_status.copy()
        key_status.append(1)

        # Stack 4 grayscale frames → [4, 360, 640]
        stacked = np.stack(frames, axis=0).astype(np.float32) / 255.0
        stacked = torch.tensor(stacked)

        keys  = torch.tensor(key_status, dtype=torch.float32)
        mouse = torch.tensor(mousepos,   dtype=torch.float32)

        return stacked, keys, mouse

class BCWrapper(nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.backbone = actor_critic.backbone
        self.actor_critic = actor_critic

    def forward(self, x):
        a, b, c = self.actor_critic.forward(x)
        return a.logits, b.mean

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = ActorCritic().to(device)
ckpt = torch.load("checkpoints/model.pt", map_location = device)
net.load_state_dict(ckpt["model"])

net = BCWrapper(net).to(device)

for param in net.parameters():
    param.requires_grad = False

for param in net.backbone.parameters():
    param.requires_grad = True

for name, param in net.named_parameters():
    print(f"{name:40s} trainable={param.requires_grad}")

optimizer = Adam(net.backbone.parameters(), lr=3e-4)

dataset = BCDataset("pretrain_cnn/labeled_data.pkl")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

key_loss_fn   = nn.BCEWithLogitsLoss()
mouse_loss_fn = nn.MSELoss()

net.train()

for epoch in range(20):
    total_loss = 0

    for frames, keys, mouse in dataloader:
        frames = frames.to(device)
        keys   = keys.to(device)
        mouse  = mouse.to(device)

        key_pred, mouse_pred = net.forward(frames)

        loss = key_loss_fn(key_pred, keys) + mouse_loss_fn(mouse_pred, mouse)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(dataloader):.4f}")

torch.save({"model": net.actor_critic.state_dict()}, "checkpoints/model_bc.pt")