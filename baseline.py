#!/usr/bin/env python
"""
baseline_pyg.py  –  Step 0 baseline for the MLOps pipeline
---------------------------------------------------------
* Loads the BACE classification set from MoleculeNet (PyTorch Geometric)
* Trains a 2-layer Graph Convolutional Network (GCN) for 5 epochs
* Reports AUROC on a held-out test set
* Saves trained weights and metrics under ./artifacts/

Runs in ≈ 5-8 minutes on a typical laptop CPU; reproducible with the
environment.yml shown earlier (PyTorch 2.2 + PyG 2.5 + Python 3.10).
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# ------------------------------------------------------------------
# 0.  Reproducibility
# ------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ------------------------------------------------------------------
# 1.  Dataset & splits
# ------------------------------------------------------------------
dataset = MoleculeNet(root="data", name="BACE")          # 1 769 molecules total
num_graphs = len(dataset)

perm = torch.randperm(num_graphs)
train_idx = perm[: int(0.8 * num_graphs)]
val_idx   = perm[int(0.8 * num_graphs): int(0.9 * num_graphs)]
test_idx  = perm[int(0.9 * num_graphs):]

train_loader = DataLoader(dataset[train_idx], batch_size=64,  shuffle=True)
val_loader   = DataLoader(dataset[val_idx],   batch_size=256)
test_loader  = DataLoader(dataset[test_idx],  batch_size=256)

# ------------------------------------------------------------------
# 2.  Model definition
# ------------------------------------------------------------------
class GCN(torch.nn.Module):
    """Two-layer GCN followed by global mean pooling and a linear head."""

    def __init__(self, in_feats: int, hidden: int = 64):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden)
        self.conv2 = GCNConv(hidden,  hidden)
        self.head  = torch.nn.Linear(hidden, 1)

    def forward(self, x, edge_index, batch):
        # Cast node features to float32 (MoleculeNet loads them as Int64)
        x = x.float()
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)          # graph-level readout
        return self.head(x).squeeze(-1)         # logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = GCN(dataset.num_node_features).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion  = torch.nn.BCEWithLogitsLoss()

# ------------------------------------------------------------------
# 3.  Train / validation / test routine
# ------------------------------------------------------------------
def run(loader, training: bool = False):
    model.train() if training else model.eval()
    total_loss, ys, preds = 0.0, [], []

    for data in loader:
        data = data.to(device)
        logits = model(data.x, data.edge_index, data.batch)
        loss = criterion(logits, data.y.float().view(-1))

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * data.num_graphs
        ys.append(data.y.cpu())
        preds.append(logits.detach().cpu())

    y_true  = torch.cat(ys).numpy()
    y_score = torch.cat(preds).sigmoid().numpy()
    auc     = roc_auc_score(y_true, y_score)

    return total_loss / len(loader.dataset), auc

# ------------------------------------------------------------------
# 4.  Training loop
# ------------------------------------------------------------------
for epoch in range(1, 6):
    tr_loss, tr_auc = run(train_loader, training=True)
    val_loss, val_auc = run(val_loader)
    print(f"Epoch {epoch:2d} │ train AUC {tr_auc:.3f} │ val AUC {val_auc:.3f}")

test_loss, test_auc = run(test_loader)
print(f"\n🧪  Baseline test AUROC: {test_auc:.3f}")

# ------------------------------------------------------------------
# 5.  Persist artefacts
# ------------------------------------------------------------------
Path("artifacts").mkdir(exist_ok=True)
torch.save(model.state_dict(), "artifacts/gcn_bace.pth")

with open("artifacts/baseline_metrics.json", "w") as fp:
    json.dump({"test_auroc": float(test_auc),
               "epochs": 5,
               "seed": SEED},
              fp, indent=2)

print("✅  Artefacts saved in ./artifacts/")
