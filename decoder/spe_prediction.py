import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn, optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from encoder.MPNN import MPNNEmbedding
from decoder.dataloader import SPEdataset, spe_collate, compute_anion_features

# ─── Config ─────────────────────────────────────────────
HIDDEN_DIM = 128
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
EXTRA_COLS = ['mw', 'molality', 'anion_volume', 'anion_mass', 'anion_charge']
# ────────────────────────────────────────────────────────

class SPE_Predictor(nn.Module):
    """Arrhenius output layer: predicts A and Ea/R, then log σ = A - Ea/R · 1/T."""

    def __init__(self, node_in_dims=31, edge_dim=6, hidden_dim=HIDDEN_DIM,
                 num_layers=6, extra_dim=5, dropout=0.1):
        super().__init__()
        self.encoder = MPNNEmbedding(node_in_dims, edge_dim, hidden_dim, num_layers)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim + extra_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),  # A, Ea/R
        )

    def forward(self, data):
        emb = self.encoder(data)
        emb = torch.cat([emb, data.extra], dim=-1)
        params = self.regressor(emb)               # [B, 2]
        A, EaR = params[:, 0:1], params[:, 1:2]
        cond = A - EaR * data.inv_temp             # Arrhenius: [B, 1]
        return cond.squeeze(-1)


def train():
    df = pd.read_csv('../data/clean_train_data.csv')
    df['inv_temp'] = 1.0 / (df['temperature'] + 273.15)
    # Anion features — compute once per unique salt, then map
    unique_salts = df['salt smiles'].fillna('').unique()
    salt_feat_map = {s: compute_anion_features(s if s else None) for s in unique_salts}
    anion_feats = df['salt smiles'].fillna('').map(salt_feat_map)
    df[['anion_volume', 'anion_mass', 'anion_charge']] = pd.DataFrame(
        anion_feats.tolist(), index=df.index)

    # Split
    np.random.seed(42)
    idx = np.random.permutation(len(df))
    n_train, n_valid = int(len(df) * 0.8), int(len(df) * 0.1)
    train_df = df.iloc[idx[:n_train]].copy()
    valid_df = df.iloc[idx[n_train:n_train + n_valid]].copy()
    test_df  = df.iloc[idx[n_train + n_valid:]].copy()

    # Normalize extra features (5 cols, NO inv_temp) and target
    extra_mean = train_df[EXTRA_COLS].mean().values.astype(np.float32)
    extra_std = train_df[EXTRA_COLS].std().values.astype(np.float32)
    extra_std[extra_std == 0] = 1.0

    y_mean = train_df['conductivity'].mean().astype(np.float32)
    y_std = train_df['conductivity'].std().astype(np.float32)

    def normalize(df_):
        df_[EXTRA_COLS] = (df_[EXTRA_COLS] - extra_mean) / extra_std
        df_['conductivity'] = (df_['conductivity'] - y_mean) / y_std
        # inv_temp stays RAW — used in physical Arrhenius formula
        return df_

    train_df = normalize(train_df)
    valid_df = normalize(valid_df)
    test_df  = normalize(test_df)

    # Datasets
    train_ds = SPEdataset(train_df, cache_path='decoder/cache_train.pt')
    valid_ds = SPEdataset(valid_df, cache_path='decoder/cache_valid.pt')
    test_ds  = SPEdataset(test_df,  cache_path='decoder/cache_test.pt')

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=spe_collate)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, collate_fn=spe_collate)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, collate_fn=spe_collate)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SPE_Predictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = loss_fn(pred, batch.y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(device)
                pred = model(batch)
                val_loss += loss_fn(pred, batch.y).item() * batch.num_graphs

        avg_train = total_loss / len(train_ds)
        avg_val = val_loss / len(valid_ds)
        print(f"Epoch {epoch:3d}  |  train: {avg_train:.4f}  |  val: {avg_val:.4f}")

    # Test (in normalized scale)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            test_loss += loss_fn(pred, batch.y).item() * batch.num_graphs
    test_loss /= len(test_ds)
    print(f"\nTest loss (normalized): {test_loss:.4f}")
    print(f"Test loss (original scale): {test_loss * y_std**2:.4f}")

    # Save model for downstream use
    torch.save(model.state_dict(), 'decoder/model.pt')
    print("Model saved: decoder/model.pt")


if __name__ == '__main__':
    train()
