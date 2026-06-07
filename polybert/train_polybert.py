import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from polybert.smiles_to_psmiles import smiles_to_psmiles
from decoder.dataloader import compute_anion_features

# ─── Config ─────────────────────────────────────────────
POLYBERT_MODEL = 'xushijie/polyBERT'
HIDDEN_DIM = 128
BATCH_SIZE = 32
EPOCHS = 500
LR = 1e-3
EXTRA_COLS = ['mw', 'molality', 'anion_volume', 'anion_mass', 'anion_charge']
# ────────────────────────────────────────────────────────


class PolyBERT_Dataset(Dataset):
    """Precompute polyBERT embeddings; fast training without full DeBERTa forward pass."""

    def __init__(self, df, polybert_model, cache_path=None):
        self.df = df.reset_index(drop=True)
        self._precompute_embeddings(polybert_model, cache_path)
        self._build_index()

    def _precompute_embeddings(self, model, cache_path):
        if cache_path and os.path.exists(cache_path):
            print(f"Loading polyBERT cache: {cache_path}")
            cached = torch.load(cache_path, weights_only=False)
            self.emb_map = cached['emb_map']
            return

        unique_smiles = self.df['smiles'].unique()
        psmiles = []
        valid_smiles = []
        for s in unique_smiles:
            p = smiles_to_psmiles(s)
            if p:
                psmiles.append(p)
                valid_smiles.append(s)

        print(f"Computing polyBERT embeddings for {len(psmiles)} polymers ...")
        embs = model.encode(psmiles, show_progress_bar=True)
        self.emb_map = dict(zip(valid_smiles, embs))

        skipped = len(unique_smiles) - len(valid_smiles)
        if skipped:
            print(f"Warning: {skipped} SMILES failed conversion, skipped.")

        if cache_path:
            torch.save({'emb_map': self.emb_map}, cache_path)
            print(f"polyBERT cache saved: {cache_path}")

    def _build_index(self):
        self.embeddings, self.extras, self.inv_temps, self.ys = [], [], [], []
        fail = 0
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            emb = self.emb_map.get(row['smiles'])
            if emb is None:
                fail += 1
                continue
            self.embeddings.append(torch.from_numpy(emb).float())
            extra = [0.0 if pd.isna(row[c]) else row[c] for c in EXTRA_COLS]
            self.extras.append(torch.tensor(extra, dtype=torch.float))
            self.inv_temps.append(torch.tensor([row['inv_temp']], dtype=torch.float))
            self.ys.append(torch.tensor([row['conductivity']], dtype=torch.float))
        if fail:
            print(f"Warning: {fail} rows skipped (missing embedding).")

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return (self.embeddings[idx], self.extras[idx],
                self.inv_temps[idx], self.ys[idx])


def collate_fn(batch):
    embs = torch.stack([b[0] for b in batch])
    extras = torch.stack([b[1] for b in batch])
    inv_temps = torch.stack([b[2] for b in batch])
    ys = torch.cat([b[3] for b in batch])
    return embs, extras, inv_temps, ys


class PolyBERT_SPE_Predictor(nn.Module):
    """Simple projection on frozen polyBERT embeddings + Arrhenius output."""

    def __init__(self, polybert_dim=600, hidden_dim=128, extra_dim=5, dropout=0.15):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(polybert_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim + extra_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, emb, extra, inv_temp):
        h = self.projection(emb)
        h = torch.cat([h, extra], dim=-1)
        params = self.regressor(h)
        A, EaR = params[:, 0:1], params[:, 1:2]
        cond = A - EaR * inv_temp
        return cond.squeeze(-1)


def train():
    df = pd.read_csv('data/clean_train_data.csv')
    df['inv_temp'] = 1.0 / (df['temperature'] + 273.15)

    # Anion features
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

    # Normalize
    extra_mean = train_df[EXTRA_COLS].mean().values.astype(np.float32)
    extra_std = train_df[EXTRA_COLS].std().values.astype(np.float32)
    extra_std[extra_std == 0] = 1.0
    y_mean = np.float32(train_df['conductivity'].mean())
    y_std = np.float32(train_df['conductivity'].std())

    def normalize(df_):
        df_[EXTRA_COLS] = (df_[EXTRA_COLS] - extra_mean) / extra_std
        df_['conductivity'] = (df_['conductivity'] - y_mean) / y_std
        return df_

    train_df = normalize(train_df)
    valid_df = normalize(valid_df)
    test_df  = normalize(test_df)

    # PolyBERT encoder (frozen)
    print(f"Loading {POLYBERT_MODEL} ...")
    polybert_model = SentenceTransformer(POLYBERT_MODEL)
    polybert_model.eval()

    # Datasets
    train_ds = PolyBERT_Dataset(train_df, polybert_model, 'polybert/cache_train.pt')
    valid_ds = PolyBERT_Dataset(valid_df, polybert_model, 'polybert/cache_valid.pt')
    test_ds  = PolyBERT_Dataset(test_df,  polybert_model, 'polybert/cache_test.pt')

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, BATCH_SIZE, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  BATCH_SIZE, collate_fn=collate_fn)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PolyBERT_SPE_Predictor().to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {total:,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = nn.MSELoss()

    best_val = float('inf')
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for emb, extra, inv_temp, y in train_loader:
            emb, extra, inv_temp, y = emb.to(device), extra.to(device), inv_temp.to(device), y.to(device)
            pred = model(emb, extra, inv_temp)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * emb.size(0)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for emb, extra, inv_temp, y in valid_loader:
                emb, extra, inv_temp, y = emb.to(device), extra.to(device), inv_temp.to(device), y.to(device)
                pred = model(emb, extra, inv_temp)
                val_loss += loss_fn(pred, y).item() * emb.size(0)

        avg_train = total_loss / len(train_ds)
        avg_val = val_loss / len(valid_ds)
        scheduler.step()

        if avg_val < best_val:
            best_val = avg_val
            best_epoch = epoch

        if epoch % 25 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}  |  train: {avg_train:.4f}  |  val: {avg_val:.4f}  |  lr: {scheduler.get_last_lr()[0]:.2e}")

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for emb, extra, inv_temp, y in test_loader:
            emb, extra, inv_temp, y = emb.to(device), extra.to(device), inv_temp.to(device), y.to(device)
            pred = model(emb, extra, inv_temp)
            test_loss += loss_fn(pred, y).item() * emb.size(0)
    test_loss /= len(test_ds)

    print(f"\nBest val: {best_val:.4f} @ epoch {best_epoch}")
    print(f"Test loss (normalized):   {test_loss:.4f}")
    print(f"Test loss (original scale): {test_loss * y_std**2:.4f}")

    torch.save(model.state_dict(), 'polybert/model.pt')
    print("Model saved: polybert/model.pt")


if __name__ == '__main__':
    train()
