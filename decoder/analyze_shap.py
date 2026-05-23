"""SHAP analysis: feature importance of extra features (especially anion volume)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import shap

from encoder.MPNN import MPNNEmbedding
from decoder.dataloader import SPEdataset, spe_collate, compute_anion_features
from decoder.spe_prediction import SPE_Predictor, EXTRA_COLS

BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── Load data ──────────────────────────────────────────
df = pd.read_csv('../data/clean_train_data.csv')
df['inv_temp'] = 1.0 / (df['temperature'] + 273.15)
unique_salts = df['salt smiles'].fillna('').unique()
salt_feat_map = {s: compute_anion_features(s if s else None) for s in unique_salts}
anion_feats = df['salt smiles'].fillna('').map(salt_feat_map)
df[['anion_volume', 'anion_mass', 'anion_charge']] = pd.DataFrame(
    anion_feats.tolist(), index=df.index)

# Validation split
np.random.seed(42)
idx = np.random.permutation(len(df))
n_train = int(len(df) * 0.8)
valid_df = df.iloc[idx[n_train:n_train + int(len(df) * 0.1)]].copy()

extra_mean = df[EXTRA_COLS].mean().values.astype(np.float32)
extra_std = df[EXTRA_COLS].std().values.astype(np.float32)
extra_std[extra_std == 0] = 1.0
valid_df[EXTRA_COLS] = (valid_df[EXTRA_COLS] - extra_mean) / extra_std

ds = SPEdataset(valid_df, cache_path=None)
loader = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=spe_collate)

# ─── Load model ─────────────────────────────────────────
model = SPE_Predictor().to(DEVICE)
model.load_state_dict(torch.load('decoder/model.pt', map_location=DEVICE, weights_only=True))
model.eval()

# ─── Pre-compute MPNN embeddings ────────────────────────
all_emb = []
all_extra = []
with torch.no_grad():
    for batch in loader:
        batch = batch.to(DEVICE)
        emb = model.encoder(batch)
        all_emb.append(emb.cpu())
        all_extra.append(batch.extra.cpu())

all_emb = torch.cat(all_emb, dim=0).numpy()     # [N, 128]
all_extra = torch.cat(all_extra, dim=0).numpy()  # [N, 6]

# ─── SHAP (fixed embedding approach) ────────────────────
# Use mean embedding as a fixed "background polymer",
# so SHAP isolates the effect of extra features only.
mean_emb = all_emb.mean(0, keepdims=True)  # [1, 128]

def predict_fn(extra_feats):
    """Same mean embedding for every sample — explains extra features alone."""
    emb = np.tile(mean_emb, (len(extra_feats), 1))
    combined = np.concatenate([emb, extra_feats], axis=1)
    combined_t = torch.tensor(combined, dtype=torch.float, device=DEVICE)
    with torch.no_grad():
        return model.regressor(combined_t).squeeze(-1).cpu().numpy()

N_BG = 100
N_EXPLAIN = 300

background = all_extra[:N_BG]
explain_extra = all_extra[:N_EXPLAIN]

print("Computing SHAP values (this may take a few minutes)...")
explainer = shap.KernelExplainer(predict_fn, background)
shap_values = explainer.shap_values(explain_extra, nsamples=200)

# ─── Plots ──────────────────────────────────────────────
import matplotlib.pyplot as plt

shap.summary_plot(shap_values, explain_extra, feature_names=EXTRA_COLS,
                  show=False)
plt.tight_layout()
plt.savefig('decoder/shap_summary.png', dpi=150, bbox_inches='tight')
print("Saved: decoder/shap_summary.png")

shap.summary_plot(shap_values, explain_extra, feature_names=EXTRA_COLS,
                  plot_type='bar', show=False)
plt.tight_layout()
plt.savefig('decoder/shap_bar.png', dpi=150, bbox_inches='tight')
print("Saved: decoder/shap_bar.png")

# Text summary
print("\nFeature importance (mean |SHAP|):")
mean_shap = np.abs(shap_values).mean(axis=0)
for name, val in sorted(zip(EXTRA_COLS, mean_shap), key=lambda x: -x[1]):
    print(f"  {name:20s}  {val:.4f}")
