"""Plot predicted conductivity vs molecular weight for PEG and PP."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from encoder import molecule_to_graph
from encoder.MPNN import MPNNEmbedding
from decoder.dataloader import compute_anion_features
from torch_geometric.data import Batch
from decoder.spe_prediction import SPE_Predictor, EXTRA_COLS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RDLogger.logger().setLevel(RDLogger.ERROR)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── Load training stats ────────────────────────────────
df = pd.read_csv(os.path.join(ROOT, 'data/clean_train_data.csv'))
df['inv_temp'] = 1.0 / (df['temperature'] + 273.15)
unique_salts = df['salt smiles'].fillna('').unique()
salt_map = {s: compute_anion_features(s if s else None) for s in unique_salts}
anion_feats = df['salt smiles'].fillna('').map(salt_map)
df[['anion_volume', 'anion_mass', 'anion_charge']] = pd.DataFrame(
    anion_feats.tolist(), index=df.index)

extra_mean = df[EXTRA_COLS].mean().values.astype(np.float32)
extra_std = df[EXTRA_COLS].std().values.astype(np.float32)
extra_std[extra_std == 0] = 1.0

# ─── Load model ─────────────────────────────────────────
model = SPE_Predictor().to(DEVICE)
model.load_state_dict(torch.load(
    os.path.join(ROOT, 'decoder/model(1).pt'),
    map_location=DEVICE, weights_only=True))
model.eval()

# ─── Define polymers ────────────────────────────────────
polymers = {
    'PEG (PEO)': '[Cu]C(OCCOCCOCCOCCOCCOCCOCCOCCOC)C[Au]',
    'PP':         '[Cu]C(C)[Au]',
}

salt_smi = 'O=S([N-]S(=O)(C(F)(F)F)=O)(C(F)(F)F)=O.[Li+]'  # LiTFSI
anion_feat = compute_anion_features(salt_smi)
temperature = 80
inv_t = 1.0 / (temperature + 273.15)

mw_range = np.logspace(np.log10(500), np.log10(500000), 50)

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

colors = {'PEG (PEO)': '#4ECDC4', 'PP': '#FF6B6B'}
markers = {'PEG (PEO)': 'o', 'PP': 's'}

for poly_name, smi in polymers.items():
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    data = molecule_to_graph.mol_to_pyg_graph(mol)
    data = data.to(DEVICE)

    # Precompute MPNN embedding (same for all MW)
    batch_single = Batch.from_data_list([data]).to(DEVICE)
    with torch.no_grad():
        emb = model.encoder(batch_single)

    preds = []
    for mw in mw_range:
        extra_raw = torch.tensor([[mw, 0.5, anion_feat[0], anion_feat[1], anion_feat[2]]],
                                  dtype=torch.float)
        extra_norm = (extra_raw - torch.tensor(extra_mean)) / torch.tensor(extra_std)
        combined = torch.cat([emb, extra_norm.to(DEVICE)], dim=-1)
        with torch.no_grad():
            params = model.regressor(combined)
            A, EaR = params[0, 0], params[0, 1]
            log_sigma = A - EaR * inv_t
        preds.append(float(log_sigma))

    # Convert to original scale
    y_mean = float(df['conductivity'].mean())
    y_std = float(df['conductivity'].std())
    preds = np.array(preds) * y_std + y_mean

    ax.plot(mw_range / 1000, preds, color=colors[poly_name],
            linewidth=2.5, label=poly_name, marker=markers[poly_name],
            markevery=8, markersize=6)

# Also add data from training set for reference
train_pegs = df[df['smiles'].str.contains('OCCOC', na=False)]
if len(train_pegs) > 0:
    ax.scatter(train_pegs['mw'] / 1000, train_pegs['conductivity'],
               c='#4ECDC4', alpha=0.15, s=10, zorder=2)

ax.set_xscale('log')
ax.set_xlabel('Molecular Weight (kg/mol)', fontsize=12)
ax.set_ylabel('Predicted log Conductivity (S/cm)', fontsize=12)
ax.set_title('Conductivity vs Molecular Weight (T=80°C, LiTFSI, 0.5 mol/kg)',
             fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Annotate
ax.annotate('Higher MW → fewer chain ends\n→ lower ionic conductivity',
            xy=(100, -4.5), fontsize=9, style='italic', alpha=0.6,
            ha='center')

plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'decoder/mw_vs_conductivity.png'),
            dpi=150, bbox_inches='tight')
print("Saved: decoder/mw_vs_conductivity.png")
