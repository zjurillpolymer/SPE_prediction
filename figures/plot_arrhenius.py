"""Arrhenius plot: log σ vs 1/T for different polymers and salts."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from torch_geometric.data import Batch
from encoder import molecule_to_graph
from encoder.MPNN import MPNNEmbedding
from decoder.dataloader import compute_anion_features
from decoder.spe_prediction import SPE_Predictor, EXTRA_COLS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RDLogger.logger().setLevel(RDLogger.ERROR)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── Load stats ─────────────────────────────────────────
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
y_mean = float(df['conductivity'].mean())
y_std = float(df['conductivity'].std())

# ─── Load model ─────────────────────────────────────────
model = SPE_Predictor().to(DEVICE)
model.load_state_dict(torch.load(
    os.path.join(ROOT, 'decoder/model(1).pt'),
    map_location=DEVICE, weights_only=True))
model.eval()

# ─── Configurations ──────────────────────────────────────
polymers = {
    'PEG (PEO)': '[Cu]C(OCCOCCOCCOCCOCCOCCOCCOCCOC)C[Au]',
    'PP':         '[Cu]C(C)[Au]',
}
salts = {
    'LiTFSI': 'O=S([N-]S(=O)(C(F)(F)F)=O)(C(F)(F)F)=O.[Li+]',
    'LiBF4':  'F[B-](F)(F)F.[Li+]',
}
molality = 1.5
mw = 10000
temps_c = np.arange(0, 101, 5)  # 0°C to 100°C

# ─── Precompute encodings ───────────────────────────────
emb_cache = {}
for name, smi in polymers.items():
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    data = molecule_to_graph.mol_to_pyg_graph(mol)
    batch = Batch.from_data_list([data]).to(DEVICE)
    with torch.no_grad():
        emb_cache[name] = model.encoder(batch)

# ─── Predict ────────────────────────────────────────────
results = []
for poly_name, emb in emb_cache.items():
    for salt_name, salt_smi in salts.items():
        anion = compute_anion_features(salt_smi)
        log_sigmas = []
        for tc in temps_c:
            inv_t = 1.0 / (tc + 273.15)
            extra_raw = torch.tensor([[mw, molality, anion[0], anion[1], anion[2]]],
                                     dtype=torch.float)
            extra_norm = (extra_raw - torch.tensor(extra_mean)) / torch.tensor(extra_std)
            combined = torch.cat([emb, extra_norm.to(DEVICE)], dim=-1)
            with torch.no_grad():
                params = model.regressor(combined)
                A, EaR = params[0, 0], params[0, 1]
                log_sigma = A - EaR * inv_t
            log_sigmas.append(float(log_sigma))
        log_sigmas = np.array(log_sigmas) * y_std + y_mean
        results.append((poly_name, salt_name, log_sigmas))

# ─── Plot ───────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))

colors = {'PEG (PEO)': '#4ECDC4', 'PP': '#FF6B6B'}
linestyles = {'LiTFSI': '-', 'LiBF4': '--'}

inv_temps = 1.0 / (np.array(temps_c) + 273.15)

for poly_name, salt_name, log_sigmas in results:
    label = f'{poly_name} + {salt_name}'
    ax.plot(inv_temps * 1000, log_sigmas,
            color=colors[poly_name], ls=linestyles[salt_name],
            linewidth=2.5, label=label)

    # Fit line to get Ea (should be perfectly linear by construction)
    slope = np.polyfit(inv_temps, log_sigmas, 1)[0]
    ea_kjmol = -slope * 8.314 / 1000
    # Annotate near the middle
    mid_idx = len(inv_temps) // 2
    ax.annotate(f'Ea = {ea_kjmol:.1f} kJ/mol',
                xy=(inv_temps[mid_idx] * 1000, log_sigmas[mid_idx]),
                fontsize=8, color=colors[poly_name],
                fontweight='bold',
                xytext=(10, 15), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color=colors[poly_name], alpha=0.5))

ax.set_xlabel('1000 / T (K⁻¹)', fontsize=12)
ax.set_ylabel('log Conductivity (S/cm)', fontsize=12)
ax.set_title('Arrhenius Plot: log σ vs 1/T',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)

# Temperature labels on top
temp_labels = [0, 25, 50, 80, 100]
temp_ticks = [1000/(t+273.15) for t in temp_labels]
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(temp_ticks)
ax2.set_xticklabels([str(t) for t in temp_labels])
ax2.set_xlabel('Temperature (°C)', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'decoder/arrhenius_plot.png'),
            dpi=150, bbox_inches='tight')
print("Saved: decoder/arrhenius_plot.png")
