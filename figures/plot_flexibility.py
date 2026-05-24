"""Log conductivity vs polymer flexibility, colored by O/N atom ratio (polarity)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.Lipinski import HeavyAtomCount

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RDLogger.logger().setLevel(RDLogger.ERROR)

# ─── Load training data ─────────────────────────────────
df = pd.read_csv(os.path.join(ROOT, 'data/clean_train_data.csv'))

poly_summary = df.groupby('smiles').agg(
    mean_cond=('conductivity', 'mean'),
    n=('conductivity', 'count')
).reset_index()

# ─── Compute polymer features ───────────────────────────
data = []
for smi in poly_summary['smiles']:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    n_heavy = HeavyAtomCount(mol)
    n_rot = Descriptors.NumRotatableBonds(mol)
    rot_frac = n_rot / max(n_heavy, 1)

    # O/N atom ratio (polarity proxy)
    n_o = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
    n_n = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
    on_ratio = (n_o + n_n) / max(n_heavy, 1)

    data.append({
        'smiles': smi,
        'rot_frac': rot_frac,
        'on_ratio': on_ratio,
        'mean_cond': poly_summary[poly_summary['smiles'] == smi]['mean_cond'].iloc[0],
        'n': poly_summary[poly_summary['smiles'] == smi]['n'].iloc[0],
    })

fdf = pd.DataFrame(data)

# ─── Plot ───────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))

sc = ax.scatter(fdf['rot_frac'], fdf['mean_cond'],
                c=fdf['on_ratio'], cmap='RdYlBu_r', alpha=0.8,
                s=fdf['n'] * 2 + 10, edgecolors='black', linewidths=0.3, zorder=3)

# Linear fit (flexibility vs conductivity)
mask = np.isfinite(fdf['rot_frac']) & np.isfinite(fdf['mean_cond'])
x, y = fdf['rot_frac'][mask], fdf['mean_cond'][mask]
coefs = np.polyfit(x, y, 1)
r2 = np.corrcoef(x, y)[0, 1] ** 2
x_line = np.linspace(x.min(), x.max(), 100)
ax.plot(x_line, np.polyval(coefs, x_line), '--', color='gray',
        linewidth=1.2, alpha=0.6, label=f'Flexibility fit (R²={r2:.3f})')

# Annotations
peo = fdf[(fdf['rot_frac'] > 0.35) & (fdf['on_ratio'] > 0.2)]
rigid = fdf[fdf['rot_frac'] < 0.1]
alkane = fdf[(fdf['on_ratio'] < 0.05) & (fdf['rot_frac'] > 0.1)]

for subset, label in [(peo, 'PEO-like (high O)'), (rigid, 'Rigid'), (alkane, 'Low polarity')]:
    if len(subset) > 0:
        rep = subset.nlargest(1, 'n').iloc[0]
        ax.annotate(label, (rep['rot_frac'], rep['mean_cond']),
                    fontsize=8, alpha=0.7, fontweight='bold',
                    xytext=(6, 6), textcoords='offset points')

ax.set_xlabel('Rotatable bonds / heavy atom  (flexibility →)', fontsize=12)
ax.set_ylabel('Mean log Conductivity (S/cm)', fontsize=12)
ax.set_title('Polymer Flexibility vs Conductivity\n(colored by O/N atom ratio: red=high polarity, blue=low)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
cb = plt.colorbar(sc, ax=ax, label='O+N / heavy atom ratio (polarity)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'decoder/flexibility_vs_conductivity.png'),
            dpi=150, bbox_inches='tight')
print("Saved: decoder/flexibility_vs_conductivity.png")

# Stats
print(f"R²(flexibility): {r2:.3f}")
print(f"R²(O/N ratio):  {np.corrcoef(fdf['on_ratio'], fdf['mean_cond'])[0,1]**2:.3f}")
print(f"Total polymers: {len(fdf)}")
