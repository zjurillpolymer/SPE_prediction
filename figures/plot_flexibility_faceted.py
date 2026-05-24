"""Faceted: flexibility vs conductivity split by polarity level + 2D heatmap."""
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

df = pd.read_csv(os.path.join(ROOT, 'data/clean_train_data.csv'))

poly_summary = df.groupby('smiles').agg(
    mean_cond=('conductivity', 'mean'),
    n=('conductivity', 'count')
).reset_index()

data = []
for smi in poly_summary['smiles']:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    n_heavy = HeavyAtomCount(mol)
    n_rot = Descriptors.NumRotatableBonds(mol)
    rot_frac = n_rot / max(n_heavy, 1)
    n_o = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
    n_n = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
    on_ratio = (n_o + n_n) / max(n_heavy, 1)

    data.append({
        'rot_frac': rot_frac,
        'on_ratio': on_ratio,
        'mean_cond': poly_summary[poly_summary['smiles'] == smi]['mean_cond'].iloc[0],
    })

fdf = pd.DataFrame(data)

# Bin polarity into 3 levels
fdf['polarity'] = pd.qcut(fdf['on_ratio'], 3, labels=['Low O/N', 'Mid O/N', 'High O/N'])

# Stats per panel
panel_stats = {}
for label in ['Low O/N', 'Mid O/N', 'High O/N']:
    sub = fdf[fdf['polarity'] == label]
    r, _ = np.polyfit(sub['rot_frac'], sub['mean_cond'], 1) if len(sub) > 1 else (0, 0)
    r2 = np.corrcoef(sub['rot_frac'], sub['mean_cond'])[0, 1]**2 if len(sub) > 2 else 0
    panel_stats[label] = {'slope': r, 'r2': r2, 'n': len(sub)}

# ─── Figure: faceted scatter + 2D grid ──────────────────
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(14, 5.5))

# Left: faceted scatter
ax1 = fig.add_subplot(121)
colors = {'Low O/N': '#4575B4', 'Mid O/N': '#FEE090', 'High O/N': '#D73027'}
for label in ['Low O/N', 'Mid O/N', 'High O/N']:
    sub = fdf[fdf['polarity'] == label]
    ax1.scatter(sub['rot_frac'], sub['mean_cond'],
                c=colors[label], alpha=0.6, s=25, edgecolors='black',
                linewidths=0.2, label=label, zorder=3)
    # Fit line
    if len(sub) > 2:
        x, y = sub['rot_frac'], sub['mean_cond']
        coefs = np.polyfit(x, y, 1)
        xl = np.linspace(x.min(), x.max(), 50)
        ax1.plot(xl, np.polyval(coefs, xl), color=colors[label],
                 linewidth=1.5, alpha=0.8)
        st = panel_stats[label]
        ax1.text(x.median(), y.min() - 0.3, f'slope={st["slope"]:+.2f}',
                 fontsize=7, color=colors[label], ha='center',
                 fontweight='bold')

ax1.set_xlabel('Rotatable bonds / heavy atom (flexibility)', fontsize=11)
ax1.set_ylabel('Mean log Conductivity (S/cm)', fontsize=11)
ax1.set_title('Flexibility → Conductivity\n(faceted by O/N ratio)', fontsize=11, fontweight='bold')
ax1.legend(fontsize=8, framealpha=0.9)
ax1.grid(True, alpha=0.3)

# Right: 2D grid (flexibility × polarity, colored by conductivity)
ax2 = fig.add_subplot(122)
# Hexbin: x=flexibility, y=O/N ratio, color=conductivity
hb = ax2.hexbin(fdf['rot_frac'], fdf['on_ratio'], C=fdf['mean_cond'],
                 gridsize=12, cmap='RdYlBu_r', alpha=0.9,
                 edgecolors='white', linewidths=0.3,
                 reduce_C_function=np.mean, mincnt=1)
ax2.set_xlabel('Rotatable bonds / heavy atom (flexibility)', fontsize=11)
ax2.set_ylabel('O+N / heavy atom ratio (polarity)', fontsize=11)
ax2.set_title('2D Map: Flexibility × Polarity → Conductivity', fontsize=11, fontweight='bold')
cb = plt.colorbar(hb, ax=ax2, label='Mean log Conductivity (S/cm)')
ax2.grid(True, alpha=0.3)

# Annotate corners
ax2.text(0.05, 0.41, 'Rigid + Polar', fontsize=8, style='italic', alpha=0.5, transform=ax2.transData)
ax2.text(0.42, 0.41, 'Flexible + Polar', fontsize=8, style='italic', alpha=0.5, transform=ax2.transData)
ax2.text(0.05, 0.01, 'Rigid + Nonpolar', fontsize=8, style='italic', alpha=0.5, transform=ax2.transData)
ax2.text(0.45, 0.01, 'Flexible + Nonpolar', fontsize=8, style='italic', alpha=0.5, transform=ax2.transData)

plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'decoder/flexibility_faceted.png'),
            dpi=150, bbox_inches='tight')
print("Saved: decoder/flexibility_faceted.png")

# Print stats
print(f"\n{'Polarity Level':15s} {'n':5s} {'Slope':8s} {'R²':6s}")
for label in ['Low O/N', 'Mid O/N', 'High O/N']:
    st = panel_stats[label]
    print(f"{label:15s} {st['n']:5d} {st['slope']:+7.3f} {st['r2']:.3f}")
