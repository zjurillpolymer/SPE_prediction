"""Compare predicted PolyInfo conductivities with real training data for similar polymers."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, AllChem
from rdkit.Chem.Lipinski import HeavyAtomCount
from scipy.spatial.distance import cdist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RDLogger.logger().setLevel(RDLogger.ERROR)
N_BITS = 128
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=N_BITS)


def create_long_smiles(smile, req_length=30):
    if "Cu" not in smile:
        return smile
    try:
        num_heavy = HeavyAtomCount(Chem.MolFromSmiles(smile)) - 2
        repeats = math.ceil(req_length / num_heavy) - 1
        if repeats > 0:
            mol = Chem.MolFromSmiles(smile)
            new_mol = mol
            for i in range(repeats):
                rxn = AllChem.ReactionFromSmarts("[Cu][*:1].[*:2][Au]>>[*:1]-[*:2]")
                results = rxn.RunReactants((mol, new_mol))
                assert len(results) == 1 and len(results[0]) == 1
                new_mol = results[0][0]
            new_smile = Chem.MolToSmiles(new_mol)
        else:
            new_smile = smile
        new_smile = new_smile.replace("[Cu]", "C").replace("[Au]", "C").replace("[Ca]", "C")
        return Chem.MolToSmiles(Chem.MolFromSmiles(new_smile))
    except:
        return None


def get_fp(smiles_list):
    fps, valid = [], []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            fps.append(list(mfpgen.GetFingerprint(mol)))
            valid.append(s)
    return np.array(fps, dtype=np.float32), valid


def get_fp_single(s):
    mol = Chem.MolFromSmiles(s)
    if mol:
        return np.array(list(mfpgen.GetFingerprint(mol)), dtype=np.float32)
    return None


# ─── Load ───────────────────────────────────────────────
train_df = pd.read_csv(os.path.join(ROOT, 'data/clean_train_data.csv'))
pred_df = pd.read_csv(os.path.join(ROOT, 'decoder/polyinfo_8salts_pred.csv'))

# Compute fingerprints for all unique train SMILES
train_smiles_arr = train_df['smiles'].dropna().unique()
train_fps, train_smi_valid = get_fp(train_smiles_arr)
train_smi_set = set(train_smi_valid)

# Build dict train_smiles -> list of conductivities
train_cond_map = {}
for smi, grp in train_df.groupby('smiles'):
    train_cond_map[smi] = grp['conductivity'].values

# Pick several diverse PolyInfo polymers covering high/mid/low similarity
poly_orig_smi = pred_df['smiles'].unique()
candidates = []
for s in poly_orig_smi:
    ext = create_long_smiles(s)
    if not ext:
        continue
    fp = get_fp_single(ext)
    if fp is None:
        continue
    sims = 1 - cdist(fp.reshape(1, -1), train_fps, metric='rogerstanimoto')[0]
    best_sim = sims.max()
    best_idx = sims.argmax()
    candidates.append({
        'orig_smi': s,
        'ext_smi': ext,
        'sim': best_sim,
        'train_smi': train_smi_valid[best_idx],
        'train_cond': train_cond_map.get(train_smi_valid[best_idx], [])
    })

# Sort by similarity and pick diverse examples
candidates.sort(key=lambda x: x['sim'], reverse=True)
# Pick top 6 covering different similarity levels
picked = candidates[:3]
# Also pick some mid and lower similarity
for c in candidates:
    if len(picked) >= 6:
        break
    if c not in picked and c['sim'] < 0.7:
        picked.append(c)

# For each picked pair, also get the nearest PolyInfo polymer's predicted conductivities
for p in picked:
    sub = pred_df[pred_df['smiles'] == p['orig_smi']]
    p['poly_preds'] = sub['pred_log_conductivity'].values

# ─── Plot ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for idx, p in enumerate(picked):
    ax = axes[idx]
    train_vals = p['train_cond']
    poly_vals = p['poly_preds']

    bp = ax.boxplot([train_vals, poly_vals], positions=[0, 1], widths=0.5,
                    patch_artist=True,
                    boxprops=dict(linewidth=1.2),
                    medianprops=dict(color='black', linewidth=2))

    bp['boxes'][0].set_facecolor('#4ECDC4')
    bp['boxes'][1].set_facecolor('#FF6B6B')

    # Scatter individual points
    np.random.seed(42)
    if len(train_vals) <= 40:
        ax.scatter(np.random.normal(0, 0.04, len(train_vals)),
                   train_vals, alpha=0.3, s=8, color='#4ECDC4', zorder=3)
    if len(poly_vals) <= 40:
        ax.scatter(np.random.normal(1, 0.04, len(poly_vals)),
                   poly_vals, alpha=0.3, s=8, color='#FF6B6B', zorder=3)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Train (real)', 'PolyInfo\n(predicted)'], fontsize=8)
    ax.set_ylabel('log Conductivity (S/cm)', fontsize=9)
    ax.set_title(f'sim={p["sim"]:.2f}', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add label for the polymer type
    ext = p['ext_smi']
    if len(ext) > 40:
        ext = ext[:37] + '...'
    ax.text(0.5, 0.02, ext, transform=ax.transAxes, fontsize=6,
            ha='center', va='bottom', style='italic', alpha=0.7)

    # n count
    ax.text(0, 0.98, f'n={len(train_vals)}', transform=ax.transAxes,
            fontsize=7, va='top', ha='center', color='#2C8580')
    ax.text(1, 0.98, f'n={len(poly_vals)}', transform=ax.transAxes,
            fontsize=7, va='top', ha='center', color='#CC4444')

legend = [
    mpatches.Patch(color='#4ECDC4', label='Training (real σ)'),
    mpatches.Patch(color='#FF6B6B', label='PolyInfo (predicted σ)')
]
fig.legend(handles=legend, loc='lower center', ncol=2, fontsize=9)

fig.suptitle('Real vs Predicted Conductivity: Training polymers vs their closest PolyInfo analogs',
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'decoder/similar_pairs_comparison.png'),
            dpi=150, bbox_inches='tight')
print(f"Saved: decoder/similar_pairs_comparison.png")
