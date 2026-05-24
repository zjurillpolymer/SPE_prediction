"""Analyze outliers: polymers that outperform their flexibility + polarity."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.Lipinski import HeavyAtomCount

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RDLogger.logger().setLevel(RDLogger.ERROR)

df = pd.read_csv(os.path.join(ROOT, 'data/clean_train_data.csv'))

# Compute features per polymer
records = []
for smi, grp in df.groupby('smiles'):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    n_heavy = HeavyAtomCount(mol)
    n_rot = Descriptors.NumRotatableBonds(mol)
    rot_frac = n_rot / max(n_heavy, 1)
    n_o = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
    n_n = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
    n_f = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)
    n_s = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16)
    on_ratio = (n_o + n_n) / max(n_heavy, 1)

    # Salt info
    has_salt = grp['salt smiles'].notna().any()
    salt_types = grp['salt smiles'].dropna().unique().tolist()

    records.append({
        'smiles': smi,
        'rot_frac': rot_frac,
        'on_ratio': on_ratio,
        'mean_cond': grp['conductivity'].mean(),
        'max_cond': grp['conductivity'].max(),
        'n_rows': len(grp),
        'has_salt': has_salt,
        'salt_types': salt_types[:3],
        'n_heavy': n_heavy,
        'n_o': n_o, 'n_n': n_n, 'n_f': n_f, 'n_s': n_s,
        'has_Cu': 'Cu' in smi,
        'has_Au': 'Au' in smi,
        'temp_min': grp['temperature'].min(), 'temp_max': grp['temperature'].max(),
        'mw_min': grp['mw'].min(), 'mw_max': grp['mw'].max(),
    })

pdf = pd.DataFrame(records)

# Predict conductivity from flexibility alone (linear fit)
x, y = pdf['rot_frac'], pdf['mean_cond']
coefs = np.polyfit(x, y, 1)
pdf['pred_from_flex'] = np.polyval(coefs, x)
pdf['residual'] = pdf['mean_cond'] - pdf['pred_from_flex']

# Top outliers (much better than expected)
outliers = pdf.nlargest(15, 'residual')

print(f"{'='*100}")
print(f"TOP 15 OUTLIERS — polymers that outperform their flexibility prediction")
print(f"{'='*100}")
print(f"\n{'SMILES':55s} {'rot_frac':>8s} {'O/N':>5s} {'mean_cond':>9s} {'predicted':>9s} "
      f"{'residual':>8s} {'O':>3s} {'F':>3s} {'S':>3s} {'salt':>5s} {'n':>4s}")
print('-' * 105)

for _, r in outliers.iterrows():
    smi = r['smiles'][:50]
    print(f"{smi:55s} {r['rot_frac']:8.3f} {r['on_ratio']:5.2f} {r['mean_cond']:+9.3f} "
          f"{r['pred_from_flex']:+9.3f} {r['residual']:+8.3f} "
          f"{r['n_o']:3d} {r['n_f']:3d} {r['n_s']:3d} {str(r['has_salt']):>5s} {r['n_rows']:4d}")

# Analyze what's special
print(f"\n{'='*100}")
print(f"PATTERN ANALYSIS")
print(f"{'='*100}")

up = pdf[pdf['residual'] > 0]
down = pdf[pdf['residual'] < 0]

print(f"\nUpside (n={len(up)}): avg O={up['n_o'].mean():.1f} F={up['n_f'].mean():.1f} S={up['n_s'].mean():.1f}")
print(f"Downside (n={len(down)}): avg O={down['n_o'].mean():.1f} F={down['n_f'].mean():.1f} S={down['n_s'].mean():.1f}")
print(f"Upside has salt: {up['has_salt'].mean()*100:.0f}%")
print(f"Downside has salt: {down['has_salt'].mean()*100:.0f}%")

# Check if certain functional groups drive outliers
print(f"\nOutlier polymer motifs:")
for _, r in outliers.iterrows():
    smi = r['smiles']
    features = []
    if 'S(=O)(=O)' in smi: features.append('sulfonyl')
    if 'C(=O)' in smi: features.append('carbonyl')
    if 'F' in smi.replace('[Au]','').replace('[Cu]','') and r['n_f'] > 3:
        features.append(f'fluorinated({r["n_f"]}F)')
    if r['has_salt']: features.append(f'salt:{r["salt_types"][0][:20]}')
    if r['n_heavy'] < 15: features.append('small molecule')
    print(f"  residual={r['residual']:+5.2f}  {' | '.join(features)}")
