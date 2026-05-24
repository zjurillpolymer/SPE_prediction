"""Similarity analysis between training and PolyInfo polymers."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, AllChem
from rdkit.Chem.Lipinski import HeavyAtomCount
from scipy.spatial.distance import cdist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RDLogger.logger().setLevel(RDLogger.ERROR)

N_BITS = 128
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=N_BITS)


def create_long_smiles(smile, req_length=30):
    """Extend monomer SMILES (same as predict_polyinfo.py)."""
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


# ─── Load data ──────────────────────────────────────────
train_df = pd.read_csv(os.path.join(ROOT, 'data/clean_train_data.csv'))
poly_df = pd.read_csv(os.path.join(ROOT, 'decoder/polyinfo_8salts_pred.csv'))

# Unique polymers in each set
train_smiles = train_df['smiles'].dropna().unique()
poly_smiles_raw = poly_df['smiles'].dropna().unique()

print(f"Training unique polymers: {len(train_smiles)}")
print(f"PolyInfo unique polymers: {len(poly_smiles_raw)}")

# Extend PolyInfo SMILES
poly_smiles = []
for s in poly_smiles_raw:
    ext = create_long_smiles(s)
    if ext:
        poly_smiles.append(ext)
    else:
        poly_smiles.append(s)

print(f"PolyInfo after extension: {len(poly_smiles)}")

# ─── Compute Morgan fingerprints ────────────────────────
def get_fp(smiles_list):
    fps = []
    valid = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            fps.append(list(mfpgen.GetFingerprint(mol)))
            valid.append(s)
    return np.array(fps, dtype=np.float32), valid

print("Computing fingerprints...")
train_fps, train_valid = get_fp(train_smiles.tolist())
poly_fps, poly_valid = get_fp(poly_smiles)
print(f"  Training: {len(train_valid)} valid")
print(f"  PolyInfo: {len(poly_valid)} valid")

# ─── Nearest neighbor similarity ────────────────────────
# Tanimoto = 1 - Rogers-Tanimoto distance
print("Computing pairwise similarities (this may take a moment)...")
dists = cdist(poly_fps, train_fps, metric='rogerstanimoto')
sims = 1 - dists                       # Tanimoto similarity
best_sim = sims.max(axis=1)            # max similarity to any training polymer
best_idx = sims.argmax(axis=1)         # index of most similar training polymer

# ─── Stats ──────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Similarity between PolyInfo polymers and training set:")
print(f"  Mean best Tanimoto: {best_sim.mean():.3f}")
print(f"  Median:             {np.median(best_sim):.3f}")
print(f"  Min:                {best_sim.min():.3f}")
print(f"  Max:                {best_sim.max():.3f}")

pct_good = (best_sim >= 0.4).mean() * 100
pct_poor = (best_sim < 0.2).mean() * 100
print(f"  % with good coverage (sim≥0.4): {pct_good:.0f}%")
print(f"  % with poor coverage (sim<0.2): {pct_poor:.0f}%")

# ─── Show extremes ──────────────────────────────────────
order = np.argsort(best_sim)
print(f"\n{'='*60}")
print("Most novel PolyInfo polymers (lowest sim to training):")
for i in order[:5]:
    print(f"  sim={best_sim[i]:.3f}  {poly_valid[i][:60]}")
    nn = train_valid[best_idx[i]]
    print(f"    nearest train: {nn[:60]}")

print(f"\nMost similar PolyInfo polymers (highest sim to training):")
for i in order[-5:]:
    print(f"  sim={best_sim[i]:.3f}  {poly_valid[i][:60]}")
    nn = train_valid[best_idx[i]]
    print(f"    nearest train: {nn[:60]}")

# ─── Plot ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].hist(best_sim, bins=40, color='steelblue', edgecolor='white')
axes[0].axvline(0.4, color='red', ls='--', label='sim=0.4')
axes[0].set_xlabel('Tanimoto similarity to nearest training polymer')
axes[0].set_ylabel('Number of PolyInfo polymers')
axes[0].set_title('Distribution of similarity')
axes[0].legend()

# Similarity vs prediction std (prediction uncertainty)
# Group PolyInfo predictions by polymer, compute std
poly_preds = poly_df.groupby('smiles')['pred_log_conductivity'].agg(['mean', 'std'])
if len(poly_preds) == len(best_sim):
    axes[1].scatter(best_sim, poly_preds['std'].values, alpha=0.3, s=8, c='steelblue')
    axes[1].set_xlabel('Tanimoto similarity')
    axes[1].set_ylabel('Std of predictions per polymer')
    axes[1].set_title('Similarity vs prediction variance')

plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'decoder/similarity_analysis.png'), dpi=150, bbox_inches='tight')
print(f"\nPlot saved to decoder/similarity_analysis.png")
