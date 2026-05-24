"""Compute activation energy Ea for PEG and PP polymers from the trained model."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from encoder import molecule_to_graph
from encoder.MPNN import MPNNEmbedding
from decoder.dataloader import compute_anion_features
from decoder.spe_prediction import SPE_Predictor, EXTRA_COLS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RDLogger.logger().setLevel(RDLogger.ERROR)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
R = 8.314  # J/(mol·K)

# ─── Load training stats ────────────────────────────────
df = pd.read_csv(os.path.join(ROOT, 'data/clean_train_data.csv'))
df['inv_temp'] = 1.0 / (df['temperature'] + 273.15)
unique_salts = df['salt smiles'].fillna('').unique()
salt_map = {s: compute_anion_features(s if s else None) for s in unique_salts}
anion_feats = df['salt smiles'].fillna('').map(salt_map)
df[['anion_volume', 'anion_mass', 'anion_charge']] = pd.DataFrame(
    anion_feats.tolist(), index=df.index)

# Use the same normalization as training
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

# ─── Define target polymers ─────────────────────────────
# PEG: PEO-like chain, same as in training data
pegs = {
    'PEG (short)': '[Cu]C(OCCOCCOCCOC)C[Au]',
    'PEG (med)':   '[Cu]C(OCCOCCOCCOCCOCCOCCOCCOCCOC)C[Au]',
}

# PP: polypropylene monomer from PolyInfo
pps = {
    'PP (monomer)': '[Cu]C(C)[Au]',
}

# Add some common salts and conditions
salts = {
    'LiTFSI': 'O=S([N-]S(=O)(C(F)(F)F)=O)(C(F)(F)F)=O.[Li+]',
    'LiBF4':  'F[B-](F)(F)F.[Li+]',
}

conditions = [
    {'molality': 0.5, 'mw': 5000, 'temperature': 80},
    {'molality': 1.5, 'mw': 5000, 'temperature': 80},
    {'molality': 4.5, 'mw': 5000, 'temperature': 80},
]


def predict_params(smiles, salt_smi, cond):
    """Return (log_sigma, A_norm, EaR_norm, A_phys, Ea_kJmol) for a single condition."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    data = molecule_to_graph.mol_to_pyg_graph(mol)

    # Build extra features
    anion = compute_anion_features(salt_smi)
    inv_t = 1.0 / (cond['temperature'] + 273.15)
    extra_raw = torch.tensor(
        [cond['mw'], cond['molality'], anion[0], anion[1], anion[2]],
        dtype=torch.float)
    extra_norm = (extra_raw - torch.tensor(extra_mean)) / torch.tensor(extra_std)
    data.extra = extra_norm.unsqueeze(0)
    data.inv_temp = torch.tensor([[inv_t]], dtype=torch.float)

    data = data.to(DEVICE)
    with torch.no_grad():
        params = model.regressor(
            torch.cat([model.encoder(data), data.extra], dim=-1))
        A_norm, EaR_norm = float(params[0, 0]), float(params[0, 1])
        log_sigma = A_norm - EaR_norm * inv_t

    # Unnormalize
    log_sigma_phys = log_sigma * y_std + y_mean
    A_phys = y_mean + y_std * A_norm
    ea_kjmol = y_std * EaR_norm * R / 1000  # kJ/mol

    return {
        'log_sigma': log_sigma_phys,
        'A_norm': A_norm,
        'EaR_norm': EaR_norm,
        'A_phys': A_phys,
        'Ea_kJmol': ea_kjmol,
    }


print(f"{'Polymer':25s} {'Salt':12s} {'T(°C)':6s} {'Molality':9s} "
      f"{'logσ':8s} {'Ea(kJ/mol)':11s} {'A_phys':8s}")
print("=" * 80)

for poly_name, smi in {**pegs, **pps}.items():
    for salt_name, salt_smi in salts.items():
        for cond in conditions:
            res = predict_params(smi, salt_smi, cond)
            if res is None:
                continue
            print(f"{poly_name:25s} {salt_name:12s} "
                  f"{cond['temperature']:3d}     "
                  f"{cond['molality']:4.1f}       "
                  f"{res['log_sigma']:+7.3f}  "
                  f"{res['Ea_kJmol']:8.2f}     "
                  f"{res['A_phys']:+7.2f}")
    print()
