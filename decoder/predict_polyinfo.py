"""Predict conductivity on PolyInfo datasets using the trained model."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Lipinski import HeavyAtomCount

from encoder import molecule_to_graph
from decoder.dataloader import compute_anion_features
from decoder.spe_prediction import SPE_Predictor, EXTRA_COLS

RDLogger.logger().setLevel(RDLogger.ERROR)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_long_smiles(smile, req_length=30):
    """Extend monomer SMILES to a polymer of req_length heavy atoms."""
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


def predict_polyinfo(polyinfo_path, model, extra_mean, extra_std, y_mean, y_std, out_path):
    """Run prediction on a PolyInfo CSV and save results."""
    df = pd.read_csv(polyinfo_path)
    print(f"Loaded {len(df)} rows from {polyinfo_path}")

    # Extend SMILES
    df['long_smiles'] = df['smiles'].apply(create_long_smiles)
    n_invalid = df['long_smiles'].isna().sum()
    if n_invalid:
        print(f"Warning: {n_invalid} SMILES could not be extended.")
    df = df.dropna(subset=['long_smiles']).reset_index(drop=True)

    # Anion features
    unique_salts = df['salt smiles'].fillna('').unique()
    salt_feat_map = {s: compute_anion_features(s if s else None) for s in unique_salts}
    anion_feats = df['salt smiles'].fillna('').map(salt_feat_map)
    df[['anion_volume', 'anion_mass', 'anion_charge']] = pd.DataFrame(
        anion_feats.tolist(), index=df.index)

    # Temperature → 1/T
    df['inv_temp'] = 1.0 / (df['temperature'] + 273.15)

    # Build graphs
    graphs = []
    extras = []
    fail = 0
    for idx in range(len(df)):
        row = df.iloc[idx]
        mol = Chem.MolFromSmiles(row['long_smiles'])
        if mol is None:
            fail += 1
            continue
        graphs.append(molecule_to_graph.mol_to_pyg_graph(mol))
        extra = [row[col] for col in EXTRA_COLS]
        extras.append(torch.tensor(extra, dtype=torch.float))

    if fail:
        print(f"Warning: {fail} molecules failed graph conversion.")

    # Batch
    for g in graphs:
        g.extra = None  # will be set manually
    batch = Batch.from_data_list(graphs)
    batch.extra = torch.stack(extras, dim=0)

    # Normalize extras
    batch.extra = (batch.extra - torch.tensor(extra_mean)) / torch.tensor(extra_std)

    # Predict
    model.eval()
    batch = batch.to(DEVICE)
    with torch.no_grad():
        pred_norm = model(batch).cpu().numpy()  # normalized

    pred_orig = pred_norm * y_std + y_mean  # original scale (log S/cm)
    df_pred = df.loc[df.index[:len(pred_orig)]].copy()
    df_pred['pred_log_conductivity'] = pred_orig

    # Save
    df_pred.to_csv(out_path, index=False)
    print(f"Predictions saved to {out_path}")
    print(f"Predicted log(σ) range: [{pred_orig.min():.2f}, {pred_orig.max():.2f}]")
    print(f"Mean predicted log(σ): {pred_orig.mean():.2f} ± {pred_orig.std():.2f}")

    # Temperature trend check (first polymer)
    first_poly = df_pred['smiles'].iloc[0]
    sub = df_pred[df_pred['smiles'] == first_poly].sort_values('temperature')
    print(f"\nTemperature trend for {first_poly[:40]}...")
    for _, r in sub[::max(1, len(sub)//5)].iterrows():
        print(f"  T={r['temperature']:5.1f}°C  →  log(σ)={r['pred_log_conductivity']:.3f}")


if __name__ == '__main__':
    # Load model & training stats (need to recompute from full training data)
    train_df = pd.read_csv(os.path.join(ROOT, 'data/clean_train_data.csv'))
    train_df['inv_temp'] = 1.0 / (train_df['temperature'] + 273.15)
    unique_salts = train_df['salt smiles'].fillna('').unique()
    salt_feat_map = {s: compute_anion_features(s if s else None) for s in unique_salts}
    anion_feats = train_df['salt smiles'].fillna('').map(salt_feat_map)
    train_df[['anion_volume', 'anion_mass', 'anion_charge']] = pd.DataFrame(
        anion_feats.tolist(), index=train_df.index)

    extra_mean = train_df[EXTRA_COLS].mean().values.astype(np.float32)
    extra_std = train_df[EXTRA_COLS].std().values.astype(np.float32)
    extra_std[extra_std == 0] = 1.0
    y_mean = train_df['conductivity'].mean().astype(np.float32)
    y_std = train_df['conductivity'].std().astype(np.float32)

    model = SPE_Predictor().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(ROOT, 'decoder/model.pt'), map_location=DEVICE, weights_only=True))
    print(f"Model loaded. Predicting on PolyInfo...\n")

    predict_polyinfo(
        os.path.join(ROOT, 'data/PolyInfo_8salts.csv'),
        model, extra_mean, extra_std, y_mean, y_std,
        os.path.join(ROOT, 'decoder/polyinfo_8salts_pred.csv')
    )
