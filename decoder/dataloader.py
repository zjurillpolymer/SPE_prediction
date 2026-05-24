import os
import pandas as pd
import numpy as np
import torch
from rdkit import RDLogger, Chem
from rdkit.Chem import AllChem, Descriptors
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from encoder import molecule_to_graph

RDLogger.logger().setLevel(RDLogger.ERROR)


def compute_anion_features(salt_smiles):
    """Extract anion from salt SMILES; return [volume, mass, charge]."""
    if pd.isna(salt_smiles) or salt_smiles == '':
        return [0.0, 0.0, 0.0]

    mol = Chem.MolFromSmiles(salt_smiles)
    if mol is None:
        return [0.0, 0.0, 0.0]

    # Find the anion (fragment with negative charge)
    frags = Chem.GetMolFrags(mol, asMols=True)
    anion = None
    for frag in frags:
        if Chem.rdmolops.GetFormalCharge(frag) < 0:
            anion = frag
            break

    if anion is None:
        return [0.0, 0.0, 0.0]

    mass = Descriptors.ExactMolWt(anion)
    charge = float(Chem.rdmolops.GetFormalCharge(anion))

    # 3D conformer for volume
    try:
        mol_3d = Chem.RWMol(anion)
        mol_3d = Chem.AddHs(mol_3d)
        AllChem.EmbedMolecule(mol_3d, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol_3d)
        volume = AllChem.ComputeMolVolume(mol_3d)
    except Exception:
        volume = 0.0

    return [volume, mass, abs(charge)]


class SPEdataset(Dataset):
    EXTRA_FEATURES = ['mw', 'molality',
                      'anion_volume', 'anion_mass', 'anion_charge']

    def __init__(self, df, task='conductivity', cache_path=None):
        self.df = df.reset_index(drop=True)
        self.task = task

        if cache_path and os.path.exists(cache_path):
            print(f"Loading cache: {cache_path}")
            cached = torch.load(cache_path, weights_only=False)
            self.data_list = cached['data_list']
            self.extra_list = cached['extra_list']
            self.inv_temp_list = cached['inv_temp_list']
            self.y_list = cached['y_list']
        else:
            self.data_list, self.extra_list, self.inv_temp_list, self.y_list = self._preprocess()
            if cache_path:
                torch.save({
                    'data_list': self.data_list,
                    'extra_list': self.extra_list,
                    'inv_temp_list': self.inv_temp_list,
                    'y_list': self.y_list,
                }, cache_path)
                print(f"Cache saved: {cache_path}")

    def _preprocess(self):
        data_list, extra_list, inv_temp_list, y_list = [], [], [], []
        fail = 0
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]

            # Polymer graph
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol is None:
                fail += 1
                continue
            data_list.append(molecule_to_graph.mol_to_pyg_graph(mol))

            # Extra features without inv_temp
            extra = [0.0 if pd.isna(row[col]) else row[col]
                     for col in self.EXTRA_FEATURES]
            extra_list.append(torch.tensor(extra, dtype=torch.float))

            # Raw inv_temp for Arrhenius formula
            inv_temp_list.append(torch.tensor(
                [np.nan_to_num(row['inv_temp'], nan=0.0)], dtype=torch.float))

            # Label
            y_list.append(torch.tensor(
                [np.nan_to_num(row[self.task], nan=0.0)], dtype=torch.float))

        if fail:
            print(f"Warning: {fail} invalid SMILES skipped.")
        return data_list, extra_list, inv_temp_list, y_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return (self.data_list[idx].clone(),
                self.extra_list[idx].clone(),
                self.inv_temp_list[idx].clone(),
                self.y_list[idx].clone())


def spe_collate(batch):
    data_list = [item[0] for item in batch]
    extras = torch.stack([item[1] for item in batch], dim=0)   # [B, 5]
    inv_temps = torch.stack([item[2] for item in batch], dim=0) # [B, 1]
    ys = torch.cat([item[3] for item in batch], dim=0)          # [B]
    batch_data = Batch.from_data_list(data_list)
    batch_data.extra = extras
    batch_data.inv_temp = inv_temps
    batch_data.y = ys
    return batch_data
