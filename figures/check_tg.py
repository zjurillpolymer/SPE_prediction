import pandas as pd
import numpy as np

import sys, os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw = pd.read_csv(os.path.join(root, 'data/PolymerElectrolyteData.csv'),
                  low_memory=False)
train = pd.read_csv(os.path.join(root, 'data/clean_train_data.csv'))

raw['Tg_num'] = pd.to_numeric(raw['Tg (oC)'], errors='coerce')
print(f"Rows with Tg: {raw['Tg_num'].notna().sum()} / {len(raw)}")
print(f"Tg range: {raw['Tg_num'].min():.1f} ~ {raw['Tg_num'].max():.1f} °C")

# Check if clean_train SMILES appear in the raw data somehow
print(f"\nUnique clean_train SMILES: {train['smiles'].nunique()}")
print(f"Unique raw 'Compound Notebook Name': {raw['Compound Notebook Name'].nunique()}")

# Check if raw has a SMILES column we can match on
smi_cols = [c for c in raw.columns if 'smiles' in c.lower() or 'smiles' in c.lower()]
print(f"\nSMILES-related columns: {smi_cols}")

# Look at the first few rows with Tg to see data format
with_tg = raw[raw['Tg_num'].notna()]
print(f"\nFirst 3 rows with Tg:")
print(with_tg[['Compound Notebook Name', 'Tg_num', 'Temperature (oC)']].head(3).to_string())
