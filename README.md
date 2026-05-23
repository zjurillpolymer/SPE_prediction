# SPE Prediction

Polymer electrolyte conductivity prediction using graph neural networks.

## Data

Data from [ChemPropPred](https://github.com/learningmatter-mit/Chem-prop-pred) — polymer electrolyte conductivity measurements including SMILES, conductivity (log S/cm), molecular weight, salt concentration, and temperature.

- `data/clean_train_data.csv` — cleaned training data
- `data/PolymerElectrolyteData.csv` — full raw dataset
- `data/PolyInfo_8salts.csv` / `data/polyinfo_5salts_4conc.csv` — PolyInfo polymer database with salt additives

## Project Structure

```
encoder/
├── molecule_to_graph.py   # RDKit → PyG graph conversion (atom & bond features)
├── MPNN.py                # MPNN embedding layer (SMILES → 128-dim vector)
├── dataloader.py          # Dataset & DataLoader
└── tox_pre.py
```

## Usage

The MPNN embedding layer takes a monomer SMILES and outputs a 128-dimensional vector:

```python
from encoder.MPNN import MPNNEmbedding

model = MPNNEmbedding(node_in_dims=31, edge_dim=6, hidden_dim=128, num_layers=6)
embedding = model(data)  # shape: [batch_size, 128]
```

## Molecular Graph Features

| Feature | Dimension | Description |
|---------|-----------|-------------|
| Atom features | 31 | atomic number, degree, formal charge, H count, hybridization, aromaticity, ring membership, chirality |
| Bond features | 6 | bond type (single/double/triple/aromatic), conjugation, ring membership |
