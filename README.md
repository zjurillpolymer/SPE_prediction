# SPE Prediction

Polymer electrolyte conductivity prediction using graph neural networks.

Predict ionic conductivity (log S/cm) of solid polymer electrolytes from polymer SMILES, salt concentration, molecular weight, and temperature.

## Data

Data from [ChemPropPred](https://github.com/learningmatter-mit/Chem-prop-pred).

| File | Description | Rows |
|------|-------------|------|
| `data/clean_train_data.csv` | Cleaned training set (229 unique polymers) | 10,114 |
| `data/PolymerElectrolyteData.csv` | Full raw experimental data | 16,009 |
| `data/PolyInfo_8salts.csv` | PolyInfo polymer database × 8 salts (no labels) | 19,704 |
| `data/polyinfo_5salts_4conc.csv` | PolyInfo × 5 salts × 4 concentrations | 32,840 |

## Features

**Polymer structure**: SMILES → RDKit molecule → PyG graph with 31 atom features + 6 bond features → MPNN → 128-dim embedding.

**Anion features**: Extracted from salt SMILES (Li⁺ removed), 3 features per anion:
- Volume (Å³) — from 3D conformer
- Exact mass (g/mol)
- Formal charge

**Extra features**: MW, molality, 1/T (Arrhenius form), + 3 anion features = 6 total.

## Model Architecture

```
SMILES → MPNNEmbedding (6× MPNNEdgeLayer) → 128-dim → concat → Linear(134→64) → ReLU → Dropout → Linear(64→1) → log σ
                                                    ↑
                                          [mw, molality, 1/T, 
                                           anion_volume, anion_mass, anion_charge]
```

## Performance

| Setting | Test loss (original scale) |
|---------|---------------------------|
| Without anion features | 0.65 |
| With anion features | **0.49** (-24%) |

## Key Findings

- **Anion charge matters most**: SHAP analysis shows charge > mass > volume for anion contribution
- **Molality sweet spot**: medium concentration (1.5 mol/kg) often outperforms 4.5 mol/kg
- **PolyInfo prediction**: 19,704 predictions across 819 polymers × 8 salts, log σ range [-9.22, -1.35]
- **Coverage**: 100% of PolyInfo polymers have Tanimoto similarity ≥ 0.4 to training set (mean 0.58)

## Project Structure

```
encoder/
├── molecule_to_graph.py    # RDKit → PyG graph conversion
├── MPNN.py                 # MPNN embedding layer
decoder/
├── dataloader.py           # Dataset + anion feature extraction
├── spe_prediction.py       # Training pipeline
├── analyze_shap.py         # SHAP feature importance
├── analyze_similarity.py   # Chemical similarity analysis (PolyInfo vs training)
├── predict_polyinfo.py     # PolyInfo conductivity prediction
└── plot_similar_pairs.py   # Real vs predicted comparison
```

## Usage

```python
from encoder.MPNN import MPNNEmbedding
from decoder.spe_prediction import SPE_Predictor

model = SPE_Predictor()
# model.load_state_dict(torch.load('decoder/model.pt'))
# output = model(batch)  # [batch_size] → log conductivity
```

Run training:
```bash
python decoder/spe_prediction.py
```

Predict on PolyInfo:
```bash
python decoder/predict_polyinfo.py
```

## Requirements

torch, torch_geometric, rdkit, pandas, numpy, shap, matplotlib
