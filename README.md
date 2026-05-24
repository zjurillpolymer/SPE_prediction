# SPE Prediction

Polymer electrolyte conductivity prediction using graph neural networks with physical constraints.

Predict ionic conductivity (log S/cm) of solid polymer electrolytes from polymer SMILES, salt chemistry, molecular weight, and temperature.

## Key Findings

### 1. Arrhenius Physical Constraint Works
Model outputs Arrhenius parameters (A, Ea/R) and applies `log σ = A − (Ea/R)·(1/T)` as a hard-coded physical formula. The learned activation energies are physically meaningful:
- **PEG (PEO)**: Ea ≈ **18 kJ/mol** — matches literature values (10–30 kJ/mol)
- **PP**: Ea ≈ **23 kJ/mol** — higher, consistent with non-polar backbone
- Ea is invariant to salt type and concentration, as physics requires

### 2. Anion Design: Charge > Mass > Volume
SHAP analysis on anion features (from RDKit 3D conformers + formal charge):
| Feature | Importance |
|---------|-----------|
| Anion charge | 0.141 |
| Anion mass | 0.098 |
| Anion volume | 0.071 |

Charge dominates — Li⁺-anion Coulomb interaction is the primary factor, not steric volume.

### 3. Flexibility + Polarity Synergy
Polymers with both high flexibility (rotatable bonds / heavy atom) and high polarity (O/N atom ratio) achieve the highest conductivity. PEO-based oligomers (short PEG chains + LiTFSI) dominate the top performers, with log σ as high as **−3.0** vs the dataset mean of **−5.4**.

Faceted analysis shows:
| Polarity | Flexibility slope (vs log σ) | R² |
|----------|------------------------------|-----|
| Low O/N | +2.59 | 0.121 |
| Mid O/N | +2.29 | 0.084 |
| High O/N | +2.54 | **0.202** |

Flexibility only translates to conductivity when sufficient Li⁺ coordination sites (O/N atoms) are present.

### 4. Key Differentiator: Salt Presence
91% of outperforming polymers have added lithium salt, vs only 43% of underperformers. Polymers relying solely on Cu/Au counterions (no added salt) consistently underperform.

### 5. PolyInfo Prediction
19,704 conductivity predictions across 819 polymers × 8 lithium salts. Range: log σ ∈ [−10.64, −0.28]. All 819 polymers have Tanimoto similarity ≥ 0.4 to the training set (mean 0.58), confirming reasonable extrapolation.

## Performance

| Setting | Test loss (original scale) |
|---------|---------------------------|
| Without anion features | 0.65 |
| With anion features | 0.49 |
| **Arrhenius output layer** | **0.54** |

## Project Structure

```
encoder/
├── MPNN.py                 # MPNN embedding layer (MessagePassing)
└── molecule_to_graph.py    # RDKit → PyG graph conversion

decoder/
├── dataloader.py           # Dataset + anion feature extraction
├── spe_prediction.py       # Training pipeline with Arrhenius output
├── predict_polyinfo.py     # PolyInfo conductivity prediction
└── compute_ea.py           # Activation energy computation

figures/
├── plot_arrhenius.py       # log σ vs 1/T (Arrhenius plot)
├── plot_flexibility.py     # Flexibility vs conductivity
├── plot_flexibility_faceted.py  # Faceted by polarity
├── plot_mw_relation.py     # Conductivity vs molecular weight
├── plot_similar_pairs.py   # Training vs PolyInfo comparison
├── analyze_shap.py         # SHAP feature importance
├── analyze_similarity.py   # Chemical similarity analysis
├── analyze_outliers.py     # Outlier pattern analysis
└── *.png                   # Generated figures
```

## Usage

```bash
# Train
python decoder/spe_prediction.py

# Predict on PolyInfo
python decoder/predict_polyinfo.py

# Compute Ea for selected polymers
python decoder/compute_ea.py

# Generate figures
python figures/plot_arrhenius.py
python figures/plot_flexibility_faceted.py
```

## Requirements

torch, torch_geometric, rdkit, pandas, numpy, scipy, shap, matplotlib
