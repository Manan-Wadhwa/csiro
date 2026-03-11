# 🌿 CSIRO Image2Biomass Prediction

A machine learning solution for the **CSIRO – Image2Biomass Prediction** Kaggle competition. The goal is to predict pasture biomass components from high‑resolution field images plus tabular metadata, evaluated with a **globally weighted R²** metric.

## What’s in this repo

- `csiro0.ipynb` — **Hybrid “gold-standard” pipeline**  
  DINOv2 + SigLIP (split-stream left/right) + engineered metadata + color statistics → CatBoost regressors.
- `csiro2.ipynb` — **Final ensemble notebook**  
  Patch-based SigLIP embeddings + semantic text-probing features + classical ML ensemble, combined with a DINO-based direct predictor, with final biological consistency constraints.

## Targets

The notebooks are set up to predict:

- `Dry_Green_g`
- `Dry_Dead_g`
- `Dry_Clover_g`
- `GDM_g`
- `Dry_Total_g`

## Pipeline highlights

### `csiro0.ipynb` (DINOv2 + SigLIP + Metadata → CatBoost)
- Metadata engineering:
  - `Sampling_Date` → day-of-year + cyclic month encoding
  - One-hot encoding: `Species`, `State`
  - Interaction: `Pre_GSHH_NDVI × Height_Ave_cm` (`Volume_Proxy`)
- Visual features:
  - Split each image into **left/right halves**
  - Extract embeddings with:
    - DINOv2 @ 518×518
    - SigLIP @ 384×384
- Extra image descriptors:
  - RGB mean/std, Excess Green (ExG), Green/Red ratio
- Model:
  - CatBoostRegressor per target with KFold CV
- Post-processing:
  - Non-negativity clamp
  - Blend `Dry_Total_g` with sum of components

### `csiro2.ipynb` (Ensemble + semantic probing)
- SigLIP patch embeddings:
  - Split images into overlapping patches (e.g., 520px with overlap) and average embeddings
- Semantic feature generation (text probing):
  - Similarity to prompt groups like `green`, `dead`, `dense`, `bare`, `clover`, etc.
  - Derived ratios (e.g., greenness, cover)
- Classical regressors + ensembling:
  - CatBoost / LightGBM / scikit-learn models (as configured in notebook)
- Final-stage constraint:
  - Enforce biomass mass-balance consistency once at the end

## How to run

### On Kaggle (recommended)
1. Create a Kaggle Notebook and enable **GPU**.
2. Add the CSIRO Biomass dataset via **“+ Add Data”**.
3. Upload and run either notebook end-to-end.
4. The notebooks include checks/auto-detection for dataset paths under `/kaggle/input/`.

### Locally
1. Install dependencies (rough guide):
   - `torch`, `timm`, `transformers`
   - `catboost`, `lightgbm`, `scikit-learn`
   - `opencv-python`, `pandas`, `numpy`, `tqdm`, `Pillow`
2. Update the dataset root paths in the config blocks:
   - In `csiro0.ipynb`: `CFG.ROOT`
   - In `csiro2.ipynb`: `cfg.DATA_PATH`, `cfg.SPLIT_PATH`, and any model paths if needed

## Notes
- GPU memory can be a bottleneck; model sizes and batch sizes are configurable.
- The notebooks are written to be competition-aligned and reproducible (fixed seeds, CV).

---
