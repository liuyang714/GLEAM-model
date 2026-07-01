# GLEAM

GLEAM is a multimodal deep learning framework for automated nephropathology diagnosis, jointly incorporating multi-stain light microscopy WSIs, immunofluorescence reports, and clinical variables. GLEAM builds upon several pretrained models: **YOLOv8** for glomeruli detection, **CONCH** (ViT-B/16) for histopathology feature extraction, and **Med-BERT** (Chinese) for immunofluorescence report encoding.

> **Note:** This work is currently under peer review.

---

## Data

**The training data are private and not publicly available due to legal, ethical, and privacy constraints.** Access can be requested via a reasonable research proposal.

The model expects the following inputs per patient:

### partition.csv (train/val split)

| train | train_label | val | val_label |
|-------|------------|-----|-----------|
| 001   | 0          | 050 | 2         |
| 002   | 3          | 051 | 0         |

### text.xlsx (IF reports + clinical variables)

| ID | immunofluorescence_report | age | sex   |
|----|--------------------------|-----|-------|
| 001| IgAN 2+, C3 1+, IgG 1+   | 45  | male  |
| 002| C3 2+, C1q 1+           | 62  | female|

### .pt feature files

One `.pt` file per patient, containing glomerular features extracted by CONCH:
- Shape: `[N, 1024, 768]` (N = number of glomeruli)
- Filename: `{patient_id}.pt`

---

## Usage

### Step 1: Glomerulus Detection

```bash
python 1segment-glomerulus/3restore_extract.py
```

### Step 2: Feature Extraction

```bash
python 2feature_extraction/extract_features.py
```

### Step 3: Classification

```bash
# Train
python -m 3classification.train_gaan --config 3classification/config.yaml
python -m 3classification.train_bert --config 3classification/config.yaml
python -m 3classification.train_fusion --config 3classification/config.yaml

# Predict
python -m 3classification.predict_fusion \
  --config 3classification/config.yaml \
  --input_excel /path/to/test.xlsx \
  --output_excel /path/to/result.xlsx \
  --image_feat_dir /path/to/features \
  --fusion_model_path /path/to/best_fusion.pt

# MEST-C component classification (IgAN grading)
python -m 3classification.MEST_C.train --task M --config 3classification/config.yaml
python -m 3classification.MEST_C.predict --task M --model_path /path/to/best_M.pt \
  --input_excel /path/to/test.xlsx --output_csv /path/to/result.csv --pt_dir /path/to/features
```

---

## Citation

Please cite the corresponding manuscript once published.
