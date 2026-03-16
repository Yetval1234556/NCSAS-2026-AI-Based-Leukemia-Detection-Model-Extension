# NCSAS 2026 — AI-Based Leukemia Detection: Model Extension

Fine-tuning [DinoBloom-G](https://github.com/marrlab/DinoBloom) (ViT-G/14, 1.1B parameters) for leukemia subtype classification, with cross-institutional generalization experiments and knowledge retention analysis.

---

## Overview

Standard CNNs trained on a single institution's data suffer severe accuracy drops when tested across institutions (~23 pp in our experiments). This project fine-tunes DinoBloom-G — a foundation model pretrained on 380K hematology images — on multi-domain leukemia data to achieve robust performance.

| Model | Accuracy |
|---|---|
| Linear probe (frozen DinoBloom-G) | 89.41% |
| Fine-tuned DinoBloom-G (ours, 75 epochs) | **98.86%** |
| EfficientNet-B0 in-distribution | 95.12% |
| EfficientNet-B0 cross-institution | 71.85% |

---

## Scripts

### `train_efficientnet_b0.py`
Trains an EfficientNet-B0 classifier on leukemia cell images using an ImageNet-pretrained backbone with a trainable classification head. Supports partial unfreezing of the last N feature blocks.

**Key flags:**

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 30 | Training epochs |
| `--lr` | 1e-4 | Learning rate |
| `--unfreeze-blocks` | 2 | Feature blocks to unfreeze (0–9) |
| `--batch-size` | 32 | Batch size |

---

### `single_institution_cnn.py`
Establishes the cross-institutional generalization baseline. Trains an EfficientNet-B0 head on C-NMC (India) and evaluates on Taleqani Hospital (Iran).

| Split | Accuracy |
|---|---|
| In-distribution (C-NMC) | 95.12% |
| Cross-institution (Taleqani) | 71.85% |
| Drop | −23.27 pp |

Outputs to `single_institution_results.txt`.

---

### `linear_probe.py`
Evaluates the general WBC recognition capability of the original DinoBloom-G backbone by training a frozen linear classifier (L-BFGS) on extracted 1536-dim features. Serves as the pre-fine-tuning baseline.

```
python linear_probe.py --backbone DinoBloom-G.pth --data-dir <path>
```

Outputs to `linear_probe_results.txt`.

---

### `compare_retention.py`
Measures how much general hematological knowledge is retained after fine-tuning. Runs a 5-NN classifier on features from both the original DinoBloom-G and the fine-tuned backbone across 21+ WBC cell types.

```
python compare_retention.py --original DinoBloom-G.pth --finetuned checkpoint_latest.pth
```

Outputs to `retention_<timestamp>.txt`.

---

### `eval_val.py`
Evaluates a trained checkpoint on the validation split, reporting per-class and macro precision, recall, and F1.

```
python eval_val.py --checkpoint bloom_leukemia.pth --workers 0
```

---

### `test_model.py`
Evaluates a trained checkpoint at both the image level and the patient level (majority vote). Requires `patient_metadata.csv` mapping patient IDs to diagnoses.

```
python test_model.py --checkpoint bloom_leukemia.pth --data-dir <path>
```

Outputs to `test_results_<timestamp>.txt`.

---

### `plot_training.py`
Reads `training_metrics.csv` and generates 4 separate training curve plots: loss, accuracy, train-val gap, and learning rate schedule.

```
python plot_training.py
```

---

## Model Architecture

```
DinoBloom-G (ViT-G/14, pretrained on 380K hematology images)
  Transformer blocks 0 to N-3   [FROZEN]
  Transformer blocks N-2 to N   [TRAINABLE]
        |
        Classification Head
              LayerNorm(1536)
              Linear(1536 -> 512)
              GELU + Dropout(0.2)
              Linear(512 -> num_classes)
```

Training setup: AdamW, CosineAnnealingLR, label smoothing 0.1, bfloat16 AMP, 75 epochs

---

## Requirements

- Python 3.10+, PyTorch 2.x with CUDA
- torchvision, Pillow, matplotlib
- DinoBloom-G.pth (original pretrained weights)
- dinov2/ package in repo root
