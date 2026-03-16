import sys
import warnings
import argparse
from pathlib import Path

warnings.filterwarnings("ignore", message="xFormers is not available")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

REPO_ROOT     = Path(__file__).parent.resolve()
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
PATCH_SIZE    = 14
FEAT_DIM      = 1536


def _ceil14(n):
    return ((n + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE

def pad_collate(batch):
    tensors, labels = zip(*batch)
    max_h = _ceil14(max(t.shape[1] for t in tensors))
    max_w = _ceil14(max(t.shape[2] for t in tensors))
    padded = [F.pad(t, (0, max_w - t.shape[2], 0, max_h - t.shape[1])) for t in tensors]
    return torch.stack(padded), list(labels)


class DinoBloomClassifier(nn.Module):
    def __init__(self, backbone, num_classes, feat_dim=FEAT_DIM):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


class ValDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (64, 64), (128, 128, 128))
        return self.tf(img), label


def load_model(ckpt_path, device):
    sys.path.insert(0, str(REPO_ROOT))
    from dinov2.hub.backbones import dinov2_vitg14
    ckpt         = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    num_classes  = ckpt["num_classes"]
    class_to_idx = ckpt["class_to_idx"]
    backbone     = dinov2_vitg14(pretrained=False, img_size=224)
    model        = DinoBloomClassifier(backbone, num_classes).to(device)
    state        = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, class_to_idx


def compute_metrics(all_true, all_pred, classes):
    per = {}
    for cls in classes:
        tp    = sum(1 for t, p in zip(all_true, all_pred) if t == cls and p == cls)
        fp    = sum(1 for t, p in zip(all_true, all_pred) if t != cls and p == cls)
        fn    = sum(1 for t, p in zip(all_true, all_pred) if t == cls and p != cls)
        total = sum(1 for t in all_true if t == cls)
        prec  = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
        rec   = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
        f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc   = tp / total * 100 if total > 0 else 0.0
        per[cls] = (acc, prec, rec, f1, tp, fp, fn, total)
    macro_p  = sum(v[1] for v in per.values()) / len(per)
    macro_r  = sum(v[2] for v in per.values()) / len(per)
    macro_f1 = sum(v[3] for v in per.values()) / len(per)
    return per, macro_p, macro_r, macro_f1


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint",  default=str(REPO_ROOT / "bloom_leukemia.pth"))
parser.add_argument("--val-txt",     default=str(REPO_ROOT / "New Data" / "val.txt"))
parser.add_argument("--batch-size",  type=int, default=32)
parser.add_argument("--workers",     type=int, default=4)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice     : {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
print(f"Checkpoint : {args.checkpoint}")
print(f"Val split  : {args.val_txt}\n")

SKIP = {"C-NMC_test_final_phase_data", "C-NMC_test_prelim_phase_data",
        "testing_data", "training_data"}

samples = []
with open(args.val_txt) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        p     = REPO_ROOT / line
        label = Path(line).parent.name
        if label in SKIP:
            continue
        samples.append((p, label))

print(f"Val images : {len(samples):,}")
from collections import defaultdict
counts = defaultdict(int)
for _, l in samples:
    counts[l] += 1
for cls in sorted(counts):
    print(f"  {cls:<12} {counts[cls]:>5,}")
print()

model, class_to_idx = load_model(args.checkpoint, device)
idx_to_class = {v: k for k, v in class_to_idx.items()}
print(f"Classes    : {len(class_to_idx)}\n")

ds     = ValDataset(samples)
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, collate_fn=pad_collate,
                    pin_memory=(device.type == "cuda"))

all_true, all_pred = [], []
print("Evaluating...")
with torch.no_grad():
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
            logits = model(imgs)
        preds = logits.argmax(1).cpu().tolist()
        for pred_idx, true_label in zip(preds, labels):
            all_true.append(true_label)
            all_pred.append(idx_to_class[pred_idx])
        if (batch_idx + 1) % 20 == 0:
            done = min((batch_idx + 1) * args.batch_size, len(samples))
            print(f"  {done:>5,} / {len(samples):,}", flush=True)

eval_classes = sorted(counts.keys())
per, macro_p, macro_r, macro_f1 = compute_metrics(all_true, all_pred, eval_classes)

total   = len(all_true)
correct = sum(1 for t, p in zip(all_true, all_pred) if t == p)
acc     = correct / total * 100

W   = 75
SEP = "=" * W
S2  = "-" * W

print()
print(SEP)
print(f"  EXTENDED DINOBLOOM EVAL  ({total:,} images, {len(eval_classes)} classes)")
print(S2)
print(f"  {'Metric':<14}  {'Value':>9}")
print(f"  {'-'*14}  {'-'*9}")
print(f"  {'Accuracy':<14}  {acc:>8.2f}%")
print(f"  {'Macro Prec':<14}  {macro_p:>8.2f}%")
print(f"  {'Macro Recall':<14}  {macro_r:>8.2f}%")
print(f"  {'Macro F1':<14}  {macro_f1:>8.2f}%")
print(S2)
print(f"  {'Class':<12}  {'Acc':>7}  {'Prec':>7}  {'Recall':>7}  {'F1':>7}  {'N':>6}")
print(f"  {'-'*12}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}")
for cls in eval_classes:
    a, p, r, f, tp, fp, fn, tot = per[cls]
    print(f"  {cls:<12}  {a:>6.1f}%  {p:>6.1f}%  {r:>6.1f}%  {f:>6.1f}%  {tot:>6,}")
print(S2)
print(f"  {'MACRO':<12}  {'':>7}  {macro_p:>6.1f}%  {macro_r:>6.1f}%  {macro_f1:>6.1f}%")
print(SEP)
print()
print(f"  Paper row (Extended DinoBloom):")
print(f"  Accuracy={acc:.2f}%  Macro P={macro_p:.1f}%  Macro R={macro_r:.1f}%  Macro F1={macro_f1:.1f}%")
print()
