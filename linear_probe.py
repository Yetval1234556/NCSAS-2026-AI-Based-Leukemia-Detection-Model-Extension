import sys
import os
import warnings
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

warnings.filterwarnings("ignore", message="xFormers is not available")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

REPO_ROOT     = Path(__file__).parent.resolve()
VALID_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
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


class CellDataset(Dataset):
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


def discover_samples(data_dir: Path):
    STRUCTURAL = {"All", "Signed slides", "Unsigned slides"}
    by_class = defaultdict(list)
    for root, _, files in os.walk(data_dir):
        root_path = Path(root)
        if root_path.name in STRUCTURAL:
            continue
        for fname in files:
            if Path(fname).suffix.lower() in VALID_EXTS:
                label = root_path.name
                if label not in STRUCTURAL:
                    by_class[label].append((root_path / fname, label))
    samples = []
    for items in by_class.values():
        samples.extend(items)
    return samples


def load_backbone(pth_path: str, device):
    sys.path.insert(0, str(REPO_ROOT))
    from dinov2.hub.backbones import dinov2_vitg14
    model = dinov2_vitg14(pretrained=False, img_size=224)
    raw   = torch.load(pth_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        for key in ("teacher", "model", "state_dict"):
            if key in raw:
                raw = raw[key]; break
    cleaned = {}
    for k, v in raw.items():
        for prefix in ("backbone.", "module.", "encoder."):
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        cleaned[k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"  Backbone loaded | missing={len(missing)}  unexpected={len(unexpected)}")
    return model.eval().to(device)


@torch.no_grad()
def extract_features(backbone, loader, device):
    feats, labels = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
            f = backbone(imgs)
        feats.append(f.float().cpu())
        labels.extend(lbls)
    return torch.cat(feats, dim=0), labels


def train_linear_probe(feats, labels, num_classes, device):
    label_set = sorted(set(labels))
    l2i = {l: i for i, l in enumerate(label_set)}
    y   = torch.tensor([l2i[l] for l in labels], dtype=torch.long).to(device)
    X   = feats.to(device)

    head = nn.Linear(FEAT_DIM, num_classes).to(device)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)

    optimizer = torch.optim.LBFGS(head.parameters(), lr=0.1, max_iter=100,
                                   line_search_fn="strong_wolfe")
    ce = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = ce(head(X), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return head, label_set


def evaluate(head, feats, labels, label_set, device):
    l2i   = {l: i for i, l in enumerate(label_set)}
    y     = torch.tensor([l2i[l] for l in labels], dtype=torch.long).to(device)
    X     = feats.to(device)
    with torch.no_grad():
        preds = head(X).argmax(1)

    acc = (preds == y).float().mean().item() * 100
    per_class = {}
    for i, cls in enumerate(label_set):
        tp    = ((preds == i) & (y == i)).sum().item()
        fp    = ((preds == i) & (y != i)).sum().item()
        fn    = ((preds != i) & (y == i)).sum().item()
        total = (y == i).sum().item()
        cls_acc = tp / total * 100     if total          else 0.0
        prec    = tp / (tp + fp) * 100 if (tp + fp) > 0  else 0.0
        rec     = tp / (tp + fn) * 100 if (tp + fn) > 0  else 0.0
        f1      = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[cls] = (cls_acc, prec, rec, f1, tp, fp, fn, total)

    n_cls      = len(label_set)
    macro_prec = sum(v[1] for v in per_class.values()) / n_cls
    macro_rec  = sum(v[2] for v in per_class.values()) / n_cls
    macro_f1   = sum(v[3] for v in per_class.values()) / n_cls
    return acc, per_class, macro_prec, macro_rec, macro_f1


parser = argparse.ArgumentParser()
parser.add_argument("--data-dir",   default=r"C:\Users\19802\Downloads\bloomi extra\All")
parser.add_argument("--backbone",   default=str(REPO_ROOT / "DinoBloom-G.pth"))
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--workers",    type=int, default=4)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice : {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

samples = discover_samples(Path(args.data_dir))
label_counts = defaultdict(int)
for _, l in samples:
    label_counts[l] += 1

print(f"Images    : {len(samples):,}")
print(f"Classes   : {len(label_counts)}")
for cls in sorted(label_counts):
    print(f"  {cls:<35}  {label_counts[cls]:>5,}")
print()

if not samples:
    print("No images found.")
    sys.exit(0)

ds     = CellDataset(samples)
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, collate_fn=pad_collate,
                    pin_memory=(device.type == "cuda"))

print("Extracting features...")
backbone = load_backbone(args.backbone, device)
feats, labels = extract_features(backbone, loader, device)
del backbone
torch.cuda.empty_cache()
print(f"  Features: {feats.shape}")

num_classes = len(set(labels))
print(f"\nTraining linear probe (L-BFGS, {num_classes} classes)...")
head, label_set = train_linear_probe(feats, labels, num_classes, device)

acc, per_class, macro_prec, macro_rec, macro_f1 = evaluate(head, feats, labels, label_set, device)

W    = 72
SEP  = "=" * W
SEP2 = "-" * W

print()
print(SEP)
print(f"  LINEAR PROBE RESULTS  ({len(samples):,} images, {num_classes} classes)")
print(SEP2)
print(f"  {'Metric':<14}  {'Value':>9}")
print(f"  {'-'*14}  {'-'*9}")
print(f"  {'Accuracy':<14}  {acc:>8.2f}%")
print(f"  {'Macro Prec':<14}  {macro_prec:>8.2f}%")
print(f"  {'Macro Recall':<14}  {macro_rec:>8.2f}%")
print(f"  {'Macro F1':<14}  {macro_f1:>8.2f}%")
print(SEP2)
print(f"  {'Class':<35}  {'Acc':>7}  {'Prec':>7}  {'Recall':>7}  {'F1':>7}  {'N':>6}")
print(f"  {'-'*35}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}")
for cls in sorted(label_set):
    a, p, r, f, tp, fp, fn, tot = per_class[cls]
    print(f"  {cls:<35}  {a:>6.1f}%  {p:>6.1f}%  {r:>6.1f}%  {f:>6.1f}%  {tot:>6,}")
print(SEP2)
print(f"  {'MACRO':<35}  {'':>7}  {macro_prec:>6.1f}%  {macro_rec:>6.1f}%  {macro_f1:>6.1f}%")
print(SEP)

out = REPO_ROOT / f"linear_probe_results.txt"
with open(out, "w", encoding="utf-8") as f:
    f.write(f"Linear Probe  {datetime.now()}\n")
    f.write(f"Backbone: {args.backbone}\n")
    f.write(f"Images: {len(samples):,}  Classes: {num_classes}\n\n")
    f.write(f"Accuracy    : {acc:.2f}%\n")
    f.write(f"Macro Prec  : {macro_prec:.2f}%\n")
    f.write(f"Macro Recall: {macro_rec:.2f}%\n")
    f.write(f"Macro F1    : {macro_f1:.2f}%\n\n")
    for cls in sorted(label_set):
        a, p, r, f1v, tp, fp, fn, tot = per_class[cls]
        f.write(f"  {cls:<35}  Acc={a:.1f}%  P={p:.1f}%  R={r:.1f}%  F1={f1v:.1f}%  N={tot}\n")
print(f"\n  Saved to: {out}")
