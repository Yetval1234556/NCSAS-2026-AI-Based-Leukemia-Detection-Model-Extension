import sys
import os
import csv
import warnings
from datetime import datetime


class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()
warnings.filterwarnings("ignore", message="xFormers is not available")
import argparse
from pathlib import Path
from collections import defaultdict

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

DIAGNOSIS_MAP = {
    "AML": "AML",
    "APL": "APML",
}


class DinoBloomClassifier(nn.Module):
    def __init__(self, backbone, num_classes: int, feat_dim: int = 1536):
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


def load_model(ckpt_path: str, device):
    sys.path.insert(0, str(REPO_ROOT))
    from dinov2.hub.backbones import dinov2_vitg14

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    num_classes  = ckpt["num_classes"]
    class_to_idx = ckpt["class_to_idx"]

    backbone = dinov2_vitg14(pretrained=False, img_size=224)
    model    = DinoBloomClassifier(backbone, num_classes).to(device)

    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"Loaded  : {ckpt_path}")
    print(f"Epoch   : {ckpt.get('epoch', '?')}   Classes: {num_classes}")
    return model, class_to_idx


PATCH_SIZE = 14

def _ceil14(n):
    return ((n + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE

def pad_collate(batch):
    tensors, labels, patients = zip(*batch)
    max_h = _ceil14(max(t.shape[1] for t in tensors))
    max_w = _ceil14(max(t.shape[2] for t in tensors))
    padded = [F.pad(t, (0, max_w - t.shape[2], 0, max_h - t.shape[1])) for t in tensors]
    return torch.stack(padded), list(labels), list(patients)


class EvalDataset(Dataset):
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
        path, label, patient_id = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (64, 64), (128, 128, 128))
        return self.tf(img), label, patient_id


def class_metrics(all_true, all_pred, classes):
    per = {}
    for cls in classes:
        tp    = sum(1 for t, p in zip(all_true, all_pred) if t == cls and p == cls)
        fp    = sum(1 for t, p in zip(all_true, all_pred) if t != cls and p == cls)
        fn    = sum(1 for t, p in zip(all_true, all_pred) if t == cls and p != cls)
        total = sum(1 for t in all_true if t == cls)

        cls_acc = tp / total * 100      if total          else 0.0
        prec    = tp / (tp + fp) * 100  if (tp + fp) > 0  else 0.0
        rec     = tp / (tp + fn) * 100  if (tp + fn) > 0  else 0.0
        f1      = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per[cls] = (cls_acc, prec, rec, f1, tp, fp, fn, total)

    macro_p  = sum(v[1] for v in per.values()) / len(per)
    macro_r  = sum(v[2] for v in per.values()) / len(per)
    macro_f1 = sum(v[3] for v in per.values()) / len(per)
    return per, macro_p, macro_r, macro_f1


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=str(REPO_ROOT / "checkpoint_latest.pth"))
parser.add_argument("--data-dir",   default=r"C:\Users\19802\Downloads\bloomi extra\All")
parser.add_argument("--metadata",   default=str(REPO_ROOT / "patient_metadata.csv"))
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--workers",    type=int, default=4)
args = parser.parse_args()

out_path = REPO_ROOT / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
_out_file = open(out_path, "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _out_file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice  : {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
print()

patient_label = {}
with open(args.metadata, newline="") as f:
    for row in csv.DictReader(f):
        pid       = row["Patient_ID"].strip()
        diagnosis = row["Diagnosis"].strip()
        mapped    = DIAGNOSIS_MAP.get(diagnosis)
        if mapped:
            patient_label[pid] = mapped
        else:
            print(f"  WARNING: no mapping for diagnosis '{diagnosis}' (patient {pid})")

print(f"Patients in metadata : {len(patient_label)}")
diag_counts = defaultdict(int)
for v in patient_label.values():
    diag_counts[v] += 1
for cls, cnt in sorted(diag_counts.items()):
    print(f"  {cls:<8} : {cnt} patients")
print()

model, class_to_idx = load_model(args.checkpoint, device)
idx_to_class = {v: k for k, v in class_to_idx.items()}

for cls in set(DIAGNOSIS_MAP.values()):
    if cls not in class_to_idx:
        print(f"  WARNING: class '{cls}' not in model — predictions may be off")
print()

data_dir = Path(args.data_dir)
samples  = []
for root, _, files in os.walk(data_dir):
    root_path  = Path(root)
    patient_id = None
    for part in root_path.parts:
        if part in patient_label:
            patient_id = part
            break
    if patient_id is None:
        continue
    label = patient_label[patient_id]
    for fname in files:
        if Path(fname).suffix.lower() in VALID_EXTS:
            samples.append((root_path / fname, label, patient_id))

print(f"Images found : {len(samples):,}")
if not samples:
    print("No images to evaluate.")
    sys.exit(0)

ds     = EvalDataset(samples)
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, collate_fn=pad_collate,
                    pin_memory=(device.type == "cuda"))

all_true_img  = []
all_pred_img  = []
patient_votes = defaultdict(lambda: defaultdict(int))
patient_label_ = {s[2]: s[1] for s in samples}

print("Evaluating...\n")
with torch.no_grad():
    for batch_idx, (imgs, labels, patients) in enumerate(loader):
        imgs = imgs.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
            logits = model(imgs)
        preds = logits.argmax(1).cpu().tolist()

        for pred_idx, true_label, patient_id in zip(preds, labels, patients):
            pred_label = idx_to_class[pred_idx]
            all_true_img.append(true_label)
            all_pred_img.append(pred_label)
            patient_votes[patient_id][pred_label] += 1

        if (batch_idx + 1) % 20 == 0:
            done = min((batch_idx + 1) * args.batch_size, len(samples))
            print(f"  {done:>6,} / {len(samples):,}", flush=True)

eval_classes = sorted(set(DIAGNOSIS_MAP.values()))
per_img, m_prec, m_rec, m_f1 = class_metrics(all_true_img, all_pred_img, eval_classes)

total_imgs    = len(all_true_img)
total_correct = sum(1 for t, p in zip(all_true_img, all_pred_img) if t == p)
total_acc     = total_correct / total_imgs * 100 if total_imgs else 0

W    = 68
SEP  = "=" * W
SEP2 = "-" * W

print()
print(SEP)
print(f"  IMAGE-LEVEL RESULTS  ({total_imgs:,} images)")
print(SEP2)
print(f"  {'Metric':<14}  {'Value':>9}")
print(f"  {'-'*14}  {'-'*9}")
print(f"  {'Accuracy':<14}  {total_acc:>8.2f}%")
print(f"  {'Macro Prec':<14}  {m_prec:>8.2f}%")
print(f"  {'Macro Recall':<14}  {m_rec:>8.2f}%")
print(f"  {'Macro F1':<14}  {m_f1:>8.2f}%")
print(SEP2)
print(f"  {'Class':<8}  {'Acc':>7}  {'Prec':>7}  {'Recall':>7}  {'F1':>7}  "
      f"{'TP':>6}  {'FP':>6}  {'FN':>6}  {'Total':>7}")
print(f"  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  "
      f"{'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}")
for cls in eval_classes:
    a, p, r, f, tp, fp, fn, tot = per_img[cls]
    bar = "#" * int(a / 5)
    print(f"  {cls:<8}  {a:>6.1f}%  {p:>6.1f}%  {r:>6.1f}%  {f:>6.1f}%  "
          f"{tp:>6,}  {fp:>6,}  {fn:>6,}  {tot:>7,}  {bar}")
print(SEP2)
print(f"  {'MACRO':<8}  {'':>7}  {m_prec:>6.1f}%  {m_rec:>6.1f}%  {m_f1:>6.1f}%")
print(SEP)

all_true_pat = []
all_pred_pat = []
pat_rows     = []

for pid in sorted(patient_votes):
    true_label = patient_label_[pid]
    votes      = patient_votes[pid]
    pred_label = max(votes, key=votes.get)
    all_true_pat.append(true_label)
    all_pred_pat.append(pred_label)
    vote_str   = "  ".join(f"{k}:{v}" for k, v in sorted(votes.items(), key=lambda x: -x[1]))
    tick       = "YES" if pred_label == true_label else "NO "
    pat_rows.append((pid, true_label, pred_label, vote_str, tick))

per_pat, pm_prec, pm_rec, pm_f1 = class_metrics(all_true_pat, all_pred_pat, eval_classes)

total_pats    = len(all_true_pat)
pat_correct   = sum(1 for t, p in zip(all_true_pat, all_pred_pat) if t == p)
pat_acc       = pat_correct / total_pats * 100 if total_pats else 0

print()
print(SEP)
print(f"  PATIENT-LEVEL RESULTS  (majority vote, {total_pats} patients)")
print(SEP2)
print(f"  {'Patient':<12}  {'True':<6}  {'Pred':<6}  {'Votes':<30}  {'OK'}")
print(f"  {'-'*12}  {'-'*6}  {'-'*6}  {'-'*30}  {'-'*4}")
for pid, true_label, pred_label, vote_str, tick in pat_rows:
    print(f"  {pid:<12}  {true_label:<6}  {pred_label:<6}  {vote_str:<30}  {tick}")

print(SEP2)
print(f"  {'Metric':<14}  {'Value':>9}")
print(f"  {'-'*14}  {'-'*9}")
print(f"  {'Accuracy':<14}  {pat_acc:>8.2f}%  ({pat_correct}/{total_pats})")
print(f"  {'Macro Prec':<14}  {pm_prec:>8.2f}%")
print(f"  {'Macro Recall':<14}  {pm_rec:>8.2f}%")
print(f"  {'Macro F1':<14}  {pm_f1:>8.2f}%")
print(SEP2)
print(f"  {'Class':<8}  {'Acc':>7}  {'Prec':>7}  {'Recall':>7}  {'F1':>7}  "
      f"{'TP':>4}  {'FP':>4}  {'FN':>4}  {'N':>4}")
print(f"  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  "
      f"{'-'*4}  {'-'*4}  {'-'*4}  {'-'*4}")
for cls in eval_classes:
    a, p, r, f, tp, fp, fn, tot = per_pat[cls]
    print(f"  {cls:<8}  {a:>6.1f}%  {p:>6.1f}%  {r:>6.1f}%  {f:>6.1f}%  "
          f"{tp:>4}  {fp:>4}  {fn:>4}  {tot:>4}")
print(SEP2)
print(f"  {'MACRO':<8}  {'':>7}  {pm_prec:>6.1f}%  {pm_rec:>6.1f}%  {pm_f1:>6.1f}%")
print(SEP)
print()

sys.stdout = sys.__stdout__
_out_file.close()
print(f"  Report saved to: {out_path}")
