import os, sys, random, warnings
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

warnings.filterwarnings("ignore")

REPO_ROOT    = Path(__file__).parent.resolve()
ARCHIVE7_DIR = REPO_ROOT / "New Data" / "extracted" / "archive7"
ARCHIVE5_DIR = REPO_ROOT / "New Data" / "extracted" / "archive5"
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SKIP = {"C-NMC_test_final_phase_data", "C-NMC_test_prelim_phase_data",
        "testing_data", "training_data"}

BINARY = {"hem": 0, "all": 1, "Benign": 0, "Early": 1, "Pre": 1, "Pro": 1}
NAMES  = ["normal", "leukemia"]

def label_of(p):
    parts = p.stem.split("_")
    if len(parts) > 1:
        c = parts[-1]
        if c and any(x.isalpha() for x in c):
            return c
    return p.parent.name

def collect(root):
    out, skip = [], defaultdict(int)
    for dp, _, files in os.walk(root):
        if Path(dp).name in SKIP: continue
        for f in files:
            if Path(f).suffix.lower() not in EXTS: continue
            p = Path(dp) / f
            l = label_of(p)
            if l in BINARY: out.append((p, BINARY[l]))
            else: skip[l] += 1
    if skip: print(f"  skipped: {dict(skip)}")
    return out

class DS(Dataset):
    def __init__(self, samples, aug=False):
        self.s = samples
        t = [transforms.Resize((224, 224)), transforms.ToTensor(),
             transforms.Normalize(MEAN, STD)]
        if aug:
            t = [transforms.RandomHorizontalFlip(),
                 transforms.RandomRotation(10)] + t
        self.tf = transforms.Compose(t)
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        p, l = self.s[i]
        try: img = Image.open(p).convert("RGB")
        except: img = Image.new("RGB", (64,64), (128,128,128))
        return self.tf(img), l

def run_eval(model, loader, device):
    model.eval()
    tp=fp=fn=tn=0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            tp += ((pred==1)&(y==1)).sum().item()
            fp += ((pred==1)&(y==0)).sum().item()
            fn += ((pred==0)&(y==1)).sum().item()
            tn += ((pred==0)&(y==0)).sum().item()
    tot = tp+fp+fn+tn
    acc  = (tp+tn)/tot*100 if tot else 0
    prec = tp/(tp+fp)*100 if (tp+fp) else 0
    rec  = tp/(tp+fn)*100 if (tp+fn) else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0
    return acc, prec, rec, f1, tp, fp, fn, tn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}" +
      (f" ({torch.cuda.get_device_name(0)})" if device.type=="cuda" else ""), flush=True)

EPOCHS = 5
BS     = 64

print("\nLoading archive7 (C-NMC)...", flush=True)
a7 = collect(ARCHIVE7_DIR)
random.seed(42); random.shuffle(a7)
split = int(0.8*len(a7))
tr, te_a7 = a7[:split], a7[split:]
print(f"  train={len(tr):,}  test={len(te_a7):,}", flush=True)

print("Loading archive5 (Taleqani)...", flush=True)
a5 = collect(ARCHIVE5_DIR)
print(f"  cross-inst={len(a5):,}", flush=True)

kw = dict(batch_size=BS, num_workers=0, pin_memory=False)
tr_ld  = DataLoader(DS(tr,   aug=True),  shuffle=True,  **kw)
a7_ld  = DataLoader(DS(te_a7,aug=False), shuffle=False, **kw)
a5_ld  = DataLoader(DS(a5,  aug=False),  shuffle=False, **kw)

print("\nBuilding EfficientNet-B0 (frozen backbone, head only)...", flush=True)
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
for p in model.parameters():
    p.requires_grad_(False)
in_f = model.classifier[1].in_features
model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_f, 2))
model = model.to(device)
head_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable params (head only): {head_params:,}", flush=True)

opt = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
ce  = nn.CrossEntropyLoss()

print(f"\nTraining head for {EPOCHS} epochs on C-NMC (single institution)...", flush=True)
best_acc, best_state = 0.0, None
for ep in range(1, EPOCHS+1):
    model.train()
    t0 = __import__("time").time()
    loss_sum = cor = tot = 0
    for x, y in tr_ld:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        out  = model(x)
        loss = ce(out, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item()
        cor += (out.argmax(1)==y).sum().item()
        tot += len(y)
    tr_acc = cor/tot*100
    va_acc, *_ = run_eval(model, a7_ld, device)
    if va_acc > best_acc:
        best_acc  = va_acc
        best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
    print(f"  Ep {ep}/{EPOCHS}  loss {loss_sum/len(tr_ld):.4f}"
          f"  train {tr_acc:.1f}%  val {va_acc:.1f}%"
          f"  best {best_acc:.1f}%  {__import__('time').time()-t0:.0f}s", flush=True)

model.load_state_dict(best_state)
print("\n" + "="*60, flush=True)
print("SINGLE-INSTITUTION CNN RESULTS", flush=True)
print("="*60, flush=True)

acc_a7, p7, r7, f7, tp,fp,fn,tn = run_eval(model, a7_ld, device)
print(f"\n(A) In-distribution  archive7 C-NMC ({len(te_a7):,} images)", flush=True)
print(f"    Acc={acc_a7:.2f}%  P={p7:.1f}%  R={r7:.1f}%  F1={f7:.1f}%", flush=True)
print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}", flush=True)

acc_a5, p5, r5, f5, tp2,fp2,fn2,tn2 = run_eval(model, a5_ld, device)
print(f"\n(B) Cross-institution  archive5 Taleqani ({len(a5):,} images)", flush=True)
print(f"    Acc={acc_a5:.2f}%  P={p5:.1f}%  R={r5:.1f}%  F1={f5:.1f}%", flush=True)
print(f"\n    Drop: {acc_a7:.2f}% -> {acc_a5:.2f}%  ({acc_a5-acc_a7:+.2f} pp)", flush=True)
print("="*60, flush=True)

out = REPO_ROOT / "single_institution_results.txt"
with open(out, "w", encoding="utf-8") as f:
    f.write(f"Single-Institution CNN  {datetime.now()}\n")
    f.write(f"Model: EfficientNet-B0 frozen backbone + trained head\n")
    f.write(f"Train: archive7 C-NMC {len(tr):,} images, {EPOCHS} epochs\n\n")
    f.write(f"(A) In-distribution archive7 ({len(te_a7):,} images):\n")
    f.write(f"    Acc={acc_a7:.2f}%  P={p7:.1f}%  R={r7:.1f}%  F1={f7:.1f}%\n")
    f.write(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}\n\n")
    f.write(f"(B) Cross-institution archive5 ({len(a5):,} images):\n")
    f.write(f"    Acc={acc_a5:.2f}%  P={p5:.1f}%  R={r5:.1f}%  F1={f5:.1f}%\n")
    f.write(f"    TP={tp2}  FP={fp2}  FN={fn2}  TN={tn2}\n\n")
    f.write(f"Drop: {acc_a7:.2f}% -> {acc_a5:.2f}%  ({acc_a5-acc_a7:+.2f} pp)\n")
print(f"\nSaved to: {out}", flush=True)
