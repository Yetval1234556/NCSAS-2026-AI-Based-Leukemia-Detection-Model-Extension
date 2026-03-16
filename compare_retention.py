import sys
import os
import warnings
import argparse
from pathlib import Path
from collections import defaultdict
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

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

warnings.filterwarnings("ignore", message="xFormers is not available")

REPO_ROOT     = Path(__file__).parent.resolve()
VALID_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
PATCH_SIZE    = 14


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


def discover_samples(data_dir: Path, max_per_class: int = 0):
    import random
    by_class = defaultdict(list)
    for root, _, files in os.walk(data_dir):
        root_path = Path(root)
        for fname in files:
            if Path(fname).suffix.lower() in VALID_EXTS:
                label = root_path.name
                by_class[label].append((root_path / fname, label))

    STRUCTURAL = {"All", "Signed slides", "Unsigned slides"}
    samples = []
    for label, items in by_class.items():
        if label in STRUCTURAL:
            continue
        if max_per_class > 0 and len(items) > max_per_class:
            items = random.sample(items, max_per_class)
        samples.extend(items)
    return samples


def load_original_backbone(pth_path: str, device):
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
    print(f"  Original DinoBloom-G loaded | missing={len(missing)}  unexpected={len(unexpected)}")
    return model.eval().to(device)


def load_finetuned_backbone(ckpt_path: str, device):
    sys.path.insert(0, str(REPO_ROOT))
    from dinov2.hub.backbones import dinov2_vitg14
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = dinov2_vitg14(pretrained=False, img_size=224)

    state = {k.replace("_orig_mod.", ""): v
             for k, v in ckpt["model_state_dict"].items()}
    backbone_state = {k[len("backbone."):]: v
                      for k, v in state.items()
                      if k.startswith("backbone.")}
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    print(f"  Fine-tuned backbone loaded  | missing={len(missing)}  unexpected={len(unexpected)}"
          f"  (epoch {ckpt.get('epoch','?')})")
    return model.eval().to(device)


@torch.no_grad()
def extract_features(backbone, loader, device, total_images):
    feats, labels = [], []
    done = 0
    t0 = __import__("time").time()
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
            f = backbone(imgs)
        f = F.normalize(f.float(), dim=-1)
        feats.append(f.cpu())
        labels.extend(lbls)
        done += len(lbls)

        elapsed = __import__("time").time() - t0
        speed   = done / elapsed if elapsed > 0 else 0
        eta     = (total_images - done) / speed if speed > 0 else 0
        pct     = done / total_images * 100
        bar     = "#" * int(pct / 2) + "." * (50 - int(pct / 2))
        print(f"\r  [{bar}] {done:>5}/{total_images}  {pct:>5.1f}%  "
              f"{speed:>5.1f} img/s  ETA {int(eta//60):02d}:{int(eta%60):02d}",
              end="", flush=True)

    print()
    elapsed = __import__("time").time() - t0
    print(f"  Done — {done:,} images in {elapsed:.1f}s  ({done/elapsed:.1f} img/s)")
    return torch.cat(feats, dim=0), labels


def knn_classify(feats, labels, k=5):
    label_set = sorted(set(labels))
    l2i  = {l: i for i, l in enumerate(label_set)}
    y    = torch.tensor([l2i[l] for l in labels])
    N    = len(y)

    print(f"  Running k-NN on {N:,} × {feats.shape[1]}-d features (CPU)...")
    t0     = __import__("time").time()
    y_pred = torch.zeros(N, dtype=torch.long)

    CHUNK = 512
    for start in range(0, N, CHUNK):
        end       = min(start + CHUNK, N)
        sim_chunk = feats[start:end] @ feats.T
        for i in range(end - start):
            sim_chunk[i, start + i] = -1e9
        topk_idx = sim_chunk.topk(k, dim=1).indices
        for i, row in enumerate(topk_idx):
            neighbor_labels = y[row].tolist()
            vote = max(set(neighbor_labels), key=neighbor_labels.count)
            y_pred[start + i] = vote

        pct = end / N * 100
        bar = "#" * int(pct / 2) + "." * (50 - int(pct / 2))
        print(f"\r  [{bar}] {end:>5}/{N}  {pct:>5.1f}%", end="", flush=True)

    print()
    elapsed = __import__("time").time() - t0
    print(f"  k-NN done in {elapsed:.1f}s")
    return y_pred, y, label_set


def compute_metrics(y_pred, y, label_set):
    N = len(y)
    acc = (y_pred == y).sum().item() / N * 100

    per_class = {}
    for i, cls in enumerate(label_set):
        tp    = ((y_pred == i) & (y == i)).sum().item()
        fp    = ((y_pred == i) & (y != i)).sum().item()
        fn    = ((y_pred != i) & (y == i)).sum().item()
        total = (y == i).sum().item()

        cls_acc = tp / total * 100      if total          else 0.0
        prec    = tp / (tp + fp) * 100  if (tp + fp) > 0  else 0.0
        rec     = tp / (tp + fn) * 100  if (tp + fn) > 0  else 0.0
        f1      = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[cls] = (cls_acc, prec, rec, f1, tp, fp, fn, total)

    n_cls      = len(label_set)
    macro_prec = sum(v[1] for v in per_class.values()) / n_cls
    macro_rec  = sum(v[2] for v in per_class.values()) / n_cls
    macro_f1   = sum(v[3] for v in per_class.values()) / n_cls

    return acc, per_class, macro_prec, macro_rec, macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",      default=r"C:\Users\19802\Downloads\bloomi extra\All")
    parser.add_argument("--original",      default=str(REPO_ROOT / "DinoBloom-G.pth"))
    parser.add_argument("--finetuned",     default=str(REPO_ROOT / "checkpoint_latest.pth"))
    parser.add_argument("--batch-size",    type=int, default=32)
    parser.add_argument("--workers",       type=int, default=4)
    parser.add_argument("--k",             type=int, default=5)
    parser.add_argument("--max-per-class", type=int, default=0,
                        help="Cap images per class (0 = no cap)")
    args = parser.parse_args()

    out_path = REPO_ROOT / f"retention_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    _out_file = open(out_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _out_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"k-NN   : k={args.k}" + (f"  |  max-per-class={args.max_per_class}" if args.max_per_class > 0 else "") + "\n")

    samples = discover_samples(Path(args.data_dir), max_per_class=args.max_per_class)
    STRUCTURAL = {"All", "Signed slides", "Unsigned slides"}
    samples    = [(p, l) for p, l in samples if l not in STRUCTURAL]

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
        return

    ds     = CellDataset(samples)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=pad_collate,
                        pin_memory=(device.type == "cuda"))

    W    = 80
    SEP  = "=" * W
    SEP2 = "-" * W

    results = {}
    for name, loader_fn in [
        ("Original DinoBloom-G", lambda: load_original_backbone(args.original, device)),
        ("Fine-tuned (ours)",    lambda: load_finetuned_backbone(args.finetuned, device)),
    ]:
        print(f"── {name} ──")
        backbone = loader_fn()
        feats, labels = extract_features(backbone, loader, device, len(samples))
        y_pred, y, label_set = knn_classify(feats, labels, k=args.k)
        acc, per_class, m_prec, m_rec, m_f1 = compute_metrics(y_pred, y, label_set)
        results[name] = (acc, per_class, m_prec, m_rec, m_f1, label_set)
        print(f"  Accuracy: {acc:.2f}%  |  Macro P: {m_prec:.2f}%  R: {m_rec:.2f}%  F1: {m_f1:.2f}%")
        del backbone
        torch.cuda.empty_cache()
        print()

    orig_acc, orig_cls, orig_mp, orig_mr, orig_mf, orig_ls = results["Original DinoBloom-G"]
    fine_acc, fine_cls, fine_mp, fine_mr, fine_mf, fine_ls = results["Fine-tuned (ours)"]
    all_classes = sorted(set(orig_ls + fine_ls))

    print(SEP)
    print(f"  RETENTION REPORT  ({args.k}-NN, {len(samples):,} images)")
    print(SEP2)
    print(f"  {'Model':<25}  {'Accuracy':>9}  {'Macro P':>8}  {'Macro R':>8}  {'Macro F1':>9}  {'Δ Acc':>7}")
    print(f"  {'-'*25}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*7}")
    print(f"  {'Original DinoBloom-G':<25}  {orig_acc:>8.2f}%  {orig_mp:>7.2f}%  {orig_mr:>7.2f}%  {orig_mf:>8.2f}%")
    delta_acc = fine_acc - orig_acc
    print(f"  {'Fine-tuned (ours)':<25}  {fine_acc:>8.2f}%  {fine_mp:>7.2f}%  {fine_mr:>7.2f}%  {fine_mf:>8.2f}%  {delta_acc:>+6.2f}%")

    print(SEP2)
    print(f"  PER-CLASS BREAKDOWN")
    print(SEP2)
    H = (f"  {'Class':<28}  "
         f"{'Orig Acc':>8}  {'Orig P':>7}  {'Orig R':>7}  {'Orig F1':>8}  "
         f"{'Ours Acc':>8}  {'Ours P':>7}  {'Ours R':>7}  {'Ours F1':>8}  "
         f"{'Δ Acc':>7}  {'Δ F1':>7}")
    print(H)
    print(f"  {'-'*28}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*8}  "
          f"{'-'*8}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*7}")

    for cls in all_classes:
        oa, op, or_, of1, *_ = orig_cls.get(cls, (0, 0, 0, 0, 0, 0, 0, 0))
        fa, fp, fr,  ff1, *_ = fine_cls.get(cls, (0, 0, 0, 0, 0, 0, 0, 0))
        d_acc = fa - oa
        d_f1  = ff1 - of1
        print(f"  {cls:<28}  "
              f"{oa:>7.1f}%  {op:>6.1f}%  {or_:>6.1f}%  {of1:>7.1f}%  "
              f"{fa:>7.1f}%  {fp:>6.1f}%  {fr:>6.1f}%  {ff1:>7.1f}%  "
              f"{d_acc:>+6.1f}%  {d_f1:>+6.1f}%")

    print(SEP2)
    print(f"  {'MACRO':<28}  "
          f"{'':>8}  {orig_mp:>6.1f}%  {orig_mr:>6.1f}%  {orig_mf:>7.1f}%  "
          f"{'':>8}  {fine_mp:>6.1f}%  {fine_mr:>6.1f}%  {fine_mf:>7.1f}%  "
          f"{fine_acc-orig_acc:>+6.1f}%  {fine_mf-orig_mf:>+6.1f}%")
    print(SEP)
    print(f"\n  Accuracy retained : {fine_acc / orig_acc * 100:.1f}% of original")
    print(f"  Macro F1 retained : {fine_mf  / orig_mf  * 100:.1f}% of original")
    print()

    sys.stdout = sys.__stdout__
    _out_file.close()
    print(f"  Report saved to: {out_path}")


if __name__ == "__main__":
    main()
