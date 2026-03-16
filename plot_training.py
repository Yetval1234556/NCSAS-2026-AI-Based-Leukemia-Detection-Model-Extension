import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).parent.resolve()
csv_path  = REPO_ROOT / "training_metrics.csv"

rows = []
with open(csv_path, newline="") as f:
    for row in csv.DictReader(f):
        rows.append({
            "epoch":      int(row["epoch"]),
            "train_loss": float(row["train_loss"]),
            "train_acc":  float(row["train_acc"]),
            "test_acc":   float(row["test_acc"]),
            "lr":         float(row["lr"]),
        })

seen, deduped = set(), []
for r in rows:
    k = (r["epoch"], round(r["lr"], 10))
    if k not in seen:
        seen.add(k)
        deduped.append(r)

run = sorted((r for r in deduped if r["epoch"] <= 75), key=lambda r: r["epoch"])
if len(run) > 75:
    run = run[-75:]

epochs     = [r["epoch"]      for r in run]
train_loss = [r["train_loss"] for r in run]
train_acc  = [r["train_acc"]  for r in run]
test_acc   = [r["test_acc"]   for r in run]
lr         = [r["lr"]         for r in run]
gap        = [tr - te for tr, te in zip(train_acc, test_acc)]

style = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#cccccc",
    "axes.grid":        True,
    "grid.color":       "#eeeeee",
    "grid.linestyle":   "-",
    "grid.linewidth":   0.8,
    "xtick.color":      "#444444",
    "ytick.color":      "#444444",
    "axes.labelcolor":  "#222222",
    "text.color":       "#222222",
}

plots = [
    ("Training Loss",        epochs, [train_loss],          ["tab:red"],                    None,              "Loss"),
    ("Accuracy",             epochs, [train_acc, test_acc], ["tab:blue", "tab:orange"],     ["Train", "Val"],  "Accuracy (%)"),
    ("Train–Val Gap",        epochs, [gap],                 ["tab:purple"],                 None,              "Gap (pp)"),
    ("Learning Rate",        epochs, [[v * 1e6 for v in lr]], ["tab:green"],                None,              "LR (×10⁻⁶)"),
]

for title, x, ys, colors, labels, ylabel in plots:
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("white")
        for i, y in enumerate(ys):
            kw = {"color": colors[i], "linewidth": 2}
            if labels:
                kw["label"] = labels[i]
            ax.plot(x, y, **kw)
        if title == "Train–Val Gap":
            ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        if labels:
            ax.legend(fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        fname = title.lower().replace(" ", "_").replace("–", "-") + ".png"
        out = REPO_ROOT / fname
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Saved: {out}")
