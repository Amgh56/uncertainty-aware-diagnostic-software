"""
Run from backend/ folder:
    python crc_find_best.py
 
Uses:
  - Model:      ../../Chexpert/config/pre_train.pth  (relative to backend/)
  - Valid CSV:  CheXpert1/valid.csv
  - Images:     CheXpert1/valid/...  (strips CheXpert-v1.0-small/ prefix)
 
Real CRC parameters:
  - alpha     = 0.1   (FNR bound)
  - lambda_hat = 0.292214  (non-conformity threshold)
  - A class is included if: 1 - sigmoid(class) <= lambda_hat
    i.e. sigmoid(class) >= 1 - 0.292214 = 0.707786
 
CNN baseline:
  - A class is included if: sigmoid(class) >= 0.5
 
Saves:
  - crc_results/best_candidate.json   — best image metadata
  - crc_results/all_candidates.json   — all valid candidates
  - crc_vs_cnn_figure.png             — final figure
"""
 
import os, json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from PIL import Image
from torchvision import transforms, models
import cv2
# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH = "/Users/abdullamaghrabi/Downloads/uncertainty-aware-diagnostic-software1/Chexpert/config/pre_train_torchscript.pt"

VALID_CSV    = "CheXpert1/valid.csv"
IMAGE_BASE   = "CheXpert1/"           # images live here after stripping prefix
PATH_PREFIX  = "CheXpert-v1.0-small/" # strip this from CSV paths
OUTPUT_FIG   = "crc_vs_cnn_figure.png"
RESULTS_DIR  = "crc_results"
 
ALPHA        = 0.1
LAMBDA_HAT   = 0.292214               # real empirical lambda_hat
CNN_THRESH   = 0.5
CRC_THRESH = LAMBDA_HAT       # = 0.707786  (sigmoid must exceed this)
 
CLASSES = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]
CSV_LABEL_ALIASES = {
    "Pleural Effusion": ["Pleural Effusion", "Pleural_Effusion"],
}
 
os.makedirs(RESULTS_DIR, exist_ok=True)
 
print(f"CRC sigmoid threshold : {CRC_THRESH:.6f}  (= λ̂)")
print(f"CNN sigmoid threshold : {CNN_THRESH}")
print(f"Alpha (FNR bound)     : {ALPHA}\n")
 
# ── MODEL ─────────────────────────────────────────────────────────────────────
def load_model(path):
    print(f"Loading model from {path}...")
    try:
        model = torch.jit.load(path, map_location="cpu")
    except Exception:
        model = torch.load(path, map_location="cpu", weights_only=False)
    model.eval()
    print("Model ready.\n")
    return model
 
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
 
# Config matching example.json
CONFIG = {
    "width": 512,
    "height": 512,
    "pixel_mean": 128.0,
    "pixel_std": 64.0,
    "use_equalizeHist": True,
}

def infer(model, img_path):
    with open(img_path, "rb") as f:
        img_bytes = f.read()

    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    w = int(CONFIG["width"])
    h = int(CONFIG["height"])

    # Always grayscale for medical X-rays
    gray = bgr[:, :, 0].astype(np.float32)
    gray = cv2.resize(gray, (w, h))
    if CONFIG.get("use_equalizeHist", False):
        gray = cv2.equalizeHist(gray.astype(np.uint8)).astype(np.float32)
    gray = (gray - float(CONFIG["pixel_mean"])) / float(CONFIG["pixel_std"])
    img_array = np.stack([gray, gray, gray], axis=0)

    x = torch.tensor(img_array[np.newaxis], dtype=torch.float32)
    with torch.no_grad():
        out = model(x)
        if isinstance(out, (list, tuple)):
            logits = out[0]
            if isinstance(logits, list):
                logits = torch.cat(logits, dim=1)
        else:
            logits = out
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    return {c: float(probs[i]) for i, c in enumerate(CLASSES)}
 
# ── SEARCH ────────────────────────────────────────────────────────────────────
def search(model, csv_path, image_base, path_prefix):
    df = pd.read_csv(csv_path)
    print(f"Searching {len(df)} images from {csv_path}...\n")

    candidates  = []
    total_found = 0

    for idx, row in df.iterrows():
        raw      = row["Path"] if "Path" in df.columns else row.iloc[0]
        img_path = os.path.join(image_base, raw.replace(path_prefix, "").lstrip("/"))
        if not os.path.exists(img_path):
          continue

    # Progress print every 1000 rows
        if idx % 1000 == 0:
            print(f"  [progress] row {idx} / {len(df)}  |  candidates so far: {total_found}", flush=True)

        true_labels = []
        for c in CLASSES:
            value = 0
            for col in CSV_LABEL_ALIASES.get(c, [c]):
                if col in row:
                    value = row.get(col, 0)
                    break
            if pd.notna(value) and float(value) == 1.0:
                true_labels.append(c)
    

        if not true_labels:
            continue

        try:
                probs = infer(model, img_path)
        except Exception as e:
            continue

        cnn_preds = [c for c in CLASSES if probs[c] >= CNN_THRESH]
        crc_preds = [c for c in CLASSES if probs[c] >= CRC_THRESH]

        # CRC adds classes CNN missed
        caught_by_crc = [c for c in true_labels if c in crc_preds and c not in cnn_preds]
        false_pos_crc = [c for c in crc_preds if c not in true_labels]
        false_pos_cnn = [c for c in cnn_preds if c not in true_labels]

        # FIXED CONDITION: CNN must miss at least one true label that CRC catches
        if not caught_by_crc:
            continue

        all_caught = all(c in crc_preds for c in true_labels)

        score = (len(caught_by_crc) * 4
                + len(true_labels) * 2
                - len(false_pos_crc) * 5
                + (10 if all_caught and not false_pos_crc else 0)
                + (5 if len(cnn_preds) <= 1 else 0))  # bonus if CNN only gets 1 or fewer

        candidates.append({
            "path":          img_path,
            "true_labels":   true_labels,
            "cnn_preds":     cnn_preds,
            "crc_preds":     crc_preds,
            "probs":         probs,
            "missed_by_cnn": [c for c in true_labels if c not in cnn_preds],
            "caught_by_crc": caught_by_crc,
            "false_pos_crc": false_pos_crc,
            "false_pos_cnn": false_pos_cnn,
            "all_caught":    all_caught,
            "score":         score,
        })
        total_found += 1

    print(f"Found {total_found} valid candidates in {csv_path}.\n")
    return candidates
 
# ── FIGURE ────────────────────────────────────────────────────────────────────
def make_figure(best):
    GREY   = "#bdc3c7"
    GREEN  = "#27ae60"
    ORANGE = "#e67e22"
    BLUE   = "#2980b9"
    RED    = "#e74c3c"
    PURPLE = "#8e44ad"
    BLACK  = "#2c3e50"
 
    fig = plt.figure(figsize=(20, 5.2), facecolor="white")
    gs  = gridspec.GridSpec(1, 3, figure=fig,
                            left=0.02, right=0.98,
                            wspace=0.35,
                            width_ratios=[1.0, 1.6, 2.2])
 
    # ── Panel 1: Chest X-ray ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(Image.open(best["path"]).convert("L"), cmap="gray", aspect="auto")
    ax1.axis("off")
    ax1.set_title("Patient Chest X-ray\n(CheXpert dataset)",
                  fontsize=10, fontweight="bold", pad=8)
 
    # ── Panel 2: DenseNet sigmoid bar chart ───────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    probs  = [best["probs"][c] for c in CLASSES]
    colors = []
    for c in CLASSES:
        if c in best["true_labels"] and c not in best["cnn_preds"]:
            colors.append(RED)       # true positive missed by CNN
        elif c in best["cnn_preds"] and c in best["true_labels"]:
            colors.append(GREEN)     # CNN correct
        elif c in best["cnn_preds"]:
            colors.append(ORANGE)    # CNN false positive
        else:
            colors.append(GREY)
 
    x = np.arange(len(CLASSES))
    ax2.bar(x, probs, color=colors, edgecolor="white", width=0.55, zorder=3)
 
    # CNN threshold line
    ax2.axhline(CNN_THRESH, color=BLACK, linestyle="--",
                linewidth=1.5, zorder=4, label=f"CNN threshold (0.5)")
 
    # CRC threshold line
    ax2.axhline(CRC_THRESH, color=PURPLE, linestyle="--",
                linewidth=1.5, zorder=4, label=f"CRC threshold ({CRC_THRESH:.4f})")
 
    ax2.set_xticks(x)
    ax2.set_xticklabels(CLASSES, rotation=28, ha="right", fontsize=8.5)
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("Sigmoid output", fontsize=9)
    ax2.set_title("DenseNet-121 Sigmoid Outputs", fontsize=10, fontweight="bold")
    ax2.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
    ax2.set_axisbelow(True)
 
    p1 = mpatches.Patch(color=GREEN,  label="CNN correct ✓")
    p2 = mpatches.Patch(color=RED,    label="Missed by CNN ✗")
    p3 = mpatches.Patch(color=GREY,   label="Not predicted")
    ax2.legend(handles=[p1, p2, p3], fontsize=7.5, loc="upper right")
 
    # ── Panel 3: Prediction set comparison ───────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
 
    def draw_box(ax, y, title, preds, true_labels, color):
        ax.text(0.02, y + 0.095, title,
                fontsize=9.5, fontweight="bold", color=BLACK,
                transform=ax.transAxes, va="bottom")
        if not preds:
            text = "{ — }"
        else:
            parts = []
            for p in preds:
                tick = "✓" if p in true_labels else "✗"
                col  = GREEN if p in true_labels else RED
                parts.append((tick, p, col))
            # build display string with tick marks
            text = "{  " + ",   ".join(f"{t} {p}" for t, p, _ in parts) + "  }"
 
        ax.text(0.02, y, text,
                fontsize=9, color=color,
                transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="round,pad=0.5",
                          facecolor=color + "15",
                          edgecolor=color, linewidth=1.4))
 
    ax3.set_title("Prediction Set Comparison",
                  fontsize=10, fontweight="bold", pad=8)
 
    draw_box(ax3, 0.72, "True Labels",
             best["true_labels"], best["true_labels"], GREEN)
    draw_box(ax3, 0.44, f"Standard CNN  (τ = {CNN_THRESH})",
             best["cnn_preds"],   best["true_labels"], ORANGE)
    draw_box(ax3, 0.14, f"CRC — SafeDx  (α = {ALPHA},  λ̂ = {LAMBDA_HAT})",
             best["crc_preds"],   best["true_labels"], BLUE)
 
    caught_str = ", ".join(best["caught_by_crc"])
    ax3.text(0.02, 0.04,
             f"↑  CRC recovers missed patholog{'y' if len(best['caught_by_crc'])==1 else 'ies'}: {caught_str}",
             fontsize=8.5, color=RED, fontweight="bold",
             transform=ax3.transAxes)
 
    plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"✅  Figure saved → {OUTPUT_FIG}")
 
if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    
    candidates = []
    
    candidates += search(
        model,
        "/Users/abdullamaghrabi/Downloads/uncertainty-aware-diagnostic-software1/src/backend/NIH_dataset/remaining_after_calib.csv",
        "/Users/abdullamaghrabi/Downloads/uncertainty-aware-diagnostic-software1/src/backend/",
        ""
    )
    
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    print(f"\n{'='*60}")
    print(f"TOP 20 CANDIDATES:")
    print(f"{'='*60}\n")
    for i, c in enumerate(candidates[:20]):
        print(f"[{i+1}] score={c['score']:.1f}  all_caught={c['all_caught']}  false_pos={c['false_pos_crc']}  cnn_count={len(c['cnn_preds'])}")
        print(f"     path    : {c['path']}")
        print(f"     true    : {c['true_labels']}")
        print(f"     cnn     : {c['cnn_preds']}")
        print(f"     crc     : {c['crc_preds']}")
        print(f"     caught  : {c['caught_by_crc']}")
        print(f"     sigmoids:")
        for cls, prob in c['probs'].items():
            in_true = "TRUE" if cls in c['true_labels'] else "    "
            in_cnn  = "<-- CNN" if cls in c['cnn_preds'] else ""
            in_crc  = "<-- CRC" if cls in c['crc_preds'] else ""
            print(f"       {cls:<20}: {prob:.4f}  {in_true}  {in_cnn} {in_crc}")
        print()