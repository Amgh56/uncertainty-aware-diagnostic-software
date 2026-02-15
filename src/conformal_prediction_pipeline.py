from pathlib import Path
import numpy as np
import pandas as pd
import json
import sys
import torch
import cv2


SCRIPT_DIR = Path(__file__).resolve().parent     
ROOT_DIR = SCRIPT_DIR.parent                   
CHEXPERT_DIR = ROOT_DIR / "Chexpert"             

sys.path.insert(0, str(CHEXPERT_DIR))
from model.classifier import Classifier

DISEASES = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]


def load_paths_labels(
    csv_path: str | Path,
    repo_root: Path | None = None,
    sample_n: int | None = None,
    seed: int = 42,
    images_root: Path | None = None, 
):
    """
    Load image paths + multi-label ground-truth from a CSV.

    Reads `csv_path` (must contain columns: "Path" + DISEASES), optionally samples `sample_n`
    rows with `seed`, resolves each image path to an absolute path, and builds a binary label
    matrix for the 5 diseases.

    Path resolution:
      - If `images_root` is provided, each CSV path is mapped by filename into `images_root`.
      - Otherwise, relative paths are resolved against `repo_root`, `SCRIPT_DIR`, or `ROOT_DIR`.

    Returns:
      abs_paths: list[Path]       # absolute resolved image paths (length N)
      labels: np.ndarray          # shape (N, 5), values in {0,1}
      pos_mask: np.ndarray        # shape (N,), True if a row has >=1 positive label
    """
    csv_path = Path(csv_path)

    if repo_root is None:
        repo_root = ROOT_DIR

    df = pd.read_csv(csv_path)
    if sample_n is not None:
        df = df.sample(n=sample_n, random_state=seed).reset_index(drop=True)


    required = ["Path"] + DISEASES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path.name}: {missing}")

    rel_paths = df["Path"].astype(str).tolist()
    abs_paths = []

    if images_root is not None:
            images_root = Path(images_root)
            for p in rel_paths:
                fname = Path(p).name           
                candidate = images_root / fname   
                if not candidate.exists():
                    raise FileNotFoundError(f"Missing image in images_root: {candidate}")
                abs_paths.append(candidate.resolve())
    else:
            for p in rel_paths:
                rel = Path(p)
                if rel.is_absolute():
                    candidate = rel
                else:
                    candidates = [repo_root / rel, SCRIPT_DIR / rel, ROOT_DIR / rel]
                    candidate = next((c for c in candidates if c.exists()), candidates[0])
                abs_paths.append(candidate.resolve())


    labels = (
        df[DISEASES]
        .fillna(0)
        .astype(float)
        .round()
        .astype(np.int64)
        .to_numpy()
    )
    if not np.isin(labels, [0, 1]).all():
        bad_vals = np.unique(labels[~np.isin(labels, [0, 1])]) 
        raise ValueError(f"Found non-binary label values: {bad_vals}")

    pos_mask = labels.sum(axis=1) > 0

    return abs_paths, labels, pos_mask


def load_chexpert_pretrained_model():
    """
    Load CheXpert pretrained `Classifier` using `config/example.json` and `pre_train.pth`.
    Returns (model, config) with model set to eval() mode.
    """
    config_dir = CHEXPERT_DIR / "config"

    with open(config_dir / "example.json", "r") as f:
        cfg = json.load(f)

    class Config: pass
    config = Config()
    for k, v in cfg.items():
        setattr(config, k, v)

    model = Classifier(config)
    state_dict = torch.load(str(config_dir / "pre_train.pth"), map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("Loaded CheXpert pretrained model")
    return model, config

def pick_device() -> str:
    """
        Specifiy where to run the model we usually use cuda as it is the fastest and stronges
    """
    if torch.cuda.is_available():
        return "cuda"
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return "cpu"
    return "mps"

def preprocess_image(img_path: Path, config) -> np.ndarray:
    """
    Read a grayscale image, resize to (config.width, config.height), optionally equalize,
    normalize with (pixel_mean, pixel_std), and return as float32 array (3, H, W).
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    img = cv2.resize(img, (config.width, config.height))

    if getattr(config, "use_equalizeHist", False):
        img = cv2.equalizeHist(img)

    img = img.astype(np.float32)
    img = (img - config.pixel_mean) / config.pixel_std

    img = np.stack([img, img, img], axis=0)  
    return img


def make_batch(img_paths, config, device: str):
    """
    Preprocess a list of image paths, stack into a batch (B,3,H,W), and return a float32
    torch tensor on the specified device.
    """
    batch_np = np.stack([preprocess_image(p, config) for p in img_paths], axis=0) 
    x = torch.tensor(batch_np, dtype=torch.float32, device=device)
    return x

def predict_batch_probs(model, x):
    """Run model inference on a batch tensor and return sigmoid probabilities (B,5) as numpy."""

    model.eval()
    with torch.no_grad():
        outputs, _ = model(x)

        if isinstance(outputs, list):
            logits = torch.cat(outputs, dim=1)
        elif isinstance(outputs, torch.Tensor):
            logits = outputs
        else:
            raise TypeError(f"Unexpected model output type: {type(outputs)}")

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        return probs

def infer_all_probs(model, config, img_paths, batch_size: int = 16):
    """Run batched inference over all images and return probabilities array (N,5)."""

    device = pick_device()
    model = model.to(device)

    all_probs = []
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i+batch_size]
        x = make_batch(batch_paths, config, device)
        probs = predict_batch_probs(model, x) 
        all_probs.append(probs)

        if (i // batch_size) % 10 == 0:
            print(f"Inferred {i}/{len(img_paths)}")

    sgmd = np.concatenate(all_probs, axis=0)  
    return sgmd


def false_negative_rate(pred_set: np.ndarray, gt_labels: np.ndarray, pos_mask: np.ndarray) -> float:
    pred_set = pred_set[pos_mask]
    gt_labels = gt_labels[pos_mask]

    denom = gt_labels.sum(axis=1)

    if (denom == 0).any():
        raise ValueError("pos_mask still contains rows with zero positives. Something is wrong.")

    num = (pred_set * gt_labels).sum(axis=1)

    recall = num / denom
    fnr = 1.0 - recall.mean()
    return float(fnr)


def compute_lamhat(cal_sgmd: np.ndarray, cal_labels: np.ndarray, cal_pos_mask: np.ndarray, alpha: float) -> float:
    sgmd = cal_sgmd[cal_pos_mask]
    labels = cal_labels[cal_pos_mask]
    n = sgmd.shape[0]

    target = ((n + 1) / n) * alpha - (1 / n)

    candidates = np.unique(sgmd)
    candidates.sort()

    lamhat = candidates[0]  

    for lam in candidates:
        pred_set = sgmd >= lam
        fnr = false_negative_rate(pred_set, labels, np.ones(n, dtype=bool)) 
        if fnr <= target:
            lamhat = lam
        else:
            break

    return float(lamhat)

def evaluate_prediction_sets(pred_sets, labels, pos_mask):
    n = pred_sets.shape[0]
    set_sizes = pred_sets.sum(axis=1)

    results = {
        "fnr": false_negative_rate(pred_sets, labels, pos_mask),
        "avg_set_size": float(set_sizes.mean()),
        "median_set_size": float(np.median(set_sizes)),
        "empty_set_rate": float((set_sizes == 0).mean()),
        "full_set_rate": float((set_sizes == 5).mean()),
        "size_distribution": {
            int(s): int((set_sizes == s).sum()) for s in range(6)
        },
    }

    per_disease = {}
    for j, d in enumerate(DISEASES):
        pos_j = labels[:, j] == 1
        n_pos = int(pos_j.sum())
        if n_pos > 0:
            recall_j = pred_sets[pos_j, j].mean()
            per_disease[d] = {
                "n_positive": n_pos,
                "recall": round(float(recall_j), 4),
                "fnr": round(1.0 - float(recall_j), 4),
                "pred_rate": round(float(pred_sets[:, j].mean()), 4),
            }
        else:
            per_disease[d] = {"n_positive": 0, "recall": None, "fnr": None}
    results["per_disease"] = per_disease

    return results


def save_lamhat_json(out_path: Path, lamhat: float, alpha: float, calib_csv: str, n_total: int, n_pos: int, batch_size: int):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "alpha": float(alpha),
        "lamhat": float(lamhat),
        "calib_csv": str(calib_csv),
        "n_calib_total": int(n_total),
        "n_calib_pos": int(n_pos),
        "batch_size": int(batch_size),
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved lamhat JSON -> {out_path}")

def prediction_set_names(probs_row: np.ndarray, lamhat: float) -> list[str]:
    """
    Given one row of probabilities shape (5,), return the list of disease names
    whose probability >= lamhat.
    """
    mask = probs_row >= lamhat
    return [DISEASES[i] for i in range(len(DISEASES)) if mask[i]]




if __name__ == "__main__":
    alpha = 0.1
    batch_size = 32

    # ── KEY CHANGE: much larger splits ──
    CAL_SAMPLE_N = 10_000
    TEST_SAMPLE_N = 30_000
    SEED = 0

    data_dir = Path(__file__).resolve().parent / "NIH_dataset"
    
    images_root_1024 = SCRIPT_DIR / "NIH_images_1024"


    # ── UPDATE: point to the new 10k calibration CSV ──
    calib_csv = data_dir / "calibration_10000.csv"
    test_csv  = data_dir / "remaining_after_calib.csv"
    art_dir   = data_dir / "artifacts"
    lamhat_path = art_dir / "lamhat.json"

    # -------------------- CALIBRATION --------------------
    cal_paths, cal_labels, cal_pos_mask = load_paths_labels(
    calib_csv, sample_n=CAL_SAMPLE_N, seed=SEED, images_root=images_root_1024
)

    print("\n==================== CALIBRATION ====================")
    print(f"  N paths: {len(cal_paths)}")
    print(f"  labels shape: {cal_labels.shape}")
    print(f"  positive rows: {int(cal_pos_mask.sum())} / {len(cal_pos_mask)}"
          f"  ({cal_pos_mask.mean():.1%})")
    print(f"  example path: {cal_paths[0]}")

    model, config = load_chexpert_pretrained_model()
    device = pick_device()
    print(f"\nDevice: {device}")

    print("\nRunning calibration inference...")
    cal_sgmd = infer_all_probs(model, config, cal_paths, batch_size=batch_size)

    print(f"\n  cal_sgmd shape: {cal_sgmd.shape}")
    print(f"  min/max: {float(cal_sgmd.min()):.6f} / {float(cal_sgmd.max()):.6f}")
    print(f"  first probs: {np.round(cal_sgmd[0], 4)}")

    lamhat = compute_lamhat(cal_sgmd, cal_labels, cal_pos_mask, alpha)

    print(f"\n── Calibration Results ──")
    print(f"  alpha:  {alpha}")
    print(f"  lamhat: {lamhat:.6f}")

    cal_pred_sets = cal_sgmd >= lamhat
    cal_eval = evaluate_prediction_sets(cal_pred_sets, cal_labels, cal_pos_mask)

    print(f"  cal FNR:           {cal_eval['fnr']:.4f}")
    print(f"  avg set size:      {cal_eval['avg_set_size']:.2f}")
    print(f"  median set size:   {cal_eval['median_set_size']:.1f}")
    print(f"  empty set rate:    {cal_eval['empty_set_rate']:.3f}")
    print(f"  full set rate:     {cal_eval['full_set_rate']:.3f}")
    print(f"  size distribution: {cal_eval['size_distribution']}")
    print(f"  per-disease:")
    for d, info in cal_eval["per_disease"].items():
        if info["recall"] is not None:
            print(f"    {d:20s}  n={info['n_positive']:5d}  recall={info['recall']:.4f}  pred_rate={info['pred_rate']:.4f}")

    save_lamhat_json(
        out_path=lamhat_path,
        lamhat=lamhat,
        alpha=alpha,
        calib_csv=str(calib_csv.relative_to(data_dir.parent)),
        n_total=len(cal_paths),
        n_pos=int(cal_pos_mask.sum()),
        batch_size=batch_size,
    )

    art_dir.mkdir(parents=True, exist_ok=True)
    np.save(art_dir / "cal_sgmd.npy", cal_sgmd)
    print(f"Saved calibration scores -> {art_dir / 'cal_sgmd.npy'}")

    # -------------------- TEST --------------------
    test_paths, test_labels, test_pos_mask = load_paths_labels(
    test_csv, sample_n=TEST_SAMPLE_N, seed=SEED, images_root=images_root_1024
)

    print(f"\n======================= TEST =======================")
    print(f"  N paths: {len(test_paths)}")
    print(f"  positive rows: {int(test_pos_mask.sum())} / {len(test_pos_mask)}"
          f"  ({test_pos_mask.mean():.1%})")

    print("\nRunning test inference...")
    test_sgmd = infer_all_probs(model, config, test_paths, batch_size=batch_size)

    test_pred_sets = test_sgmd >= lamhat
    test_eval = evaluate_prediction_sets(test_pred_sets, test_labels, test_pos_mask)

    print(f"\n── Test Results (lamhat={lamhat:.6f}) ──")
    print(f"  test FNR:          {test_eval['fnr']:.4f}")
    print(f"  avg set size:      {test_eval['avg_set_size']:.2f}")
    print(f"  median set size:   {test_eval['median_set_size']:.1f}")
    print(f"  empty set rate:    {test_eval['empty_set_rate']:.3f}")
    print(f"  full set rate:     {test_eval['full_set_rate']:.3f}")
    print(f"  size distribution: {test_eval['size_distribution']}")
    print(f"  per-disease:")
    for d, info in test_eval["per_disease"].items():
        if info["recall"] is not None:
            print(f"    {d:20s}  n={info['n_positive']:5d}  recall={info['recall']:.4f}  pred_rate={info['pred_rate']:.4f}")

    np.save(art_dir / "test_sgmd.npy", test_sgmd)
    print(f"Saved test scores -> {art_dir / 'test_sgmd.npy'}")

    # Save full evaluation as JSON for the report
    eval_path = art_dir / "evaluation.json"
    eval_payload = {
        "alpha": alpha,
        "lamhat": lamhat,
        "calibration": cal_eval,
        "test": test_eval,
        "n_cal": len(cal_paths),
        "n_test": len(test_paths),
    }
    # Convert numpy types for JSON
    def _convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, dict): return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_convert(v) for v in obj]
        return obj

    with open(eval_path, "w") as f:
        json.dump(_convert(eval_payload), f, indent=2)
    print(f"Saved evaluation -> {eval_path}")

    # ── Sample prediction sets ──
    print(f"\n[Test] 10 example prediction sets:")
    rng = np.random.default_rng(SEED)
    k = min(10, len(test_paths))
    idxs = rng.choice(len(test_paths), size=k, replace=False)

    for i in idxs:
        pred_names = prediction_set_names(test_sgmd[i], lamhat)
        true_names = [DISEASES[j] for j in range(5) if test_labels[i, j] == 1]

        print(f"\n  idx {i}")
        print(f"    Path: {test_paths[i]}")
        print(f"    True: {true_names}")
        print(f"    Probs: {np.round(test_sgmd[i], 4)}")
        print(f"    Prediction set ({len(pred_names)}): {pred_names}")
