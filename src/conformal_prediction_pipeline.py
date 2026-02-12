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
):
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
    for p in rel_paths:
        rel = Path(p)
        if rel.is_absolute():
            candidate = rel
        else:
            candidates = [
                repo_root / rel,
                SCRIPT_DIR / rel,
                ROOT_DIR / rel,
            ]
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
    if torch.cuda.is_available():
        return "cuda"
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return "cpu"
    return "mps"

def preprocess_image(img_path: Path, config) -> np.ndarray:

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
    batch_np = np.stack([preprocess_image(p, config) for p in img_paths], axis=0) 
    x = torch.tensor(batch_np, dtype=torch.float32, device=device)
    return x

def predict_batch_probs(model, x):

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
    """
    Step 2.4:
    Runs inference on ALL img_paths in batches and returns:
      sgmd: np.ndarray of shape (N,5)

    Uses:
      - pick_device()
      - make_batch()
      - predict_batch_probs()
    """
    device = pick_device()
    model = model.to(device)

    all_probs = []
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i+batch_size]
        x = make_batch(batch_paths, config, device)
        probs = predict_batch_probs(model, x)  # (B,5) numpy
        all_probs.append(probs)

        if (i // batch_size) % 10 == 0:
            print(f"Inferred {i}/{len(img_paths)}")

    sgmd = np.concatenate(all_probs, axis=0)  # (N,5)
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
    # Only use rows with >=1 true positive (your rule)
    sgmd = cal_sgmd[cal_pos_mask]
    labels = cal_labels[cal_pos_mask]
    n = sgmd.shape[0]

    target = ((n + 1) / n) * alpha - (1 / n)

    # Candidate thresholds: all unique predicted probabilities in calibration
    candidates = np.unique(sgmd)
    candidates.sort()

    lamhat = candidates[0]  # default smallest threshold (largest sets)

    for lam in candidates:
        pred_set = sgmd >= lam
        fnr = false_negative_rate(pred_set, labels, np.ones(n, dtype=bool))  # mask already applied
        if fnr <= target:
            lamhat = lam
        else:
            break

    return float(lamhat)

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
    CAL_SAMPLE_N = 2500     
    TEST_SAMPLE_N = 5000     
    SEED = 0

    data_dir = Path(__file__).resolve().parent / "NIH_dataset"
    calib_csv = data_dir / "calibration_2500.csv"
    test_csv  = data_dir / "remaining_after_calib.csv"
    art_dir   = data_dir / "artifacts"
    lamhat_path = art_dir / "lamhat.json"

    # -------------------- CALIBRATION --------------------
    cal_paths, cal_labels, cal_pos_mask = load_paths_labels(
        calib_csv, sample_n=CAL_SAMPLE_N, seed=SEED
    )

    print("\n==================== CALIBRATION ====================")
    print("Loaded calibration:")
    print("  N paths:", len(cal_paths))
    print("  labels shape:", cal_labels.shape)
    print("  positive rows (used for FNR):", int(cal_pos_mask.sum()), "/", len(cal_pos_mask))
    print("  example path:", cal_paths[0])
    print("  example labels:", dict(zip(DISEASES, cal_labels[0].tolist())))

    model, config = load_chexpert_pretrained_model()
    device = pick_device()
    print("\nDevice selected:", device)

    cal_sgmd = infer_all_probs(model, config, cal_paths, batch_size=batch_size)

    print("\nCalibration inference output:")
    print("  cal_sgmd shape:", cal_sgmd.shape)
    print("  cal_sgmd min/max:", float(cal_sgmd.min()), float(cal_sgmd.max()))
    print("  first probs row:", np.round(cal_sgmd[0], 4))

    lamhat = compute_lamhat(cal_sgmd, cal_labels, cal_pos_mask, alpha)
    fnr_cal = false_negative_rate((cal_sgmd >= lamhat), cal_labels, cal_pos_mask)

    print("\nCalibration results:")
    print("  alpha:", alpha)
    print("  lamhat:", lamhat)
    print("  calibration FNR at lamhat:", fnr_cal)

    # Save lamhat for reuse
    save_lamhat_json(
        out_path=lamhat_path,
        lamhat=lamhat,
        alpha=alpha,
        calib_csv="NIH_dataset/calibration_2500.csv",
        n_total=len(cal_paths),
        n_pos=int(cal_pos_mask.sum()),
        batch_size=batch_size,
    )

    # Optional: cache calibration scores (useful on Iridis to avoid recompute)
    art_dir.mkdir(parents=True, exist_ok=True)
    np.save(art_dir / "cal_sgmd.npy", cal_sgmd)
    print(f"Saved calibration scores -> {art_dir / 'cal_sgmd.npy'}")

    # -------------------- TEST --------------------
    test_paths, test_labels, test_pos_mask = load_paths_labels(
        test_csv, sample_n=TEST_SAMPLE_N, seed=SEED
    )

    print("\n======================= TEST =======================")
    print("Loaded test:")
    print("  N paths:", len(test_paths))
    print("  positive rows (used for FNR):", int(test_pos_mask.sum()), "/", len(test_pos_mask))

    test_sgmd = infer_all_probs(model, config, test_paths, batch_size=batch_size)

    fnr_test = false_negative_rate((test_sgmd >= lamhat), test_labels, test_pos_mask)

    print("\nTest results (using calibration lamhat):")
    print("  test FNR:", fnr_test)

    np.save(art_dir / "test_sgmd.npy", test_sgmd)
    print(f"Saved test scores -> {art_dir / 'test_sgmd.npy'}")

    print("\n[Test] 10 example prediction sets:")
    rng = np.random.default_rng(SEED)
    k = min(10, len(test_paths))
    idxs = rng.choice(len(test_paths), size=k, replace=False)

    for i in idxs:
        pred_names = prediction_set_names(test_sgmd[i], lamhat)
        true_names = [DISEASES[j] for j in range(5) if test_labels[i, j] == 1]

        print(f"\nExample idx {i}")
        print("  Path:", test_paths[i])
        print("  True:", true_names)
        print("  Probs:", np.round(test_sgmd[i], 4))
        print("  Prediction set:", pred_names)
