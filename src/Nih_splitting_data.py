import os
import numpy as np
import pandas as pd

INPUT_CSV = "NIH_dataset/nih_chexpert_like.csv"

DISEASES = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]

TARGET_POS_PER_DISEASE = 500
DESIRED_CALIB_SIZE = 2500
SEED = 42

# So that if we take a patient image we remove it from the whole dataset to avoid data leakage 
USE_PATIENT_EXCLUSION = True 

# returns the first part of the path 
def extract_patient_id(path: str) -> str:
    base = os.path.basename(str(path))
    return base.split("_")[0]

#
def make_constrained_calibration_set(
    df: pd.DataFrame,
    diseases: list[str],
    target_pos: int,
    desired_size: int,
    seed: int = 42,
    use_patient_exclusion: bool = True,
):
    rng = np.random.default_rng(seed)
    pos = (df[diseases].fillna(0).astype(float).values == 1.0).astype(np.int8)
    n, k = pos.shape

    total_pos = pos.sum(axis=0)
    for j, d in enumerate(diseases):
        if total_pos[j] < target_pos:
            raise ValueError(
                f"Not enough positives for {d}: have {total_pos[j]}, need {target_pos}"
            )

    remaining_mask = np.ones(n, dtype=bool)
    selected_mask = np.zeros(n, dtype=bool)

    counts = np.zeros(k, dtype=int)

    pos_indices = [np.where(pos[:, j] == 1)[0] for j in range(k)]

    max_iters = desired_size * 5  # safety
    iters = 0

    while (counts < target_pos).any():
        iters += 1
        if iters > max_iters:
            raise RuntimeError(
                "Selection got stuck. Try increasing desired_size or lowering target_pos."
            )

        deficits = target_pos - counts
        j = int(np.argmax(deficits)) 

        cand = pos_indices[j]
        cand = cand[remaining_mask[cand]]
        if cand.size == 0:
            raise RuntimeError(f"No remaining candidates to satisfy {diseases[j]} deficit.")

        underfilled = (counts < target_pos)
        gains = pos[cand][:, underfilled].sum(axis=1)

        best_gain = gains.max()
        best = cand[gains == best_gain]

        chosen = int(rng.choice(best))

        selected_mask[chosen] = True
        remaining_mask[chosen] = False
        counts += pos[chosen]

    selected_idx = np.where(selected_mask)[0].tolist()

    if len(selected_idx) > desired_size:
        rng.shuffle(selected_idx)
        counts_now = pos[selected_idx].sum(axis=0)

        keep = set(selected_idx)
        for idx in selected_idx:
            if len(keep) <= desired_size:
                break
            if np.all(counts_now - pos[idx] >= target_pos):
                keep.remove(idx)
                counts_now -= pos[idx]

        selected_idx = sorted(list(keep))

    if len(selected_idx) < desired_size:
        remaining_idx = np.where(remaining_mask)[0]
        need = desired_size - len(selected_idx)
        if need > remaining_idx.size:
            raise RuntimeError("Not enough remaining samples to top up to desired_size.")
        extra = rng.choice(remaining_idx, size=need, replace=False).tolist()
        selected_idx = sorted(selected_idx + extra)

    calib_df = df.iloc[selected_idx].copy()

    if use_patient_exclusion and "Path" in df.columns:
        calib_df["_patient_id"] = calib_df["Path"].apply(extract_patient_id)
        heldout_patients = set(calib_df["_patient_id"].unique().tolist())

        rest_df = df.copy()
        rest_df["_patient_id"] = rest_df["Path"].apply(extract_patient_id)
        rest_df = rest_df[~rest_df["_patient_id"].isin(heldout_patients)].copy()
        calib_df.drop(columns=["_patient_id"], inplace=True)
        rest_df.drop(columns=["_patient_id"], inplace=True)
    else:
        rest_df = df.drop(index=calib_df.index).copy()

    calib_pos = (calib_df[diseases].fillna(0).astype(float).values == 1.0).sum(axis=0)

    report = {
        "calib_size": len(calib_df),
        "rest_size": len(rest_df),
        "positives_per_disease_in_calib": {diseases[i]: int(calib_pos[i]) for i in range(k)},
    }

    return calib_df, rest_df, report


if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)

    calib_df, rest_df, report = make_constrained_calibration_set(
        df=df,
        diseases=DISEASES,
        target_pos=TARGET_POS_PER_DISEASE,
        desired_size=DESIRED_CALIB_SIZE,
        seed=SEED,
        use_patient_exclusion=USE_PATIENT_EXCLUSION,
    )

    calib_path = "NIH_dataset/calibration_2500.csv"
    rest_path = "NIH_dataset/remaining_after_calib.csv"

    calib_df.to_csv(calib_path, index=False)
    rest_df.to_csv(rest_path, index=False)

    print("Saved:")
    print(f"  - {calib_path}  (held-out calibration set)")
    print(f"  - {rest_path}   (use this for train/val/test splits)")
    print("\nReport:")
    print(report)

