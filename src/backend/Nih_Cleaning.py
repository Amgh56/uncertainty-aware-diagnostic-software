import pandas as pd
import numpy as np
from pathlib import Path

CHEX_COLUMNS = [
    "Path","Sex","Age","Frontal/Lateral","AP/PA",
    "No Finding","Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity","Lung Lesion",
    "Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax",
    "Pleural Effusion","Pleural Other","Fracture","Support Devices"
]
BASE_DIR = Path(__file__).resolve().parent
NIH_CSV = BASE_DIR / "NIH_dataset" / "Data_Entry_2017.csv"
NIH_IMAGES_REL = "NIH_dataset/images-224/images-224"

def parse_age(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    if s.endswith("Y"): s = s[:-1]
    try: return int(s)
    except: return np.nan

def normalize_sex(x):
    if pd.isna(x): return ""
    x = str(x).strip().upper()
    return {"M": "Male", "F": "Female"}.get(x, "")

def frontal_lateral(view_pos):
    if pd.isna(view_pos): return "Frontal"
    vp = str(view_pos).strip().upper()
    if "LAT" in vp: return "Lateral"
    return "Frontal"

def ap_pa(view_pos, frlat):
    if frlat != "Frontal": return ""
    if pd.isna(view_pos): return ""
    vp = str(view_pos).strip().upper()
    if vp.startswith("AP"): return "AP"
    if vp.startswith("PA"): return "PA"
    return ""

def split_findings(s):
    if pd.isna(s) or str(s).strip() == "":
        return set()
    return set(x.strip() for x in str(s).split("|") if x.strip())

# Here we are mapping each disease with 1 if it appear and 0 if other wise. 
def build_chex_labels(findings):
    out = {c: 0.0 for c in CHEX_COLUMNS if c not in ["Path","Sex","Age","Frontal/Lateral","AP/PA"]}

    if "No Finding" in findings:
        out["No Finding"] = 1.0
        return out

    if "Cardiomegaly" in findings: out["Cardiomegaly"] = 1.0
    if "Edema" in findings: out["Edema"] = 1.0
    if "Consolidation" in findings: out["Consolidation"] = 1.0
    if "Pneumonia" in findings: out["Pneumonia"] = 1.0
    if "Atelectasis" in findings: out["Atelectasis"] = 1.0
    if "Pneumothorax" in findings: out["Pneumothorax"] = 1.0
    if "Effusion" in findings: out["Pleural Effusion"] = 1.0
    if "Pleural_Thickening" in findings: out["Pleural Other"] = 1.0
    if ("Mass" in findings) or ("Nodule" in findings): out["Lung Lesion"] = 1.0
    if "Infiltration" in findings: out["Lung Opacity"] = 1.0

    return out

def main():
    df = pd.read_csv(NIH_CSV)
    rows = []

    for _, r in df.iterrows():
        findings = split_findings(r["Finding Labels"])
        labels = build_chex_labels(findings)

        frlat = frontal_lateral(r.get("View Position", ""))

        row = {
            "Path": f"{NIH_IMAGES_REL}/{r['Image Index']}",
            "Sex": normalize_sex(r.get("Patient Gender", "")),
            "Age": parse_age(r.get("Patient Age", "")),
            "Frontal/Lateral": frlat,
            "AP/PA": ap_pa(r.get("View Position", ""), frlat),
            **labels
        }
        rows.append(row)

    out = pd.DataFrame(rows)

    out["Age"] = pd.to_numeric(out["Age"], errors="coerce").astype("Int64")

    for c in CHEX_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan

    out = out[CHEX_COLUMNS]

    label_cols = [c for c in CHEX_COLUMNS if c not in ["Path","Sex","Age","Frontal/Lateral","AP/PA"]]
    out[label_cols] = out[label_cols].astype(float)

    out_path = BASE_DIR / "NIH_dataset" / "nih_chexpert_like.csv"
    out.to_csv(out_path, index=False)
    
    print("Saved:", out_path)
    print("Shape:", out.shape)
    print("Columns:", list(out.columns))
if __name__ == "__main__":
    main()