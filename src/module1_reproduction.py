import json
import sys
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from pathlib import Path

CLASSES = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
CHEXPERT_CODE_DIR = ROOT_DIR / "Chexpert"
CONFIG_DIR = CHEXPERT_CODE_DIR / "config"
DATA_DIR = SCRIPT_DIR / "CheXpert1"
VALID_CSV = DATA_DIR / "valid.csv"


sys.path.insert(0, str(CHEXPERT_CODE_DIR))
from model.classifier import Classifier

# function purpose: Load config + build the model + load pretrained weights.
def load_model1():
    config_path = CONFIG_DIR / "example.json"
    with open(config_path, "r") as f:
        cfg = json.load(f)

    class Config:
        pass

    config = Config()
    for k, v in cfg.items():
        setattr(config, k, v)

    model = Classifier(config)
    weights_path = CONFIG_DIR / "pre_train.pth"
    state_dict = torch.load(str(weights_path), map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    print("Model1 loaded successfully.")
    return model, config

# transform an image to tensor sensor 
def preprocess_image(img_path: Path, config):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    img = cv2.resize(img, (config.width, config.height))

    if getattr(config, "use_equalizeHist", False):
        img = cv2.equalizeHist(img)

    img = img.astype(np.float32)
    img = (img - config.pixel_mean) / config.pixel_std
    img = np.stack([img, img, img], axis=0)
    img = np.expand_dims(img, axis=0)

    return torch.tensor(img, dtype=torch.float32)

# here where we take a image and predict 
def predict(model, config, img_paths):
    if isinstance(img_paths, (str, Path)):
        img_paths = [img_paths]

    model.eval()
    final_outputs = []

    for p in img_paths:
        p = Path(p)
        x = preprocess_image(p, config)

        with torch.no_grad():
            outputs, _ = model(x)

            if isinstance(outputs, list):
                logits = torch.cat(outputs, dim=1) 
            elif isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                raise TypeError(f"Unexpected model output type: {type(outputs)}")

            probs = torch.sigmoid(logits).cpu().numpy().flatten().tolist()

        final_outputs.append(probs)

    return final_outputs


def evaluate_model(model, config):
    df = pd.read_csv(VALID_CSV)

    y_true, y_pred = [], []

    for _, row in df.iterrows():
        rel = str(row["Path"]).replace("CheXpert-v1.0-small/", "")
        img_path = DATA_DIR / rel  

        probs = predict(model, config, img_path)[0]
        labels = []
        for name in CLASSES:
            val = row[name]
            if pd.isna(val):
                labels.append(0)
            elif val == -1:
                labels.append(0)
            else:
                labels.append(int(val))

        y_true.append(labels)
        y_pred.append(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    aucs = []
    for i in range(len(CLASSES)):
        if len(np.unique(y_true[:, i])) < 2:
            aucs.append(float("nan"))
        else:
            aucs.append(roc_auc_score(y_true[:, i], y_pred[:, i]))

    return aucs


if __name__ == "__main__":
    model, config = load_model1()
    auc_scores = evaluate_model(model, config)

    for name, score in zip(CLASSES, auc_scores):
        print(f"{name}: {score:.4f}" if not np.isnan(score) else f"{name}: NaN (only one class in labels)")

    mean_auc = np.nanmean(auc_scores)
    print(f"\nMean AUC: {mean_auc:.4f}")