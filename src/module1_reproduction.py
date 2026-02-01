import os
import sys
import json
import torch
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CLASSES = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]

# Get project root directory (go one level up) and point to the Model1 folder

# Get project root folder (the folder above ip_implemntation)
BASE_DIR = os.path.dirname(__file__)

# Model1 path

MODEL1_DIR = os.path.join(BASE_DIR, "Model1")
MODEL1_MODEL_DIR = os.path.join(MODEL1_DIR, "model")


CHEXPERT_DIR = os.path.join(BASE_DIR, "CheXpert1", "CheXpert-v1.0-small")

sys.path.append(MODEL1_DIR)
sys.path.append(MODEL1_MODEL_DIR)


# Now the import will work
from classifier import Classifier


def load_model1():

    # extracting the model infor from the config
    config_path = os.path.join(MODEL1_DIR, "config", "example.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    class Config: 
        pass

    config = Config()
    for k, v in cfg.items():
        setattr(config, k, v)

    model = Classifier(config)

    weights_path = os.path.join(MODEL1_DIR, "config", "pre_train.pth")
    state_dict = torch.load(weights_path, map_location="cpu") 
    model.load_state_dict(state_dict, strict=True) # putting the weights inside the model 


    print("Model1 loaded successfully.")
    return model, config

def preprocess_image(img_path, config):

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (config.width, config.height))

    if config.use_equalizeHist:
        img = cv2.equalizeHist(img)

    img = img.astype(np.float32)

    img = (img - config.pixel_mean) / config.pixel_std

    img = np.stack([img, img, img], axis=0)   
    img = np.expand_dims(img, axis=0)         

    return torch.tensor(img, dtype=torch.float32)

def predict(model, config, img_paths):

    if isinstance(img_paths, str):
        img_paths = [img_paths]

    model.eval()
    final_outputs = []

    for p in img_paths:
        x = preprocess_image(p, config)

        with torch.no_grad():
            outputs, _ = model(x)  # outputs can be list or tensor

            # If the model returns a list of tensors [B,1] x 5 → concat to [B,5]
            if isinstance(outputs, list):
                logits = torch.cat(outputs, dim=1)   # shape: [B, 5]
            elif isinstance(outputs, torch.Tensor):
                logits = outputs                      # already a tensor
            else:
                raise TypeError(f"Unexpected model output type: {type(outputs)}")

            probs = torch.sigmoid(logits)             # shape: [B, 5]
            probs = probs.cpu().numpy().flatten().tolist()

        final_outputs.append(probs)

    return final_outputs

def evaluate_model(model, config):
    # Use your downloaded CheXpert small dataset
    dev_path = os.path.join(CHEXPERT_DIR, "valid.csv")

    df = pd.read_csv(dev_path)

    y_true = []
    y_pred = []

    for idx, row in df.iterrows():
        # Correct image path
        img_path = os.path.join(BASE_DIR, "CheXpert1", row["Path"])

        # Prediction
        probs = predict(model, config, img_path)[0]

        # Ground truth labels (map -1 → 0 simply)
        labels = [
            0 if row["Cardiomegaly"] == -1 else int(row["Cardiomegaly"]),
            0 if row["Edema"] == -1 else int(row["Edema"]),
            0 if row["Consolidation"] == -1 else int(row["Consolidation"]),
            0 if row["Atelectasis"] == -1 else int(row["Atelectasis"]),
            0 if row["Pleural Effusion"] == -1 else int(row["Pleural Effusion"])
        ]

        y_true.append(labels)
        y_pred.append(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    aucs = []
    for i in range(5):
        auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        aucs.append(auc)

    return aucs

if __name__ == "__main__":
 model, config = load_model1()
 auc_scores = evaluate_model(model, config)

 for name, score in zip(CLASSES, auc_scores):
    print(f"{name}: {score:.4f}") 

    mean_auc = sum(auc_scores) / len(auc_scores)
print(f"\nMean AUC: {mean_auc:.4f}")