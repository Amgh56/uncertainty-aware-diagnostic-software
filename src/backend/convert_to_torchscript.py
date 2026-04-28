import json
import sys
from pathlib import Path

import torch

# --- paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
CHEXPERT_DIR = ROOT_DIR / "Chexpert"
CONFIG_DIR = CHEXPERT_DIR / "config"

STATE_DICT_PATH = CONFIG_DIR / "pre_train.pth"
OUTPUT_PATH = CONFIG_DIR / "pre_train_torchscript.pt"
EXAMPLE_JSON = CONFIG_DIR / "example.json"

sys.path.insert(0, str(CHEXPERT_DIR))
from model.classifier import Classifier  


class _InferenceWrapper(torch.nn.Module):
    """
    Thin wrapper that returns a single logits tensor of shape (B, n_classes).

    Many models return tuples (logits, feature_maps) or lists — TorchScript
    trace cannot handle empty lists or heterogeneous tuples.  This wrapper
    collapses the output into a single tensor so tracing always succeeds.

    NOTE: returns raw logits (no sigmoid) — the calibration pipeline applies
    sigmoid itself.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            logits = out[0]
            if isinstance(logits, list):
                logits = torch.cat(logits, dim=1)
        else:
            logits = out
        return logits


def main():
    # 1. Load config
    with open(EXAMPLE_JSON) as f:
        cfg_dict = json.load(f)

    class Config:
        pass

    config = Config()
    for k, v in cfg_dict.items():
        setattr(config, k, v)

    # 2. Build model + load state dict
    model = Classifier(config)
    state_dict = torch.load(str(STATE_DICT_PATH), map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"Loaded state dict from {STATE_DICT_PATH}")

    # 3. Wrap so the output is a single tensor (traceable)
    wrapper = _InferenceWrapper(model)
    wrapper.eval()

    # 4. Trace with a dummy input matching the model's expected size
    dummy = torch.zeros(1, 3, config.height, config.width)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, dummy)

    # 5. Save as TorchScript
    torch.jit.save(traced, str(OUTPUT_PATH))
    print(f"Saved TorchScript model → {OUTPUT_PATH}")
    print(f"Size: {OUTPUT_PATH.stat().st_size / (1024*1024):.1f} MB")
    print()
    print("You can now upload this .pt file to the Developer dashboard.")


if __name__ == "__main__":
    main()
