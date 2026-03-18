export default function DeveloperRequirementsCards() {
  return (
    <div className="dev-req-grid">
      <div className="dev-req-card panel">
        <div className="dev-req-card-header">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 002 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0022 16z" />
          </svg>
          <h4 className="dev-req-card-title">Pre-trained model</h4>
        </div>
        <p className="dev-req-text">
          Upload a supported pre-trained model file. SafeDx expects a model that accepts
          <code>(B, 3, H, W)</code> input tensors and returns <code>(B, n_classes)</code> multilabel logits.
        </p>
        <table className="dev-req-table">
          <thead><tr><th>Format</th><th>Status</th></tr></thead>
          <tbody>
            <tr><td>TorchScript <code>.pt</code></td><td className="dev-req-ok">Accepted (recommended)</td></tr>
            <tr><td>Full model <code>.pth</code></td><td className="dev-req-ok">Accepted</td></tr>
            <tr><td>State dict <code>.pth</code></td><td className="dev-req-no">Not accepted</td></tr>
          </tbody>
        </table>
        <details className="dev-req-convert">
          <summary>Have a state dict? Here&apos;s how to convert it</summary>
          <pre className="dev-format-pre">{`import torch

# 1. Load your model architecture + weights
model = YourModel()
model.load_state_dict(torch.load("weights.pth", map_location="cpu"))
model.eval()

# 2. Trace with a dummy input matching your image size
dummy = torch.zeros(1, 3, H, W)
traced = torch.jit.trace(model, dummy)

# 3. Save — upload the resulting .pt file
torch.jit.save(traced, "model.pt")`}</pre>
        </details>
      </div>

      <div className="dev-req-card panel">
        <div className="dev-req-card-header">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
            <polyline points="14 2 14 8 20 8" />
          </svg>
          <h4 className="dev-req-card-title">Calibration dataset (.zip)</h4>
        </div>
        <p className="dev-req-text">
          Provide an unseen calibration dataset zip. The model should not have seen these examples during training.
          The archive must include an <code>images/</code> folder and a <code>labels.csv</code> file.
        </p>
        <pre className="dev-format-pre">{`dataset.zip
├── images/
│   ├── img001.png
│   ├── img002.jpg
│   └── ...
└── labels.csv`}</pre>
        <p className="dev-req-text" style={{ marginTop: 8 }}><strong>labels.csv</strong> — first column must be <code>filename</code>, all other columns are treated as labels:</p>
        <pre className="dev-format-pre">{`filename,LabelA,LabelB,LabelC
img001.png,1,0,0
img002.jpg,0,1,1`}</pre>
        <ul className="dev-req-rules">
          <li>Label values must be <strong>0</strong> or <strong>1</strong></li>
          <li>Minimum <strong>50</strong> labelled images</li>
          <li>Every filename must match a file in <code>images/</code></li>
          <li>Number of label columns must match the model&apos;s output classes</li>
          <li>Max size: <strong>2 GB</strong></li>
        </ul>
      </div>

      <div className="dev-req-card panel">
        <div className="dev-req-card-header">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="3" />
            <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z" />
          </svg>
          <h4 className="dev-req-card-title">Config (.json) — optional</h4>
        </div>
        <p className="dev-req-text">
          If your model resizes images or applies normalisation before prediction, provide a matching config JSON so SafeDx can preprocess the calibration dataset correctly.
        </p>
        <pre className="dev-format-pre">{`{
  "width": 512,
  "height": 512,
  "pixel_mean": 128.0,
  "pixel_std": 64.0,
  "use_equalizeHist": true
}`}</pre>
        <table className="dev-req-table">
          <thead><tr><th>Field</th><th>Required</th></tr></thead>
          <tbody>
            <tr><td><code>width</code></td><td>Yes</td></tr>
            <tr><td><code>height</code></td><td>Yes</td></tr>
            <tr><td><code>pixel_mean</code></td><td>Yes</td></tr>
            <tr><td><code>pixel_std</code></td><td>Yes</td></tr>
            <tr><td><code>use_equalizeHist</code></td><td>No</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
