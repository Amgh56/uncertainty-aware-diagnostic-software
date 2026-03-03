import { useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { API_URL } from "../api/diagnosticApi";
import UploadCard from "./developer/UploadCard";
import JobsTable from "./developer/JobsTable";

export default function DeveloperDashboard() {
  const { token, doctor, logout } = useAuth();
  const navigate = useNavigate();

  const [modelFile, setModelFile] = useState(null);
  const [configFile, setConfigFile] = useState(null);
  const [datasetFile, setDatasetFile] = useState(null);
  const [alpha, setAlpha] = useState("0.1");
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState(null);
  const [submitSuccess, setSubmitSuccess] = useState(null);
  // Incremented to force JobsTable to refresh after a new job is created
  const [jobsKey, setJobsKey] = useState(0);

  const canSubmit = modelFile && datasetFile && !submitting; // config is optional

  async function handleSubmit(e) {
    e.preventDefault();
    if (!canSubmit) return;

    const alphaNum = parseFloat(alpha);
    if (isNaN(alphaNum) || alphaNum <= 0 || alphaNum >= 1) {
      setSubmitError("Alpha must be a number strictly between 0 and 1 (e.g. 0.1).");
      return;
    }

    setSubmitting(true);
    setSubmitError(null);
    setSubmitSuccess(null);

    try {
      const form = new FormData();
      form.append("model_file", modelFile);
      if (configFile) form.append("config_file", configFile);
      form.append("dataset_file", datasetFile);
      form.append("alpha", alphaNum);

      const res = await fetch(`${API_URL}/developer/jobs?alpha=${alphaNum}`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body: form,
      });

      if (!res.ok) {
        let detail = "Failed to create calibration job.";
        try { detail = (await res.json()).detail ?? detail; } catch {}
        throw new Error(detail);
      }

      const job = await res.json();
      setSubmitSuccess(`Job ${job.id.slice(0, 8)} queued successfully.`);
      setModelFile(null);
      setConfigFile(null);
      setDatasetFile(null);
      setJobsKey((k) => k + 1);
    } catch (err) {
      setSubmitError(err.message);
    } finally {
      setSubmitting(false);
    }
  }

  // Called by JobsTable when all active jobs reach a terminal state
  const handleJobsSettled = useCallback(() => {}, []);

  return (
    <div className="dashboard-root">
      {/* ── Header ── */}
      <header className="dash-header">
        <div className="dash-header-inner">
          <div className="dash-header-left">
            <div className="dash-logo-icon">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
              </svg>
            </div>
            <div>
              <h1 className="dash-header-title">Uncertainty-Aware Diagnostic System</h1>
              <p className="dash-header-subtitle">Developer / Researcher mode</p>
            </div>
          </div>
          <div className="dash-header-right" style={{ display: "flex", gap: 10 }}>
            <button className="nav-btn" onClick={() => navigate("/home")} style={{ color: "var(--text-soft)" }}>
              Clinician view
            </button>
            <button className="nav-btn logout-btn" onClick={logout}>Logout</button>
          </div>
        </div>
      </header>

      <main className="dash-main">
        {/* ── Page heading ── */}
        <div className="home-top-bar" style={{ marginBottom: 8 }}>
          <div>
            <h2 className="home-welcome">
              Welcome, {doctor?.full_name?.split(" ")[0] ?? "Researcher"}
            </h2>
            <p className="home-welcome-sub">
              Upload a pretrained model and labelled calibration dataset to run the
              conformal calibration pipeline and download a calibrated threshold (λ̂).
            </p>
          </div>
        </div>

        {/* ── Upload requirements ── */}
        <div className="dev-req-grid">
          {/* Model format */}
          <div className="dev-req-card panel">
            <div className="dev-req-card-header">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 002 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0022 16z" />
              </svg>
              <h4 className="dev-req-card-title">Model file (.pt / .pth)</h4>
            </div>
            <p className="dev-req-text">
              Your model must take <code>(B, 3, H, W)</code> input and return <code>(B, n_classes)</code> logits.
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
              <summary>Have a state dict? Here's how to convert it</summary>
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

          {/* Dataset format */}
          <div className="dev-req-card panel">
            <div className="dev-req-card-header">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
                <polyline points="14 2 14 8 20 8" />
              </svg>
              <h4 className="dev-req-card-title">Dataset zip (.zip)</h4>
            </div>
            <p className="dev-req-text">
              A single zip archive containing an <code>images/</code> folder and a <code>labels.csv</code> file.
              A wrapper folder inside the zip is OK.
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
              <li>Number of label columns must match the model's output classes</li>
              <li>Max size: <strong>2 GB</strong></li>
            </ul>
          </div>

          {/* Config format */}
          <div className="dev-req-card panel">
            <div className="dev-req-card-header">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="3" />
                <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z" />
              </svg>
              <h4 className="dev-req-card-title">Config (optional, .json)</h4>
            </div>
            <p className="dev-req-text">
              Upload if your model expects a specific image size or normalisation.
              If omitted, images are used at their native size with no preprocessing.
            </p>
            <pre className="dev-format-pre">{`{
  "width": 512,
  "height": 512,
  "pixel_mean": 128.0,
  "pixel_std": 64.0,
  "use_equalizeHist": true
}`}</pre>
            <table className="dev-req-table">
              <thead><tr><th>Field</th><th>Required</th><th>Description</th></tr></thead>
              <tbody>
                <tr><td><code>width</code></td><td>Yes</td><td>Resize width (px)</td></tr>
                <tr><td><code>height</code></td><td>Yes</td><td>Resize height (px)</td></tr>
                <tr><td><code>pixel_mean</code></td><td>Yes</td><td>Pixel mean for normalisation</td></tr>
                <tr><td><code>pixel_std</code></td><td>Yes</td><td>Pixel std for normalisation</td></tr>
                <tr><td><code>use_equalizeHist</code></td><td>No</td><td>Apply histogram equalisation (default: false)</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* ── Upload form ── */}
        <form onSubmit={handleSubmit} className="dev-upload-form panel">
          <h3 className="dev-section-title">New Calibration Job</h3>

          <div className="dev-upload-row">
            <UploadCard
              title="Model File"
              accept=".pth,.pt"
              hint="TorchScript (.pt) or full saved model (.pth). Max 500 MB."
              file={modelFile}
              onChange={setModelFile}
              disabled={submitting}
            />
            <UploadCard
              title="Config (optional)"
              accept=".json"
              hint="Preprocessing config — image size + normalisation. See requirements above."
              file={configFile}
              onChange={setConfigFile}
              disabled={submitting}
            />
            <UploadCard
              title="Calibration Dataset"
              accept=".zip"
              hint="ZIP with images/ folder + labels.csv. Min 50 images, max 2 GB."
              file={datasetFile}
              onChange={setDatasetFile}
              disabled={submitting}
            />
          </div>

          {/* Alpha input */}
          <div className="dev-alpha-row">
            <label htmlFor="alpha-input" className="dev-alpha-label">
              Miscoverage rate α
              <span className="dev-alpha-hint"> — target false-negative rate (e.g. 0.1 = 90% coverage)</span>
            </label>
            <input
              id="alpha-input"
              type="number"
              className="dev-alpha-input"
              value={alpha}
              min="0.01"
              max="0.99"
              step="0.01"
              onChange={(e) => setAlpha(e.target.value)}
              disabled={submitting}
            />
          </div>

          {/* Feedback */}
          {submitError && (
            <div className="error-bar">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#dc2626" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <line x1="15" y1="9" x2="9" y2="15" />
                <line x1="9" y1="9" x2="15" y2="15" />
              </svg>
              <span>{submitError}</span>
            </div>
          )}
          {submitSuccess && (
            <div className="error-bar" style={{ borderColor: "#bbf7d0", background: "#f0fdf4", color: "#059669" }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#059669" strokeWidth="2">
                <polyline points="20 6 9 17 4 12" />
              </svg>
              <span>{submitSuccess}</span>
            </div>
          )}

          <button
            type="submit"
            className="home-new-btn"
            disabled={!canSubmit}
            style={{ alignSelf: "flex-start", marginTop: 4 }}
          >
            {submitting ? (
              <>
                <span className="spinner-small" />
                Uploading…
              </>
            ) : (
              <>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <polygon points="5 3 19 12 5 21 5 3" />
                </svg>
                Run Calibration
              </>
            )}
          </button>
        </form>

        {/* ── Jobs history ── */}
        <div className="dev-jobs-section">
          <h3 className="dev-section-title" style={{ marginBottom: 12 }}>Calibration Jobs</h3>
          <JobsTable key={jobsKey} token={token} onNewJob={handleJobsSettled} />
        </div>
      </main>
    </div>
  );
}
