import { useState, useCallback } from "react";
import { useAuth } from "../context/AuthContext";
import { createCalibrationJob } from "./api/developerApi";
import UploadCard from "./UploadCard";
import JobsTable from "./JobsTable";
import DeveloperLayout from "./DeveloperLayout";

export default function DeveloperDashboard() {
  const { token, doctor } = useAuth();

  const [modelFile, setModelFile] = useState<File | null>(null);
  const [configFile, setConfigFile] = useState<File | null>(null);
  const [datasetFile, setDatasetFile] = useState<File | null>(null);
  const [alpha, setAlpha] = useState("0.1");
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitSuccess, setSubmitSuccess] = useState<string | null>(null);
  // Incremented to force JobsTable to refresh after a new job is created
  const [jobsKey, setJobsKey] = useState(0);

  const canSubmit = modelFile && datasetFile && !submitting; // config is optional

  async function handleSubmit(e: React.FormEvent) {
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
      const job = await createCalibrationJob(modelFile, datasetFile, configFile, alphaNum, token!);
      setSubmitSuccess(`Job ${job.id.slice(0, 8)} queued successfully.`);
      setModelFile(null);
      setConfigFile(null);
      setDatasetFile(null);
      setJobsKey((k) => k + 1);
    } catch (err) {
      setSubmitError((err as Error).message);
    } finally {
      setSubmitting(false);
    }
  }

  // Called by JobsTable when all active jobs reach a terminal state
  const handleJobsSettled = useCallback(() => {}, []);

  return (
    <DeveloperLayout
      title="Calibrate Your Model"
      subtitle="Upload the required files, configure your calibration input, and track job progress."
    >
      <section className="developer-floating-shell">
        <div className="home-top-bar" style={{ marginBottom: 8 }}>
          <div>
            <h2 className="home-welcome">
              {doctor ? `Welcome, ${doctor.full_name}` : "Welcome"}
            </h2>
            <p className="home-welcome-sub">
              Start a new calibration run for your trained multilabel model and monitor the resulting jobs below.
            </p>
          </div>
        </div>

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
              title="Calibration Dataset"
              accept=".zip"
              hint="ZIP with images/ folder + labels.csv. Min 50 images, max 2 GB."
              file={datasetFile}
              onChange={setDatasetFile}
              disabled={submitting}
            />
            <UploadCard
              title="Config (optional)"
              accept=".json"
              hint="Preprocessing config — image size + normalisation."
              file={configFile}
              onChange={setConfigFile}
              disabled={submitting}
            />
          </div>

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

        <div className="dev-jobs-section">
          <h3 className="dev-section-title" style={{ marginBottom: 12 }}>Calibration Jobs</h3>
          <JobsTable key={jobsKey} token={token!} onNewJob={handleJobsSettled} />
        </div>
      </section>
    </DeveloperLayout>
  );
}
