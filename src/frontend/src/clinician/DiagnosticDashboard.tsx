import { useCallback, useEffect, useRef, useState } from "react";
import {
  predictImage,
  regeneratePrediction,
  listClinicianModels,
  Patient,
  PredictionResponse,
  ClinicianModel,
} from "./api/clinicianApi";
import { useAuth } from "../context/AuthContext";
import PatientSelector from "./PatientSelector";
import ClinicianLayout from "./ClinicianLayout";

const uncertaintyColors: Record<string, { bg: string; text: string; border: string }> = {
  Low: { bg: "#dcfce7", text: "#166534", border: "#86efac" },
  Medium: { bg: "#fff7ed", text: "#9a3412", border: "#fdba74" },
  High: { bg: "#fef2f2", text: "#991b1b", border: "#fca5a5" },
};

const statusColors: Record<string, { bg: string; text: string; border: string }> = {
  true: { bg: "#eff6ff", text: "#1e40af", border: "#93c5fd" },
  false: { bg: "#f9fafb", text: "#6b7280", border: "#e5e7eb" },
};

const verdictStyles: Record<string, { color: string; bg: string }> = {
  good: { color: "#059669", bg: "#f0fdf4" },
  review: { color: "#d97706", bg: "#fffbeb" },
};

export default function DiagnosticDashboard() {
  const { token } = useAuth();

  // Model selection
  const [availableModels, setAvailableModels] = useState<ClinicianModel[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>("");
  const [modelsLoading, setModelsLoading] = useState(true);

  const [patient, setPatient] = useState<Patient | null>(null);
  const [image, setImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [results, setResults] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [zoom, setZoom] = useState(100);
  const [alphaInput, setAlphaInput] = useState<string>("");
  const [regenerating, setRegenerating] = useState(false);
  const [regenError, setRegenError] = useState<string | null>(null);
  const [regenSuccess, setRegenSuccess] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const selectedModel = availableModels.find((m) => m.id === selectedModelId) ?? null;

  // Fetch available models on mount
  useEffect(() => {
    if (!token) return;
    listClinicianModels(token)
      .then((res) => {
        setAvailableModels(res.models);
        // Auto-select first model if available
        if (res.models.length > 0) {
          setSelectedModelId(res.models[0].id);
        }
      })
      .catch(() => {})
      .finally(() => setModelsLoading(false));
  }, [token]);

  const handleFile = useCallback((file: File | undefined) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("Please upload an image file (PNG, JPEG)");
      return;
    }
    setImage(file);
    setError(null);
    setResults(null);
    const reader = new FileReader();
    reader.onload = (event) => setImagePreview((event.target?.result as string) || null);
    reader.readAsDataURL(file);
  }, []);

  const handleDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      setDragOver(false);
      handleFile(event.dataTransfer.files?.[0]);
    },
    [handleFile]
  );

  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => setDragOver(false);
  const handleUploadClick = () => fileInputRef.current?.click();

  const handlePredict = async () => {
    if (!image || !patient) return;

    setLoading(true);
    setError(null);

    try {
      const data = await predictImage(
        image,
        patient.id,
        token!,
        selectedModelId || undefined,
      );
      setResults(data);
      setAlphaInput(String(data.alpha));
    } catch (requestError) {
      setError((requestError as Error).message || "Failed to connect to the server");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setPatient(null);
    setImage(null);
    setImagePreview(null);
    setResults(null);
    setError(null);
    setZoom(100);
    setAlphaInput("");
    setRegenError(null);
    setRegenSuccess(null);
  };

  // Number of labels (from selected model or default 5)
  const numLabels = selectedModel?.num_labels ?? results?.findings?.length ?? 5;

  return (
    <ClinicianLayout
      title="New Diagnosis"
      subtitle={!patient
        ? "Select a patient to start your diagnosis."
        : "Upload a chest X-ray and run a prediction."}
      patientBar={patient ? (
        <div className="patient-banner-info">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2" />
            <circle cx="12" cy="7" r="4" />
          </svg>
          <span className="patient-banner-name">{patient.first_name} {patient.last_name}</span>
          <span className="patient-banner-mrn">MRN: {patient.mrn}</span>
          <button className="patient-banner-change" onClick={handleReset}>Change Patient</button>
        </div>
      ) : undefined}
    >
        {/* ══════ Step 1: Patient Selection (full-width) ══════ */}
        {!patient && (
          <section className="panel patient-step-panel">
            <div className="patient-form-wrapper">
              <PatientSelector onPatientReady={(p) => setPatient(p)} />
            </div>
          </section>
        )}

        {/* ══════ Step 2: Model + Upload + Results ══════ */}
        {patient && (
          <>
            {/* Model Selector */}
            <div className="model-selector-section">
              <div className="model-selector-header">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="3" y="7" width="18" height="14" rx="2" />
                  <path d="M7 7V5a2 2 0 012-2h6a2 2 0 012 2v2" />
                </svg>
                <h3>Select Diagnostic Model</h3>
              </div>

              {modelsLoading ? (
                <div style={{ padding: "12px 0", color: "#64748b", fontSize: "0.85rem" }}>
                  Loading available models...
                </div>
              ) : availableModels.length === 0 ? (
                <div className="model-selector-empty">
                  <p>No published models available for clinical use yet.</p>
                  <p style={{ fontSize: "0.8rem", color: "#94a3b8" }}>
                    Ask a developer to publish a model for clinician use.
                  </p>
                </div>
              ) : (
                <>
                  <select
                    className="model-selector-dropdown"
                    value={selectedModelId}
                    onChange={(e) => {
                      setSelectedModelId(e.target.value);
                      setResults(null);
                    }}
                  >
                    <option value="">-- Select a model --</option>
                    {availableModels.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name} (v{m.version}) — {m.modality}
                      </option>
                    ))}
                  </select>

                  {selectedModel && (
                    <div className="model-info-card">
                      <div className="model-info-top">
                        <span className="model-info-name">{selectedModel.name}</span>
                        <span className="model-info-version">v{selectedModel.version}</span>
                        {(() => {
                          const vs = verdictStyles[selectedModel.validation_verdict];
                          return vs ? (
                            <span style={{
                              padding: "2px 8px", borderRadius: 12, fontSize: 11,
                              fontWeight: 600, color: vs.color, background: vs.bg,
                            }}>
                              {selectedModel.validation_verdict.charAt(0).toUpperCase() +
                                selectedModel.validation_verdict.slice(1)}
                            </span>
                          ) : null;
                        })()}
                      </div>
                      <p className="model-info-desc">{selectedModel.description}</p>
                      <div className="model-info-meta">
                        <span>Modality: {selectedModel.modality}</span>
                        <span>Labels: {selectedModel.num_labels}</span>
                        <span>Alpha: {selectedModel.alpha}</span>
                        <span>Coverage: {Math.round((1 - selectedModel.alpha) * 100)}%</span>
                        {selectedModel.developer_name && (
                          <span>By: {selectedModel.developer_name}</span>
                        )}
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>

            <div className="dash-grid">
              <section className="panel">
                <div className="panel-header">
                  <div>
                    <h2 className="panel-title">{selectedModel ? selectedModel.name : "\u00A0"}</h2>
                    <p className="panel-subtitle">
                      Patient: {patient.first_name} {patient.last_name} (MRN: {patient.mrn})
                    </p>
                  </div>
                  {image && (
                    <select className="mode-select" defaultValue="original">
                      <option value="original">Original</option>
                      <option value="enhanced">Enhanced</option>
                    </select>
                  )}
                </div>

                <div className="panel-body">
                  {!imagePreview ? (
                <div
                  className={`dropzone ${dragOver ? "dropzone-active" : ""}`}
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onClick={handleUploadClick}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      handleUploadClick();
                    }
                  }}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    className="hidden-input"
                    onChange={(event) => handleFile(event.target.files?.[0])}
                  />
                  <div className="dropzone-icon">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke={dragOver ? "#2563eb" : "#94a3b8"} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                      <polyline points="17 8 12 3 7 8" />
                      <line x1="12" y1="3" x2="12" y2="15" />
                    </svg>
                  </div>
                  <p className="dropzone-title">{dragOver ? "Drop image here" : "Drag and drop a chest X-ray"}</p>
                  <p className="dropzone-hint">or click to browse files</p>
                  <p className="dropzone-formats">PNG and JPEG supported</p>
                </div>
              ) : (
                <div className="image-container">
                  <img
                    src={imagePreview}
                    alt="Uploaded chest X-ray"
                    className="xray-image"
                    style={{ transform: `scale(${zoom / 100})` }}
                  />
                </div>
              )}
            </div>

            {imagePreview && (
              <div className="panel-footer">
                <div className="zoom-controls">
                  <button className="zoom-btn" onClick={() => setZoom((value) => Math.max(50, value - 25))}>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="11" cy="11" r="8" />
                      <line x1="21" y1="21" x2="16.65" y2="16.65" />
                      <line x1="8" y1="11" x2="14" y2="11" />
                    </svg>
                  </button>
                  <button className="zoom-btn" onClick={() => setZoom((value) => Math.min(200, value + 25))}>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="11" cy="11" r="8" />
                      <line x1="21" y1="21" x2="16.65" y2="16.65" />
                      <line x1="11" y1="8" x2="11" y2="14" />
                      <line x1="8" y1="11" x2="14" y2="11" />
                    </svg>
                  </button>
                  <button className="zoom-btn" onClick={() => setZoom(100)}>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polyline points="1 4 1 10 7 10" />
                      <polyline points="23 20 23 14 17 14" />
                      <path d="M20.49 9A9 9 0 005.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 013.51 15" />
                    </svg>
                  </button>
                </div>
                <span className="zoom-label">Zoom: {zoom}%</span>
              </div>
            )}

            <div className="action-bar">
              {image && !results && (
                <button className="predict-btn" onClick={handlePredict} disabled={loading}>
                  {loading ? (
                    <>
                      <span className="spinner" />
                      Analysing...
                    </>
                  ) : (
                    <>
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
                      </svg>
                      Run Prediction
                    </>
                  )}
                </button>
              )}

              {results && results.model_info && (
                <div style={{ width: "100%", padding: "0.75rem 1rem", background: "#f8fafc", borderRadius: "12px", border: "1px solid #e2e8f0" }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "0.6rem", flexWrap: "wrap" }}>
                    <span style={{ fontSize: "0.82rem", fontWeight: 500, color: "#475569" }}>α =</span>
                    <input
                      type="number"
                      min={0.01}
                      max={0.99}
                      step={0.01}
                      value={alphaInput}
                      disabled={regenerating}
                      onChange={(e) => { setAlphaInput(e.target.value); setRegenError(null); setRegenSuccess(null); }}
                      style={{
                        width: "72px", padding: "0.38rem 0.5rem",
                        border: "1px solid #cbd5e1", borderRadius: "8px",
                        fontSize: "0.85rem", color: "#1e293b", background: "#fff",
                      }}
                    />
                    <span style={{ fontSize: "0.8rem", color: "#64748b" }}>
                      {Math.round((1 - parseFloat(alphaInput || "0.1")) * 100)}% coverage
                    </span>

                    <button
                      disabled={regenerating || alphaInput === String(results.alpha)}
                      onClick={async () => {
                        setRegenerating(true);
                        setRegenError(null);
                        setRegenSuccess(null);
                        try {
                          const result = await regeneratePrediction(results.id, parseFloat(alphaInput), token!);
                          setResults(result);
                          setAlphaInput(String(result.alpha));
                          setRegenSuccess(`Prediction set updated for α = ${result.alpha}`);
                        } catch (err: unknown) {
                          setRegenError(err instanceof Error ? err.message : "Update failed");
                        } finally {
                          setRegenerating(false);
                        }
                      }}
                      style={{
                        minWidth: "120px", padding: "0.42rem 1.1rem", fontSize: "0.82rem", fontWeight: 600,
                        color: (regenerating || alphaInput === String(results.alpha)) ? "#94a3b8" : "#fff",
                        background: (regenerating || alphaInput === String(results.alpha)) ? "#e2e8f0" : "#2563eb",
                        border: "1px solid transparent",
                        borderColor: (regenerating || alphaInput === String(results.alpha)) ? "#cbd5e1" : "#2563eb",
                        borderRadius: "8px",
                        cursor: (regenerating || alphaInput === String(results.alpha)) ? "not-allowed" : "pointer",
                        display: "flex", alignItems: "center", justifyContent: "center", gap: "0.35rem",
                        transition: "all 0.15s ease",
                      }}
                    >
                      {regenerating && <span className="spinner" style={{ width: 14, height: 14 }} />}
                      {regenerating ? "Updating..." : "Update Results"}
                    </button>

                    <button
                      onClick={handleReset}
                      style={{
                        minWidth: "120px", padding: "0.42rem 1.1rem", fontSize: "0.82rem", fontWeight: 600,
                        color: "#2563eb", background: "#fff",
                        border: "1px solid #bfdbfe", borderRadius: "8px",
                        cursor: "pointer",
                        display: "flex", alignItems: "center", justifyContent: "center",
                        transition: "all 0.15s ease",
                      }}
                    >
                      Reset
                    </button>
                  </div>

                  <p style={{ margin: "0.5rem 0 0", fontSize: "0.72rem", color: "#94a3b8", textAlign: "center" }}>
                    Adjust the alpha risk level to update the prediction set and uncertainty thresholds.
                  </p>

                  {regenError && (
                    <p style={{ margin: "0.35rem 0 0", fontSize: "0.78rem", color: "#dc2626", textAlign: "center" }}>{regenError}</p>
                  )}
                  {regenSuccess && !regenError && (
                    <p style={{ margin: "0.35rem 0 0", fontSize: "0.78rem", color: "#059669", textAlign: "center" }}>{regenSuccess}</p>
                  )}
                </div>
              )}

              {(patient || image || results) && !(results && results.model_info) && (
                <button className="reset-btn" onClick={handleReset}>
                  Reset
                </button>
              )}
            </div>
          </section>

          <section className="panel">
            <div className="panel-header">
              <div className="results-header">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="10" />
                  <line x1="12" y1="16" x2="12" y2="12" />
                  <line x1="12" y1="8" x2="12.01" y2="8" />
                </svg>
                <div>
                  <h2 className="panel-title">Model Predictions and Uncertainty</h2>
                  <p className="panel-subtitle">
                    {results?.model_info
                      ? `Using: ${results.model_info.name} (v${results.model_info.version})`
                      : "Prediction set generated by the diagnostic support model"}
                  </p>
                </div>
              </div>
            </div>

            <div className="panel-body">
              {!results && !loading && (
                <div className="empty-state">
                  <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="#cbd5e1" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                    <line x1="3" y1="9" x2="21" y2="9" />
                    <line x1="9" y1="21" x2="9" y2="9" />
                  </svg>
                  <p className="empty-title">No predictions yet</p>
                  <p className="empty-hint">Upload a chest X-ray and click &quot;Run Prediction&quot; to generate a conformal prediction set.</p>
                </div>
              )}

              {loading && (
                <div className="empty-state">
                  <div className="spinner-large" />
                  <p className="empty-title">Running inference...</p>
                  <p className="empty-hint">Preprocessing image and computing prediction set.</p>
                </div>
              )}

              {results && (
                <div className="results-container">
                  <div className="summary-row">
                    <div className="summary-card">
                      <p className="summary-label">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="2">
                          <polyline points="23 6 13.5 15.5 8.5 10.5 1 18" />
                          <polyline points="17 6 23 6 23 12" />
                        </svg>
                        Top Predicted Finding
                      </p>
                      <p className="summary-value">
                        {results.top_finding}
                        <span className="summary-prob">({results.top_probability.toFixed(2)})</span>
                      </p>
                    </div>

                    <div className="summary-card">
                      <p className="summary-label">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#059669" strokeWidth="2">
                          <path d="M22 11.08V12a10 10 0 11-5.93-9.14" />
                          <polyline points="22 4 12 14.01 9 11.01" />
                        </svg>
                        Prediction Set Size
                      </p>
                      <p className="summary-value">
                        {results.prediction_set_size} labels
                        <span className="coverage-value">at {results.coverage} coverage</span>
                      </p>
                    </div>
                  </div>

                  <div className="table-section">
                    <p className="table-title">All Findings</p>
                    <div className="table-header">
                      <span className="col-finding">Finding</span>
                      <span className="col-centered">Probability</span>
                      <span className="col-centered">Uncertainty</span>
                      <span className="col-centered">Status</span>
                    </div>

                    {results.findings.map((findingRow, index) => {
                      const uncertaintyPalette = uncertaintyColors[findingRow.uncertainty] ?? uncertaintyColors.High;
                      const statusPalette = statusColors[findingRow.in_prediction_set.toString()] ?? statusColors.false;

                      return (
                        <div
                          key={findingRow.finding}
                          className={`table-row ${findingRow.in_prediction_set ? "in-set" : ""}`}
                          style={{ animationDelay: `${index * 80}ms` }}
                        >
                          <span className="col-finding finding-name">{findingRow.finding}</span>
                          <span className="col-centered">
                            <span className="prob-value">{findingRow.probability.toFixed(2)}</span>
                          </span>
                          <span className="col-centered">
                            <span
                              className="badge"
                              style={{
                                backgroundColor: uncertaintyPalette.bg,
                                color: uncertaintyPalette.text,
                                borderColor: uncertaintyPalette.border,
                              }}
                            >
                              {findingRow.uncertainty}
                            </span>
                          </span>
                          <span className="col-centered">
                            <span
                              className="badge"
                              style={{
                                backgroundColor: statusPalette.bg,
                                color: statusPalette.text,
                                borderColor: statusPalette.border,
                                fontWeight: findingRow.in_prediction_set ? 600 : 400,
                              }}
                            >
                              {findingRow.in_prediction_set ? "In prediction set" : "Not in set"}
                            </span>
                          </span>
                        </div>
                      );
                    })}
                  </div>

                  <div className="note-box">
                    <p className="note-text">
                      <strong>Note:</strong> Findings marked &quot;In prediction set&quot; have a &gt;{results.coverage} chance of containing the true diagnosis.
                      Lower uncertainty indicates higher model confidence. Threshold (lamhat) = {results.lamhat.toFixed(4)}.
                    </p>
                  </div>

                  <div className="tech-details">
                    <p className="tech-title">Technical Details</p>
                    <div className="tech-grid">
                      <div>
                        <span className="tech-label">alpha (risk level)</span>
                        <span className="tech-value">{results.alpha}</span>
                      </div>
                      <div>
                        <span className="tech-label">lamhat (threshold)</span>
                        <span className="tech-value">{results.lamhat.toFixed(6)}</span>
                      </div>
                      <div>
                        <span className="tech-label">Coverage</span>
                        <span className="tech-value">{results.coverage}</span>
                      </div>
                      <div>
                        <span className="tech-label">Set size</span>
                        <span className="tech-value">{results.prediction_set_size} / {numLabels}</span>
                      </div>
                      {results.model_info && (
                        <div>
                          <span className="tech-label">Model</span>
                          <span className="tech-value">{results.model_info.name} v{results.model_info.version}</span>
                        </div>
                      )}
                    </div>
                  </div>

                </div>
              )}
            </div>
          </section>
        </div>

        {error && (
          <div className="error-bar">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#dc2626" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <line x1="15" y1="9" x2="9" y2="15" />
              <line x1="9" y1="9" x2="15" y2="15" />
            </svg>
            <span>{error}</span>
            <button className="error-close" onClick={() => setError(null)}>
              x
            </button>
          </div>
        )}
          </>
        )}
    </ClinicianLayout>
  );
}
