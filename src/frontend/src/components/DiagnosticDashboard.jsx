import { useCallback, useRef, useState } from "react";
import { predictImage } from "../api/diagnosticApi";

const uncertaintyColors = {
  Low: { bg: "#dcfce7", text: "#166534", border: "#86efac" },
  Medium: { bg: "#fff7ed", text: "#9a3412", border: "#fdba74" },
  High: { bg: "#fef2f2", text: "#991b1b", border: "#fca5a5" },
};

const statusColors = {
  true: { bg: "#eff6ff", text: "#1e40af", border: "#93c5fd" },
  false: { bg: "#f9fafb", text: "#6b7280", border: "#e5e7eb" },
};

export default function DiagnosticDashboard() {
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [zoom, setZoom] = useState(100);

  const fileInputRef = useRef(null);

  const handleFile = useCallback((file) => {
    if (!file) {
      return;
    }

    if (!file.type.startsWith("image/")) {
      setError("Please upload an image file (PNG, JPEG)");
      return;
    }

    setImage(file);
    setError(null);
    setResults(null);

    const reader = new FileReader();
    reader.onload = (event) => setImagePreview(event.target?.result || null);
    reader.readAsDataURL(file);
  }, []);

  const handleDrop = useCallback(
    (event) => {
      event.preventDefault();
      setDragOver(false);
      const file = event.dataTransfer.files?.[0];
      handleFile(file);
    },
    [handleFile]
  );

  const handleDragOver = (event) => {
    event.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => setDragOver(false);

  const handleUploadClick = () => fileInputRef.current?.click();

  const handlePredict = async () => {
    if (!image) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await predictImage(image);
      setResults(data);
    } catch (requestError) {
      setError(requestError.message || "Failed to connect to the server");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setImage(null);
    setImagePreview(null);
    setResults(null);
    setError(null);
    setZoom(100);
  };

  return (
    <div className="dashboard-root">
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
              <p className="dash-header-subtitle">Conformal prediction for chest X-ray analysis</p>
            </div>
          </div>
          <div className="dash-header-right">
            <span className="dash-header-badge">CheXpert Model</span>
            <span className="dash-header-badge">FNR &lt;= 10%</span>
          </div>
        </div>
      </header>

      <main className="dash-main">
        <div className="dash-grid">
          <section className="panel">
            <div className="panel-header">
              <div>
                <h2 className="panel-title">Chest X-ray</h2>
                {image ? (
                  <p className="panel-subtitle">{image.name}</p>
                ) : (
                  <p className="panel-subtitle">Upload a posteroanterior (PA) chest radiograph</p>
                )}
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

              {(image || results) && (
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
                  <p className="panel-subtitle">Prediction set generated by the diagnostic support model</p>
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
                  <p className="empty-hint">Upload a chest X-ray and click "Run Prediction" to generate a conformal prediction set.</p>
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
                      const uncertaintyPalette = uncertaintyColors[findingRow.uncertainty];
                      const statusPalette = statusColors[findingRow.in_prediction_set.toString()];

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
                      <strong>Note:</strong> Findings marked "In prediction set" have a &gt;{results.coverage} chance of containing the true diagnosis.
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
                        <span className="tech-value">{results.prediction_set_size} / 5</span>
                      </div>
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
      </main>
    </div>
  );
}
