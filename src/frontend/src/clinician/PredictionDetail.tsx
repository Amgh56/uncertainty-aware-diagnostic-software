import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { ArrowLeft, ZoomIn, ZoomOut } from "lucide-react";
import { useAuth } from "../context/AuthContext";
import { getPrediction, PredictionResponse } from "./api/clinicianApi";
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

export default function PredictionDetail() {
  const { id } = useParams();
  const { token } = useAuth();
  const navigate = useNavigate();
  const [data, setData] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [zoom, setZoom] = useState(100);

  useEffect(() => {
    getPrediction(id!, token!)
      .then((result) => {
        setData(result);
      })
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  }, [id, token]);

  if (loading) {
    return (
      <ClinicianLayout title="" subtitle="">
        <div className="panel clinician-feedback-panel">
          <div className="empty-state">
            <div className="spinner-large" />
            <p className="empty-title">Loading prediction...</p>
          </div>
        </div>
      </ClinicianLayout>
    );
  }

  if (error || !data) {
    return (
      <ClinicianLayout title="" subtitle="">
        <div className="panel clinician-feedback-panel">
          <div className="empty-state">
            <p className="empty-title">Error: {error ?? "Unknown error"}</p>
            <button className="home-page-btn" onClick={() => navigate("/home")} style={{ marginTop: 16 }}>
              Back to Dashboard
            </button>
          </div>
        </div>
      </ClinicianLayout>
    );
  }

  const numLabels = data.findings?.length ?? 5;
  const patientProfilePath = `/patients/${data.patient.id}`;

  return (
    <ClinicianLayout
      title="Case Detail"
      subtitle={`${data.patient.first_name} ${data.patient.last_name} (MRN: ${data.patient.mrn})`}
      headerBefore={
        <button
          className="profile-back-btn"
          onClick={() => navigate(patientProfilePath)}
        >
          <ArrowLeft size={18} />
          <span>Back to Patient Profile</span>
        </button>
      }
    >
        <div className="dash-grid">
          <section className="panel">
            <div className="panel-header">
              <div>
                <h2 className="panel-title">Chest X-ray</h2>
                <p className="panel-subtitle">
                  {data.patient.first_name} {data.patient.last_name} - {new Date(data.created_at).toLocaleString()}
                </p>
              </div>
            </div>
            <div className="panel-body">
              <div className="image-container">
                <img
                  src={data.image_path}
                  alt="Chest X-ray"
                  className="xray-image"
                  style={{ transform: `scale(${zoom / 100})` }}
                />
              </div>
            </div>
            <div className="panel-footer">
              <div className="zoom-controls">
                <button
                  className="zoom-btn"
                  onClick={() => setZoom((value) => Math.max(50, value - 25))}
                  aria-label="Zoom out"
                >
                  <ZoomOut size={16} />
                </button>
                <button
                  className="zoom-btn"
                  onClick={() => setZoom((value) => Math.min(200, value + 25))}
                  aria-label="Zoom in"
                >
                  <ZoomIn size={16} />
                </button>
              </div>
              <span className="zoom-label">Zoom: {zoom}%</span>
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
                    {data.model_info
                      ? `Model: ${data.model_info.name} (v${data.model_info.version})`
                      : "Legacy model (pre-platform)"}
                  </p>
                </div>
              </div>
            </div>

            <div className="panel-body">
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
                      {data.top_finding}
                      <span className="summary-prob">({data.top_probability.toFixed(2)})</span>
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
                      {data.prediction_set_size} labels
                      <span className="coverage-value">at {data.coverage} coverage</span>
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

                  {data.findings.map((findingRow, index) => {
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
                    <strong>Note:</strong> Findings marked &quot;In prediction set&quot; have a &gt;{data.coverage} chance of containing the true diagnosis.
                    Lower uncertainty indicates higher model confidence. Threshold (lamhat) = {data.lamhat.toFixed(4)}.
                  </p>
                </div>

                <div className="tech-details">
                  <p className="tech-title">Technical Details</p>
                  <div className="tech-grid">
                    <div>
                      <span className="tech-label">alpha (risk level)</span>
                      <span className="tech-value">{data.alpha}</span>
                    </div>
                    <div>
                      <span className="tech-label">lamhat (threshold)</span>
                      <span className="tech-value">{data.lamhat.toFixed(6)}</span>
                    </div>
                    <div>
                      <span className="tech-label">Coverage</span>
                      <span className="tech-value">{data.coverage}</span>
                    </div>
                    <div>
                      <span className="tech-label">Set size</span>
                      <span className="tech-value">{data.prediction_set_size} / {numLabels}</span>
                    </div>
                    <div>
                      <span className="tech-label">Model</span>
                      <span className="tech-value">
                        {data.model_info
                          ? `${data.model_info.name} v${data.model_info.version}`
                          : "Legacy (built-in)"}
                      </span>
                    </div>
                  </div>
                </div>

              </div>
            </div>
          </section>
        </div>
    </ClinicianLayout>
  );
}
