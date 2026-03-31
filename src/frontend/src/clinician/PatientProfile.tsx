import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { ArrowLeft, Activity, Layers, Clock, Eye } from "lucide-react";
import { useAuth } from "../context/AuthContext";
import { getPatientPredictions, PatientPredictionsResponse } from "./api/clinicianApi";
import ClinicianLayout from "./ClinicianLayout";

export default function PatientProfile() {
  const { id } = useParams<{ id: string }>();
  const { token } = useAuth();
  const navigate = useNavigate();
  const [data, setData] = useState<PatientPredictionsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id || !token) return;
    getPatientPredictions(Number(id), token)
      .then(setData)
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  }, [id, token]);

  const patient = data?.patient;
  const predictions = data?.predictions ?? [];

  return (
    <ClinicianLayout
      title={patient ? `${patient.first_name} ${patient.last_name}` : "Patient Profile"}
      subtitle={patient ? `MRN: ${patient.mrn}` : "Loading patient data..."}
    >
      <button
        className="profile-back-btn"
        onClick={() => navigate("/home")}
      >
        <ArrowLeft size={18} />
        <span>Back to Dashboard</span>
      </button>

      {loading && (
        <div className="panel" style={{ padding: 60, textAlign: "center" }}>
          <div className="spinner-large" style={{ margin: "0 auto 16px" }} />
          <p className="empty-title">Loading patient profile...</p>
        </div>
      )}

      {error && (
        <div className="error-bar">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#dc2626" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <line x1="15" y1="9" x2="9" y2="15" />
            <line x1="9" y1="9" x2="15" y2="15" />
          </svg>
          <span>{error}</span>
        </div>
      )}

      {!loading && !error && predictions.length === 0 && (
        <div className="panel" style={{ padding: 60, textAlign: "center" }}>
          <p className="empty-title">No diagnoses yet</p>
          <p className="empty-hint">This patient has no predictions recorded.</p>
        </div>
      )}

      {!loading && !error && predictions.length > 0 && (
        <section className="profile-predictions">
          <h3 className="profile-section-title">
            Diagnosis History
            <span className="profile-count-badge">{predictions.length}</span>
          </h3>

          <div className="profile-card-grid">
            {predictions.map((pred) => (
              <div key={pred.id} className="profile-prediction-card">
                <div className="profile-card-image-wrap">
                  <img
                    src={pred.image_path}
                    alt={`X-ray — ${pred.top_finding}`}
                    className="profile-card-image"
                    loading="lazy"
                  />
                </div>

                <div className="profile-card-body">
                  <div className="profile-card-top">
                    <span className="profile-card-finding">{pred.top_finding}</span>
                    <span className="profile-card-date">
                      <Clock size={13} />
                      {new Date(pred.created_at).toLocaleDateString("en-US", {
                        year: "numeric",
                        month: "short",
                        day: "numeric",
                      })}
                    </span>
                  </div>

                  <div className="profile-card-stats">
                    <div className="profile-card-stat">
                      <Activity size={14} />
                      <span className="profile-card-stat-label">Confidence</span>
                      <span className="profile-card-stat-value">{(pred.top_probability * 100).toFixed(1)}%</span>
                    </div>
                    <div className="profile-card-stat">
                      <Layers size={14} />
                      <span className="profile-card-stat-label">Set Size</span>
                      <span className="profile-card-stat-value">{pred.prediction_set_size}</span>
                    </div>
                  </div>

                  <button
                    className="profile-card-view-btn"
                    onClick={() => navigate(`/predictions/${pred.id}`)}
                  >
                    <Eye size={14} />
                    <span>View case</span>
                  </button>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </ClinicianLayout>
  );
}
