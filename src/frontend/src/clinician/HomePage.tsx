import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { getPatients, PatientWithStats } from "./api/clinicianApi";
import ClinicianLayout from "./ClinicianLayout";

const PAGE_SIZE = 5;

export default function HomePage() {
  const { token, doctor } = useAuth();
  const navigate = useNavigate();
  const [patients, setPatients] = useState<PatientWithStats[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);

  useEffect(() => {
    getPatients(token!)
      .then((data) => setPatients(data.patients))
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  }, [token]);

  // Only show patients that have at least one prediction (image uploaded)
  const withPredictions = patients.filter((p) => p.prediction_count > 0);

  const filtered = withPredictions.filter((p) => {
    const q = search.toLowerCase();
    return (
      p.first_name.toLowerCase().includes(q) ||
      p.last_name.toLowerCase().includes(q) ||
      p.mrn.toLowerCase().includes(q)
    );
  });

  // Reset to first page when search changes
  useEffect(() => {
    setPage(0);
  }, [search]);

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const start = page * PAGE_SIZE;
  const pageItems = filtered.slice(start, start + PAGE_SIZE);

  return (
    <ClinicianLayout
      title={`Welcome Back, ${doctor?.full_name?.split(" ").pop() || "Doctor"}`}
      subtitle="Uncertainty-Aware Diagnostic System - Manage your cases and patient records"
    >
      <section className="dashboard-floating-shell">
        <div className="home-top-bar">
          <div>
            <h2 className="home-welcome">Your Patient Cases</h2>
            <p className="home-welcome-sub">Review previous predictions and reopen saved cases.</p>
          </div>
        </div>

        <div className="home-search-wrapper">
          <svg className="home-search-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <input
            className="home-search-input"
            type="text"
            placeholder="Search patients by name or file number..."
            aria-label="Search patients"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        <h3 className="home-section-title">Previous Cases</h3>

        {loading && (
          <div className="panel" style={{ padding: 60, textAlign: "center" }}>
            <div className="spinner-large" style={{ margin: "0 auto 16px" }} />
            <p className="empty-title">Loading patients...</p>
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

        {!loading && !error && filtered.length === 0 && (
          <div className="panel" style={{ padding: 60, textAlign: "center" }}>
            <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="#cbd5e1" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2" />
              <circle cx="12" cy="7" r="4" />
            </svg>
            <p className="empty-title">{search ? "No patients match your search" : "No patients yet"}</p>
            <p className="empty-hint">Use the sidebar New Diagnosis view to add a patient and run a prediction.</p>
          </div>
        )}

        <div className="home-patient-list">
          {pageItems.map((patient) => (
            <div key={patient.id} className="home-patient-card">
              <div className="home-patient-info">
                <div className="home-patient-top">
                  <span className="home-patient-name">{patient.first_name} {patient.last_name}</span>
                  <span className="home-patient-mrn">{patient.mrn}</span>
                  {patient.last_top_finding && (
                    <span className="home-patient-finding-badge">
                      {patient.last_top_finding}
                    </span>
                  )}
                </div>
                <div className="home-patient-meta">
                  <span className="home-patient-meta-item">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                      <line x1="3" y1="9" x2="21" y2="9" />
                      <line x1="9" y1="21" x2="9" y2="9" />
                    </svg>
                    {patient.prediction_count} prediction{patient.prediction_count !== 1 ? "s" : ""}
                  </span>
                  {patient.last_prediction_at && (
                    <span className="home-patient-meta-item">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
                        <line x1="16" y1="2" x2="16" y2="6" />
                        <line x1="8" y1="2" x2="8" y2="6" />
                        <line x1="3" y1="10" x2="21" y2="10" />
                      </svg>
                      Last session: {new Date(patient.last_prediction_at).toLocaleDateString()}
                    </span>
                  )}
                </div>
              </div>
              <button
                className="home-view-btn"
                onClick={() => {
                  if (patient.last_prediction_id) {
                    navigate(`/predictions/${patient.last_prediction_id}`);
                  }
                }}
                disabled={!patient.last_prediction_id}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                  <circle cx="12" cy="12" r="3" />
                </svg>
                View Case
              </button>
            </div>
          ))}
        </div>

        {filtered.length > PAGE_SIZE && (
          <div className="home-pagination">
            <span className="home-pagination-info">
              {start + 1}-{Math.min(start + PAGE_SIZE, filtered.length)} of {filtered.length} patients
            </span>
            <div className="home-pagination-btns">
              <button
                className="home-page-btn"
                disabled={page === 0}
                onClick={() => setPage((p) => p - 1)}
              >
                Previous
              </button>
              <button
                className="home-page-btn"
                disabled={page >= totalPages - 1}
                onClick={() => setPage((p) => p + 1)}
              >
                Next
              </button>
            </div>
          </div>
        )}
      </section>
    </ClinicianLayout>
  );
}
