import { useEffect, useState } from "react";
import { createPatient, getPatients, Patient, PatientWithStats } from "./api/clinicianApi";
import { useAuth } from "../context/AuthContext";

type Tab = "existing" | "new";

interface PatientSelectorProps {
  onPatientReady: (patient: Patient) => void;
  onTabChange?: (tab: "existing" | "new") => void;
}

export default function PatientSelector({ onPatientReady, onTabChange }: PatientSelectorProps) {
  const { token } = useAuth();
  const [tab, setTab] = useState<Tab>("existing");

  // Existing patient state
  const [patients, setPatients] = useState<PatientWithStats[]>([]);
  const [patientsLoading, setPatientsLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [selectedPatient, setSelectedPatient] = useState<PatientWithStats | null>(null);

  // New patient state
  const [mrn, setMrn] = useState("");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!token) return;
    getPatients(token)
      .then((res) => setPatients(res.patients))
      .catch(() => {})
      .finally(() => setPatientsLoading(false));
  }, [token]);

  const filtered = patients.filter((p) => {
    const q = search.toLowerCase();
    return (
      p.first_name.toLowerCase().includes(q) ||
      p.last_name.toLowerCase().includes(q) ||
      p.mrn.toLowerCase().includes(q)
    );
  });

  const handleSelectExisting = () => {
    if (selectedPatient) {
      onPatientReady({
        id: selectedPatient.id,
        mrn: selectedPatient.mrn,
        first_name: selectedPatient.first_name,
        last_name: selectedPatient.last_name,
      });
    }
  };

  const handleCreateNew = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const patient = await createPatient(mrn, firstName, lastName, token!);
      onPatientReady(patient);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="patient-selector">
      <div className="patient-selector-icon">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2" />
          <circle cx="12" cy="7" r="4" />
        </svg>
      </div>
      <h3 className="panel-title">Patient Selection</h3>
      <p className="panel-subtitle">
        {tab === "existing"
          ? "Select a patient to start your diagnosis."
          : "Add the new patient to start your diagnosis."}
      </p>

      {/* Tabs */}
      <div className="patient-selector-tabs">
        <button
          className={`patient-selector-tab${tab === "existing" ? " patient-selector-tab--active" : ""}`}
          onClick={() => { setTab("existing"); setError(null); onTabChange?.("existing"); }}
        >
          Select Existing
        </button>
        <button
          className={`patient-selector-tab${tab === "new" ? " patient-selector-tab--active" : ""}`}
          onClick={() => { setTab("new"); setError(null); onTabChange?.("new"); }}
        >
          Add New
        </button>
      </div>

      {/* Existing patient tab */}
      {tab === "existing" && (
        <div className="patient-selector-body">
          <div className="patient-selector-search-wrap">
            <svg className="patient-selector-search-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
            <input
              className="patient-selector-search"
              type="text"
              placeholder="Search by name or MRN..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>

          {patientsLoading && (
            <div className="patient-selector-empty">
              <div className="spinner" />
              <span>Loading patients...</span>
            </div>
          )}

          {!patientsLoading && filtered.length === 0 && (
            <div className="patient-selector-empty">
              <p>{search ? "No patients match your search." : "No patients found. Add a new patient to get started."}</p>
            </div>
          )}

          {!patientsLoading && filtered.length > 0 && (
            <div className="patient-selector-list">
              {filtered.slice(0, 10).map((p) => (
                <div
                  key={p.id}
                  className={`patient-selector-item${selectedPatient?.id === p.id ? " patient-selector-item--selected" : ""}`}
                  onClick={() => setSelectedPatient(selectedPatient?.id === p.id ? null : p)}
                >
                  <div className="patient-selector-item-info">
                    <span className="patient-selector-item-name">
                      {p.first_name} {p.last_name}
                    </span>
                    <span className="patient-selector-item-mrn">{p.mrn}</span>
                  </div>
                  {p.prediction_count > 0 && (
                    <span className="patient-selector-item-count">
                      {p.prediction_count} prediction{p.prediction_count !== 1 ? "s" : ""}
                    </span>
                  )}
                </div>
              ))}
            </div>
          )}

          <button
            type="button"
            className="auth-submit-btn"
            disabled={!selectedPatient}
            onClick={handleSelectExisting}
          >
            Confirm Patient
          </button>
        </div>
      )}

      {/* New patient tab */}
      {tab === "new" && (
        <form onSubmit={handleCreateNew} className="patient-form">
          <div className="auth-field">
            <label className="auth-label-text" htmlFor="patient-mrn">Medical Record Number (MRN)</label>
            <input
              id="patient-mrn"
              className="auth-input-control"
              value={mrn}
              onChange={(e) => setMrn(e.target.value)}
              required
              placeholder="e.g. MRN-123456"
            />
          </div>
          <div className="patient-form-row">
            <div className="auth-field">
              <label className="auth-label-text" htmlFor="patient-first-name">First Name</label>
              <input
                id="patient-first-name"
                className="auth-input-control"
                value={firstName}
                onChange={(e) => setFirstName(e.target.value)}
                required
                placeholder="First name"
              />
            </div>
            <div className="auth-field">
              <label className="auth-label-text" htmlFor="patient-last-name">Last Name</label>
              <input
                id="patient-last-name"
                className="auth-input-control"
                value={lastName}
                onChange={(e) => setLastName(e.target.value)}
                required
                placeholder="Last name"
              />
            </div>
          </div>
          {error && <div className="auth-error">{error}</div>}
          <button type="submit" className="auth-submit-btn" disabled={submitting}>
            {submitting ? "Registering..." : "Confirm Patient"}
          </button>
        </form>
      )}
    </div>
  );
}
