import { useState } from "react";
import { createPatient } from "../api/diagnosticApi";
import { useAuth } from "../context/AuthContext";

export default function PatientForm({ onPatientReady }) {
  const { token } = useAuth();
  const [mrn, setMrn] = useState("");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);

    try {
      const patient = await createPatient(mrn, firstName, lastName, token);
      onPatientReady(patient);
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="patient-form-container">
      <div className="patient-form-icon">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2" />
          <circle cx="12" cy="7" r="4" />
        </svg>
      </div>
      <h3 className="panel-title">Patient Information</h3>
      <p className="panel-subtitle">Enter patient details before running prediction</p>
      <form onSubmit={handleSubmit} className="patient-form">
        <label className="auth-label">
          Medical Record Number (MRN)
          <input className="auth-input" value={mrn}
                 onChange={(e) => setMrn(e.target.value)} required placeholder="e.g. MRN-123456" />
        </label>
        <div className="patient-form-row">
          <label className="auth-label">
            First Name
            <input className="auth-input" value={firstName}
                   onChange={(e) => setFirstName(e.target.value)} required />
          </label>
          <label className="auth-label">
            Last Name
            <input className="auth-input" value={lastName}
                   onChange={(e) => setLastName(e.target.value)} required />
          </label>
        </div>
        {error && <div className="auth-error">{error}</div>}
        <button type="submit" className="auth-submit-btn" disabled={submitting}>
          {submitting ? "Registering..." : "Confirm Patient"}
        </button>
      </form>
    </div>
  );
}
