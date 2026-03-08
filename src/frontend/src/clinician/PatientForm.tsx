import { useState } from "react";
import { createPatient, Patient } from "./api/clinicianApi";
import { useAuth } from "../context/AuthContext";

interface PatientFormProps {
  onPatientReady: (patient: Patient) => void;
}

export default function PatientForm({ onPatientReady }: PatientFormProps) {
  const { token } = useAuth();
  const [mrn, setMrn] = useState("");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
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
    </div>
  );
}
