import { useState } from "react";
import { useAuth } from "../context/AuthContext";
import {
  publishModel,
  type PublishModelPayload,
  type ValidationData,
} from "./api/developerApi";

interface PublishModelDialogProps {
  jobId: string;
  validation: ValidationData;
  onPublished: () => void;
  onCancel: () => void;
}

const MODALITY_OPTIONS = [
  "Chest X-Ray",
  "Dermatoscopy",
  "Retinal OCT",
  "CT Scan",
  "MRI",
  "Ultrasound",
  "Other",
];

export default function PublishModelDialog({
  jobId,
  validation,
  onPublished,
  onCancel,
}: PublishModelDialogProps) {
  const { token } = useAuth();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [version, setVersion] = useState("1.0.0");
  const [modality, setModality] = useState("Chest X-Ray");
  const [customModality, setCustomModality] = useState("");
  const [intendedUse, setIntendedUse] = useState("");
  const [visibility, setVisibility] = useState("private");
  const [consentChecked, setConsentChecked] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const labels = validation.label_names;
  const isBlocked = validation.verdict === "unreliable" || validation.verdict === "review";
  const needsConsent = visibility !== "private";
  const effectiveModality = modality === "Other" ? customModality.trim() : modality;
  const canSubmit =
    !isBlocked &&
    !submitting &&
    name.trim() &&
    description.trim() &&
    version.trim() &&
    effectiveModality &&
    intendedUse.trim() &&
    (!needsConsent || consentChecked);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit || !token) return;

    setSubmitting(true);
    setError(null);

    const payload: PublishModelPayload = {
      calibration_job_id: jobId,
      name: name.trim(),
      description: description.trim(),
      version: version.trim(),
      modality: effectiveModality,
      intended_use: intendedUse.trim(),
      labels,
      visibility,
      consent_agreed: needsConsent ? consentChecked : false,
    };

    try {
      await publishModel(payload, token);
      onPublished();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="publish-dialog-overlay" onClick={onCancel}>
      <div
        className="publish-dialog"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="publish-dialog-header">
          <h2 className="publish-dialog-title">Publish Model</h2>
          <button
            type="button"
            className="publish-dialog-close"
            onClick={onCancel}
            aria-label="Close"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        {isBlocked && (
          <div className="publish-dialog-warning">
            This model cannot be published. Only models with a <strong>Good</strong> verdict are allowed to be published. Please recalibrate with a larger or higher-quality dataset to improve the model's reliability.
          </div>
        )}

        <form onSubmit={handleSubmit} className="publish-dialog-form">
          <div className="publish-field">
            <label htmlFor="pub-name">Model Name *</label>
            <input
              id="pub-name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., CheXpert Chest X-Ray Classifier"
              maxLength={150}
              disabled={isBlocked}
            />
          </div>

          <div className="publish-field">
            <label htmlFor="pub-desc">Description *</label>
            <textarea
              id="pub-desc"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Short description of what the model does"
              rows={2}
              disabled={isBlocked}
            />
          </div>

          <div className="publish-row-2">
            <div className="publish-field">
              <label htmlFor="pub-version">Version *</label>
              <input
                id="pub-version"
                type="text"
                value={version}
                onChange={(e) => setVersion(e.target.value)}
                placeholder="1.0.0"
                maxLength={20}
                disabled={isBlocked}
              />
            </div>
            <div className="publish-field">
              <label htmlFor="pub-modality">Modality *</label>
              <select
                id="pub-modality"
                value={modality}
                onChange={(e) => {
                  setModality(e.target.value);
                  if (e.target.value !== "Other") setCustomModality("");
                }}
                disabled={isBlocked}
              >
                {MODALITY_OPTIONS.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
              {modality === "Other" && (
                <input
                  type="text"
                  value={customModality}
                  onChange={(e) => setCustomModality(e.target.value)}
                  placeholder="Enter your modality type"
                  maxLength={100}
                  disabled={isBlocked}
                  style={{ marginTop: 6 }}
                />
              )}
            </div>
          </div>

          <div className="publish-field">
            <label htmlFor="pub-use">Intended Use *</label>
            <textarea
              id="pub-use"
              value={intendedUse}
              onChange={(e) => setIntendedUse(e.target.value)}
              placeholder="e.g., Screening aid for thoracic conditions in adult patients"
              rows={2}
              disabled={isBlocked}
            />
          </div>

          {/* Labels (read-only) */}
          <div className="publish-field">
            <label>Labels (from calibration dataset)</label>
            <div className="publish-labels-list">
              {labels.map((l, i) => (
                <span key={i} className="publish-label-chip">
                  {i + 1}. {l}
                </span>
              ))}
            </div>
          </div>

          {/* Auto-filled calibration info */}
          <div className="publish-auto-row">
            <span>Alpha: {validation.job_alpha}</span>
            <span>Lamhat: {validation.job_lamhat.toFixed(4)}</span>
            <span>
              Verdict:{" "}
              <span className={`publish-verdict publish-verdict--${validation.verdict}`}>
                {validation.verdict.charAt(0).toUpperCase() + validation.verdict.slice(1)}
              </span>
            </span>
          </div>

          {/* Visibility */}
          <fieldset className="publish-field" disabled={isBlocked}>
            <legend>Who should have access to this model?</legend>
            <div className="publish-radio-group">
              {[
                { value: "private", label: "Keep Private (only you can see it)" },
                { value: "clinician", label: "Release for Clinician Use" },
                { value: "community", label: "Release to Developer Community" },
                { value: "clinician_and_community", label: "Release for Both Clinician and Community Use" },
              ].map((opt) => (
                <label key={opt.value} className="publish-radio-label">
                  <input
                    type="radio"
                    name="visibility"
                    value={opt.value}
                    checked={visibility === opt.value}
                    onChange={(e) => {
                      setVisibility(e.target.value);
                      setConsentChecked(false);
                    }}
                  />
                  <span>{opt.label}</span>
                </label>
              ))}
            </div>
          </fieldset>

          {/* Consent */}
          {needsConsent && !isBlocked && (
            <div className="publish-consent-box">
              <h4>Release Consent</h4>
              <p>By releasing this model, you acknowledge that:</p>
              <ul>
                <li>Your model artifact, metadata, and calibration parameters will be shared with the selected audience</li>
                <li>For clinician release: this model may be used as a diagnostic aid for patients</li>
                <li>You are responsible for the quality and validity of your model within the scope of your validation</li>
                <li>Your name will be displayed as the model creator</li>
                <li>You can change visibility or deactivate the model at any time</li>
                <li>Real clinical deployment requires formal review beyond this platform's validation</li>
              </ul>
              <label className="publish-consent-check">
                <input
                  type="checkbox"
                  checked={consentChecked}
                  onChange={(e) => setConsentChecked(e.target.checked)}
                />
                <span>I understand and agree to these terms</span>
              </label>
            </div>
          )}

          {error && (
            <div className="publish-error">{error}</div>
          )}

          <div className="publish-dialog-actions">
            <button
              type="button"
              className="val-btn val-btn-outline"
              onClick={onCancel}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="val-btn val-btn-primary"
              disabled={!canSubmit}
            >
              {submitting ? "Publishing..." : "Publish Model"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
