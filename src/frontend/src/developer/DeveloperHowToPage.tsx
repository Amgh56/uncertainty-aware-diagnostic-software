import { useState } from "react";
import { useNavigate } from "react-router-dom";
import DeveloperLayout from "./DeveloperLayout";
import DeveloperRequirementsCards from "./DeveloperRequirementsCards";

const steps = [
  "Prepare your trained multilabel model in a supported .pt or .pth format.",
  "Prepare an unseen calibration dataset zip containing images/ and labels.csv.",
  "Optionally prepare a config JSON if your model depends on resizing or normalisation.",
  "Open the calibration page in SafeDx.",
  "Upload the required files.",
  "Set the target alpha / miscoverage value.",
  "Run calibration.",
  "Wait for the job to complete in the jobs table.",
  "Download the calibrated result once it is ready.",
];

export default function DeveloperHowToPage() {
  const navigate = useNavigate();
  const [requirementsOpen, setRequirementsOpen] = useState(false);
  const [stepsOpen, setStepsOpen] = useState(false);
  const [tipsOpen, setTipsOpen] = useState(false);

  return (
    <DeveloperLayout
      title="How to Calibrate Your Model"
      subtitle="Learn what SafeDx needs from you and how the calibration workflow works before you start uploading files."
    >
      <section className="developer-floating-shell developer-guide-shell">
        <section className="developer-guide-hero">
          <div className="developer-guide-hero-copy">
            <p className="developer-guide-kicker">Developer Guide</p>
            <h2 className="developer-guide-title">How to Calibrate Your Model</h2>
            <p className="developer-guide-text">
              Use SafeDx to calibrate your trained model and better represent predictive uncertainty before deployment.
            </p>
          </div>
        </section>

        <div className="developer-guide-grid">
          <section className="developer-guide-card">
            <h3 className="developer-guide-card-title">Why calibrate my model?</h3>
            <p className="developer-guide-card-text">
              A model can make strong predictions and still be poorly calibrated. Confidence scores alone do not tell you whether uncertainty is being expressed responsibly. Calibration helps align model outputs with real-world uncertainty so predictions are easier to interpret and trust.
            </p>
          </section>

          <section className="developer-guide-card">
            <h3 className="developer-guide-card-title">What is calibration?</h3>
            <p className="developer-guide-card-text">
              You have already trained your model. SafeDx helps you calibrate that model so its predictions communicate uncertainty more clearly and more responsibly. The current SafeDx calibration workflow supports multilabel models and guides you through the exact files required to run calibration correctly.
            </p>
          </section>
        </div>

        <section className="developer-guide-card">
          <h3 className="developer-guide-card-title">What is SafeDx?</h3>
          <p className="developer-guide-card-text">
            SafeDx provides a guided calibration workflow for supported models. It helps you upload the required artifacts, run calibration jobs, monitor progress, and download the calibrated result and thresholds once processing is complete.
          </p>
        </section>

        <section className="developer-guide-section">
          <button
            type="button"
            className="developer-accordion-toggle"
            onClick={() => setRequirementsOpen((o) => !o)}
            aria-expanded={requirementsOpen}
          >
            <div className="developer-accordion-toggle-left">
              <h3 className="developer-guide-section-title">What do we need from you?</h3>
              <p className="developer-guide-section-text">
                To calibrate your model successfully, SafeDx needs a compatible trained model, an unseen calibration dataset, and optionally a config file that describes the preprocessing used before inference.
              </p>
            </div>
            <svg
              className={`developer-accordion-chevron${requirementsOpen ? " open" : ""}`}
              width="20" height="20" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
            >
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </button>
          {requirementsOpen && (
            <div className="developer-accordion-body">
              <DeveloperRequirementsCards />
            </div>
          )}
        </section>

        <section className="developer-guide-section">
          <button
            type="button"
            className="developer-accordion-toggle"
            onClick={() => setStepsOpen((o) => !o)}
            aria-expanded={stepsOpen}
          >
            <div className="developer-accordion-toggle-left">
              <h3 className="developer-guide-section-title">How to use SafeDx</h3>
              <p className="developer-guide-section-text">
                Follow these steps to move from preparation to a completed calibration job.
              </p>
            </div>
            <svg
              className={`developer-accordion-chevron${stepsOpen ? " open" : ""}`}
              width="20" height="20" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
            >
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </button>
          {stepsOpen && (
            <div className="developer-accordion-body">
              <div className="developer-steps">
                {steps.map((step, index) => (
                  <div key={step} className="developer-step-card">
                    <span className="developer-step-number">{index + 1}</span>
                    <p className="developer-step-text">{step}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>

        <section className="developer-guide-section">
          <button
            type="button"
            className="developer-accordion-toggle"
            onClick={() => setTipsOpen((o) => !o)}
            aria-expanded={tipsOpen}
          >
            <div className="developer-accordion-toggle-left">
              <h3 className="developer-guide-section-title">Tips on Calibrating</h3>
              <p className="developer-guide-section-text">
                Follow these tips to improve the quality and reliability of your calibration results.
              </p>
            </div>
            <svg
              className={`developer-accordion-chevron${tipsOpen ? " open" : ""}`}
              width="20" height="20" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
            >
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </button>
          {tipsOpen && (
            <div className="developer-accordion-body">
              <div className="developer-tips-grid">
                <div className="developer-tip-card">
                  <div className="developer-tip-icon">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                    </svg>
                  </div>
                  <div>
                    <h4 className="developer-tip-title">Use an unseen dataset</h4>
                    <p className="developer-tip-text">
                      Your calibration dataset must be entirely separate from your training data. The model should not have encountered any of these examples during training — this ensures the calibrated thresholds generalise to real-world inputs.
                    </p>
                  </div>
                </div>

                <div className="developer-tip-card">
                  <div className="developer-tip-icon">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                      <line x1="3" y1="9" x2="21" y2="9" />
                      <line x1="9" y1="21" x2="9" y2="9" />
                    </svg>
                  </div>
                  <div>
                    <h4 className="developer-tip-title">Keep the dataset structure consistent</h4>
                    <p className="developer-tip-text">
                      Ensure the image dimensions, label columns, and preprocessing pipeline match what your model was trained on. Mismatched input shapes or label ordering will produce unreliable calibration results.
                    </p>
                  </div>
                </div>

                <div className="developer-tip-card">
                  <div className="developer-tip-icon">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="12" y1="20" x2="12" y2="10" />
                      <line x1="18" y1="20" x2="18" y2="4" />
                      <line x1="6" y1="20" x2="6" y2="16" />
                    </svg>
                  </div>
                  <div>
                    <h4 className="developer-tip-title">Reserve enough data for calibration</h4>
                    <p className="developer-tip-text">
                      A larger calibration set produces more stable and reliable thresholds. As a rule of thumb, reserve at least 10–20% of your total labelled data for calibration — the more representative samples you include, the better the result.
                    </p>
                  </div>
                </div>

                <div className="developer-tip-card">
                  <div className="developer-tip-icon">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="9 11 12 14 22 4" />
                      <path d="M21 12v7a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2h11" />
                    </svg>
                  </div>
                  <div>
                    <h4 className="developer-tip-title">Validate your calibration result</h4>
                    <p className="developer-tip-text">
                      After your calibration job completes, use the <strong>Validate Calibration</strong> feature to verify that the output thresholds meet your target miscoverage rate. This step confirms the calibration is working as expected before deployment.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </section>

        <section className="developer-guide-cta">
          <div>
            <h3 className="developer-guide-card-title">Ready to calibrate?</h3>
            <p className="developer-guide-card-text">
              When your files are ready, move to the calibration workspace and run your job.
            </p>
          </div>
          <button
            type="button"
            className="developer-primary-cta"
            onClick={() => navigate("/developer/calibrate")}
          >
            Calibrate My Model
          </button>
        </section>
      </section>
    </DeveloperLayout>
  );
}
