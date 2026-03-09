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
          <div className="developer-guide-section-head">
            <h3 className="developer-guide-section-title">What do we need from you?</h3>
            <p className="developer-guide-section-text">
              To calibrate your model successfully, SafeDx needs a compatible trained model, an unseen calibration dataset, and optionally a config file that describes the preprocessing used before inference.
            </p>
          </div>
          <DeveloperRequirementsCards />
        </section>

        <section className="developer-guide-section">
          <div className="developer-guide-section-head">
            <h3 className="developer-guide-section-title">How to use SafeDx</h3>
            <p className="developer-guide-section-text">
              Follow these steps to move from preparation to a completed calibration job.
            </p>
          </div>

          <div className="developer-steps">
            {steps.map((step, index) => (
              <div key={step} className="developer-step-card">
                <span className="developer-step-number">{index + 1}</span>
                <p className="developer-step-text">{step}</p>
              </div>
            ))}
          </div>
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
