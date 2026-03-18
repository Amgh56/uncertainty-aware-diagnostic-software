import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import DeveloperLayout from "./DeveloperLayout";
import DeveloperRequirementsCards from "./DeveloperRequirementsCards";

const tocItems = [
  { id: "what-is-calibration", label: "What is calibration?" },
  { id: "alpha-risk-control", label: "Alpha & risk control" },
  { id: "what-safedx-does", label: "What SafeDx does" },
  { id: "what-youll-need", label: "What you'll need" },
  { id: "quick-start", label: "Quick start" },
  { id: "tips", label: "Tips" },
];

export { tocItems };

export default function DeveloperHowToPage() {
  const navigate = useNavigate();
  const [activeSection, setActiveSection] = useState(tocItems[0].id);
  const sectionRefs = useRef<Record<string, HTMLElement | null>>({});

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.id);
          }
        }
      },
      { rootMargin: "-20% 0px -60% 0px", threshold: 0 }
    );

    for (const item of tocItems) {
      const el = sectionRefs.current[item.id];
      if (el) observer.observe(el);
    }

    return () => observer.disconnect();
  }, []);

  const setRef = (id: string) => (el: HTMLElement | null) => {
    sectionRefs.current[id] = el;
  };

  return (
    <DeveloperLayout
      title="Developer Guide"
      subtitle="Learn how SafeDx helps you calibrate, validate, and release your multi-label diagnostic model — with uncertainty you can trust."
      activeSection={activeSection}
    >
      <div className="guide-tutorial-card">
        {/* Header */}
        <div className="guide-tutorial-header">
          <div className="guide-tutorial-badges">
            <span className="guide-badge guide-badge--dev">DEVELOPER GUIDE</span>
            <span className="guide-badge guide-badge--time">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10" />
                <polyline points="12 6 12 12 16 14" />
              </svg>
              8 min read
            </span>
          </div>
        </div>

        {/* Section: What is calibration? */}
        <section id="what-is-calibration" ref={setRef("what-is-calibration")} className="guide-tutorial-section">
          <h2 className="guide-tutorial-section-title">What is calibration?</h2>
          <p className="guide-tutorial-text">
            When you train a classification model, it learns to output confidence scores for each label. But these scores don't always reflect true probability. A model might say it's 90% confident about a diagnosis when, in reality, it's only right 60% of the time. This gap between predicted confidence and actual accuracy is what calibration addresses.
          </p>
          <p className="guide-tutorial-text">
            Calibration adjusts your model's outputs so that the confidence scores align with real-world outcomes. After calibration, when your model says 80%, it genuinely means "this label is correct about 80% of the time." This is critical in medical diagnostics, where doctors and systems rely on these scores to make decisions.
          </p>
        </section>

        <hr className="guide-tutorial-divider" />

        {/* Section: Alpha & risk control */}
        <section id="alpha-risk-control" ref={setRef("alpha-risk-control")} className="guide-tutorial-section">
          <h2 className="guide-tutorial-section-title">Understanding alpha and risk control</h2>
          <p className="guide-tutorial-text">
            At the heart of SafeDx's calibration is a concept called <strong>alpha (α)</strong> — your chosen risk tolerance. Alpha represents the maximum miscoverage rate you're willing to accept. If you set α&nbsp;=&nbsp;0.1, you're saying: "I accept that at most 10% of the true labels might be missed in my prediction sets." A lower alpha is stricter and produces larger, more conservative prediction sets. A higher alpha is more relaxed and produces smaller sets.
          </p>
          <p className="guide-tutorial-text">
            Risk control is the mechanism that turns your alpha into action. Using your calibration dataset, SafeDx computes a threshold (λ̂) that determines which labels make it into the final prediction set. This threshold is calculated so that the coverage guarantee — the probability that the true label is included — is at least 1&nbsp;−&nbsp;α. In practical terms, risk control gives you a statistical guarantee on how reliable your model's outputs will be.
          </p>

        </section>

        <hr className="guide-tutorial-divider" />

        {/* Section: What SafeDx does */}
        <section id="what-safedx-does" ref={setRef("what-safedx-does")} className="guide-tutorial-section">
          <h2 className="guide-tutorial-section-title">What SafeDx does for you</h2>
          <p className="guide-tutorial-text">
            SafeDx is a complete calibration platform for multi-label pre-trained models. You upload your trained model, provide a calibration dataset, and SafeDx handles the rest. It runs the calibration job, computes the optimal thresholds, and gives you a downloadable calibrated result ready for deployment.
          </p>
          <p className="guide-tutorial-text">
            But calibration is just the beginning. Once your model is calibrated, SafeDx also lets you <strong>validate your calibration</strong> to confirm that the output thresholds meet your target miscoverage rate — so you can be confident everything works as expected before going live. Beyond validation, SafeDx provides two release pathways: you can <strong>release your model to researchers</strong> for further study and collaboration, or <strong>release it to doctors</strong> through our dedicated clinical release feature, making your model accessible to the people who will use it in practice.
          </p>
        </section>

        <hr className="guide-tutorial-divider" />

        {/* Section: What you'll need */}
        <section id="what-youll-need" ref={setRef("what-youll-need")} className="guide-tutorial-section">
          <h2 className="guide-tutorial-section-title">What do we need from you?</h2>
          <p className="guide-tutorial-text">
            To calibrate your model successfully, SafeDx needs a compatible trained model, an unseen calibration dataset, and optionally a config file that describes the preprocessing used before inference.
          </p>
          <DeveloperRequirementsCards />
        </section>

        <hr className="guide-tutorial-divider" />

        {/* Section: Quick start */}
        <section id="quick-start" ref={setRef("quick-start")} className="guide-tutorial-section">
          <h2 className="guide-tutorial-section-title">Quick start</h2>
          <div className="guide-quick-steps">
            <div className="guide-quick-step">
              <span className="guide-quick-step-number">1</span>
              <div>
                <h4 className="guide-quick-step-title">Prepare your files</h4>
                <p className="guide-quick-step-text">
                  Export your model as a <code>.pt</code> or <code>.pth</code> file, organise your unseen calibration images into a zip with <code>labels.csv</code>, and create a <code>config.json</code> if your model needs specific preprocessing.
                </p>
              </div>
            </div>
            <div className="guide-quick-step">
              <span className="guide-quick-step-number">2</span>
              <div>
                <h4 className="guide-quick-step-title">Upload everything to SafeDx</h4>
                <p className="guide-quick-step-text">
                  In the calibration workspace, upload your model, dataset, and config. SafeDx validates each file automatically before proceeding.
                </p>
              </div>
            </div>
            <div className="guide-quick-step">
              <span className="guide-quick-step-number">3</span>
              <div>
                <h4 className="guide-quick-step-title">Run calibration</h4>
                <p className="guide-quick-step-text">
                  Start the job. SafeDx runs inference on your calibration data, computes optimal thresholds based on your chosen alpha, and generates the calibrated result. You can monitor progress in the jobs table.
                </p>
              </div>
            </div>
            <div className="guide-quick-step">
              <span className="guide-quick-step-number">4</span>
              <div>
                <h4 className="guide-quick-step-title">Download and validate</h4>
                <p className="guide-quick-step-text">
                  Once complete, download your calibrated result and thresholds. Then use the Validate Calibration feature to confirm the output meets your target miscoverage rate.
                </p>
              </div>
            </div>
            <div className="guide-quick-step">
              <span className="guide-quick-step-number">5</span>
              <div>
                <h4 className="guide-quick-step-title">Release your model</h4>
                <p className="guide-quick-step-text">
                  When you're confident in the results, release your calibrated model — either to researchers for further study, or to doctors through the clinical release feature.
                </p>
              </div>
            </div>
          </div>
        </section>

        <hr className="guide-tutorial-divider" />

        {/* Section: Tips */}
        <section id="tips" ref={setRef("tips")} className="guide-tutorial-section">
          <h2 className="guide-tutorial-section-title">Tips for better calibration</h2>
          <div className="guide-tips-grid">
            <div className="guide-tip-card">
              <div className="guide-tip-header">
                <span className="guide-tip-icon">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="10" />
                    <line x1="12" y1="8" x2="12" y2="12" />
                    <line x1="12" y1="16" x2="12.01" y2="16" />
                  </svg>
                </span>
                <h4 className="guide-tip-title">Use an unseen dataset</h4>
              </div>
              <p className="guide-tip-text">
                Your calibration dataset must be entirely separate from your training data. The model should not have encountered any of these examples during training — this ensures the calibrated thresholds generalise to real-world inputs.
              </p>
            </div>
            <div className="guide-tip-card">
              <div className="guide-tip-header">
                <span className="guide-tip-icon">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="3" y="3" width="7" height="7" /><rect x="14" y="3" width="7" height="7" />
                    <rect x="3" y="14" width="7" height="7" /><rect x="14" y="14" width="7" height="7" />
                  </svg>
                </span>
                <h4 className="guide-tip-title">Keep the dataset structure consistent</h4>
              </div>
              <p className="guide-tip-text">
                Ensure the image dimensions, label columns, and preprocessing pipeline match what your model was trained on. Mismatched input shapes or label ordering will produce unreliable calibration results.
              </p>
            </div>
            <div className="guide-tip-card">
              <div className="guide-tip-header">
                <span className="guide-tip-icon">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="18" y1="20" x2="18" y2="10" /><line x1="12" y1="20" x2="12" y2="4" /><line x1="6" y1="20" x2="6" y2="14" />
                  </svg>
                </span>
                <h4 className="guide-tip-title">Reserve enough data for calibration</h4>
              </div>
              <p className="guide-tip-text">
                A larger calibration set produces more stable and reliable thresholds. As a rule of thumb, reserve at least 10–20% of your total labelled data for calibration — the more representative samples you include, the better the result.
              </p>
            </div>
            <div className="guide-tip-card">
              <div className="guide-tip-header">
                <span className="guide-tip-icon">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M9 11l3 3L22 4" />
                    <path d="M21 12v7a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2h11" />
                  </svg>
                </span>
                <h4 className="guide-tip-title">Validate your calibration result</h4>
              </div>
              <p className="guide-tip-text">
                After your calibration job completes, use the <strong>Validate Calibration</strong> feature to verify that the output thresholds meet your target miscoverage rate. This step confirms the calibration is working as expected before deployment.
              </p>
            </div>
          </div>
        </section>

        {/* CTA */}
        <div className="guide-tutorial-cta">
          <p className="guide-tutorial-cta-text">
            When your files are ready, head to the <a className="guide-tutorial-cta-link" href="/developer/calibrate" onClick={(e) => { e.preventDefault(); navigate("/developer/calibrate"); }}>calibration workspace</a> and start your job.
          </p>
        </div>
      </div>
    </DeveloperLayout>
  );
}
