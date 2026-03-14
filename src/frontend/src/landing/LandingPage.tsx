import { useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

/* ── Marquee data ── */
const devMarquee = [
  "For Developers",
  "For Researchers",
  "Model-First Architecture",
  "Plug-and-Play Risk Control",
  "Scalable Calibration Pipeline",
  "Uncertainty-Aware Evaluation",
  "Multilabel Model Support",
  "Calibration Workflows",
];

const clinicalMarquee = [
  "For Clinicians",
  "Confidence with Uncertainty",
  "Interpretable Predictions",
  "Case Review Workflows",
  "Patient-Centered Analysis",
  "Actionable AI Insights",
  "Visual Diagnostic Support",
  "Clearer Decision Support",
];

/* ── Dev features data ── */
const devFeatures = [
  {
    title: "Upload and Calibrate Models",
    text: "Support for TorchScript and full PyTorch models out of the box.",
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
        <polyline points="17 8 12 3 7 8" />
        <line x1="12" y1="3" x2="12" y2="15" />
      </svg>
    ),
  },
  {
    title: "Run Calibration Pipelines",
    text: "Execute conformal prediction calibration with a single click and monitor job progress in real time.",
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polygon points="5 3 19 12 5 21 5 3" />
      </svg>
    ),
  },
  {
    title: "Download Thresholds",
    text: "Retrieve calibrated thresholds and metrics as structured JSON, ready for integration.",
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
        <polyline points="7 10 12 15 17 10" />
        <line x1="12" y1="15" x2="12" y2="3" />
      </svg>
    ),
  },
  {
    title: "Validate Results",
    text: "Verify your calibration output meets the target miscoverage rate before deploying to production.",
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="9 11 12 14 22 4" />
        <path d="M21 12v7a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2h11" />
      </svg>
    ),
  },
  {
    title: "Guided Instructions",
    text: "Step-by-step tutorials and format guides so you always know what files are required and why.",
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M2 3h6a4 4 0 014 4v14a3 3 0 00-3-3H2z" />
        <path d="M22 3h-6a4 4 0 00-4 4v14a3 3 0 013-3h7z" />
      </svg>
    ),
  },
  {
    title: "Multilabel Support",
    text: "First-class support for multilabel classification models with per-class uncertainty quantification.",
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="3" width="7" height="7" />
        <rect x="14" y="3" width="7" height="7" />
        <rect x="14" y="14" width="7" height="7" />
        <rect x="3" y="14" width="7" height="7" />
      </svg>
    ),
  },
];

/* ── Marquee band component ── */
function MarqueeBand({
  items,
  variant,
}: {
  items: string[];
  variant: "dev" | "clinical";
}) {
  // Duplicate items for seamless loop
  const doubled = [...items, ...items];
  return (
    <div className={`landing-marquee-band landing-marquee-band--${variant}`}>
      <div className="landing-marquee-track">
        {doubled.map((item, i) => (
          <span key={i} className="landing-marquee-item">
            <span className="landing-marquee-dot" />
            {item}
          </span>
        ))}
      </div>
    </div>
  );
}

/* ── Page ── */
export default function LandingPage() {
  const { token } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (token) {
      navigate("/home", { replace: true });
    }
  }, [token, navigate]);

  if (token) return null;

  return (
    <div className="landing-page">
      {/* ── Navbar ── */}
      <nav className="landing-nav">
        <span className="landing-nav-brand">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M22 12H18L15 21L9 3L6 12H2" />
          </svg>
          SafeDx
        </span>
        <div className="landing-nav-links">
          <a href="#how-it-works" className="landing-nav-link">How It Works</a>
          <a href="#about" className="landing-nav-link">About</a>
          <a href="#developers" className="landing-nav-link">Developers</a>
          <Link to="/login" className="landing-nav-link">Sign In</Link>
          <Link to="/register" className="landing-nav-cta">Get Started</Link>
        </div>
      </nav>

      {/* ── Hero ── */}
      <section className="landing-section landing-hero">
        <div className="landing-hero-visual">
          <div className="landing-hero-visual-inner">
            <div className="landing-hero-ring" />
            <div className="landing-hero-ring" />
            <svg className="landing-hero-pulse" width="120" height="120" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 12H18L15 21L9 3L6 12H2" />
            </svg>
          </div>
        </div>

        <div className="landing-hero-copy">
          <p className="landing-hero-kicker">
            <span className="landing-hero-kicker-dot" />
            Uncertainty-Aware Diagnostics
          </p>
          <h1 className="landing-hero-title">
            Medical AI You Can <span>Actually Trust</span>
          </h1>
          <p className="landing-hero-subtitle">
            SafeDx helps clinicians and developers work with AI predictions more responsibly
            by making uncertainty visible, actionable, and easier to understand.
          </p>
          <div className="landing-hero-actions">
            <Link to="/register" className="landing-btn-primary">
              Explore the Platform
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="5" y1="12" x2="19" y2="12" />
                <polyline points="12 5 19 12 12 19" />
              </svg>
            </Link>
            <a href="#how-it-works" className="landing-btn-secondary">
              Learn How It Works
            </a>
          </div>
        </div>
      </section>

      {/* ── Marquee bands ── */}
      <div className="landing-marquee-section">
        <MarqueeBand items={devMarquee} variant="dev" />
        <MarqueeBand items={clinicalMarquee} variant="clinical" />
      </div>

      {/* ── Value pillars ── */}
      <section className="landing-section landing-pillars">
        <p className="landing-section-label">Why SafeDx</p>
        <h2 className="landing-section-heading">
          AI predictions are only useful when you can trust them
        </h2>
        <p className="landing-section-desc">
          SafeDx brings calibration, transparency, and clinical context together in one platform
          so every prediction carries the uncertainty information it should.
        </p>

        <div className="landing-pillars-grid">
          <div className="landing-pillar-card">
            <div className="landing-pillar-icon landing-pillar-icon--blue">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 12H18L15 21L9 3L6 12H2" />
              </svg>
            </div>
            <h3 className="landing-pillar-title">Uncertainty-Aware Predictions</h3>
            <p className="landing-pillar-text">
              Every prediction includes calibrated uncertainty information so clinicians know
              exactly how confident the model is and where its limits lie.
            </p>
          </div>

          <div className="landing-pillar-card">
            <div className="landing-pillar-icon landing-pillar-icon--teal">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
              </svg>
            </div>
            <h3 className="landing-pillar-title">Calibration for Safer AI</h3>
            <p className="landing-pillar-text">
              Run conformal prediction calibration on your own models to align confidence
              scores with real-world reliability before clinical deployment.
            </p>
          </div>

          <div className="landing-pillar-card">
            <div className="landing-pillar-icon landing-pillar-icon--green">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="2" y="3" width="20" height="14" rx="2" ry="2" />
                <line x1="8" y1="21" x2="16" y2="21" />
                <line x1="12" y1="17" x2="12" y2="21" />
              </svg>
            </div>
            <h3 className="landing-pillar-title">Clinical + Technical Workflows</h3>
            <p className="landing-pillar-text">
              One unified platform for clinicians reviewing cases and developers calibrating
              models — designed around how each role actually works.
            </p>
          </div>
        </div>
      </section>

      {/* ── How it works ── */}
      <section className="landing-how" id="how-it-works">
        <div className="landing-how-inner">
          <div className="landing-how-header">
            <p className="landing-section-label">How It Works</p>
            <h2 className="landing-section-heading">
              From image upload to actionable insight
            </h2>
            <p className="landing-section-desc">
              Three simple steps take a clinician from a raw image to a transparent,
              uncertainty-aware diagnostic recommendation.
            </p>
          </div>

          <div className="landing-steps-grid">
            <div className="landing-step">
              <span className="landing-step-number">1</span>
              <div className="landing-step-connector" />
              <h3 className="landing-step-title">Upload Case</h3>
              <p className="landing-step-text">
                The clinician uploads a patient chest X-ray image into SafeDx.
                The interface is simple — select a patient, attach the image, and submit.
              </p>
            </div>
            <div className="landing-step">
              <span className="landing-step-number">2</span>
              <div className="landing-step-connector" />
              <h3 className="landing-step-title">AI Analysis</h3>
              <p className="landing-step-text">
                SafeDx runs the image through a calibrated model and returns predictions
                together with conformal prediction sets that quantify uncertainty per finding.
              </p>
            </div>
            <div className="landing-step">
              <span className="landing-step-number">3</span>
              <h3 className="landing-step-title">Review &amp; Decide</h3>
              <p className="landing-step-text">
                The clinician reviews each finding, its probability, and whether it falls
                inside the prediction set — then makes an informed clinical decision.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ── About ── */}
      <section className="landing-section landing-about" id="about">
        <div className="landing-about-grid">
          <div className="landing-about-copy">
            <p className="landing-section-label">About SafeDx</p>
            <h2 className="landing-section-heading">
              Making uncertainty visible in medical AI
            </h2>
            <p className="landing-about-text">
              SafeDx is a medical AI platform designed to make predictive uncertainty
              more visible and useful for both clinicians and developers. A model can
              produce strong predictions and still be poorly calibrated — confidence
              scores alone don't tell you whether uncertainty is being communicated
              responsibly.
            </p>
            <p className="landing-about-text">
              SafeDx combines calibration workflows, multilabel model support, and
              role-specific interfaces so teams can interact with diagnostic AI more
              transparently. Developers calibrate their models and validate the results.
              Clinicians review predictions alongside clear uncertainty information
              before making decisions.
            </p>
            <p className="landing-about-text">
              The goal is not to replace clinical judgement — it's to give clinicians
              the uncertainty context they need to exercise it better.
            </p>
          </div>
          <div className="landing-about-visual">
            <div className="landing-about-visual-inner">
              <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
              </svg>
              <span className="landing-about-visual-label">Safer AI, Better Decisions</span>
            </div>
          </div>
        </div>
      </section>

      {/* ── Developer section ── */}
      <section className="landing-devs" id="developers">
        <div className="landing-devs-inner">
          <p className="landing-section-label">For Developers &amp; Researchers</p>
          <h2 className="landing-section-heading">
            Built for the teams behind the models
          </h2>
          <p className="landing-section-desc">
            Everything you need to calibrate your trained model, validate the output,
            and download production-ready thresholds — guided step by step.
          </p>

          <div className="landing-devs-grid">
            {devFeatures.map((feat) => (
              <div key={feat.title} className="landing-dev-card">
                <div className="landing-dev-card-icon">{feat.icon}</div>
                <h4 className="landing-dev-card-title">{feat.title}</h4>
                <p className="landing-dev-card-text">{feat.text}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Final CTA ── */}
      <section className="landing-cta">
        <div className="landing-cta-inner">
          <p className="landing-section-label">Get Started</p>
          <h2 className="landing-section-heading">
            Ready to explore SafeDx?
          </h2>
          <p className="landing-section-desc">
            Discover a more transparent and uncertainty-aware approach to medical AI.
            Whether you're a clinician reviewing cases or a developer calibrating models,
            SafeDx is built for you.
          </p>
          <div className="landing-cta-actions">
            <Link to="/register" className="landing-btn-primary">
              Get Started
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="5" y1="12" x2="19" y2="12" />
                <polyline points="12 5 19 12 12 19" />
              </svg>
            </Link>
            <a href="#how-it-works" className="landing-btn-secondary">
              Learn More
            </a>
          </div>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="landing-footer">
        <div className="landing-footer-inner">
          <div>
            <span className="landing-footer-brand">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 12H18L15 21L9 3L6 12H2" />
              </svg>
              SafeDx
            </span>
            <span className="landing-footer-tagline">
              Uncertainty-aware medical diagnostics
            </span>
          </div>
          <div className="landing-footer-links">
            <a href="#how-it-works" className="landing-footer-link">How It Works</a>
            <a href="#about" className="landing-footer-link">About</a>
            <a href="#developers" className="landing-footer-link">Developers</a>
            <Link to="/login" className="landing-footer-link">Sign In</Link>
            <Link to="/register" className="landing-footer-link">Register</Link>
          </div>
          <span className="landing-footer-copy">&copy; 2026 SafeDx</span>
        </div>
      </footer>
    </div>
  );
}
