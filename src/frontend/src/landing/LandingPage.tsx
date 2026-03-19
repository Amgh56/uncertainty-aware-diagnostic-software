import { useEffect, useState, useRef } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import doctorsIllustration from "../assets/doctors-illustration.svg";
import {
  Stethoscope,
  BarChart3,
  Eye,
  FlaskConical,
  Code,
  Activity,
  Shield,
  Monitor,
  Play,
  Download,
  ChevronDown,
  Linkedin,
  Github,
  Mail,
  Phone,
  ShieldCheck,
  FileCheck,
  Users,
  UploadCloud,
  ClipboardCheck,
} from "lucide-react";

/* ── Marquee data ── */
const marqueeItems: { text: string; audience: "clinician" | "dev" }[] = [
  { text: "Patient-Centered Analysis", audience: "clinician" },
  { text: "Calibration Workflows", audience: "dev" },
  { text: "Actionable AI Insights", audience: "clinician" },
  { text: "Multilabel Model Support", audience: "dev" },
  { text: "Visual Diagnostic Support", audience: "clinician" },
  { text: "Conformal Prediction", audience: "dev" },
  { text: "Clearer Decision Support", audience: "clinician" },
  { text: "Threshold Control", audience: "dev" },
  { text: "Uncertainty-Aware Predictions", audience: "clinician" },
  { text: "Validation Metrics", audience: "dev" },
  { text: "Explainable Results", audience: "clinician" },
  { text: "Research-Ready Pipelines", audience: "dev" },
  { text: "Safer Clinical Confidence", audience: "clinician" },
  { text: "Model Upload & Export", audience: "dev" },
];

/* ── Dev features data ── */
const devFeatures = [
  {
    title: "Upload and Calibrate Models",
    text: "Support for TorchScript and full PyTorch models out of the box.",
    icon: UploadCloud,
  },
  {
    title: "Run Calibration Pipelines",
    text: "Execute conformal prediction calibration with a single click and monitor job progress in real time.",
    icon: Play,
  },
  {
    title: "Download Thresholds",
    text: "Retrieve calibrated thresholds and metrics as structured JSON, ready for integration.",
    icon: Download,
  },
  {
    title: "Validate Results",
    text: "Verify your calibration output meets the target miscoverage rate before deploying to production.",
    icon: FileCheck,
  },
  {
    title: "Release to Researchers",
    text: "Share your calibrated model with the research community for further study and collaboration.",
    icon: FlaskConical,
  },
  {
    title: "Release to Doctors",
    text: "Deploy your calibrated model through the clinical release feature, making it accessible to clinicians.",
    icon: Users,
  },
];

/* ── FAQ data ── */
const faqItems = [
  {
    q: "What is SafeDx?",
    a: "SafeDx is a medical AI platform that helps developers calibrate their machine learning models and gives clinicians uncertainty-aware diagnostic tools. It combines calibration workflows with clinical review interfaces.",
  },
  {
    q: "What is model calibration?",
    a: "Calibration adjusts your model's confidence scores so they align with real-world accuracy. After calibration, when your model says 80% confident, it genuinely means the label is correct about 80% of the time.",
  },
  {
    q: "What is conformal prediction?",
    a: "Conformal prediction is a statistical framework that produces prediction sets with coverage guarantees. Instead of a single prediction, you get a set of labels that is guaranteed to contain the true label with a probability you choose (1 \u2212 \u03b1).",
  },
  {
    q: "Who is SafeDx for?",
    a: "SafeDx serves two audiences: developers and researchers who need to calibrate and validate their models before deployment, and clinicians who need to review AI predictions with clear uncertainty information.",
  },
  {
    q: "What model formats does SafeDx support?",
    a: "SafeDx currently supports TorchScript (.pt) files (recommended) and full model (.pth) files for multilabel classification models.",
  },
  {
    q: "Is SafeDx free to use?",
    a: "SafeDx is currently available for research and demonstration purposes. Contact us for more information about access.",
  },
];

/* ── FAQ Item component ── */
function FaqItem({ q, a }: { q: string; a: string }) {
  const [open, setOpen] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);

  return (
    <div className={`landing-faq-item${open ? " landing-faq-item--open" : ""}`}>
      <button
        type="button"
        className="landing-faq-trigger"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span>{q}</span>
        <ChevronDown size={18} className="landing-faq-chevron" />
      </button>
      <div
        className="landing-faq-content"
        ref={contentRef}
        style={{ maxHeight: open ? contentRef.current?.scrollHeight : 0 }}
      >
        <p className="landing-faq-answer">{a}</p>
      </div>
    </div>
  );
}

/* ── Page ── */
export default function LandingPage() {
  const { token } = useAuth();
  const navigate = useNavigate();
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    if (token) navigate("/home", { replace: true });
  }, [token, navigate]);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 10);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // Smooth scroll for anchor links
  const scrollTo = (id: string) => (e: React.MouseEvent) => {
    e.preventDefault();
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
  };

  if (token) return null;

  return (
    <div className="landing-page">
      {/* ═══ SECTION 1: NAVBAR ═══ */}
      <nav className={`landing-nav${scrolled ? " landing-nav--scrolled" : ""}`}>
        <div className="landing-nav-inner">
          <div className="landing-nav-left">
            <span className="landing-nav-brand">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2563eb" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 12H18L15 21L9 3L6 12H2" />
              </svg>
              SafeDx
            </span>
          </div>
          <div className="landing-nav-center">
            <a href="#about" onClick={scrollTo("about")} className="landing-nav-link">About</a>
            <a href="#faq" onClick={scrollTo("faq")} className="landing-nav-link">FAQ</a>
            <a href="#developers" onClick={scrollTo("developers")} className="landing-nav-link">Developers</a>
            <a href="#clinicians" onClick={scrollTo("clinicians")} className="landing-nav-link">Clinicians</a>
          </div>
          <div className="landing-nav-right">
            <Link to="/login" className="landing-nav-signin">Sign In</Link>
          </div>
        </div>
      </nav>

      {/* ═══ SECTION 2: HERO ═══ */}
      <section className="landing-hero">
        <div className="landing-hero-inner">
          <span className="landing-hero-badge">
            <span className="landing-hero-badge-dot" />
            UNCERTAINTY-AWARE DIAGNOSTICS
          </span>
          <h1 className="landing-hero-title">
            Medical AI You Can{" "}
            <span className="landing-hero-accent">Actually Trust</span>
          </h1>
          <p className="landing-hero-subtitle">
            SafeDx helps clinicians and developers work with AI predictions more responsibly
            by making uncertainty visible, actionable, and easier to understand.
          </p>
          <div className="landing-hero-actions">
            <Link to="/register" className="landing-btn-primary">
              Explore the Platform
            </Link>
            <a href="#why-safedx" onClick={scrollTo("why-safedx")} className="landing-btn-outline">
              Learn How It Works
            </a>
          </div>
        </div>
      </section>

      {/* ═══ SECTION 3: MARQUEE ═══ */}
      <div className="landing-marquee-section">
        <div className="landing-marquee-bar">
          <div className="landing-marquee-track">
            {[...marqueeItems, ...marqueeItems, ...marqueeItems].map((item, i) => (
              <span key={i} className={`landing-marquee-phrase landing-marquee-phrase--${item.audience}`}>
                <span className={`landing-marquee-dot landing-marquee-dot--${item.audience}`} />
                {item.audience === "clinician" ? (
                  <Stethoscope size={13} className="landing-marquee-icon" />
                ) : (
                  <Code size={13} className="landing-marquee-icon" />
                )}
                {item.text}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* ═══ SECTION 4: WHY SAFEDX ═══ */}
      <section className="landing-section" id="why-safedx">
        <div className="landing-section-inner">
          <div className="landing-section-header">
            <span className="landing-badge">WHY SAFEDX</span>
            <h2 className="landing-heading">
              AI predictions are only useful when you can trust them
            </h2>
            <p className="landing-subheading">
              SafeDx brings calibration, transparency, and clinical context together in one platform
              so every prediction carries the uncertainty information it should.
            </p>
          </div>
          <div className="landing-pillars-grid">
            <div className="landing-pillar-card">
              <div className="landing-pillar-icon landing-pillar-icon--blue">
                <Activity size={22} />
              </div>
              <h3 className="landing-pillar-title">Uncertainty-Aware Predictions</h3>
              <p className="landing-pillar-text">
                Every prediction includes calibrated uncertainty information so clinicians know
                exactly how confident the model is and where its limits lie.
              </p>
            </div>
            <div className="landing-pillar-card">
              <div className="landing-pillar-icon landing-pillar-icon--teal">
                <Shield size={22} />
              </div>
              <h3 className="landing-pillar-title">Calibration for Safer AI</h3>
              <p className="landing-pillar-text">
                Run conformal prediction calibration on your own models to align confidence
                scores with real-world reliability before clinical deployment.
              </p>
            </div>
            <div className="landing-pillar-card">
              <div className="landing-pillar-icon landing-pillar-icon--green">
                <Monitor size={22} />
              </div>
              <h3 className="landing-pillar-title">Clinical + Technical Workflows</h3>
              <p className="landing-pillar-text">
                One unified platform for clinicians reviewing cases and developers calibrating
                models — designed around how each role actually works.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ═══ SECTION 5: FOR CLINICIANS ═══ */}
      <section className="landing-clinicians" id="clinicians">
        <div className="landing-section-inner">
          <div className="landing-section-header landing-section-header--center">
            <span className="landing-badge">FOR CLINICIANS</span>
            <h2 className="landing-heading">From image upload to actionable insight</h2>
            <p className="landing-subheading">
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
                The clinician selects a calibrated model, uploads a medical image, and submits
                the case. SafeDx supports any image type as long as a matching model is available.
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

      {/* ═══ SECTION 6: FOR DEVELOPERS & RESEARCHERS (DARK) ═══ */}
      <section className="landing-devs" id="developers">
        <div className="landing-section-inner">
          <div className="landing-section-header landing-section-header--center">
            <span className="landing-badge landing-badge--dark">FOR DEVELOPERS &amp; RESEARCHERS</span>
            <h2 className="landing-heading landing-heading--light">
              Built for the teams behind the models
            </h2>
            <p className="landing-subheading landing-subheading--light">
              Everything you need to calibrate your trained model, validate the output,
              and download production-ready thresholds — guided step by step.
            </p>
          </div>
          <div className="landing-devs-grid">
            {devFeatures.map((feat) => (
              <div key={feat.title} className="landing-dev-card">
                <div className="landing-dev-card-icon">
                  <feat.icon size={18} />
                </div>
                <h4 className="landing-dev-card-title">{feat.title}</h4>
                <p className="landing-dev-card-text">{feat.text}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ═══ SECTION 7: ABOUT ═══ */}
      <section className="landing-section landing-about" id="about">
        <div className="landing-section-inner">
          <div className="landing-about-grid">
            <div className="landing-about-copy">
              <span className="landing-badge">ABOUT SAFEDX</span>
              <h2 className="landing-heading">Making uncertainty visible in medical AI</h2>
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
              <div className="landing-about-visual-card">
                <div className="landing-about-illustration-wrap">
                  <div className="landing-about-glow" />
                  <img
                    src={doctorsIllustration}
                    alt="Clinician illustration"
                    className="landing-about-illustration"
                  />
                </div>
                <div className="landing-about-card-text">
                  <span className="landing-about-visual-label">Safer AI, Better Decisions</span>
                  <span className="landing-about-visual-sub">Designed to help clinicians interpret uncertainty with more confidence</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ═══ SECTION 8: FAQ ═══ */}
      <section className="landing-section landing-faq-section" id="faq">
        <div className="landing-section-inner landing-faq-inner">
          <div className="landing-section-header landing-section-header--center">
            <span className="landing-badge">FREQUENTLY ASKED QUESTIONS</span>
            <h2 className="landing-heading">Got questions? We've got answers</h2>
          </div>
          <div className="landing-faq-list">
            {faqItems.map((item) => (
              <FaqItem key={item.q} q={item.q} a={item.a} />
            ))}
          </div>
        </div>
      </section>

      {/* ═══ FOOTER ═══ */}
      <footer className="landing-footer">
        <div className="landing-footer-main">
          <div className="landing-footer-brand-col">
            <div className="landing-footer-brand">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 12H18L15 21L9 3L6 12H2" />
              </svg>
              <span>SafeDx</span>
            </div>
            <p className="landing-footer-desc">
              SafeDx is an AI-powered medical diagnostic platform that helps clinicians
              and researchers use validated machine learning models more safely, clearly,
              and confidently through explainability and uncertainty-aware workflows.
            </p>
            <div className="landing-footer-socials">
              <a href="https://www.linkedin.com/in/abdullahmaghrabi/" target="_blank" rel="noopener noreferrer" className="landing-footer-social" aria-label="LinkedIn"><Linkedin size={16} /></a>
              <a href="https://github.com/Amgh56" target="_blank" rel="noopener noreferrer" className="landing-footer-social" aria-label="GitHub"><Github size={16} /></a>
            </div>
          </div>

          <div className="landing-footer-col">
            <h4 className="landing-footer-col-title">Quick Links</h4>
            <a href="#" className="landing-footer-link">Home</a>
            <a href="#about" onClick={scrollTo("about")} className="landing-footer-link">About</a>
            <a href="#why-safedx" onClick={scrollTo("why-safedx")} className="landing-footer-link">Features</a>
            <a href="#faq" onClick={scrollTo("faq")} className="landing-footer-link">FAQ</a>
          </div>

          <div className="landing-footer-col">
            <h4 className="landing-footer-col-title">Contact / Support</h4>
            <a href="mailto:Abdullahmmmaghrabi@gmail.com" className="landing-footer-contact">
              <Mail size={14} />
              <span>Abdullahmmmaghrabi@gmail.com</span>
            </a>
            <a href="tel:+447503485510" className="landing-footer-contact">
              <Phone size={14} />
              <span>+447503485510</span>
            </a>
          </div>
        </div>

        <div className="landing-footer-bottom">
          <span>&copy; 2026 SafeDx. All rights reserved.</span>
          <div className="landing-footer-bottom-links">
            <a href="https://www.linkedin.com/in/abdullahmaghrabi/" target="_blank" rel="noopener noreferrer" className="landing-footer-bottom-link">LinkedIn</a>
            <a href="https://github.com/Amgh56" target="_blank" rel="noopener noreferrer" className="landing-footer-bottom-link">GitHub</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
