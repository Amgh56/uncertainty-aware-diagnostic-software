import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import AuthLayout from "./AuthLayout";
import PasswordInput from "./PasswordInput";

type Role = "clinician" | "developer";

function ClinicianIcon() {
  return (
    <svg className="auth-role-icon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M12 5V19M5 12H19"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function DeveloperIcon() {
  return (
    <svg className="auth-role-icon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M8 8L4 12L8 16M16 8L20 12L16 16"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export default function RegisterPage() {
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<Role>("clinician");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const { register } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const result = await register(email, password, fullName, role);
      if (result.is_verified === false) {
        navigate("/verify-email", { state: { email, role }, replace: true });
      } else {
        navigate(role === "developer" ? "/developer" : "/home", { replace: true });
      }
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <AuthLayout
      title="Create Account"
      subtitle="Join the diagnostic platform"
      footerText="Already have an account?"
      footerLinkLabel="Sign In"
      footerLinkTo="/login"
    >
      <section className="auth-panel" aria-label="Register form">
        <form onSubmit={handleSubmit} className="auth-form">
          <div className="auth-role-section">
            <div className="auth-role-toggle" role="group" aria-label="Select role">
              <button
                type="button"
                aria-pressed={role === "clinician"}
                className={`auth-role-option ${role === "clinician" ? "auth-role-option--active" : ""}`}
                onClick={() => setRole("clinician")}
              >
                <ClinicianIcon />
                Clinician
              </button>
              <button
                type="button"
                aria-pressed={role === "developer"}
                className={`auth-role-option ${role === "developer" ? "auth-role-option--active" : ""}`}
                onClick={() => setRole("developer")}
              >
                <DeveloperIcon />
                Developer
              </button>
            </div>
          </div>

          <div className="auth-field">
            <label className="auth-label-text" htmlFor="register-full-name">Full Name</label>
            <input
              id="register-full-name"
              type="text"
              className="auth-input-control"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              autoComplete="name"
              required
              placeholder={role === "developer" ? "Dr. Alice Researcher" : "Dr. John Smith"}
            />
          </div>
          <div className="auth-field">
            <label className="auth-label-text" htmlFor="register-email">Email</label>
            <input
              id="register-email"
              type="email"
              className="auth-input-control"
              value={email}
              placeholder="doctor@example.com"
              onChange={(e) => setEmail(e.target.value)}
              autoComplete="email"
              required
            />
          </div>
          <PasswordInput
            id="register-password"
            label="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            minLength={6}
            placeholder="Create a secure password"
            autoComplete="new-password"
            required
          />

          {error && <div className="auth-error">{error}</div>}

          <button type="submit" className="auth-submit-btn" disabled={submitting}>
            {submitting ? <><span className="spinner" /> Creating account...</> : "Register"}
          </button>
        </form>
      </section>
    </AuthLayout>
  );
}
