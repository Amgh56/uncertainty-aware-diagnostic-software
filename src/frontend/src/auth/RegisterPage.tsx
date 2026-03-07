import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export default function RegisterPage() {
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("clinician");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const { register } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await register(email, password, fullName, role);
      navigate(role === "developer" ? "/developer" : "/home", { replace: true });
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-logo">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#2563eb"
               strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
          </svg>
        </div>
        <h1 className="auth-title">Create Account</h1>
        <p className="auth-subtitle">Register to use the Diagnostic System</p>

        <form onSubmit={handleSubmit} className="auth-form">
          {/* Role toggle */}
          <div className="auth-label">
            <div className="auth-role-toggle">
              <button
                type="button"
                className={`auth-role-btn ${role === "clinician" ? "auth-role-btn--active" : ""}`}
                onClick={() => setRole("clinician")}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
                </svg>
                Clinician
              </button>
              <button
                type="button"
                className={`auth-role-btn ${role === "developer" ? "auth-role-btn--active" : ""}`}
                onClick={() => setRole("developer")}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="16 18 22 12 16 6" />
                  <polyline points="8 6 2 12 8 18" />
                </svg>
                Developer
              </button>
            </div>
          </div>

          <label className="auth-label">
            Full Name
            <input type="text" className="auth-input" value={fullName}
                   onChange={(e) => setFullName(e.target.value)} required
                   placeholder={role === "developer" ? "Dr. Alice Researcher" : "Dr. John Smith"} />
          </label>
          <label className="auth-label">
            Email
            <input type="email" className="auth-input" value={email} placeholder="Abdullahmmmaghrabi@gmail.com"
                   onChange={(e) => setEmail(e.target.value)} required />
          </label>
          <label className="auth-label">
            Password
            <input type="password" className="auth-input" value={password} 
                   onChange={(e) => setPassword(e.target.value)} required
                   minLength={6} placeholder="123456@"/>
          </label>

          {error && <div className="auth-error">{error}</div>}

          <button type="submit" className="auth-submit-btn" disabled={submitting}>
            {submitting ? <><span className="spinner" /> Creating account...</> : "Register"}
          </button>
        </form>

        <p className="auth-footer">
          Already have an account? <Link to="/login" className="auth-link">Sign In</Link>
        </p>
      </div>
    </div>
  );
}
