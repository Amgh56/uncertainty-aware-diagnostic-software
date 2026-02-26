import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export default function RegisterPage() {
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const { register } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await register(email, password, fullName);
      navigate("/home", { replace: true });
    } catch (err) {
      setError(err.message);
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
          <label className="auth-label">
            Full Name
            <input type="text" className="auth-input" value={fullName}
                   onChange={(e) => setFullName(e.target.value)} required
                   placeholder="Dr. John Smith" />
          </label>
          <label className="auth-label">
            Email
            <input type="email" className="auth-input" value={email}
                   onChange={(e) => setEmail(e.target.value)} required />
          </label>
          <label className="auth-label">
            Password
            <input type="password" className="auth-input" value={password}
                   onChange={(e) => setPassword(e.target.value)} required
                   minLength={6} placeholder="At least 6 characters" />
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
