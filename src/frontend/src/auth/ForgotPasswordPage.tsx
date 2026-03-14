import { useState } from "react";
import AuthLayout from "./AuthLayout";
import { API_URL } from "../config";

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [sent, setSent] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const res = await fetch(`${API_URL}/auth/forgot-password`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email }),
      });
      if (!res.ok) {
        const payload = await res.json().catch(() => ({}));
        throw new Error(payload.detail || "Something went wrong. Please try again.");
      }
      setSent(true);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <AuthLayout
      title="Forgot Password"
      subtitle="We'll send a reset link to your email"
      footerText="Remembered your password?"
      footerLinkLabel="Sign In"
      footerLinkTo="/login"
    >
      <section className="auth-panel" aria-label="Forgot password form">
        {sent ? (
          <div className="auth-success-box">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none"
              stroke="#059669" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <polyline points="9 12 11 14 15 10" />
            </svg>
            <h3 className="auth-success-title">Check your inbox</h3>
            <p className="auth-success-text">
              If <strong>{email}</strong> is registered with SafeDx, you'll receive
              a password reset link within a few minutes.
            </p>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="auth-form">
            <div className="auth-field">
              <label className="auth-label-text" htmlFor="forgot-email">Email address</label>
              <input
                id="forgot-email"
                type="email"
                className="auth-input-control"
                value={email}
                placeholder="doctor@example.com"
                onChange={(e) => setEmail(e.target.value)}
                autoComplete="email"
                required
              />
            </div>

            {error && <div className="auth-error">{error}</div>}

            <button type="submit" className="auth-submit-btn" disabled={submitting}>
              {submitting ? <><span className="spinner" /> Sending...</> : "Send Reset Link"}
            </button>
          </form>
        )}
      </section>
    </AuthLayout>
  );
}
