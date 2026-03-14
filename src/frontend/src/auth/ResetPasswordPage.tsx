import { useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import AuthLayout from "./AuthLayout";
import PasswordInput from "./PasswordInput";
import { API_URL } from "../config";

export default function ResetPasswordPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  const email = searchParams.get("email") ?? "";
  const token = searchParams.get("token") ?? "";
  const timestamp = parseInt(searchParams.get("ts") ?? "0", 10);

  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [done, setDone] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isValidLink = email && token && timestamp > 0;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (password !== confirm) {
      setError("Passwords do not match.");
      return;
    }
    if (password.length < 6) {
      setError("Password must be at least 6 characters.");
      return;
    }

    setSubmitting(true);
    try {
      const res = await fetch(`${API_URL}/auth/reset-password`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, token, timestamp, new_password: password }),
      });
      const payload = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(payload.detail || "Reset failed. Please try again.");
      setDone(true);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  if (!isValidLink) {
    return (
      <AuthLayout
        title="Invalid Link"
        subtitle="This reset link is missing required information"
        footerText="Need a new link?"
        footerLinkLabel="Forgot Password"
        footerLinkTo="/forgot-password"
      >
        <section className="auth-panel">
          <div className="auth-error" style={{ marginTop: 0 }}>
            This reset link is invalid or incomplete. Please request a new one.
          </div>
        </section>
      </AuthLayout>
    );
  }

  return (
    <AuthLayout
      title="Reset Password"
      subtitle="Choose a new password for your account"
      footerText="Remembered your password?"
      footerLinkLabel="Sign In"
      footerLinkTo="/login"
    >
      <section className="auth-panel" aria-label="Reset password form">
        {done ? (
          <div className="auth-success-box">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none"
              stroke="#059669" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="20 6 9 17 4 12" />
            </svg>
            <h3 className="auth-success-title">Password updated</h3>
            <p className="auth-success-text">
              Your password has been changed successfully.
            </p>
            <button
              type="button"
              className="auth-submit-btn"
              style={{ marginTop: 16 }}
              onClick={() => navigate("/login", { replace: true })}
            >
              Sign In
            </button>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="auth-form">
            <div className="auth-field">
              <label className="auth-label-text">Account</label>
              <p className="auth-reset-email">{email}</p>
            </div>

            <PasswordInput
              id="reset-password"
              label="New Password"
              value={password}
              placeholder="At least 6 characters"
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="new-password"
              required
            />
            <PasswordInput
              id="reset-confirm"
              label="Confirm Password"
              value={confirm}
              placeholder="Repeat your new password"
              onChange={(e) => setConfirm(e.target.value)}
              autoComplete="new-password"
              required
            />

            {error && <div className="auth-error">{error}</div>}

            <button type="submit" className="auth-submit-btn" disabled={submitting}>
              {submitting ? <><span className="spinner" /> Updating...</> : "Update Password"}
            </button>
          </form>
        )}
      </section>
    </AuthLayout>
  );
}
