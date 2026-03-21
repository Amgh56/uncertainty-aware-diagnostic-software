import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { fetchCurrentUser } from "./api/authApi";
import AuthLayout from "./AuthLayout";
import PasswordInput from "./PasswordInput";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const result = await login(email, password);
      if (result.is_verified === false) {
        navigate("/verify-email", { state: { email }, replace: true });
        return;
      }
      const me = await fetchCurrentUser(result.access_token);
      navigate(me.role === "developer" ? "/developer" : "/home", { replace: true });
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <AuthLayout
      title="Welcome Back"
      subtitle="Uncertainty-aware diagnostics you can trust"
      footerText="Don't have an account?"
      footerLinkLabel="Register"
      footerLinkTo="/register"
    >
      <section className="auth-panel" aria-label="Sign in form">
        <form onSubmit={handleSubmit} className="auth-form">
          <div className="auth-field">
            <label className="auth-label-text" htmlFor="login-email">Email</label>
            <input
              id="login-email"
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
            id="login-password"
            label="Password"
            value={password}
            placeholder="Enter your password"
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password"
            required
          />
          <div className="auth-forgot-row">
            <Link to="/forgot-password" className="auth-link auth-forgot-link">
              Forgot password?
            </Link>
          </div>

          {error && <div className="auth-error">{error}</div>}

          <button type="submit" className="auth-submit-btn" disabled={submitting}>
            {submitting ? <><span className="spinner" /> Signing in...</> : "Sign In"}
          </button>
        </form>
      </section>
    </AuthLayout>
  );
}
