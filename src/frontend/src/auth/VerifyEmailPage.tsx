import { useCallback, useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { verifyEmailOtp, resendEmailOtp, fetchCurrentUser } from "./api/authApi";
import { useAuth } from "../context/AuthContext";
import AuthLayout from "./AuthLayout";

const OTP_LENGTH = 6;
const RESEND_COOLDOWN = 60;

function maskEmail(email: string): string {
  const [local, domain] = email.split("@");
  if (!domain) return email;
  if (local.length <= 2) return `${local}@${domain}`;
  return `${local[0]}${"*".repeat(Math.min(local.length - 2, 6))}${local[local.length - 1]}@${domain}`;
}

export default function VerifyEmailPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { token } = useAuth();
  const state = location.state as { email?: string; role?: string } | null;
  const email = state?.email || "";

  const [otp, setOtp] = useState<string[]>(Array(OTP_LENGTH).fill(""));
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [verifying, setVerifying] = useState(false);
  const [resendCooldown, setResendCooldown] = useState(RESEND_COOLDOWN);
  const [resending, setResending] = useState(false);
  const inputRefs = useRef<(HTMLInputElement | null)[]>([]);

  // Redirect if no email
  useEffect(() => {
    if (!email) navigate("/register", { replace: true });
  }, [email, navigate]);

  // Resend countdown
  useEffect(() => {
    if (resendCooldown <= 0) return;
    const timer = setInterval(() => setResendCooldown((v) => v - 1), 1000);
    return () => clearInterval(timer);
  }, [resendCooldown]);

  // Auto-focus first input
  useEffect(() => {
    inputRefs.current[0]?.focus();
  }, []);

  const handleChange = useCallback(
    (index: number, value: string) => {
      if (!/^\d*$/.test(value)) return;

      const newOtp = [...otp];
      newOtp[index] = value.slice(-1);
      setOtp(newOtp);
      setError(null);

      if (value && index < OTP_LENGTH - 1) {
        inputRefs.current[index + 1]?.focus();
      }
    },
    [otp]
  );

  const handleKeyDown = useCallback(
    (index: number, e: React.KeyboardEvent) => {
      if (e.key === "Backspace" && !otp[index] && index > 0) {
        inputRefs.current[index - 1]?.focus();
      }
    },
    [otp]
  );

  const handlePaste = useCallback(
    (e: React.ClipboardEvent) => {
      e.preventDefault();
      const pasted = e.clipboardData.getData("text").replace(/\D/g, "").slice(0, OTP_LENGTH);
      if (!pasted) return;
      const newOtp = [...otp];
      for (let i = 0; i < pasted.length; i++) {
        newOtp[i] = pasted[i];
      }
      setOtp(newOtp);
      setError(null);
      const focusIndex = Math.min(pasted.length, OTP_LENGTH - 1);
      inputRefs.current[focusIndex]?.focus();
    },
    [otp]
  );

  const handleVerify = async () => {
    const code = otp.join("");
    if (code.length < OTP_LENGTH) {
      setError("Please enter the full 6-digit code.");
      return;
    }

    setVerifying(true);
    setError(null);
    try {
      await verifyEmailOtp(email, code);
      setSuccess(true);
      // Redirect after short success display
      setTimeout(async () => {
        if (token) {
          try {
            const me = await fetchCurrentUser(token);
            navigate(me.role === "developer" ? "/developer" : "/home", { replace: true });
          } catch {
            navigate("/login", { replace: true });
          }
        } else {
          navigate("/login", { replace: true });
        }
      }, 1200);
    } catch (err) {
      setError((err as Error).message);
      setOtp(Array(OTP_LENGTH).fill(""));
      inputRefs.current[0]?.focus();
    } finally {
      setVerifying(false);
    }
  };

  const handleResend = async () => {
    if (resendCooldown > 0 || resending) return;
    setResending(true);
    setError(null);
    try {
      await resendEmailOtp(email);
      setResendCooldown(RESEND_COOLDOWN);
      setOtp(Array(OTP_LENGTH).fill(""));
      inputRefs.current[0]?.focus();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setResending(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleVerify();
  };

  if (!email) return null;

  return (
    <AuthLayout
      title="Verify Your Email"
      subtitle="One last step to secure your account"
      footerText="Wrong email?"
      footerLinkLabel="Go back to Register"
      footerLinkTo="/register"
    >
      <section className="auth-panel" aria-label="Email verification">
        <form onSubmit={handleSubmit} className="auth-form">
          <div className="otp-description">
            <p>
              We sent a 6-digit code to <strong>{maskEmail(email)}</strong>.
              Enter it below to continue.
            </p>
          </div>

          <div className="otp-inputs" onPaste={handlePaste}>
            {otp.map((digit, i) => (
              <input
                key={i}
                ref={(el) => { inputRefs.current[i] = el; }}
                type="text"
                inputMode="numeric"
                maxLength={1}
                value={digit}
                onChange={(e) => handleChange(i, e.target.value)}
                onKeyDown={(e) => handleKeyDown(i, e)}
                className={`otp-input${error ? " otp-input--error" : ""}${success ? " otp-input--success" : ""}`}
                disabled={verifying || success}
                autoComplete="one-time-code"
              />
            ))}
          </div>

          {error && <div className="auth-error">{error}</div>}
          {success && <div className="otp-success">Email verified successfully!</div>}

          <button
            type="submit"
            className="auth-submit-btn"
            disabled={verifying || success || otp.join("").length < OTP_LENGTH}
          >
            {verifying ? (
              <><span className="spinner" /> Verifying...</>
            ) : success ? (
              "Verified!"
            ) : (
              "Verify Email"
            )}
          </button>

          <div className="otp-resend-row">
            <span className="otp-resend-text">Didn't receive the code?</span>
            <button
              type="button"
              className="otp-resend-btn"
              onClick={handleResend}
              disabled={resendCooldown > 0 || resending}
            >
              {resending
                ? "Sending..."
                : resendCooldown > 0
                  ? `Resend in ${resendCooldown}s`
                  : "Resend Code"}
            </button>
          </div>
        </form>
      </section>
    </AuthLayout>
  );
}
