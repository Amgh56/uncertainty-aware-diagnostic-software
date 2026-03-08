import type { ReactNode } from "react";
import { Link } from "react-router-dom";

interface AuthLayoutProps {
  title: string;
  subtitle: string;
  footerText: string;
  footerLinkLabel: string;
  footerLinkTo: string;
  children: ReactNode;
}

function PulseIcon() {
  return (
    <svg
      className="auth-mark-icon"
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <path
        d="M22 12H18L15 21L9 3L6 12H2"
        stroke="currentColor"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export default function AuthLayout({
  title,
  subtitle,
  footerText,
  footerLinkLabel,
  footerLinkTo,
  children,
}: AuthLayoutProps) {
  return (
    <div className="auth-page">
      <div className="auth-shell">
        <header className="auth-header">
          <div className="auth-mark">
            <PulseIcon />
          </div>
          <div className="auth-copy">
            <h1 className="auth-title">{title}</h1>
            <p className="auth-subtitle">{subtitle}</p>
          </div>
        </header>

        {children}

        <p className="auth-footer">
          {footerText} <Link to={footerLinkTo} className="auth-link">{footerLinkLabel}</Link>
        </p>
      </div>
    </div>
  );
}
