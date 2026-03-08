import { useEffect, useState, type ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

interface ClinicianLayoutProps {
  title: string;
  subtitle: string;
  children: ReactNode;
}

function PulseIcon() {
  return (
    <svg
      className="clinician-brand-icon-svg"
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

function DashboardIcon() {
  return (
    <svg className="clinician-nav-icon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <rect x="3" y="3" width="7" height="7" rx="2" stroke="currentColor" strokeWidth="1.8" />
      <rect x="14" y="3" width="7" height="7" rx="2" stroke="currentColor" strokeWidth="1.8" />
      <rect x="3" y="14" width="7" height="7" rx="2" stroke="currentColor" strokeWidth="1.8" />
      <rect x="14" y="14" width="7" height="7" rx="2" stroke="currentColor" strokeWidth="1.8" />
    </svg>
  );
}

function PatientIcon() {
  return (
    <svg className="clinician-nav-icon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M20 21V19C20 16.7909 18.2091 15 16 15H8C5.79086 15 4 16.7909 4 19V21"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="12" cy="8" r="4" stroke="currentColor" strokeWidth="1.8" />
    </svg>
  );
}

function LogoutIcon() {
  return (
    <svg className="clinician-nav-icon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M9 21H5C3.89543 21 3 20.1046 3 19V5C3 3.89543 3.89543 3 5 3H9"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M16 17L21 12L16 7"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M21 12H9"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function ChevronIcon({ direction }: { direction: "left" | "right" | "down" }) {
  const d = direction === "left"
    ? "M15 18L9 12L15 6"
    : direction === "right"
      ? "M9 18L15 12L9 6"
      : "M6 9L12 15L18 9";

  return (
    <svg className="clinician-chevron-icon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d={d} stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function getInitials(fullName: string) {
  return fullName
    .split(" ")
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase() || "")
    .join("");
}

export default function ClinicianLayout({ title, subtitle, children }: ClinicianLayoutProps) {
  const { pathname } = useLocation();
  const { doctor, logout } = useAuth();
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem("clinician-sidebar-collapsed") === "1");

  const dashboardActive = pathname === "/home" || pathname.startsWith("/predictions/");
  const newPatientActive = pathname === "/dashboard";
  const doctorInitials = doctor ? getInitials(doctor.full_name) : "DR";

  useEffect(() => {
    localStorage.setItem("clinician-sidebar-collapsed", collapsed ? "1" : "0");
  }, [collapsed]);

  return (
    <div className={`clinician-layout${collapsed ? " clinician-layout--collapsed" : ""}`}>
      <aside className="clinician-sidebar">
        <div className="clinician-sidebar-top">
          <div className="clinician-brand">
            <div className="clinician-brand-icon">
              <PulseIcon />
            </div>
            <div className="clinician-brand-copy">
              <span className="clinician-brand-kicker">Uncertainty-aware</span>
              <span className="clinician-brand-title">Diagnostic System</span>
              <span className="clinician-brand-subtitle">Medical workspace</span>
            </div>
          </div>

          <button
            type="button"
            className="clinician-sidebar-toggle"
            onClick={() => setCollapsed((value) => !value)}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            aria-pressed={collapsed}
          >
            <ChevronIcon direction={collapsed ? "right" : "left"} />
          </button>
        </div>

        <div className="clinician-nav-section">
          <p className="clinician-nav-heading">Explore</p>
          <nav className="clinician-nav" aria-label="Clinician navigation">
            <Link
              to="/home"
              className={`clinician-nav-item${dashboardActive ? " clinician-nav-item--active" : ""}`}
              aria-current={dashboardActive ? "page" : undefined}
              title={collapsed ? "Dashboard" : undefined}
            >
              <DashboardIcon />
              <span>Dashboard</span>
            </Link>
            <Link
              to="/dashboard"
              className={`clinician-nav-item${newPatientActive ? " clinician-nav-item--active" : ""}`}
              aria-current={newPatientActive ? "page" : undefined}
              title={collapsed ? "New Patient" : undefined}
            >
              <PatientIcon />
              <span>New Patient</span>
            </Link>
          </nav>
        </div>

        <div className="clinician-sidebar-footer">
          {doctor && (
            <div className="clinician-user-card" title={collapsed ? doctor.full_name : undefined}>
              <div className="clinician-user-avatar">{doctorInitials}</div>
              <span className="clinician-user-name">{doctor.full_name}</span>
              <span className="clinician-user-chevron">
                <ChevronIcon direction="down" />
              </span>
            </div>
          )}

          <button
            type="button"
            className="clinician-logout"
            onClick={logout}
            title={collapsed ? "Logout" : undefined}
          >
            <LogoutIcon />
            <span>Logout</span>
          </button>
        </div>
      </aside>

      <div className="clinician-content-shell">
        <main className="clinician-main">
          <header className="clinician-page-header">
            <div>
              <h1 className="clinician-page-title">{title}</h1>
              <p className="clinician-page-subtitle">{subtitle}</p>
            </div>
          </header>

          {children}
        </main>
      </div>
    </div>
  );
}
