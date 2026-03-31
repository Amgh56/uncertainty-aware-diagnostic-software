import { useEffect, useRef, useState, type ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";
import { Stethoscope } from "lucide-react";
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


function ModelsIcon() {
  return (
    <svg className="clinician-nav-icon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M12 2L2 7L12 12L22 7L12 2Z"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M2 17L12 22L22 17"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M2 12L12 17L22 12"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
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

function ChevronIcon({ direction }: { direction: "left" | "right" | "down" | "up" }) {
  const d = direction === "left"
    ? "M15 18L9 12L15 6"
    : direction === "right"
      ? "M9 18L15 12L9 6"
      : direction === "up"
        ? "M6 15L12 9L18 15"
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

function HamburgerIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <line x1="3" y1="6" x2="21" y2="6" />
      <line x1="3" y1="12" x2="21" y2="12" />
      <line x1="3" y1="18" x2="21" y2="18" />
    </svg>
  );
}

export default function ClinicianLayout({ title, subtitle, children }: ClinicianLayoutProps) {
  const { pathname } = useLocation();
  const { doctor, logout } = useAuth();
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem("clinician-sidebar-collapsed") === "1");
  const [profileMenuOpen, setProfileMenuOpen] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const profileMenuRef = useRef<HTMLDivElement | null>(null);

  const dashboardActive = pathname === "/home" || pathname.startsWith("/predictions/");
  const newPatientActive = pathname === "/dashboard";
  const modelsActive = pathname === "/models";
  const doctorInitials = doctor ? getInitials(doctor.full_name) : "DR";

  useEffect(() => {
    localStorage.setItem("clinician-sidebar-collapsed", collapsed ? "1" : "0");
  }, [collapsed]);

  useEffect(() => {
    const handlePointerDown = (event: MouseEvent) => {
      if (!profileMenuRef.current?.contains(event.target as Node)) {
        setProfileMenuOpen(false);
      }
    };

    document.addEventListener("mousedown", handlePointerDown);
    return () => document.removeEventListener("mousedown", handlePointerDown);
  }, []);

  useEffect(() => {
    setProfileMenuOpen(false);
    setMobileOpen(false);
  }, [pathname, collapsed]);

  return (
    <div className={`clinician-layout${collapsed ? " clinician-layout--collapsed" : ""}${mobileOpen ? " clinician-layout--mobile-open" : ""}`}>

      {/* Mobile top bar — only visible on small screens */}
      <div className="clinician-mobile-topbar">
        <button
          type="button"
          className="clinician-mobile-hamburger"
          onClick={() => setMobileOpen(true)}
          aria-label="Open navigation"
        >
          <HamburgerIcon />
        </button>
        <div className="clinician-mobile-brand-wrap">
          <div className="clinician-mobile-brand-icon">
            <PulseIcon />
          </div>
          <span className="clinician-mobile-brand">SafeDx</span>
        </div>
      </div>

      {/* Overlay backdrop — tapping it closes the drawer */}
      {mobileOpen && (
        <div
          className="clinician-mobile-backdrop"
          onClick={() => setMobileOpen(false)}
          aria-hidden="true"
        />
      )}

      <aside className="clinician-sidebar">
        <div className="clinician-sidebar-top">
          <div className="clinician-brand">
            <div className="clinician-brand-icon">
              <PulseIcon />
            </div>
            <div className="clinician-brand-copy">
              <span className="clinician-brand-title">SafeDx</span>
            </div>
          </div>

          <button
            type="button"
            className="clinician-sidebar-toggle"
            onClick={() => setCollapsed((value) => !value)}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            aria-pressed={collapsed}
          >
            <HamburgerIcon />
          </button>
        </div>

        <div className="clinician-sidebar-divider" aria-hidden="true" />

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
              title={collapsed ? "New Diagnosis" : undefined}
            >
              <Stethoscope className="clinician-nav-icon" />
              <span>New Diagnosis</span>
            </Link>
            <Link
              to="/models"
              className={`clinician-nav-item${modelsActive ? " clinician-nav-item--active" : ""}`}
              aria-current={modelsActive ? "page" : undefined}
              title={collapsed ? "Calibrated Models" : undefined}
            >
              <ModelsIcon />
              <span>Calibrated Models</span>
            </Link>
          </nav>
        </div>

        <div className="clinician-sidebar-footer">
          {doctor && (
            <div className="clinician-user-menu-wrap" ref={profileMenuRef}>
              <button
                type="button"
                className={`clinician-user-card${profileMenuOpen ? " clinician-user-card--active" : ""}`}
                title={collapsed ? doctor.full_name : undefined}
                aria-label={collapsed ? doctor.full_name : "Open profile menu"}
                aria-expanded={profileMenuOpen}
                aria-haspopup="menu"
                onClick={() => setProfileMenuOpen((value) => !value)}
              >
                <div className="clinician-user-avatar">{doctorInitials}</div>
                <span className="clinician-user-name">{doctor.full_name}</span>
                <span className="clinician-user-chevron">
                  <ChevronIcon direction={profileMenuOpen ? "up" : "down"} />
                </span>
              </button>

              {profileMenuOpen && (
                <div className="clinician-user-menu" role="menu">
                  <button
                    type="button"
                    className="clinician-user-menu-item"
                    role="menuitem"
                    onClick={logout}
                  >
                    <LogoutIcon />
                    <span>Logout</span>
                  </button>
                </div>
              )}
            </div>
          )}
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
