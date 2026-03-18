import { useEffect, useRef, useState, type ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { tocItems } from "./DeveloperHowToPage";

interface DeveloperLayoutProps {
  title: string;
  subtitle: string;
  children: ReactNode;
  activeSection?: string;
}

function PulseIcon() {
  return (
    <svg className="developer-brand-icon-svg" viewBox="0 0 24 24" fill="none" aria-hidden="true">
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

function GuideIcon() {
  return (
    <svg className="developer-nav-icon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M4 5.5C4 4.67157 4.67157 4 5.5 4H11V20H5.5C4.67157 20 4 19.3284 4 18.5V5.5Z"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M20 5.5C20 4.67157 19.3284 4 18.5 4H13V20H18.5C19.3284 20 20 19.3284 20 18.5V5.5Z"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path d="M8 8H8.01M8 11H8.01M8 14H8.01" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}

function CalibrateIcon() {
  return (
    <svg className="developer-nav-icon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M12 3L19 7V17L12 21L5 17V7L12 3Z"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M9 12L11 14L15 10"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function ValidateIcon() {
  return (
    <svg className="developer-nav-icon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M3 3v18h18"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M7 16l4-6 4 4 5-8"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M18 6h2v2"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function ModelsIcon() {
  return (
    <svg className="developer-nav-icon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <rect
        x="3" y="7" width="18" height="14" rx="2"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M7 7V5a2 2 0 012-2h6a2 2 0 012 2v2"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M12 11v4M10 13h4"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
    </svg>
  );
}

function LogoutIcon() {
  return (
    <svg className="developer-nav-icon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
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
    <svg className="developer-chevron-icon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
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

export default function DeveloperLayout({ title, subtitle, children, activeSection }: DeveloperLayoutProps) {
  const { pathname } = useLocation();
  const { doctor, logout } = useAuth();
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem("developer-sidebar-collapsed") === "1");
  const [profileMenuOpen, setProfileMenuOpen] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const profileMenuRef = useRef<HTMLDivElement | null>(null);
  const guideActive = pathname === "/developer/how-to-calibrate" || pathname === "/developer";
  const calibrateActive = pathname === "/developer/calibrate";
  const validateActive = pathname === "/developer/validate";
  const modelsActive = pathname === "/developer/models";
  const developerInitials = doctor ? getInitials(doctor.full_name) : "DR";

  useEffect(() => {
    localStorage.setItem("developer-sidebar-collapsed", collapsed ? "1" : "0");
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
    <div className={`developer-layout${collapsed ? " developer-layout--collapsed" : ""}${mobileOpen ? " developer-layout--mobile-open" : ""}`}>

      {/* Mobile top bar */}
      <div className="developer-mobile-topbar">
        <button
          type="button"
          className="developer-mobile-hamburger"
          onClick={() => setMobileOpen(true)}
          aria-label="Open navigation"
        >
          <HamburgerIcon />
        </button>
        <div className="developer-mobile-brand-wrap">
          <div className="developer-mobile-brand-icon">
            <PulseIcon />
          </div>
          <span className="developer-mobile-brand">SafeDx</span>
        </div>
      </div>

      {mobileOpen && (
        <div
          className="developer-mobile-backdrop"
          onClick={() => setMobileOpen(false)}
          aria-hidden="true"
        />
      )}

      <aside className="developer-sidebar">
        <div className="developer-sidebar-top">
          <button
            type="button"
            className="developer-sidebar-toggle"
            onClick={() => setCollapsed((value) => !value)}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            aria-pressed={collapsed}
          >
            <HamburgerIcon />
          </button>

          <div className="developer-brand">
            <div className="developer-brand-icon">
              <PulseIcon />
            </div>
            <div className="developer-brand-copy">
              <span className="developer-brand-title">SafeDx</span>
            </div>
          </div>
        </div>

        <div className="developer-sidebar-divider" aria-hidden="true" />

        <div className="developer-nav-section">
          <p className="developer-nav-heading">Workspace</p>
          <nav className="developer-nav" aria-label="Developer navigation">
            <div className="developer-nav-group">
              <Link
                to="/developer/how-to-calibrate"
                className={`developer-nav-item${guideActive ? " developer-nav-item--active" : ""}`}
                aria-current={guideActive ? "page" : undefined}
                title={collapsed ? "Developer Guide" : undefined}
              >
                <GuideIcon />
                <span>Developer Guide</span>
              </Link>
              {guideActive && !collapsed && (
                <div className="developer-toc-dropdown">
                  {tocItems.map((item) => (
                    <a
                      key={item.id}
                      href={`#${item.id}`}
                      className={`developer-toc-item${activeSection === item.id ? " developer-toc-item--active" : ""}`}
                      onClick={(e) => {
                        e.preventDefault();
                        document.getElementById(item.id)?.scrollIntoView({ behavior: "smooth" });
                      }}
                    >
                      {item.label}
                    </a>
                  ))}
                </div>
              )}
            </div>
            <Link
              to="/developer/calibrate"
              className={`developer-nav-item${calibrateActive ? " developer-nav-item--active" : ""}`}
              aria-current={calibrateActive ? "page" : undefined}
              title={collapsed ? "Calibrate Your Model" : undefined}
            >
              <CalibrateIcon />
              <span>Calibrate Your Model</span>
            </Link>
            <Link
              to="/developer/validate"
              className={`developer-nav-item${validateActive ? " developer-nav-item--active" : ""}`}
              aria-current={validateActive ? "page" : undefined}
              title={collapsed ? "Validate Your Calibration" : undefined}
            >
              <ValidateIcon />
              <span>Validate Your Calibration</span>
            </Link>
            <Link
              to="/developer/models"
              className={`developer-nav-item${modelsActive ? " developer-nav-item--active" : ""}`}
              aria-current={modelsActive ? "page" : undefined}
              title={collapsed ? "Calibrated Models" : undefined}
            >
              <ModelsIcon />
              <span>Calibrated Models</span>
            </Link>
          </nav>
        </div>

        <div className="developer-sidebar-footer">
          {doctor && (
            <div className="developer-user-menu-wrap" ref={profileMenuRef}>
              <button
                type="button"
                className={`developer-user-card${profileMenuOpen ? " developer-user-card--active" : ""}`}
                title={collapsed ? doctor.full_name : undefined}
                aria-label={collapsed ? doctor.full_name : "Open profile menu"}
                aria-expanded={profileMenuOpen}
                aria-haspopup="menu"
                onClick={() => setProfileMenuOpen((value) => !value)}
              >
                <div className="developer-user-avatar">{developerInitials}</div>
                <span className="developer-user-name">{doctor.full_name}</span>
                <span className="developer-user-chevron">
                  <ChevronIcon direction={profileMenuOpen ? "up" : "down"} />
                </span>
              </button>

              {profileMenuOpen && (
                <div className="developer-user-menu" role="menu">
                  <button
                    type="button"
                    className="developer-user-menu-item"
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

      <div className="developer-content-shell">
        <main className="developer-main">
          <header className="developer-page-header">
            <div>
              <h1 className="developer-page-title">{title}</h1>
              <p className="developer-page-subtitle">{subtitle}</p>
            </div>
          </header>

          {children}
        </main>
      </div>
    </div>
  );
}
