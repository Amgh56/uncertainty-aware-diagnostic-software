import { Navigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

/**
 * Route guard that requires:
 *  1. A valid JWT (same as ProtectedRoute)
 *  2. doctor.role === "developer"
 *
 * Non-developers are redirected to /home.
 * Unauthenticated users are redirected to /login.
 */
export default function DeveloperRoute({ children }) {
  const { token, doctor, loading } = useAuth();

  if (loading) {
    return (
      <div style={{ display: "flex", justifyContent: "center", alignItems: "center", height: "100vh" }}>
        <div className="spinner-large" />
      </div>
    );
  }

  if (!token) {
    return <Navigate to="/login" replace />;
  }

  if (doctor && doctor.role !== "developer") {
    return <Navigate to="/home" replace />;
  }

  return children;
}
