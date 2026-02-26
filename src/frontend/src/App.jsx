import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import ProtectedRoute from "./components/ProtectedRoute";
import LoginPage from "./components/LoginPage";
import RegisterPage from "./components/RegisterPage";
import HomePage from "./components/HomePage";
import DiagnosticDashboard from "./components/DiagnosticDashboard";
import PredictionDetail from "./components/PredictionDetail";
import "./styles/diagnostic-dashboard.css";
import "./styles/auth.css";

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route path="/home" element={
            <ProtectedRoute><HomePage /></ProtectedRoute>
          } />
          <Route path="/dashboard" element={
            <ProtectedRoute><DiagnosticDashboard /></ProtectedRoute>
          } />
          <Route path="/predictions/:id" element={
            <ProtectedRoute><PredictionDetail /></ProtectedRoute>
          } />
          <Route path="*" element={<Navigate to="/home" replace />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}
