import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import ProtectedRoute from "./auth/ProtectedRoute";
import DeveloperRoute from "./auth/DeveloperRoute";
import LoginPage from "./auth/LoginPage";
import RegisterPage from "./auth/RegisterPage";
import ForgotPasswordPage from "./auth/ForgotPasswordPage";
import ResetPasswordPage from "./auth/ResetPasswordPage";
import HomePage from "./clinician/HomePage";
import DiagnosticDashboard from "./clinician/DiagnosticDashboard";
import PredictionDetail from "./clinician/PredictionDetail";
import CalibratedModelsPage from "./clinician/CalibratedModelsPage";
import DeveloperDashboard from "./developer/DeveloperDashboard";
import DeveloperHowToPage from "./developer/DeveloperHowToPage";
import ValidateCalibrationPage from "./developer/ValidateCalibrationPage";
import ModelLibraryPage from "./developer/ModelLibraryPage";
import LandingPage from "./landing/LandingPage";
import "./styles/diagnostic-dashboard.css";
import "./styles/auth.css";
import "./styles/landing.css";

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route path="/forgot-password" element={<ForgotPasswordPage />} />
          <Route path="/reset-password" element={<ResetPasswordPage />} />
          <Route path="/home" element={
            <ProtectedRoute><HomePage /></ProtectedRoute>
          } />
          <Route path="/dashboard" element={
            <ProtectedRoute><DiagnosticDashboard /></ProtectedRoute>
          } />
          <Route path="/predictions/:id" element={
            <ProtectedRoute><PredictionDetail /></ProtectedRoute>
          } />
          <Route path="/models" element={
            <ProtectedRoute><CalibratedModelsPage /></ProtectedRoute>
          } />
          <Route path="/developer" element={
            <DeveloperRoute><Navigate to="/developer/how-to-calibrate" replace /></DeveloperRoute>
          } />
          <Route path="/developer/how-to-calibrate" element={
            <DeveloperRoute><DeveloperHowToPage /></DeveloperRoute>
          } />
          <Route path="/developer/calibrate" element={
            <DeveloperRoute><DeveloperDashboard /></DeveloperRoute>
          } />
          <Route path="/developer/validate" element={
            <DeveloperRoute><ValidateCalibrationPage /></DeveloperRoute>
          } />
          <Route path="/developer/models" element={
            <DeveloperRoute><ModelLibraryPage /></DeveloperRoute>
          } />
          <Route path="/" element={<LandingPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}
