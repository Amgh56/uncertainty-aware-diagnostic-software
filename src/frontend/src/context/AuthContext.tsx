import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from "react";
import { loginRequest, registerRequest, fetchCurrentUser, type Doctor, type TokenResponse } from "../auth/api/authApi";

interface AuthContextValue {
  token: string | null;
  doctor: Doctor | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<TokenResponse>;
  register: (email: string, password: string, fullName: string, role?: string) => Promise<TokenResponse>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(() => localStorage.getItem("token"));
  const [doctor, setDoctor] = useState<Doctor | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!token) {
      setLoading(false);
      setDoctor(null);
      return;
    }

    let cancelled = false;
    fetchCurrentUser(token)
      .then((data) => {
        if (!cancelled) {
          setDoctor(data);
          setLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) {
          localStorage.removeItem("token");
          setToken(null);
          setDoctor(null);
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [token]);

  const login = useCallback(async (email: string, password: string) => {
    const data = await loginRequest(email, password);
    localStorage.setItem("token", data.access_token);
    setToken(data.access_token);
    return data;
  }, []);

  const register = useCallback(async (email: string, password: string, fullName: string, role: string = "clinician") => {
    const data = await registerRequest(email, password, fullName, role);
    localStorage.setItem("token", data.access_token);
    setToken(data.access_token);
    return data;
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem("token");
    setToken(null);
    setDoctor(null);
  }, []);

  return (
    <AuthContext.Provider value={{ token, doctor, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
