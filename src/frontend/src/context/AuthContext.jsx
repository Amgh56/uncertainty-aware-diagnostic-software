import { createContext, useContext, useState, useEffect, useCallback } from "react";
import { API_URL } from "../api/diagnosticApi";

const AuthContext = createContext(null);

async function parseAuthError(response, fallbackMessage) {
  try {
    const payload = await response.json();
    return payload.detail || payload.message || fallbackMessage;
  } catch {
    return fallbackMessage;
  }
}

function toNetworkErrorMessage() {
  return `Cannot reach API at ${API_URL}. Make sure backend is running and CORS allows your frontend URL.`;
}

export function AuthProvider({ children }) {
  const [token, setToken] = useState(() => localStorage.getItem("token"));
  const [doctor, setDoctor] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!token) {
      setLoading(false);
      setDoctor(null);
      return;
    }

    let cancelled = false;
    fetch(`${API_URL}/auth/me`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => {
        if (!res.ok) throw new Error("Invalid token");
        return res.json();
      })
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

  const login = useCallback(async (email, password) => {
    let res;
    try {
      res = await fetch(`${API_URL}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
    } catch {
      throw new Error(toNetworkErrorMessage());
    }
    if (!res.ok) {
      throw new Error(await parseAuthError(res, "Login failed"));
    }
    const data = await res.json();
    localStorage.setItem("token", data.access_token);
    setToken(data.access_token);
  }, []);

  const register = useCallback(async (email, password, fullName) => {
    let res;
    try {
      res = await fetch(`${API_URL}/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password, full_name: fullName }),
      });
    } catch {
      throw new Error(toNetworkErrorMessage());
    }
    if (!res.ok) {
      throw new Error(await parseAuthError(res, "Registration failed"));
    }
    const data = await res.json();
    localStorage.setItem("token", data.access_token);
    setToken(data.access_token);
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
