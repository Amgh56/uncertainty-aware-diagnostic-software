import { API_URL } from "../../config";

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface Doctor {
  id: number;
  email: string;
  full_name: string;
  role: string;
}

async function parseError(response: Response, fallback: string): Promise<string> {
  try {
    const payload = await response.json();
    return payload.detail || payload.message || fallback;
  } catch {
    return fallback;
  }
}

function networkError(): string {
  return `Cannot reach API at ${API_URL}. Make sure backend is running and CORS allows your frontend URL.`;
}

export async function loginRequest(email: string, password: string): Promise<TokenResponse> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
  } catch {
    throw new Error(networkError());
  }
  if (!res.ok) {
    throw new Error(await parseError(res, "Login failed"));
  }
  return res.json();
}

export async function registerRequest(
  email: string,
  password: string,
  fullName: string,
  role: string = "clinician",
): Promise<TokenResponse> {
  const endpoint = role === "developer"
    ? `${API_URL}/developer/register`
    : `${API_URL}/auth/register`;

  let res: Response;
  try {
    res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password, full_name: fullName }),
    });
  } catch {
    throw new Error(networkError());
  }
  if (!res.ok) {
    throw new Error(await parseError(res, "Registration failed"));
  }
  return res.json();
}

export async function fetchCurrentUser(token: string): Promise<Doctor> {
  const res = await fetch(`${API_URL}/auth/me`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error("Invalid token");
  return res.json();
}
