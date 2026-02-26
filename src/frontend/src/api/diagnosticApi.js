const DEFAULT_API_URL = "http://localhost:8000";

export const API_URL = import.meta.env.VITE_API_URL || DEFAULT_API_URL;

async function parseError(response) {
  try {
    const payload = await response.json();
    return payload.detail || payload.message || "Request failed";
  } catch {
    return "Request failed";
  }
}

function authHeaders(token) {
  return { Authorization: `Bearer ${token}` };
}

export async function predictImage(file, patientId, token) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("patient_id", patientId);

  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: authHeaders(token),
    body: formData,
  });

  if (!response.ok) {
    const message = await parseError(response);
    throw new Error(message);
  }

  return response.json();
}

export async function getHealth() {
  const response = await fetch(`${API_URL}/health`);

  if (!response.ok) {
    const message = await parseError(response);
    throw new Error(message);
  }

  return response.json();
}

export async function createPatient(mrn, firstName, lastName, token) {
  const response = await fetch(`${API_URL}/patients`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify({ mrn, first_name: firstName, last_name: lastName }),
  });
  if (!response.ok) {
    const message = await parseError(response);
    throw new Error(message);
  }
  return response.json();
}

export async function getPatients(token) {
  const response = await fetch(`${API_URL}/patients`, {
    headers: authHeaders(token),
  });
  if (!response.ok) {
    const message = await parseError(response);
    throw new Error(message);
  }
  return response.json();
}

export async function getHistory(token) {
  const response = await fetch(`${API_URL}/history`, {
    headers: authHeaders(token),
  });
  if (!response.ok) {
    const message = await parseError(response);
    throw new Error(message);
  }
  return response.json();
}

export async function getPrediction(predictionId, token) {
  const response = await fetch(`${API_URL}/predictions/${predictionId}`, {
    headers: authHeaders(token),
  });
  if (!response.ok) {
    const message = await parseError(response);
    throw new Error(message);
  }
  return response.json();
}
