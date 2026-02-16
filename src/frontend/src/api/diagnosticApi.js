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

export async function predictImage(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
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
