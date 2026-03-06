import { API_URL } from "./diagnosticApi";

function authHeaders(token) {
  return { Authorization: `Bearer ${token}` };
}

async function parseError(response) {
  try {
    const payload = await response.json();
    return payload.detail || payload.message || "Request failed";
  } catch {
    return "Request failed";
  }
}

export async function createCalibrationJob(modelFile, datasetFile, configFile, alpha, token) {
  const form = new FormData();
  form.append("model_file", modelFile);
  if (configFile) form.append("config_file", configFile);
  form.append("dataset_file", datasetFile);

  const response = await fetch(`${API_URL}/developer/jobs?alpha=${alpha}`, {
    method: "POST",
    headers: authHeaders(token),
    body: form,
  });

  if (!response.ok) {
    const message = await parseError(response);
    throw new Error(message);
  }

  return response.json();
}

export async function listJobs(token) {
  const response = await fetch(`${API_URL}/developer/jobs`, {
    headers: authHeaders(token),
  });

  if (!response.ok) {
    const message = await parseError(response);
    throw new Error(message);
  }

  return response.json();
}

export async function downloadJobResult(jobId, token) {
  const response = await fetch(`${API_URL}/developer/jobs/${jobId}/result`, {
    headers: authHeaders(token),
  });

  if (!response.ok) {
    const message = await parseError(response);
    throw new Error(message);
  }

  return response.blob();
}

export async function deleteJob(jobId, token) {
  const response = await fetch(`${API_URL}/developer/jobs/${jobId}`, {
    method: "DELETE",
    headers: authHeaders(token),
  });

  if (!response.ok) {
    const message = await parseError(response);
    throw new Error(message);
  }
}
