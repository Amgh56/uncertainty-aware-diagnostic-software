import { API_URL } from "../../config";

export interface CalibrationJob {
  id: string;
  status: string;
  model_filename: string;
  config_filename: string | null;
  dataset_filename: string;
  alpha: number;
  result_json: string | null;
  error_message: string | null;
  created_at: string;
  completed_at: string | null;
}

export interface JobListResponse {
  jobs: CalibrationJob[];
}

async function parseError(response: Response): Promise<string> {
  try {
    const payload = await response.json();
    return payload.detail || payload.message || "Request failed";
  } catch {
    return "Request failed";
  }
}

function authHeaders(token: string): Record<string, string> {
  return { Authorization: `Bearer ${token}` };
}

export async function createCalibrationJob(
  modelFile: File,
  datasetFile: File,
  configFile: File | null,
  alpha: number,
  token: string,
): Promise<CalibrationJob> {
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

export async function listJobs(token: string): Promise<JobListResponse> {
  const response = await fetch(`${API_URL}/developer/jobs`, {
    headers: authHeaders(token),
  });

  if (!response.ok) {
    const message = await parseError(response);
    throw new Error(message);
  }

  return response.json();
}

export async function downloadJobResult(jobId: string, token: string): Promise<Blob> {
  const response = await fetch(`${API_URL}/developer/jobs/${jobId}/result`, {
    headers: authHeaders(token),
  });

  if (!response.ok) {
    const message = await parseError(response);
    throw new Error(message);
  }

  return response.blob();
}

export async function deleteJob(jobId: string, token: string): Promise<void> {
  const response = await fetch(`${API_URL}/developer/jobs/${jobId}`, {
    method: "DELETE",
    headers: authHeaders(token),
  });

  if (!response.ok) {
    const message = await parseError(response);
    throw new Error(message);
  }
}
