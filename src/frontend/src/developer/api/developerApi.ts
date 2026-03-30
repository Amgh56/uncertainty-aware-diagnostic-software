import { API_URL } from "../../config";

export interface CalibrationJob {
  id: string;
  display_name: string | null;
  status: string;
  model_filename: string;
  config_filename: string | null;
  dataset_filename: string;
  alpha: number;
  result_json: string | null;
  error_message: string | null;
  validation_verdict: string | null;
  is_published: boolean;
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

/* ── Validation ─────────────────────────────────────────── */

export interface SweepPoint {
  alpha: number;
  lamhat: number;
  empirical_fnr: number;
  avg_set_size: number;
}

export interface ValidationData {
  sweep: SweepPoint[];
  job_alpha: number;
  job_lamhat: number;
  job_fnr: number;
  job_avg_set_size: number;
  n_samples: number;
  n_positive: number;
  label_names: string[];
  verdict: "good" | "review" | "unreliable";
  violations: number;
  monotonic_breaks: number;
}

export async function fetchValidationData(
  jobId: string,
  token: string,
): Promise<ValidationData> {
  const response = await fetch(`${API_URL}/developer/jobs/${jobId}/validation`, {
    headers: authHeaders(token),
  });

  if (!response.ok) {
    const message = await parseError(response);
    throw new Error(message);
  }

  return response.json();
}

export async function regenerateValidation(
  jobId: string,
  token: string,
): Promise<ValidationData> {
  const response = await fetch(`${API_URL}/developer/jobs/${jobId}/validation`, {
    method: "POST",
    headers: authHeaders(token),
  });

  if (!response.ok) {
    const message = await parseError(response);
    throw new Error(message);
  }

  return response.json();
}

export function getArtifactDownloadUrl(jobId: string, filename: string): string {
  return `${API_URL}/developer/jobs/${jobId}/validation/download/${filename}`;
}

/* ── Published Models ──────────────────────────────────────── */

export interface PublishedModelSummary {
  id: string;
  name: string;
  description: string;
  version: string;
  modality: string;
  intended_use: string;
  num_labels: number;
  alpha: number;
  lamhat: number;
  validation_verdict: string;
  visibility: string;
  is_active: boolean;
  developer_name: string | null;
  created_at: string;
}

export interface PublishedModelDetail extends PublishedModelSummary {
  calibration_job_id: string;
  developer_id: number;
  intended_use: string;
  artifact_type: string;
  labels_json: string;
  lamhat_result_json: string | null;
  validation_metrics_json: string | null;
  consent_given_at: string | null;
  updated_at: string;
}

export interface PublishModelPayload {
  calibration_job_id: string;
  name: string;
  description: string;
  version: string;
  modality: string;
  intended_use: string;
  labels: string[];
  visibility: string;
  consent_agreed: boolean;
}

export async function publishModel(
  data: PublishModelPayload,
  token: string,
): Promise<{ id: string }> {
  const response = await fetch(`${API_URL}/models/publish`, {
    method: "POST",
    headers: { ...authHeaders(token), "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
  return response.json();
}

export async function listMyModels(
  token: string,
): Promise<{ models: PublishedModelSummary[] }> {
  const response = await fetch(`${API_URL}/models/mine`, {
    headers: authHeaders(token),
  });
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
  return response.json();
}

export async function listCommunityModels(
  token: string,
  params?: { search?: string; modality?: string; verdict?: string; sort?: string },
): Promise<{ models: PublishedModelSummary[] }> {
  const qs = new URLSearchParams();
  if (params?.search) qs.set("search", params.search);
  if (params?.modality) qs.set("modality", params.modality);
  if (params?.verdict) qs.set("verdict", params.verdict);
  if (params?.sort) qs.set("sort", params.sort);
  const url = `${API_URL}/models/community${qs.toString() ? "?" + qs.toString() : ""}`;
  const response = await fetch(url, { headers: authHeaders(token) });
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
  return response.json();
}

export async function getModelDetail(
  modelId: string,
  token: string,
): Promise<PublishedModelDetail> {
  const response = await fetch(`${API_URL}/models/${modelId}`, {
    headers: authHeaders(token),
  });
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
  return response.json();
}

export async function updateModelVisibility(
  modelId: string,
  visibility: string,
  consentAgreed: boolean,
  token: string,
): Promise<void> {
  const response = await fetch(`${API_URL}/models/${modelId}/visibility`, {
    method: "PATCH",
    headers: { ...authHeaders(token), "Content-Type": "application/json" },
    body: JSON.stringify({ visibility, consent_agreed: consentAgreed }),
  });
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
}

export async function toggleModelActive(
  modelId: string,
  isActive: boolean,
  token: string,
): Promise<void> {
  const response = await fetch(`${API_URL}/models/${modelId}/active`, {
    method: "PATCH",
    headers: { ...authHeaders(token), "Content-Type": "application/json" },
    body: JSON.stringify({ is_active: isActive }),
  });
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
}

export async function updateModelDetails(
  modelId: string,
  details: { description?: string; intended_use?: string },
  token: string,
): Promise<{ id: string; description: string; intended_use: string }> {
  const response = await fetch(`${API_URL}/models/${modelId}/details`, {
    method: "PATCH",
    headers: { ...authHeaders(token), "Content-Type": "application/json" },
    body: JSON.stringify(details),
  });
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
  return response.json();
}

export async function downloadModelArtifact(
  modelId: string,
  token: string,
): Promise<{ blob: Blob; filename: string }> {
  const response = await fetch(`${API_URL}/models/${modelId}/download`, {
    headers: authHeaders(token),
  });
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
  const disposition = response.headers.get("Content-Disposition") ?? "";
  const match = disposition.match(/filename="([^"]+)"/);
  const filename = match ? match[1] : `model_${modelId.slice(0, 8)}.zip`;
  const blob = await response.blob();
  return { blob, filename };
}
