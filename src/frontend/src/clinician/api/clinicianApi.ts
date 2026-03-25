import { API_URL } from "../../config";

export interface Patient {
  id: number;
  mrn: string;
  first_name: string;
  last_name: string;
}

export interface Finding {
  finding: string;
  probability: number;
  uncertainty: string;
  in_prediction_set: boolean;
}

export interface ModelInfo {
  id: string;
  name: string;
  version: string;
  modality: string;
  num_labels: number;
  validation_verdict: string;
}

export interface ClinicianModel {
  id: string;
  name: string;
  description: string;
  version: string;
  modality: string;
  num_labels: number;
  alpha: number;
  lamhat: number;
  validation_verdict: string;
  visibility: string;
  is_active: boolean;
  developer_name: string | null;
  created_at: string;
}

export interface PredictionResponse {
  id: number;
  patient: Patient;
  image_path: string;
  findings: Finding[];
  top_finding: string;
  top_probability: number;
  prediction_set_size: number;
  coverage: number;
  alpha: number;
  lamhat: number;
  model_info: ModelInfo | null;
  created_at: string;
}

export interface PatientWithStats extends Patient {
  prediction_count: number;
  last_prediction_at: string | null;
  last_prediction_id: number | null;
  last_top_finding: string | null;
}

export interface PatientListResponse {
  patients: PatientWithStats[];
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

export async function predictImage(
  file: File,
  patientId: number,
  token: string,
  modelId?: string,
): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("patient_id", String(patientId));
  if (modelId) {
    formData.append("model_id", modelId);
  }

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

export async function listClinicianModels(token: string): Promise<{ models: ClinicianModel[] }> {
  const response = await fetch(`${API_URL}/models/clinician`, {
    headers: authHeaders(token),
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

export async function createPatient(mrn: string, firstName: string, lastName: string, token: string): Promise<Patient> {
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

export async function getPatients(token: string): Promise<PatientListResponse> {
  const response = await fetch(`${API_URL}/patients`, {
    headers: authHeaders(token),
  });
  if (!response.ok) {
    const message = await parseError(response);
    throw new Error(message);
  }
  return response.json();
}

export async function getHistory(token: string): Promise<PredictionResponse[]> {
  const response = await fetch(`${API_URL}/history`, {
    headers: authHeaders(token),
  });
  if (!response.ok) {
    const message = await parseError(response);
    throw new Error(message);
  }
  return response.json();
}

export async function getPrediction(predictionId: string, token: string): Promise<PredictionResponse> {
  const response = await fetch(`${API_URL}/predictions/${predictionId}`, {
    headers: authHeaders(token),
  });
  if (!response.ok) {
    const message = await parseError(response);
    throw new Error(message);
  }
  return response.json();
}

export async function regeneratePrediction(predictionId: number, alpha: number, token: string): Promise<PredictionResponse> {
  const response = await fetch(`${API_URL}/predictions/${predictionId}/regenerate`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify({ alpha }),
  });
  if (!response.ok) {
    const message = await parseError(response);
    throw new Error(message);
  }
  return response.json();
}
