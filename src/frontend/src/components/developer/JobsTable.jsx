import { useEffect, useRef, useState } from "react";
import { API_URL } from "../../api/diagnosticApi";

const POLL_INTERVAL_MS = 5000;

const STATUS_BADGE = {
  queued:  { label: "Queued",  color: "#64748b", bg: "#f1f5f9" },
  running: { label: "Running", color: "#d97706", bg: "#fffbeb" },
  done:    { label: "Done",    color: "#059669", bg: "#f0fdf4" },
  failed:  { label: "Failed",  color: "#dc2626", bg: "#fef2f2" },
};

function StatusBadge({ status }) {
  const s = STATUS_BADGE[status] ?? { label: status, color: "#64748b", bg: "#f1f5f9" };
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 5,
      padding: "3px 10px", borderRadius: 20, fontSize: 12, fontWeight: 600,
      color: s.color, background: s.bg,
    }}>
      {status === "running" && (
        <span style={{
          width: 8, height: 8, borderRadius: "50%",
          background: s.color, display: "inline-block",
          animation: "dev-pulse 1.2s ease-in-out infinite",
        }} />
      )}
      {s.label}
    </span>
  );
}

/**
 * Props:
 *   token       — JWT
 *   onNewJob    — called when a terminal (done/failed) status is reached, useful to re-enable form
 */
export default function JobsTable({ token, onNewJob }) {
  const [jobs, setJobs] = useState([]);
  const [loadError, setLoadError] = useState(null);
  const pollRef = useRef(null);

  async function fetchJobs() {
    try {
      const res = await fetch(`${API_URL}/developer/jobs`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) return;
      const data = await res.json();
      setJobs(data.jobs ?? []);
      setLoadError(null);
    } catch {
      setLoadError("Could not reach the server.");
    }
  }

  useEffect(() => {
    fetchJobs();
    pollRef.current = setInterval(fetchJobs, POLL_INTERVAL_MS);
    return () => clearInterval(pollRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  // Notify parent when any running/queued job reaches terminal state
  useEffect(() => {
    const hasActive = jobs.some((j) => j.status === "queued" || j.status === "running");
    if (!hasActive && onNewJob) onNewJob();
  }, [jobs, onNewJob]);

  async function handleDownload(jobId) {
    const res = await fetch(`${API_URL}/developer/jobs/${jobId}/result`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!res.ok) {
      alert("Result not available yet.");
      return;
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `lamhat_${jobId.slice(0, 8)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  async function handleDelete(jobId) {
    if (!window.confirm("Delete this job and all uploaded files?")) return;
    await fetch(`${API_URL}/developer/jobs/${jobId}`, {
      method: "DELETE",
      headers: { Authorization: `Bearer ${token}` },
    });
    setJobs((prev) => prev.filter((j) => j.id !== jobId));
  }

  if (loadError) {
    return <p className="dev-jobs-error">{loadError}</p>;
  }

  if (jobs.length === 0) {
    return (
      <div className="dev-jobs-empty">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#cbd5e1" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
          <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
          <polyline points="14 2 14 8 20 8" />
        </svg>
        <p>No calibration jobs yet. Upload a model and dataset to get started.</p>
      </div>
    );
  }

  return (
    <div className="dev-jobs-table-wrapper">
      <table className="dev-jobs-table">
        <thead>
          <tr>
            <th>Job ID</th>
            <th>Model</th>
            <th>Dataset</th>
            <th>Alpha</th>
            <th>Status</th>
            <th>Created</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {jobs.map((job) => (
            <tr key={job.id}>
              <td className="dev-job-id" title={job.id}>{job.id.slice(0, 8)}&hellip;</td>
              <td className="dev-job-filename" title={job.model_filename}>{job.model_filename}</td>
              <td className="dev-job-filename" title={job.dataset_filename}>{job.dataset_filename}</td>
              <td>{job.alpha}</td>
              <td><StatusBadge status={job.status} /></td>
              <td>{new Date(job.created_at).toLocaleString()}</td>
              <td>
                <div style={{ display: "flex", gap: 6 }}>
                  {job.status === "done" && (
                    <button
                      className="dev-action-btn dev-action-btn--download"
                      onClick={() => handleDownload(job.id)}
                      title="Download lamhat.json"
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                        <polyline points="7 10 12 15 17 10" />
                        <line x1="12" y1="15" x2="12" y2="3" />
                      </svg>
                      Download
                    </button>
                  )}
                  {job.status === "failed" && job.error_message && (
                    <span className="dev-job-error-tooltip" title={job.error_message}>
                      Error
                    </span>
                  )}
                  <button
                    className="dev-action-btn dev-action-btn--delete"
                    onClick={() => handleDelete(job.id)}
                    title="Delete job"
                    disabled={job.status === "running"}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="3 6 5 6 21 6" />
                      <path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6" />
                      <path d="M10 11v6M14 11v6" />
                    </svg>
                  </button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
