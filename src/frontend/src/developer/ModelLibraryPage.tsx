import { useEffect, useState } from "react";
import { useAuth } from "../context/AuthContext";
import DeveloperLayout from "./DeveloperLayout";
import {
  listCommunityModels,
  listMyModels,
  getModelDetail,
  downloadModelArtifact,
  updateModelVisibility,
  updateModelDetails,
  toggleModelActive,
  type PublishedModelSummary,
  type PublishedModelDetail,
} from "./api/developerApi";

type Tab = "community" | "mine";

const VERDICT_STYLES: Record<string, { color: string; bg: string }> = {
  good: { color: "#059669", bg: "#f0fdf4" },
  review: { color: "#d97706", bg: "#fffbeb" },
};

function VerdictBadge({ verdict }: { verdict: string }) {
  const s = VERDICT_STYLES[verdict] ?? { color: "#64748b", bg: "#f1f5f9" };
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 4,
      padding: "2px 8px", borderRadius: 12, fontSize: 11, fontWeight: 600,
      color: s.color, background: s.bg,
    }}>
      <span style={{ width: 6, height: 6, borderRadius: "50%", background: s.color }} />
      {verdict.charAt(0).toUpperCase() + verdict.slice(1)}
    </span>
  );
}

function VisibilityBadge({ visibility }: { visibility: string }) {
  const labels: Record<string, string> = {
    private: "Private",
    clinician: "Clinician",
    community: "Community",
    clinician_and_community: "Clinician & Community",
  };
  return (
    <span style={{
      padding: "2px 8px", borderRadius: 12, fontSize: 11, fontWeight: 500,
      color: "#4338ca", background: "#eef2ff",
    }}>
      {labels[visibility] ?? visibility}
    </span>
  );
}

export default function ModelLibraryPage() {
  const { token } = useAuth();
  const [tab, setTab] = useState<Tab>("community");
  const [models, setModels] = useState<PublishedModelSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [modalityFilter, setModalityFilter] = useState("");
  const [selectedModel, setSelectedModel] = useState<PublishedModelDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [editing, setEditing] = useState(false);
  const [editDescription, setEditDescription] = useState("");
  const [editIntendedUse, setEditIntendedUse] = useState("");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!token) return;
    setLoading(true);
    const fetch = tab === "community"
      ? listCommunityModels(token, {
          search: search || undefined,
          modality: modalityFilter || undefined,
        })
      : listMyModels(token);

    fetch
      .then((res) => setModels(res.models))
      .catch(() => setModels([]))
      .finally(() => setLoading(false));
  }, [token, tab, search, modalityFilter]);

  async function handleViewDetail(modelId: string) {
    if (!token) return;
    setDetailLoading(true);
    setEditing(false);
    try {
      const detail = await getModelDetail(modelId, token);
      setSelectedModel(detail);
    } catch {
      alert("Failed to load model details.");
    } finally {
      setDetailLoading(false);
    }
  }

  function startEditing() {
    if (!selectedModel) return;
    setEditDescription(selectedModel.description);
    setEditIntendedUse(selectedModel.intended_use);
    setEditing(true);
  }

  async function handleSaveDetails() {
    if (!token || !selectedModel) return;
    setSaving(true);
    try {
      const result = await updateModelDetails(
        selectedModel.id,
        { description: editDescription, intended_use: editIntendedUse },
        token,
      );
      setSelectedModel({ ...selectedModel, description: result.description, intended_use: result.intended_use });
      setModels((prev) =>
        prev.map((m) => m.id === selectedModel.id ? { ...m, description: result.description } : m),
      );
      setEditing(false);
    } catch (err) {
      alert((err as Error).message);
    } finally {
      setSaving(false);
    }
  }

  async function handleDownload(modelId: string, modelName?: string) {
    if (!token) return;
    try {
      const { blob } = await downloadModelArtifact(modelId, token);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${(modelName ?? modelId.slice(0, 8)).replace(/\s+/g, "_")}.zip`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      alert("Download failed.");
    }
  }

  async function handleToggleActive(modelId: string, isActive: boolean) {
    if (!token) return;
    try {
      await toggleModelActive(modelId, isActive, token);
      setModels((prev) =>
        prev.map((m) => (m.id === modelId ? { ...m, is_active: isActive } : m))
      );
      if (selectedModel?.id === modelId) {
        setSelectedModel({ ...selectedModel, is_active: isActive });
      }
    } catch {
      alert("Failed to update.");
    }
  }

  async function handleVisibilityChange(modelId: string, visibility: string) {
    if (!token) return;
    const needsConsent = visibility !== "private";
    if (needsConsent) {
      const ok = window.confirm(
        "Changing visibility to a wider audience requires consent. " +
        "By confirming, you agree that your model will be shared as described in the release terms."
      );
      if (!ok) return;
    }
    try {
      await updateModelVisibility(modelId, visibility, needsConsent, token);
      setModels((prev) =>
        prev.map((m) => (m.id === modelId ? { ...m, visibility } : m))
      );
      if (selectedModel?.id === modelId) {
        setSelectedModel({ ...selectedModel, visibility });
      }
    } catch (err) {
      alert((err as Error).message);
    }
  }

  // Extract unique modalities for filter
  const modalities = [...new Set(models.map((m) => m.modality))].sort();

  return (
    <DeveloperLayout
      title="Calibrated Models"
      subtitle="Browse validated, calibrated models shared by the SafeDx community, or manage your own."
    >
      <div className="val-container">
        <div className="profile-content-card">
        {/* Tabs */}
        <div className="model-lib-tabs">
          <button
            className={`model-lib-tab${tab === "community" ? " model-lib-tab--active" : ""}`}
            onClick={() => setTab("community")}
          >
            Community Models
          </button>
          <button
            className={`model-lib-tab${tab === "mine" ? " model-lib-tab--active" : ""}`}
            onClick={() => setTab("mine")}
          >
            My Published Models
          </button>
        </div>

        {/* Filters (community tab) */}
        {tab === "community" && (
          <div className="model-lib-filters">
            <input
              type="text"
              className="model-lib-search"
              placeholder="Search by name or description..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
            <select
              className="model-lib-filter-select"
              value={modalityFilter}
              onChange={(e) => setModalityFilter(e.target.value)}
            >
              <option value="">All Modalities</option>
              {modalities.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div className="val-card val-center">
            <div className="val-spinner" />
            <p>Loading models...</p>
          </div>
        )}

        {/* Empty state */}
        {!loading && models.length === 0 && (
          <div className="val-card val-center" style={{ padding: "40px 20px" }}>
            <p style={{ color: "#64748b" }}>
              {tab === "community"
                ? "No community models available yet. Be the first to publish!"
                : "You haven't published any models yet. Validate a calibration and publish it."}
            </p>
          </div>
        )}

        {/* Model cards grid */}
        {!loading && models.length > 0 && (
          <div className="model-lib-grid">
            {models.map((model) => (
              <div key={model.id} className="model-card">
                <div className="model-card-header">
                  <h3 className="model-card-name">{model.name}</h3>
                  <span className="model-card-version">v{model.version}</span>
                </div>
                <p className="model-card-desc">{model.description}</p>
                <div className="model-card-meta">
                  <span>Modality: {model.modality}</span>
                  <span>Labels: {model.num_labels}</span>
                  <span>Alpha: {model.alpha}</span>
                  <span>Lamhat: {model.lamhat.toFixed(4)}</span>
                </div>
                <div className="model-card-footer">
                  <div className="model-card-badges">
                    <VerdictBadge verdict={model.validation_verdict} />
                    {tab === "mine" && <VisibilityBadge visibility={model.visibility} />}
                    {tab === "mine" && !model.is_active && (
                      <span style={{
                        padding: "2px 8px", borderRadius: 12, fontSize: 11,
                        fontWeight: 500, color: "#dc2626", background: "#fef2f2",
                      }}>
                        Inactive
                      </span>
                    )}
                  </div>
                  <div className="model-card-info">
                    {model.developer_name && (
                      <span className="model-card-author">By: {model.developer_name}</span>
                    )}
                    <span className="model-card-date">
                      {new Date(model.created_at).toLocaleDateString()}
                    </span>
                  </div>
                </div>
                <div className="model-card-actions">
                  <button
                    className="val-btn val-btn-outline"
                    onClick={() => handleViewDetail(model.id)}
                    disabled={detailLoading}
                  >
                    View Details
                  </button>
                  {tab === "community" && (
                    <button
                      className="val-btn val-btn-outline"
                      onClick={() => handleDownload(model.id, model.name)}
                    >
                      Download
                    </button>
                  )}
                  {tab === "mine" && (
                    <button
                      className="val-btn val-btn-outline"
                      onClick={() => handleToggleActive(model.id, !model.is_active)}
                    >
                      {model.is_active ? "Deactivate" : "Activate"}
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
        </div>

        {/* Detail Modal */}
        {selectedModel && (
          <div className="publish-dialog-overlay" onClick={() => setSelectedModel(null)}>
            <div className="publish-dialog" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 640 }}>
              <div className="publish-dialog-header">
                <h2 className="publish-dialog-title">{selectedModel.name}</h2>
                <button
                  type="button"
                  className="publish-dialog-close"
                  onClick={() => setSelectedModel(null)}
                  aria-label="Close"
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <line x1="18" y1="6" x2="6" y2="18" />
                    <line x1="6" y1="6" x2="18" y2="18" />
                  </svg>
                </button>
              </div>

              <div className="model-detail-body">
                <div className="model-detail-row">
                  <strong>Version:</strong> {selectedModel.version}
                </div>
                <div className="model-detail-row">
                  <strong>Modality:</strong> {selectedModel.modality}
                </div>
                <div className="model-detail-row">
                  <strong>Description:</strong>{" "}
                  {editing ? (
                    <textarea
                      value={editDescription}
                      onChange={(e) => setEditDescription(e.target.value)}
                      style={{ width: "100%", marginTop: 4, padding: 8, borderRadius: 6, border: "1px solid #e2e8f0", minHeight: 60, fontFamily: "inherit", fontSize: "inherit" }}
                    />
                  ) : (
                    selectedModel.description
                  )}
                </div>
                <div className="model-detail-row">
                  <strong>Intended Use:</strong>{" "}
                  {editing ? (
                    <textarea
                      value={editIntendedUse}
                      onChange={(e) => setEditIntendedUse(e.target.value)}
                      style={{ width: "100%", marginTop: 4, padding: 8, borderRadius: 6, border: "1px solid #e2e8f0", minHeight: 60, fontFamily: "inherit", fontSize: "inherit" }}
                    />
                  ) : (
                    selectedModel.intended_use
                  )}
                </div>
                <div className="model-detail-row">
                  <strong>Labels:</strong>{" "}
                  {(() => {
                    try {
                      return JSON.parse(selectedModel.labels_json).join(", ");
                    } catch {
                      return selectedModel.labels_json;
                    }
                  })()}
                </div>
                <div className="model-detail-row">
                  <strong>Artifact Type:</strong> {selectedModel.artifact_type}
                </div>
                <div className="model-detail-row">
                  <strong>Alpha:</strong> {selectedModel.alpha} | <strong>Lamhat:</strong> {selectedModel.lamhat.toFixed(4)}
                </div>
                <div className="model-detail-row">
                  <strong>Verdict:</strong> <VerdictBadge verdict={selectedModel.validation_verdict} />
                </div>
                {selectedModel.validation_metrics_json && (
                  <div className="model-detail-row">
                    <strong>Metrics:</strong>
                    <pre className="model-detail-metrics">
                      {JSON.stringify(JSON.parse(selectedModel.validation_metrics_json), null, 2)}
                    </pre>
                  </div>
                )}
                <div className="model-detail-row">
                  <strong>Creator:</strong> {selectedModel.developer_name}
                </div>
                <div className="model-detail-row">
                  <strong>Published:</strong> {new Date(selectedModel.created_at).toLocaleString()}
                </div>

                {/* Visibility management for own models */}
                {tab === "mine" && (
                  <div className="model-detail-row" style={{ marginTop: 16 }}>
                    <strong>Visibility:</strong>{" "}
                    <select
                      value={selectedModel.visibility}
                      onChange={(e) => handleVisibilityChange(selectedModel.id, e.target.value)}
                      style={{ marginLeft: 8, padding: "4px 8px", borderRadius: 6, border: "1px solid #e2e8f0" }}
                    >
                      <option value="private">Private</option>
                      <option value="clinician">Clinician</option>
                      <option value="community">Community</option>
                      <option value="clinician_and_community">Clinician & Community</option>
                    </select>
                  </div>
                )}
              </div>

              <div className="publish-dialog-actions">
                <button className="val-btn val-btn-outline" onClick={() => { setSelectedModel(null); setEditing(false); }}>
                  Close
                </button>
                {tab === "mine" && !editing && (
                  <button className="val-btn val-btn-outline" onClick={startEditing}>
                    Edit Details
                  </button>
                )}
                {tab === "mine" && editing && (
                  <>
                    <button className="val-btn val-btn-outline" onClick={() => setEditing(false)}>
                      Cancel
                    </button>
                    <button className="val-btn val-btn-primary" onClick={handleSaveDetails} disabled={saving}>
                      {saving ? "Saving..." : "Save"}
                    </button>
                  </>
                )}
                <button className="val-btn val-btn-primary" onClick={() => handleDownload(selectedModel.id, selectedModel.name)}>
                  Download Package (.zip)
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </DeveloperLayout>
  );
}
