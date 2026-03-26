import { useEffect, useState } from "react";
import { useAuth } from "../context/AuthContext";
import ClinicianLayout from "./ClinicianLayout";
import { listClinicianModels, type ClinicianModel } from "./api/clinicianApi";

const VERDICT_STYLES: Record<string, { color: string; bg: string }> = {
  good: { color: "#059669", bg: "#f0fdf4" },
  review: { color: "#d97706", bg: "#fffbeb" },
};

function VerdictBadge({ verdict }: { verdict: string }) {
  const s = VERDICT_STYLES[verdict] ?? { color: "#64748b", bg: "#f1f5f9" };
  const verdictLabel =
    verdict === "good"
      ? "Good"
      : verdict === "review"
        ? "Review"
        : verdict === "unreliable"
          ? "Calibration May Be Unreliable"
          : verdict.charAt(0).toUpperCase() + verdict.slice(1);

  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 4,
      padding: "2px 8px", borderRadius: 12, fontSize: 11, fontWeight: 600,
      color: s.color, background: s.bg,
    }}>
      <span style={{ width: 6, height: 6, borderRadius: "50%", background: s.color }} />
      {verdictLabel}
    </span>
  );
}

export default function CalibratedModelsPage() {
  const { token } = useAuth();
  const [models, setModels] = useState<ClinicianModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [modalityFilter, setModalityFilter] = useState("");
  const [selectedModel, setSelectedModel] = useState<ClinicianModel | null>(null);

  useEffect(() => {
    if (!token) return;
    setLoading(true);
    listClinicianModels(token)
      .then((res) => setModels(res.models))
      .catch(() => setModels([]))
      .finally(() => setLoading(false));
  }, [token]);

  const filtered = models.filter((m) => {
    const matchesSearch =
      !search ||
      m.name.toLowerCase().includes(search.toLowerCase()) ||
      m.description.toLowerCase().includes(search.toLowerCase()) ||
      (m.developer_name ?? "").toLowerCase().includes(search.toLowerCase());
    const matchesModality = !modalityFilter || m.modality === modalityFilter;
    return matchesSearch && matchesModality;
  });

  const modalities = [...new Set(models.map((m) => m.modality))].sort();

  return (
    <ClinicianLayout
      title={"Calibrated Models"}
      subtitle={"Browse all calibrated models available for diagnostic use."}
    >
      <div className="val-container">
        {/* Filters */}
        <div className="model-lib-filters">
          <input
            type="text"
            className="model-lib-search"
            placeholder={"Search by name, description, or developer..."}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <select
            className="model-lib-filter-select"
            value={modalityFilter}
            onChange={(e) => setModalityFilter(e.target.value)}
          >
            <option value="">{"All Modalities"}</option>
            {modalities.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>

        {/* Loading */}
        {loading && (
          <div className="val-card val-center">
            <div className="val-spinner" />
            <p>{"Loading models..."}</p>
          </div>
        )}

        {/* Empty state */}
        {!loading && filtered.length === 0 && (
          <div className="val-card val-center" style={{ padding: "40px 20px" }}>
            <p style={{ color: "#64748b" }}>
              {search || modalityFilter
                ? "No models match your search criteria."
                : "No calibrated models are available yet."}
            </p>
          </div>
        )}

        {/* Model cards grid */}
        {!loading && filtered.length > 0 && (
          <div className="model-lib-grid">
            {filtered.map((model) => (
              <div key={model.id} className="model-card">
                <div className="model-card-header">
                  <h3 className="model-card-name">{model.name}</h3>
                  <span className="model-card-version">v{model.version}</span>
                </div>
                <p className="model-card-desc">{model.description}</p>
                <div className="model-card-meta">
                  <span>{"Modality:"} {model.modality}</span>
                  <span>{"Labels:"} {model.num_labels}</span>
                  <span>{"Alpha:"} {model.alpha}</span>
                  <span>{"Lamhat:"} {model.lamhat.toFixed(4)}</span>
                </div>
                <div className="model-card-footer">
                  <div className="model-card-badges">
                    <VerdictBadge verdict={model.validation_verdict} />
                    {!model.is_active && (
                      <span style={{
                        padding: "2px 8px", borderRadius: 12, fontSize: 11,
                        fontWeight: 500, color: "#dc2626", background: "#fef2f2",
                      }}>
                        {"Inactive"}
                      </span>
                    )}
                  </div>
                  <div className="model-card-info">
                    {model.developer_name && (
                      <span className="model-card-author">{"By:"} {model.developer_name}</span>
                    )}
                    <span className="model-card-date">
                      {new Date(model.created_at).toLocaleDateString()}
                    </span>
                  </div>
                </div>
                <div className="model-card-actions">
                  <button
                    className="val-btn val-btn-outline"
                    onClick={() => setSelectedModel(model)}
                  >
                    {"View Details"}
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}

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
                  aria-label={"Close"}
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <line x1="18" y1="6" x2="6" y2="18" />
                    <line x1="6" y1="6" x2="18" y2="18" />
                  </svg>
                </button>
              </div>

              <div className="model-detail-body">
                <div className="model-detail-row">
                  <strong>{"Version:"}</strong> {selectedModel.version}
                </div>
                <div className="model-detail-row">
                  <strong>{"Modality:"}</strong> {selectedModel.modality}
                </div>
                <div className="model-detail-row">
                  <strong>{"Description:"}</strong> {selectedModel.description}
                </div>
                <div className="model-detail-row" style={{ display: "flex", alignItems: "flex-start", gap: "0.5rem", flexWrap: "wrap" }}>
                  <strong style={{ whiteSpace: "nowrap" }}>{"Labels:"}</strong>
                  <div className="publish-labels-list" style={{ margin: 0 }}>
                    {(selectedModel.labels && selectedModel.labels.length > 0
                      ? selectedModel.labels
                      : []
                    ).map((l, i) => (
                      <span key={i} className="publish-label-chip">
                        {l}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="model-detail-row">
                  <strong>{"Alpha:"}</strong> {selectedModel.alpha} | <strong>{"Lamhat:"}</strong> {selectedModel.lamhat.toFixed(4)}
                </div>
                <div className="model-detail-row">
                  <strong>{"Verdict:"}</strong> <VerdictBadge verdict={selectedModel.validation_verdict} />
                </div>
                {selectedModel.developer_name && (
                  <div className="model-detail-row">
                    <strong>{"Developer:"}</strong> {selectedModel.developer_name}
                  </div>
                )}
                <div className="model-detail-row">
                  <strong>{"Published:"}</strong> {new Date(selectedModel.created_at).toLocaleString()}
                </div>
              </div>

              <div className="publish-dialog-actions">
                <button className="val-btn val-btn-outline" onClick={() => setSelectedModel(null)}>
                  {"Close"}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </ClinicianLayout>
  );
}
