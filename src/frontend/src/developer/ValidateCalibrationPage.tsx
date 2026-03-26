import { useEffect, useState, useRef } from "react";
import { useAuth } from "../context/AuthContext";
import DeveloperLayout from "./DeveloperLayout";
import {
  CalibrationJob,
  ValidationData,
  fetchValidationData,
  listJobs,
  regenerateValidation,
} from "./api/developerApi";
import PublishModelDialog from "./PublishModelDialog";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceDot,
} from "recharts";

export default function ValidateCalibrationPage() {
  const { token } = useAuth();
  const fnrChartRef = useRef<HTMLDivElement>(null);
  const sizeChartRef = useRef<HTMLDivElement>(null);

  const [completedJobs, setCompletedJobs] = useState<CalibrationJob[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string>("");
  const [validation, setValidation] = useState<ValidationData | null>(null);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [needsRegenerate, setNeedsRegenerate] = useState(false);

  // Fetch completed jobs on mount
  useEffect(() => {
    if (!token) return;
    listJobs(token).then((res) => {
      const done = res.jobs.filter((j) => j.status === "done");
      setCompletedJobs(done);
    });
  }, [token]);

  // Load validation data when job selected
  useEffect(() => {
    if (!selectedJobId || !token) {
      setValidation(null);
      return;
    }
    setLoading(true);
    setError(null);
    setNeedsRegenerate(false);
    fetchValidationData(selectedJobId, token)
      .then((data) => {
        setValidation(data);
      })
      .catch((err) => {
        if (err.message.includes("not found") || err.message.includes("regenerate")) {
          setNeedsRegenerate(true);
        } else {
          setError(err.message);
        }
      })
      .finally(() => setLoading(false));
  }, [selectedJobId, token]);

  async function handleRegenerate() {
    if (!selectedJobId || !token) return;
    setGenerating(true);
    setError(null);
    try {
      const data = await regenerateValidation(selectedJobId, token);
      setValidation(data);
      setNeedsRegenerate(false);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setGenerating(false);
    }
  }

  function downloadChart(ref: React.RefObject<HTMLDivElement | null>, name: string) {
    const container = ref.current;
    if (!container) return;

    // Recharts nests SVGs — grab the last (deepest) one which has the actual chart
    const allSvgs = container.querySelectorAll("svg");
    const svg = allSvgs[allSvgs.length - 1];
    if (!svg) return;

    const clone = svg.cloneNode(true) as SVGElement;
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    // Copy computed dimensions
    const box = svg.getBoundingClientRect();
    clone.setAttribute("width", String(box.width));
    clone.setAttribute("height", String(box.height));

    // Embed white background
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("width", String(box.width));
    rect.setAttribute("height", String(box.height));
    rect.setAttribute("fill", "#ffffff");
    clone.insertBefore(rect, clone.firstChild);

    // Inline computed styles so the SVG renders correctly standalone
    const origTexts = svg.querySelectorAll("text");
    const cloneTexts = clone.querySelectorAll("text");
    origTexts.forEach((orig, i) => {
      const cs = window.getComputedStyle(orig);
      cloneTexts[i]?.setAttribute(
        "style",
        `font-family:${cs.fontFamily};font-size:${cs.fontSize};fill:${cs.fill};`
      );
    });

    const serialized = new XMLSerializer().serializeToString(clone);
    const blob = new Blob([serialized], { type: "image/svg+xml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${name}_${selectedJobId.slice(0, 8)}.svg`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function downloadValidationJSON() {
    if (!validation) return;
    const blob = new Blob([JSON.stringify(validation, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `validation_${selectedJobId.slice(0, 8)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const [showPublish, setShowPublish] = useState(false);
  const [publishedSuccess, setPublishedSuccess] = useState(false);

  const selectedJob = completedJobs.find((j) => j.id === selectedJobId);

  const verdictConfig = {
    good: { label: "Calibration Looks Good", cls: "val-verdict--good" },
    review: { label: "Calibration Needs Review", cls: "val-verdict--review" },
    unreliable: { label: "Calibration May Be Unreliable", cls: "val-verdict--unreliable" },
  };

  // Prepare chart data with target line
  const chartData = validation?.sweep.map((s) => ({
    ...s,
    target_fnr: s.alpha,
  }));

  return (
    <DeveloperLayout
      title="Validate Your Calibration"
      subtitle="Review validation plots and summary metrics to understand whether your calibration behaves correctly."
    >
      <div className="val-container">
        {/* ── Job Selection Card ── */}
        <div className="val-card">
          <h3 className="val-card-title">
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <rect x="2" y="3" width="16" height="14" rx="2" stroke="currentColor" strokeWidth="1.5" fill="none" />
              <path d="M6 7h8M6 10h5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
            Select a Completed Calibration Job
          </h3>
          {completedJobs.length === 0 ? (
            <p className="val-empty">No completed calibration jobs found. Complete a calibration first.</p>
          ) : (
            <select
              className="val-select"
              value={selectedJobId}
              onChange={(e) => setSelectedJobId(e.target.value)}
            >
              <option value="">-- Select a job --</option>
              {completedJobs.map((job) => (
                <option key={job.id} value={job.id}>
                  {job.display_name ?? job.id.slice(0, 8)} — alpha {job.alpha}
                  {job.completed_at ? ` — ${new Date(job.completed_at).toLocaleDateString()}` : ""}
                </option>
              ))}
            </select>
          )}
        </div>

        {/* ── Loading / Error / Regenerate ── */}
        {loading && (
          <div className="val-card val-center">
            <div className="val-spinner" />
            <p>Loading validation data...</p>
          </div>
        )}

        {error && (
          <div className="val-card val-error-card">
            <p>{error}</p>
          </div>
        )}

        {needsRegenerate && !loading && (
          <div className="val-card val-center">
            <p style={{ marginBottom: 16 }}>
              Validation artifacts are not available for this job. Generate them now to view plots and metrics.
            </p>
            <button
              className="val-btn val-btn-primary"
              onClick={handleRegenerate}
              disabled={generating}
            >
              {generating ? "Generating..." : "Generate Validation Plots"}
            </button>
            {generating && <p className="val-hint">This may take a few minutes — re-running inference on your dataset.</p>}
          </div>
        )}

        {/* ── Results ── */}
        {validation && selectedJob && (
          <>
            {/* ── Summary Metrics ── */}
            <div className="val-metrics-row">
              <div className="val-metric-card">
                <span className="val-metric-label">Selected Alpha</span>
                <span className="val-metric-value">{validation.job_alpha}</span>
              </div>
              <div className="val-metric-card">
                <span className="val-metric-label">Calibrated Threshold</span>
                <span className="val-metric-value">{validation.job_lamhat.toFixed(4)}</span>
              </div>
              <div className="val-metric-card">
                <span className="val-metric-label">Empirical FNR</span>
                <span className="val-metric-value">{(validation.job_fnr * 100).toFixed(1)}%</span>
              </div>
              <div className="val-metric-card">
                <span className="val-metric-label">Avg Set Size</span>
                <span className="val-metric-value">{validation.job_avg_set_size.toFixed(2)}</span>
              </div>
              <div className="val-metric-card">
                <span className="val-metric-label">Samples</span>
                <span className="val-metric-value">{validation.n_samples.toLocaleString()}</span>
              </div>
              <div className={`val-metric-card ${verdictConfig[validation.verdict].cls}`}>
                <span className="val-metric-label">Validation Status</span>
                <span className="val-metric-value val-verdict-text">
                  {verdictConfig[validation.verdict].label}
                </span>
              </div>
            </div>

            {/* ── Plots ── */}
            <div className="val-plots-row">
              {/* Plot 1: FNR vs Alpha */}
              <div className="val-card val-plot-card">
                <h3 className="val-card-title">Empirical FNR vs Alpha</h3>
                <p className="val-card-caption">
                  Compares the observed false-negative rate against the target (y = alpha) across different alpha values.
                </p>
                <div className="val-chart-wrapper" ref={fnrChartRef}>
                  <ResponsiveContainer width="100%" height={340}>
                    <LineChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 36 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                      <XAxis
                        dataKey="alpha"
                        label={{ value: "Alpha", position: "insideBottom", offset: -20, style: { fill: "#64748B" } }}
                        tick={{ fontSize: 12, fill: "#64748B" }}
                      />
                      <YAxis
                        label={{ value: "FNR", angle: -90, position: "insideLeft", offset: 10, style: { fill: "#64748B" } }}
                        tick={{ fontSize: 12, fill: "#64748B" }}
                      />
                      <Tooltip
                        contentStyle={{ borderRadius: 8, border: "1px solid #E2E8F0", fontSize: 13 }}
                        formatter={(value, name) => [
                          typeof value === "number" ? (value * 100).toFixed(1) + "%" : String(value),
                          name === "empirical_fnr" ? "Empirical FNR" : "Target (y=alpha)",
                        ]}
                        labelFormatter={(v) => `Alpha = ${v}`}
                      />
                  
                      <Line
                        type="monotone"
                        dataKey="target_fnr"
                        stroke="#94A3B8"
                        strokeWidth={2}
                        strokeDasharray="6 4"
                        dot={false}
                      />
                      <Line
                        type="monotone"
                        dataKey="empirical_fnr"
                        stroke="#1F6FEB"
                        strokeWidth={2.5}
                        dot={false}
                        activeDot={{ r: 5 }}
                      />
                      <ReferenceDot
                        x={validation.job_alpha}
                        y={validation.job_fnr}
                        r={6}
                        fill="#1F6FEB"
                        stroke="#fff"
                        strokeWidth={2}
                      />
                      <ReferenceLine
                        x={validation.job_alpha}
                        stroke="#1F6FEB"
                        strokeDasharray="3 3"
                        strokeWidth={1}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Plot 2: Avg Set Size vs Alpha */}
              <div className="val-card val-plot-card">
                <h3 className="val-card-title">Average Prediction-Set Size vs Alpha</h3>
                <p className="val-card-caption">
                  Shows how the average number of predicted labels changes as the target miscoverage rate increases.
                </p>
                <div className="val-chart-wrapper" ref={sizeChartRef}>
                  <ResponsiveContainer width="100%" height={340}>
                    <LineChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 36 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                      <XAxis
                        dataKey="alpha"
                        label={{ value: "Alpha", position: "insideBottom", offset: -20, style: { fill: "#64748B" } }}
                        tick={{ fontSize: 12, fill: "#64748B" }}
                      />
                      <YAxis
                        label={{ value: "Avg Set Size", angle: -90, position: "insideLeft", offset: 10, style: { fill: "#64748B" } }}
                        tick={{ fontSize: 12, fill: "#64748B" }}
                      />
                      <Tooltip
                        contentStyle={{ borderRadius: 8, border: "1px solid #E2E8F0", fontSize: 13 }}
                        formatter={(value) => [
                          typeof value === "number" ? value.toFixed(2) : String(value),
                          "Avg Set Size",
                        ]}
                        labelFormatter={(v) => `Alpha = ${v}`}
                      />
                      <Line
                        type="monotone"
                        dataKey="avg_set_size"
                        stroke="#0F766E"
                        strokeWidth={2.5}
                        dot={false}
                        activeDot={{ r: 5 }}
                      />
                      <ReferenceDot
                        x={validation.job_alpha}
                        y={validation.job_avg_set_size}
                        r={6}
                        fill="#0F766E"
                        stroke="#fff"
                        strokeWidth={2}
                      />
                      <ReferenceLine
                        x={validation.job_alpha}
                        stroke="#0F766E"
                        strokeDasharray="3 3"
                        strokeWidth={1}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            {/* ── Interpretation Section ── */}
            <div className="val-card">
              <h3 className="val-card-title">
                <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                  <circle cx="10" cy="10" r="8" stroke="currentColor" strokeWidth="1.5" fill="none" />
                  <path d="M10 9v4M10 7h.01" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                </svg>
                Interpreting Your Results
              </h3>

              <div className="val-interp-grid">
                <div className="val-interp-block">
                  <h4>Empirical FNR vs Alpha</h4>
                  <ul>
                    <li>
                      <strong>What it shows:</strong> The blue curve represents the observed false-negative rate at each alpha level.
                      The dashed line represents the ideal target where FNR equals alpha.
                    </li>
                    <li>
                      <strong>Good result:</strong> The blue curve stays close to or below the dashed target line.
                      This means your model's calibration is controlling the false-negative rate as intended.
                    </li>
                    <li>
                      <strong>Warning sign:</strong> If the blue curve is consistently above the dashed line,
                      the calibration is not meeting the target — the model misses more true positives than expected.
                    </li>
                    <li>
                      <strong>Small deviations</strong> are normal and expected, especially at extreme alpha values.
                      Persistent, large gaps indicate a calibration problem.
                    </li>
                  </ul>
                </div>

                <div className="val-interp-block">
                  <h4>Average Prediction-Set Size vs Alpha</h4>
                  <ul>
                    <li>
                      <strong>What it shows:</strong> How many labels the model includes in its prediction set on average,
                      as the miscoverage tolerance increases.
                    </li>
                    <li>
                      <strong>Good result:</strong> A smooth, generally decreasing curve. As alpha increases (more tolerance for error),
                      the model can afford to include fewer labels.
                    </li>
                    <li>
                      <strong>Warning sign:</strong> Erratic jumps, flat regions, or an increasing trend
                      may indicate instability in the calibration or issues with the model or dataset.
                    </li>
                    <li>
                      <strong>Tradeoff:</strong> A lower alpha means higher safety (fewer missed diagnoses)
                      but larger prediction sets. This plot helps you understand that tradeoff for your model.
                    </li>
                  </ul>
                </div>
              </div>
            </div>

            {/* ── Final Verdict ── */}
            <div className={`val-card val-verdict-card ${verdictConfig[validation.verdict].cls}`}>
              <h3 className="val-card-title">Validation Verdict</h3>
              <div className="val-verdict-body">
                <span className="val-verdict-badge">
                  {verdictConfig[validation.verdict].label}
                </span>
                <p className="val-verdict-explain">
                  {validation.verdict === "good" && (
                    <>
                      Your calibration is performing well. The empirical false-negative rate stays close to the target
                      across alpha values, and the prediction-set size curve behaves as expected. You can confidently use
                      this calibrated threshold (lambda = {validation.job_lamhat.toFixed(4)}) for inference at alpha = {validation.job_alpha}. Or you can change the alpha to a value that is more sutiable after looking at the graphs.
                    </>
                  )}
                  {validation.verdict === "review" && (
                    <>
                      Your calibration shows some deviations from the expected behavior. There are {validation.violations} alpha
                      point{validation.violations !== 1 ? "s" : ""} where the FNR exceeds the target by more than 5%.
                      Consider reviewing your dataset quality, increasing the calibration dataset size, or adjusting your alpha.
                    </>
                  )}
                  {validation.verdict === "unreliable" && (
                    <>
                      Your calibration may not be reliable. The false-negative rate significantly exceeds the target at multiple
                      alpha levels ({validation.violations} violations detected), and/or the prediction-set size curve is unstable
                      ({validation.monotonic_breaks} non-monotonic breaks). We recommend recalibrating with a larger or
                      higher-quality dataset, or investigating your model's predictions.
                    </>
                  )}
                </p>
              </div>
            </div>

            {/* ── Download Section ── */}
            <div className="val-card">
              <h3 className="val-card-title">
                <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                  <path d="M10 3v10m0 0l-3-3m3 3l3-3M4 15h12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                Downloads
              </h3>
              <p className="val-card-caption" style={{ marginBottom: 16 }}>
                Export your validation results and plots.
              </p>
              <div className="val-download-row">
                <button className="val-btn val-btn-outline" onClick={downloadValidationJSON}>
                  <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
                    <path d="M10 3v10m0 0l-3-3m3 3l3-3M4 15h12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  Validation Report (.json)
                </button>
                <button className="val-btn val-btn-outline" onClick={() => downloadChart(fnrChartRef, "fnr_vs_alpha")}>
                  <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
                    <path d="M10 3v10m0 0l-3-3m3 3l3-3M4 15h12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  FNR Plot (.svg)
                </button>
                <button className="val-btn val-btn-outline" onClick={() => downloadChart(sizeChartRef, "set_size_vs_alpha")}>
                  <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
                    <path d="M10 3v10m0 0l-3-3m3 3l3-3M4 15h12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  Set Size Plot (.svg)
                </button>
              </div>
            </div>

            {/* ── Publish Section ── */}
            <div className={`val-card ${validation.verdict === "unreliable" ? "val-publish-disabled" : "val-publish-card"}`}>
              <h3 className="val-card-title">
                <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                  <rect x="3" y="5" width="14" height="12" rx="2" stroke="currentColor" strokeWidth="1.5" fill="none" />
                  <path d="M7 5V3a3 3 0 016 0v2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                </svg>
                Publish This Model
              </h3>
              {validation.verdict === "unreliable" ? (
                <p style={{ color: "#dc2626", margin: 0 }}>
                  This calibration has an "unreliable" verdict and cannot be published.
                  Please recalibrate with a larger or higher-quality dataset.
                </p>
              ) : selectedJob?.is_published || publishedSuccess ? (
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#059669" strokeWidth="2.5" strokeLinecap="round">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                  <span style={{ color: "#059669", fontWeight: 600 }}>
                    This model has been published successfully.
                  </span>
                </div>
              ) : (
                <>
                  <p style={{ margin: "0 0 12px" }}>
                    Your calibration passed validation with verdict:{" "}
                    <strong style={{ color: validation.verdict === "good" ? "#059669" : "#d97706" }}>
                      {validation.verdict.toUpperCase()}
                    </strong>.
                    {validation.verdict === "review" && " Publishing is allowed but proceed with caution."}
                    {" "}You can now publish this as a reusable model package.
                  </p>
                  <button
                    className="val-btn val-btn-primary"
                    onClick={() => setShowPublish(true)}
                  >
                    Publish Model
                  </button>
                </>
              )}
            </div>

            {/* Publish Dialog */}
            {showPublish && validation && (
              <PublishModelDialog
                jobId={selectedJobId}
                validation={validation}
                onPublished={() => {
                  setShowPublish(false);
                  setPublishedSuccess(true);
                  // Update the job in the list to reflect published status
                  setCompletedJobs((prev) =>
                    prev.map((j) =>
                      j.id === selectedJobId ? { ...j, is_published: true } : j
                    )
                  );
                }}
                onCancel={() => setShowPublish(false)}
              />
            )}
          </>
        )}
      </div>
    </DeveloperLayout>
  );
}
