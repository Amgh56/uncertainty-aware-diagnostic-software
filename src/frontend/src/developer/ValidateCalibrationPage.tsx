import { useEffect, useState, useRef } from "react";
import { useAuth } from "../context/AuthContext";
import DeveloperLayout from "./DeveloperLayout";
import {
  CalibrationJob,
  ValidationData,
  fetchValidationData,
  listJobs,
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
} from "recharts";

const ALPHA_AXIS_TICKS = [0, 0.2, 0.4, 0.6, 0.8, 1];

export default function ValidateCalibrationPage() {
  const { token } = useAuth();
  const fnrChartRef = useRef<HTMLDivElement>(null);
  const sizeChartRef = useRef<HTMLDivElement>(null);

  const [completedJobs, setCompletedJobs] = useState<CalibrationJob[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string>("");
  const [validation, setValidation] = useState<ValidationData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const formatAlpha = (value: unknown) => {
    const numericValue = Number(value);
    return Number.isFinite(numericValue) ? numericValue.toFixed(1) : String(value);
  };

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
    fetchValidationData(selectedJobId, token)
      .then((data) => setValidation(data))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [selectedJobId, token]);

  function downloadChartAsPng(ref: React.RefObject<HTMLDivElement | null>, name: string) {
    const container = ref.current;
    if (!container) return;

    // Recharts renders as SVG internally — we convert it to PNG via canvas
    const allSvgs = container.querySelectorAll("svg");
    const chartSvg = allSvgs[allSvgs.length - 1];
    if (!chartSvg) return;

    const clone = chartSvg.cloneNode(true) as SVGElement;
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    const box = chartSvg.getBoundingClientRect();
    clone.setAttribute("width", String(box.width));
    clone.setAttribute("height", String(box.height));

    const background = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    background.setAttribute("width", String(box.width));
    background.setAttribute("height", String(box.height));
    background.setAttribute("fill", "#ffffff");
    clone.insertBefore(background, clone.firstChild);

    // Inline text styles so they render correctly outside the browser context
    const origTexts = chartSvg.querySelectorAll("text");
    const cloneTexts = clone.querySelectorAll("text");
    origTexts.forEach((orig, i) => {
      const cs = window.getComputedStyle(orig);
      cloneTexts[i]?.setAttribute(
        "style",
        `font-family:${cs.fontFamily};font-size:${cs.fontSize};fill:${cs.fill};`
      );
    });

    const serialized = new XMLSerializer().serializeToString(clone);
    const intermediateUrl = URL.createObjectURL(
      new Blob([serialized], { type: "image/svg+xml;charset=utf-8" })
    );
    const image = new Image();

    image.onload = () => {
      const scale = 2;
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(box.width * scale));
      canvas.height = Math.max(1, Math.round(box.height * scale));
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        URL.revokeObjectURL(intermediateUrl);
        return;
      }
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(intermediateUrl);

      const a = document.createElement("a");
      a.href = canvas.toDataURL("image/png");
      a.download = `${name}_${selectedJobId.slice(0, 8)}.png`;
      a.click();
    };

    image.onerror = () => URL.revokeObjectURL(intermediateUrl);
    image.src = intermediateUrl;
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
                        type="number"
                        domain={[0, 1]}
                        ticks={ALPHA_AXIS_TICKS}
                        label={{ value: "Alpha", position: "insideBottom", offset: -20, style: { fill: "#64748B" } }}
                        tick={{ fontSize: 12, fill: "#64748B" }}
                        tickFormatter={formatAlpha}
                      />
                      <YAxis
                        domain={[0, 1]}
                        ticks={ALPHA_AXIS_TICKS}
                        label={{ value: "FNR", angle: -90, position: "insideLeft", offset: 10, style: { fill: "#64748B" } }}
                        tick={{ fontSize: 12, fill: "#64748B" }}
                        tickFormatter={formatAlpha}
                      />
                      <Tooltip
                        contentStyle={{ borderRadius: 8, border: "1px solid #E2E8F0", fontSize: 13 }}
                        formatter={(value, name) => [
                          typeof value === "number" ? (value * 100).toFixed(1) + "%" : String(value),
                          name === "empirical_fnr" ? "Empirical FNR" : "Target (y=alpha)",
                        ]}
                        labelFormatter={(v) => `Alpha = ${formatAlpha(v)}`}
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
                        type="number"
                        domain={[0, 1]}
                        ticks={ALPHA_AXIS_TICKS}
                        label={{ value: "Alpha", position: "insideBottom", offset: -20, style: { fill: "#64748B" } }}
                        tick={{ fontSize: 12, fill: "#64748B" }}
                        tickFormatter={formatAlpha}
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
                        labelFormatter={(v) => `Alpha = ${formatAlpha(v)}`}
                      />
                      <Line
                        type="monotone"
                        dataKey="avg_set_size"
                        stroke="#0F766E"
                        strokeWidth={2.5}
                        dot={false}
                        activeDot={{ r: 5 }}
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
                <button className="val-btn val-btn-outline" onClick={() => downloadChartAsPng(fnrChartRef, "fnr_vs_alpha")}>
                  <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
                    <path d="M10 3v10m0 0l-3-3m3 3l3-3M4 15h12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  FNR Plot (.png)
                </button>
                <button className="val-btn val-btn-outline" onClick={() => downloadChartAsPng(sizeChartRef, "set_size_vs_alpha")}>
                  <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
                    <path d="M10 3v10m0 0l-3-3m3 3l3-3M4 15h12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  Set Size Plot (.png)
                </button>
              </div>
            </div>

            {/* ── Publish Section ── */}
            <div className={`val-card ${validation.verdict === "good" ? "val-publish-card" : "val-publish-disabled"}`}>
              <h3 className="val-card-title">
                <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                  <rect x="3" y="5" width="14" height="12" rx="2" stroke="currentColor" strokeWidth="1.5" fill="none" />
                  <path d="M7 5V3a3 3 0 016 0v2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                </svg>
                Publish This Model
              </h3>
              {selectedJob?.is_published || publishedSuccess ? (
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#059669" strokeWidth="2.5" strokeLinecap="round">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                  <span style={{ color: "#059669", fontWeight: 600 }}>
                    This model has been published successfully.
                  </span>
                </div>
              ) : validation.verdict === "good" ? (
                <>
                  <p style={{ margin: "0 0 12px" }}>
                    Your calibration passed with a <strong style={{ color: "#059669" }}>GOOD</strong> verdict.
                    {" "}You can now publish this as a reusable model package.
                  </p>
                  <button
                    className="val-btn val-btn-primary"
                    onClick={() => setShowPublish(true)}
                  >
                    Publish Model
                  </button>
                </>
              ) : validation.verdict === "review" ? (
                <p style={{ color: "#d97706", margin: 0 }}>
                  This calibration has a <strong>"review"</strong> verdict and cannot be published.
                  Please recalibrate with a larger or higher-quality dataset to achieve a "good" verdict.
                </p>
              ) : (
                <p style={{ color: "#dc2626", margin: 0 }}>
                  This calibration has an <strong>"unreliable"</strong> verdict and cannot be published.
                  Please recalibrate with a larger or higher-quality dataset.
                </p>
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
