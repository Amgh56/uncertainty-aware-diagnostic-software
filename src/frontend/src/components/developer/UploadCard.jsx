/**
 * Reusable upload card for a single file input.
 *
 * Props:
 *   title       — card heading (e.g. "Model Weights")
 *   accept      — file input accept string (e.g. ".pth")
 *   hint        — short description shown below title
 *   file        — currently selected File (or null)
 *   onChange    — callback(File)
 *   disabled    — grey-out while job is running
 */
export default function UploadCard({ title, accept, hint, file, onChange, disabled }) {
  function handleChange(e) {
    const selected = e.target.files?.[0];
    if (selected) onChange(selected);
  }

  function handleDrop(e) {
    e.preventDefault();
    if (disabled) return;
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) onChange(dropped);
  }

  function handleDragOver(e) {
    e.preventDefault();
  }

  const inputId = `upload-${title.replace(/\s+/g, "-").toLowerCase()}`;

  return (
    <div className="dev-upload-card" onDrop={handleDrop} onDragOver={handleDragOver}>
      <p className="dev-upload-card-title">{title}</p>
      <p className="dev-upload-card-hint">{hint}</p>

      <label
        htmlFor={inputId}
        className={`dev-upload-drop-zone${file ? " dev-upload-drop-zone--selected" : ""}${disabled ? " dev-upload-drop-zone--disabled" : ""}`}
      >
        {file ? (
          <>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#059669" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="20 6 9 17 4 12" />
            </svg>
            <span className="dev-upload-filename">{file.name}</span>
            <span className="dev-upload-filesize">({(file.size / (1024 * 1024)).toFixed(1)} MB)</span>
          </>
        ) : (
          <>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="16 16 12 12 8 16" />
              <line x1="12" y1="12" x2="12" y2="21" />
              <path d="M20.39 18.39A5 5 0 0018 9h-1.26A8 8 0 103 16.3" />
            </svg>
            <span className="dev-upload-drop-label">Click or drag & drop</span>
            <span className="dev-upload-accept">{accept} file</span>
          </>
        )}
      </label>

      <input
        id={inputId}
        type="file"
        accept={accept}
        onChange={handleChange}
        disabled={disabled}
        style={{ display: "none" }}
      />
    </div>
  );
}
