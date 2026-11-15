import { useEffect, useRef, useState } from "react";

const API_ROOT = import.meta.env.VITE_API_BASE_URL || "";
const DISCLAIMER =
  "Educational research prototype. Not a medical device. Works only on dermatoscopic lesion images; phone photos or non-skin images give meaningless results.";
const MAX_FILE_BYTES = 5 * 1024 * 1024;
const ACCEPTED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
const CLASS_NAMES = {
  akiec: "Actinic keratoses",
  bcc: "Basal cell carcinoma",
  bkl: "Benign keratosis-like lesions",
  df: "Dermatofibroma",
  mel: "Melanoma",
  nv: "Melanocytic nevi",
  vasc: "Vascular lesions",
};
const CLASS_ORDER = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"];

function apiUrl(path) {
  return `${API_ROOT}${path}`;
}

function asPercent(value, digits = 1) {
  return new Intl.NumberFormat(undefined, {
    style: "percent",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

function asDecimal(value, digits = 3) {
  return Number(value).toFixed(digits);
}

async function responseError(response) {
  let body = {};
  try {
    body = await response.json();
  } catch {
    // Use the status based message when the server did not return JSON.
  }
  const detail = typeof body.detail === "string" ? body.detail : "";
  if (response.status === 413) return "That file is too large. Choose an image under 5 MiB.";
  if (response.status === 415) return "Unsupported file type. Choose a JPEG, PNG, or WebP image.";
  if (response.status === 422) return detail || "The image could not be read. Try another file.";
  if (response.status === 503) return "The model is unavailable. Check that the backend loaded its Stage B checkpoint.";
  if (response.status >= 500) return detail || "The backend could not complete this request.";
  return detail || `The request failed (${response.status}).`;
}

function App() {
  const [activeTab, setActiveTab] = useState("analyse");
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [analyseError, setAnalyseError] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalysing, setIsAnalysing] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [metricsError, setMetricsError] = useState("");
  const [metricsLoading, setMetricsLoading] = useState(false);
  const fileInput = useRef(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return undefined;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  async function loadMetrics(force = false) {
    if (metrics && !force) return;
    setMetricsLoading(true);
    setMetricsError("");
    try {
      const response = await fetch(apiUrl("/api/metrics"));
      if (!response.ok) throw new Error(await responseError(response));
      setMetrics(await response.json());
    } catch (error) {
      setMetricsError(
        error instanceof TypeError
          ? "Cannot reach the backend. Start the API server and try again."
          : error.message,
      );
    } finally {
      setMetricsLoading(false);
    }
  }

  useEffect(() => {
    loadMetrics();
  }, []);

  function selectFile(nextFile) {
    setAnalyseError("");
    setResult(null);
    if (!nextFile) return;
    if (!ACCEPTED_TYPES.has(nextFile.type)) {
      setFile(null);
      setAnalyseError("Unsupported file type. Choose a JPEG, PNG, or WebP image.");
      return;
    }
    if (nextFile.size > MAX_FILE_BYTES) {
      setFile(null);
      setAnalyseError("That file is too large. Choose an image under 5 MiB.");
      return;
    }
    setFile(nextFile);
  }

  function onFileChange(event) {
    selectFile(event.target.files?.[0]);
    event.target.value = "";
  }

  function onDrop(event) {
    event.preventDefault();
    setIsDragging(false);
    selectFile(event.dataTransfer.files?.[0]);
  }

  async function analyseImage() {
    if (!file || isAnalysing) return;
    setIsAnalysing(true);
    setAnalyseError("");
    setResult(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const response = await fetch(apiUrl("/api/predict"), { method: "POST", body: form });
      if (!response.ok) throw new Error(await responseError(response));
      setResult(await response.json());
    } catch (error) {
      setAnalyseError(
        error instanceof TypeError
          ? "Cannot reach the backend. Check that the API server is running."
          : error.message,
      );
    } finally {
      setIsAnalysing(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="DermaLens home">
          <span className="brand-mark" aria-hidden="true">D</span>
          <span>
            <strong>DermaLens</strong>
            <small>Image research prototype</small>
          </span>
        </a>
        <span className="header-tag">Educational use only</span>
      </header>

      <section className="page-heading" id="top">
        <div>
          <p className="eyebrow">HAM10000 · ResNet18 · Grad-CAM</p>
          <h1>Explore a skin lesion image</h1>
          <p className="page-intro">
            Review the model output and its attention overlay, or inspect the held-out evaluation summary.
          </p>
        </div>
      </section>

      <div className="tabbar" role="tablist" aria-label="DermaLens sections">
        <button
          id="tab-analyse"
          className={activeTab === "analyse" ? "tab active" : "tab"}
          role="tab"
          aria-selected={activeTab === "analyse"}
          aria-controls="panel-analyse"
          onClick={() => setActiveTab("analyse")}
        >
          Analyse
        </button>
        <button
          id="tab-model"
          className={activeTab === "model" ? "tab active" : "tab"}
          role="tab"
          aria-selected={activeTab === "model"}
          aria-controls="panel-model"
          onClick={() => {
            setActiveTab("model");
            loadMetrics();
          }}
        >
          Model &amp; Limitations
        </button>
      </div>

      {activeTab === "analyse" ? (
        <section id="panel-analyse" role="tabpanel" aria-labelledby="tab-analyse">
          <div className="notice" role="note">
            <span className="notice-icon" aria-hidden="true">!</span>
            <p>{DISCLAIMER}</p>
          </div>

          <div className="analyse-layout">
            <section className="card upload-card" aria-labelledby="upload-title">
              <div className="card-heading">
                <div>
                  <p className="eyebrow">Image input</p>
                  <h2 id="upload-title">Choose an image</h2>
                </div>
                {file && <span className="file-size">{(file.size / 1024).toFixed(0)} KB</span>}
              </div>

              <button
                type="button"
                className={`dropzone${isDragging ? " dragging" : ""}`}
                onClick={() => fileInput.current?.click()}
                onDragOver={(event) => { event.preventDefault(); setIsDragging(true); }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={onDrop}
                aria-label="Choose or drop a dermatoscopic image"
              >
                {previewUrl ? (
                  <img className="selected-preview" src={previewUrl} alt="Selected image preview" />
                ) : (
                  <span className="drop-content">
                    <span className="upload-icon" aria-hidden="true">↑</span>
                    <strong>Drop an image here</strong>
                    <span>or browse your files</span>
                    <small>JPEG, PNG, or WebP · up to 5 MiB</small>
                  </span>
                )}
              </button>
              <input
                ref={fileInput}
                className="visually-hidden"
                type="file"
                accept="image/jpeg,image/png,image/webp"
                onChange={onFileChange}
                aria-label="Select image file"
              />

              {file && <div className="selected-file"><span className="file-dot" />{file.name}</div>}
              {analyseError && <div className="error-banner" role="alert">{analyseError}</div>}
              <button
                type="button"
                className="primary-button"
                disabled={!file || isAnalysing}
                onClick={analyseImage}
              >
                {isAnalysing ? <><span className="spinner" /> Analysing image…</> : "Analyse image"}
              </button>
              {isAnalysing && <p className="loading-copy" aria-live="polite">Generating prediction and Grad-CAM overlay…</p>}
            </section>

            {result ? (
              <section className="result-area" aria-live="polite" aria-label="Analysis result">
                <div className="card prediction-card">
                  <div className="prediction-copy">
                    <p className="eyebrow">Model prediction</p>
                    <h2>{result.predicted_class_name}</h2>
                    <span className="class-code">{result.predicted_class}</span>
                  </div>
                  <div className="score-block">
                    <span className="score-label">Model score</span>
                    <strong>{asPercent(result.confidence)}</strong>
                    <small>Not a probability of disease</small>
                  </div>
                  <div className="inference-time">
                    <span>Inference time</span>
                    <strong>{Number(result.inference_time_ms).toFixed(1)} ms</strong>
                  </div>
                </div>

                <div className="card probability-card">
                  <div className="card-heading compact-heading">
                    <div>
                      <p className="eyebrow">Seven-class output</p>
                      <h3>Class probabilities</h3>
                    </div>
                    <span className="subtle-label">Model scores</span>
                  </div>
                  <div className="probability-list">
                    {CLASS_ORDER.map((code) => {
                      const value = result.class_probabilities[code];
                      return (
                        <div className="probability-row" key={code}>
                          <div className="probability-label">
                            <span>{CLASS_NAMES[code]}</span>
                            <strong>{asPercent(value)}</strong>
                          </div>
                          <div className="bar-track" role="meter" aria-label={`${CLASS_NAMES[code]} model probability`} aria-valuemin="0" aria-valuemax="1" aria-valuenow={value}>
                            <div className={`bar-fill${code === result.predicted_class ? " selected" : ""}`} style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div className="image-comparison">
                  <figure className="card image-card">
                    <figcaption><span>01</span> Original image</figcaption>
                    <img src={previewUrl} alt="Uploaded dermatoscopic image" />
                  </figure>
                  <figure className="card image-card">
                    <figcaption><span>02</span> Grad-CAM overlay</figcaption>
                    <img
                      src={`data:${result.gradcam_overlay_media_type};base64,${result.gradcam_overlay_base64}`}
                      alt="Grad-CAM overlay showing model attention for the predicted class"
                    />
                  </figure>
                </div>
                <p className="cam-caption">Grad-CAM shows model attention, not medical correctness.</p>
              </section>
            ) : (
              <section className="empty-result card" aria-live="polite">
                <div className="empty-icon" aria-hidden="true">◌</div>
                <h2>Your result will appear here</h2>
                <p>Select a dermatoscopic image, then analyse it to view the model output and attention overlay.</p>
                <div className="empty-rule" />
                <small>Images are processed locally by the configured DermaLens backend.</small>
              </section>
            )}
          </div>
        </section>
      ) : (
        <section id="panel-model" role="tabpanel" aria-labelledby="tab-model" className="model-panel">
          {metricsLoading && !metrics && <div className="card loading-card"><span className="spinner" /> Loading metrics…</div>}
          {metricsError && (
            <div className="error-banner metrics-error" role="alert">
              {metricsError}
              <button className="text-button" type="button" onClick={() => loadMetrics(true)}>Retry</button>
            </div>
          )}
          {metrics && (
            <>
              <div className="metrics-heading">
                <div>
                  <p className="eyebrow">Held-out evaluation</p>
                  <h2>Model performance</h2>
                </div>
                <span className="test-count">{metrics.test_image_count.toLocaleString()} test images</span>
              </div>

              <div className="metric-grid">
                <MetricCard label="Accuracy" value={asPercent(metrics.accuracy)} detail="Overall correct predictions" />
                <MetricCard label="Macro F1" value={asDecimal(metrics.macro_f1)} detail={`95% CI ${asDecimal(metrics.macro_f1_bootstrap_95_ci[0])}–${asDecimal(metrics.macro_f1_bootstrap_95_ci[1])}`} />
                <MetricCard label="Macro AUC" value={asDecimal(metrics.macro_ovr_roc_auc)} detail="One-vs-rest" />
                <MetricCard label="Melanoma recall" value={asPercent(metrics.melanoma_argmax_recall)} detail="Argmax classification" />
              </div>

              <div className="baseline-note">
                <span className="baseline-icon" aria-hidden="true">↗</span>
                <p>
                  Always predicting the majority class (<strong>nv</strong>) gives{" "}
                  <strong>{asPercent(metrics.always_predict_nv_baseline.accuracy)}</strong> accuracy, so accuracy alone is misleading.
                </p>
              </div>

              <div className="model-grid">
                <section className="card table-card">
                  <div className="card-heading">
                    <div>
                      <p className="eyebrow">Test set · argmax</p>
                      <h3>Per-class results</h3>
                    </div>
                  </div>
                  <div className="table-scroll">
                    <table>
                      <thead>
                        <tr><th>Class</th><th>Test count</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
                      </thead>
                      <tbody>
                        {CLASS_ORDER.map((code) => {
                          const row = metrics.per_class_metrics_argmax[code];
                          return (
                            <tr key={code}>
                              <th scope="row"><span className="table-class-code">{code}</span>{CLASS_NAMES[code]}</th>
                              <td>{metrics.test_class_counts[code]}</td>
                              <td>{asDecimal(row.precision)}</td>
                              <td>{asDecimal(row.recall)}</td>
                              <td>{asDecimal(row["f1-score"])}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </section>

                <section className="card confusion-card">
                  <div className="card-heading">
                    <div>
                      <p className="eyebrow">Rows: actual · columns: predicted</p>
                      <h3>Confusion matrix</h3>
                    </div>
                  </div>
                  <img
                    className="confusion-image"
                    src={apiUrl("/api/metrics/confusion-matrix")}
                    alt="Seven-class test confusion matrix"
                  />
                </section>
              </div>

              <section className="card limitations-card">
                <div className="card-heading">
                  <div>
                    <p className="eyebrow">Read before interpreting</p>
                    <h3>Limitations</h3>
                  </div>
                </div>
                <ul className="limitations-list">
                  <li>Dermatoscopic data from Austria and Australia, skewed toward lighter skin tones, and not validated on other populations.</li>
                  <li>The split is lesion-level, not patient-level; the dataset has no patient ID.</li>
                  <li>Melanoma recall is {asPercent(metrics.melanoma_argmax_recall)}; many melanomas were missed.</li>
                  <li>Rare classes have small test counts, as shown in the table above.</li>
                  <li>Training was CPU-only with partial fine-tuning of the network.</li>
                  <li>The model has no out-of-distribution detection.</li>
                </ul>
              </section>
              <p className="metrics-footnote">Educational research results only. The test set was evaluated once.</p>
            </>
          )}
        </section>
      )}

      <footer className="footer">
        <span>DermaLens · Educational research prototype</span>
        <span>HAM10000 · ResNet18 · Grad-CAM</span>
      </footer>
    </main>
  );
}

function MetricCard({ label, value, detail }) {
  return (
    <section className="card metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </section>
  );
}

export default App;
