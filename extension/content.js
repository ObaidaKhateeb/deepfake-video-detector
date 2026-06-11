/**
 * content.js — Deepfake Video Detector browser extension
 *
 * Scans the page for <video> elements, injects an "Analyze" button over each
 * one, and on click sends the video URL to the local Python server for
 * analysis.  Results are rendered in an overlay panel on the page.
 */

// ── Helpers ───────────────────────────────────────────────────────────────────

function escHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * Try to resolve a usable (non-blob) src URL from a <video> element.
 * Returns { url, isBlob } where isBlob means the page uses a streaming
 * format (MSE / blob:) that the server cannot download.
 */
function resolveVideoSrc(video) {
  const candidates = [
    video.currentSrc,
    video.src,
    ...Array.from(video.querySelectorAll("source")).map((s) => s.src),
  ].filter(Boolean);

  const direct = candidates.find((u) => /^https?:\/\//i.test(u));
  if (direct) return { url: direct, isBlob: false };

  const blob = candidates.find((u) => u.startsWith("blob:"));
  if (blob) return { url: blob, isBlob: true };

  return { url: null, isBlob: false };
}

function scoreColor(score) {
  if (score < 0.35) return "#4ade80";
  if (score < 0.65) return "#facc15";
  return "#f87171";
}

function verdictClass(verdict) {
  if (verdict === "Likely Real") return "dvd-verdict-real";
  if (verdict === "Likely Fake") return "dvd-verdict-fake";
  return "dvd-verdict-suspicious";
}

// ── DOM helpers ───────────────────────────────────────────────────────────────

function removePanel(wrapper) {
  wrapper.querySelector(".dvd-results-panel")?.remove();
}

function showError(wrapper, msg) {
  removePanel(wrapper);
  const panel = document.createElement("div");
  panel.className = "dvd-results-panel dvd-error";
  panel.innerHTML = `
    <div class="dvd-panel-header">
      <span class="dvd-panel-title">⚠ Deepfake Detector</span>
      <button class="dvd-close-btn" title="Close">✕</button>
    </div>
    <div class="dvd-error-msg">${escHtml(msg)}</div>
  `;
  panel.querySelector(".dvd-close-btn").addEventListener("click", () => panel.remove());
  wrapper.appendChild(panel);
}

function showLoading(wrapper) {
  removePanel(wrapper);
  const panel = document.createElement("div");
  panel.className = "dvd-results-panel dvd-loading";
  panel.innerHTML = `
    <div class="dvd-panel-header">
      <span class="dvd-panel-title">Deepfake Detector</span>
      <button class="dvd-close-btn" title="Cancel">✕</button>
    </div>
    <div class="dvd-loading-body">
      <div class="dvd-spinner-lg"></div>
      <div class="dvd-loading-text">Downloading &amp; analysing video…<br>
        <span class="dvd-loading-sub">This may take 15–60 seconds.</span>
      </div>
    </div>
  `;
  panel.querySelector(".dvd-close-btn").addEventListener("click", () => panel.remove());
  wrapper.appendChild(panel);
  return panel;
}

function showResults(wrapper, data) {
  removePanel(wrapper);

  const pct = Math.round(data.overall_score * 100);

  const barsHtml = Object.entries(data.components)
    .map(([label, comp]) => {
      const s = Math.round(comp.score * 100);
      const color = scoreColor(comp.score);
      const conf = Math.round(comp.confidence * 100);
      return `
        <div class="dvd-param-row" title="Confidence: ${conf}%">
          <span class="dvd-param-label">${escHtml(label)}</span>
          <div class="dvd-bar-track">
            <div class="dvd-bar-fill" style="width:${s}%;background:${color};"></div>
          </div>
          <span class="dvd-param-pct">${s}%</span>
        </div>`;
    })
    .join("");

  const detailsHtml = Object.entries(data.components)
    .map(([label, comp]) => {
      if (!comp.details.length) return "";
      const lines = comp.details
        .map((d) => `<div class="dvd-detail-line">${escHtml(d)}</div>`)
        .join("");
      return `<div class="dvd-detail-section"><b>${escHtml(label)}</b>${lines}</div>`;
    })
    .filter(Boolean)
    .join("");

  const panel = document.createElement("div");
  panel.className = "dvd-results-panel";
  panel.innerHTML = `
    <div class="dvd-panel-header">
      <span class="dvd-panel-title">Deepfake Detector</span>
      <button class="dvd-close-btn" title="Close">✕</button>
    </div>
    <div class="dvd-verdict-row">
      <div class="dvd-score-circle">
        <span class="dvd-score-num" style="color:${scoreColor(data.overall_score)}">${pct}%</span>
        <span class="dvd-score-sub">fake probability</span>
      </div>
      <div class="dvd-verdict-badge ${verdictClass(data.verdict)}">${escHtml(data.verdict)}</div>
    </div>
    <div class="dvd-section-title">Parameter Breakdown</div>
    <div class="dvd-bars">${barsHtml}</div>
    <details class="dvd-details-section">
      <summary class="dvd-details-toggle">Analysis Details</summary>
      <div class="dvd-details-content">${detailsHtml || "No details available."}</div>
    </details>
    <div class="dvd-footer">Analysis is heuristic. Results are indicative, not definitive.</div>
  `;

  panel.querySelector(".dvd-close-btn").addEventListener("click", () => panel.remove());
  wrapper.appendChild(panel);
}

// ── Core flow ─────────────────────────────────────────────────────────────────

async function handleAnalyze(video, wrapper, btn) {
  const { url, isBlob } = resolveVideoSrc(video);

  if (isBlob) {
    showError(
      wrapper,
      "This video uses a streaming/blob URL that cannot be downloaded by the server.\n" +
        "Try downloading the video file and analysing it with the desktop app instead."
    );
    return;
  }

  if (!url) {
    showError(wrapper, "Could not find a direct video URL on this element.");
    return;
  }

  btn.disabled = true;
  btn.innerHTML = '<span class="dvd-spinner"></span> Analysing…';

  const loadingPanel = showLoading(wrapper);
  let aborted = false;
  loadingPanel.querySelector(".dvd-close-btn").addEventListener("click", () => {
    aborted = true;
  });

  try {
    const response = await new Promise((resolve, reject) => {
      chrome.runtime.sendMessage({ type: "analyze", url }, (reply) => {
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
        } else {
          resolve(reply);
        }
      });
    });

    if (aborted) return;

    if (response.error) {
      showError(wrapper, response.error);
    } else {
      showResults(wrapper, response.data);
    }
  } catch (err) {
    if (aborted) return;
    showError(wrapper, `Extension error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.innerHTML = "🔍 Re-analyse";
  }
}

// ── Attach button to a <video> element ───────────────────────────────────────

function attachButton(video) {
  // Guard: skip if already processed, inside an iframe we don't own, or tiny
  if (video.dataset.dvdDone) return;
  video.dataset.dvdDone = "1";

  const rect = video.getBoundingClientRect();
  if (rect.width > 0 && rect.width < 80) return; // skip tiny thumbnails

  // Wrap the video in a relative-positioned container
  const wrapper = document.createElement("div");
  wrapper.className = "dvd-wrapper";

  // Preserve the video's computed display value (block / inline-block / etc.)
  const cs = window.getComputedStyle(video);
  const disp = cs.display;
  wrapper.style.display = disp === "inline" || disp === "inline-block"
    ? "inline-block"
    : "block";
  wrapper.style.position = "relative";

  // If the video has explicit width/height attributes, mirror them on the wrapper
  if (video.hasAttribute("width"))  wrapper.style.width  = video.getAttribute("width") + "px";
  if (video.hasAttribute("height")) wrapper.style.height = video.getAttribute("height") + "px";

  video.parentNode.insertBefore(wrapper, video);
  wrapper.appendChild(video);

  // Analyze button (top-right corner of the video)
  const btn = document.createElement("button");
  btn.className = "dvd-analyze-btn";
  btn.innerHTML = "🔍 Analyse";
  btn.title = "Analyse this video for deepfake / synthetic manipulation";
  wrapper.appendChild(btn);

  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    e.preventDefault();
    handleAnalyze(video, wrapper, btn);
  });
}

// ── Page scanning ─────────────────────────────────────────────────────────────

function scanPage() {
  document.querySelectorAll("video").forEach(attachButton);
}

// Observe dynamically added videos (SPAs, lazy-loaded players, etc.)
const observer = new MutationObserver((mutations) => {
  for (const mut of mutations) {
    for (const node of mut.addedNodes) {
      if (node.nodeType !== 1) continue;
      if (node.tagName === "VIDEO") {
        attachButton(node);
      } else if (typeof node.querySelectorAll === "function") {
        node.querySelectorAll("video").forEach(attachButton);
      }
    }
  }
});

observer.observe(document.documentElement, { childList: true, subtree: true });

// Initial scan + delayed re-scans for lazy content
scanPage();
setTimeout(scanPage, 1500);
setTimeout(scanPage, 4000);
