// popup.js — checks whether the local analysis server is reachable

const SERVER = "http://127.0.0.1:7177";

const dot        = document.getElementById("dot");
const statusText = document.getElementById("status-text");
const checkBtn   = document.getElementById("check-btn");

async function checkServer() {
  dot.className = "status-dot";
  statusText.textContent = "Checking…";
  checkBtn.disabled = true;

  try {
    const resp = await fetch(`${SERVER}/status`, { signal: AbortSignal.timeout(4000) });
    if (resp.ok) {
      dot.className = "status-dot ok";
      statusText.textContent = "Server running — ready to analyse";
    } else {
      throw new Error(`HTTP ${resp.status}`);
    }
  } catch {
    dot.className = "status-dot err";
    statusText.textContent = "Server not found — run: python server.py";
  } finally {
    checkBtn.disabled = false;
  }
}

checkBtn.addEventListener("click", checkServer);
checkServer();
