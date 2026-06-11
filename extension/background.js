// background.js — Deepfake Video Detector service worker
// Handles fetch requests on behalf of content.js to bypass page CSP restrictions.

const SERVER = "http://127.0.0.1:7177";

chrome.runtime.onInstalled.addListener(() => {
  console.log("Deepfake Video Detector extension installed.");
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "analyze") {
    fetch(`${SERVER}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: message.url }),
    })
      .then(async (resp) => {
        const data = await resp.json();
        if (!resp.ok || data.error) {
          sendResponse({ error: data.error ?? `Server error ${resp.status}` });
        } else {
          sendResponse({ data });
        }
      })
      .catch((err) => {
        sendResponse({
          error:
            "Cannot reach the local server at 127.0.0.1:7177.\n" +
            "Make sure server.py is running: python server.py",
        });
      });

    return true; // keep message channel open for async response
  }

  if (message.type === "status") {
    fetch(`${SERVER}/status`)
      .then((r) => r.json())
      .then((d) => sendResponse({ ok: true, data: d }))
      .catch(() => sendResponse({ ok: false }));
    return true;
  }
});
