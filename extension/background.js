// background.js — Deepfake Video Detector service worker
// Minimal service worker required by Manifest V3.

chrome.runtime.onInstalled.addListener(() => {
  console.log("Deepfake Video Detector extension installed.");
});
