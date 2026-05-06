const params = new URLSearchParams(window.location.search);
const apiOverride = params.get("api");
const defaultBase =
  window.location.port === "8000"
    ? window.location.origin
    : "http://localhost:8000";
const apiBase = apiOverride ?? defaultBase;
const API_URL = `${apiBase.replace(/\/$/, "")}/predict`;

const form = document.getElementById("predict-form");
const textInput = document.getElementById("news-text");
const submitButton = document.getElementById("predict-btn");
const resultLabel = document.getElementById("result-label");
const resultProb = document.getElementById("result-prob");
const resultError = document.getElementById("result-error");

function setBusy(isBusy) {
  submitButton.disabled = isBusy;
  submitButton.textContent = isBusy ? "Analyzing..." : "Analyze";
}

function setResult({ label, probability, error }) {
  if (error) {
    resultError.textContent = error;
  } else {
    resultError.textContent = "";
  }

  resultLabel.textContent = label ?? "-";
  if (typeof probability === "number") {
    resultProb.textContent = `${(probability * 100).toFixed(1)}%`;
  } else {
    resultProb.textContent = "-";
  }
}

async function predict(text) {
  const response = await fetch(API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    let message = "Prediction failed";
    try {
      const payload = await response.json();
      if (payload.detail) {
        message = payload.detail;
      }
    } catch (error) {
      message = "Prediction failed";
    }
    throw new Error(message);
  }

  return response.json();
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const text = textInput.value.trim();
  if (!text) {
    setResult({ label: "-", probability: null, error: "Enter some text" });
    return;
  }

  setBusy(true);
  setResult({ label: "-", probability: null, error: "" });

  try {
    const result = await predict(text);
    setResult({
      label: result.label ?? result.raw_prediction ?? "-",
      probability: result.probability,
      error: "",
    });
  } catch (error) {
    setResult({ label: "-", probability: null, error: error.message });
  } finally {
    setBusy(false);
  }
});
