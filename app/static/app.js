const form = document.getElementById("riskForm");
const modelBadge = document.getElementById("modelBadge");
const riskProbability = document.getElementById("riskProbability");
const riskBand = document.getElementById("riskBand");
const predictionText = document.getElementById("predictionText");
const repaymentText = document.getElementById("repaymentText");
const riskDial = document.getElementById("riskDial");
const driversList = document.getElementById("driversList");
const aucMetric = document.getElementById("aucMetric");
const f1Metric = document.getElementById("f1Metric");
const accuracyMetric = document.getElementById("accuracyMetric");
const submitButton = form.querySelector("button[type='submit']");

function percent(v) {
  return v == null ? "—" : `${Math.round(v * 100)}%`;
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    modelBadge.textContent = data.model_loaded
      ? `● ${data.model} model loaded`
      : "○ Training required";
    modelBadge.style.color = data.model_loaded ? "#2fd391" : "#f6b73c";
  } catch {
    modelBadge.textContent = "Health check unavailable";
  }
}

function renderDrivers(drivers) {
  if (!drivers?.length) {
    driversList.innerHTML = `<div class="empty-state">No explanation available.</div>`;
    return;
  }

  const maxImpact = Math.max(...drivers.map(d => Math.abs(d.impact)), 0.0001);

  driversList.innerHTML = drivers.map(d => {
    const width = Math.max(8, Math.round((Math.abs(d.impact) / maxImpact) * 100));
    const cls = d.impact > 0 ? "impact-up" : "impact-down";
    return `
      <div class="driver">
        <div class="driver-top">
          <strong>${d.label}</strong>
          <span class="${cls}">${d.direction}</span>
        </div>
        <div class="driver-bar">
          <div class="driver-fill" style="width:${width}%"></div>
        </div>
      </div>
    `;
  }).join("");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const data = Object.fromEntries(new FormData(form).entries());

  for (const key of Object.keys(data)) {
    data[key] = Number(data[key]);
  }

  submitButton.disabled = true;
  submitButton.innerHTML = "Scoring risk...";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.detail || "Prediction failed.");
    }

    const probability = result.default_probability;
    const degrees = Math.round(probability * 360);

    riskProbability.textContent = percent(probability);
    riskDial.style.background =
      `conic-gradient(#f6b73c ${degrees}deg, #27313d ${degrees}deg)`;

    riskBand.textContent = `${result.risk_band} RISK`;
    riskBand.className = `risk-band ${result.risk_band.toLowerCase()}`;

    predictionText.textContent = result.predicted_class;
    repaymentText.textContent =
      `Estimated repayment probability: ${percent(result.repayment_probability)}.`;

    aucMetric.textContent = percent(result.roc_auc);
    f1Metric.textContent = percent(result.f1);
    accuracyMetric.textContent = percent(result.accuracy);

    renderDrivers(result.top_drivers);
  } catch (error) {
    predictionText.textContent = "Assessment unavailable";
    repaymentText.textContent = error.message;
  } finally {
    submitButton.disabled = false;
    submitButton.innerHTML = `Run risk assessment <span>→</span>`;
  }
});

checkHealth();
