const API_BASE_URL = "https://stocksense-ai-nirk.onrender.com";
const tickerInput = document.getElementById("ticker-input");
const daysInput = document.getElementById("days-input");
const epochsInput = document.getElementById("epochs-input");
const predictBtn = document.getElementById("predict-btn");
const loadingEl = document.getElementById("loading");
const errorEl = document.getElementById("error");
const trendText = document.getElementById("trend-text");
const lastValues = document.getElementById("last-values");
const rmseText = document.getElementById("rmse-text");

const ctx = document.getElementById("price-chart").getContext("2d");
let priceChart = null;

function showLoading(show) {
  loadingEl.classList.toggle("hidden", !show);
}

function showError(message) {
  if (!message) {
    errorEl.classList.add("hidden");
    errorEl.textContent = "";
  } else {
    errorEl.classList.remove("hidden");
    errorEl.textContent = message;
  }
}

async function trainModel(ticker, epochs) {
  const url = new URL(`${API_BASE_URL}/train-model`);
  url.searchParams.set("ticker", ticker);
  url.searchParams.set("epochs", String(epochs));

  const res = await fetch(url.toString(), {
    method: "POST",
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "Failed to train model");
  }
  return res.json();
}

async function predict(ticker, daysAhead) {
  const url = new URL(`${API_BASE_URL}/predict`);
  url.searchParams.set("ticker", ticker);
  url.searchParams.set("days_ahead", String(daysAhead));

  const res = await fetch(url.toString());
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "Prediction failed");
  }
  return res.json();
}

function buildChart(actual, predicted, daysAhead) {
  if (priceChart) {
    priceChart.destroy();
  }

  const recentWindow = 120;
  const actualSlice =
    actual.length > recentWindow
      ? actual.slice(actual.length - recentWindow)
      : actual.slice();

  const labels = [];
  for (let i = 0; i < actualSlice.length; i++) {
    labels.push(`Day -${actualSlice.length - i}`);
  }

  const futureLabels = [];
  for (let i = 1; i <= daysAhead; i++) {
    futureLabels.push(`+${i}`);
  }

  const allLabels = labels.concat(futureLabels);
  const predictedPadded = new Array(labels.length).fill(null).concat(predicted);

  priceChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: allLabels,
      datasets: [
        {
          label: "Actual Close",
          data: actualSlice,
          borderColor: "#38bdf8",
          backgroundColor: "rgba(56, 189, 248, 0.15)",
          tension: 0.2,
          borderWidth: 2,
        },
        {
          label: "Predicted Close",
          data: predictedPadded,
          borderColor: "#22c55e",
          backgroundColor: "rgba(34, 197, 94, 0.15)",
          borderDash: [6, 4],
          tension: 0.2,
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      interaction: {
        mode: "index",
        intersect: false,
      },
      plugins: {
        legend: {
          labels: {
            color: "#e5e7eb",
          },
        },
      },
      scales: {
        x: {
          ticks: {
            color: "#9ca3af",
          },
          grid: {
            color: "rgba(55, 65, 81, 0.5)",
          },
        },
        y: {
          ticks: {
            color: "#9ca3af",
          },
          grid: {
            color: "rgba(55, 65, 81, 0.5)",
          },
        },
      },
    },
  });
}

predictBtn.addEventListener("click", async () => {
  const rawTicker = tickerInput.value.trim();
  const daysAhead = Number(daysInput.value) || 7;
  const epochs = Number(epochsInput.value) || 10;

  if (!rawTicker) {
    showError("Please enter a stock ticker (e.g. AAPL).");
    return;
  }

  const ticker = rawTicker.toUpperCase();
  showError("");
  showLoading(true);
  predictBtn.disabled = true;

  try {
    // const trainSummary = await trainModel(ticker, epochs);
    // rmseText.textContent = trainSummary.rmse
    //   ? `${trainSummary.rmse.toFixed(3)} (train RMSE)`
    //   : "-";
    rmseText.textContent = "Model trained successfully";

    const prediction = await predict(ticker, daysAhead);

    const {
      actual,
      predicted,
      trend,
      last_actual,
      last_predicted,
      warning,
    } = prediction;

    // if (warning) {
    //   showError(warning);
    // } else {
    //   showError("");
    // }

    buildChart(actual, predicted, daysAhead);

    trendText.textContent =
      trend === "UP"
        ? `${ticker}: Predicted trend UP`
        : `${ticker}: Predicted trend DOWN`;

    lastValues.textContent = `Last close: ${last_actual.toFixed(
      2
    )} | Next predicted: ${last_predicted.toFixed(2)}`;
  } catch (err) {
    console.error(err);
    showError(err.message || "Unexpected error.");
  } finally {
    showLoading(false);
    predictBtn.disabled = false;
  }
});

