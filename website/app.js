"use strict";

const copyButton = document.getElementById("copy-citation");
copyButton.addEventListener("click", async () => {
  const citation = document.getElementById("citation-text");
  const status = document.getElementById("copy-status");
  try {
    await navigator.clipboard.writeText(citation.textContent);
    copyButton.textContent = "Copied";
    status.textContent = "BibTeX copied to clipboard.";
    window.setTimeout(() => { copyButton.textContent = "Copy BibTeX"; }, 2200);
  } catch {
    const range = document.createRange();
    range.selectNodeContents(citation);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = "Citation selected. Use your browser's copy command.";
  }
});

async function initializeRewardLab() {
  const response = await fetch("assets/cifar-trajectories.json");
  if (!response.ok) throw new Error("Recorded histories could not be loaded.");
  const data = await response.json();
  const canvas = document.getElementById("reward-canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas is unavailable.");
  const slider = document.getElementById("update-slider");
  const params = new URL(location.href).searchParams;
  const names = ["weak", "medium", "strong"];
  let selection = names.includes(params.get("model")) ? params.get("model") : "strong";
  const requestedUpdate = Number(params.get("update"));
  let update = Number.isInteger(requestedUpdate) && requestedUpdate >= 1 && requestedUpdate <= 80 ? requestedUpdate : 80;
  const colors = { ink: "#202335", accent: "#4c4fc8", grid: "#ededf2", muted: "#53576a" };
  const observations = {
    weak: "The Weak model shows a visible gap between SDE and ODE rewards.",
    medium: "The Medium model brings the SDE and ODE reward trajectories closer.",
    strong: "The Strong model shows substantial overlap between the two reward trajectories."
  };

  function axisStep(range) {
    const raw = range / 4;
    const base = 10 ** Math.floor(Math.log10(raw));
    return [1, 2, 2.5, 5, 10].find(step => step * base >= raw) * base;
  }

  function draw() {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!width || !height) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const model = data.models[selection];
    const margin = { left: width < 500 ? 52 : 66, right: width < 500 ? 12 : 24, top: 22, bottom: 52 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    let lower = Infinity, upper = -Infinity;
    for (const key of ["sde", "ode"]) {
      model[key].mean.forEach((value, i) => {
        lower = Math.min(lower, value - model[key].std[i]);
        upper = Math.max(upper, value + model[key].std[i]);
      });
    }
    const padding = (upper - lower) * 0.08;
    const step = axisStep(upper - lower + padding * 2);
    const min = Math.floor((lower - padding) / step) * step;
    const max = Math.ceil((upper + padding) / step) * step;
    const x = i => margin.left + i / 79 * plotWidth;
    const y = value => margin.top + (max - value) / (max - min) * plotHeight;
    ctx.font = "14px Manrope, Arial, sans-serif";
    ctx.textBaseline = "middle";
    ctx.lineWidth = 1;
    for (let value = min; value <= max + step / 10; value += step) {
      ctx.beginPath(); ctx.strokeStyle = colors.grid;
      ctx.moveTo(margin.left, y(value)); ctx.lineTo(width - margin.right, y(value)); ctx.stroke();
      ctx.fillStyle = colors.muted; ctx.textAlign = "right";
      ctx.fillText(value.toFixed(2), margin.left - 12, y(value));
    }
    const ticks = width < 500 ? [1, 40, 80] : [1, 20, 40, 60, 80];
    for (const tick of ticks) {
      ctx.fillStyle = colors.muted; ctx.textAlign = "center";
      ctx.fillText(String(tick), x(tick - 1), height - margin.bottom + 21);
    }
    ctx.font = "16px Manrope, Arial, sans-serif";
    ctx.fillStyle = colors.ink; ctx.textAlign = "center";
    ctx.fillText("Fine-tune update", margin.left + plotWidth / 2, height - 8);
    if (width < 500) {
      // Keep the axis title above the plot on phones so it cannot overlap ticks.
      ctx.textAlign = "left"; ctx.fillText("Reward", margin.left, 9);
    } else {
      ctx.save(); ctx.translate(15, margin.top + plotHeight / 2); ctx.rotate(-Math.PI / 2);
      ctx.fillText("Reward", 0, 0); ctx.restore();
    }

    ctx.save(); ctx.beginPath(); ctx.rect(margin.left, margin.top, plotWidth, plotHeight); ctx.clip();
    for (const key of ["ode", "sde"]) {
      const values = model[key];
      ctx.beginPath();
      values.mean.forEach((value, i) => { if (i === 0) ctx.moveTo(x(i), y(value + values.std[i])); else ctx.lineTo(x(i), y(value + values.std[i])); });
      for (let i = 79; i >= 0; i--) ctx.lineTo(x(i), y(values.mean[i] - values.std[i]));
      ctx.closePath(); ctx.fillStyle = key === "sde" ? "rgba(76,79,200,.10)" : "rgba(32,35,53,.07)"; ctx.fill();
    }
    for (const key of ["ode", "sde"]) {
      ctx.beginPath(); model[key].mean.forEach((value, i) => { if (i === 0) ctx.moveTo(x(i), y(value)); else ctx.lineTo(x(i), y(value)); });
      ctx.setLineDash(key === "ode" ? [6, 5] : []);
      ctx.strokeStyle = key === "sde" ? colors.accent : colors.ink;
      ctx.lineWidth = key === "sde" ? 2.8 : 2.2; ctx.lineJoin = "round"; ctx.lineCap = "round"; ctx.stroke();
    }
    ctx.setLineDash([3, 5]); ctx.lineWidth = 1; ctx.strokeStyle = "rgba(32,35,53,.32)";
    ctx.beginPath(); ctx.moveTo(x(update - 1), margin.top); ctx.lineTo(x(update - 1), height - margin.bottom); ctx.stroke();
    ctx.restore(); ctx.setLineDash([]);
    for (const key of ["ode", "sde"]) {
      ctx.beginPath(); ctx.arc(x(update - 1), y(model[key].mean[update - 1]), 4.5, 0, Math.PI * 2);
      ctx.fillStyle = key === "sde" ? colors.accent : colors.ink; ctx.fill(); ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.stroke();
    }
  }

  function render(shareState = false) {
    const model = data.models[selection];
    const title = selection[0].toUpperCase() + selection.slice(1);
    slider.value = update;
    slider.style.setProperty("--range-fill", `${(update - 1) / 79 * 100}%`);
    document.getElementById("quality-value").textContent = title;
    document.getElementById("update-value").textContent = update;
    document.querySelectorAll("[data-model]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.model === selection)));
    document.getElementById("metric-sde").textContent = model.sde.mean[update - 1].toFixed(3);
    document.getElementById("metric-ode").textContent = model.ode.mean[update - 1].toFixed(3);
    document.getElementById("metric-violation").textContent = model.violation_rate.toFixed(3);
    document.getElementById("metric-fm").textContent = model.fm_loss.mean[update - 1].toFixed(3);
    document.getElementById("lab-observation").textContent = observations[selection];
    document.getElementById("lab-source").textContent = `Mean and one-standard-deviation bands over seeds 42, 43, and 44. After seed averaging, ${model.violations} of ${model.sde_improvements} SDE-improving transitions have an ODE reward decrease.`;
    canvas.setAttribute("aria-label", `${title} base model, recorded CIFAR-10 SDE and ODE rewards over eighty updates. Update ${update}: SDE ${model.sde.mean[update - 1].toFixed(3)}, ODE ${model.ode.mean[update - 1].toFixed(3)}. Conditional violation rate ${model.violation_rate.toFixed(3)}.`);
    if (shareState) {
      const url = new URL(location.href);
      url.searchParams.set("model", selection); url.searchParams.set("update", update);
      history.replaceState(null, "", url);
    }
    draw();
  }

  document.querySelectorAll("[data-model]").forEach(button => button.addEventListener("click", () => { selection = button.dataset.model; render(true); }));
  slider.addEventListener("input", () => { update = Number(slider.value); render(true); });
  document.getElementById("reset-lab").addEventListener("click", () => { selection = "strong"; update = 80; render(true); });
  new ResizeObserver(draw).observe(canvas);
  await document.fonts.ready;
  document.getElementById("reward-lab").setAttribute("aria-busy", "false");
  render();
}

initializeRewardLab().catch(() => {
  document.getElementById("reward-lab").setAttribute("aria-busy", "false");
  document.getElementById("lab-status").textContent = "The interactive records could not be loaded. The complete published figures and original histories are available below and in the repository.";
});
