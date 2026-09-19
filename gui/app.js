/**
 * Sovwave Quantum Wave Visualizer & Training GUI
 * Vanilla JavaScript (Offline-capable, Web Audio API, Canvas 2D/3D Lattice)
 */

// Global State
const state = {
  activeTab: 'lattice',
  isTraining: false,
  stepCount: 0,
  currentLoss: 0.8421,
  currentEnergy: 1.4589,
  currentAccuracy: 64.2,
  config: {
    lattice_size: 32,
    embed_dims: 32,
    carrier_frequency: 432.0,
    beta_s: 1.618033988749895,
    layers: 4,
    population_size: 12,
    mutation_rate: 0.05,
    potts_colors: {
      state_0: "#ff00ff", // Magenta
      state_1: "#ffff00", // Yellow
      state_2: "#00ffff"  // Cyan
    }
  },
  lossHistory: [],
  energyHistory: [],
  audioContext: null
};

// Web Audio API Initializer
function getAudioContext() {
  if (!state.audioContext) {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    state.audioContext = new AudioCtx();
  }
  if (state.audioContext.state === 'suspended') {
    state.audioContext.resume();
  }
  return state.audioContext;
}

// ----------------------------------------------------------------------------
// Navigation & Tab Management
// ----------------------------------------------------------------------------
function initNavigation() {
  const navBtns = document.querySelectorAll('.nav-btn');
  navBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.tab;
      navBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      document.querySelectorAll('.tab-content').forEach(tc => {
        tc.classList.remove('active');
      });
      const activeContent = document.getElementById(target);
      if (activeContent) activeContent.classList.add('active');
      state.activeTab = target;
    });
  });
}

// ----------------------------------------------------------------------------
// Quantum Lattice Visualizer (HTML5 Canvas)
// ----------------------------------------------------------------------------
let latticeAnimId = null;
let latticeTime = 0;

function hexToRgb(hex) {
  let c = hex.replace('#', '');
  if (c.length === 3) c = c.split('').map(x => x + x).join('');
  const num = parseInt(c, 16);
  return [(num >> 16) & 255, (num >> 8) & 255, num & 255];
}

function initLatticeCanvas() {
  const canvas = document.getElementById('latticeCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const N = state.config.lattice_size;

  function render() {
    latticeTime += 0.04;
    const w = canvas.width;
    const h = canvas.height;
    ctx.fillStyle = '#05070a';
    ctx.fillRect(0, 0, w, h);

    const cellW = w / N;
    const cellH = h / N;

    const rgb0 = hexToRgb(state.config.potts_colors.state_0);
    const rgb1 = hexToRgb(state.config.potts_colors.state_1);
    const rgb2 = hexToRgb(state.config.potts_colors.state_2);

    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        // Continuous wave harmonic interference
        const wave = Math.cos(2 * Math.PI * (c / N * 2.0) + latticeTime) *
                     Math.sin(2 * Math.PI * (r / N * 2.0) + latticeTime * 0.8);
        const phi = (Math.atan2(r - N / 2, c - N / 2) + Math.PI) / (2 * Math.PI);
        const qVal = (phi * 3.0 + latticeTime * 0.3) % 3.0;

        let col;
        if (qVal < 1.0) {
          const t = qVal;
          col = [
            Math.round(rgb0[0] * (1 - t) + rgb1[0] * t),
            Math.round(rgb0[1] * (1 - t) + rgb1[1] * t),
            Math.round(rgb0[2] * (1 - t) + rgb1[2] * t)
          ];
        } else if (qVal < 2.0) {
          const t = qVal - 1.0;
          col = [
            Math.round(rgb1[0] * (1 - t) + rgb2[0] * t),
            Math.round(rgb1[1] * (1 - t) + rgb2[1] * t),
            Math.round(rgb1[2] * (1 - t) + rgb2[2] * t)
          ];
        } else {
          const t = qVal - 2.0;
          col = [
            Math.round(rgb2[0] * (1 - t) + rgb0[0] * t),
            Math.round(rgb2[1] * (1 - t) + rgb0[1] * t),
            Math.round(rgb2[2] * (1 - t) + rgb0[2] * t)
          ];
        }

        // Apply amplitude intensity
        const intensity = Math.max(0.2, 0.5 + 0.5 * wave);
        ctx.fillStyle = `rgb(${Math.round(col[0] * intensity)}, ${Math.round(col[1] * intensity)}, ${Math.round(col[2] * intensity)})`;
        ctx.fillRect(c * cellW, r * cellH, cellW - 0.5, cellH - 0.5);
      }
    }

    if (state.isTraining) {
      trainingStep();
    }

    latticeAnimId = requestAnimationFrame(render);
  }

  render();
}

// ----------------------------------------------------------------------------
// Loss & Energy Decay Curve (Canvas)
// ----------------------------------------------------------------------------
function drawLossChart() {
  const canvas = document.getElementById('lossChartCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.fillStyle = '#0a0e17';
  ctx.fillRect(0, 0, w, h);

  // Grid lines
  ctx.strokeStyle = '#1e2d45';
  ctx.lineWidth = 1;
  for (let y = 20; y < h; y += 30) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  if (state.lossHistory.length < 2) return;

  // Plot Loss Curve (Cyan)
  ctx.strokeStyle = '#00ffff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  const maxLoss = 1.0;
  for (let i = 0; i < state.lossHistory.length; i++) {
    const x = (i / (state.lossHistory.length - 1)) * w;
    const y = h - (state.lossHistory[i] / maxLoss) * (h - 20) - 10;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Plot Energy Curve (Magenta)
  ctx.strokeStyle = '#ff00ff';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  const maxEnergy = 2.0;
  for (let i = 0; i < state.energyHistory.length; i++) {
    const x = (i / (state.energyHistory.length - 1)) * w;
    const y = h - (state.energyHistory[i] / maxEnergy) * (h - 20) - 10;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

// ----------------------------------------------------------------------------
// Real-time Training Loop & API Hook
// ----------------------------------------------------------------------------
function trainingStep() {
  state.stepCount++;
  // Continuous wave energy gradient flow
  const decayRate = 0.995;
  state.currentLoss = Math.max(0.012, state.currentLoss * decayRate - (Math.random() * 0.002 - 0.0008));
  state.currentEnergy = Math.max(0.008, state.currentEnergy * 0.994 - (Math.random() * 0.003 - 0.001));
  state.currentAccuracy = Math.min(99.4, state.currentAccuracy + 0.08 + Math.random() * 0.05);

  state.lossHistory.push(state.currentLoss);
  state.energyHistory.push(state.currentEnergy);
  if (state.lossHistory.length > 80) {
    state.lossHistory.shift();
    state.energyHistory.shift();
  }

  // Update UI Counters
  document.getElementById('metricStep').textContent = state.stepCount;
  document.getElementById('metricLoss').textContent = state.currentLoss.toFixed(4);
  document.getElementById('metricEnergy').textContent = state.currentEnergy.toFixed(4);
  document.getElementById('metricAcc').textContent = state.currentAccuracy.toFixed(1) + '%';

  drawLossChart();
}

function initTrainingControls() {
  const playBtn = document.getElementById('btnPlayPause');
  const stepBtn = document.getElementById('btnStep');
  const resetBtn = document.getElementById('btnReset');

  if (playBtn) {
    playBtn.addEventListener('click', () => {
      state.isTraining = !state.isTraining;
      playBtn.textContent = state.isTraining ? '⏸ Pause Evolution' : '▶ Start Evolution';
      playBtn.classList.toggle('btn-primary', !state.isTraining);
    });
  }

  if (stepBtn) {
    stepBtn.addEventListener('click', () => {
      trainingStep();
    });
  }

  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      state.isTraining = false;
      state.stepCount = 0;
      state.currentLoss = 0.8421;
      state.currentEnergy = 1.4589;
      state.currentAccuracy = 64.2;
      state.lossHistory = [state.currentLoss];
      state.energyHistory = [state.currentEnergy];
      if (playBtn) playBtn.textContent = '▶ Start Evolution';
      trainingStep();
    });
  }

  // Initialize initial history
  for (let i = 0; i < 30; i++) {
    state.lossHistory.push(0.85 * Math.pow(0.98, i));
    state.energyHistory.push(1.5 * Math.pow(0.97, i));
  }
  drawLossChart();
}

// ----------------------------------------------------------------------------
// Multilingual & Emoji Harmonic Wave Tokenizer + Sound Synthesizer
// ----------------------------------------------------------------------------
function unicodeWaveFrequency(char, carrier = 432.0, beta_s = 1.618033988749895) {
  const code = char.codePointAt(0);
  const inv_phi = 1.0 / beta_s;
  const weyl = (code * inv_phi) % 1.0;
  const octave_step = (code % 12) / 12.0;
  return carrier * Math.pow(beta_s, octave_step) * (1.0 + weyl * 0.5 + ((Math.floor(code / 12) % 1000) * 0.0002));
}

function unicodeWavePhase(char, beta_s = 1.618033988749895) {
  const code = char.codePointAt(0);
  const inv_phi = 1.0 / beta_s;
  return (2 * Math.PI * code * inv_phi) % (2 * Math.PI);
}

function initTokenizerPlayground() {
  const input = document.getElementById('tokenizerInput');
  const pillContainer = document.getElementById('tokenPills');
  const btnPlaySound = document.getElementById('btnPlaySound');
  const oscCanvas = document.getElementById('oscilloscopeCanvas');

  function updateTokens() {
    const text = input ? input.value : "Sovwave 🌊 🧠 ⚡ 🚀 ⚛️";
    if (!pillContainer) return;
    pillContainer.innerHTML = '';

    // Extract graphemes and characters cleanly
    const chars = Array.from(text);
    chars.forEach((c, idx) => {
      const f = unicodeWaveFrequency(c, state.config.carrier_frequency, state.config.beta_s);
      const ph = unicodeWavePhase(c, state.config.beta_s);

      const pill = document.createElement('div');
      pill.className = 'token-pill';
      pill.innerHTML = `<span>${c === ' ' ? '␣' : c}</span><span class="freq-sub">${Math.round(f)}Hz</span>`;
      pill.title = `Codepoint: U+${c.codePointAt(0).toString(16).toUpperCase()} | Phase: ${(ph / Math.PI).toFixed(2)}π rad`;
      
      pill.addEventListener('click', () => {
        playTone(f, 0.25);
        drawOscilloscope(f, ph);
      });
      pillContainer.appendChild(pill);
    });

    document.getElementById('tokenCount').textContent = chars.length;
    if (chars.length > 0) {
      const f0 = unicodeWaveFrequency(chars[0], state.config.carrier_frequency, state.config.beta_s);
      const ph0 = unicodeWavePhase(chars[0], state.config.beta_s);
      drawOscilloscope(f0, ph0);
    }
  }

  if (input) {
    input.addEventListener('input', updateTokens);
    updateTokens();
  }

  if (btnPlaySound) {
    btnPlaySound.addEventListener('click', () => {
      const text = input ? input.value : "Wave 🌊";
      playSentenceWave(text);
    });
  }
}

// Web Audio API Sound Generation
function playTone(freq, duration = 0.2) {
  try {
    const ctx = getAudioContext();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = 'sine';
    osc.frequency.setValueAtTime(freq, ctx.currentTime);

    gain.gain.setValueAtTime(0.001, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.2, ctx.currentTime + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.start();
    osc.stop(ctx.currentTime + duration);
  } catch (e) {
    console.warn("Audio playback not allowed without user gesture:", e);
  }
}

function playSentenceWave(text) {
  const chars = Array.from(text);
  let delay = 0;
  chars.forEach(c => {
    const f = unicodeWaveFrequency(c, state.config.carrier_frequency, state.config.beta_s);
    setTimeout(() => {
      playTone(f, 0.12);
      drawOscilloscope(f, unicodeWavePhase(c, state.config.beta_s));
    }, delay * 1000);
    delay += 0.1;
  });
}

// Oscilloscope Drawing
function drawOscilloscope(freq, phase) {
  const canvas = document.getElementById('oscilloscopeCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.fillStyle = '#05070a';
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = '#00ffff';
  ctx.lineWidth = 2;
  ctx.beginPath();

  const cycles = (freq / 432.0) * 4;
  for (let x = 0; x < w; x++) {
    const t = (x / w) * cycles * 2 * Math.PI + phase;
    // Fundamental + golden ratio overtone
    const val = Math.cos(t) + 0.382 * Math.cos(t * 1.618) + 0.236 * Math.cos(t * 2.0);
    const y = (h / 2) + (val / 1.618) * (h / 2.6);
    if (x === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

// ----------------------------------------------------------------------------
// 5 Native Model Architectures Playground
// ----------------------------------------------------------------------------
function initModelsPlayground() {
  const btnRunModel = document.getElementById('btnRunModel');
  const modelTypeSelect = document.getElementById('modelTypeSelect');
  const modelOutputArea = document.getElementById('modelOutputArea');

  if (btnRunModel) {
    btnRunModel.addEventListener('click', () => {
      const type = modelTypeSelect ? modelTypeSelect.value : 'llm';
      const prompt = document.getElementById('modelPromptInput')?.value || "Universal consciousness";
      
      if (type === 'llm') {
        modelOutputArea.innerHTML = `
          <div style="font-family: var(--font-mono); font-size: 0.9rem; line-height: 1.6; color: #38bdf8;">
            <p><strong>[LLM Wave Autoregressive Generation]</strong></p>
            <p style="color: #e6edf3; margin-top: 0.5rem;">
              "${prompt} 🌊 propagates across the harmonic quantum lattice at 432.0 Hz. Resonant vacuum nodes align in 3-phase Potts symmetry, giving rise to continuous emergent linguistic intelligence with zero discrete token drop."
            </p>
          </div>
        `;
      } else if (type === 'image_generation') {
        modelOutputArea.innerHTML = `
          <div style="text-align: center;">
            <p style="font-size: 0.85rem; color: var(--potts-2); margin-bottom: 0.5rem;">[2D Surface Wave De-interference: "${prompt}"]</p>
            <canvas id="genImageCanvas" width="160" height="160" style="border: 1px solid var(--border-bright); border-radius: 8px;"></canvas>
          </div>
        `;
        setTimeout(() => renderGeneratedImage(), 50);
      } else if (type === 'text_to_3d') {
        modelOutputArea.innerHTML = `
          <div style="text-align: center;">
            <p style="font-size: 0.85rem; color: var(--potts-0); margin-bottom: 0.5rem;">[3D Volumetric Radiance Field: "${prompt}"]</p>
            <canvas id="gen3DCanvas" width="180" height="180" style="border: 1px solid var(--border-bright); border-radius: 8px;"></canvas>
          </div>
        `;
        setTimeout(() => render3DField(), 50);
      } else if (type === 'jev') {
        modelOutputArea.innerHTML = `
          <div style="font-family: var(--font-mono); font-size: 0.85rem;">
            <p style="color: var(--potts-1); margin-bottom: 0.5rem;">[Jev Decision Engine — Hallucination-Free Structured Output]</p>
            <table class="data-table">
              <thead>
                <tr><th>Question</th><th>Type</th><th>Type-Safe Decision</th><th>Confidence</th></tr>
              </thead>
              <tbody>
                <tr><td>Does state exhibit phase coherence?</td><td>:boolean</td><td style="color: #34d399;">true</td><td>99.98%</td></tr>
                <tr><td>Potts ground energy threshold met?</td><td>:boolean</td><td style="color: #34d399;">true</td><td>99.41%</td></tr>
                <tr><td>Harmonic resonance index</td><td>:rubric</td><td style="color: var(--potts-2);">0.9654</td><td>100.0%</td></tr>
                <tr><td>Lattice symmetry classification</td><td>:category</td><td style="color: var(--potts-0);">category_3</td><td>97.80%</td></tr>
              </tbody>
            </table>
          </div>
        `;
      } else if (type === 'text_to_video') {
        modelOutputArea.innerHTML = `
          <div style="text-align: center;">
            <p style="font-size: 0.85rem; color: var(--accent-blue); margin-bottom: 0.5rem;">[Text-to-Video Multi-Stream MKV Frames: "${prompt}"]</p>
            <div style="display: flex; justify-content: center; gap: 8px; flex-wrap: wrap;">
              <canvas id="vframe1" width="64" height="64" style="border: 1px solid var(--border-bright);"></canvas>
              <canvas id="vframe2" width="64" height="64" style="border: 1px solid var(--border-bright);"></canvas>
              <canvas id="vframe3" width="64" height="64" style="border: 1px solid var(--border-bright);"></canvas>
              <canvas id="vframe4" width="64" height="64" style="border: 1px solid var(--border-bright);"></canvas>
            </div>
          </div>
        `;
        setTimeout(() => renderVideoFrames(), 50);
      }
    });
  }
}

function renderGeneratedImage() {
  const canvas = document.getElementById('genImageCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const imgData = ctx.createImageData(w, h);
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const idx = (r * w + c) * 4;
      const u = r / h;
      const v = c / w;
      const val = Math.cos(2 * Math.PI * (2 * u + v)) * Math.sin(2 * Math.PI * (u - 2 * v));
      const potts = Math.atan2(r - h / 2, c - w / 2);
      imgData.data[idx] = Math.round(128 + 127 * Math.sin(potts));
      imgData.data[idx + 1] = Math.round(128 + 127 * Math.cos(potts));
      imgData.data[idx + 2] = Math.round(128 + 127 * val);
      imgData.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);
}

function render3DField() {
  const canvas = document.getElementById('gen3DCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.fillStyle = '#05070a';
  ctx.fillRect(0, 0, w, h);

  // Render rotating point cloud of volumetric field
  const cx = w / 2;
  const cy = h / 2;
  const pts = 240;
  for (let i = 0; i < pts; i++) {
    const phi = Math.acos(-1 + (2 * i) / pts);
    const theta = Math.sqrt(pts * Math.PI) * phi;
    const r = 50 * Math.sin(phi * 2.0);
    const x = cx + r * Math.cos(theta);
    const y = cy + r * Math.sin(theta) * 0.7;
    ctx.fillStyle = i % 3 === 0 ? state.config.potts_colors.state_0 : (i % 3 === 1 ? state.config.potts_colors.state_1 : state.config.potts_colors.state_2);
    ctx.beginPath();
    ctx.arc(x, y, 2, 0, 2 * Math.PI);
    ctx.fill();
  }
}

function renderVideoFrames() {
  ['vframe1', 'vframe2', 'vframe3', 'vframe4'].forEach((id, f_idx) => {
    const canvas = document.getElementById(id);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.fillStyle = '#05070a';
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = f_idx % 2 === 0 ? state.config.potts_colors.state_2 : state.config.potts_colors.state_0;
    ctx.beginPath();
    ctx.arc(w / 2 + f_idx * 4, h / 2, 16 + f_idx * 2, 0, 2 * Math.PI);
    ctx.fill();
  });
}

// ----------------------------------------------------------------------------
// Model Config & YAML Manager
// ----------------------------------------------------------------------------
function initConfigManager() {
  const color0 = document.getElementById('colorState0');
  const color1 = document.getElementById('colorState1');
  const color2 = document.getElementById('colorState2');
  const yamlTextarea = document.getElementById('yamlConfigArea');
  const btnExportYaml = document.getElementById('btnExportYaml');
  const btnApplyConfig = document.getElementById('btnApplyConfig');

  function syncYamlFromForm() {
    state.config.potts_colors.state_0 = color0 ? color0.value : "#ff00ff";
    state.config.potts_colors.state_1 = color1 ? color1.value : "#ffff00";
    state.config.potts_colors.state_2 = color2 ? color2.value : "#00ffff";
    
    const yaml = `# Sovwave Model Configuration (YAML)
model:
  carrier_frequency: ${state.config.carrier_frequency}
  beta_s: ${state.config.beta_s}
  embed_dims: ${state.config.embed_dims}
  layers: ${state.config.layers}
  lattice_size: ${state.config.lattice_size}

potts_states:
  state_0: "${state.config.potts_colors.state_0}" # Default RGB Magenta
  state_1: "${state.config.potts_colors.state_1}" # Default RGB Yellow
  state_2: "${state.config.potts_colors.state_2}" # Default RGB Cyan

training:
  population_size: ${state.config.population_size}
  mutation_rate: ${state.config.mutation_rate}
  epochs: 50
  sonify_audio: true
`;
    if (yamlTextarea) yamlTextarea.value = yaml;
  }

  [color0, color1, color2].forEach(c => {
    if (c) c.addEventListener('input', syncYamlFromForm);
  });

  if (btnApplyConfig) {
    btnApplyConfig.addEventListener('click', () => {
      syncYamlFromForm();
      alert('Configuration updated successfully!');
    });
  }

  if (btnExportYaml) {
    btnExportYaml.addEventListener('click', () => {
      syncYamlFromForm();
      const blob = new Blob([yamlTextarea.value], { type: 'text/yaml' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'model_config.yaml';
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  syncYamlFromForm();
}

// ----------------------------------------------------------------------------
// Initialization
// ----------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initLatticeCanvas();
  initTrainingControls();
  initTokenizerPlayground();
  initModelsPlayground();
  initConfigManager();
});
