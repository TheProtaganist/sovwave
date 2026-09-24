/* tokenizer.js — Sovwave Continuous & Phonetic Wave Tokenizer Demo
 * Implements:
 * 1. Phonetic Continuous Wave Tokenizer (IPA Formants F1/F2/F3, 963 Hz Carrier, Speech Wave Packets)
 * 2. Harmonic Wave Tokenizer (Subword frequency harmonics, 432 Hz Carrier, Circular Phase Angles)
 * 3. Real-Time Web Audio API Synthesis (Acoustic pronunciation & binaural resonance)
 */

"use strict";

// ============================================================================
// CONSTANTS & VOCABULARIES
// ============================================================================

const PHONETIC_CARRIER = 963.0; // Hz (Crown / Universal harmonic)
const HARMONIC_CARRIER = 432.0; // Hz (Verdi / Sacred tuning)
const GOLDEN_RATIO = 1.618033988749895; // Φ
const TWO_PI = 2 * Math.PI;

// IPA Acoustic Formant Frequencies (F1, F2, F3 in Hz) matching PhoneticWaveTokenizer.jl
const IPA_FORMANTS = {
  // Vowels
  'a': [730.0, 1090.0, 2440.0],  // /a/ father
  'e': [530.0, 1840.0, 2480.0],  // /e/ bed
  'i': [270.0, 2290.0, 3010.0],  // /i/ bee
  'o': [570.0, 840.0,  2410.0],  // /o/ boat
  'u': [300.0, 870.0,  2240.0],  // /u/ boot
  'ə': [500.0, 1500.0, 2500.0],  // /ə/ schwa
  'æ': [660.0, 1720.0, 2410.0],  // /æ/ cat
  'ʌ': [640.0, 1190.0, 2390.0],  // /ʌ/ cup
  'ɔ': [510.0, 890.0,  2300.0],  // /ɔ/ caught
  'ɛ': [550.0, 1750.0, 2500.0],  // /ɛ/ open-mid
  'ɪ': [390.0, 1990.0, 2550.0],  // /ɪ/ bit
  'ʊ': [440.0, 1020.0, 2240.0],  // /ʊ/ put
  'ɨ': [340.0, 1450.0, 2400.0],  // /ɨ/ close central

  // Consonants: Plosives & Stops
  'p': [100.0, 1000.0, 2500.0],
  'b': [100.0, 1000.0, 2500.0],
  't': [200.0, 1700.0, 2600.0],
  'd': [200.0, 1700.0, 2600.0],
  'k': [300.0, 2000.0, 3000.0],
  'g': [300.0, 2000.0, 3000.0],

  // Fricatives & Sibilants
  'f': [150.0, 2000.0, 4000.0],
  'v': [150.0, 2000.0, 4000.0],
  's': [200.0, 4000.0, 8000.0],
  'z': [200.0, 4000.0, 8000.0],
  'ʃ': [300.0, 2500.0, 6000.0],  // "sh"
  'ʒ': [300.0, 2500.0, 6000.0],  // "zh"
  'h': [500.0, 1500.0, 2500.0],
  'θ': [200.0, 3000.0, 7000.0],  // "th" voiceless
  'ð': [200.0, 3000.0, 7000.0],  // "th" voiced

  // Nasals, Liquids, Approximants
  'm': [280.0, 1200.0, 2400.0],
  'n': [280.0, 1700.0, 2600.0],
  'ŋ': [280.0, 2100.0, 2700.0],  // "ng"
  'l': [360.0, 1300.0, 2500.0],
  'r': [420.0, 1300.0, 1600.0],
  'w': [300.0, 870.0,  2240.0],
  'j': [270.0, 2290.0, 3010.0],  // "y"

  // Punctuation & Cadence
  '.': [150.0, 450.0,  900.0],
  ',': [250.0, 750.0,  1500.0],
  ';': [200.0, 600.0,  1200.0],
  ':': [220.0, 660.0,  1320.0],
  '!': [450.0, 1350.0, 2700.0],
  '?': [300.0, 900.0,  1800.0],
  '-': [180.0, 540.0,  1080.0],
  ' ': [120.0, 360.0,  720.0]
};

// Grapheme to spoken phonetic expansions
const GRAPHEME_EXPANSIONS = {
  "th": "θ", "sh": "ʃ", "ch": "tʃ", "ng": "ŋ", "ph": "f", "wh": "w", "ck": "k",
  "ee": "i", "oo": "u", "ea": "i", "ou": "u", "ai": "e", "ay": "e", "oa": "o",
  "0": "ziro", "1": "wʌn", "2": "tu", "3": "θri", "4": "for", "5": "faɪv",
  "6": "sɪks", "7": "sɛvən", "8": "et", "9": "naɪn",
  "+": "plʌs", "=": "ikwəl", "*": "taɪmz", "/": "dɪvaɪd",
  "==": "ikwəlz", "!=": "not_ikwəl", "<=": "lɛs_ikwəl", ">=": "gretər_ikwəl"
};

// Harmonic BPE vocabulary builder
function buildHarmonicVocabulary() {
  const vocab = new Map();
  const invVocab = [];
  let id = 0;
  const reg = (tok) => { vocab.set(tok, id); invVocab.push(tok); id++; };

  ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>", "<MASK>"].forEach(reg);
  for (let c = 32; c <= 126; c++) reg(String.fromCharCode(c));
  ["\n", "\t", "\r"].forEach(reg);

  ["th", "he", "in", "er", "an", "re", "on", "at", "en", "nd", "ti", "es", "or", "te", "of",
   "ed", "is", "it", "al", "ar", "st", "to", "nt", "ng", "se", "ha", "as", "ou", "io", "le",
   "the", "and", "for", "are", "but", "not", "you", "all", "any", "can", "her", "was", "one",
   "our", "out", "day", "get", "has", "him", "his", "how", "man", "new", "now", "old", "see",
   "two", "way", "who", "boy", "did", "its", "let", "put", "say", "she", "too", "use"].forEach(reg);

  ["==", "!=", "<=", ">=", "->", "=>", "+=", "-=", "*=", "/="].forEach(reg);
  ["🌊", "🧠", "⚡", "🚀", "⚛️", "🔮", "🎵", "🎶", "🔊", "🌐", "✨", "💡"].forEach(reg);
  return { vocab, invVocab };
}

const HARMONIC_DATA = buildHarmonicVocabulary();
const VOCAB = HARMONIC_DATA.vocab;

// Global State
let currentTokenizerMode = "phonetic"; // "phonetic" | "harmonic"
let audioCtx = null;
let currentTokens = [];

// ============================================================================
// PHONETIC WAVE TOKENIZER LOGIC
// ============================================================================

function textToPhoneticTokens(text) {
  if (!text) return [];
  const words = text.split(/(\s+|[.,!?;:\-+*/=()\[\]{}])/g).filter(Boolean);
  const result = [];

  for (const w of words) {
    if (!w) continue;

    // Check punctuation or whitespace
    if (/^\s+$/.test(w)) {
      result.push({
        text: "␣",
        display: "space",
        ipa: "[pause]",
        formants: IPA_FORMANTS[' '],
        carrier: PHONETIC_CARRIER,
        durationMs: 90,
        type: "cadence"
      });
      continue;
    }

    if (IPA_FORMANTS[w]) {
      result.push({
        text: w,
        display: w,
        ipa: `[${w}]`,
        formants: IPA_FORMANTS[w],
        carrier: PHONETIC_CARRIER,
        durationMs: 180,
        type: "cadence"
      });
      continue;
    }

    // Number or symbol expansion
    let lower = w.toLowerCase();
    for (const [pattern, repl] of Object.entries(GRAPHEME_EXPANSIONS)) {
      lower = lower.replace(new RegExp(pattern, "g"), repl);
    }

    // Syllable / character breakdown
    const chars = Array.from(lower);
    let i = 0;
    while (i < chars.length) {
      let ch = chars[i];
      let formants = IPA_FORMANTS[ch];

      if (!formants) {
        // Fallback formant synthesis from UTF code
        const code = ch.codePointAt(0);
        const f1 = 200.0 + (code % 600);
        const f2 = 800.0 + ((code * 7) % 1500);
        const f3 = 2000.0 + ((code * 13) % 1500);
        formants = [f1, f2, f3];
      }

      const isVowel = "aeiouəæʌɔɛɪʊɨ".includes(ch);
      result.push({
        text: chars[i],
        display: chars[i],
        ipa: `/${ch}/`,
        formants: formants,
        carrier: PHONETIC_CARRIER,
        durationMs: isVowel ? 140 : 80,
        type: isVowel ? "vowel" : "consonant"
      });
      i++;
    }
  }

  return result;
}

// ============================================================================
// HARMONIC SUBWORD TOKENIZER LOGIC
// ============================================================================

function unicodeWaveFreq(char) {
  const u = char.codePointAt(0);
  const invPhi = 1.0 / GOLDEN_RATIO;
  const weyl = (u * invPhi) % 1.0;
  const octaveStep = (u % 12) / 12.0;
  return HARMONIC_CARRIER * Math.pow(GOLDEN_RATIO, octaveStep) * (1.0 + weyl * 0.5);
}

function unicodeWavePhase(char) {
  const u = char.codePointAt(0);
  return (TWO_PI * u * (1.0 / GOLDEN_RATIO)) % TWO_PI;
}

function textToHarmonicTokens(text) {
  if (!text) return [];
  const chars = Array.from(text);
  const tokens = [];
  let idx = 0;

  while (idx < chars.length) {
    let matched = false;
    for (let len = Math.min(4, chars.length - idx); len >= 2; len--) {
      const sub = chars.slice(idx, idx + len).join("");
      if (VOCAB.has(sub)) {
        tokens.push({
          text: sub,
          display: sub,
          id: VOCAB.get(sub),
          freq: tokenWaveFreq(sub),
          phase: tokenWavePh(sub),
          carrier: HARMONIC_CARRIER,
          type: "subword"
        });
        idx += len;
        matched = true;
        break;
      }
    }
    if (!matched) {
      const ch = chars[idx];
      tokens.push({
        text: ch,
        display: ch === " " ? "␣" : ch,
        id: VOCAB.get(ch) || -1,
        freq: unicodeWaveFreq(ch),
        phase: unicodeWavePhase(ch),
        carrier: HARMONIC_CARRIER,
        type: "char"
      });
      idx++;
    }
  }
  return tokens;
}

function tokenWaveFreq(token) {
  let totalF = 0.0, weightSum = 0.0;
  for (let i = 0; i < token.length; i++) {
    const w = Math.pow(GOLDEN_RATIO, -i);
    totalF += w * unicodeWaveFreq(token[i]);
    weightSum += w;
  }
  return totalF / weightSum;
}

function tokenWavePh(token) {
  let ph = 0.0;
  for (let i = 0; i < token.length; i++) {
    ph += unicodeWavePhase(token[i]) * Math.pow(GOLDEN_RATIO, -i);
  }
  return ph % TWO_PI;
}

// ============================================================================
// WEB AUDIO API REAL-TIME SYNTHESIS
// ============================================================================

function getAudioContext() {
  if (!audioCtx) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    audioCtx = new AudioContextClass();
  }
  if (audioCtx.state === 'suspended') {
    audioCtx.resume();
  }
  return audioCtx;
}

function playSingleToken(token) {
  const ctx = getAudioContext();
  const now = ctx.currentTime;
  const masterGain = ctx.createGain();
  masterGain.connect(ctx.destination);

  if (currentTokenizerMode === "phonetic") {
    const formants = token.formants || [500, 1500, 2500];
    const dur = (token.durationMs || 120) / 1000.0;

    masterGain.gain.setValueAtTime(0.001, now);
    masterGain.gain.exponentialRampToValueAtTime(0.18, now + 0.02);
    masterGain.gain.exponentialRampToValueAtTime(0.001, now + dur);

    // 3 Formant Oscillators (F1, F2, F3) + Sub-harmonic Carrier
    const freqs = [token.carrier * 0.5, formants[0], formants[1], formants[2]];
    const amps  = [0.25, 0.4, 0.25, 0.1];

    freqs.forEach((f, idx) => {
      const osc = ctx.createOscillator();
      const oscGain = ctx.createGain();
      osc.type = idx === 0 ? "triangle" : "sine";
      osc.frequency.setValueAtTime(f, now);
      oscGain.gain.value = amps[idx];
      osc.connect(oscGain);
      oscGain.connect(masterGain);
      osc.start(now);
      osc.stop(now + dur + 0.05);
    });
  } else {
    // Harmonic Subword synthesis
    const dur = 0.16;
    const f = token.freq || 432.0;

    masterGain.gain.setValueAtTime(0.001, now);
    masterGain.gain.exponentialRampToValueAtTime(0.2, now + 0.02);
    masterGain.gain.exponentialRampToValueAtTime(0.001, now + dur);

    [f, f * GOLDEN_RATIO, f * 2].forEach((freqVal, idx) => {
      const osc = ctx.createOscillator();
      osc.type = "sine";
      osc.frequency.setValueAtTime(freqVal, now);
      const g = ctx.createGain();
      g.gain.value = 0.3 / (idx + 1);
      osc.connect(g);
      g.connect(masterGain);
      osc.start(now);
      osc.stop(now + dur + 0.05);
    });
  }
}

function playAllTokens() {
  const ctx = getAudioContext();
  let time = ctx.currentTime + 0.05;

  currentTokens.forEach((token) => {
    const durSec = currentTokenizerMode === "phonetic" 
      ? (token.durationMs || 100) / 1000.0 
      : 0.14;

    const masterGain = ctx.createGain();
    masterGain.connect(ctx.destination);
    masterGain.gain.setValueAtTime(0.001, time);
    masterGain.gain.exponentialRampToValueAtTime(0.15, time + 0.015);
    masterGain.gain.exponentialRampToValueAtTime(0.001, time + durSec);

    if (currentTokenizerMode === "phonetic") {
      const formants = token.formants || [500, 1500, 2500];
      [token.carrier * 0.5, ...formants].forEach((f, idx) => {
        const osc = ctx.createOscillator();
        osc.type = idx === 0 ? "triangle" : "sine";
        osc.frequency.setValueAtTime(f, time);
        const g = ctx.createGain();
        g.gain.value = idx === 0 ? 0.2 : 0.3;
        osc.connect(g);
        g.connect(masterGain);
        osc.start(time);
        osc.stop(time + durSec);
      });
    } else {
      const f = token.freq || 432.0;
      const osc = ctx.createOscillator();
      osc.type = "sine";
      osc.frequency.setValueAtTime(f, time);
      osc.connect(masterGain);
      osc.start(time);
      osc.stop(time + durSec);
    }

    time += durSec + 0.01;
  });
}

// ============================================================================
// CANVAS VISUALIZATION (Formant Spectrogram & Standing Wave Phase Ring)
// ============================================================================

function drawWaveCanvas(canvasId, tokens) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.width = canvas.offsetWidth * dpr;
  const h = canvas.height = canvas.offsetHeight * dpr;

  ctx.clearRect(0, 0, w, h);

  if (!tokens || tokens.length === 0) {
    ctx.fillStyle = 'rgba(255,255,255,0.2)';
    ctx.font = `${14 * dpr}px 'JetBrains Mono', monospace`;
    ctx.textAlign = 'center';
    ctx.fillText('Type above or select a preset to synthesize continuous acoustic waves', w / 2, h / 2);
    return;
  }

  const n = Math.min(tokens.length, 32);
  const colW = w / n;

  if (currentTokenizerMode === "phonetic") {
    // Formant Spectrogram
    const maxFreq = 6000.0;
    tokens.slice(0, n).forEach((tok, i) => {
      const x = i * colW;
      const formants = tok.formants || [500, 1500, 2500];

      // Draw vertical formant energy bands
      formants.forEach((f, idx) => {
        const normY = 1.0 - (f / maxFreq);
        const y = normY * (h - 30 * dpr);
        const bandH = Math.max(8 * dpr, (14 - idx * 3) * dpr);

        const colors = [
          'rgba(0, 255, 255, 0.85)',   // F1: Cyan
          'rgba(255, 0, 255, 0.85)',   // F2: Magenta
          'rgba(255, 255, 0, 0.85)'    // F3: Yellow
        ];

        ctx.fillStyle = colors[idx] || colors[0];
        ctx.fillRect(x + 2 * dpr, y - bandH / 2, colW - 4 * dpr, bandH);

        // Formant frequency label
        ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
        ctx.font = `${8 * dpr}px 'JetBrains Mono', monospace`;
        ctx.textAlign = 'center';
        ctx.fillText(`${Math.round(f)}`, x + colW / 2, y - bandH / 2 - 2 * dpr);
      });

      // Token label at bottom
      ctx.fillStyle = tok.type === 'vowel' ? '#00ffff' : (tok.type === 'cadence' ? '#ffff00' : '#e6edf3');
      ctx.font = `bold ${10 * dpr}px 'JetBrains Mono', monospace`;
      ctx.textAlign = 'center';
      ctx.fillText(tok.display, x + colW / 2, h - 8 * dpr);
    });
  } else {
    // Harmonic Frequency Spectrum
    const maxF = 4000.0;
    tokens.slice(0, n).forEach((tok, i) => {
      const x = i * colW;
      const f = tok.freq || 432.0;
      const barH = (f / maxF) * (h - 35 * dpr);
      const y = h - barH - 20 * dpr;

      const grad = ctx.createLinearGradient(0, y, 0, h - 20 * dpr);
      grad.addColorStop(0, '#00ffff');
      grad.addColorStop(1, '#ff00ff');

      ctx.fillStyle = grad;
      ctx.fillRect(x + 2 * dpr, y, colW - 4 * dpr, barH);

      ctx.fillStyle = 'rgba(255, 255, 255, 0.75)';
      ctx.font = `${8 * dpr}px 'JetBrains Mono', monospace`;
      ctx.textAlign = 'center';
      ctx.fillText(`${Math.round(f)}Hz`, x + colW / 2, y - 4 * dpr);

      ctx.fillStyle = '#fff';
      ctx.font = `bold ${10 * dpr}px 'JetBrains Mono', monospace`;
      ctx.fillText(tok.display, x + colW / 2, h - 6 * dpr);
    });
  }
}

// ============================================================================
// UI REFRESH & TOKEN RENDERING
// ============================================================================

const PRESETS = {
  english: "The universe is an interconnected web of vibrations and acoustic fields.",
  dialogue: "Hello, how are you? I am doing well, thank you!",
  math: "Two plus two equals four, and 5 + 5 = 10.",
  code: "def add(a, b):\n    return a + b",
  multilingual: "Quantum Resonance 🌊 波浪智能 الذكاء الموجي Волновая интерференция",
  logic: "If all roses are flowers and all flowers need sunlight, then all roses need sunlight."
};

function setTokenizerMode(mode) {
  currentTokenizerMode = mode;
  document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.mode === mode);
  });
  tokenizeText();
}

function loadPreset(key) {
  const input = document.getElementById('inputText');
  if (input && PRESETS[key]) {
    input.value = PRESETS[key];
    tokenizeText();
  }
}

function tokenizeText() {
  const input = document.getElementById('inputText');
  const text = input ? input.value : "";
  const t0 = performance.now();

  if (currentTokenizerMode === "phonetic") {
    currentTokens = textToPhoneticTokens(text);
  } else {
    currentTokens = textToHarmonicTokens(text);
  }

  const latencyUs = (performance.now() - t0) * 1000;
  const nChars = Array.from(text).length;
  const nTokens = currentTokens.length;
  const compRatio = nChars > 0 ? (nTokens / nChars) : 0;
  const throughput = latencyUs > 0 ? (nTokens / (latencyUs / 1e6)) : 0;

  // Update Stats Cards
  const setEl = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  setEl('charCount', nChars);
  setEl('tokenCount', nTokens);
  setEl('compressionRatio', compRatio.toFixed(3));
  setEl('latency', latencyUs.toFixed(1));
  setEl('throughput', Math.round(throughput).toLocaleString());

  const modeBadge = document.getElementById('activeModeBadge');
  if (modeBadge) {
    modeBadge.textContent = currentTokenizerMode === 'phonetic'
      ? 'Acoustic IPA Formant Waveform (963 Hz)'
      : 'Harmonic Golden Ratio Overtone Series (432 Hz)';
  }

  // Render Token Pills
  const container = document.getElementById('tokensDisplay');
  if (container) {
    container.innerHTML = '';
    currentTokens.forEach((tok) => {
      const pill = document.createElement('div');
      pill.className = `token-pill ${tok.type || ''}`;
      pill.title = "Click to synthesize & hear this token's acoustic waveform";

      if (currentTokenizerMode === "phonetic") {
        const [f1, f2, f3] = tok.formants || [500, 1500, 2500];
        pill.innerHTML = `
          <div class="tok-label"><strong>"${escapeHtml(tok.display)}"</strong> <span class="tok-ipa">${tok.ipa}</span></div>
          <div class="tok-meta">
            <span class="f1">F₁:${Math.round(f1)}</span>
            <span class="f2">F₂:${Math.round(f2)}</span>
            <span class="f3">F₃:${Math.round(f3)}</span>
          </div>
        `;
      } else {
        pill.innerHTML = `
          <div class="tok-label"><strong>"${escapeHtml(tok.display)}"</strong></div>
          <div class="tok-meta">
            <span class="f1">${Math.round(tok.freq)} Hz</span>
            <span class="f2">φ=${tok.phase.toFixed(2)}</span>
          </div>
        `;
      }

      pill.addEventListener('click', () => {
        playSingleToken(tok);
        pill.classList.add('pulse');
        setTimeout(() => pill.classList.remove('pulse'), 250);
      });

      container.appendChild(pill);
    });
  }

  drawWaveCanvas('waveformCanvas', currentTokens);
}

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Global initialization
window.addEventListener('DOMContentLoaded', () => {
  const input = document.getElementById('inputText');
  if (input) {
    input.addEventListener('input', tokenizeText);
  }

  const playBtn = document.getElementById('btnPlayAudio');
  if (playBtn) {
    playBtn.addEventListener('click', playAllTokens);
  }

  document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => setTokenizerMode(btn.dataset.mode));
  });

  tokenizeText();
});
