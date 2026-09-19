/* tokenizer.js — Harmonic Wave Tokenizer Demo
 * Implements frequency-based multi-character tokenization (Tournament Winner)
 * Mirrors the Julia implementation from src/Audio/WaveML/Tokenizer.jl
 */

"use strict";

// ============================================================================
// CORE CONSTANTS
// ============================================================================

const CARRIER_FREQUENCY = 432.0;  // Sacred 432 Hz tuning
const GOLDEN_RATIO = 1.618033988749895;  // Φ
const TWO_PI = 2 * Math.PI;

// ============================================================================
// VOCABULARY (Mirrors default_tokenizer)
// ============================================================================

function buildDefaultVocabulary() {
  const vocab = new Map();
  const invVocab = [];
  let id = 0;
  
  // Helper to register token
  const register = (token) => {
    vocab.set(token, id);
    invVocab.push(token);
    id++;
  };
  
  // Special tokens
  ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>", "<MASK>"].forEach(register);
  
  // ASCII printable (space through ~)
  for (let c = 32; c <= 126; c++) {
    register(String.fromCharCode(c));
  }
  
  // Whitespace
  ["\n", "\t", "\r"].forEach(register);
  
  // Common English subwords (frequency-based)
  ["th", "he", "in", "er", "an", "re", "on", "at", "en", "nd", "ti", "es", "or", "te", "of",
   "ed", "is", "it", "al", "ar", "st", "to", "nt", "ng", "se", "ha", "as", "ou", "io", "le",
   "ve", "co", "me", "de", "hi", "ri", "ro", "ic", "ne", "ea", "ra", "ce", "li", "ch", "ll",
   "mo", "ni", "wa", "mp", "ut", "ing", "ion", "ave",
   "the", "and", "for", "are", "but", "not", "you", "all", "any", "can", "her", "was", "one",
   "our", "out", "day", "get", "has", "him", "his", "how", "man", "new", "now", "old", "see",
   "two", "way", "who", "boy", "did", "its", "let", "put", "say", "she", "too", "use",
   "wave", "data", "true", "false", "token", "computing", "function", "return", "loss", "model", 
   "layer", "input", "state", "train", "infer", "harmonic"].forEach(register);
  
  // Latin extended
  ["á", "é", "í", "ó", "ú", "ñ", "ç", "ü", "ö", "ä", "à", "è", "ì", "ò", "ù"].forEach(register);
  
  // Greek
  ["α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "λ", "μ", "π", "φ", "ψ", "ω", "Φ", "Ψ", "Ω"].forEach(register);
  
  // Cyrillic (basic)
  ["а", "б", "в", "г", "д", "е", "ж", "з", "и", "к", "л", "м", "н", "о", "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "я"].forEach(register);
  
  // Math symbols
  ["∂", "∇", "∫", "∑", "∏", "√", "∞", "≈", "≠", "≤", "≥", "±", "×", "÷", "ℏ"].forEach(register);
  
  // Common operators
  ["==", "!=", "<=", ">=", "->", "=>", "+=", "-=", "*=", "/=", "::"].forEach(register);
  
  // Emojis
  ["🌊", "🧠", "⚡", "🚀", "⚛️", "🔮", "🎵", "🎶", "🔊", "🌐", "🌌", "✨", "🌟", "💡", "🔥", "🌈", "💻", "🤖"].forEach(register);
  
  return { vocab, invVocab };
}

const VOCAB_DATA = buildDefaultVocabulary();
const VOCAB = VOCAB_DATA.vocab;
const INV_VOCAB = VOCAB_DATA.invVocab;

// ============================================================================
// FREQUENCY AND PHASE COMPUTATION
// ============================================================================

function unicodeWaveFrequency(char) {
  const u = char.codePointAt(0);
  const invPhi = 1.0 / GOLDEN_RATIO;
  const weyl = (u * invPhi) % 1.0;
  const octaveStep = (u % 12) / 12.0;
  const f = CARRIER_FREQUENCY * Math.pow(GOLDEN_RATIO, octaveStep) * 
            (1.0 + weyl * 0.5 + Math.floor(u / 12) % 1000 * 0.0002);
  return f;
}

function unicodeWavePhase(char) {
  const u = char.codePointAt(0);
  const invPhi = 1.0 / GOLDEN_RATIO;
  return (TWO_PI * u * invPhi) % TWO_PI;
}

function tokenWaveFrequency(token) {
  if (token.length === 0) return CARRIER_FREQUENCY;
  if (token.length === 1) return unicodeWaveFrequency(token);
  
  let totalF = 0.0;
  let weightSum = 0.0;
  
  for (let i = 0; i < token.length; i++) {
    const w = Math.pow(GOLDEN_RATIO, -(i));
    totalF += w * unicodeWaveFrequency(token[i]);
    weightSum += w;
  }
  
  return totalF / weightSum;
}

function tokenWavePhase(token) {
  if (token.length === 0) return 0.0;
  if (token.length === 1) return unicodeWavePhase(token);
  
  let ph = 0.0;
  for (let i = 0; i < token.length; i++) {
    ph += unicodeWavePhase(token[i]) * Math.pow(GOLDEN_RATIO, -(i));
  }
  
  return ph % TWO_PI;
}

// ============================================================================
// WAVEFORM GENERATION
// ============================================================================

function generateWaveform(token, nSamples = 32, sampleRate = 48000.0, t0 = 0.0) {
  const freq = tokenWaveFrequency(token);
  const phase = tokenWavePhase(token);
  const dt = 1.0 / sampleRate;
  
  // Harmonics: fundamental, octave, fifth, golden overtone
  const harmonics = [freq, 2 * freq, 1.5 * freq, GOLDEN_RATIO * freq];
  const samples = [];
  
  for (let n = 0; n < nSamples; n++) {
    const t = t0 + n * dt;
    
    // Fundamental + overtones with golden decay
    let val = Math.cos(TWO_PI * freq * t + phase);
    val += 0.382 * Math.cos(TWO_PI * harmonics[1] * t + phase * 2.0);
    val += 0.236 * Math.cos(TWO_PI * harmonics[2] * t + phase * 1.5);
    val += 0.146 * Math.cos(TWO_PI * harmonics[3] * t + phase * GOLDEN_RATIO);
    
    // Envelope windowing (exponential decay)
    const envelope = Math.exp(-n / (nSamples * 0.8));
    samples.push(val * envelope);
  }
  
  // Normalize
  const norm = Math.sqrt(samples.reduce((sum, s) => sum + s * s, 0));
  if (norm > 1e-6) {
    for (let i = 0; i < samples.length; i++) {
      samples[i] /= norm;
    }
  }
  
  const energy = samples.reduce((sum, s) => sum + s * s, 0);
  
  return {
    token,
    frequency: freq,
    phase: phase,
    harmonics: harmonics,
    samples: samples,
    energy: energy
  };
}

// ============================================================================
// TOKENIZER
// ============================================================================

function tokenizeEnhanced(text) {
  if (!text || text.length === 0) return [];
  
  const chars = Array.from(text);
  const nChars = chars.length;
  const tokens = [];
  
  // Learn bigram frequencies
  const bigramFreq = new Map();
  for (let i = 0; i < nChars - 1; i++) {
    const bigram = chars[i] + chars[i + 1];
    bigramFreq.set(bigram, (bigramFreq.get(bigram) || 0) + 1);
  }
  
  // Tokenize
  let idx = 0;
  while (idx < nChars) {
    let matched = false;
    
    for (const len of [8, 6, 4, 2]) {
      if (idx + len <= nChars) {
        const sub = chars.slice(idx, idx + len).join('');
        const inVocab = VOCAB.has(sub);
        const isFreqBigram = (len === 2 && (bigramFreq.get(sub) || 0) >= 2);
        
        if (inVocab || isFreqBigram) {
          tokens.push({
            text: sub,
            id: VOCAB.get(sub) || -1,
            length: len,
            isMultiChar: len > 1,
            isEmoji: /\p{Emoji}/u.test(sub),
            isFrequencyBased: isFreqBigram && !inVocab
          });
          idx += len;
          matched = true;
          break;
        }
      }
    }
    
    if (!matched) {
      const ch = chars[idx];
      tokens.push({
        text: ch,
        id: VOCAB.get(ch) || -1,
        length: 1,
        isMultiChar: false,
        isEmoji: /\p{Emoji}/u.test(ch),
        isFrequencyBased: false
      });
      idx++;
    }
  }
  
  return tokens;
}

// ============================================================================
// VISUALIZATION
// ============================================================================

function visualizeWaveforms(tokens, canvasId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.offsetWidth * window.devicePixelRatio;
  const h = canvas.height = canvas.offsetHeight * window.devicePixelRatio;
  
  ctx.clearRect(0, 0, w, h);
  
  if (tokens.length === 0) {
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.font = '14px monospace';
    ctx.fillText('No tokens to visualize', w / 2 - 80, h / 2);
    return;
  }
  
  // Generate waveforms for visualization (limit to first 16 tokens for performance)
  const visTokens = tokens.slice(0, 16);
  const waveforms = visTokens.map(t => generateWaveform(t.text));
  
  // Draw frequency bars
  const barWidth = w / visTokens.length;
  const maxFreq = 8000.0;  // 8 kHz max
  
  waveforms.forEach((wf, i) => {
    const x = i * barWidth;
    const barHeight = (wf.frequency / maxFreq) * h * 0.8;
    const y = h - barHeight;
    
    // Color gradient based on frequency
    const hue = (wf.frequency / maxFreq) * 280;  // Blue to purple spectrum
    ctx.fillStyle = `hsla(${hue}, 70%, 60%, 0.8)`;
    ctx.fillRect(x + 2, y, barWidth - 4, barHeight);
    
    // Token label
    ctx.fillStyle = 'rgba(255,255,255,0.9)';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(wf.token.length > 4 ? wf.token.slice(0, 3) + '…' : wf.token, 
                 x + barWidth / 2, h - 5);
    
    // Frequency label
    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    ctx.font = '9px monospace';
    ctx.fillText(`${Math.round(wf.frequency)}Hz`, x + barWidth / 2, y - 5);
  });
}

// ============================================================================
// UI UPDATES
// ============================================================================

const PRESETS = {
  english: "The quick brown fox jumps over the lazy dog. Tokenization should compress common English patterns efficiently.",
  code: "def tokenize(text):\n    return [token for token in text.split() if token not in stopwords]",
  julia: "function wave_tokenize(tok::WaveTokenizer, text::String)::Vector{WaveForm}\n    return tokenize(tok, text)\nend",
  math: "∫∂ψ/∂t = -iℏ∇²ψ + V(x)ψ, Φ = (1+√5)/2 ≈ 1.618, π ≈ 3.14159, e^(iπ) + 1 = 0",
  emoji: "🌊🧠⚡🚀⚛️🔮🎵🎶🔊 Wave computing 🌐🌌✨🌟💡 transforms data 🔥🌈 into sound 🎵",
  multilingual: "English Español Français Deutsch 中文 日本語 한국어 العربية עברית हिन्दी Русский",
  cyrillic: "Привет мир! Это текст на русском языке для проверки токенизации кириллицы.",
  arabic: "السلام عليكم. هذا نص باللغة العربية لاختبار تجزئة الكلمات والحروف.",
  cjk: "中文分詞測試：量子波動計算，將數據轉換為聲波頻率。日本語：波動計算システム。",
  mixed: "Wave 🌊 computing: ∂ψ/∂t + código混合text العربية with tokens!=chars && subwords>>chars"
};

function loadPreset(name) {
  const textarea = document.getElementById('inputText');
  textarea.value = PRESETS[name] || "";
  tokenizeText();
}

function tokenizeText() {
  const text = document.getElementById('inputText').value;
  const t0 = performance.now();
  const tokens = tokenizeEnhanced(text);
  const latencyMs = performance.now() - t0;
  
  // Update statistics
  const nChars = Array.from(text).length;  // Proper UTF-8 char count
  const nTokens = tokens.length;
  const compressionRatio = nChars > 0 ? nTokens / nChars : 0;
  const multiCharCount = tokens.filter(t => t.isMultiChar).length;
  const latencyUs = latencyMs * 1000;
  const throughput = nTokens > 0 ? (nTokens / latencyMs) * 1000 : 0;
  
  document.getElementById('charCount').textContent = nChars;
  document.getElementById('tokenCount').textContent = nTokens;
  document.getElementById('compressionRatio').textContent = compressionRatio.toFixed(3);
  document.getElementById('multiCharTokens').textContent = multiCharCount;
  document.getElementById('latency').textContent = latencyUs.toFixed(1);
  document.getElementById('throughput').textContent = Math.round(throughput).toLocaleString();
  
  // Comparison stats
  document.getElementById('charLevelTokens').textContent = nChars + " tokens";
  document.getElementById('bpeTokens').textContent = Math.round(nChars * 0.7) + " tokens";
  document.getElementById('sovwaveTokens').textContent = nTokens + " tokens";
  const gain = nChars > 0 ? ((nChars - nTokens) / nChars * 100) : 0;
  document.getElementById('compressionGain').textContent = gain.toFixed(1) + "%";
  
  // Render tokens with frequency highlighting
  const tokensDisplay = document.getElementById('tokensDisplay');
  tokensDisplay.innerHTML = '';
  
  tokens.forEach((token, idx) => {
    const wf = generateWaveform(token.text);
    const div = document.createElement('div');
    div.className = 'token-item';
    
    // Color coding based on token type
    if (token.isMultiChar) div.classList.add('multi-char-token');
    if (token.isEmoji) div.classList.add('emoji-token');
    
    // Special highlight for frequency-based tokens (learned from text)
    const freqIndicator = token.isFrequencyBased ? ' 🔥' : '';
    
    div.innerHTML = `
      <div class="token-text">${escapeHtml(token.text)}${freqIndicator}</div>
      <div class="token-info">
        f=${Math.round(wf.frequency)}Hz φ=${wf.phase.toFixed(2)} E=${wf.energy.toFixed(3)}
      </div>
    `;
    
    const tooltipText = token.isFrequencyBased 
      ? `Token: "${token.text}" [FREQUENCY-LEARNED] | Freq: ${Math.round(wf.frequency)} Hz | Phase: ${wf.phase.toFixed(3)} rad | Energy: ${wf.energy.toFixed(4)} | Length: ${token.length} chars | Learned from text bigram frequency`
      : `Token: "${token.text}" | Freq: ${Math.round(wf.frequency)} Hz | Phase: ${wf.phase.toFixed(3)} rad | Energy: ${wf.energy.toFixed(4)} | Length: ${token.length} chars`;
    
    div.title = tooltipText;
    
    tokensDisplay.appendChild(div);
  });
  
  // Visualize waveforms
  visualizeWaveforms(tokens, 'waveformCanvas');
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// Initialize on load
window.addEventListener('DOMContentLoaded', () => {
  tokenizeText();
});
