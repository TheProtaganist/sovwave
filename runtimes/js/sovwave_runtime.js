/**
 * sovwave_runtime.js — Sovwave MKV Model Runtime (Node.js / Browser)
 *
 * Loads a trained Sovwave .mkv model and runs inference with zero Julia dependency.
 * Node.js: requires ffmpeg in PATH.
 * Browser: call SovwaveModel.fromArrayBuffer(mkvBuffer, metaYamlString).
 *
 * Usage (Node.js):
 *   const { SovwaveModel } = require('./sovwave_runtime');
 *   const model = await SovwaveModel.load('my_model.mkv', 'my_model_meta.yaml');
 *   const outputs = await model.predict([[0.1, 0.2, 0.3, 0.4]]);
 *   console.log(outputs);
 *
 * Usage (Browser, after fetching the MKV as ArrayBuffer):
 *   const model = await SovwaveModel.fromArrayBuffer(mkvBuffer, metaYamlText);
 *   const outputs = await model.predict([[0.1, 0.2, 0.3, 0.4]]);
 */

'use strict';

// ── Wave forward pass ─────────────────────────────────────────────────────────

function waveForward(layer, inputValues, t = 0.0) {
    const { nodes, embedDim, amplitudes, phases, frequencies,
            fractalScales, fractalDims, waveSpeeds, omega } = layer;
    const inLen = inputValues.length;
    const output = new Array(nodes).fill(0.0);

    for (let i = 0; i < nodes; i++) {
        const beta    = fractalScales[i];
        const df      = fractalDims[i];
        const vSpd    = waveSpeeds[i];
        const fracEnv = df / 1.5;
        let nodeSum   = 0.0;

        for (let j = 0; j < embedDim; j++) {
            const inVal = j < inLen ? inputValues[j] : 0.5;
            const amp   = amplitudes[i][j];
            const ph    = phases[i][j];
            const freq  = frequencies[i][j];

            // Speed-aware phase: -1 = unlimited
            const xEff  = vSpd === -1.0 ? inVal : inVal / Math.max(vSpd, 1e-12);
            const angle = omega * 0.001 * freq * xEff + ph - t;
            nodeSum    += amp * Math.sin(angle);
        }

        output[i] = (nodeSum / Math.sqrt(embedDim)) * beta * fracEnv;
    }
    return output;
}

// ── Pixel → weight decoding ───────────────────────────────────────────────────

function decodePixels(rawBytes, w, h, numLayers, nodes, embedDim, omega, betaS = 1.618033988749895) {
    const bytesPerFrame = w * h * 3;
    const layers = [];
    const TWO_PI = 2 * Math.PI;

    for (let l = 0; l < numLayers; l++) {
        const frameOff = l * bytesPerFrame;
        const amps  = Array.from({ length: nodes }, () => new Array(embedDim).fill(0.5));
        const phs   = Array.from({ length: nodes }, () => new Array(embedDim).fill(0.0));
        const freqs = Array.from({ length: nodes }, () => new Array(embedDim).fill(1.0));
        const fscales = new Array(nodes).fill(betaS);
        const fdims   = new Array(nodes).fill(1.5);
        const wspeeds = new Array(nodes).fill(1.0);

        for (let r = 0; r < nodes; r++) {
            for (let c = 0; c < embedDim; c++) {
                const px = frameOff + (r * w + c) * 3;
                if (px + 2 < rawBytes.length) {
                    amps[r][c]  = (rawBytes[px]     / 255.0) * 2.0;
                    phs[r][c]   = (rawBytes[px + 1] / 255.0) * TWO_PI;
                    freqs[r][c] = Math.max(0.1, (rawBytes[px + 2] / 255.0) * 4.0);
                }
            }
        }

        layers.push({ nodes, embedDim, amplitudes: amps, phases: phs,
                      frequencies: freqs, fractalScales: fscales,
                      fractalDims: fdims, waveSpeeds: wspeeds, omega });
    }
    return layers;
}

// ── YAML parser (minimal, no dependencies) ─────────────────────────────────

function parseMetaYaml(text) {
    const cfg = { nodes: 64, embedDim: 64, numLayers: 3, omega: 432.0, tFrames: 3, betaS: 1.618033988749895 };
    if (!text) return cfg;
    const lines = text.split('\n');
    let inModel = false;
    for (const line of lines) {
        if (/^model:/.test(line)) { inModel = true; continue; }
        if (inModel && /^\S/.test(line)) { inModel = false; }
        if (inModel) {
            const m = line.match(/^\s+(\w+):\s*(.+)/);
            if (!m) continue;
            const [, key, val] = m;
            if (key === 'nodes')      cfg.nodes      = parseInt(val);
            if (key === 'embed_dims') cfg.embedDim   = parseInt(val);
            if (key === 'layers')     cfg.numLayers  = parseInt(val);
            if (key === 'omega')      cfg.omega       = parseFloat(val);
            if (key === 't_frames')   cfg.tFrames     = parseInt(val);
            if (key === 'beta_s')     cfg.betaS       = parseFloat(val);
        }
    }
    return cfg;
}

// ── Model class ───────────────────────────────────────────────────────────────

class SovwaveModel {
    constructor(layers, nodes, embedDim, omega = 432.0, tFrames = 3) {
        this.layers   = layers;
        this.nodes    = nodes;
        this.embedDim = embedDim;
        this.omega    = omega;
        this.tFrames  = tFrames;
    }

    /**
     * Load model from MKV file (Node.js).
     * @param {string} mkvPath     - path to .mkv model file
     * @param {string} [metaPath]  - path to _meta.yaml (auto-detected if omitted)
     * @returns {Promise<SovwaveModel>}
     */
    static async load(mkvPath, metaPath) {
        if (typeof require === 'undefined') {
            throw new Error('SovwaveModel.load() is Node.js only. Use fromArrayBuffer() in the browser.');
        }
        const fs = require('fs');
        const path = require('path');

        // Auto-detect meta
        const actualMeta = metaPath || mkvPath.replace(/\.mkv$/i, '_meta.yaml');
        let metaText = '';
        if (fs.existsSync(actualMeta)) {
            metaText = fs.readFileSync(actualMeta, 'utf8');
        }
        const cfg = parseMetaYaml(metaText);

        const w = cfg.embedDim % 2 === 0 ? cfg.embedDim : cfg.embedDim + 1;
        const h = cfg.nodes    % 2 === 0 ? cfg.nodes    : cfg.nodes    + 1;

        const rawBytes = await _extractStreamNode(mkvPath, w, h);
        const layers   = decodePixels(rawBytes, w, h, cfg.numLayers,
                                      cfg.nodes, cfg.embedDim, cfg.omega, cfg.betaS);
        return new SovwaveModel(layers, cfg.nodes, cfg.embedDim, cfg.omega, cfg.tFrames);
    }

    /**
     * Load model from ArrayBuffer (Browser).
     * @param {ArrayBuffer} mkvBuffer
     * @param {string} [metaYamlText]
     * @returns {Promise<SovwaveModel>}
     */
    static async fromArrayBuffer(mkvBuffer, metaYamlText = '') {
        // In the browser, we use ffmpeg.wasm or a server-side decode endpoint.
        // For simplicity, this implementation expects pre-extracted raw RGB bytes
        // passed as a second ArrayBuffer, or calls a /sovwave/decode endpoint.
        const cfg    = parseMetaYaml(metaYamlText);
        const w      = cfg.embedDim % 2 === 0 ? cfg.embedDim : cfg.embedDim + 1;
        const h      = cfg.nodes    % 2 === 0 ? cfg.nodes    : cfg.nodes    + 1;
        // Treat entire buffer as pre-decoded raw RGB (stream was pre-extracted)
        const raw    = new Uint8Array(mkvBuffer);
        const layers = decodePixels(raw, w, h, cfg.numLayers,
                                    cfg.nodes, cfg.embedDim, cfg.omega, cfg.betaS);
        return new SovwaveModel(layers, cfg.nodes, cfg.embedDim, cfg.omega, cfg.tFrames);
    }

    /** Forward pass through all layers */
    forward(inputValues, t = 0.0) {
        let current = [...inputValues];
        for (const layer of this.layers) {
            const accum = new Array(layer.nodes).fill(0.0);
            for (let frame = 0; frame < this.tFrames; frame++) {
                const tOff = t + 2 * Math.PI * frame / (this.omega + 1e-12);
                const out  = waveForward(layer, current, tOff);
                for (let k = 0; k < layer.nodes; k++) accum[k] += out[k];
            }
            const scale = 1.0 / Math.sqrt(this.tFrames);
            current = accum.map(v => v * scale);
        }
        return current;
    }

    /** Batch prediction */
    predict(inputs, t = 0.0) {
        return inputs.map(inp => this.forward(inp, t));
    }
}

// ── Node.js ffmpeg stream extractor ──────────────────────────────────────────

function _extractStreamNode(mkvPath, w, h) {
    const { execFileSync } = require('child_process');
    // Try data stream (0:v:1) first
    try {
        const buf = execFileSync('ffmpeg', [
            '-loglevel', 'error', '-i', mkvPath,
            '-map', '0:v:1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'
        ]);
        if (buf && buf.length > 0) return new Uint8Array(buf);
    } catch (_) { /* fallthrough */ }

    // Fallback: visual stream with downscale
    const buf = execFileSync('ffmpeg', [
        '-loglevel', 'error', '-i', mkvPath,
        '-map', '0:v:0', '-vf', `scale=${w}:${h}:flags=neighbor`,
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'
    ]);
    return new Uint8Array(buf);
}

// ── Module export ─────────────────────────────────────────────────────────────

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SovwaveModel, waveForward, decodePixels };
}
