/**
 * sovwave_runtime.js — Sovwave MKV Model Runtime (Node.js / Browser)
 *
 * Loads a trained Sovwave .mkv model and runs inference with zero Julia dependency.
 * Supports:
 * 1. Physical zero-DC nodal wave superposition:
 *      E_i = sum_j (A_{i, j} * cos(phi_{i, j}) * x_j)
 *      psi_i = sin(E_i)
 *      hat_psi = psi / ||psi||_2
 * 2. Lossless MKV container attachment extraction (waveml_model.bin)
 * 3. Optical RGB video frame decoding (Centroid Kernel Sampling fallback)
 */

'use strict';

// ── Wave forward pass ─────────────────────────────────────────────────────────

function waveForward(layer, inputValues, t = 0.0) {
    const { nodes, embedDim, amplitudes, phases } = layer;
    const inLen = inputValues.length;
    const output = new Array(nodes).fill(0.0);

    for (let i = 0; i < nodes; i++) {
        let Ei = 0.0;
        const rowAmps = amplitudes[i];
        const rowPhs  = phases[i];
        const limit = Math.min(inLen, embedDim);
        for (let j = 0; j < limit; j++) {
            Ei += rowAmps[j] * Math.cos(rowPhs[j]) * inputValues[j];
        }
        output[i] = Math.sin(Ei);
    }

    // Unit L2 sphere projection
    let sqSum = 0.0;
    for (let i = 0; i < nodes; i++) sqSum += output[i] * output[i];
    const nrm = Math.sqrt(sqSum);
    if (nrm > 1e-6) {
        for (let i = 0; i < nodes; i++) output[i] /= nrm;
    }
    return output;
}

// ── Pixel → weight decoding (Optical Fallback) ─────────────────────────────────

function decodePixels(rawBytes, w, h, numLayers, nodes, embedDim, omega, betaS = 1.618033988749895) {
    const bytesPerFrame = w * h * 3;
    const layers = [];
    const TWO_PI = 2 * Math.PI;
    const bw = Math.max(1, Math.floor(w / embedDim));
    const bh = Math.max(1, Math.floor(h / nodes));
    const halfK = Math.max(1, Math.floor(bw / 4));

    for (let l = 0; l < numLayers; l++) {
        const frameOff = l * bytesPerFrame;
        const amps  = Array.from({ length: nodes }, () => new Array(embedDim).fill(0.5));
        const phs   = Array.from({ length: nodes }, () => new Array(embedDim).fill(0.0));
        const freqs = Array.from({ length: nodes }, () => new Array(embedDim).fill(1.0));
        const fscales = new Array(nodes).fill(betaS);
        const fdims   = new Array(nodes).fill(1.5);
        const wspeeds = new Array(nodes).fill(1.0);

        for (let r = 0; r < nodes; r++) {
            const yc = Math.round((r + 0.5) * bh);
            for (let c = 0; c < embedDim; c++) {
                const xc = Math.round((c + 0.5) * bw);
                let rAcc = 0.0, gAcc = 0.0, bAcc = 0.0, count = 0;
                for (let dy = -halfK; dy <= halfK; dy++) {
                    for (let dx = -halfK; dx <= halfK; dx++) {
                        const pxX = Math.min(w - 1, Math.max(0, xc + dx));
                        const pyY = Math.min(h - 1, Math.max(0, yc + dy));
                        const idx = frameOff + (pyY * w + pxX) * 3;
                        if (idx + 2 < rawBytes.length) {
                            rAcc += rawBytes[idx];
                            gAcc += rawBytes[idx + 1];
                            bAcc += rawBytes[idx + 2];
                            count++;
                        }
                    }
                }
                if (count > 0) {
                    let a = (rAcc / count / 255.0) * 2.0;
                    let p = (gAcc / count / 255.0) * TWO_PI;
                    if (a < 0.0) {
                        a = -a;
                        p = (p + Math.PI) % TWO_PI;
                    }
                    amps[r][c]  = a;
                    phs[r][c]   = p;
                    freqs[r][c] = Math.max(0.1, (bAcc / count / 255.0) * 4.0);
                }
            }
        }

        layers.push({ nodes, embedDim, amplitudes: amps, phases: phs,
                      frequencies: freqs, fractalScales: fscales,
                      fractalDims: fdims, waveSpeeds: wspeeds, omega });
    }
    return layers;
}

// ── Mini YAML parser ──────────────────────────────────────────────────────────

function parseMetaYaml(text) {
    const cfg = {
        nodes:      64,
        embedDim:   64,
        numLayers:  3,
        omega:      432.0,
        tFrames:    1,
        betaS:      1.618033988749895,
    };
    if (!text) return cfg;

    const lines = text.split('\n');
    let inModel = false;
    for (const rawLine of lines) {
        const line = rawLine.trim();
        if (line.startsWith('model:')) { inModel = true; continue; }
        if (inModel && line && !rawLine.startsWith(' ') && !rawLine.startsWith('\t')) {
            inModel = false;
        }
        if (inModel) {
            const [key, ...rest] = line.split(':');
            const val = rest.join(':').trim();
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
    constructor(layers, nodes, embedDim, omega = 432.0, tFrames = 1) {
        this.layers   = layers;
        this.nodes    = nodes;
        this.embedDim = embedDim;
        this.omega    = omega;
        this.tFrames  = tFrames;
    }

    /**
     * Load model from MKV file (Node.js).
     */
    static async load(mkvPath, metaPath) {
        if (typeof require === 'undefined') {
            throw new Error('SovwaveModel.load() is Node.js only. Use fromArrayBuffer() in the browser.');
        }
        const fs = require('fs');
        const os = require('os');
        const path = require('path');
        const { execFileSync } = require('child_process');

        // 1. Try lossless Matroska container attachment extraction
        if (mkvPath.toLowerCase().endsWith('.mkv')) {
            const tmpBin = path.join(os.tmpdir(), `sw_weights_${Date.now()}_${Math.random().toString(36).slice(2)}.dat`);
            try {
                execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-dump_attachment:t:0', tmpBin, '-i', mkvPath, '-f', 'null', '-']);
                if (fs.existsSync(tmpBin) && fs.statSync(tmpBin).size >= 36) {
                    const buf = fs.readFileSync(tmpBin);
                    fs.unlinkSync(tmpBin);
                    if (buf.toString('utf8', 0, 4) === 'SOVW') {
                        const numLayers = buf.readInt32LE(8);
                        const nodes     = buf.readInt32LE(12);
                        const embedDim  = buf.readInt32LE(16);
                        const omega     = buf.readDoubleLE(20);
                        let off = 36;
                        const layers = [];
                        for (let l = 0; l < numLayers; l++) {
                            const floatsPerMat = nodes * embedDim;
                            const amps = Array.from({ length: nodes }, () => new Array(embedDim).fill(0.0));
                            const phs  = Array.from({ length: nodes }, () => new Array(embedDim).fill(0.0));
                            const freqs = Array.from({ length: nodes }, () => new Array(embedDim).fill(1.0));

                            const ampVals = [];
                            for (let k = 0; k < floatsPerMat; k++) {
                                ampVals.push(buf.readFloatLE(off));
                                off += 4;
                            }
                            const phVals = [];
                            for (let k = 0; k < floatsPerMat; k++) {
                                phVals.push(buf.readFloatLE(off));
                                off += 4;
                            }
                            const freqVals = [];
                            for (let k = 0; k < floatsPerMat; k++) {
                                freqVals.push(buf.readFloatLE(off));
                                off += 4;
                            }
                            const scales = [];
                            for (let k = 0; k < nodes; k++) {
                                scales.push(buf.readFloatLE(off));
                                off += 4;
                            }

                            let idx = 0;
                            for (let c = 0; c < embedDim; c++) {
                                for (let r = 0; r < nodes; r++) {
                                    let a = ampVals[idx];
                                    let p = phVals[idx];
                                    if (a < 0.0) {
                                        a = -a;
                                        p = (p + Math.PI) % (2.0 * Math.PI);
                                    }
                                    amps[r][c] = a;
                                    phs[r][c]  = p;
                                    freqs[r][c] = freqVals[idx];
                                    idx++;
                                }
                            }
                            layers.push({ nodes, embedDim, amplitudes: amps, phases: phs, frequencies: freqs, fractalScales: scales, omega });
                        }
                        return new SovwaveModel(layers, nodes, embedDim, omega, 1);
                    }
                }
            } catch (e) {
                if (fs.existsSync(tmpBin)) try { fs.unlinkSync(tmpBin); } catch (_) {}
            }
        }

        // 2. Optical video frame decoding fallback
        const actualMeta = metaPath || mkvPath.replace(/\.mkv$/i, '_meta.yaml');
        let metaText = '';
        if (fs.existsSync(actualMeta)) {
            metaText = fs.readFileSync(actualMeta, 'utf8');
        }
        const cfg = parseMetaYaml(metaText);

        const w = 640;
        const h = 480;

        const rawBytes = _extractStreamNode(mkvPath);
        const layers   = decodePixels(rawBytes, w, h, cfg.numLayers,
                                      cfg.nodes, cfg.embedDim, cfg.omega, cfg.betaS);
        return new SovwaveModel(layers, cfg.nodes, cfg.embedDim, cfg.omega, cfg.tFrames);
    }

    /**
     * Load model from ArrayBuffer (Browser).
     */
    static async fromArrayBuffer(mkvBuffer, metaYamlText = '') {
        const cfg    = parseMetaYaml(metaYamlText);
        const w      = 640;
        const h      = 480;
        const raw    = new Uint8Array(mkvBuffer);
        const layers = decodePixels(raw, w, h, cfg.numLayers,
                                    cfg.nodes, cfg.embedDim, cfg.omega, cfg.betaS);
        return new SovwaveModel(layers, cfg.nodes, cfg.embedDim, cfg.omega, cfg.tFrames);
    }

    /** Forward pass through all layers */
    forward(inputValues, t = 0.0) {
        let current = [...inputValues];
        for (const layer of this.layers) {
            current = waveForward(layer, current, t);
        }
        return current;
    }

    /** Batch prediction */
    predict(inputs, t = 0.0) {
        return inputs.map(inp => this.forward(inp, t));
    }
}

// ── Node.js ffmpeg stream extractor ──────────────────────────────────────────

function _extractStreamNode(mkvPath) {
    const { execFileSync } = require('child_process');
    const buf = execFileSync('ffmpeg', [
        '-loglevel', 'error', '-i', mkvPath,
        '-map', '0:v:0',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'
    ]);
    return new Uint8Array(buf);
}

// ── Module export ─────────────────────────────────────────────────────────────

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SovwaveModel, waveForward, decodePixels };
}
