/**
 * sovwave_runtime.hpp — Sovwave MKV Model Runtime (C++17, header-only)
 *
 * Loads a trained Sovwave .mkv model and runs inference with zero Julia dependency.
 * Requires: C++17 compiler, ffmpeg in PATH.
 *
 * Usage:
 *   #include "sovwave_runtime.hpp"
 *   auto model = sovwave::SovwaveModel::load("my_model.mkv", "my_model_meta.yaml");
 *   auto output = model.predict({{0.1, 0.2, 0.3, 0.4}});
 */

#pragma once

#include <cmath>
#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>

namespace sovwave {

// ── Layer parameter struct ────────────────────────────────────────────────────

struct WaveLayerParams {
    int nodes;
    int embed_dim;
    std::vector<std::vector<double>> amplitudes;  // [nodes][embed_dim]
    std::vector<std::vector<double>> phases;
    std::vector<std::vector<double>> frequencies;
    std::vector<double> fractal_scales;
    std::vector<double> fractal_dims;   // Hausdorff D_f ∈ [1.0, 3.0]
    std::vector<double> wave_speeds;    // -1.0 = unlimited
    double omega;

    WaveLayerParams(int n, int d, double omega_)
        : nodes(n), embed_dim(d), omega(omega_),
          amplitudes(n, std::vector<double>(d, 0.5)),
          phases    (n, std::vector<double>(d, 0.0)),
          frequencies(n, std::vector<double>(d, 1.0)),
          fractal_scales(n, 1.618033988749895),
          fractal_dims  (n, 1.5),
          wave_speeds   (n, 1.0) {}
};

// ── Forward pass ──────────────────────────────────────────────────────────────

inline std::vector<double> wave_forward(
    const WaveLayerParams& layer,
    const std::vector<double>& input_values,
    double t = 0.0
) {
    int n   = layer.nodes;
    int d   = layer.embed_dim;
    int ilen = static_cast<int>(input_values.size());
    std::vector<double> output(n, 0.0);

    for (int i = 0; i < n; ++i) {
        double beta    = layer.fractal_scales[i];
        double d_f     = layer.fractal_dims[i];
        double v_spd   = layer.wave_speeds[i];
        double frac_env = d_f / 1.5;
        double node_sum = 0.0;

        for (int j = 0; j < d; ++j) {
            double in_val = (j < ilen) ? input_values[j] : 0.5;
            double amp    = layer.amplitudes[i][j];
            double ph     = layer.phases[i][j];
            double freq   = layer.frequencies[i][j];

            // Speed-aware phase: -1.0 = unlimited (no speed divisor)
            double x_eff = (v_spd == -1.0) ? in_val : in_val / std::max(v_spd, 1e-12);
            double angle = layer.omega * 0.001 * freq * x_eff + ph - t;
            node_sum += amp * std::sin(angle);
        }

        output[i] = (node_sum / std::sqrt(static_cast<double>(d))) * beta * frac_env;
    }
    return output;
}

// ── Pixel → weight decoding ───────────────────────────────────────────────────

inline std::vector<WaveLayerParams> decode_pixels(
    const std::vector<uint8_t>& raw_bytes,
    int w, int h, int num_layers,
    int nodes, int embed_dim,
    double omega, double beta_s = 1.618033988749895
) {
    std::vector<WaveLayerParams> layers;
    layers.reserve(num_layers);
    int bytes_per_frame = w * h * 3;

    for (int l = 0; l < num_layers; ++l) {
        WaveLayerParams lp(nodes, embed_dim, omega);
        std::fill(lp.fractal_scales.begin(), lp.fractal_scales.end(), beta_s);
        int frame_off = l * bytes_per_frame;

        for (int r = 0; r < nodes; ++r) {
            for (int c = 0; c < embed_dim; ++c) {
                int px = frame_off + (r * w + c) * 3;
                if (px + 2 < static_cast<int>(raw_bytes.size())) {
                    lp.amplitudes[r][c]   = (raw_bytes[px]     / 255.0) * 2.0;
                    lp.phases[r][c]       = (raw_bytes[px + 1] / 255.0) * (2.0 * M_PI);
                    lp.frequencies[r][c]  = std::max(0.1, (raw_bytes[px + 2] / 255.0) * 4.0);
                }
            }
        }
        layers.push_back(std::move(lp));
    }
    return layers;
}

// ── ffmpeg pipe helper ────────────────────────────────────────────────────────

inline std::vector<uint8_t> read_pipe(const std::string& cmd) {
    std::vector<uint8_t> buf;
#ifdef _WIN32
    FILE* pipe = _popen(cmd.c_str(), "rb");
#else
    FILE* pipe = popen(cmd.c_str(), "r");
#endif
    if (!pipe) throw std::runtime_error("Failed to open ffmpeg pipe");
    uint8_t chunk[4096];
    while (true) {
        size_t n = fread(chunk, 1, sizeof(chunk), pipe);
        if (n == 0) break;
        buf.insert(buf.end(), chunk, chunk + n);
    }
#ifdef _WIN32
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    return buf;
}

// ── Model class ───────────────────────────────────────────────────────────────

class SovwaveModel {
public:
    std::vector<WaveLayerParams> layers;
    int nodes, embed_dim, t_frames;
    double omega;

    SovwaveModel(std::vector<WaveLayerParams> l, int n, int d, double om, int tf)
        : layers(std::move(l)), nodes(n), embed_dim(d), omega(om), t_frames(tf) {}

    /**
     * Load a SovwaveModel from an MKV video file.
     * @param mkv_path  Path to the .mkv model file
     * @param meta_path Path to the _meta.yaml config (empty string = auto-detect)
     */
    static SovwaveModel load(const std::string& mkv_path, const std::string& meta_path = "") {
        int    nodes      = 64;
        int    embed_dim  = 64;
        int    num_layers = 3;
        double omega      = 432.0;
        int    t_frames   = 3;
        double beta_s     = 1.618033988749895;

        // Parse YAML manually (no dependency) — simple key: value search
        std::string actual_meta = meta_path;
        if (actual_meta.empty()) {
            // Auto-detect: replace .mkv with _meta.yaml
            actual_meta = mkv_path;
            auto pos = actual_meta.rfind(".mkv");
            if (pos != std::string::npos) actual_meta.replace(pos, 4, "_meta.yaml");
        }
        if (!actual_meta.empty()) {
            std::ifstream mf(actual_meta);
            if (mf.is_open()) {
                std::string line;
                bool in_model = false;
                while (std::getline(mf, line)) {
                    if (line.find("model:") != std::string::npos) { in_model = true; continue; }
                    if (in_model && !line.empty() && line[0] != ' ' && line[0] != '\t') in_model = false;
                    auto parse_int = [&](const std::string& key) -> int {
                        auto p = line.find(key);
                        if (p != std::string::npos) return std::stoi(line.substr(p + key.size()));
                        return -1;
                    };
                    auto parse_dbl = [&](const std::string& key) -> double {
                        auto p = line.find(key);
                        if (p != std::string::npos) return std::stod(line.substr(p + key.size()));
                        return -1.0;
                    };
                    if (in_model) {
                        int v; double d;
                        if ((v = parse_int("nodes: "))      > 0) nodes      = v;
                        if ((v = parse_int("embed_dims: ")) > 0) embed_dim  = v;
                        if ((v = parse_int("layers: "))     > 0) num_layers = v;
                        if ((d = parse_dbl("omega: "))      > 0) omega      = d;
                        if ((d = parse_dbl("beta_s: "))     > 0) beta_s     = d;
                        if ((v = parse_int("t_frames: "))   > 0) t_frames   = v;
                    }
                }
            }
        }

        int w = (embed_dim % 2 == 0) ? embed_dim : embed_dim + 1;
        int h = (nodes     % 2 == 0) ? nodes     : nodes     + 1;

        // Extract FFV1 data stream via ffmpeg
        auto raw = _extract_stream(mkv_path, w, h);
        auto lps = decode_pixels(raw, w, h, num_layers, nodes, embed_dim, omega, beta_s);
        return SovwaveModel(std::move(lps), nodes, embed_dim, omega, t_frames);
    }

    /** Forward pass through all layers */
    std::vector<double> forward(const std::vector<double>& input, double t = 0.0) const {
        std::vector<double> current = input;
        for (const auto& layer : layers) {
            std::vector<double> accum(layer.nodes, 0.0);
            for (int frame = 0; frame < t_frames; ++frame) {
                double t_off = t + 2.0 * M_PI * frame / (omega + 1e-12);
                auto out = wave_forward(layer, current, t_off);
                for (int k = 0; k < layer.nodes; ++k) accum[k] += out[k];
            }
            double scale = 1.0 / std::sqrt(static_cast<double>(t_frames));
            for (auto& v : accum) v *= scale;
            current = std::move(accum);
        }
        return current;
    }

    /** Batch prediction */
    std::vector<std::vector<double>> predict(
        const std::vector<std::vector<double>>& inputs,
        double t = 0.0
    ) const {
        std::vector<std::vector<double>> results;
        results.reserve(inputs.size());
        for (const auto& inp : inputs) results.push_back(forward(inp, t));
        return results;
    }

private:
    static std::vector<uint8_t> _extract_stream(const std::string& path, int w, int h) {
        // Try data stream (0:v:1) first
        std::string cmd1 = "ffmpeg -loglevel error -i \"" + path +
                           "\" -map 0:v:1 -f rawvideo -pix_fmt rgb24 - 2>/dev/null";
        auto raw = read_pipe(cmd1);
        if (!raw.empty()) return raw;

        // Fallback: visual stream with downscale
        std::string cmd2 = "ffmpeg -loglevel error -i \"" + path +
                           "\" -map 0:v:0 -vf scale=" + std::to_string(w) + ":" + std::to_string(h) +
                           ":flags=neighbor -f rawvideo -pix_fmt rgb24 -";
        return read_pipe(cmd2);
    }
};

} // namespace sovwave
