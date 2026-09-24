/**
 * sovwave_runtime.hpp — Sovwave MKV Model Runtime (C++17, header-only)
 *
 * Loads a trained Sovwave .mkv model and runs inference with zero Julia dependency.
 * Supports:
 * 1. Physical zero-DC nodal wave superposition:
 *      E_i = sum_j (A_{i, j} * cos(phi_{i, j}) * x_j)
 *      psi_i = sin(E_i)
 *      hat_psi = psi / ||psi||_2
 * 2. Lossless MKV container attachment extraction (waveml_model.bin)
 * 3. Optical RGB video frame decoding (Centroid Kernel Sampling fallback)
 *
 * Usage:
 *   #include "sovwave_runtime.hpp"
 *   auto model = sovwave::SovwaveModel::load("spark_model.mkv");
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
#include <unistd.h>
#include <cstdlib>

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
        double E_i = 0.0;
        const auto& row_amps = layer.amplitudes[i];
        const auto& row_phs  = layer.phases[i];
        int limit = std::min(ilen, d);
        for (int j = 0; j < limit; ++j) {
            E_i += row_amps[j] * std::cos(row_phs[j]) * input_values[j];
        }
        output[i] = std::sin(E_i);
    }

    // Unit L2 sphere projection
    double sq_sum = 0.0;
    for (int i = 0; i < n; ++i) sq_sum += output[i] * output[i];
    double nrm = std::sqrt(sq_sum);
    if (nrm > 1e-6) {
        for (int i = 0; i < n; ++i) output[i] /= nrm;
    }
    return output;
}

// ── ffmpeg pipe helper ────────────────────────────────────────────────────────

inline std::vector<uint8_t> read_pipe(const std::string& cmd) {
    std::vector<uint8_t> buf;
#ifdef _WIN32
    FILE* pipe = _popen(cmd.c_str(), "rb");
#else
    FILE* pipe = popen(cmd.c_str(), "r");
#endif
    if (!pipe) return buf;
    uint8_t chunk[4096];
    size_t n;
    while ((n = std::fread(chunk, 1, sizeof(chunk), pipe)) > 0) {
        buf.insert(buf.end(), chunk, chunk + n);
    }
#ifdef _WIN32
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    return buf;
}

// ── Pixel → weight decoding (Optical Fallback) ─────────────────────────────────

inline std::vector<WaveLayerParams> decode_pixels(
    const std::vector<uint8_t>& raw_bytes,
    int w, int h, int num_layers,
    int nodes, int embed_dim,
    double omega, double beta_s = 1.618033988749895
) {
    std::vector<WaveLayerParams> layers;
    layers.reserve(num_layers);
    int bytes_per_frame = w * h * 3;
    int bw = std::max(1, w / embed_dim);
    int bh = std::max(1, h / nodes);
    int half_k = std::max(1, bw / 4);

    for (int l = 0; l < num_layers; ++l) {
        WaveLayerParams lp(nodes, embed_dim, omega);
        std::fill(lp.fractal_scales.begin(), lp.fractal_scales.end(), beta_s);
        int frame_off = l * bytes_per_frame;

        for (int r = 0; r < nodes; ++r) {
            int yc = static_cast<int>(std::round((r + 0.5) * bh));
            for (int c = 0; c < embed_dim; ++c) {
                int xc = static_cast<int>(std::round((c + 0.5) * bw));
                double r_acc = 0.0, g_acc = 0.0, b_acc = 0.0;
                int count = 0;
                for (int dy = -half_k; dy <= half_k; ++dy) {
                    for (int dx = -half_k; dx <= half_k; ++dx) {
                        int px = std::min(w - 1, std::max(0, xc + dx));
                        int py = std::min(h - 1, std::max(0, yc + dy));
                        int idx = frame_off + (py * w + px) * 3;
                        if (idx + 2 < static_cast<int>(raw_bytes.size())) {
                            r_acc += raw_bytes[idx];
                            g_acc += raw_bytes[idx + 1];
                            b_acc += raw_bytes[idx + 2];
                            count++;
                        }
                    }
                }
                if (count > 0) {
                    double a = (r_acc / count / 255.0) * 2.0;
                    double p = (g_acc / count / 255.0) * (2.0 * M_PI);
                    if (a < 0.0) {
                        a = -a;
                        p = std::fmod(p + M_PI, 2.0 * M_PI);
                    }
                    lp.amplitudes[r][c]  = a;
                    lp.phases[r][c]      = p;
                    lp.frequencies[r][c] = std::max(0.1, (b_acc / count / 255.0) * 4.0);
                } else {
                    lp.amplitudes[r][c]  = 0.5;
                    lp.phases[r][c]      = 0.0;
                    lp.frequencies[r][c] = 1.0;
                }
            }
        }
        layers.push_back(std::move(lp));
    }
    return layers;
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
        // 1. Try lossless Matroska container attachment extraction first
        if (mkv_path.size() >= 4 && mkv_path.substr(mkv_path.size() - 4) == ".mkv") {
            char tmp_template[] = "/tmp/sw_weights_XXXXXX";
            int fd = mkstemp(tmp_template);
            if (fd != -1) {
                close(fd);
                std::string tmp_bin(tmp_template);
                std::string dump_cmd = "ffmpeg -y -loglevel error -dump_attachment:t:0 \"" + tmp_bin + "\" -i \"" + mkv_path + "\" -f null -";
                int ret = std::system(dump_cmd.c_str());
                std::ifstream bf(tmp_bin, std::ios::binary);
                if (ret == 0 && bf.is_open()) {
                    bf.seekg(0, std::ios::end);
                    size_t sz = bf.tellg();
                    bf.seekg(0, std::ios::beg);
                    if (sz >= 36) {
                        std::vector<uint8_t> raw(sz);
                        bf.read(reinterpret_cast<char*>(raw.data()), sz);
                        bf.close();
                        std::remove(tmp_bin.c_str());

                        if (raw[0] == 'S' && raw[1] == 'O' && raw[2] == 'V' && raw[3] == 'W') {
                            int num_layers_att = *reinterpret_cast<const int32_t*>(&raw[8]);
                            int nodes_att      = *reinterpret_cast<const int32_t*>(&raw[12]);
                            int embed_dim_att  = *reinterpret_cast<const int32_t*>(&raw[16]);
                            double omega_att   = *reinterpret_cast<const double*>(&raw[20]);
                            size_t off = 36;
                            std::vector<WaveLayerParams> att_layers;
                            att_layers.reserve(num_layers_att);

                            for (int l = 0; l < num_layers_att; ++l) {
                                WaveLayerParams lp(nodes_att, embed_dim_att, omega_att);
                                size_t floats_per_mat = nodes_att * embed_dim_att;
                                const float* amp_ptr  = reinterpret_cast<const float*>(&raw[off]);
                                off += floats_per_mat * sizeof(float);
                                const float* ph_ptr   = reinterpret_cast<const float*>(&raw[off]);
                                off += floats_per_mat * sizeof(float);
                                const float* freq_ptr = reinterpret_cast<const float*>(&raw[off]);
                                off += floats_per_mat * sizeof(float);
                                const float* sc_ptr   = reinterpret_cast<const float*>(&raw[off]);
                                off += nodes_att * sizeof(float);

                                size_t idx = 0;
                                for (int c = 0; c < embed_dim_att; ++c) {
                                    for (int r = 0; r < nodes_att; ++r) {
                                        double a = amp_ptr[idx];
                                        double p = ph_ptr[idx];
                                        if (a < 0.0) {
                                            a = -a;
                                            p = std::fmod(p + M_PI, 2.0 * M_PI);
                                        }
                                        lp.amplitudes[r][c]  = a;
                                        lp.phases[r][c]      = p;
                                        lp.frequencies[r][c] = freq_ptr[idx];
                                        idx++;
                                    }
                                }
                                for (int r = 0; r < nodes_att; ++r) {
                                    lp.fractal_scales[r] = sc_ptr[r];
                                }
                                att_layers.push_back(std::move(lp));
                            }
                            return SovwaveModel(std::move(att_layers), nodes_att, embed_dim_att, omega_att, 1);
                        }
                    }
                    if (bf.is_open()) bf.close();
                }
                std::remove(tmp_bin.c_str());
            }
        }

        // 2. Optical video frame decoding fallback
        int    nodes      = 64;
        int    embed_dim  = 64;
        int    num_layers = 3;
        double omega      = 432.0;
        int    t_frames   = 1;
        double beta_s     = 1.618033988749895;

        std::string actual_meta = meta_path;
        if (actual_meta.empty()) {
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

        int w = 640;
        int h = 480;

        auto raw = _extract_stream(mkv_path);
        auto lps = decode_pixels(raw, w, h, num_layers, nodes, embed_dim, omega, beta_s);
        return SovwaveModel(std::move(lps), nodes, embed_dim, omega, t_frames);
    }

    /** Forward pass through all layers */
    std::vector<double> forward(const std::vector<double>& input, double t = 0.0) const {
        std::vector<double> current = input;
        for (const auto& layer : layers) {
            current = wave_forward(layer, current, t);
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
    static std::vector<uint8_t> _extract_stream(const std::string& path) {
        std::string cmd = "ffmpeg -loglevel error -i \"" + path +
                          "\" -map 0:v:0 -f rawvideo -pix_fmt rgb24 -";
        return read_pipe(cmd);
    }
};

} // namespace sovwave
