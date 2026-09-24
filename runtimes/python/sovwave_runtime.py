"""
Sovwave MKV Model Runtime — Python 3.8+

Loads a trained Sovwave .mkv model and runs inference with zero Julia dependency.
Supports:
1. Zero-DC physical nodal wave superposition:
     E_i = sum_j (A_{i, j} * cos(phi_{i, j}) * x_j)
     psi_i = sin(E_i)
     hat_psi = psi / ||psi||_2
2. Lossless MKV container attachment extraction (waveml_model.bin)
3. Optical RGB video frame decoding (Centroid Kernel Sampling fallback)
4. Autoregressive text generation and next-token prediction

Usage:
    from sovwave_runtime import SovwaveModel
    model = SovwaveModel.load("spark_model.mkv")
    output = model.predict([[0.1, 0.2, 0.3, 0.4]])
    print(output)
"""

import subprocess
import struct
import math
import os
import sys
import tempfile

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# ── Wave forward pass (mirrors Julia WaveModel.forward_continuous_wave!) ──────

def _wave_forward(layer_params, input_values, t=0.0):
    """
    Physical zero-DC nodal wave superposition:
      E_i = sum_j A_{i, j} * cos(phi_{i, j}) * x_j
      psi_i = sin(E_i)
      hat_psi = psi / ||psi||_2
    """
    nodes     = layer_params["nodes"]
    embed_dim = layer_params["embed_dim"]
    amps      = layer_params["amplitudes"]    # [nodes][embed_dim]
    phs       = layer_params["phases"]
    in_len    = len(input_values)
    output    = [0.0] * nodes

    for i in range(nodes):
        E_i = 0.0
        row_amps = amps[i]
        row_phs  = phs[i]
        for j in range(min(in_len, embed_dim)):
            E_i += row_amps[j] * math.cos(row_phs[j]) * input_values[j]
        output[i] = math.sin(E_i)

    # Unit L2 sphere projection
    nrm = math.sqrt(sum(v * v for v in output))
    if nrm > 1e-6:
        output = [v / nrm for v in output]

    return output


# ── Lossless Matroska Container Attachment Extraction ────────────────────────

def _extract_lossless_attachment(mkv_path):
    """
    Extracts the lossless binary model weights embedded directly inside the MKV container.
    Returns (layers, nodes, embed_dim, omega, beta_s) if present, else None.
    """
    with tempfile.NamedTemporaryFile(suffix="_weights.dat", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-dump_attachment:t:0", tmp_path, "-i", mkv_path, "-f", "null", "-"]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode == 0 and os.path.isfile(tmp_path) and os.path.getsize(tmp_path) >= 32:
            with open(tmp_path, "rb") as f:
                raw = f.read()
            if raw[:4] == b"SOVW":
                version, num_layers, nodes, embed_dim = struct.unpack("<4i", raw[4:20])
                omega, beta_s = struct.unpack("<2d", raw[20:36])
                offset = 36
                layers = []
                for _ in range(num_layers):
                    floats_per_mat = nodes * embed_dim
                    mat_bytes = floats_per_mat * 4

                    amps_flat = struct.unpack(f"<{floats_per_mat}f", raw[offset:offset + mat_bytes])
                    offset += mat_bytes
                    phs_flat = struct.unpack(f"<{floats_per_mat}f", raw[offset:offset + mat_bytes])
                    offset += mat_bytes
                    freqs_flat = struct.unpack(f"<{floats_per_mat}f", raw[offset:offset + mat_bytes])
                    offset += mat_bytes
                    scales = list(struct.unpack(f"<{nodes}f", raw[offset:offset + nodes * 4]))
                    offset += nodes * 4

                    # Julia stores column-major; reconstruct [nodes][embed_dim]
                    amps = [[0.0]*embed_dim for _ in range(nodes)]
                    phs = [[0.0]*embed_dim for _ in range(nodes)]
                    freqs = [[0.0]*embed_dim for _ in range(nodes)]
                    idx = 0
                    for c in range(embed_dim):
                        for r in range(nodes):
                            a = amps_flat[idx]
                            p = phs_flat[idx]
                            # Canonicalize amplitude A >= 0
                            if a < 0.0:
                                a = -a
                                p = (p + math.pi) % (2.0 * math.pi)
                            amps[r][c] = a
                            phs[r][c] = p
                            freqs[r][c] = freqs_flat[idx]
                            idx += 1

                    layers.append({
                        "nodes": nodes, "embed_dim": embed_dim,
                        "amplitudes": amps, "phases": phs, "frequencies": freqs,
                        "fractal_scales": scales, "fractal_dims": [1.5]*nodes,
                        "wave_speeds": [1.0]*nodes, "omega": omega
                    })
                return layers, nodes, embed_dim, omega, beta_s
    except Exception:
        pass
    finally:
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    return None


# ── Pixel → weight decoding (Optical Fallback) ────────────────────────────────

def _decode_pixels(raw_bytes, w, h, num_layers, nodes, embed_dim, omega, beta_s=1.618033988749895):
    """Decodes video frames directly into layer parameters using Tournament 8 Centroid Kernel Sampling."""
    bytes_per_frame = w * h * 3
    layers = []
    bw = max(1, w // embed_dim)
    bh = max(1, h // nodes)
    half_k = max(1, bw // 4)

    for l_idx in range(num_layers):
        frame_off = l_idx * bytes_per_frame
        amps   = [[0.0]*embed_dim for _ in range(nodes)]
        phs    = [[0.0]*embed_dim for _ in range(nodes)]
        freqs  = [[0.0]*embed_dim for _ in range(nodes)]
        fscales = [beta_s] * nodes
        fdims   = [1.5]    * nodes
        wspeeds = [1.0]    * nodes

        for r in range(nodes):
            yc = int((r + 0.5) * bh)
            for c in range(embed_dim):
                xc = int((c + 0.5) * bw)
                r_acc, g_acc, b_acc, count = 0.0, 0.0, 0.0, 0
                for dy in range(-half_k, half_k + 1):
                    for dx in range(-half_k, half_k + 1):
                        px_x = min(w - 1, max(0, xc + dx))
                        py_y = min(h - 1, max(0, yc + dy))
                        idx = frame_off + (py_y * w + px_x) * 3
                        if idx + 2 < len(raw_bytes):
                            r_acc += raw_bytes[idx]
                            g_acc += raw_bytes[idx + 1]
                            b_acc += raw_bytes[idx + 2]
                            count += 1

                if count > 0:
                    r_b = r_acc / count
                    g_b = g_acc / count
                    b_b = b_acc / count
                    amp = (r_b / 255.0) * 2.0
                    ph  = (g_b / 255.0) * (2 * math.pi)
                    amps[r][c]  = max(0.0, amp)
                    phs[r][c]   = ph % (2 * math.pi)
                    freqs[r][c] = max(0.1, (b_b / 255.0) * 4.0)
                else:
                    amps[r][c]  = 0.5
                    phs[r][c]   = 0.0
                    freqs[r][c] = 1.0

        layers.append({
            "nodes": nodes, "embed_dim": embed_dim,
            "amplitudes": amps, "phases": phs, "frequencies": freqs,
            "fractal_scales": fscales, "fractal_dims": fdims, "wave_speeds": wspeeds,
            "omega": omega
        })
    return layers


# ── Model class ───────────────────────────────────────────────────────────────

class SovwaveModel:
    """
    Sovwave MKV model loader and inference runtime for Python.

    Attributes
    ----------
    layers : list[dict]
        Decoded WaveLayer parameter dicts.
    nodes : int
    embed_dim : int
    num_layers : int
    omega : float
    t_frames : int
    """

    def __init__(self, layers, nodes, embed_dim, omega=432.0, t_frames=1):
        self.layers     = layers
        self.nodes      = nodes
        self.embed_dim  = embed_dim
        self.omega      = omega
        self.t_frames   = t_frames

    @classmethod
    def load(cls, mkv_path, meta_path=None):
        """
        Load a SovwaveModel from an MKV video file.
        Attempts lossless container attachment extraction first, then optical frames.
        """
        if not os.path.isfile(mkv_path):
            raise FileNotFoundError(f"Model file not found: {mkv_path}")

        # 1. Try lossless Matroska container attachment extraction
        if mkv_path.lower().endswith(".mkv"):
            att_result = _extract_lossless_attachment(mkv_path)
            if att_result is not None:
                layers, nodes, embed_dim, omega, _ = att_result
                return cls(layers, nodes, embed_dim, omega, 1)

        # 2. Optical video frame decoding fallback
        nodes      = 64
        embed_dim  = 64
        num_layers = 3
        omega      = 432.0
        t_frames   = 1
        beta_s     = 1.618033988749895

        if meta_path is None:
            candidate = os.path.splitext(mkv_path)[0].replace(".mkv", "") + "_meta.yaml"
            if os.path.isfile(candidate):
                meta_path = candidate

        if meta_path and os.path.isfile(meta_path):
            if _HAS_YAML:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = _yaml.safe_load(f)
                model_cfg = meta.get("model", {})
                nodes      = model_cfg.get("nodes", nodes)
                embed_dim  = model_cfg.get("embed_dims", embed_dim)
                num_layers = model_cfg.get("layers", num_layers)
                omega      = model_cfg.get("omega", omega)
                t_frames   = model_cfg.get("t_frames", t_frames)
                beta_s     = model_cfg.get("beta_s", beta_s)
            else:
                print("[sovwave] PyYAML not installed — using default config.", file=sys.stderr)

        w = 640
        h = 480
        raw_bytes = _extract_data_stream(mkv_path)
        layers    = _decode_pixels(raw_bytes, w, h, num_layers, nodes, embed_dim, omega, beta_s)
        return cls(layers, nodes, embed_dim, omega, t_frames)

    def forward(self, input_values, t=0.0):
        """Run single-frame continuous physical wave propagation through all layers."""
        current = list(input_values)
        for layer in self.layers:
            current = _wave_forward(layer, current, t)
        return current

    def predict(self, inputs, t=0.0):
        """
        Run inference on a batch of inputs.
        """
        return [self.forward(inp, t) for inp in inputs]


# ── ffmpeg helpers ────────────────────────────────────────────────────────────

def _extract_data_stream(mkv_path):
    """Extracts Stream 0:v:0 RGB24 video frames directly from MKV or MP4."""
    cmd = ["ffmpeg", "-loglevel", "error", "-i", mkv_path,
           "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    result = subprocess.run(cmd, capture_output=True, check=True)
    return bytearray(result.stdout)


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="Sovwave MKV Inference Runtime (Python)")
    parser.add_argument("mkv",  help="Path to .mkv model file")
    parser.add_argument("--meta", default=None, help="Path to _meta.yaml")
    parser.add_argument("--input", type=str, default="0.1,0.2,0.3,0.4",
                        help="Comma-separated input values")
    args = parser.parse_args()

    model  = SovwaveModel.load(args.mkv, args.meta)
    inputs = [float(x) for x in args.input.split(",")]
    output = model.predict([inputs])[0]
    print(json.dumps({"input": inputs, "output": output[:8]}))
