"""
Sovwave MKV Model Runtime — Python 3.8+

Loads a trained Sovwave .mkv model and runs inference with zero Julia dependency.
Requires: Python 3.8+, ffmpeg in PATH, PyYAML (optional but recommended).

Usage:
    from sovwave_runtime import SovwaveModel
    model = SovwaveModel.load("my_model.mkv", "my_model_meta.yaml")
    outputs = model.predict([[0.1, 0.2, 0.3, 0.4]])
    print(outputs)
"""

import subprocess
import struct
import math
import os
import sys

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# ── Wave forward pass (mirrors Julia WaveLayer.forward!) ─────────────────────

def _wave_forward(layer_params, input_values, t=0.0):
    """
    CPU forward pass for one WaveLayer.
    layer_params: dict with keys 'nodes', 'embed_dim', 'amplitudes', 'phases',
                  'frequencies', 'fractal_scales', 'fractal_dims', 'wave_speeds', 'omega'
    """
    nodes     = layer_params["nodes"]
    embed_dim = layer_params["embed_dim"]
    amps      = layer_params["amplitudes"]    # list[list[float]], shape [nodes][embed_dim]
    phs       = layer_params["phases"]
    freqs     = layer_params["frequencies"]
    fscales   = layer_params["fractal_scales"]
    fdims     = layer_params["fractal_dims"]
    wspeeds   = layer_params["wave_speeds"]
    omega     = layer_params["omega"]
    in_len    = len(input_values)
    output    = [0.0] * nodes

    for i in range(nodes):
        beta    = fscales[i]
        d_f     = fdims[i]
        v_spd   = wspeeds[i]
        frac_env = d_f / 1.5

        node_sum = 0.0
        for j in range(embed_dim):
            in_val = input_values[j] if j < in_len else 0.5
            amp  = amps[i][j]
            ph   = phs[i][j]
            freq = freqs[i][j]

            # Speed-aware phase
            x_eff = in_val if v_spd == -1.0 else in_val / max(v_spd, 1e-12)
            angle = omega * 0.001 * freq * x_eff + ph - t
            node_sum += amp * math.sin(angle)

        node_wave = (node_sum / math.sqrt(embed_dim)) * beta * frac_env
        output[i] = node_wave

    return output


# ── Pixel → weight decoding (mirrors Julia rgb_frames_to_model) ──────────────

def _decode_pixels(raw_bytes, w, h, num_layers, nodes, embed_dim, omega, beta_s=1.618033988749895):
    """Decodes FFV1 raw RGB24 bytes into layer parameter dicts."""
    bytes_per_frame = w * h * 3
    layers = []
    for l_idx in range(num_layers):
        frame_off = l_idx * bytes_per_frame
        amps   = [[0.0]*embed_dim for _ in range(nodes)]
        phs    = [[0.0]*embed_dim for _ in range(nodes)]
        freqs  = [[0.0]*embed_dim for _ in range(nodes)]
        fscales = [beta_s] * nodes
        fdims   = [1.5]    * nodes
        wspeeds = [1.0]    * nodes

        for r in range(nodes):
            for c in range(embed_dim):
                px = frame_off + ((r * w) + c) * 3
                if px + 2 < len(raw_bytes):
                    r_b = raw_bytes[px]
                    g_b = raw_bytes[px + 1]
                    b_b = raw_bytes[px + 2]
                    amps[r][c]  = (r_b / 255.0) * 2.0
                    phs[r][c]   = (g_b / 255.0) * (2 * math.pi)
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

    def __init__(self, layers, nodes, embed_dim, omega=432.0, t_frames=3):
        self.layers     = layers
        self.nodes      = nodes
        self.embed_dim  = embed_dim
        self.omega      = omega
        self.t_frames   = t_frames

    @classmethod
    def load(cls, mkv_path, meta_path=None):
        """
        Load a SovwaveModel from an MKV video file.

        Parameters
        ----------
        mkv_path : str  — path to the .mkv model file
        meta_path : str — path to the _meta.yaml config (optional; uses defaults if missing)

        Returns
        -------
        SovwaveModel
        """
        if not os.path.isfile(mkv_path):
            raise FileNotFoundError(f"Model file not found: {mkv_path}")

        # ── Load metadata ──────────────────────────────────────────────────
        nodes      = 64
        embed_dim  = 64
        num_layers = 3
        omega      = 432.0
        t_frames   = 3
        beta_s     = 1.618033988749895

        if meta_path is None:
            # Auto-detect companion YAML
            candidate = os.path.splitext(mkv_path)[0].replace(".mkv", "") + "_meta.yaml"
            if os.path.isfile(candidate):
                meta_path = candidate

        if meta_path and os.path.isfile(meta_path):
            if _HAS_YAML:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = yaml.safe_load(f)
                model_cfg = meta.get("model", {})
                nodes      = model_cfg.get("nodes", nodes)
                embed_dim  = model_cfg.get("embed_dims", embed_dim)
                num_layers = model_cfg.get("layers", num_layers)
                omega      = model_cfg.get("omega", omega)
                t_frames   = model_cfg.get("t_frames", t_frames)
                beta_s     = model_cfg.get("beta_s", beta_s)
            else:
                print("[sovwave] PyYAML not installed — using default config. Run: pip install pyyaml", file=sys.stderr)

        # ── Extract data stream via ffmpeg ─────────────────────────────────
        w = embed_dim if embed_dim % 2 == 0 else embed_dim + 1
        h = nodes     if nodes     % 2 == 0 else nodes     + 1

        raw_bytes = _extract_data_stream(mkv_path, w, h, num_layers)
        layers    = _decode_pixels(raw_bytes, w, h, num_layers, nodes, embed_dim, omega, beta_s)
        return cls(layers, nodes, embed_dim, omega, t_frames)

    def forward(self, input_values, t=0.0):
        """Run one forward pass through all layers."""
        current = list(input_values)
        for layer in self.layers:
            accum = [0.0] * layer["nodes"]
            for frame in range(self.t_frames):
                t_offset = t + 2 * math.pi * frame / (self.omega + 1e-12)
                out = _wave_forward(layer, current, t_offset)
                accum = [accum[k] + out[k] for k in range(len(accum))]
            scale = 1.0 / math.sqrt(self.t_frames)
            current = [v * scale for v in accum]
        return current

    def predict(self, inputs, t=0.0):
        """
        Run inference on a batch of inputs.

        Parameters
        ----------
        inputs : list[list[float]]  — batch of input vectors
        t : float                   — starting time offset

        Returns
        -------
        list[list[float]]           — batch of output vectors
        """
        return [self.forward(inp, t) for inp in inputs]


# ── ffmpeg helpers ────────────────────────────────────────────────────────────

def _extract_data_stream(mkv_path, w, h, num_layers):
    """Extracts the FFV1 data stream (stream 0:v:1) from the MKV."""
    # Try dedicated data stream first
    try:
        cmd = ["ffmpeg", "-loglevel", "error", "-i", mkv_path,
               "-map", "0:v:1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        result = subprocess.run(cmd, capture_output=True, check=True)
        if result.stdout:
            return bytearray(result.stdout)
    except subprocess.CalledProcessError:
        pass

    # Fallback: stream 0:v:0 with downscale
    downscale = f"scale={w}:{h}:flags=neighbor"
    cmd = ["ffmpeg", "-loglevel", "error", "-i", mkv_path,
           "-map", "0:v:0", "-vf", downscale, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
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
