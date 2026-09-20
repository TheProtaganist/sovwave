// SovwaveRuntime.cs — Sovwave MKV Model Runtime (.NET 6+)
//
// Loads a trained Sovwave .mkv model and runs inference with zero Julia dependency.
// Requires: .NET 6+, ffmpeg in PATH.
//
// Usage:
//   var model = await SovwaveModel.LoadAsync("my_model.mkv", "my_model_meta.yaml");
//   double[][] outputs = model.Predict(new[] { new[] { 0.1, 0.2, 0.3, 0.4 } });
//   Console.WriteLine(string.Join(", ", outputs[0]));
//
// Compile: dotnet run  (or add to your .csproj)

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sovwave
{
    // ── WaveLayerParams ───────────────────────────────────────────────────────

    internal class WaveLayerParams
    {
        public int Nodes, EmbedDim;
        public double[][] Amplitudes;    // [nodes][embedDim]
        public double[][] Phases;
        public double[][] Frequencies;
        public double[]   FractalScales;
        public double[]   FractalDims;   // Hausdorff D_f ∈ [1.0, 3.0]
        public double[]   WaveSpeeds;    // -1.0 = unlimited
        public double     Omega;

        public WaveLayerParams(int nodes, int embedDim, double omega)
        {
            Nodes        = nodes;
            EmbedDim     = embedDim;
            Omega        = omega;
            Amplitudes   = Jagged(nodes, embedDim, 0.5);
            Phases       = Jagged(nodes, embedDim, 0.0);
            Frequencies  = Jagged(nodes, embedDim, 1.0);
            FractalScales = Enumerable.Repeat(1.618033988749895, nodes).ToArray();
            FractalDims   = Enumerable.Repeat(1.5,               nodes).ToArray();
            WaveSpeeds    = Enumerable.Repeat(1.0,               nodes).ToArray();
        }

        private static double[][] Jagged(int rows, int cols, double fill)
        {
            var m = new double[rows][];
            for (int i = 0; i < rows; i++) {
                m[i] = new double[cols];
                Array.Fill(m[i], fill);
            }
            return m;
        }
    }

    // ── SovwaveModel ──────────────────────────────────────────────────────────

    public class SovwaveModel
    {
        private readonly List<WaveLayerParams> _layers;
        private readonly int    _nodes, _embedDim, _tFrames;
        private readonly double _omega;

        private SovwaveModel(List<WaveLayerParams> layers, int nodes, int embedDim,
                              double omega, int tFrames)
        {
            _layers   = layers;
            _nodes    = nodes;
            _embedDim = embedDim;
            _omega    = omega;
            _tFrames  = tFrames;
        }

        // ── Static loader ─────────────────────────────────────────────────────

        /// <summary>Load a SovwaveModel from an MKV video file.</summary>
        /// <param name="mkvPath">Path to the .mkv model file</param>
        /// <param name="metaPath">Path to the _meta.yaml config (null = auto-detect)</param>
        public static async Task<SovwaveModel> LoadAsync(string mkvPath, string? metaPath = null)
        {
            int    nodes      = 64;
            int    embedDim   = 64;
            int    numLayers  = 3;
            double omega      = 432.0;
            int    tFrames    = 3;
            double betaS      = 1.618033988749895;

            // Auto-detect companion YAML
            string actualMeta = metaPath
                ?? System.Text.RegularExpressions.Regex.Replace(mkvPath, @"(?i)\.mkv$", "_meta.yaml");

            if (File.Exists(actualMeta))
            {
                bool inModel = false;
                foreach (var line in await File.ReadAllLinesAsync(actualMeta))
                {
                    if (line.Trim() == "model:") { inModel = true; continue; }
                    if (inModel && line.Length > 0 && !char.IsWhiteSpace(line[0])) inModel = false;
                    if (!inModel) continue;

                    var tl = line.Trim();
                    if (tl.StartsWith("nodes: "))      nodes      = int.Parse(tl.Split(": ")[1]);
                    if (tl.StartsWith("embed_dims: ")) embedDim   = int.Parse(tl.Split(": ")[1]);
                    if (tl.StartsWith("layers: "))     numLayers  = int.Parse(tl.Split(": ")[1]);
                    if (tl.StartsWith("omega: "))      omega      = double.Parse(tl.Split(": ")[1]);
                    if (tl.StartsWith("t_frames: "))   tFrames    = int.Parse(tl.Split(": ")[1]);
                    if (tl.StartsWith("beta_s: "))     betaS      = double.Parse(tl.Split(": ")[1]);
                }
            }

            int w = 640;
            int h = 480;

            byte[] raw    = await ExtractStreamAsync(mkvPath);
            var layerList = DecodePixels(raw, w, h, numLayers, nodes, embedDim, omega, betaS);
            return new SovwaveModel(layerList, nodes, embedDim, omega, tFrames);
        }

        // ── Forward pass ──────────────────────────────────────────────────────

        private static double[] WaveForward(WaveLayerParams layer, double[] input, double t)
        {
            int n    = layer.Nodes;
            int d    = layer.EmbedDim;
            int ilen = input.Length;
            var output = new double[n];

            for (int i = 0; i < n; i++)
            {
                double beta    = layer.FractalScales[i];
                double df      = layer.FractalDims[i];
                double vSpd    = layer.WaveSpeeds[i];
                double fracEnv = df / 1.5;
                double nodeSum = 0.0;

                for (int j = 0; j < d; j++)
                {
                    double inVal = j < ilen ? input[j] : 0.5;
                    double amp   = layer.Amplitudes[i][j];
                    double ph    = layer.Phases[i][j];
                    double freq  = layer.Frequencies[i][j];

                    // Speed-aware phase: -1.0 = unlimited
                    double xEff  = vSpd == -1.0 ? inVal : inVal / Math.Max(vSpd, 1e-12);
                    double angle = layer.Omega * 0.001 * freq * xEff + ph - t;
                    nodeSum     += amp * Math.Sin(angle);
                }
                output[i] = (nodeSum / Math.Sqrt(d)) * beta * fracEnv;
            }
            return output;
        }

        /// <summary>Single-sample forward pass through all layers.</summary>
        public double[] Forward(double[] input, double t = 0.0)
        {
            double[] current = (double[])input.Clone();
            foreach (var layer in _layers)
            {
                var accum = new double[layer.Nodes];
                for (int frame = 0; frame < _tFrames; frame++)
                {
                    double tOff = t + 2.0 * Math.PI * frame / (_omega + 1e-12);
                    var    @out = WaveForward(layer, current, tOff);
                    for (int k = 0; k < layer.Nodes; k++) accum[k] += @out[k];
                }
                double scale = 1.0 / Math.Sqrt(_tFrames);
                for (int k = 0; k < accum.Length; k++) accum[k] *= scale;
                current = accum;
            }
            return current;
        }

        /// <summary>Batch prediction.</summary>
        public double[][] Predict(double[][] inputs, double t = 0.0)
            => inputs.Select(inp => Forward(inp, t)).ToArray();

        // ── Pixel decoding ────────────────────────────────────────────────────

        private static List<WaveLayerParams> DecodePixels(
            byte[] raw, int w, int h, int numLayers,
            int nodes, int embedDim, double omega, double betaS)
        {
            var layers        = new List<WaveLayerParams>();
            int bytesPerFrame = w * h * 3;
            double twoPi      = 2.0 * Math.PI;
            int bw            = Math.Max(1, w / embedDim);
            int bh            = Math.Max(1, h / nodes);
            int halfK         = Math.Max(1, bw / 4);

            for (int l = 0; l < numLayers; l++)
            {
                var lp = new WaveLayerParams(nodes, embedDim, omega);
                Array.Fill(lp.FractalScales, betaS);
                int frameOff = l * bytesPerFrame;

                for (int r = 0; r < nodes; r++)
                {
                    int yc = (int)Math.Round((r + 0.5) * bh);
                    for (int c = 0; c < embedDim; c++)
                    {
                        int xc = (int)Math.Round((c + 0.5) * bw);
                        double rAcc = 0.0, gAcc = 0.0, bAcc = 0.0;
                        int count = 0;
                        for (int dy = -halfK; dy <= halfK; dy++)
                        {
                            for (int dx = -halfK; dx <= halfK; dx++)
                            {
                                int px = Math.Min(w - 1, Math.Max(0, xc + dx));
                                int py = Math.Min(h - 1, Math.Max(0, yc + dy));
                                int idx = frameOff + (py * w + px) * 3;
                                if (idx + 2 < raw.Length)
                                {
                                    rAcc += raw[idx];
                                    gAcc += raw[idx + 1];
                                    bAcc += raw[idx + 2];
                                    count++;
                                }
                            }
                        }
                        if (count > 0)
                        {
                            lp.Amplitudes[r][c]  = (rAcc / count / 255.0) * 2.0;
                            lp.Phases[r][c]      = (gAcc / count / 255.0) * twoPi;
                            lp.Frequencies[r][c] = Math.Max(0.1, (bAcc / count / 255.0) * 4.0);
                        }
                        else
                        {
                            lp.Amplitudes[r][c]  = 0.5;
                            lp.Phases[r][c]      = 0.0;
                            lp.Frequencies[r][c] = 1.0;
                        }
                    }
                }
                layers.Add(lp);
            }
            return layers;
        }

        // ── ffmpeg extractor ──────────────────────────────────────────────────

        private static async Task<byte[]> ExtractStreamAsync(string mkvPath)
        {
            return await RunFfmpegAsync(
                "ffmpeg",
                $"-loglevel error -i \"{mkvPath}\" -map 0:v:0 -f rawvideo -pix_fmt rgb24 -");
        }

        private static async Task<byte[]> RunFfmpegAsync(string exe, string args)
        {
            var psi = new ProcessStartInfo(exe, args)
            {
                RedirectStandardOutput = true,
                RedirectStandardError  = true,
                UseShellExecute        = false,
                CreateNoWindow         = true
            };
            using var proc = Process.Start(psi) ?? throw new Exception("Failed to start ffmpeg");
            using var ms   = new MemoryStream();
            await proc.StandardOutput.BaseStream.CopyToAsync(ms);
            await proc.WaitForExitAsync();
            return ms.ToArray();
        }

        // ── CLI entrypoint ────────────────────────────────────────────────────

        public static async Task Main(string[] args)
        {
            if (args.Length < 1) {
                Console.Error.WriteLine("Usage: SovwaveRuntime <model.mkv> [meta.yaml] [0.1,0.2,...]");
                return;
            }
            string mkvPath  = args[0];
            string? meta    = args.Length > 1 ? args[1] : null;
            double[] input  = args.Length > 2
                ? args[2].Split(',').Select(double.Parse).ToArray()
                : new[] { 0.1, 0.2, 0.3, 0.4 };

            var    model  = await SovwaveModel.LoadAsync(mkvPath, meta);
            double[] output = model.Forward(input);
            Console.WriteLine($"Input:  [{string.Join(", ", input)}]");
            Console.WriteLine($"Output: [{string.Join(", ", output.Take(8).Select(v => $"{v:F6}"))}]");
        }
    }
}
