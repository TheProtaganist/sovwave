import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * SovwaveRuntime.java — Sovwave MKV Model Runtime (Java 11+)
 *
 * Loads a trained Sovwave .mkv model and runs inference with zero Julia dependency.
 * Requires: Java 11+, ffmpeg in PATH.
 *
 * Usage:
 *   SovwaveModel model = SovwaveModel.load("my_model.mkv", "my_model_meta.yaml");
 *   double[][] outputs = model.predict(new double[][]{{0.1, 0.2, 0.3, 0.4}});
 *   System.out.println(Arrays.toString(outputs[0]));
 *
 * Compile:  javac SovwaveRuntime.java
 * Run:      java SovwaveRuntime my_model.mkv
 */

// ── WaveLayerParams ───────────────────────────────────────────────────────────

class WaveLayerParams {
    int nodes, embedDim;
    double[][] amplitudes;   // [nodes][embedDim]
    double[][] phases;
    double[][] frequencies;
    double[]   fractalScales;
    double[]   fractalDims;   // Hausdorff D_f ∈ [1.0, 3.0]
    double[]   waveSpeeds;    // -1.0 = unlimited
    double omega;

    WaveLayerParams(int nodes, int embedDim, double omega) {
        this.nodes     = nodes;
        this.embedDim  = embedDim;
        this.omega     = omega;
        amplitudes     = new double[nodes][embedDim];
        phases         = new double[nodes][embedDim];
        frequencies    = new double[nodes][embedDim];
        fractalScales  = new double[nodes];
        fractalDims    = new double[nodes];
        waveSpeeds     = new double[nodes];

        for (int i = 0; i < nodes; i++) {
            for (int j = 0; j < embedDim; j++) {
                amplitudes[i][j]  = 0.5;
                phases[i][j]      = 0.0;
                frequencies[i][j] = 1.0;
            }
            fractalScales[i] = 1.618033988749895;
            fractalDims[i]   = 1.5;
            waveSpeeds[i]    = 1.0;
        }
    }
}

// ── SovwaveModel ──────────────────────────────────────────────────────────────

public class SovwaveModel {
    private final List<WaveLayerParams> layers;
    private final int nodes, embedDim, tFrames;
    private final double omega;

    private SovwaveModel(List<WaveLayerParams> layers, int nodes, int embedDim,
                          double omega, int tFrames) {
        this.layers   = layers;
        this.nodes    = nodes;
        this.embedDim = embedDim;
        this.omega    = omega;
        this.tFrames  = tFrames;
    }

    // ── Static loader ─────────────────────────────────────────────────────────

    /**
     * Load a SovwaveModel from an MKV video file.
     * @param mkvPath  Path to the .mkv model file
     * @param metaPath Path to the _meta.yaml config (null = auto-detect)
     */
    public static SovwaveModel load(String mkvPath, String metaPath) throws IOException {
        int    nodes      = 64;
        int    embedDim   = 64;
        int    numLayers  = 3;
        double omega      = 432.0;
        int    tFrames    = 3;
        double betaS      = 1.618033988749895;

        // Auto-detect meta YAML
        String actualMeta = metaPath;
        if (actualMeta == null) {
            actualMeta = mkvPath.replaceAll("(?i)\\.mkv$", "_meta.yaml");
        }
        File metaFile = new File(actualMeta);
        if (metaFile.exists()) {
            try (BufferedReader br = new BufferedReader(new FileReader(metaFile))) {
                String line;
                boolean inModel = false;
                while ((line = br.readLine()) != null) {
                    if (line.trim().equals("model:")) { inModel = true; continue; }
                    if (inModel && !line.isEmpty() && !Character.isWhitespace(line.charAt(0))) inModel = false;
                    if (!inModel) continue;
                    String tl = line.trim();
                    if (tl.startsWith("nodes: "))      nodes      = Integer.parseInt(tl.split(": ")[1]);
                    if (tl.startsWith("embed_dims: ")) embedDim   = Integer.parseInt(tl.split(": ")[1]);
                    if (tl.startsWith("layers: "))     numLayers  = Integer.parseInt(tl.split(": ")[1]);
                    if (tl.startsWith("omega: "))      omega      = Double.parseDouble(tl.split(": ")[1]);
                    if (tl.startsWith("t_frames: "))   tFrames    = Integer.parseInt(tl.split(": ")[1]);
                    if (tl.startsWith("beta_s: "))     betaS      = Double.parseDouble(tl.split(": ")[1]);
                }
            } catch (Exception ignored) {}
        }

        int w = 640;
        int h = 480;

        byte[] raw = extractStream(mkvPath);
        List<WaveLayerParams> layerList = decodePixels(raw, w, h, numLayers, nodes, embedDim, omega, betaS);
        return new SovwaveModel(layerList, nodes, embedDim, omega, tFrames);
    }

    public static SovwaveModel load(String mkvPath) throws IOException {
        return load(mkvPath, null);
    }

    // ── Forward pass ──────────────────────────────────────────────────────────

    private static double[] waveForward(WaveLayerParams layer, double[] input, double t) {
        int n   = layer.nodes;
        int d   = layer.embedDim;
        int ilen = input.length;
        double[] output = new double[n];

        for (int i = 0; i < n; i++) {
            double beta    = layer.fractalScales[i];
            double df      = layer.fractalDims[i];
            double vSpd    = layer.waveSpeeds[i];
            double fracEnv = df / 1.5;
            double nodeSum = 0.0;

            for (int j = 0; j < d; j++) {
                double inVal = j < ilen ? input[j] : 0.5;
                double amp   = layer.amplitudes[i][j];
                double ph    = layer.phases[i][j];
                double freq  = layer.frequencies[i][j];

                double xEff  = (vSpd == -1.0) ? inVal : inVal / Math.max(vSpd, 1e-12);
                double angle = layer.omega * 0.001 * freq * xEff + ph - t;
                nodeSum     += amp * Math.sin(angle);
            }
            output[i] = (nodeSum / Math.sqrt(d)) * beta * fracEnv;
        }
        return output;
    }

    /** Single sample forward pass */
    public double[] forward(double[] input, double t) {
        double[] current = input.clone();
        for (WaveLayerParams layer : layers) {
            double[] accum = new double[layer.nodes];
            for (int frame = 0; frame < tFrames; frame++) {
                double tOff = t + 2 * Math.PI * frame / (omega + 1e-12);
                double[] out = waveForward(layer, current, tOff);
                for (int k = 0; k < layer.nodes; k++) accum[k] += out[k];
            }
            double scale = 1.0 / Math.sqrt(tFrames);
            for (int k = 0; k < accum.length; k++) accum[k] *= scale;
            current = accum;
        }
        return current;
    }

    public double[] forward(double[] input) { return forward(input, 0.0); }

    /** Batch prediction */
    public double[][] predict(double[][] inputs, double t) {
        double[][] results = new double[inputs.length][];
        for (int i = 0; i < inputs.length; i++) results[i] = forward(inputs[i], t);
        return results;
    }

    public double[][] predict(double[][] inputs) { return predict(inputs, 0.0); }

    // ── Pixel decoding ────────────────────────────────────────────────────────

    private static List<WaveLayerParams> decodePixels(
        byte[] raw, int w, int h, int numLayers,
        int nodes, int embedDim, double omega, double betaS
    ) {
        List<WaveLayerParams> layers = new ArrayList<>();
        int bytesPerFrame = w * h * 3;
        double TWO_PI = 2.0 * Math.PI;
        int bw = Math.max(1, w / embedDim);
        int bh = Math.max(1, h / nodes);
        int halfK = Math.max(1, bw / 4);

        for (int l = 0; l < numLayers; l++) {
            WaveLayerParams lp = new WaveLayerParams(nodes, embedDim, omega);
            Arrays.fill(lp.fractalScales, betaS);
            int frameOff = l * bytesPerFrame;

            for (int r = 0; r < nodes; r++) {
                int yc = (int) Math.round((r + 0.5) * bh);
                for (int c = 0; c < embedDim; c++) {
                    int xc = (int) Math.round((c + 0.5) * bw);
                    double rAcc = 0.0, gAcc = 0.0, bAcc = 0.0;
                    int count = 0;
                    for (int dy = -halfK; dy <= halfK; dy++) {
                        for (int dx = -halfK; dx <= halfK; dx++) {
                            int px = Math.min(w - 1, Math.max(0, xc + dx));
                            int py = Math.min(h - 1, Math.max(0, yc + dy));
                            int idx = frameOff + (py * w + px) * 3;
                            if (idx + 2 < raw.length) {
                                rAcc += (raw[idx] & 0xFF);
                                gAcc += (raw[idx + 1] & 0xFF);
                                bAcc += (raw[idx + 2] & 0xFF);
                                count++;
                            }
                        }
                    }
                    if (count > 0) {
                        lp.amplitudes[r][c]  = (rAcc / count / 255.0) * 2.0;
                        lp.phases[r][c]      = (gAcc / count / 255.0) * TWO_PI;
                        lp.frequencies[r][c] = Math.max(0.1, (bAcc / count / 255.0) * 4.0);
                    } else {
                        lp.amplitudes[r][c]  = 0.5;
                        lp.phases[r][c]      = 0.0;
                        lp.frequencies[r][c] = 1.0;
                    }
                }
            }
            layers.add(lp);
        }
        return layers;
    }

    // ── ffmpeg extractor ──────────────────────────────────────────────────────

    private static byte[] extractStream(String mkvPath) throws IOException {
        return runFfmpeg(new String[]{
            "ffmpeg", "-loglevel", "error", "-i", mkvPath,
            "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
        });
    }

    private static byte[] runFfmpeg(String[] cmd) throws IOException {
        Process proc = new ProcessBuilder(cmd)
            .redirectErrorStream(false)
            .start();
        byte[] data = proc.getInputStream().readAllBytes();
        try { proc.waitFor(); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
        return data;
    }

    // ── CLI entrypoint ────────────────────────────────────────────────────────

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage: java SovwaveRuntime <model.mkv> [meta.yaml] [in1,in2,...]");
            System.exit(1);
        }
        String mkvPath  = args[0];
        String metaPath = args.length > 1 ? args[1] : null;
        double[] input  = { 0.1, 0.2, 0.3, 0.4 };
        if (args.length > 2) {
            String[] parts = args[2].split(",");
            input = new double[parts.length];
            for (int i = 0; i < parts.length; i++) input[i] = Double.parseDouble(parts[i]);
        }

        SovwaveModel model = SovwaveModel.load(mkvPath, metaPath);
        double[] output = model.forward(input);
        System.out.println("Input:  " + Arrays.toString(input));
        System.out.println("Output: " + Arrays.toString(Arrays.copyOf(output, Math.min(8, output.length))));
    }
}
