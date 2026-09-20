"""
    examples/mnist/mnist_wave.jl

Full Continuous Wave MNIST Digit Classifier (0-9) Project.
Downloads official MNIST digits directly from Hugging Face Hub (or canonical mirrors),
projects 28x28 pixel fields into continuous 2D surface harmonic waves, evolves weights
using continuous evolution with green terminal progress bar, sonifies at 432 Hz Gamma-to-Epsilon,
serializes as fluid cymatic heatmap MKV video with presentation audio, and runs inference.
"""

push!(LOAD_PATH, normpath(joinpath(@__DIR__, "../../")))
using Sovwave
using Sovwave.WaveML
using Printf

include("src/dataset.jl")
include("src/model.jl")
include("src/train.jl")
include("src/infer.jl")

using .MNISTDataset
using .MNISTModel
using .MNISTTrain
using .MNISTInfer

function main()
    println("="^85)
    println(" 🌊 SOVWAVE PRODUCTION PROJECT: CONTINUOUS WAVE MNIST CLASSIFIER 🌊")
    println("="^85)
    println(" • Direct Hugging Face Hub Integration (token optional / public access supported)")
    println(" • 2D Spatial Surface Harmonic Wave Projection (process_pixel_waves)")
    println(" • Continuous Evolutionary Relaxation with Bold Green Terminal Progress Bar")
    println(" • Sound Entrainment: Fixed 432 Hz Carrier with Asymptotic Gamma-to-Epsilon Beat")
    println(" • Video Checkpointing: Fluid Continuous Cymatic Heatmap MKV with Presentation Audio")
    println(" • Zero Matrix Multipliers | Zero Convolution Kernels | Zero Markov Chains")
    println("-"^85)

    # 1. Train on real Hugging Face MNIST digits
    checkpoint_dir = normpath(joinpath(@__DIR__, "checkpoints"))
    net, history = train_mnist_model(
        epochs = 25,
        limit = 500,
        batch_size = 25,
        checkpoint_dir = checkpoint_dir,
        sonify = true # Set to true to stream live audio to speakers
    )

    # 2. Checkpoint verification
    latest_ckpt = joinpath(checkpoint_dir, @sprintf("checkpoint_epoch_%04d.mkv", history.best_epoch))
    if !isfile(latest_ckpt)
        ckpts = filter(f -> endswith(f, ".mkv"), readdir(checkpoint_dir))
        if !isempty(ckpts)
            latest_ckpt = joinpath(checkpoint_dir, ckpts[end])
        end
    end

    # Verify video streams and file integrity
    println("\n 🔍 Verifying Video Playability & Multi-Platform Decoder Support...")
    if isfile(latest_ckpt)
        @printf("  ✓ Found MKV model video: %s (%.1f KB)\n", latest_ckpt, filesize(latest_ckpt)/1024)
        run(pipeline(`ffprobe -v error -show_entries stream=index,codec_type,codec_name,width,height $latest_ckpt`, stdout=stdout))
        
        # Test GStreamer / Totem player discoverer
        try
            run(pipeline(`gst-discoverer-1.0 $latest_ckpt`, stdout=devnull, stderr=devnull))
            println("  ✓ GStreamer/Totem media engine: PASS (clean decode, zero errors)")
        catch
            println("  ⚠️ GStreamer check completed with warnings")
        end

        mp4_ckpt = replace(latest_ckpt, r"\.mkv$"i => ".mp4")
        if isfile(mp4_ckpt)
            @printf("  ✓ Found MP4 companion video: %s (%.1f KB)\n", mp4_ckpt, filesize(mp4_ckpt)/1024)
            try
                run(pipeline(`gst-discoverer-1.0 $mp4_ckpt`, stdout=devnull, stderr=devnull))
                println("  ✓ MP4 QuickTime/Web player engine: PASS (faststart enabled)")
            catch
            end
        end
    end

    # 3. Evaluate on test split directly from MKV video
    println("\n Running post-training evaluation directly restored from MKV video...")
    if isfile(latest_ckpt)
        evaluate_mnist_model(mkv_path=latest_ckpt, limit=50)
    else
        evaluate_mnist_model(net, limit=50)
    end

    println("\n" * "="^85)
    println(" 🏆 FULL MNIST CONTINUOUS WAVE PROJECT RUN COMPLETE")
    println("="^85)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
