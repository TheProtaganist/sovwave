# 💾 Checkpoint MKV Saving Guide

## Quick Start

Train with automatic checkpoint saving:

```julia
using Sovwave

# Your model and data
model = WaveModel(cfg)
inputs = [rand(32) for _ in 1:100]
targets = [rand(16) for _ in 1:100]

# Train with checkpoints every 10 epochs
trained_model, history = train!(
    model, inputs, targets, cfg.train;
    checkpoint_dir = "my_checkpoints",  # Where to save
    checkpoint_every = 10,              # Save every N epochs
    verbose = true
)
```

## What Gets Saved

Every checkpoint includes:

1. **`.mkv` file**: Dual-stream video with model weights + visualization
2. **`.mp4` file**: Single-stream version (double-click to view!)
3. **`_meta.yaml` file**: Configuration and metadata

## Viewing Checkpoints

### Option 1: Video Player (Instant Visual Inspection)
```bash
# Double-click the .mp4 file in your file browser, or:
vlc my_checkpoints/checkpoint_epoch_0010.mp4
mpv my_checkpoints/checkpoint_epoch_0010.mp4
ffplay my_checkpoints/checkpoint_epoch_0010.mp4
```

**What you'll see**:
- Potts q-state domain visualization (magenta/yellow/cyan)
- Wave lattice evolution from step n to step n_final
- Visual representation of learned quantum states

### Option 2: Load in Julia (Resume Training)
```julia
using Sovwave

# Load checkpoint
checkpoint = load_model("my_checkpoints/checkpoint_epoch_0010.mkv")

# Inspect
summary = inspect_model(checkpoint)
println("Loaded model from epoch 10")
println("Parameters: ", summary[:total_parameters])
println("Energy: ", checkpoint.total_energy)

# Resume training from checkpoint
new_model, history = train!(checkpoint, inputs, targets, cfg.train)
```

### Option 3: Extract Frames with ffmpeg
```bash
# Extract visual frames
ffmpeg -i checkpoint_epoch_0010.mkv -map 0:v:0 frame_%04d.png

# Extract model weight frames (lossless)
ffmpeg -i checkpoint_epoch_0010.mkv -map 0:v:1 -c copy weights.mkv
```

## Configuration Options

```julia
train!(model, inputs, targets, cfg;
    checkpoint_dir = "checkpoints",     # Directory to save (auto-created)
    checkpoint_every = 10,              # Save every N epochs (default: 10)
    audio_cfg = WaveAudioConfig(...),  # Optional audio config
    verbose = true                      # Show checkpoint save messages
)
```

## Checkpoint Naming Convention

```
checkpoints/
├── checkpoint_epoch_0010.mkv        # Epoch 10 full model
├── checkpoint_epoch_0010.mp4        # Epoch 10 viewable video
├── checkpoint_epoch_0010_meta.yaml  # Epoch 10 metadata
├── checkpoint_epoch_0020.mkv        # Epoch 20 full model
├── checkpoint_epoch_0020.mp4        # Epoch 20 viewable video
└── checkpoint_epoch_0020_meta.yaml  # Epoch 20 metadata
```

## Benefits

✅ **Resume Training**: Load any checkpoint and continue training  
✅ **Visual Progress**: Watch model evolution as a video  
✅ **No Extra Storage**: Checkpoints are tiny (3-5 KB typical)  
✅ **Universal Viewing**: MP4 files work everywhere  
✅ **Lossless Weights**: Exact model state preserved in MKV  
✅ **Metadata Included**: YAML files store full configuration  

## Performance Impact

- **Checkpoint save time**: ~5-20ms per checkpoint
- **Storage per checkpoint**: 3-10 KB (depends on model size)
- **Training slowdown**: <0.1% (only saves on checkpoint epochs)

Checkpoints are saved **asynchronously** and don't block training.

## Advanced: Custom Video Rendering

```julia
train!(model, inputs, targets, cfg;
    checkpoint_dir = "checkpoints",
    checkpoint_every = 5,
    video_cfg = WaveVideoConfig(
        frames = 16,              # More frames = longer video
        fps = 8,                  # Faster playback
        render_mode = :potts_model_q_state_domains,
        pixel_scale = 4,          # Larger pixels
        target_height = 720       # HD resolution
    )
)
```

## Troubleshooting

### Checkpoints not saving?
- Check that `checkpoint_dir` path is writable
- Ensure `ffmpeg` is installed: `which ffmpeg`
- Set `verbose=true` to see error messages

### Can't view MP4 files?
- MP4 export requires `ffmpeg` with H.264 codec
- Install: `sudo apt install ffmpeg` (Linux) or `brew install ffmpeg` (Mac)

### Checkpoint files too large?
- Reduce `video_cfg.frames` (default: 8)
- Use `include_audio=false` in save_model options
- Checkpoints scale with model size (nodes × layers)

## Example: Long Training Session

```julia
using Sovwave

# Large dataset, long training
cfg = WaveMLConfig(
    train = WaveTrainConfig(epochs=1000, batch_size=32)
)

model = WaveModel(cfg)

# Save checkpoints every 50 epochs
trained, history = train!(
    model, large_inputs, large_targets, cfg.train;
    checkpoint_dir = "long_training_checkpoints",
    checkpoint_every = 50,
    verbose = true
)

# Result: 20 checkpoints saved (epochs 50, 100, 150, ..., 1000)
# Can resume from any checkpoint if training is interrupted
```

## Integration with Existing Code

The checkpoint feature is **backward compatible**:

```julia
# Old code (still works, no checkpoints saved)
train!(model, inputs, targets, cfg.train)

# New code (checkpoints saved)
train!(model, inputs, targets, cfg.train; checkpoint_dir="ckpt")
```

If you don't specify `checkpoint_dir`, no checkpoints are saved.

---

**See Also**:
- `specs/CPU_Optimization_Winners.md` - Training performance improvements
- `test_checkpoint_saving.jl` - Full example with verification
- `docs/WaveML_User_Guide.md` - Complete WaveML documentation
