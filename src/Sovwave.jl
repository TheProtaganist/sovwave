"""
    Sovwave

From-scratch Julia library for pure wave-based computation, quantum acoustic synthesis,
and continuous morphogenetic wave intelligence (Self-Organizing Vacuum / SOV dynamics).

Zero matrix multiplication (no CUDA, no discrete CPU matmul paradigm), zero Markov chains,
zero backpropagation. Computes via continuous harmonic wave interference in physical manifolds.
"""
module Sovwave

using Base # explicit self-containment intent (empty deps; nothing but stdlib/Base)

const VERSION = v"0.1.0"

# --- guarded includes --------------------------------------------------------
const _SRC_DIR = joinpath(@__DIR__)

for _rel in (
    "Core/Grid.jl",
    "Core/WaveField.jl",
    "Core/Constants.jl",
    "Encoding/FrequencySpectrum.jl",
    "Encoding/HarmonicEncoder.jl",
    "Potentials/SacredGeometry.jl",
    "Potentials/FlowerOfLife.jl",
    "Potentials/MetatronsCube.jl",
    "Simulation/MorphogeneticSimulator.jl",
    "Simulation/TimeEvolution.jl",
    "Simulation/EnergyFunctionals.jl",
    "Extraction/EigenState.jl",
    "Extraction/CymaticExtractor.jl",
    "Optimization/WaveOptimizer.jl",
    "Optimization/GradientFlow.jl",
    "Optimization/AdamWave.jl",
    "Loss/WaveLoss.jl",
    "Loss/FrequencyDomainMSE.jl",
    "Loss/PhaseCoherenceLoss.jl",
    "Loss/FreeEnergyLoss.jl",
    "Activation/WaveActivation.jl",
    "Activation/WaveSigmoid.jl",
    "Activation/WaveReLU.jl",
    "Activation/WaveTanh.jl",
    "Networks/SOVLayer.jl",
    "Networks/SOVDense.jl",
    "Networks/SOVNetwork.jl",
    "Training/TrainingLoop.jl",
    "Training/Validation.jl",
    "Training/EarlyStopping.jl",
    "Inference/SingleSample.jl",
    "Inference/BatchInference.jl",
    "Data/DataLoader.jl",
    "Data/Preprocessing.jl",
    "Data/Augmentation.jl",
    "Persistence/ModelMetadata.jl",
    "Persistence/MP4Storage.jl",
    "Persistence/GGUFConverter.jl",
    "WaveProtocols/WaveProtocol.jl",
    "WaveProtocols/SignalProcessing.jl",
    "WaveProtocols/AudioProtocols.jl",
    "WaveProtocols/EMFProtocols.jl",
    "WaveProtocols/RadioProtocols.jl",
    "WaveProtocols/TestRunner.jl",
    "WaveProtocols/ProtocolLogger.jl",
    "Audio/Audio.jl",
)
    _p = joinpath(_SRC_DIR, _rel)
    if isfile(_p)
        include(_p)
    else
        @debug "Sovwave: $(_rel) not implemented yet — excluded from stub build"
    end
end

# Backwards compatibility alias
const Aetheria = Sovwave
export Sovwave, Aetheria

end # module Sovwave
