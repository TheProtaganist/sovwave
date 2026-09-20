"""
    examples/mnist/src/dataset.jl

Hugging Face MNIST Continuous Wave Dataset Pipeline.
Downloads official MNIST images directly from Hugging Face Hub (with optional HF Token support
or public token-free access) and projects 28x28 pixel fields into continuous 2D surface harmonic waves.
"""

module MNISTDataset

using Printf
using Sovwave
using Sovwave.WaveML

export load_mnist_dataset, stream_mnist_dataset

"""
    load_mnist_dataset(; split="train", limit=nothing, streaming=false, token=nothing, embed_dim=32)

Downloads real MNIST digits directly from Hugging Face Hub / mirrors and formats into a continuous `WaveDataset`.
Tokens are optional: works completely token-free for public access, or with explicit `token` / `ENV["HF_TOKEN"]`.
"""
function load_mnist_dataset(;
    split::String = "train",
    limit::Union{Nothing, Int} = nothing,
    streaming::Bool = false,
    token::Union{Nothing, String} = nothing,
    embed_dim::Int = 32,
    batch_size::Int = 16
)
    return load_hf_dataset(
        "mnist";
        split = split,
        limit = limit,
        streaming = streaming,
        token = token,
        embed_dim = embed_dim,
        batch_size = batch_size
    )
end

"""
    stream_mnist_dataset(; split="train", batch_size=16, kwargs...)

Returns an asynchronous streaming iterator `WaveDataStreamer` over the Hugging Face MNIST dataset.
"""
function stream_mnist_dataset(;
    split::String = "train",
    batch_size::Int = 16,
    kwargs...
)
    return load_mnist_dataset(; split=split, streaming=true, batch_size=batch_size, kwargs...)
end

end # module MNISTDataset
