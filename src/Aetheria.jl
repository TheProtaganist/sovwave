"""
    Aetheria

Legacy backwards compatibility bridge for `Sovwave.jl`.
Allows older code, tests, and scripts referencing `Aetheria` to continue functioning seamlessly.
"""
module Aetheria

if !isdefined(parentmodule(Aetheria), :Sovwave)
    include(joinpath(@__DIR__, "Sovwave.jl"))
end
using .Sovwave: Sovwave, VERSION, Audio, _SRC_DIR

export Sovwave, Aetheria, Audio

end # module Aetheria