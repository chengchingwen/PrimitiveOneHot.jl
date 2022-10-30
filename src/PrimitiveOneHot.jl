module PrimitiveOneHot

using NNlib
using Requires

export OneHotArray

abstract type AbstractOneHotArray{N} <: AbstractArray{Bool,  N} end

include("primitive.jl")
include("array.jl")

@init @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
    CUDA.functional() && include("gpu.jl")
end

include("op.jl")

end
