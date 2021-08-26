module PrimitiveOneHot

using NNlib

export OneHotArray

abstract type AbstractOneHotArray{N} <: AbstractArray{Bool,  N} end

include("primitive.jl")
include("array.jl")
include("gpu.jl")
include("op.jl")

end
