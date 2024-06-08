module PrimitiveOneHotGPUArraysExt

using PrimitiveOneHot
using GPUArrays

PrimitiveOneHot._mapreduce(::AnyGPUArray, arg...; kw...) = GPUArrays._mapreduce(arg...; kw...)

end
