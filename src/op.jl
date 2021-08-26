using ChainRulesCore
ChainRulesCore.@non_differentiable OneHot(args...)
ChainRulesCore.@non_differentiable OneHotArray(args...)

# gather
using NNlib: gather, scatter

NNlib.gather(src::AbstractArray{Tsrc, Nsrc},
             idx::OneHotArray) where {Tsrc, Nsrc} = gather(src, reinterpret(Int32, idx))

NNlib.gather!(dst::AbstractArray, src::AbstractArray, idx::OneHotArray) =
    NNlib.gather!(dst, src, reinterpret(Int32, idx))

function ChainRulesCore.rrule(::typeof(NNlib.gather!), dst::AbstractArray, src::AbstractArray, idx::OneHotArray)
    _idx = reinterpret(Int32, idx)
    return rrule(NNlib.gather!, dst, src, _idx)
end

# scatter

NNlib.scatter!(op, dst::AbstractArray, src::AbstractArray, idx::OneHotArray) =
    NNlib.scatter!(op, dst, src, reinterpret(Int32, idx))

NNlib.scatter(op, src::AbstractArray{Tsrc,Nsrc}, idx::OneHotArray;
              init = nothing, dstsize = nothing) where {Tsrc, Nsrc} =
                  NNlib.scatter(op, src, reinterpret(Int32, idx); init, dstsize)

function ChainRulesCore.rrule(::typeof(NNlib.scatter!), op, dst::AbstractArray, src::AbstractArray, idx::OneHotArray)
    _idx = reinterpret(Int32, idx)
    return rrule(NNlib.scatter!, op, dst, src, _idx)
end

function ChainRulesCore.rrule(::typeof(NNlib.scatter), op, src::AbstractArray, idx::OneHotArray; kws...)
    _idx = reinterpret(Int32, idx)
    return rrule(NNlib.scatter, op, src, _idx; kws...)
end

function Base.:(*)(A::AbstractMatrix, oa::AbstractOneHotArray)
    size(A, 2) == onehotsize(oa) || throw(DimensionMismatch("Matrix column must correspond with OneHot size: $(size(A, 2)) != $(onehotsize(oa))"))
    return A[:, reinterpret(Int32, oa)]
end

