using ChainRulesCore
ChainRulesCore.@non_differentiable OneHot(args...)
ChainRulesCore.@non_differentiable OneHotArray(args...)

function ChainRulesCore.rrule(::typeof(reinterpret), ::Type{T}, oa::AbstractOneHotArray) where T
    return reinterpret(T, oa), function reinterpret_onehot_pullback(_)
        (NoTangent(), NoTangent(), NoTangent())
    end
end

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

# fast mul

function Base.:(*)(A::AbstractMatrix, oa::OneHotArray)
    size(A, 2) == onehotsize(oa) || throw(DimensionMismatch("Matrix column must correspond with OneHot size: $(size(A, 2)) != $(onehotsize(oa))"))
    return gather(A, oa)
end

function Base.:(*)(A::AbstractMatrix, oa::OneHot)
    size(A, 2) == onehotsize(oa) || throw(DimensionMismatch("Matrix column must correspond with OneHot size: $(size(A, 2)) != $(onehotsize(oa))"))
    return A[:, Int32(oa)]
end

@init @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" begin
    Flux.onecold(oa::AbstractOneHotArray) = reinterpret(Int32, oa)
end
