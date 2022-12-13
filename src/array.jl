# onehot array
struct OneHotArray{K, N, var"N+1", A<:AbstractArray{OneHot{K}, N}} <: AbstractOneHotArray{var"N+1"}
    onehots::A
    function OneHotArray{K, N, var"N+1", A}(onehots::A) where {K, N, var"N+1", A<:AbstractArray{OneHot{K}, N}}
        @assert K isa UInt32
        @assert N+1 == var"N+1"
        return new{K, N, var"N+1", A}(onehots)
    end
end

OneHotArray(onehots::A) where {K, A<:AbstractArray{OneHot{K}}} = OneHotArray{K, ndims(onehots), ndims(onehots)+1, A}(onehots)
OneHotArray{K}(indices::A) where {K, A<:AbstractArray{<:Union{Int32, UInt32}}} = OneHotArray(reinterpret(OneHot(K), indices))
OneHotArray{K}(xs) where K = OneHotArray(OneHot(K).(xs))
OneHotArray(k, xs::A) where {A<:AbstractArray{<:Integer}} = OneHotArray{UInt32(k)}(xs)

OneHotArray(k, o::OneHot) = OneHotArray([OneHot(k, o)])
OneHotArray(o::OneHot) = OneHotArray([o])

OneHotArray{K, N}(xs::AbstractArray{T, N}) where {K, N, T} = OneHotArray{K, N, N+1}(xs)
function OneHotArray{K, N, var"N+1"}(xs::AbstractArray{T, N}) where {K, N, var"N+1", T}
    @assert N+1 == var"N+1"
    return OneHotArray(K, xs)
end

const OneHotVector{K} = OneHot{K}
const OneHotMatrix{K} = OneHotArray{K, 1}

onehotsize(::OneHotArray{K}) where K = Int(K)

Base.reinterpret(::Type{T}, oa::OneHotArray) where T = reinterpret(T, parent(oa))

# array interface
Base.size(oa::OneHotArray{K}) where K = (onehotsize(oa), size(oa.onehots)...)

Base.@propagate_inbounds function Base.getindex(oa::OneHotArray{K, N}, i, is::Vararg{Int, N}) where {K, N}
    @boundscheck checkbounds(oa, i, is...)
    return oa.onehots[is...][i]
end

Base.@propagate_inbounds function Base.getindex(oa::OneHotArray{K, N}, i::Colon, is::Vararg{Int, N}) where {K, N}
    @boundscheck checkbounds(oa, i, is...)
    return oa.onehots[is...]
end

Base.@propagate_inbounds function Base.getindex(oa::OneHotArray{K}, i::Colon, is...) where {K}
    @boundscheck checkbounds(oa, i, is...)
    return OneHotArray(oa.onehots[is...])
end

Base.@propagate_inbounds function Base.getindex(oa::OneHotArray{K}, i::Integer, is::Vararg{Colon, N}) where {K, N}
    @boundscheck checkbounds(oa, i, is...)
    return map(x->x[i], oa.onehots)
end

Base.similar(o::OneHotArray, ::Type{T}, dims::Dims{N}) where {T, N} = similar(o.onehots, T, dims)

Base.parent(o::OneHotArray) = o.onehots

# printing

function Base.summary(io::IO, oa::OneHotArray)
    join(io, size(oa), 'x')
    print(io, " OneHotArray{", onehotsize(oa), ", ", ndims(oa), ", ")
    Base.showarg(io, oa.onehots, true)
    print(io, "}")
end

# cat

is1(::Val{V}) where V = is1(V)
is1(v) = isone(v)
predecessor(::Val{V}) where V = Val(V-1)
predecessor(v) = Val(v - 1)

Base.vcat(xss::OneHot{K}...) where K = cat(xss...; dims=Val(1))
Base.hcat(xss::OneHot{K}...) where K = cat(xss...; dims=Val(2))

function Base.cat(xss::OneHot{K}...; dims) where K
    if is1(dims)
        @warn "concat OneHot{$K} along dimension 1."
        return Base._cat(Val(1), xss...)
    else
        yss = reshape(collect(xss), reverse(Base.rdims(predecessor(dims), axes(xss))))
        return OneHotArray(yss)
    end
end

Base.vcat(xss::OneHotArray{K}...) where K = cat(xss...; dims=Val(1))
Base.hcat(xss::OneHotArray{K}...) where K = cat(xss...; dims=Val(2))

function Base.cat(xss::OneHotArray{K}...; dims) where K
    if is1(dims)
        @warn "concat OneHotArray{$K} along dimension 1."
        return Base._cat(Val(1), xss...)
    else
        sdims = predecessor(dims)
        xidss = map(parent, xss)
        ret = cat(xidss...; dims=sdims)
        return OneHotArray(ret)
    end
end

# view

function Base.view(parent::OneHotArray, ::Colon, I...)
    onehots = parent.onehots
    v = view(onehots, I...)
    return OneHotArray(v)
end

# reshape

function ohreshape(parent::OneHotArray{K}, dims) where K
    onehots = parent.onehots
    return OneHotArray(reshape(onehots, dims))
end

function Base.reshape(parent::OneHotArray{K}, dims::Dims) where K
    isequal(prod(dims), length(parent)) || throw(DimensionMismatch("new dimensions $(dims) must be consistent with array size $(length(parent))"))
    return isequal(K, first(dims)) ?
        ohreshape(parent, Base.tail(dims)) :
        Base._reshape(parent, dims)
end

Base.reshape(parent::OneHotArray{K}, dims::Tuple{Vararg{Union{Colon, Int}}}) where K = reshape(parent, Base._reshape_uncolon(parent, dims))

# adapt

import Adapt: adapt, adapt_structure
adapt_structure(T, oa::OneHotArray) = OneHotArray(adapt(T, oa.onehots))
