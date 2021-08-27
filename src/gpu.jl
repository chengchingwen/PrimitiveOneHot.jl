# gpu

import .CUDA
import .CUDA: CuArray, CuArrayStyle, @allowscalar
import Adapt: WrappedArray

const CuOneHotArray{K, N, var"N+1"} = OneHotArray{K, N, var"N+1", <: CuArray{OneHot{K}, N}}
const WrappedCuOneHotArray{K, N} = WrappedArray{Bool, N, CuOneHotArray{K, N}, CuOneHotArray{K, N}}

Base.print_array(io::IO, oa::Union{CuOneHotArray, WrappedCuOneHotArray}) = Base.print_array(io, adapt(Array, oa))

Base._show_nonempty(io::IO, oa::Union{CuOneHotArray, WrappedCuOneHotArray}, prefix::String) =
    Base._show_nonempty(io, adapt(Array, oa), prefix)
Base._show_empty(io::IO, oa::Union{CuOneHotArray, WrappedCuOneHotArray}) =
    Base._show_empty(io, adapt(Array, oa))

Base.convert(::Type{T}, oa::Union{CuOneHotArray, WrappedCuOneHotArray}) where {T<:Array} =
    Base.convert(T, adapt(Array, oa))

Base.Array{T, N}(oa::Union{CuOneHotArray, WrappedCuOneHotArray}) where {T, N} = Array{T, N}(adapt(Array, oa))
Base.collect(oa::Union{CuOneHotArray, WrappedCuOneHotArray}) = collect(adapt(Array, oa))

Base.BroadcastStyle(::Type{<:CuOneHotArray{K, N}}) where {K, N} = CuArrayStyle{N+1}()

# avoid array copy
function Base.copyto!(dst::CuArray, oa::OneHotArray)
    dst .= oa
    return dst
end

CUDA.CuArray{T, N, B}(oa::Union{CuOneHotArray, WrappedCuOneHotArray}) where {T, K, N, B, var"N+1", A <: CuArray} =
    copyto!(similar(oa, T), oa)

using Base.Cartesian

function Base.findmax(oa::CuOneHotArray; dims=:)
    if dims == Colon()
        i = findfirst(!iszero, parent(oa))
        isnothing(i) && return (false, CartesianIndex{ndims(oa)}())
        v = @allowscalar oa.onehots[i]
        return (true, CartesianIndex(Int(v), i))
    elseif isone(dims)
        onehots = parent(oa)
        vs = map(!iszero, onehots)
        function f1(x, cidx)
            iszero(x) ? CartesianIndex(1, cidx) : CartesianIndex(Int(x), cidx)
        end
        is = map(f1, onehots, CartesianIndices(onehots))
        return (reshape(vs, 1, size(vs)...), reshape(is, 1, size(is)...))
    else
        return findminmax(>, oa; init=false, dims=dims)
    end
end

function findminmax(binop, a::CuOneHotArray; init, dims)
    function f(t1, t2)
        (x, i), (y, j) = t1, t2

        iszero(i) && return t2
        iszero(j) && return t1
        binop(x, y) && return t1
        x == y && return (x, Base.min(i, j))
        return t2
    end

    indx = ndims(a) == 1 ? (eachindex(a), 1) :
                           (CartesianIndices(a), zero(CartesianIndex{ndims(a)}))

    if dims == Colon()
        mapreduce(tuple, f, a, indx[1]; init = (init, indx[2]))
    else
        res = CUDA.GPUArrays._mapreduce(tuple, f, a, indx[1];
                                        init = (init, indx[2]), dims=dims)
        vals = map(x->x[1], res)
        inds = map(x->x[2], res)
        CUDA.unsafe_free!(res)
        return (vals, inds)
    end
end
