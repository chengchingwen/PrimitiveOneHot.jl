using Base: @_noinline_meta, @_inline_meta
using Core: is_top_bit_set
using Core.Intrinsics: bitcast, trunc_int, sext_int, zext_int, sle_int, eq_int, and_int

# onehot erorr
struct OneHotEncodeError <: Exception
    K
    val
    OneHotEncodeError(@nospecialize(K), @nospecialize(val)) = (@_noinline_meta; new(K, val))
end

function Base.showerror(io::IO, e::OneHotEncodeError)
    print(io, "OneHotEncodeError: cannot encode ")
    print(io, e.val)
    print(io, " with OneHot{")
    print(io, e.K)
    print(io, '}')
end

throw_onehotencode_error(K, val) = (@_noinline_meta; throw(OneHotEncodeError(K, val)))

# onehot encode
primitive type OneHot{K} <: AbstractOneHotArray{1} 32 end

OneHot(k) = OneHot{UInt32(k)}
OneHot{K}(x) where K = convert(OneHot(K), x)
OneHot(k, x) = OneHot{k}(x)

onehotsize(::OneHot{K}) where K = Int(K)

# array interface

Base.size(o::OneHot) = (onehotsize(o),)
Base.@propagate_inbounds function Base.getindex(o::OneHot, i::I) where {I<:Integer}
    @boundscheck checkbounds(o, i)
    return convert(I, o) == i
end

Base.@propagate_inbounds Base.getindex(o::OneHot, i::Colon) = o

Base.argmax(o::OneHot) = Int(o)
Base.argmin(o::OneHot) = isone(o) ? 2 : 1

# printing

function Base.showarg(io::IO, x::OneHot, toplevel)
    toplevel || print(io, "::")
    print(io, "OneHot{", onehotsize(x), '}')
end

# convert

Base.UInt32(o::OneHot) = bitcast(UInt32, o)
Base.UInt64(o::OneHot) = zext_int(UInt64, o)
Base.Int32(o::OneHot) = bitcast(Int32, o)
Base.Int64(o::OneHot) = zext_int(Int64, o)

Base.convert(::Type{Any}, o::OneHot) = o
Base.convert(::Type{OneHot{K}}, o::OneHot{K}) where {K} = o
Base.convert(::Type{OneHot{K}}, o::OneHot) where {K} = OneHot(K, UInt32(o))
Base.convert(::Type{UInt32},  o::OneHot) = UInt32(o)
Base.convert(::Type{To}, o::OneHot) where {To<:Number} = convert(To, UInt32(o))
Base.convert(::Type{T}, o::OneHot) where {T<:Array} = T(o)

Base.reinterpret(::Type{T}, o::OneHot) where {T} = reinterpret(T, UInt32(o))

# one

Base.one(o::O) where {O<:OneHot} = convert(O, 0x00000001)
Base.one(::Type{<:OneHot{K}}) where K = OneHot(K, 1)
Base.isone(o::OneHot) = isone(convert(UInt32, o))

# zero

Base.zero(o::O) where {O<:OneHot} = convert(O, 0x00000000)
Base.zero(::Type{<:OneHot{K}}) where K = OneHot(K, 0)
Base.iszero(o::OneHot) = iszero(convert(UInt32, o))

# number

Base.typemin(::Type{OneHot{K}}) where K = OneHot(K, 0)
Base.typemax(::Type{OneHot{K}}) where K = OneHot(K, K)

Base.isless(a::OneHot, b::OneHot) = isless(UInt32(a), UInt32(b))
Base.:(==)(a::OneHot, b::OneHot) = UInt32(a) == UInt32(b)

# bit-op

function check_onehot_top_bit(::Type{OneHot{K}}, x) where {K}
    @_inline_meta
    is_top_bit_set(x) && throw_onehotencode_error(K, x)
    x
end

function check_onehot_encode(ot::Type{OneHot{K}}, x) where {K}
    @_inline_meta
    sle_int(x, K) || throw_onehotencode_error(K, x)
    bitcast(ot, x)
end

function checked_onehot_trunc_sint(ot::Type{OneHot{K}}, x::From) where {K, From}
    @_inline_meta
    y = trunc_int(UInt32, x)
    back = sext_int(From, y)
    eq_int(x, back) || throw_onehotencode_error(K, x)
    check_onehot_encode(ot, y)
end

function checked_onehot_trunc_uint(ot::Type{OneHot{K}}, x::From) where {K, From}
    @_inline_meta
    y = trunc_int(UInt32, x)
    back = zext_int(From, y)
    eq_int(x, back) || throw_onehotencode_error(K, x)
    check_onehot_encode(ot, y)
end

Base.convert(ot::Type{OneHot{K}}, x::Int8) where {K} = check_onehot_encode(ot, sext_int(UInt32, check_onehot_top_bit(ot, x)))
Base.convert(ot::Type{OneHot{K}}, x::Int16) where {K} = check_onehot_encode(ot, sext_int(UInt32, check_onehot_top_bit(ot, x)))
Base.convert(ot::Type{OneHot{K}}, x::Int32) where {K} = check_onehot_encode(ot, bitcast(UInt32, check_onehot_top_bit(ot, x)))
Base.convert(ot::Type{OneHot{K}}, x::Int64) where {K} = checked_onehot_trunc_sint(ot, check_onehot_top_bit(ot, x))
Base.convert(ot::Type{OneHot{K}}, x::Int128) where {K} = checked_onehot_trunc_sint(ot, check_onehot_top_bit(ot, x))
Base.convert(ot::Type{OneHot{K}}, x::UInt8) where {K} = check_onehot_encode(ot, zext_int(UInt32, x))
Base.convert(ot::Type{OneHot{K}}, x::UInt16) where {K} = check_onehot_encode(ot, zext_int(UInt32, x))
Base.convert(ot::Type{OneHot{K}}, x::UInt32) where {K} = check_onehot_encode(ot, x)
Base.convert(ot::Type{OneHot{K}}, x::UInt64) where {K} = checked_onehot_trunc_uint(ot, x)
Base.convert(ot::Type{OneHot{K}}, x::UInt128) where {K} = checked_onehot_trunc_uint(ot, x)
Base.convert(ot::Type{OneHot{K}}, x::Bool) where {K} = and_int(zext_int(ot, x), Base.convert(ot, 0x1))
