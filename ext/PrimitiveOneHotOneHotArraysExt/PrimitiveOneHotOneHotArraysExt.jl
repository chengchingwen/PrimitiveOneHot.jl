module PrimitiveOneHotOneHotArraysExt

using PrimitiveOneHot
using OneHotArrays

OneHotArrays.onecold(oa::PrimitiveOneHot.AbstractOneHotArray) = reinterpret(Int32, oa)

end
