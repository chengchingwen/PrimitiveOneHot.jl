module PrimitiveOneHotMetalExt

using PrimitiveOneHot
using Metal

Metal.@device_override @noinline PrimitiveOneHot.throw_onehotencode_error(K, val) = Metal.@print_and_throw "OneHotEncodeError"


end
