using PrimitiveOneHot
using Test

const tests = [
    "primitive",
    "array",
    "op",
]

function should_test_cuda()
    e = get(ENV, "JL_PKG_TEST_CUDA", false)
    e isa Bool && return e
    if e isa String
        x = tryparse(Bool, e)
        return isnothing(x) ? false : x
    else
        return false
    end
end

if should_test_cuda()
    push!(tests, "gpu")
end

@testset "PrimitiveOneHot.jl" begin
    for t in tests
        fp = joinpath(dirname(@__FILE__), "$t.jl")
        @info "Test $(uppercase(t))"
        include(fp)
    end
end
