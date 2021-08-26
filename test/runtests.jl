using PrimitiveOneHot
using Test

const tests = [
    "primitive",
    "array",
]

@testset "PrimitiveOneHot.jl" begin
    for t in tests
        fp = joinpath(dirname(@__FILE__), "$t.jl")
        @info "Test $(uppercase(t))"
        include(fp)
    end
end
