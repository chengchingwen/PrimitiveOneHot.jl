using Test, Pkg
function testing_gpu()
    e = get(ENV, "JL_PKG_TEST_GPU", nothing)
    isnothing(e) && return nothing
    if e isa String
        x = lowercase(e)
        if isempty(x)
            return nothing
        elseif x == "cuda"
            return :cuda
        elseif x == "amdgpu"
            return :amdgpu
        elseif x == "metal"
            return :metal
        end
    end
    error("Unknown value for `JL_PKG_TEST_GPU`: $x")
end

const GPUBACKEND = testing_gpu()
if isnothing(GPUBACKEND)
    const USE_GPU = false
else
    const USE_GPU = true
    if GPUBACKEND == :cuda
        Pkg.add(["CUDA"])
        using CUDA
        CUDA.allowscalar(false)
        device(x) = cu(x)
        const GpuArray = CuArray
    elseif GPUBACKEND == :amdgpu
        Pkg.add(["AMDGPU"])
        using AMDGPU
        AMDGPU.allowscalar(false)
        device(x) = roc(x)
        const GpuArray = RocArray
    elseif GPUBACKEND == :metal
        Pkg.add(["Metal"])
        using Metal
        Metal.allowscalar(false)
        device(x) = mtl(x)
        const GpuArray = MtlArray
    end
end
@show GPUBACKEND
@show USE_GPU

using PrimitiveOneHot

const tests = [
    "primitive",
    "array",
    "op",
]

if USE_GPU
    push!(tests, "gpu")
end

@testset "PrimitiveOneHot.jl" begin
    for t in tests
        fp = joinpath(dirname(@__FILE__), "$t.jl")
        @info "Test $(uppercase(t))"
        include(fp)
    end
end
