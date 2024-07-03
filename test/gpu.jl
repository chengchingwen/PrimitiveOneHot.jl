@testset "GPU" begin
    using Adapt
    import LinearAlgebra

    oa = OneHotArray(10, rand(1:10, 5,5))
    @test collect(device(oa)) == collect(oa)
    @test device(copy(oa)) == copy(device(oa))

    coa = device(oa)
    @test adapt(Array, findmax(coa)) == findmax(oa)
    @test adapt(Array, findmax(coa; dims=1)) == findmax(oa; dims=1)
    @test adapt(Array, findmax(coa; dims=2)) == findmax(oa; dims=2)
    @test adapt(Array, findmax(coa; dims=3)) == findmax(oa; dims=3)

    @test coa[:, :,  1] isa OneHotArray && parent(coa[:, :,  1]) isa GpuArray
    
    @test adapt(Array, LinearAlgebra.tril(coa[:,:,1])) == LinearAlgebra.tril(collect(oa[:,:,1]))

    @test sprint(Base.print_array, coa) == sprint(Base.print_array, oa)

    using Zygote
    ca = device(randn(5,  30))
    cb = device(OneHotArray(30, ones(Int, 20)))

    fa = zeros(Float32, size(ca))
    fa[:, 1] .= 20
    @test collect(Zygote.gradient(pca->sum(pca * cb), ca)[1]) == fa
end
