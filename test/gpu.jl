@testset "CUDA" begin
    using Adapt
    using CUDA
    import LinearAlgebra
    
    oa = OneHotArray(10, rand(1:10, 5,5))
    @test collect(cu(oa)) == collect(oa)
    
    coa = cu(oa)
    @test adapt(Array, findmax(coa)) == findmax(oa)
    @test adapt(Array, findmax(coa; dims=1)) == findmax(oa; dims=1)
    @test adapt(Array, findmax(coa; dims=2)) == findmax(oa; dims=2)

    @test coa[:, :,  1] isa PrimitiveOneHot.CuOneHotArray
    
    @test adapt(Array, LinearAlgebra.tril(coa[:,:,1])) == LinearAlgebra.tril(collect(oa[:,:,1]))

end
