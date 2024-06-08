@testset "Op" begin
    using NNlib: gather, gather!
    using Zygote
    src = Float32[3, 4, 5, 6, 7]
    index = Int32[
        1 2 3 4;
        4 2 1 3;
        3 5 5 3;
    ]
    output = Float32[
        3 4 5 6;
        6 4 3 5;
        5 7 7 5;
    ]

    oa = OneHotArray(10, index)
    @test gather(src, oa) == gather(src, index)
    @test gather!(similar(src, size(index)), src, oa) == output

    w = randn(3, 10)
    @test_throws DimensionMismatch randn(3,3) * oa
    @test w * oa == gather(w, oa)

    a = randn(5,  30)
    b = OneHotArray(30, ones(Int, 20))
    y = zeros(Float32, size(a))
    y[:, 1] .= 20
    @test Zygote.gradient((a, b)->sum(a * b), a, b)[1] == y
end
