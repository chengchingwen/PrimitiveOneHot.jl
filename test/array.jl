@testset "OneHotArray" begin
  idx = rand(5:10)
  ot = OneHot(10)
  o = OneHot{10}(idx)
  ind = rand(1:10, 5, 5)
  oa = OneHotArray{10}(ind)

  @test PrimitiveOneHot.onehotsize(oa) == 10
  @test OneHotArray(15, oa) |> PrimitiveOneHot.onehotsize == 15

  @test size(oa) == (10, 5, 5)
  @test oa[:, 5, 5] == oa.onehots[5, 5]
  @test oa[:, 5, :] == OneHotArray(oa.onehots[5, :])
  @test oa[ind[1], 1, 1]

  @test vcat(o, o) == vcat(collect(o), collect(o))
  @test hcat(o, o).onehots == [o,o]
  @test cat(o, o; dims=1) == vcat(collect(o), collect(o))
  @test cat(o, o; dims=2).onehots == [o,o]

  @test vcat(oa, oa) == vcat(collect(oa), collect(oa))
  @test hcat(oa, oa).onehots == vcat(oa.onehots, oa.onehots)
  @test cat(oa, oa; dims=1) == vcat(collect(oa), collect(oa))
  @test cat(oa, oa; dims=2).onehots == vcat(oa.onehots, oa.onehots)
  @test cat(oa, oa; dims=3).onehots == cat(oa.onehots, oa.onehots; dims=2)

  @test reshape(oa, 10, 25) isa OneHotArray
  @test reshape(oa, 10, :) isa OneHotArray
  @test reshape(oa, :, 25) isa OneHotArray
  @test reshape(oa, 50, :) isa Base.ReshapedArray{Bool, 2}
  @test reshape(oa, 5, 10, 5) isa Base.ReshapedArray{Bool, 3}

end
