@testset "OneHot primitive" begin
  using PrimitiveOneHot: OneHot
  idx = rand(5:10)
  ot = OneHot(10)
  o = OneHot{10}(idx)

  @test PrimitiveOneHot.onehotsize(o) == 10
  @test o === OneHot(10, idx)
  @test o === ot(idx)
  @test o == OneHot(11, idx)
  @test o !== OneHot(11, idx)

  @test size(o) == (10,)
  for i = 1:10
    @test o[i] == (i == idx)
  end
  @test o[:] === o
  @test_throws BoundsError o[11]

  @test UInt32(o) == UInt32(idx)
  @test UInt64(o) == UInt64(idx)
  @test Int32(o) == Int32(idx)
  @test Int64(o) == Int64(idx)

  @test one(o) < o

  @test convert(UInt32, o) == UInt32(idx)
  @test convert(UInt64, o) == UInt64(idx)
  @test convert(Int32, o) == Int32(idx)
  @test convert(Int64, o) == Int64(idx)
  @test convert(Int8, o) == Int8(idx)
  @test_throws InexactError convert(Bool, o)
  @test convert(ot, idx) == o
  @test convert(ot, UInt8(idx)) == o
  @test convert(ot, UInt16(idx)) == o
  @test convert(ot, UInt32(idx)) == o
  @test convert(ot, UInt64(idx)) == o
  @test convert(ot, UInt128(idx)) == o
  @test convert(ot, Int8(idx)) == o
  @test convert(ot, Int16(idx)) == o
  @test convert(ot, Int32(idx)) == o
  @test convert(ot, Int64(idx)) == o
  @test convert(ot, Int128(idx)) == o

  @test convert(ot, true) == OneHot(10, 1)
  @test convert(ot, false) == OneHot(10, 0)

  @test convert(OneHot(15), o) == OneHot(15, idx)
  @test convert(OneHot(idx+1), o) == OneHot(idx+1, idx)
  @test_throws PrimitiveOneHot.OneHotEncodeError convert(OneHot(idx-1), o)

  @test convert(Array, o) == begin
    z = zeros(Bool, 10)
    z[idx] = true
    z
  end
  @test reinterpret(Int32, o) == Int32(idx)

  @test one(o) === OneHot(10, 1)
  @test one(o) === one(typeof(o))
  @test isone(one(o))

  @test zero(o) === OneHot(10, 0)
  @test zero(o) === zero(typeof(o))
  @test !iszero(o)
  @test iszero(zero(o))

  @test_throws PrimitiveOneHot.OneHotEncodeError convert(ot, -1)
  @test_throws PrimitiveOneHot.OneHotEncodeError convert(ot, 12)
  @test_throws PrimitiveOneHot.OneHotEncodeError convert(ot, typemax(Int64))
  @test_throws PrimitiveOneHot.OneHotEncodeError convert(ot, typemax(UInt64))

  @test iszero(typemin(typeof(o)))
  @test Int(typemax(typeof(o))) == 10

  @test argmax(o) == argmax(collect(o))
  @test argmin(o) == argmin(collect(o))
  @test argmin(one(o)) == 2

end
