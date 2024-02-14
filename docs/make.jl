using PrimitiveOneHot
using Documenter

DocMeta.setdocmeta!(PrimitiveOneHot, :DocTestSetup, :(using PrimitiveOneHot); recursive=true)

makedocs(;
    modules=[PrimitiveOneHot],
    authors="chengchingwen <chengchingwen214@gmail.com> and contributors",
    sitename="PrimitiveOneHot.jl",
    format=Documenter.HTML(;
        canonical="https://chengchingwen.github.io/PrimitiveOneHot.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/chengchingwen/PrimitiveOneHot.jl",
    devbranch="master",
)
