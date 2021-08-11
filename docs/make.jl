using PrimitiveOneHot
using Documenter

DocMeta.setdocmeta!(PrimitiveOneHot, :DocTestSetup, :(using PrimitiveOneHot); recursive=true)

makedocs(;
    modules=[PrimitiveOneHot],
    authors="chengchingwen <adgjl5645@hotmail.com> and contributors",
    repo="https://github.com/chengchingwen/PrimitiveOneHot.jl/blob/{commit}{path}#{line}",
    sitename="PrimitiveOneHot.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://chengchingwen.github.io/PrimitiveOneHot.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/chengchingwen/PrimitiveOneHot.jl",
)
