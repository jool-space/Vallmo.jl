using Vallmo
using Documenter

DocMeta.setdocmeta!(Vallmo, :DocTestSetup, :(using Vallmo); recursive=true)

makedocs(;
    modules=[Vallmo],
    authors="AntonOresten <antonoresten@proton.me> and contributors",
    sitename="Vallmo.jl",
    format=Documenter.HTML(;
        canonical="https://jool-space.github.io/Vallmo.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jool-space/Vallmo.jl",
    devbranch="main",
)
