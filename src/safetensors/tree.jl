# tree.jl — checkpoint keys → a tree of weights.
#
# HF checkpoint keys are already paths: "model.layers.0.self_attn.q_proj.weight".
# This converts them, once and eagerly, into the tree they spell — dotted
# segments become nested `Tree`s, runs of integer segments become Vectors
# (0-indexed keys, 1-indexed access). `Tree` is a NamedTuple wearing a
# readable `show`: construction stays anonymous and concrete, and per-model
# loaders stop existing.
#
#     weights = tree(RawTensors(path))
#     weights.model.layers[1].self_attn.q_proj.weight

"""
    Tree

A NamedTuple with tree-shaped printing. Nodes are `Tree`s or Vectors of
`Tree`s; leaves are whatever the checkpoint held. Access is by property:
`t.model.layers[1].self_attn.q_proj.weight`.
"""
struct Tree{T<:NamedTuple}
    fields::T
end

Base.getproperty(t::Tree, name::Symbol) = getfield(t, :fields)[name]
Base.propertynames(t::Tree) = keys(getfield(t, :fields))
Base.pairs(t::Tree) = pairs(getfield(t, :fields))
Base.:(==)(a::Tree, b::Tree) = getfield(a, :fields) == getfield(b, :fields)

tree(d::AbstractDict{String}) = _tree([split(k, '.') => v for (k, v) in d])

function _tree(entries)
    if any(isempty(first(e)) for e in entries)
        length(entries) == 1 || error("key is both a leaf and a prefix: $(entries)")
        return last(only(entries))
    end
    groups = Dict{SubString{String},Vector{eltype(entries)}}()
    for (segments, v) in entries
        push!(get!(Vector{eltype(entries)}, groups, first(segments)), segments[2:end] => v)
    end
    if all(!isnothing(tryparse(Int, k)) for k in keys(groups))
        0:length(groups)-1 == sort!(parse.(Int, collect(keys(groups)))) ||
            error("integer keys are not contiguous from 0: $(sort(collect(keys(groups))))")
        return [_tree(groups[string(i)]) for i in 0:length(groups)-1]
    end
    return Tree((; (Symbol(k) => _tree(groups[k]) for k in sort!(collect(keys(groups))))...))
end

# ── printing ─────────────────────────────────────────────────────────────────

Base.show(io::IO, t::Tree) = print(io, "Tree(", join(propertynames(t), ", "), ")")

function Base.show(io::IO, ::MIME"text/plain", t::Tree)
    print(io, "Tree")
    _show(io, t, "")
end

function _show(io::IO, t::Tree, prefix::String)
    ps = pairs(t)
    for (i, (name, v)) in enumerate(ps)
        last = i == length(ps)
        print(io, '\n', prefix, last ? "└─ " : "├─ ", name)
        _shownode(io, v, prefix * (last ? "   " : "│  "))
    end
end

_shownode(io::IO, t::Tree, prefix::String) = _show(io, t, prefix)
_shownode(io::IO, x, prefix::String) = print(io, " = ", summary(x))

function _shownode(io::IO, v::Vector{<:Tree}, prefix::String)
    print(io, "[", length(v), "], each:")   # homogeneous by construction; show the shape once
    _show(io, first(v), prefix)
end
