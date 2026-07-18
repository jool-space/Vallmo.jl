# splitaxis — carve a packed axis into a tuple of views: the slot layout
# as one declaration. Equal sections or explicit sizes. Views of views
# collapse (SubArray reindexes into the parent), so splits compose with
# reshape without stacking wrappers. `@constprop :aggressive` lets literal
# call sites constant-fold the Vals, so the tuple length infers without
# Val ceremony at the call.
#
#     gate, up = splitaxis(gu, 2)
#     q, k, v  = splitaxis(conv3, (HkL, HkL, HvL); dims = 2)

function _split(x::AbstractArray{<:Any,N}, ::Val{sections}, ::Val{dims}) where {N,sections,dims}
    d = size(x, dims)
    s, r = divrem(d, sections)
    iszero(r) || throw(DimensionMismatch(
        "size(x, $dims) = $d does not divide into $sections sections"))
    return ntuple(
        i -> view(x, ntuple(j -> j == dims ? (s*(i-1)+1:s*i) : (:), Val(N))...),
        Val(sections))
end

function _split(x::AbstractArray{<:Any,N}, sizes::NTuple{S,Int}, ::Val{dims}) where {N,S,dims}
    d = size(x, dims)
    cs = cumsum(sizes)
    cs[end] == d || throw(DimensionMismatch(
        "sum(sizes) = $(cs[end]) does not match size(x, $dims) = $d"))
    inds = (0, cs...)
    return ntuple(Val(S)) do k
        i, j = inds[k], inds[k+1]
        view(x, ntuple(n -> n == dims ? (i+1:j) : (:), Val(N))...)
    end
end

Base.@constprop :aggressive splitaxis(x, sections::Int; dims::Int = 1) =
    _split(x, Val(sections), Val(dims))

Base.@constprop :aggressive splitaxis(x, sizes::NTuple{S,Int}; dims::Int = 1) where {S} =
    _split(x, sizes, Val(dims))
