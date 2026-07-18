# @sizes — bind size variables and check shape consistency in one declaration.
#
#     @sizes begin
#         Q => (Dqk, Hq, B)
#         K => (Dqk, Lkv, Hkv, B)
#         V => (Dv, Lkv, Hkv, B)
#     end
#
# The first occurrence of a name binds it (`Dqk = size(Q, 1)`); every later
# occurrence checks it (`size(K, 1) == Dqk || throw(DimensionMismatch(...))`),
# so shared names *are* the consistency contract. `_` skips a dim; a literal
# or expression checks against its value (`R => (3, 3, N)`, `X => (1, B)`).
# Each array's ndims is checked against its tuple. A `nothing` array skips
# its whole line — optional operands mean "if present, must conform" — so
# names bound only there stay unbound, loudly, if used. Errors are
# `DimensionMismatch` — the Base convention for non-conforming extents
# (`mul!`, broadcast, `copyto!`) — and name the binding site:
# "size(K, 1) = 100 does not match Dqk = 576 (= size(Q, 1))".

macro sizes(ex)
    lines = Meta.isexpr(ex, :block) ? ex.args : [ex]
    bound = Dict{Symbol,String}()          # name => "size(Q, 1)", the binding site
    out = Expr(:block)
    for line in lines
        line isa LineNumberNode && continue
        Meta.isexpr(line, :call, 3) && line.args[1] === :(=>) ||
            error("@sizes: expected `A => (d1, d2, ...)`, got `$line`")
        arr, dims = line.args[2], line.args[3]
        Meta.isexpr(dims, :tuple) || error("@sizes: expected a size tuple, got `$dims`")
        name = string(arr)
        n = length(dims.args)
        entry = Expr(:block)

        push!(entry.args, :(
            ndims($(esc(arr))) == $n || throw(DimensionMismatch(string(
                $name, " has ", ndims($(esc(arr))), " dimensions, expected ", $n,
                " from ", $(string(dims)))))
        ))

        for (d, s) in enumerate(dims.args)
            site = "size($name, $d)"
            if s === :_
                continue
            elseif s isa Symbol && !haskey(bound, s)
                bound[s] = site
                push!(entry.args, :($(esc(s)) = size($(esc(arr)), $d)))
            else
                expected = s isa Symbol ? string(s) : string("`", s, "`")
                origin = s isa Symbol ? string(" (= ", bound[s], ")") : ""
                push!(entry.args, :(
                    size($(esc(arr)), $d) == $(esc(s)) || throw(DimensionMismatch(string(
                        $site, " = ", size($(esc(arr)), $d), " does not match ",
                        $expected, " = ", $(esc(s)), $origin)))
                ))
            end
        end

        push!(out.args, :($(esc(arr)) === nothing || $entry))
    end
    push!(out.args, :nothing)
    out
end
