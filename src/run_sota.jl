# Run TSPDrone.jl (HGA-TAC lineage) on the exported instances. Plain-text I/O (no JSON dep).
# Usage: julia run_sota.jl <sota_input.txt> <sota_output.txt>
using TSPDrone

inp = ARGS[1]; out = ARGS[2]
lines = readlines(inp)
open(out, "w") do io
    i = 1
    while i <= length(lines)
        isempty(strip(lines[i])) && (i += 1; continue)
        h = split(lines[i])                       # ID id n tcf dcf
        id = h[2]; tcf = parse(Float64, h[4]); dcf = parse(Float64, h[5])
        x = parse.(Float64, split(lines[i+1])[2:end])
        y = parse.(Float64, split(lines[i+2])[2:end])
        t = @elapsed r = solve_tspd(x, y, tcf, dcf)
        println(io, id, " ", r.total_cost, " ", t)
        flush(io)
        println(stderr, "done ", id, " cost=", round(r.total_cost, digits=4), " (", round(t, digits=2), "s)")
        i += 3
    end
end
println(stderr, "ALL DONE")
