# Matched-budget TSPDrone.jl side. Usage: julia run_sota_matched.jl <sota_input.txt> <out.txt> <id>
# Solves ONE instance (the <id>-th block), repeat-until-budget, all replicates recorded.
using TSPDrone

inp = ARGS[1]; out = ARGS[2]; want = parse(Int, ARGS[3])
budget(n, kind) = kind == "B" ? 30.0 : (n <= 21 ? 35.0 : (n <= 51 ? 140.0 : 430.0))
lines = readlines(inp)
i = 1; blk = 0
while i <= length(lines)
    global i, blk
    isempty(strip(lines[i])) && (i += 1; continue)
    blk += 1
    if blk == want
        h = split(lines[i])
        id = h[2]; n = parse(Int, h[3]); tcf = parse(Float64, h[4]); dcf = parse(Float64, h[5])
        kind = startswith(id, "bench") ? "B" : "S"
        x = parse.(Float64, split(lines[i+1])[2:end])
        y = parse.(Float64, split(lines[i+2])[2:end])
        solve_tspd(rand(10), rand(10), 1.0, 0.5)          # JIT warmup outside the clock
        T = budget(n, kind)
        t0 = time(); best = Inf; reps = String[]
        while time() - t0 < T
            r = solve_tspd(x, y, tcf, dcf)
            el = time() - t0
            best = min(best, r.total_cost)
            push!(reps, string(r.total_cost, ":", round(el, digits=2)))
        end
        open(out, "w") do io
            println(io, id, " ", best, " ", round(time() - t0, digits=2), " ",
                    length(reps), " ", join(reps, ","))
        end
        exit(0)
    end
    i += 3
end
error("block not found")
