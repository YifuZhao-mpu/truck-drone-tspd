# §F held-out long-budget cross-check: ONE seeded replicate of one public comparator.
# Usage: julia --project=experiments/SOTA/julia run_heldout_comparator.jl \
#            <heldout_input.txt> <block> <dps|hgatac> <seed> <cap_s> <out.txt>
# Protocol (drafts/revision/revision_protocol.md §F, same shape as the archived v2
# matched comparison): default package settings; JIT warm-up on a throwaway instance OFF
# the clock; the package's RNG seeded once with Random.seed!(<seed>); complete solver runs
# launched back to back until the wall-clock cap; every inner run's cost and cumulative
# time appended to <out.txt> as soon as it finishes (so a killed process loses nothing
# already done).  The hard cutoff (inner runs finishing after the cap are excluded) is
# applied symmetrically by the Python driver.
# Output lines:  RUN <k> <cost> <cumulative_s>      one per completed inner run
#                BEST <cost> <truck_route> <drone_route>   routes of the best inner run,
#                comma-separated node indices exactly as the package returns them
using Random
using TSPDrone
using TSPDroneHGATAC

inp = ARGS[1]; want = parse(Int, ARGS[2]); solver = ARGS[3]
seed = parse(Int, ARGS[4]); cap = parse(Float64, ARGS[5]); out = ARGS[6]

function read_block(path, want)
    lines = readlines(path)
    i = 1; blk = 0
    while i <= length(lines)
        if isempty(strip(lines[i]))
            i += 1; continue
        end
        blk += 1
        if blk == want
            h = split(lines[i])
            id = h[2]; n = parse(Int, h[3]); tcf = parse(Float64, h[4]); dcf = parse(Float64, h[5])
            x = parse.(Float64, split(lines[i+1])[2:end])
            y = parse.(Float64, split(lines[i+2])[2:end])
            return id, n, tcf, dcf, x, y
        end
        i += 3
    end
    error("block $want not found in $path")
end

id, n, tcf, dcf, x, y = read_block(inp, want)

if solver == "dps"
    TSPDrone.solve_tspd(rand(10), rand(10), 1.0, 0.5)              # JIT warm-up, off the clock
    solve() = TSPDrone.solve_tspd(x, y, tcf, dcf)
    cost_of(r) = r.total_cost
    routes_of(r) = (r.truck_route, r.drone_route)
elseif solver == "hgatac"
    redirect_stdout(devnull) do
        TSPDroneHGATAC.solve_tspd(rand(10), rand(10); truck_speed=1.0, drone_speed=2.0)
    end
    solve() = redirect_stdout(devnull) do
        TSPDroneHGATAC.solve_tspd(x, y; truck_speed=1.0 / tcf, drone_speed=1.0 / dcf)
    end
    cost_of(r) = r.best_total_cost
    routes_of(r) = (r.best_truck_route, r.best_drone_route)
else
    error("unknown solver $solver")
end

Random.seed!(seed)
io = open(out, "w")
best = Inf; best_routes = nothing; k = 0
t0 = time()
while time() - t0 < cap
    global best, best_routes, k
    r = solve()
    el = time() - t0
    k += 1
    c = cost_of(r)
    println(io, "RUN ", k, " ", repr(c), " ", round(el, digits=3)); flush(io)
    if c < best && el <= cap          # routes kept only for a run inside the cap
        best = c; best_routes = routes_of(r)
    end
end
if best_routes !== nothing
    println(io, "BEST ", repr(best), " ", join(best_routes[1], ","), " ", join(best_routes[2], ","))
end
close(io)
