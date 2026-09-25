# §F pre-gate smoke test: both comparators on two random instances, seeded, single thread.
using Random, TSPDrone, TSPDroneHGATAC
for inst in 1:2
    Random.seed!(1000 + inst)
    x = rand(10); y = rand(10)
    Random.seed!(1)
    t0 = time(); r1 = TSPDrone.solve_tspd(x, y, 1.0, 0.5); t1 = time() - t0
    Random.seed!(1)
    t0 = time(); r2 = redirect_stdout(devnull) do
        TSPDroneHGATAC.solve_tspd(x, y; truck_speed=1.0, drone_speed=2.0)
    end; t2 = time() - t0
    Random.seed!(1)
    r1b = TSPDrone.solve_tspd(x, y, 1.0, 0.5)
    println("instance ", inst, " DPS cost ", r1.total_cost, " (", round(t1, digits=2), " s; reseeded repeat ", r1b.total_cost,
            ") HGA-TAC cost ", r2.best_total_cost, " (", round(t2, digits=2), " s)")
end
println("threads ", Threads.nthreads(), " julia ", VERSION)
