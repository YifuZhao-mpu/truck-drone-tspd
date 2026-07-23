"""
Drone energy model (Dorling et al. 2017, affine power-in-payload).
Power P = a*mass + b; energy on a leg = P * flight_time. Drone carries payload outbound
(launch->customer), returns empty (customer->rendezvous). Truck energy ~ distance.

energy_drone(order, ops, Dt, alpha) -> total drone energy over all sorties.
evaluate_solution(...) -> full audit of one decoded solution (time, distances, energies),
computed directly from the ops with the correct matrix per vehicle (truck=Dtr, drone=Ddr);
this is the single evaluation path used by every v2 experiment driver.
"""
import numpy as np

A = 1.0    # power coefficient per unit payload mass
B = 1.0    # base (frame+avionics) power
PAYLOAD = 1.0  # unit payload per drone customer (homogeneous parcels)
TRUCK_E = 0.3  # truck energy per unit distance (for reporting / combined accounting)


def energy_drone(order, ops, Dt, alpha):
    seq = order
    e = 0.0
    for op in ops:
        if op[0] == "sortie":
            i, j, k = op[1], op[2], op[3]
            t_out = Dt[seq[i], seq[j]] / alpha
            t_back = Dt[seq[j], seq[k]] / alpha
            e += (A * PAYLOAD + B) * t_out + B * t_back
        elif op[0] == "msortie":
            i, S, k = op[1], op[2], op[3]
            for j in S:
                t_out = Dt[seq[i], seq[j]] / alpha
                t_back = Dt[seq[j], seq[k]] / alpha
                e += (A * PAYLOAD + B) * t_out + B * t_back
    return e


def truck_distance(order, ops, Dt):
    seq = order
    d = 0.0
    for op in ops:
        if op[0] == "truck":
            d += Dt[seq[op[1]], seq[op[2]]]
        elif op[0] in ("sortie", "msortie"):
            i, k = op[1], op[-1]
            skip = set([op[2]]) if op[0] == "sortie" else set(op[2])
            kept = [p for p in range(i, k + 1) if p not in skip]
            d += sum(Dt[seq[kept[a]], seq[kept[a + 1]]] for a in range(len(kept) - 1))
    return d


def evaluate_solution(order, ops, Dtr, Ddr, alpha, cl=None, cb=None, te=None, payload=None):
    """
    Full audit of a decoded solution. order = [0, ..., 0]; ops from any split decoder
    (truck / sortie / msortie). Truck legs use Dtr, drone legs use Ddr (two-matrix safe).
    Energy: drone power = A*payload + B loaded (outbound), B empty (return); truck energy
    = TRUCK_E per unit distance. Returns dict with makespan, truck_dist, drone_dist,
    e_drone, e_truck, e_total, n_sorties.
    """
    cl = A * (PAYLOAD if payload is None else payload) if cl is None else cl
    cb = B if cb is None else cb
    te = TRUCK_E if te is None else te
    seq = order
    ms = 0.0
    truck_d = 0.0
    drone_d = 0.0
    e_drone = 0.0
    n_sorties = 0
    for op in ops:
        if op[0] == "truck":
            i, k = op[1], op[2]
            leg = Dtr[seq[i], seq[k]]
            ms += leg
            truck_d += leg
        else:
            i, k = op[1], op[-1]
            js = (op[2],) if op[0] == "sortie" else tuple(op[2])
            n_sorties += len(js)
            kept = [p for p in range(i, k + 1) if p not in set(js)]
            tr = sum(Dtr[seq[kept[a]], seq[kept[a + 1]]] for a in range(len(kept) - 1))
            truck_d += tr
            dmax = 0.0
            for j in js:
                out_d = Ddr[seq[i], seq[j]]
                back_d = Ddr[seq[j], seq[k]]
                drone_d += out_d + back_d
                t_out = out_d / alpha
                t_back = back_d / alpha
                e_drone += cl * t_out + cb * (t_out + t_back)
                if t_out + t_back > dmax:
                    dmax = t_out + t_back
            ms += max(tr, dmax)
    e_truck = te * truck_d
    return {"makespan": ms, "truck_dist": truck_d, "drone_dist": drone_d,
            "e_drone": e_drone, "e_truck": e_truck, "e_total": e_drone + e_truck,
            "n_sorties": n_sorties}
