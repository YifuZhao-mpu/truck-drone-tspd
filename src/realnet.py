"""
Real road-network instances from OpenStreetMap (round-7 roadmap item: at least one
realistic road/demand case).

Two ~1.2 km-radius urban districts with contrasting street patterns:
 - "manhattan": Midtown Manhattan, New York (regular grid)
 - "paris": north-east central Paris (organic/radial pattern)

Truck travel = shortest-path distance on the (undirected, largest-component) drive
network; drone travel = straight-line Euclidean. Coordinates are projected to meters and
normalized by the largest bounding-box side, so alpha and endurance are directly
comparable to the synthetic unit-square experiments (E=1.0 corresponds to a round-trip
flight budget of ~one district diameter, ~2.4 km here). One-way restrictions are ignored
(undirected simplification, disclosed). Customers and the depot are sampled from road
nodes (seeded); the depot is the node nearest the district centroid, matching the
synthetic central-depot convention.
"""
import os

import networkx as nx
import numpy as np

DISTRICTS = {
    "manhattan": {"point": (40.7549, -73.9840), "dist": 1200},
    "paris": {"point": (48.8630, 2.3770), "dist": 1200},
}
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "realnet")


def get_district(name):
    """Load (from cache) or fetch the projected, undirected, largest-component graph."""
    import osmnx as ox
    ox.settings.use_cache = True
    ox.settings.cache_folder = os.path.join(CACHE_DIR, "osm_cache")
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{name}.graphml")
    if os.path.exists(path):
        G = ox.load_graphml(path)
    else:
        cfg = DISTRICTS[name]
        G = ox.graph_from_point(cfg["point"], dist=cfg["dist"], network_type="drive",
                                simplify=True)
        G = ox.project_graph(G)
        ox.save_graphml(G, path)
    Gu = nx.Graph()
    for u, v, data in G.edges(data=True):
        w = float(data.get("length", 1.0))
        if Gu.has_edge(u, v):
            if w < Gu[u][v]["length"]:
                Gu[u][v]["length"] = w
        else:
            Gu.add_edge(u, v, length=w)
    for node, data in G.nodes(data=True):
        if node in Gu:
            Gu.nodes[node]["x"] = float(data["x"])
            Gu.nodes[node]["y"] = float(data["y"])
    comp = max(nx.connected_components(Gu), key=len)
    return Gu.subgraph(comp).copy()


def build_instance(G, n, seed):
    """Sample depot (centroid-nearest node) + n customer nodes; return coords normalized
    by the largest bbox side, truck shortest-path matrix, drone Euclidean matrix, and the
    scale (meters per normalized unit)."""
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes)
    xs = np.array([G.nodes[v]["x"] for v in nodes])
    ys = np.array([G.nodes[v]["y"] for v in nodes])
    scale = float(max(xs.max() - xs.min(), ys.max() - ys.min()))
    cx, cy = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2
    depot = nodes[int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))]
    cust_pool = [v for v in nodes if v != depot]
    cust = list(rng.choice(len(cust_pool), size=n, replace=False))
    chosen = [depot] + [cust_pool[i] for i in cust]
    coords = np.array([[(G.nodes[v]["x"] - xs.min()) / scale,
                        (G.nodes[v]["y"] - ys.min()) / scale] for v in chosen])
    # truck: shortest-path distances between all chosen nodes (Dijkstra per source)
    idx = {v: i for i, v in enumerate(chosen)}
    Dtr = np.zeros((n + 1, n + 1))
    for v in chosen:
        dist = nx.single_source_dijkstra_path_length(G, v, weight="length")
        for w in chosen:
            Dtr[idx[v], idx[w]] = dist[w] / scale
    Dtr = (Dtr + Dtr.T) / 2.0  # numerical symmetry
    diff = coords[:, None, :] - coords[None, :, :]
    Ddr = np.sqrt((diff ** 2).sum(-1))
    inst = {"coords": coords, "n": n, "seed": int(seed), "depot": "center",
            "kind": "realnet"}
    return inst, Dtr, Ddr, scale


def circuity(Dtr, Ddr):
    """Mean truck/drone distance ratio over all distinct pairs (road circuity factor)."""
    m = Ddr > 1e-9
    return float((Dtr[m] / Ddr[m]).mean())
