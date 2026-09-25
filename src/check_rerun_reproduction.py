"""Independently re-run every archived-field reproduction gate on v3 artifacts."""
import json
import os

from run_h1_v3 import reproduction_gate as h1_gate
from run_h2_v3 import reproduction_gate as h2_gate
from run_h3_v3 import reproduction_gate as h3_gate, winner_row
from run_p0_v3 import reproduction_gate as p0_gate


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
PATHS = {
    "H1": os.path.join(ROOT, "experiments", "H1-alns-backbone", "results", "h1_v3.json"),
    "H2": os.path.join(ROOT, "experiments", "H2-design-space", "results", "h2_v3.json"),
    "H3": os.path.join(ROOT, "experiments", "H3-time-energy", "results", "h3_v3.json"),
    "P0": os.path.join(ROOT, "experiments", "P0-robustness", "results", "p0_v3.json"),
}


def load(name):
    with open(PATHS[name], encoding="utf-8") as handle:
        return json.load(handle)


def main():
    h1 = load("H1")
    h2 = load("H2")
    h3 = load("H3")
    p0 = load("P0")
    gates = {
        "H1": h1_gate(h1["benchmark_rows"], h1["synthetic_rows"], 160),
        "H2": h2_gate(h2["rows"], 3600),
        "H3": h3_gate([winner_row(row) for row in h3["rows"]], 1260),
        "P0": p0_gate(p0["rows"], 1200),
    }
    if any(gate["status"] != "PASS" for gate in gates.values()):
        raise RuntimeError(gates)
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
