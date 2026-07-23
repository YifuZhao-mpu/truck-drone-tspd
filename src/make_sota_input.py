"""Convert instances.json -> sota_input.txt (plain text, no Julia JSON dependency)."""
import os, json
D = os.path.join(os.path.dirname(__file__), "..", "experiments", "SOTA", "results")
items = json.load(open(os.path.join(D, "instances.json")))
with open(os.path.join(D, "sota_input.txt"), "w") as f:
    for it in items:
        f.write(f"ID {it['id']} {it['n']} {it['truck_cost_factor']} {it['drone_cost_factor']}\n")
        f.write("X " + " ".join(repr(v) for v in it["x"]) + "\n")
        f.write("Y " + " ".join(repr(v) for v in it["y"]) + "\n")
print(f"wrote sota_input.txt ({len(items)} instances)")
