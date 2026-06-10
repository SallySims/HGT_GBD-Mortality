# ============================================================
# 03_graph_summary.py
# HGT Spatial Dependence — GBD East Africa
# Graph Structure Summary Printer
#
# Part of: HGT_GBD-Mortality
# Repository: https://github.com/SallySims/HGT_GBD-Mortality
#
# Run order: 03 of 11
# Prerequisites: Run files 01 through 02 first
# ============================================================

# ============================================================
# CELL 3: GRAPH STRUCTURE SUMMARY (addresses reviewer clarity concern)
# ============================================================

def print_graph_summary(data, tag):
    """Print full node/edge/attribute dimensions for reproducibility."""
    print(f"\n{'='*60}")
    print(f"Graph Summary: {tag}")
    print(f"{'='*60}")
    print(f"Node types: {data.node_types}")
    for nt in data.node_types:
        print(f"  '{nt}' nodes: {data[nt].x.shape[0]}  feature_dim={data[nt].x.shape[1]}")
    print(f"Edge types: {data.edge_types}")
    for et in data.edge_types:
        et_str = str(et)
        ei = data[et].edge_index
        print(f"  {et_str}: {ei.shape[1]} edges", end="")
        if hasattr(data[et], 'edge_attr'):
            print(f"  attr_dim={data[et].edge_attr.shape[1]}", end="")
        print()
    print(f"{'='*60}")

print("Graph summary printer defined — will be called during model runs.")
