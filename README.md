# VANET-GNN Stable Clustering — Fresh Implementation

Fresh implementation workspace for the Major Project:
**GNN-Based Stable Clustering for Intelligent Vehicular Networks (VANETs).**

## Dataset
- `data/raw/mobility.tcl`
- `data/raw/output.rou.xml`

The mobility trace is time-indexed and contains vehicle movement information,
allowing dynamic VANET graph construction and temporal stability analysis.

## Pipeline
Raw mobility files
→ parser
→ data audit
→ preprocessing
→ time windows
→ dynamic distance-based graphs
→ K-Means / Spectral / GAE baselines
→ GraphSAGE
→ clustering
→ cluster-head selection
→ stability evaluation
→ comparison
→ ablation studies
→ final results

Experimental results must be generated fresh; do not copy results from the paper.
