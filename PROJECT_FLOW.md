# Fresh VANET-GNN Implementation Flow

## 1. Parse
Input:
- `data/raw/mobility.tcl`
- `data/raw/output.rou.xml`

Run:
`python -m src.main parse`

Create:
`data/interim/mobility_parsed.csv`

Expected core fields:
`time, vehicle_id, x, y, speed`

First verify the exact syntax of `mobility.tcl` before treating the parser output as experimental data.

## 2. Audit
Run:
`python -m src.main inspect`

Check:
- records
- unique vehicles
- time range
- time-step distribution
- missing values
- duplicates
- coordinate ranges
- speed ranges
- records per vehicle
- vehicle appearance/disappearance

## 3. Preprocess
- remove exact duplicates
- validate numeric fields
- handle missing values
- sort by vehicle and time
- preserve temporal ordering
- save `data/processed/mobility_clean.csv`

## 4. Time windows
Split the trace into configurable windows, for example:
- 0–9 s
- 10–19 s
- 20–29 s

The window size must be configurable and later tested as an experiment.

## 5. Dynamic graph
For each window:
- vehicle = node
- `(x, y)` = position
- distance between vehicles = graph relationship
- edge if distance <= configured communication range

Start with the paper's approximately 200 m threshold, then perform sensitivity experiments.

Do not use feature-space KNN now that actual positions are available.

## 6. Baselines
Implement:
1. K-Means
2. Spectral Clustering
3. Graph Autoencoder (GAE)

## 7. GraphSAGE
Initial architecture:
Input → SAGEConv → ReLU → Dropout → SAGEConv → ReLU → embedding

Then cluster the learned embeddings.

## 8. Cluster-head selection
Keep cluster formation and cluster-head selection as separate modules.
Use a reproducible rule based only on available information.

## 9. Stability evaluation
Because the data are time-indexed, investigate:
- cluster lifetime
- connectivity duration
- vehicle retention
- vehicles leaving clusters
- number of clusters over time
- cluster-head changes
- re-clustering frequency
- fragmentation
- overhead metrics supported by the data

Do not fabricate unsupported measurements.

## 10. Clustering metrics
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index

## 11. Comparison
Compare K-Means, Spectral, GAE and GraphSAGE under a common evaluation protocol.
For stochastic methods, use fixed seeds and repeated runs.

## 12. Ablation
Test:
- time-window size
- communication range
- cluster count
- GraphSAGE architecture
- feature subsets
- GAE vs GraphSAGE
- random seeds

## 13. Visualization
Generate:
- vehicle trajectories
- speed distributions
- graph snapshots
- cluster snapshots
- cluster lifetime distribution
- connectivity duration
- number of clusters over time
- vehicle departures
- cluster-head changes
- embedding plots
- model comparisons

## 14. Reproducibility
Save configuration, seed, preprocessing settings, graph settings, model parameters,
cluster labels, metrics and figures for every experiment.
