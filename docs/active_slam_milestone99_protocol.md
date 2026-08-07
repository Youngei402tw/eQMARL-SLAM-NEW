# Active-SLAM Milestone99 Protocol

This protocol is an isolated successor to the faithful and bounded-pose
experiments. It does not modify their YAML files or output directories.

## Objective

- Maximum episode length: 250 steps
- Success and termination: 99% reachable-cell coverage
- Bounded SLAM pose estimates: enabled
- Shared dense reward:

```text
10.0 * coverage_gain
+ 0.5 * uncertainty_reduction
- 0.1 * collisions
- 0.01 step_cost
```

- One-time coverage bonuses:

```text
90% -> +0.5
95% -> +1.0
98% -> +2.0
99% -> +5.0 and terminate
```

A bonus is paid only when a transition first crosses its threshold. Reaching
multiple thresholds in one transition pays each newly crossed bonus.

## Seed Contract

For training seed `S`, episode `e` uses procedural map seed `S + e`. All four
methods use the same map for the same training seed and episode, which makes
framework comparisons paired.

The default seed blocks are spaced so their map ranges do not overlap:

```text
pilot: 13000 14000 15000
full:  16000 17000 18000 19000 20000
```

With 1,000 full episodes, seed 16000 uses maps 16000 through 16999, seed
17000 uses maps 17000 through 17999, and so on. The submission script rejects
custom seed lists whose map ranges overlap.

## Cluster Training

Submit pilot jobs:

```bash
bash scripts/submit_vanda_milestone99.sh pilot
```

Submit full jobs:

```bash
bash scripts/submit_vanda_milestone99.sh full
```

The submission command starts a local PBS monitor that periodically releases
user-held `slam-m99-*` jobs. Run the monitor manually when needed:

```bash
bash scripts/submit_vanda_milestone99.sh monitor
```

The cluster worker is `train_vanda_milestone99.pbs`. Cluster jobs only train
and save results; analysis remains local after result migration.

## Outputs And Analysis

Full results use these isolated prefixes:

```text
experiment_output/active_slam_milestone99_full_eqmarl_psi+
experiment_output/active_slam_milestone99_full_qfctde
experiment_output/active_slam_milestone99_full_fctde
experiment_output/active_slam_milestone99_full_sctde
```

After migrating all 20 full runs, audit them locally:

```bash
CUDA_VISIBLE_DEVICES=-1 python scripts/analyze_active_slam_full.py \
  --protocol milestone99
```

The visualization notebook automatically prefers complete milestone99 full
results when they are present. Saved metrics include reward components and
the first step reaching 90%, 95%, 98%, and 99%; `-1` means the threshold was
not reached before termination or truncation.
