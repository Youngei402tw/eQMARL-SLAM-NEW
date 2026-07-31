# Active-SLAM Faithful Baseline Handoff

Read this file first when starting a new Active-SLAM development session.
It records the protected baseline, the evidence obtained from it, why it
worked, and the remaining research questions. Do not overwrite this protocol
while developing improvements.

## Protected Baseline

- Name: `active-slam-faithful-pilot-v1`
- Implementation commit: `62a99cf3f4786644980d7c36e5d8052ce338b94a`
- Branch at preservation time: `main`
- Four methods only: eQMARL Psi+, qfCTDE, fCTDE, and sCTDE
- Pilot seeds: `0 1 2`
- Pilot budget: 400 episodes, at most 250 steps per episode
- Canonical output prefix: `experiment_output/active_slam_faithful_pilot_*`
- Full output prefix: `experiment_output/active_slam_faithful_full_*`

The Git tag `active-slam-faithful-pilot-v1` identifies the preserved code and
documentation. Use `git show active-slam-faithful-pilot-v1` to inspect it.
The four experiment YAML files are protected by
`test_active_slam_configs_follow_four_method_minigrid_protocol`; an accidental
change to the baseline parameters should fail the test suite.

The raw results are intentionally ignored by Git. Their tracked SHA-256
manifest is `docs/active_slam_faithful_pilot_v1.sha256`. Validate local files
from the repository root with:

```bash
sha256sum -c docs/active_slam_faithful_pilot_v1.sha256
```

The manifest checks integrity but is not a backup. Preserve the four
`active_slam_faithful_pilot_*` directories on separate storage before deleting
cluster or local artifacts.

## Supported Result

All 12 pilot jobs completed all 400 episodes. No run triggered the collapse
monitor. Seeds in directory names, resolved configurations, algorithm state,
environment state, model initialization, and episode resets are consistent.

Final-100 values below are the mean and seed standard deviation over seeds
0, 1, and 2.

| Method | Coverage | Success | Forward | Steps | IoU | Pose RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eQMARL Psi+ | 0.9036 +/- 0.0019 | 96% +/- 3.6% | 0.781 | 91 | 0.525 | 1.329 |
| qfCTDE | 0.9022 +/- 0.0020 | 95% +/- 2.6% | 0.782 | 90 | 0.521 | 1.380 |
| fCTDE | 0.7299 +/- 0.0908 | 17% +/- 16.8% | 0.314 | 236 | 0.368 | 2.495 |
| sCTDE | 0.7535 +/- 0.0338 | 22% +/- 11.1% | 0.345 | 230 | 0.391 | 2.291 |

Per-seed final-100 coverage and success:

| Method | Seed 0 | Seed 1 | Seed 2 |
| --- | ---: | ---: | ---: |
| eQMARL Psi+ | 0.9046 / 99% | 0.9014 / 92% | 0.9048 / 97% |
| qfCTDE | 0.9039 / 96% | 0.9000 / 92% | 0.9028 / 97% |
| fCTDE | 0.8224 / 35% | 0.6409 / 2% | 0.7263 / 13% |
| sCTDE | 0.7729 / 24% | 0.7731 / 32% | 0.7145 / 10% |

The defensible conclusion is:

> Under the faithful pilot protocol, eQMARL and qfCTDE are statistically
> similar, and both substantially outperform fCTDE and sCTDE.

Do not claim that Psi+ entanglement beats qfCTDE from this pilot. Their paired
final-100 differences are very small: eQMARL has +0.00138 coverage, +1
percentage point success, and 0.052 lower pose RMSE, while qfCTDE has 0.259
better episode reward. These differences are smaller than seed variation.

## The Four Methods

All methods use the same shared decentralized actor:

```text
147 inputs -> Dense(100, ReLU) -> Dense(3, Softmax)
```

Only the centralized training critic differs:

- eQMARL Psi+: eight qubits, four per robot, five partite layers, Psi+ input
  entanglement between robot partitions.
- qfCTDE: a centralized eight-qubit, five-layer critic without the Psi+ input
  state used by eQMARL.
- fCTDE: flatten both observations, Dense(100, ReLU), scalar value.
- sCTDE: separate per-robot Dense(100, ReLU) processing, late aggregation,
  scalar value.

Critic parameter counts are 3,817 for eQMARL/qfCTDE, 29,601 for fCTDE, and
29,801 for sCTDE.

## Environment And Reward

- Two robots on a procedurally generated 24x24 occupancy grid
- Nine obstacle rectangles and a 90% coverage success target
- 36 LiDAR beams, range 8, with LiDAR and odometry noise
- Three MiniGrid-ordered actions: left, right, forward
- Per-robot ego-centric observation: 7x7x3 = 147 normalized features
- Critic input: exactly the same two 147-feature robot observations
- Episode time limit: 250 steps

The shared step reward is:

```text
10.0 * coverage_gain
+ 0.5 * uncertainty_reduction
- 0.1 * collisions
- 0.01 step_cost
```

Covariance grows with motion process noise. A scan can reduce covariance only
when it observes new cells, and a step with no new cells cannot receive a
positive uncertainty-reduction reward. This rule is essential; do not remove
it without creating a separate ablation protocol.

## Training Contract

- Algorithm: shared-policy MAA2C
- Episodes/steps for full training: 1,000 / 250
- Discount: `gamma = 0.99`
- Actor Adam learning rate: `1e-4`
- Entropy coefficient: `0.01 -> 0.001` linearly over 500 episodes
- Team reward aggregation: mean, because both robots receive the same reward
- Advantage normalization: enabled
- Global gradient clip norm: 1.0
- Classical critic Adam learning rate: `1e-4`
- Quantum optimizer rates: `1e-3, 1e-3, 1e-2, 1e-2`
- Quantum encoder weights: not trainable
- Actor and critic checkpoints: every 100 episodes

Seeds are applied before model construction to Python, NumPy, and TensorFlow.
The standalone job override sets both algorithm and environment seeds. Episode
`e` resets the environment with `seed + e`. This makes method comparisons
paired on reproducible map sequences while retaining independent seeds.

## What We Did Correctly

### 1. Closed The Covariance Reward Loophole

Previously, repeated scans of known space could lower covariance and generate
reward without exploration. In the old eQMARL run, reward rose from 1.46 to
1.62 between the first and fourth 100-episode windows while coverage fell from
0.785 to 0.690 and success fell from 18% to 0%. The objective rewarded the
wrong behavior.

The new covariance and reward rules require new observations. New eQMARL
coverage rises from 0.807 to 0.904, success from 34% to 96%, and forward action
frequency from 34% to 78% over the pilot.

### 2. Preserved Exploration Long Enough To Learn

The entropy schedule starts at 0.01 instead of using 0.001 from episode zero.
For eQMARL, policy entropy moves from 0.992 to 0.840, 0.648, and 0.549 across
the four windows. Exploration declines gradually instead of collapsing into
turning before the critic becomes useful.

### 3. Corrected Reward Scale

The environment copies one team reward to both robots. Summing those copies
doubled TD targets. Mean aggregation represents one team reward and produces
the intended target scale.

### 4. Stabilized Optimization

Advantage normalization prevents episode length and reward scale from
arbitrarily changing actor update magnitude. Global norm clipping limits
outlier updates. Reducing the quantum readout rate from 0.1 to 0.01 avoids an
aggressive optimizer group that likely destabilized the earlier quantum
critics.

### 5. Restored The MiniGrid-Faithful Information Boundary

Every critic receives shape `(2, 147)`. We removed the duplicated 121-feature
global descriptor that gave both partitions the full map and both poses. This
restores the intended distinction between partite, centralized, and separated
critics and avoids compressing each 268-feature partition into 12 quantum
angles. The earlier `active_slam_minigrid_*` run also used 147 features, so
this correction is necessary but does not alone explain the improvement.

### 6. Made The Evidence Reproducible

Resolved configs, seed-specific directories, action diagnostics, optimizer
diagnostics, periodic checkpoints, and protocol-specific output roots make it
possible to audit the result. Seed 1 is the weakest seed for both quantum
methods, further confirming that paired environment difficulty is working.

## Remaining Issues

### 1. Full-Horizon Stability Is Unproven

The pilot ends at episode 400. Entropy reaches its final value only at episode
500. Earlier quantum policies degraded late in training, so the important next
test is whether performance remains stable through episode 1,000.

### 2. Entanglement Has Not Beaten qfCTDE

The pilot supports a quantum-critic advantage over the current classical
baselines, not an eQMARL-specific advantage over qfCTDE. More held-out seeds
are required, and a tie must be reported as a tie.

### 3. Baseline Optimizer Fairness Needs Attention

The classical critic uses `1e-4`; quantum parameter groups use rates up to
`1e-2`. After changing reward scale and normalization, the best classical
critic rate may have changed. Classical policies remain highly stochastic in
the final window (entropy about 0.93 versus 0.55 for quantum methods). A
reviewer may reasonably ask for validation-only classical LR tuning.

### 4. Reward Values Are Not Comparable Across Old And New Protocols

The covariance update changed reward semantics. New successful episodes can
have negative total reward. Compare coverage, success, IoU, pose error, and
behavior across protocols; do not interpret the old positive reward as better.

### 5. Statistical Power Is Still Small

Three pilot seeds are sufficient to reject the old single-seed conclusion but
not to establish small eQMARL-versus-qfCTDE differences. Final reporting needs
held-out seeds and confidence intervals over per-seed summaries.

### 6. The SLAM Backend Is A Research Abstraction

This is a transparent grid SLAM backend, not a production ROS/GTSAM system.
Claims should be scoped to active occupancy-grid exploration unless validated
on a more realistic simulator or dataset.

## Next Session Plan

First, reproduce the protected baseline without changing code:

```bash
git switch -c active-slam-improvements active-slam-faithful-pilot-v1
conda activate eQMARL_SLAM
python -m pytest -q
```

Run the confirmatory full comparison on held-out seeds, excluding pilot seeds
0, 1, and 2:

```bash
SLAM_FULL_SEEDS="3 4 5 6 7" bash scripts/submit_vanda_full.sh
```

Improvement work must follow these rules:

1. Do not edit the four faithful YAML files in place. Copy them to a clearly
   named improvement or ablation protocol.
2. Use a new output prefix so faithful, ablation, and full results never mix.
3. Change one factor at a time and use validation seeds separate from final
   evaluation seeds.
4. Keep all four requested methods; do not replace qfCTDE with a no-entanglement
   eQMARL variant.
5. Do not handicap baselines or select seeds based on the desired ranking.
6. Preserve coverage, success, action fractions, entropy, loss, gradient, and
   checkpoint diagnostics.
7. Keep every edited or newly created source/configuration file below 500
   lines. Do not rewrite the user-owned visualization notebook unnecessarily.

The first improvement question should be full-horizon stability. The second
should be whether validation-only optimizer tuning improves the classical
baselines. Only after those checks should circuit depth, beta, quantum rates,
or entropy schedules be tuned.
