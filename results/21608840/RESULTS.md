# Campaign results — SLURM run 21608840

| | |
|---|---|
| commit | `eefcc1dff0c41e5ca3b8a35cdfd844cc34e1cf63` |
| branch | `main` |
| dirty | `no` |
| cluster | NCSA Delta, partition `cpu`, account `bhio-delta-cpu` |
| environment | rooted4d 0.11.0, numpy 2.3.5, python 3.12.13 |

| experiment | runtime |
|---|---:|
| `t1_annular_resolution` | 1.9 s |
| `t2_soft_attachment` | 0.3 s |
| `t3_gauge_ablation` | 0.0 s |
| `p4_periodicity_phase` | 0.1 s |
| `t5_identifiability` | 0.5 s |
| `t6_noise_stability` | 0.2 s |
| `e1_synthetic_recovery` | 10.0 s |
| `e5_baseline_ladder` | 346.6 s |
| `t5t6_solver_jacobian` | 26.1 s |
| `e6_image_domain` | 466.7 s |
| `e9_posterior_calibration` | 144.2 s |
| `e10_family_generalization` | 154.8 s |

## T1 — exact attachment across annular densities

| nodes | local | physical | equality rows | rank | null-space | residual |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 61 | 37 | 72 | 72 | 111 | 0 mm |
| 24 | 121 | 73 | 144 | 144 | 219 | 0 mm |
| 32 | 161 | 97 | 192 | 192 | 291 | 0 mm |
| 48 | 241 | 145 | 288 | 288 | 435 | 0 mm |
| 64 | 321 | 193 | 384 | 384 | 579 | 0 mm |
| 96 | 481 | 289 | 576 | 576 | 867 | 0 mm |

## E5 — locked baseline ladder

| arm | max attach (mm) | held-out CMR | held-out TEE | phase (cyc) | alpha | identified | failures |
|---|---:|---:|---:|---:|---:|---|---:|
| `cmr_only` | 0 | 0.5744 | 12.28 | - | 0 | no | 0 |
| `tee_only` | 0 | 0.7081 | 0.7337 | 0.308 | 0.2997 | yes | 0 |
| `late_fusion` | 0 | 0.5744 | 0.7337 | 0.0733 | - | - | 0 |
| `joint_soft_attachment` | 0.1196 | 0.8405 | 0.8288 | 0.3773 | 0.275 | yes | 0 |
| `joint_exact_no_gauges` | 0 | 0.5678 | 0.7277 | 0.01183 | 0 | no | 0 |
| `rooted4d_full` | 0 | 0.5675 | 0.7275 | 0.01167 | 2.207 | yes | 0 |

Medians over all problems. Paired comparisons against the full model:

| comparison | mean difference | 95% CI | p | favours |
|---|---:|---|---:|---|
| `full_vs_late_fusion__heldout_cmr_mm` | -0.0060 | [-0.0086, -0.0039] | 0.0001 | rooted4d_full |
| `full_vs_late_fusion__heldout_tee_mm` | -0.0044 | [-0.0058, -0.0031] | 0.0001 | rooted4d_full |
| `full_vs_joint_soft_attachment__heldout_cmr_mm` | -0.2734 | [-0.3175, -0.2291] | 0.0001 | rooted4d_full |
| `full_vs_joint_soft_attachment__heldout_tee_mm` | -0.1229 | [-0.1484, -0.1000] | 0.0001 | rooted4d_full |
| `full_vs_joint_exact_no_gauges__heldout_cmr_mm` | +0.0006 | [-0.0002, +0.0014] | 0.1583 | joint_exact_no_gauges |
| `full_vs_joint_exact_no_gauges__heldout_tee_mm` | +0.0005 | [-0.0002, +0.0012] | 0.2107 | joint_exact_no_gauges |
| `full_vs_tee_only__heldout_cmr_mm` | -0.1622 | [-0.2004, -0.1258] | 0.0001 | rooted4d_full |
| `full_vs_tee_only__heldout_tee_mm` | -0.0044 | [-0.0058, -0.0031] | 0.0001 | rooted4d_full |

Weakly identified arms (alpha far below the full model): `cmr_only`, `joint_exact_no_gauges`, `joint_soft_attachment`, `tee_only`

> An arm whose smallest non-gauge eigenvalue is far below the full model's sits in a nearly flat valley, so which point the regularizer settles on depends on floating-point rounding. Observed directly: across two machines the tee_only arm reproduced its held-out TEE error to 1e-13 but its held-out CMR extrapolation to only 1e-1. Cross-modal numbers from weakly identified arms should be read as indicative, not as reproducible measurements.

> A null result for full versus joint_exact_no_gauges on held-out error is the predicted outcome, not a failure. Theorem 3 concerns uniqueness of the identity/motion/session split conditional on the trajectories; removing the gauges leaves the trajectory space, and therefore the predictions, unchanged. What the gauges buy is visible in the alpha column instead: without them the identity and the common session mode trade off freely, so the arm's smallest non-gauge eigenvalue collapses to zero and the decomposition is unidentifiable even though it predicts just as well. That is the purpose the baseline lock assigns this arm, to test identifiability rather than generic fusion. Parameter recovery is measured separately in t3_gauge_ablation.

## E6 — image-domain refinement

| quantity | before | after | improved in |
|---|---:|---:|---:|
| heldout_cmr_planes | 0.01007 | 4.95e-04 | 100% of seeds |
| heldout_tee_phases | 0.04951 | 0.02544 | 100% of seeds |
| translation_mm | 2.742 | 0.7601 | 100% of seeds |
| phase_cycles | 0.04 | 0.01188 | 67% of seeds |
| rotation_deg | 2.366 | 2.597 | 50% of seeds |

Median 76.22 s per fit. Section 7.6 gates met: yes.

> Translation, held-out planes and held-out phases improve on every seed, but rotation does not reliably improve. Rotation is observable only through the TEE sector, and at this grid density the angular evidence is thin. A finer sector or an explicitly angular term is the next thing to try; the section 7.6 rotation gate is currently met by margin rather than by improvement.

> A central-difference Jacobian needs two renders per parameter per iteration, so only pose, circular phase, and the leading shape modes are refined here. A full image-domain joint solve needs an analytic or adjoint gradient, which is not implemented.

## Theorems 5 and 6 on the solver Jacobian

- predicted gauge dimension: [5]
- predicted gauge count matches the measured numerical null space: **yes**
- unrestricted alpha is zero (the gauges are the whole null space): yes
- identifiable after gauge removal: **yes**, alpha median 2.31

## E10 — held-out generator families (anti-inverse-crime)

| family | held-out CMR | held-out TEE | rot (deg) | trans (mm) | phase (cyc) | s7.6 |
|---|---:|---:|---:|---:|---:|---|
| `harmonic_anatomy_motion_v1` | 0.5661 | 0.7226 | 0.1384 | 0.07603 | 0.01451 | pass |
| `spline_loading_motion_v1` | 1.071 | 1.15 | 0.3351 | 0.1372 | 0.1406 | **FAIL** |
| `heldout_pathology_mechanics_v1` | 1.209 | 1.49 | 1.694 | 0.2615 | 0.3138 | **FAIL** |

- `spline_loading_motion_v1`: heldout_cmr_mm x1.89, heldout_tee_mm x1.59, phase_error_cycles x9.69
- `heldout_pathology_mechanics_v1`: heldout_cmr_mm x2.14, heldout_tee_mm x2.06, phase_error_cycles x21.63

Attachment exactly zero under every family: **yes**.

> Exact attachment is a structural property and holds under every family. Accuracy is not structural: held-out error and circular phase recovery degrade substantially once the generator stops matching the solver's assumptions, and the section 7.6 phase gate is missed on the held-out families. This is the honest limit of every other number in this campaign.

## Remaining verdicts

### `t2_soft_attachment`

a finite penalty leaves O(1/lambda) drift; exact assembly does not

- slope: median -0.9943, range [-0.9971, -0.9781] over n=60
- slope_within_0p05_of_minus_one: yes
- max_identity_error: 2.21e-08
- smallest_soft_residual_any_run: 3.25e-08
- exact_assembly_residual: 0
- soft_never_reaches_exact: yes

### `t3_gauge_ablation`

distinct parameter triples generate identical trajectories; only the two centering gauges select the true one

- ambiguity_is_real: `{"parameter_space_gap": {"n": 30, "median": 1.4686379400914653, "mean": 1.5926442672802537, "p90": 2.397308023517129, "max": 2.7805877683919844, "min": 0.574293`
- full_gauges_identity_error: median 4.44e-16, range [2.22e-16, 8.88e-16] over n=30
- no_motion_gauge_identity_error: median 1.041, range [0.8622, 1.345] over n=30
- no_session_gauge_identity_error: median 0.7345, range [0.08189, 1.729] over n=30
- full_gauge_reparameterization_drift: median 5.55e-16, range [4.44e-16, 8.88e-16] over n=30
- full_gauges_are_unique: yes
- full_gauges_recover_truth: yes
- motion_gauge_is_necessary: yes
- session_gauge_is_necessary: yes

### `p4_periodicity_phase`

Fourier motion closes the cycle; the warp is a degree-one circle map

- max_seam_error_mm: 2.90e-14
- max_cycle_mean_error_mm: 4.97e-14
- all_warps_strictly_increasing: yes
- smallest_increment_observed: 4.43e-06
- max_total_turn_error: 4.44e-16

### `t5_identifiability`

identifiability requires ker(J_C) and ker(J_E) to intersect trivially

- alpha_both_modalities: median 6.054, range [2.615, 11.21] over n=45
- alpha_cmr_only: median 0, range [0, 0] over n=15
- alpha_tee_only: median 2.663, range [0.8994, 6.659] over n=15
- cmr_only_is_unidentifiable: yes
- joint_beats_worst_single_modality: yes
- weakest_block_when_tee_removed: ['motion_amplitude', 'pose_rotation']

### `t6_noise_stability`

local error stays below 4||eps||/sqrt(alpha(1-c_nl))

- coverage: 1
- all_within_bound: yes
- slack_ratio: median 341.9, range [134.1, 1328] over n=200
- truth_norm: 13.82

### `e1_synthetic_recovery`

known pose and circular phase are recovered under the intended noise regime

- oracle: `{"translation_mm": {"n": 72, "median": 0.02107804384289635, "mean": 0.022356559474443448, "p90": 0.03493720142207327, "max": 0.0546827851328019, "min": 0.006331`
- non_oracle: `{"translation_mm": {"n": 72, "median": 0.011783980123685625, "mean": 0.011289677954674206, "p90": 0.014402873780279157, "max": 0.016477387507530025, "min": 0.00`
- non_oracle_phase_offset_error_cycles: median 0.02, range [0.02, 0.03] over n=72
- cmr_frame_quantization_cycles: 0.05

### `e6_image_domain`

the CMR plane and TEE sector likelihood operators drive the joint fit and improve error on hidden planes and hidden phases

- heldout_cmr_planes: `{"before": {"n": 6, "median": 0.010067205410450697, "mean": 0.009330361538256208, "p90": 0.011722365394234657, "max": 0.012423785403370857, "min": 0.00474575720`
- heldout_tee_phases: `{"before": {"n": 6, "median": 0.04951142147183418, "mean": 0.05558381415903568, "p90": 0.0814459603279829, "max": 0.10078510642051697, "min": 0.0341619290411472`
- translation_mm: `{"before": {"n": 6, "median": 2.7418978310416646, "mean": 2.9607623372573415, "p90": 4.249612481293614, "max": 5.3433371254104465, "min": 1.3156686230195118}, "`
- phase_cycles: `{"before": {"n": 6, "median": 0.040000000000000036, "mean": 0.040000000000000036, "p90": 0.040000000000000036, "max": 0.040000000000000036, "min": 0.04000000000`
- rotation_deg: `{"before": {"n": 6, "median": 2.365649803778642, "mean": 2.2311792159158523, "p90": 3.0673758584455806, "max": 3.7100867428204722, "min": 0.46074208416818263}, `
- seconds_per_fit: median 76.22, range [74.86, 77.98] over n=6
- meets_section_7_6_gates: yes
- rotation_is_the_weak_variable: Translation, held-out planes and held-out phases improve on every seed, but rotation does not reliably improve. Rotation is observable only through the TEE sector, and at this grid density the angular evidence is thin. A finer sector or an explicitly angular term is the next thing to try; the section 7.6 rotation gate is currently met by margin rather than by improvement.
- cost_is_the_limiting_factor: A central-difference Jacobian needs two renders per parameter per iteration, so only pose, circular phase, and the leading shape modes are refined here. A full image-domain joint solve needs an analytic or adjoint gradient, which is not implemented.

### `e9_posterior_calibration`

a gauge-restricted Laplace posterior propagated to scalar measurements yields split-conformal intervals with near-nominal coverage

- calibrated: `{"phase_offset_cycles@0.90": {"name": "phase_offset_cycles", "coverage_target": 0.9, "quantile": 0.05113743281525418, "n_calibration": 22, "circular": true, "at`
- gauge_variance_is_zero: yes
- n_calibration: 22
- minimum_calibration_size: `{"0.90": 9, "0.95": 19}`
- all_targets_attainable: yes
- n_evaluation: 22
- fraction_within_five_points_of_nominal: 0.3333
- fraction_within_one_standard_error: 0.3333
- fraction_meeting_the_conformal_guarantee: 0.8333
- any_interval_under_covers: yes
- one_sided_criterion_note: Split conformal guarantees coverage at least the target in finite samples, so over-coverage is correct conservative behaviour and only under-coverage is a failure. The two-sided five-point criterion in the protocol marks valid conservative intervals as miscalibrated, so the one-sided fraction is reported alongside it.
- coverage_resolution_note: The protocol's five-point criterion is below what this many cases can resolve: the binomial standard error on a 0.90 target at about twenty evaluation cases is already six points. The fraction within one standard error is the honest statistic at this sample size, and a real calibration claim needs the hundreds of cases the protocol assumes.
- caveat: Coverage is measured on synthetic cases drawn from the same generator as the calibration half, so this demonstrates the machinery, not calibration under distribution shift. The protocol requires the locked test family and an external centre before any coverage claim is made.

### `e10_family_generalization`

in-distribution accuracy does not by itself establish robustness to a different generator

- per_family: `{"harmonic_anatomy_motion_v1": {"heldout_cmr_mm": {"n": 12, "median": 0.5660752020270997, "mean": 0.5626935352949879, "p90": 0.5829991692503904, "max": 0.585769`
- degradation_vs_train: `{"spline_loading_motion_v1": {"heldout_cmr_mm": 1.8913913521352341, "heldout_tee_mm": 1.5918840154671523, "phase_error_cycles": 9.685218342591616}, "heldout_pat`
- families_failing_section_7_6: ['spline_loading_motion_v1', 'heldout_pathology_mechanics_v1']
- attachment_survives_every_family: yes
- headline: Exact attachment is a structural property and holds under every family. Accuracy is not structural: held-out error and circular phase recovery degrade substantially once the generator stops matching the solver's assumptions, and the section 7.6 phase gate is missed on the held-out families. This is the honest limit of every other number in this campaign.

## Scope

Every number here comes from synthetic geometry with known truth. None of it is evidence about real CMR or TEE data, clinical accuracy, or the image-domain likelihood operators, which are not yet wired into the joint solver.
