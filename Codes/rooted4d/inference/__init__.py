"""Non-oracle cross-modal initialization, joint inference, and diagnostics."""

from .cross_modal import (
    CrossModalAlignmentResult,
    annular_frame_features,
    infer_cross_modal_alignment,
)
from .joint import (
    Observations,
    SceneFit,
    SolverConfig,
    fit_joint_scene,
    initial_parameters,
    latent_trajectory,
    predict,
)
from .phase_pose import RecoveryResult, kabsch, recover_synthetic_phase_pose
from .posterior import (
    ConformalInterval,
    LaplacePosterior,
    calibrate_split_conformal,
    empirical_coverage,
    fit_laplace_posterior,
    propagate_measurements,
)
from .scene import SceneModel, SceneParameters, smooth_shape_basis

__all__ = [
    "ConformalInterval",
    "CrossModalAlignmentResult",
    "LaplacePosterior",
    "Observations",
    "RecoveryResult",
    "SceneFit",
    "SceneModel",
    "SceneParameters",
    "SolverConfig",
    "annular_frame_features",
    "calibrate_split_conformal",
    "empirical_coverage",
    "fit_laplace_posterior",
    "fit_joint_scene",
    "infer_cross_modal_alignment",
    "initial_parameters",
    "kabsch",
    "latent_trajectory",
    "predict",
    "propagate_measurements",
    "recover_synthetic_phase_pose",
    "smooth_shape_basis",
]
