r"""A differentiable-by-finite-difference forward model for Theorems 5 and 6.

Theorems 5 and 6 are statements about the evidence Jacobian ``J_E`` of the real
inverse problem.  The joint solver that will eventually supply that Jacobian is
still an open readiness gate, so this module builds an explicit, small forward
map over the nuisance variables the specification names -- TEE pose, circular
phase offset, shared-motion amplitude, and session-discrepancy scale -- and
finite-differences it.

That is enough to run the two required studies honestly:

* the missing-view/modality factorial of Theorem 5, because dropping CMR planes
  or TEE phases changes which rows of ``J_E`` exist;
* the noise sweep of Theorem 6, because the map is smooth and its local
  linearization is exact in the limit of small perturbations.

It is *not* the production observation operator.  Results carry the
``analytic_jacobian`` provenance flag so no downstream summary can present them
as evidence about the image-domain likelihoods.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rooted4d.geometry import RigidTransform, make_toy_complex
from rooted4d.motion import FourierMotion, MonotoneCircularPhaseWarp

#: Contiguous latent blocks, in the order used by :func:`pack_parameters`.
PARAMETER_BLOCKS: dict[str, slice] = {
    "pose_rotation": slice(0, 3),
    "pose_translation": slice(3, 6),
    "phase_offset": slice(6, 7),
    "motion_amplitude": slice(7, 10),
    "session_scale": slice(10, 12),
}
N_PARAMETERS = 12


def pack_parameters(
    pose_axis_angle: np.ndarray,
    pose_translation: np.ndarray,
    phase_offset: float,
    motion_amplitude: np.ndarray,
    session_scale: np.ndarray,
) -> np.ndarray:
    """Assemble the latent vector whose identifiability is being measured."""

    return np.concatenate(
        [
            np.asarray(pose_axis_angle, dtype=np.float64).reshape(3),
            np.asarray(pose_translation, dtype=np.float64).reshape(3),
            np.asarray([phase_offset], dtype=np.float64),
            np.asarray(motion_amplitude, dtype=np.float64).reshape(3),
            np.asarray(session_scale, dtype=np.float64).reshape(2),
        ]
    )


@dataclass
class CoupledForwardModel:
    """Predict CMR and TEE point evidence from the packed latent vector.

    ``cmr_frames`` and ``tee_frames`` select which acquisitions exist.  Passing a
    subset is exactly the missing-view intervention the Theorem 5 contract asks
    for; passing an empty CMR set is the TEE-only arm and vice versa.
    """

    n_annulus: int = 24
    fourier_order: int = 3
    n_cmr_frames: int = 20
    n_tee_frames: int = 24
    seed: int = 0
    cmr_frames: np.ndarray | None = None
    tee_frames: np.ndarray | None = None
    cmr_vertex_fraction: float = 1.0
    tee_vertex_fraction: float = 1.0
    _built: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.cmr_vertex_fraction <= 1.0:
            raise ValueError("cmr_vertex_fraction must lie in (0, 1]")
        if not 0.0 < self.tee_vertex_fraction <= 1.0:
            raise ValueError("tee_vertex_fraction must lie in (0, 1]")
        rng = np.random.default_rng(self.seed)
        complex_ = make_toy_complex(n_annulus=self.n_annulus)
        base = complex_.global_vertices
        groups = complex_.groups

        coefficients = np.zeros((2 * self.fourier_order, *base.shape))
        coefficients[0, :, :2] = -0.075 * base[:, :2]
        coefficients[0, :, 2] = -0.012 * base[:, 2]
        coefficients[1, :, 0] = -0.018 * base[:, 1]
        coefficients[1, :, 1] = 0.018 * base[:, 0]
        if self.fourier_order >= 2:
            coefficients[2, groups["leaflet_free"], 2] = -4.0
        for harmonic in range(3, self.fourier_order + 1):
            cosine = 2 * (harmonic - 1)
            coefficients[cosine] = rng.normal(0.0, 0.35 / harmonic, size=base.shape)

        session_static = np.zeros_like(base)
        session_dynamic = np.zeros_like(base)
        session_static[groups["tee_visible"], :2] = 0.012 * base[groups["tee_visible"], :2]
        session_dynamic[groups["tee_visible"], 2] = 0.55

        cmr_all = np.arange(self.n_cmr_frames)
        tee_all = np.arange(self.n_tee_frames)
        cmr_frames = cmr_all if self.cmr_frames is None else np.asarray(self.cmr_frames, dtype=int)
        tee_frames = tee_all if self.tee_frames is None else np.asarray(self.tee_frames, dtype=int)
        if cmr_frames.size and (cmr_frames.min() < 0 or cmr_frames.max() >= self.n_cmr_frames):
            raise ValueError("cmr_frames indexes a frame that does not exist")
        if tee_frames.size and (tee_frames.min() < 0 or tee_frames.max() >= self.n_tee_frames):
            raise ValueError("tee_frames indexes a frame that does not exist")
        if cmr_frames.size == 0 and tee_frames.size == 0:
            raise ValueError("at least one modality must retain a frame")

        def subsample(indices: np.ndarray, fraction: float, salt: int) -> np.ndarray:
            keep = max(3, int(round(fraction * len(indices))))
            picker = np.random.default_rng(self.seed + salt)
            return np.sort(picker.choice(indices, size=min(keep, len(indices)), replace=False))

        self._built = {
            "base": base,
            "coefficients": coefficients,
            "session_static": session_static,
            "session_dynamic": session_dynamic,
            "cmr_times": np.linspace(0.0, 1.0, self.n_cmr_frames, endpoint=False)[cmr_frames],
            "tee_times": np.linspace(0.0, 1.0, self.n_tee_frames, endpoint=False)[tee_frames],
            "cmr_vertices": subsample(groups["cmr_visible"], self.cmr_vertex_fraction, 101),
            "tee_vertices": subsample(groups["tee_visible"], self.tee_vertex_fraction, 202),
            "phase_logits": np.array([-0.4, 0.2, 0.8, -0.1, 0.5, -0.6, 0.1, 0.4]),
        }

    @property
    def reference_parameters(self) -> np.ndarray:
        """The truth used as the linearization point ``theta*``."""

        return pack_parameters(
            pose_axis_angle=np.array([0.08, -0.12, 0.18]),
            pose_translation=np.array([7.0, -4.0, 11.0]),
            phase_offset=0.13,
            motion_amplitude=np.ones(3),
            session_scale=np.ones(2),
        )

    def predict(self, parameters: np.ndarray) -> np.ndarray:
        """Return the stacked CMR and TEE point evidence for one latent vector."""

        theta = np.asarray(parameters, dtype=np.float64)
        if theta.shape != (N_PARAMETERS,):
            raise ValueError(f"parameters must have shape ({N_PARAMETERS},)")
        built = self._built
        amplitude = theta[PARAMETER_BLOCKS["motion_amplitude"]]
        coefficients = built["coefficients"].copy()
        for index, scale in enumerate(amplitude):
            if 2 * index < len(coefficients):
                coefficients[2 * index] = coefficients[2 * index] * scale
        motion = FourierMotion(base=built["base"], coefficients=coefficients)

        static_scale, dynamic_scale = theta[PARAMETER_BLOCKS["session_scale"]]
        warp = MonotoneCircularPhaseWarp(
            built["phase_logits"], offset=float(theta[PARAMETER_BLOCKS["phase_offset"]][0])
        )
        pose = RigidTransform.from_axis_angle(
            theta[PARAMETER_BLOCKS["pose_rotation"]],
            theta[PARAMETER_BLOCKS["pose_translation"]],
        )

        def residual(phases: np.ndarray) -> np.ndarray:
            modulation = np.cos(2.0 * np.pi * phases)[..., None, None]
            return (
                static_scale * built["session_static"]
                + modulation * dynamic_scale * built["session_dynamic"]
            )

        pieces = []
        if built["cmr_times"].size:
            cmr_geometry = motion.evaluate(built["cmr_times"]) - 0.5 * residual(built["cmr_times"])
            pieces.append(cmr_geometry[:, built["cmr_vertices"], :].reshape(-1))
        if built["tee_times"].size:
            tee_phases = warp.evaluate(built["tee_times"])
            tee_geometry = motion.evaluate(tee_phases) + 0.5 * residual(tee_phases)
            posed = pose.apply(tee_geometry[:, built["tee_vertices"], :])
            pieces.append(posed.reshape(-1))
        return np.concatenate(pieces)

    def jacobian(self, parameters: np.ndarray | None = None, step: float = 1e-6) -> np.ndarray:
        """Central-difference evidence Jacobian at ``parameters``.

        A central difference is used rather than a forward one because the
        second-order truncation error is what keeps the measured smallest
        eigenvalue from being dominated by differencing noise.
        """

        theta = self.reference_parameters if parameters is None else np.asarray(parameters, float)
        if step <= 0.0:
            raise ValueError("step must be positive")
        columns = []
        for index in range(N_PARAMETERS):
            offset = np.zeros(N_PARAMETERS)
            offset[index] = step
            forward = self.predict(theta + offset)
            backward = self.predict(theta - offset)
            columns.append((forward - backward) / (2.0 * step))
        return np.stack(columns, axis=1)
