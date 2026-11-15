r"""Differentiable scene evaluation and exact Jacobians.

Mirrors the parts of ``rooted4d.inference.joint`` that the image-domain residual
depends on, so the Jacobian can be taken by forward-mode autodiff in one batched
pass instead of by a hundred and twenty-six finite-difference evaluations.

The parameter packing follows the NumPy solver exactly, because a port whose
vector layout differs cannot be checked against it, and an unchecked port of a
numerical routine is not worth having.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .geometry import surface_likelihood

__all__ = ["TorchScene", "image_residuals", "jacobian", "point_residuals"]


def _tensor(value) -> torch.Tensor:
    return torch.as_tensor(np.asarray(value, dtype=np.float64), dtype=torch.float64)


@dataclass
class TorchScene:
    """Static geometry and evidence, held as tensors."""

    template: torch.Tensor
    shape_basis: torch.Tensor
    motion_basis: torch.Tensor
    session_basis: torch.Tensor
    n_modes: int
    n_motion: int
    n_session: int
    fourier_order: int
    session_order: int
    session_includes_offset: bool
    n_phase_nodes: int
    sessions: tuple[str, ...] = ("cmr", "tee")
    #: Apply the cross-session gauge (``_apply_gauges``) to the session blocks,
    #: in the geometry and in the session prior alike.
    centre_sessions: bool = True

    @classmethod
    def from_model(cls, model, config) -> "TorchScene":
        return cls(
            template=_tensor(model.template_physical),
            shape_basis=_tensor(model.shape_basis),
            motion_basis=_tensor(model.bases.motion),
            session_basis=_tensor(model.bases.session),
            n_modes=int(model.n_modes),
            n_motion=int(model.bases.n_motion),
            n_session=int(model.bases.n_session),
            fourier_order=int(config.fourier_order),
            session_order=int(config.session_order),
            session_includes_offset=bool(config.session_includes_offset),
            n_phase_nodes=int(model.n_phase_nodes),
            sessions=tuple(model.sessions),
            centre_sessions=bool(config.enforce_gauges and config.use_session),
        )

    @property
    def motion_rows(self) -> int:
        return 2 * self.fourier_order

    @property
    def session_rows(self) -> int:
        return 2 * self.session_order + (1 if self.session_includes_offset else 0)

    def unpack(self, vector: torch.Tensor) -> dict:
        """Split a packed parameter vector, in the NumPy solver's layout."""

        cursor = 0

        def take(count: int) -> torch.Tensor:
            nonlocal cursor
            block = vector[cursor : cursor + count]
            cursor += count
            return block

        identity = take(self.n_modes)
        motion = take(self.motion_rows * self.n_motion).reshape(
            self.motion_rows, self.n_motion
        )
        session = {
            name: take(self.session_rows * self.n_session).reshape(
                self.session_rows, self.n_session
            )
            for name in self.sessions
        }
        axis_angle = take(3)
        translation = take(3)
        phase_offset = take(1)[0]
        logits = take(self.n_phase_nodes)
        return {
            "identity": identity, "motion": motion, "session": session,
            "axis_angle": axis_angle, "translation": translation,
            "phase_offset": phase_offset, "logits": logits,
        }


def _periodic_basis(phases: torch.Tensor, order: int) -> torch.Tensor:
    columns = []
    for harmonic in range(1, order + 1):
        angle = 2.0 * torch.pi * harmonic * phases
        columns.extend((torch.cos(angle), torch.sin(angle)))
    return torch.stack(columns, dim=-1)


def _session_design(phases: torch.Tensor, scene: TorchScene) -> torch.Tensor:
    oscillatory = _periodic_basis(phases, scene.session_order)
    if not scene.session_includes_offset:
        return oscillatory
    constant = torch.ones(oscillatory.shape[:-1] + (1,), dtype=oscillatory.dtype)
    return torch.cat([constant, oscillatory], dim=-1)


def _rotation(axis_angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues rotation from an axis-angle vector.

    No wrapping is applied here, but callers should note the parameterization is
    unbounded and redundant beyond pi: a gradient method with nothing to stop it
    will happily walk to a magnitude of fourteen radians, which is over four
    full turns and numerically far from where it started. ``axis_angle_penalty``
    in the prior block is what keeps the iterate in the principal branch.
    """

    # Written with the *unnormalized* skew, so no division by the angle appears
    # and the expression stays analytic at zero.
    #
    # Normalizing the axis and clamping the norm makes the value correct at the
    # identity and the derivative identically zero there: the clamp has no
    # gradient below its threshold, the normalized axis is zero, and sin(0) is
    # zero, so every term of the product rule vanishes. Automatic
    # differentiation then returned an exactly zero rotation Jacobian where
    # central differences give 1.0, and any optimizer starting from zero
    # rotation left rotation at the identity for the whole run while reporting
    # finite gradients.
    #
    # In this form the coefficients are sin(t)/t and (1 - cos t)/t^2, both
    # analytic at t = 0 with values 1 and 1/2. They are evaluated by series
    # below a threshold, where the closed forms lose precision, and everything
    # is expressed through t^2 = sum(axis_angle^2), which is smooth.
    theta_sq = (axis_angle * axis_angle).sum()
    small = theta_sq < 1e-8
    # The substitution has to happen *before* the square root, not after.
    # torch.where backpropagates through the inputs of both branches, and
    # d(sqrt)/dx is infinite at zero, so taking sqrt(theta_sq) first and
    # substituting afterwards makes the whole Jacobian NaN at the identity --
    # trading a silently zero derivative for a loudly absent one.
    guarded_sq = torch.where(small, torch.ones_like(theta_sq), theta_sq)
    theta = torch.sqrt(guarded_sq)
    sin_over = torch.where(
        small,
        1.0 - theta_sq / 6.0 + theta_sq * theta_sq / 120.0,
        torch.sin(theta) / theta,
    )
    cos_over = torch.where(
        small,
        0.5 - theta_sq / 24.0 + theta_sq * theta_sq / 720.0,
        (1.0 - torch.cos(theta)) / (theta * theta),
    )
    zero = torch.zeros((), dtype=axis_angle.dtype)
    skew = torch.stack([
        torch.stack([zero, -axis_angle[2], axis_angle[1]]),
        torch.stack([axis_angle[2], zero, -axis_angle[0]]),
        torch.stack([-axis_angle[1], axis_angle[0], zero]),
    ])
    eye = torch.eye(3, dtype=axis_angle.dtype)
    return eye + sin_over * skew + cos_over * (skew @ skew)


def _warp(phases: torch.Tensor, logits: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """Monotone circular phase warp, matching MonotoneCircularPhaseWarp.

    The offset is added *after* the piecewise-linear map and before the wrap,
    matching ``evaluate`` which is ``mod(offset + interp(t), 1)``. Adding it
    before the map instead is a different function, and a silently different one:
    both are monotone circular warps with the same parameters, so nothing raises
    and the residual is simply wrong. That mistake made this port disagree with
    the NumPy definition by 1.4 in the residual before it was caught.
    """

    increments = torch.nn.functional.softplus(logits) + 1e-6
    increments = increments / increments.sum()
    targets = torch.cat([torch.zeros(1, dtype=logits.dtype), torch.cumsum(increments, 0)])
    sources = torch.linspace(0.0, 1.0, logits.numel() + 1, dtype=logits.dtype)
    index = torch.clamp(torch.bucketize(phases, sources) - 1, 0, logits.numel() - 1)
    span = sources[index + 1] - sources[index]
    fraction = (phases - sources[index]) / span
    base = targets[index] + fraction * (targets[index + 1] - targets[index])
    return torch.remainder(offset + base, 1.0)


def _session_blocks(scene: TorchScene, unpacked: dict) -> dict:
    """The session blocks after the cross-session gauge, as ``_apply_gauges``."""

    if not scene.centre_sessions:
        return unpacked["session"]
    mean = torch.stack([unpacked["session"][n] for n in scene.sessions]).mean(dim=0)
    return {name: unpacked["session"][name] - mean for name in scene.sessions}


def _coordinates(scene: TorchScene, unpacked: dict, phases: torch.Tensor, session: str):
    # The geometry uses the gauged session blocks, the ones the prior charges.
    # It used the raw blocks, so their cross-session mean moved every frame at
    # no prior cost (E120).
    latent = unpacked["identity"][None, :].expand(phases.shape[0], -1)
    shared = _periodic_basis(phases, scene.fourier_order) @ unpacked["motion"]
    latent = latent + shared @ scene.motion_basis.T
    residual = _session_design(phases, scene) @ _session_blocks(scene, unpacked)[session]
    latent = latent + residual @ scene.session_basis.T
    displacement = latent @ scene.shape_basis.T
    return scene.template[None, :, :] + displacement.reshape(
        phases.shape[0], scene.template.shape[0], 3
    )


def point_residuals(vector, scene: TorchScene, evidence: dict) -> torch.Tensor:
    """Residuals against point evidence."""

    unpacked = scene.unpack(vector)
    pieces = []
    if evidence.get("cmr_points") is not None:
        geometry = _coordinates(scene, unpacked, evidence["cmr_times"], "cmr")
        pieces.append(
            (geometry[:, evidence["cmr_vertices"], :] - evidence["cmr_points"]).reshape(-1)
        )
    if evidence.get("tee_points") is not None:
        warped = _warp(evidence["tee_times"], unpacked["logits"], unpacked["phase_offset"])
        geometry = _coordinates(scene, unpacked, warped, "tee")
        posed = geometry @ _rotation(unpacked["axis_angle"]).T + unpacked["translation"]
        pieces.append(
            (posed[:, evidence["tee_vertices"], :] - evidence["tee_points"]).reshape(-1)
        )
    return torch.cat(pieces) if pieces else torch.zeros(0, dtype=torch.float64)


def image_residuals(vector, scene: TorchScene, evidence: dict) -> torch.Tensor:
    """Residuals against image-likelihood evidence at frozen query points."""

    unpacked = scene.unpack(vector)
    pieces = []
    if evidence.get("cmr_values") is not None:
        geometry = _coordinates(scene, unpacked, evidence["cmr_image_times"], "cmr")
        predicted = torch.stack([
            torch.stack([
                surface_likelihood(
                    evidence["cmr_query"] + shift * evidence["cmr_normal"],
                    frame, evidence["cmr_faces"], sigma_mm=evidence["sigma_mm"],
                )
                for shift in evidence["cmr_through_plane"]
            ]).mean(dim=0)
            for frame in geometry
        ])
        pieces.append(
            evidence["cmr_weight"] ** 0.5 * (predicted - evidence["cmr_values"]).reshape(-1)
        )
    if evidence.get("tee_values") is not None:
        warped = _warp(
            evidence["tee_image_times"], unpacked["logits"], unpacked["phase_offset"]
        )
        geometry = _coordinates(scene, unpacked, warped, "tee")
        posed = geometry @ _rotation(unpacked["axis_angle"]).T + unpacked["translation"]
        predicted = torch.stack([
            surface_likelihood(
                evidence["tee_query"], frame, evidence["tee_faces"],
                sigma_mm=evidence["sigma_mm"],
            )
            for frame in posed
        ])
        difference = predicted - evidence["tee_values"]
        tolerance = float(evidence.get("shadow_tolerance", 1.0))
        if tolerance < 1.0:
            # Acoustic shadowing can only *remove* signal, never add it, so a
            # query point where the model predicts more echo than was observed
            # is evidence of either bad geometry or a shadow, while a point
            # predicting less is evidence of bad geometry alone. Weighting the
            # two the same lets an unmodelled shadow pull the surface away from
            # the anatomy to reproduce the missing signal, which is measurably
            # what happens: the fitted solution reaches a third of the cost of
            # the truth. Down-weighting over-prediction is the robustness a
            # deployment needs and does not require knowing where the shadow is,
            # which reading it from the generator seed would.
            over = difference > 0.0
            difference = torch.where(over, difference * tolerance**0.5, difference)
        pieces.append(evidence["tee_weight"] ** 0.5 * difference.reshape(-1))
    return torch.cat(pieces) if pieces else torch.zeros(0, dtype=torch.float64)


def jacobian(function, vector: torch.Tensor) -> torch.Tensor:
    """Exact Jacobian by batched forward-mode autodiff.

    Forward mode rather than reverse: the cost of reverse mode scales with the
    number of *residuals*, which here is thousands, while forward mode scales
    with the number of parameters, which is tens. ``vmap`` batches the tangents
    so the whole Jacobian is one pass rather than one pass per parameter, which
    is what makes this cheaper than the differencing it replaces rather than
    merely more accurate than it.
    """

    return torch.func.jacfwd(function)(vector)


def prior_residuals(vector, scene: TorchScene, weights: dict) -> torch.Tensor:
    """Prior residuals, matching the NumPy solver's blocks.

    Included so the torch objective is the same function the NumPy solver
    minimizes. Optimizing a different objective faster is not a port, and the
    comparison between the two would be meaningless.
    """

    unpacked = scene.unpack(vector)
    identity = weights["identity"] ** 0.5 * unpacked["identity"]
    if weights.get("identity_multiplier") is not None:
        # The covariance-aware prior: the NumPy solver's per-coefficient factor,
        # computed there by ``identity_prior_multiplier`` and carried here as a
        # constant, so the two objectives stay the same function.
        identity = identity * _tensor(weights["identity_multiplier"])
    pieces = [
        identity,
        weights["motion"] ** 0.5 * unpacked["motion"].reshape(-1),
    ]
    if weights.get("session", 0.0) > 0.0:
        # The cross-session gauge centres the session blocks before they are
        # penalized, exactly as _apply_gauges does; penalizing the raw blocks
        # would charge the prior for a direction the gauge removes.
        blocks = _session_blocks(scene, unpacked)
        centred = torch.stack([blocks[n] for n in scene.sessions])
        pieces.append(weights["session"] ** 0.5 * centred.reshape(-1))
    if weights.get("axis_angle", 0.0) > 0.0:
        # Keep the rotation in the principal branch. Without it a first-order
        # method drifted to |omega| = 14.8 rad, more than four full turns, and
        # reported a 162 degree pose error from a 4.7 degree start.
        magnitude = torch.linalg.norm(unpacked["axis_angle"])
        excess = torch.clamp(magnitude - torch.pi, min=0.0)
        pieces.append((weights["axis_angle"] ** 0.5 * excess).reshape(1))
    if weights.get("phase_smoothness", 0.0) > 0.0:
        increments = torch.nn.functional.softplus(unpacked["logits"]) + 1e-6
        increments = increments / increments.sum()
        pieces.append(
            weights["phase_smoothness"] ** 0.5
            * torch.diff(increments)
            * increments.numel()
        )
    return torch.cat(pieces)


def fit_image_lbfgs(
    start: torch.Tensor,
    scene: TorchScene,
    evidence: dict,
    weights: dict,
    *,
    iterations: int = 300,
    learning_rate: float = 0.4,
):
    """Fit image evidence with a first-order method on exact gradients.

    Gauss-Newton needs a Jacobian, which costs about 27 residual evaluations by
    forward-mode autodiff and 102 by central differences. A gradient costs less
    than one. That ratio, not raw arithmetic speed, is what the port buys: a
    first-order method takes a hundred steps in the time a differenced
    Gauss-Newton step takes one.
    """

    vector = start.clone().detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [vector], lr=learning_rate, max_iter=iterations,
        history_size=25, line_search_fn="strong_wolfe",
        tolerance_grad=1e-10, tolerance_change=1e-14,
    )
    history: list[float] = []

    def closure():
        optimizer.zero_grad()
        residual = torch.cat([
            image_residuals(vector, scene, evidence),
            landmark_residuals(vector, scene, evidence),
            prior_residuals(vector, scene, weights),
        ])
        cost = (residual**2).sum()
        cost.backward()
        history.append(float(cost.detach()))
        return cost

    optimizer.step(closure)
    return vector.detach(), history


def evidence_tensors(evidence, cmr_weight: float, tee_weight: float) -> dict:
    """Convert an :class:`ImageEvidence` into the tensor dictionary used here."""

    def real(value) -> torch.Tensor:
        return torch.as_tensor(np.asarray(value, dtype=np.float64), dtype=torch.float64)

    def index(value) -> torch.Tensor:
        return torch.as_tensor(np.asarray(value, dtype=np.int64), dtype=torch.int64)

    return {
        "cmr_query": real(evidence.cmr_points),
        "cmr_values": real(evidence.cmr_values),
        "cmr_faces": index(evidence.cmr_faces),
        "cmr_image_times": real(evidence.cmr_times),
        "cmr_normal": real(evidence.cmr_normal),
        "cmr_through_plane": real(evidence.cmr_through_plane),
        "cmr_weight": float(cmr_weight),
        "tee_query": real(evidence.tee_points),
        "tee_values": real(evidence.tee_values),
        "tee_faces": index(evidence.tee_faces),
        "tee_image_times": real(evidence.tee_times),
        "tee_weight": float(tee_weight),
        "sigma_mm": float(evidence.sigma_mm),
    }


def prior_weights(config, model=None) -> dict:
    """Prior weights from a SolverConfig, in the form the torch objective wants.

    Passing the model carries the covariance-aware identity prior across, when
    the configuration asks for it and the model has training scales; without
    the model the prior is isotropic, which is what every earlier caller used.
    """

    multiplier = None
    if model is not None:
        from rooted4d.inference.joint import identity_prior_multiplier

        factor = identity_prior_multiplier(model, config, int(model.n_modes))
        multiplier = None if factor is None else np.asarray(factor, dtype=np.float64)
    return {
        "identity_multiplier": multiplier,
        "identity": float(config.identity_prior_weight),
        "motion": float(config.motion_prior_weight),
        "session": float(config.session_prior_weight) if config.use_session else 0.0,
        "phase_smoothness": (
            float(config.phase_smoothness_weight) if config.estimate_phase_warp else 0.0
        ),
        "axis_angle": 1e4,
    }


def landmark_residuals(vector, scene: TorchScene, evidence: dict) -> torch.Tensor:
    r"""Residuals for a few anatomical landmarks with *known* correspondence.

    Section 19 measured why image evidence cannot recover pose: a surface
    likelihood is correspondence-free, and rotating a near-axisymmetric annulus
    about its own axis reproduces very nearly the same surface, so the objective
    changes by one percent under a 180-degree rotation where point evidence
    changes by a factor of 189. The unconstrained direction is not a numerical
    problem and no optimizer fixes it; the evidence simply does not contain the
    information.

    A landmark does contain it. A point that is known to be *this* vertex, not
    merely some point on the surface, breaks the rotational symmetry directly.
    This is not a synthetic convenience: the anterolateral and posteromedial
    commissures are standard annotated landmarks on the mitral annulus, the
    production template specification already names them among the frozen
    landmarks, and ``infer_cross_modal_alignment`` already assumes annular curves
    anchored at a commissure.

    Landmarks are observed in the TEE frame, so the prediction is posed.
    """

    if evidence.get("landmark_points") is None:
        return torch.zeros(0, dtype=torch.float64)
    unpacked = scene.unpack(vector)
    warped = _warp(
        evidence["landmark_times"], unpacked["logits"], unpacked["phase_offset"]
    )
    geometry = _coordinates(scene, unpacked, warped, "tee")
    posed = geometry @ _rotation(unpacked["axis_angle"]).T + unpacked["translation"]
    difference = posed[:, evidence["landmark_vertices"], :] - evidence["landmark_points"]
    return evidence["landmark_weight"] ** 0.5 * difference.reshape(-1)
