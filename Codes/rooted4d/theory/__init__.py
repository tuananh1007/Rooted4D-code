"""Executable falsification checks for the section 4.9 theorem stack.

The specification pairs every theorem with a required implementation check.
This package supplies those checks so a claim can be refuted by running code
rather than only by reading a proof:

===========  ==============================================================
Result       Check
===========  ==============================================================
Theorem 1    :func:`check_assembly_nullspace_equivalence`
Theorem 2    :func:`soft_attachment_sweep`
Theorem 3    :class:`rooted4d.priors.SessionDecomposition` canonicalization
Prop. 4      :class:`rooted4d.motion.FourierMotion` seam error and
             :class:`rooted4d.motion.MonotoneCircularPhaseWarp` increments
Theorem 5    :func:`constrained_identifiability`
Theorem 6    :func:`noise_stability_bound`
===========  ==============================================================
"""

from .assembly import (
    AssemblyEquivalenceReport,
    check_assembly_nullspace_equivalence,
    spanning_tree_equalities,
)
from .identifiability import (
    ConstrainedIdentifiability,
    NoiseStabilityBound,
    constrained_identifiability,
    noise_stability_bound,
)
from .soft_penalty import (
    SoftPenaltyPoint,
    SoftPenaltySweep,
    soft_attachment_stationary_point,
    soft_attachment_sweep,
)

__all__ = [
    "AssemblyEquivalenceReport",
    "ConstrainedIdentifiability",
    "NoiseStabilityBound",
    "SoftPenaltyPoint",
    "SoftPenaltySweep",
    "check_assembly_nullspace_equivalence",
    "constrained_identifiability",
    "noise_stability_bound",
    "soft_attachment_stationary_point",
    "soft_attachment_sweep",
    "spanning_tree_equalities",
]
