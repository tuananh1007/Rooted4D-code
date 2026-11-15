"""Gauge-fixed identity, shared-motion, and session-discrepancy priors."""

from .decomposition import GaugedDecomposition, SessionDecomposition
from .gauges import BasisValidation, GaugedLatentBases

__all__ = [
    "BasisValidation",
    "GaugedDecomposition",
    "GaugedLatentBases",
    "SessionDecomposition",
]
