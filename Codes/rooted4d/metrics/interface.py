"""Attachment diagnostics."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from rooted4d.constraints import AttachmentMap


def attachment_residual_over_time(
    attachment_map: AttachmentMap,
    parts: Mapping[str, np.ndarray],
) -> float:
    """Maximum duplicated-interface disagreement across all supplied phases."""

    return attachment_map.max_residual(parts)

