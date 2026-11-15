r"""ACDC short-axis slices for segmentation training, split-lock respecting.

ACDC labels are 0 background, 1 right ventricle, 2 myocardium, 3 left-ventricle
cavity. The downstream fitter reads label 3, so that class carries the weight,
but all four are predicted because dropping the others makes the boundary between
myocardium and cavity harder rather than easier to place.

In-plane spacing varies across ACDC, roughly 1.37 to 1.68 mm, so slices are
resampled to a common 1.5 mm and centre-cropped. A network trained on raw pixel
grids would be learning each site's spacing along with the anatomy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

__all__ = ["ACDCSlices", "load_patient_splits", "TARGET_SPACING_MM", "PATCH"]

TARGET_SPACING_MM = 1.5
PATCH = 128
CAVITY_LABEL = 3


def load_patient_splits(path: str | Path) -> dict[str, str]:
    """Patient to split map, derived from the frozen global split lock."""

    return json.loads(Path(path).read_text())


def _resample_and_crop(plane: np.ndarray, spacing: tuple[float, float], order: str):
    """Bring one slice to the common spacing and patch size."""

    tensor = torch.as_tensor(np.ascontiguousarray(plane), dtype=torch.float32)[None, None]
    scale = (spacing[0] / TARGET_SPACING_MM, spacing[1] / TARGET_SPACING_MM)
    size = (max(1, int(round(plane.shape[0] * scale[0]))),
            max(1, int(round(plane.shape[1] * scale[1]))))
    mode = "bilinear" if order == "linear" else "nearest"
    kwargs = {"align_corners": False} if mode == "bilinear" else {}
    resampled = functional.interpolate(tensor, size=size, mode=mode, **kwargs)[0, 0]

    output = torch.zeros((PATCH, PATCH), dtype=resampled.dtype)
    rows = min(PATCH, resampled.shape[0])
    columns = min(PATCH, resampled.shape[1])
    src_r = (resampled.shape[0] - rows) // 2
    src_c = (resampled.shape[1] - columns) // 2
    dst_r = (PATCH - rows) // 2
    dst_c = (PATCH - columns) // 2
    output[dst_r : dst_r + rows, dst_c : dst_c + columns] = resampled[
        src_r : src_r + rows, src_c : src_c + columns
    ]
    return output


class ACDCSlices(torch.utils.data.Dataset):
    """Short-axis slices for one split, with labels."""

    def __init__(self, root: str | Path, splits: dict[str, str], split: str,
                 max_patients: int | None = None):
        import nibabel as nib

        self.images: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []
        self.patients: list[str] = []
        seen: set[str] = set()
        for path in sorted(Path(root).rglob("*_gt.nii.gz")):
            patient = path.parent.name
            if splits.get(patient) != split:
                continue
            if max_patients is not None and patient not in seen and len(seen) >= max_patients:
                continue
            seen.add(patient)
            image_path = Path(str(path).replace("_gt.nii.gz", ".nii.gz"))
            if not image_path.exists():
                continue
            label_volume = np.asarray(nib.load(str(path)).dataobj)
            image_object = nib.load(str(image_path))
            image_volume = np.asarray(image_object.dataobj, dtype=np.float32)
            spacing = tuple(float(v) for v in image_object.header.get_zooms()[:2])
            for index in range(label_volume.shape[2]):
                labels = label_volume[:, :, index]
                if labels.max() == 0:
                    continue
                plane = image_volume[:, :, index]
                # Per-slice intensity normalization. MR intensity has no fixed
                # scale and varies between scanners, so a network trained on raw
                # values learns the site rather than the anatomy.
                finite = plane[np.isfinite(plane)]
                low, high = np.percentile(finite, [1.0, 99.0]) if finite.size else (0.0, 1.0)
                plane = np.clip((plane - low) / max(high - low, 1e-6), 0.0, 1.0)
                self.images.append(_resample_and_crop(plane, spacing, "linear"))
                self.labels.append(
                    _resample_and_crop(labels.astype(np.float32), spacing, "nearest").long()
                )
                self.patients.append(patient)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        return self.images[index][None], self.labels[index]

    @property
    def n_patients(self) -> int:
        return len(set(self.patients))
