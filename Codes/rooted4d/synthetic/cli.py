"""Generate and validate a dependency-light synthetic smoke case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from rooted4d.inference import recover_synthetic_phase_pose

from .benchmark import generate_case, save_case


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rooted4d-smoke")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--config", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    parameters: dict[str, object] = {"seed": args.seed}
    acceptance_attachment = 1e-10
    acceptance_seam = 1e-10
    if args.config:
        raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        parameters.update(
            {
                "seed": raw.get("seed", args.seed),
                "n_cmr_frames": raw.get("n_cmr_frames", 20),
                "n_tee_frames": raw.get("n_tee_frames", 24),
                "fourier_order": raw.get("fourier_order", 3),
                "cmr_noise_mm": raw.get("cmr_noise_mm", 0.35),
                "tee_noise_mm": raw.get("tee_noise_mm", 0.45),
                "tee_dropout_fraction": raw.get("tee_dropout_fraction", 0.20),
                "tee_axis_angle_rad": tuple(raw["tee_pose"]["axis_angle_rad"]),
                "tee_translation_mm": tuple(raw["tee_pose"]["translation_mm"]),
                "phase_logits": tuple(raw["phase_warp"]["logits"]),
                "phase_offset": raw["phase_warp"]["offset_cycles"],
            }
        )
        acceptance_attachment = raw.get("acceptance", {}).get(
            "attachment_residual_mm", acceptance_attachment
        )
        acceptance_seam = raw.get("acceptance", {}).get("periodic_seam_mm", acceptance_seam)

    case = generate_case(**parameters)
    array_path, metadata_path = save_case(case, args.output)
    recovery = recover_synthetic_phase_pose(case)
    summary = {
        "case": str(array_path.resolve()),
        "metadata": str(metadata_path.resolve()),
        "attachment_residual_mm": case.attachment_residual_mm,
        "periodic_seam_mm": case.periodic_seam_mm,
        "phase_minimum_increment": case.phase_minimum_increment,
        "phase_monotone": case.phase_minimum_increment > 0.0,
        "oracle_initializer": recovery.as_dict(),
    }
    print(json.dumps(summary, indent=2))
    passed = (
        case.attachment_residual_mm <= acceptance_attachment
        and case.periodic_seam_mm <= acceptance_seam
        and case.phase_minimum_increment > 0.0
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
