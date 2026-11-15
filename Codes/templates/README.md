# Production-template assets

This directory intentionally contains specifications and manifest examples but no claimed
production mesh. Medical images, third-party meshes, and patient-derived geometry are not
bundled. `rooted4d.geometry.load_frozen_template` is the executable production loader and
accepts only a frozen manifest whose NPZ asset matches its SHA-256.

The final frozen release must provide:

```text
templates/frozen_v1/
  manifest.yaml
  template_arrays.npz
  quality_report.json
```

`template_arrays.npz` stores vertices, tetrahedra, annular indices, and each component's
local-to-global map, local faces, and annular attachment indices. `manifest.yaml` names
those keys and records units, coordinate system, template ID, landmarks, and asset hash.

The assembly operator is generated only from `local_to_physical.npz`. Patient-specific correspondences and barycentric projections belong in experiment outputs and must never be written back into the template manifest.

Do not change connectivity after the manifest status becomes `frozen`. A new connectivity requires a new template version and complete regeneration of priors, patient fits, and evaluation results.

## Real-derived semi-synthetic donors

`donor_manifest.example.json` is the input contract for locally prepared donors. Every
donor NPZ must contain a `vertices` array fitted to the same frozen topology and must
be listed with a research-pseudonymous ID, anatomy kind, source label, and SHA-256.
Do not use filenames, accession numbers, or clinical identifiers as donor IDs. The
manifest and any patient-derived meshes remain governed data and are not included in
the released package.
