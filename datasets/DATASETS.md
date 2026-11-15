# Dataset acquisition and evidence roles

**Source audit:** 23 August 2026. Recheck every provider page immediately before acquisition or publication.

No public resource identified here contains the complete target: same-patient, dynamic cine CMR and 4D TEE with LV–LA–mitral anatomy across the cardiac cycle. The resources below train or test components; a governed local paired cohort remains necessary for the central real-patient claim.

| Key | Current access | Appropriate Rooted4D use | Important limitation |
|---|---|---|---|
| `acdc` | [Official ACDC page](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html); provider portal | CMR segmentation, ED/ES geometry, sparse-slice simulation | Short-axis CMR; no leaflets or paired TEE |
| `mnms` | [Official M&Ms page](https://www.ub.edu/mnms/) | Multi-vendor CMR robustness | No TEE or leaflet motion |
| `mnms2` | [Official M&Ms-2 page](https://www.ub.edu/mnms-2/) | Multi-view CMR robustness, including four-chamber data | No paired TEE |
| `mitea` | [Institutional-email request](https://www.cardiacatlas.org/mitea/) | Real CMR-informed 3DE ventricular bridge; scan/rescan stability | 134 subjects, ED/ES LV; not 4D TEE leaflet data |
| `mvseg2023` | [Gated Hugging Face release](https://huggingface.co/datasets/pcarnahan/MVSeg2023); accept conditions and authenticate | Static anterior/posterior leaflet initializer | Single frame at end diastole; no full-cycle motion |
| `vecheart_vhmesh` | [VecHeart repository and linked release](https://github.com/Scalsol/VecHeart) | Registered surface anatomy prior | No mitral leaflets or paired evidence; upstream licensing needs review |
| `ukbb_cmr` | [UK Biobank application](https://www.ukbiobank.ac.uk/use-our-data/apply-for-access/) | Large population anatomy/motion prior | New applications were paused as of this audit, with late-2026 reopening intended; no paired TEE |
| `mvaa2026` | [Zenodo challenge proposal](https://zenodo.org/records/19726755) | Release watch only | The audited record contains a proposal PDF, not image cases |
| `local_paired_cmr_4dtee` | Institutional collaborator, IRB/ethics, DUA | Decisive held-out real-patient evaluation | Not public; must remain locked and governed |

## Corrections discovered in the source audit

1. **MVAA 2026 is not presently a downloadable image dataset.** The Zenodo record exposes a challenge-proposal PDF only. Do not count it in an acquired-data table or implementation timeline until the organizers release cases and terms.
2. **MTCNet is not presently a runnable baseline.** Its official repository states that code and data are “coming soon.” Keep it in the release watch rather than substituting an unofficial reimplementation.
3. **MVSeg2023 is gated despite being publicly discoverable.** The user must log in, accept conditions, and share the requested contact information. The catalog therefore requires explicit provider-term acknowledgement before automated retrieval.
4. **The MVSeg2023 card should be audited after download.** Its displayed total and displayed split counts are internally inconsistent; use the actual downloaded manifest for sample accounting.
5. **UK Biobank is not an immediate bootstrap source for a new team.** The official page stated on 1 July 2026 that new applications were paused, with an intention to resume in late 2026.

## Recommended acquisition order

1. Generate the leakage-audited semi-synthetic Rooted4D benchmark immediately and keep the small CI configuration in continuous integration.
2. Obtain ACDC and M&Ms-2 for the CMR branch.
3. Accept and download MVSeg2023 for the static TEE initializer.
4. Submit the MITEA request using an official institutional email address.
5. Fit permitted CMR- and TEE-derived donors to the frozen topology, create a local hashed donor manifest, and regenerate the proof-of-concept benchmark without redistributing provider data.
6. Use separate closed real cohorts for unimodal/component and stress evaluations; confirm a same-patient CMR–4D-TEE cohort only before making an end-to-end clinical fusion claim.
7. Treat UK Biobank, MVAA, MTCNet, and newly released CineMesh4D resources as monitored additions, not blocking prerequisites.

## Downloading MVSeg2023

First accept the conditions on the dataset page. Then install and authenticate the official Hugging Face client:

```bash
python -m pip install -U huggingface_hub
hf auth login
hf download pcarnahan/MVSeg2023 \
  --repo-type dataset \
  --local-dir data/raw/mvseg2023
```

Equivalent Rooted4D command:

```bash
rooted4d-data fetch mvseg2023 \
  --dest data/raw/mvseg2023 \
  --acknowledge-provider-terms
```

Never put authentication tokens into config files, shell history committed to Git, notebooks, or experiment metadata.

## Data integrity and split rules

- Store raw provider files read-only under `data/raw/<dataset>`.
- Record provider version/retrieval date and a local SHA-256 manifest.
- Split at the patient level before generating frames, slices, patches, or pseudo-labels.
- Keep the real paired test cohort inaccessible to training and hyperparameter code.
- Never pool echo scan/rescan or CMR/TEE sessions from one patient across splits.
- Keep provider data outside released archives; publish only derived data permitted by the applicable agreement.
