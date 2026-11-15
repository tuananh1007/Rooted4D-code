#!/bin/bash
# E120: the build of E106's no-shadow twin as it ran on Delta (job 23929426, under
# biii-delta-cpu, code at 80bd163), submitted as a batch script. It builds from the
# committed twin config, whose relative donor path resolves from Codes/configs/e106;
# the e106-twin stage of submit_e120.sbatch built from a copy in the results root,
# where that path does not resolve, and failed before generating anything.
set -euo pipefail
cd /work/nvme/bhio/avu2/Rooted4D/code
[ -L data ] || ln -sfT /work/hdd/bhio/avu2/Rooted4D/data data
source .venv/bin/activate
export PYTHONPATH="$PWD/Codes" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
grep -E '^(commit|dirty)=' PROVENANCE.txt
E120=/work/nvme/bhio/avu2/Rooted4D/results/e120; mkdir -p $E120/e106
cp -f PROVENANCE.txt $E120/PROVENANCE_e106-twin_${SLURM_JOB_ID}.txt
CONFIG=Codes/configs/e106/v3n96o_noshadow.yaml   # the committed twin config E106 built from; its donor path resolves from here
python Codes/scripts/experiments/e106_image_diagnosis.py config $E120/e106/twin.yaml
cmp $E120/e106/twin.yaml $CONFIG && echo "regenerated config equals $CONFIG"
TWIN=data/derived/e120-v3n96o-noshadow
[ -f $TWIN/audit.json ] || python -m rooted4d.synthetic.dataset_cli generate --config $CONFIG --output $TWIN
python -m rooted4d.synthetic.dataset_cli audit --root $TWIN | tail -6
CHECK=results/benchmark/E106/E106_twin_check.json
[ -f $CHECK ] && cp -f $CHECK $E120/e106/E106_twin_check_stored.json
status=0; python Codes/scripts/experiments/e106_image_diagnosis.py check data/derived/e100-v3n96o $TWIN || status=$?
cp -f $CHECK $E120/e106/E106_twin_check.json
[ -f $E120/e106/E106_twin_check_stored.json ] && cp -f $E120/e106/E106_twin_check_stored.json $CHECK
echo "twin check exit $status"; exit $status
