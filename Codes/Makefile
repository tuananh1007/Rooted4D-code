# This Makefile lives in Codes/ but its targets act on the workspace, which is
# the directory above: the split lock, the configurations and the data all
# live there. WORKSPACE is resolved from the Makefile's own location so `make`
# behaves the same from either directory.
WORKSPACE := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))..)

.PHONY: install test smoke semisynthetic semisynthetic-audit data-list baselines-list clean

install:
	python -m pip install -e $(WORKSPACE)/Codes

test:
	cd $(WORKSPACE) && PYTHONPATH=Codes python -m unittest discover -s Codes/tests -v


smoke:
	rooted4d-smoke --output outputs/smoke_case --seed 7

semisynthetic:
	cd $(WORKSPACE) && rooted4d-semisynthetic generate --config Codes/configs/semisynthetic_proof_of_concept.yaml --output outputs/semisynthetic-poc-v1

semisynthetic-audit:
	rooted4d-semisynthetic audit --root outputs/semisynthetic-poc-v1

data-list:
	rooted4d-data list

baselines-list:
	cd $(WORKSPACE) && python Codes/scripts/bootstrap_baselines.py --list

clean:
	python -c "from pathlib import Path; [p.unlink() for p in Path('.').rglob('*.pyc')]"
