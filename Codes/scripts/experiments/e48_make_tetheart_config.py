"""E48: derive an ACDC-only TetHeart config from their static one.

Their `unified_mix_tet_base.py` concatenates three datasets -- 914 (ACDC), 933
(M&Ms) and 943 (M&Ms-2) -- for train, val and test. We hold ground-truth meshes
only for ACDC, because 4DMR ships ACDC checkpoints and nothing equivalent for
the other two, so the other datasets are removed.

That is a real reduction in training data against their setup and is reported as
one. It is also the only honest option: including 933 or 943 without meshes
would fail their loader, and inventing meshes for them would compound the
ground-truth substitution rather than contain it.

The transformation is textual and structural rather than a config-system
override, so the result is a complete standalone file that can be read and
checked without knowing mmcv's merge semantics. It is verified to parse as
Python and to contain no reference to the removed tasks.
"""

import ast
import os
import re
import sys
from pathlib import Path

SOURCE = Path(os.environ["R4D_TETHEART_CONFIG"])
DEST = Path(os.environ["R4D_CONFIG_OUT"])
DATA_ROOT = os.environ["R4D_CONFIG_DATA_ROOT"]
DROP_TASKS = ("933", "943")


def drop_dataset_blocks(text):
    """Remove each dict(...) entry whose task is one we have no meshes for."""
    lines = text.split("\n")
    keep, index, removed = [], 0, 0
    while index < len(lines):
        line = lines[index]
        if re.match(r"\s*dict\($", line):
            depth, block = 0, []
            start = index
            while index < len(lines):
                block.append(lines[index])
                depth += lines[index].count("(") - lines[index].count(")")
                index += 1
                if depth <= 0:
                    break
            body = "\n".join(block)
            if any(("task='%s'" % task) in body or ('task="%s"' % task) in body
                   for task in DROP_TASKS):
                removed += 1
                continue
            keep.extend(lines[start:index])
            continue
        keep.append(line)
        index += 1
    return "\n".join(keep), removed


def main() -> int:
    text = SOURCE.read_text()
    text, removed = drop_dataset_blocks(text)

    # Point at the staged root, which contains only the patients this run may
    # train on. The split is enforced by directory contents rather than by their
    # internal KFold, which shuffles with a fixed seed and knows nothing about
    # our lock.
    text, count = re.subn(r"^data_root\s*=.*$",
                          "data_root = %r" % DATA_ROOT, text, flags=re.M)
    if count != 1:
        print("expected exactly one data_root assignment, found %d" % count, file=sys.stderr)
        return 1

    header = (
        "# Derived from TetHeart's configs/Unified/tet/static/unified_mix_tet_base.py\n"
        "# by Codes/scripts/experiments/e48_make_tetheart_config.py.\n"
        "#\n"
        "# Two changes, both recorded in Docs/EXTERNAL_COMPARISON_AUDIT.md:\n"
        "#   * the M&Ms (933) and M&Ms-2 (943) datasets are removed, because we hold\n"
        "#     ground-truth meshes only for ACDC. This reduces the training data\n"
        "#     relative to their setup.\n"
        "#   * data_root points at a staging tree holding only the patients this run\n"
        "#     is permitted to see, which is how our split lock is enforced; their\n"
        "#     own KFold shuffles with a fixed seed and does not know about it.\n"
        "#     Pooling our train and validation patients was a mistake: their fold 0\n"
        "#     then trained on 22 of our 29 validation patients, so the root now\n"
        "#     holds our training split alone and their internal fold splits only\n"
        "#     within it.\n"
        "#\n"
        "# The optimizer still reads FusedAdam. In this environment that name is\n"
        "# provided by Codes/scripts/external/apex_shim and resolves to\n"
        "# torch.optim.AdamW, because apex will not build against this cluster's\n"
        "# CUDA toolkits. Runs using this config must report AdamW.\n\n"
    )
    text = header + text

    try:
        ast.parse(text)
    except SyntaxError as exc:
        print("generated config does not parse: %s" % exc, file=sys.stderr)
        return 1
    for task in DROP_TASKS:
        if ("task='%s'" % task) in text:
            print("task %s still present after filtering" % task, file=sys.stderr)
            return 1

    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(text)
    print("removed %d dataset blocks for tasks %s" % (removed, ", ".join(DROP_TASKS)))
    print("data_root -> %s" % DATA_ROOT)
    print("wrote %s (%d lines)" % (DEST, text.count("\n")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
