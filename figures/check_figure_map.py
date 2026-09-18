#!/usr/bin/env python3
"""Check that figures/ notebooks and outputs/ agree.

Naming rule for the public repo: each ``figN_<name>.ipynb`` writes exactly one figure,
``figN_<name>.png`` (same stem), so a reader goes from a paper figure to its notebook with no
lookup table. This script enforces it. Run after touching any figure:

    python figures/check_figure_map.py

It fails on:
  * a notebook that writes no .png, or more than one (one figure per notebook)
  * a notebook whose .png does not match its own stem
  * a .png in outputs/ that no notebook writes (an orphan from a rename)
"""
import json
import re
import sys
from pathlib import Path

FIGURES = Path(__file__).resolve().parent
OUTPUTS = FIGURES.parent / "outputs"
PNG_IN_CODE = re.compile(r"""['"]([A-Za-z0-9_.\-]+\.png)['"]""")


def pngs_written(nb_path: Path) -> set[str]:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        for c in nb["cells"] if c["cell_type"] == "code"
    )
    return set(PNG_IN_CODE.findall(code))


def main() -> int:
    errors: list[str] = []
    writes: dict[str, str] = {}

    for nb in sorted(FIGURES.glob("fig*.ipynb")):
        pngs = pngs_written(nb)
        if not pngs:
            errors.append(f"{nb.name}: writes no .png at all")
            continue
        if len(pngs) > 1:
            errors.append(f"{nb.name}: writes several figures {sorted(pngs)}; one figure per notebook")
            continue
        png = pngs.pop()
        expected = nb.stem + ".png"
        if png != expected:
            errors.append(f"{nb.name}: writes {png}, so it should write {expected}")
        writes[png] = nb.stem

    if OUTPUTS.exists():
        for png in sorted(OUTPUTS.glob("*.png")):
            if png.name not in writes:
                errors.append(f"outputs/{png.name}: no notebook writes this file (orphan?)")

    for e in errors:
        print("ERROR:", e)
    print(f"\n{len(writes)} figures, {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
