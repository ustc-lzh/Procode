"""Small path helpers for the released Fig. 3 analysis modules.

Large H5AD and NPY inputs are deliberately external.  The command-line
entry point passes their root explicitly; this module only defines paths
inside the compact Fig. 3 code tree.
"""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_WORKBOOK = PACKAGE_ROOT / "fig3_data.xlsx"
DEFAULT_DATA_ROOT = Path(os.environ.get("FIG3_DATA_ROOT", PACKAGE_ROOT / "largefiles")).expanduser()
DEFAULT_OUTPUT = PACKAGE_ROOT / "results"


def make_output_dir(path: Path) -> Path:
    """Create one requested output directory and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path

