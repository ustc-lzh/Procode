"""Default input and output paths for Figure 3."""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_WORKBOOK = PACKAGE_ROOT / "fig3_data.xlsx"
DEFAULT_DATA_ROOT = Path(os.environ.get("FIG3_DATA_ROOT", PACKAGE_ROOT / "largefiles")).expanduser()
DEFAULT_OUTPUT = PACKAGE_ROOT / "results"

