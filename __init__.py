# RSTEM/__init__.py
import sys
from pathlib import Path

# Path to Rob_coding/
ROOT = Path(__file__).resolve().parents[1]

# Path to ExpertPI-0.5.1 (containing the expertpi package)
EXPERTPI_DIR = ROOT / "ExpertPI-0.5.1"

if str(EXPERTPI_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERTPI_DIR))