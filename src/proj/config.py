from pathlib import Path

# Root of the repository (two levels up from this file)
ROOT = Path(__file__).resolve().parents[2]

SRC = ROOT / "src"
DATA = ROOT / "data"
RAW = DATA / "raw"
MODELS = ROOT / "models"