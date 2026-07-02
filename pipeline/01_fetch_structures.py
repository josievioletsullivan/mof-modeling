"""Stage 01: pull CIFs from Materials Project into data/structures/."""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import config
from src.structures import fetch_structures

if __name__ == "__main__":
    saved = fetch_structures(config.TARGETS, config.STRUCTURES)
    print(f"\nsaved {len(saved)} CIFs -> {config.STRUCTURES}")
