from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.spatial import build_and_save_spatial_cache

if __name__ == "__main__":
    build_and_save_spatial_cache()
    print("Spatial cache artifact saved.")
