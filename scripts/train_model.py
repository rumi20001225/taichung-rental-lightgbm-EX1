from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import train_and_save_model

if __name__ == "__main__":
    bundle = train_and_save_model()
    print("LightGBM model artifact saved.")
    print(bundle["metrics"])
