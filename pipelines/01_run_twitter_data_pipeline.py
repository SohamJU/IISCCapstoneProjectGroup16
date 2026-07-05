import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    script_path = PROJECT_ROOT / "src" / "data" / "twitter_data_pipeline" / "run_twitter_preprocessing.py"
    subprocess.run(
        [sys.executable, str(script_path), "--dataset", "twitter", "--twitter-path", str(PROJECT_ROOT / "archive" / "twcs" / "twcs.csv")],
        check=True,
    )
