"""Install the project's declared dependencies into the active interpreter."""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    requirements = Path(__file__).resolve().parents[1] / "requirements.txt"
    raise SystemExit(subprocess.call([sys.executable, "-m", "pip", "install", "-r", str(requirements)]))
