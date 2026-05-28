import subprocess
import sys
from pathlib import Path


APP_FILE = Path(__file__).resolve().parent / "app" / "streamlit_app.py"


def main():
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_FILE),
    ]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
