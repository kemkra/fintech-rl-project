import subprocess
import sys
from pathlib import Path


APP_FILE = Path(__file__).resolve().parent / "app" / "streamlit_app.py"


def running_inside_streamlit():
    try:
        from streamlit import runtime
    except Exception:
        return False
    return bool(runtime.exists())


def main():
    if running_inside_streamlit():
        import streamlit as st

        st.error("This helper file should not be used as the Streamlit Cloud entry point.")
        st.write("Set the Streamlit Community Cloud main file path to:")
        st.code("app/streamlit_app.py")
        st.write("Use this helper only for local command-line startup:")
        st.code("python run_app.py")
        return

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
