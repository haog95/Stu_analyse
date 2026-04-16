"""启动 Streamlit 仪表盘"""
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

subprocess.run([
    sys.executable, "-m", "streamlit", "run",
    str(project_root / "src" / "visualization" / "dashboard.py"),
    "--server.port", "8501",
])
