import os
import sys
from pathlib import Path


def _has_port_arg(argv):
    for i, arg in enumerate(argv):
        if arg == "--port":
            return True
        if arg.startswith("--port="):
            return True
        
        if arg in ("-p",):
            return True
        if i > 0 and argv[i - 1] in ("-p",):
            return True
    return False


def main():
    argv = sys.argv[1:]
    if not _has_port_arg(argv):
        argv = ["--port", "26000"] + argv

    target = Path(__file__).with_name("AirVLNSimulatorServerTool.py")
    if not target.exists():
        raise FileNotFoundError(f"Cannot find target server tool: {target}")

    os.execv(sys.executable, [sys.executable, str(target)] + argv)


if __name__ == "__main__":
    main()

