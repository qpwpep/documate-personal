from pathlib import Path

_CURRENT_FILE_PATH = Path(__file__).resolve()

def project_root_path():
    root = _CURRENT_FILE_PATH.parent.parent.parent
    return root