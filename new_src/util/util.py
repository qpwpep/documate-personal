import os
from pathlib import Path

_CURRENT_FILE_PATH = Path(__file__).resolve()

def get_project_root_path():
    root = _CURRENT_FILE_PATH.parent.parent.parent
    return root

def get_save_text_output_dir():
    root = get_project_root_path()
    path = os.path.join(root, 'output/save_text')
    return path