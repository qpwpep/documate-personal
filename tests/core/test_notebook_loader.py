import copy
import json
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

from src.notebook_loader import canonicalize_notebook_payload, load_canonical_notebook


def _raw_notebook_without_ids() -> dict:
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Demo"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["print('hello')\n"],
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


class NotebookLoaderTest(unittest.TestCase):
    def test_load_canonical_notebook_adds_deterministic_ids_without_warnings(self) -> None:
        with TemporaryDirectory() as temp_dir:
            notebook_path = Path(temp_dir) / "missing_ids.ipynb"
            notebook_path.write_text(
                json.dumps(_raw_notebook_without_ids(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                first = load_canonical_notebook(notebook_path)
                second = load_canonical_notebook(notebook_path)

            self.assertEqual(caught, [])
            self.assertEqual(first.added_cell_id_count, 2)
            self.assertEqual(second.added_cell_id_count, 2)
            self.assertEqual(
                [cell.get("id") for cell in first.notebook.cells],
                [cell.get("id") for cell in second.notebook.cells],
            )

    def test_fixture_notebooks_are_checked_in_with_canonical_ids(self) -> None:
        fixtures_dir = Path("data/benchmarks/fixtures/uploads")
        notebook_paths = sorted(fixtures_dir.glob("*.ipynb"))
        self.assertTrue(notebook_paths)

        for notebook_path in notebook_paths:
            with self.subTest(notebook=notebook_path.name):
                original = json.loads(notebook_path.read_text(encoding="utf-8"))
                self.assertTrue(all(str(cell.get("id") or "").strip() for cell in original.get("cells", [])))

                stripped = copy.deepcopy(original)
                for cell in stripped.get("cells", []):
                    if isinstance(cell, dict):
                        cell.pop("id", None)

                canonicalized, added = canonicalize_notebook_payload(stripped)
                self.assertEqual(added, len(original.get("cells", [])))
                self.assertEqual(
                    [cell.get("id") for cell in original.get("cells", [])],
                    [cell.get("id") for cell in canonicalized.get("cells", [])],
                )


if __name__ == "__main__":
    unittest.main()
