"""Public-repository packaging invariants; no empirical analysis is rerun."""
import ast
import csv
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class AnalysisOnlyRelease(unittest.TestCase):
    def test_no_panel_drawing_tree(self):
        self.assertFalse((ROOT / "plotting").exists())
        self.assertFalse((ROOT / "panels").exists())

    def test_no_history_or_release_recovery_tree(self):
        self.assertFalse((ROOT / "history").exists())
        self.assertFalse((ROOT / "QA").exists())
        for path in ROOT.rglob("*"):
            if path.is_file():
                low = path.as_posix().lower()
                self.assertNotIn("history_pre_", low)
                self.assertNotIn("history_before_", low)

    def test_no_graphics_imports_in_scientific_source(self):
        for path in (ROOT / "src").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    names = [item.name.split(".")[0] for item in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module.split(".")[0]]
                else:
                    names = []
                self.assertFalse(set(names) & {"matplotlib", "seaborn", "plotly", "fitz"}, str(path))

    def test_current_analysis_map(self):
        with (ROOT / "manifests/RESULT_SOURCE_ANALYSIS_MAP.tsv").open() as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        self.assertEqual(len({row["panel"] for row in rows}), 34)
        self.assertTrue(all(row["analysis_provenance_status"] for row in rows))
        self.assertTrue(all("plot" not in row["analysis_paths_in_GitHub"].lower() for row in rows))

    def test_minimal_public_docs_exist(self):
        for rel in [
            "README.md", "CITATION.cff", "LICENSE",
            "docs/RUNBOOK.md", "docs/METHODS_AND_SCOPE.md",
            "docs/REPRODUCIBILITY.md", "docs/RELEASE_QA.md",
        ]:
            self.assertTrue((ROOT / rel).is_file(), rel)

if __name__ == "__main__":
    unittest.main()
