import json
import unittest
from pathlib import Path


class KaggleHumanNotebookTests(unittest.TestCase):
    def test_notebook_cells_compile_and_tests_are_disabled_by_default(self):
        path = Path(__file__).resolve().parents[1] / "kaggle_human_study.ipynb"
        notebook = json.loads(path.read_text())
        self.assertEqual(notebook["nbformat"], 4)
        cells = notebook["cells"]
        self.assertTrue(cells)
        self.assertEqual(cells[0]["cell_type"], "markdown")
        code = []
        for index, cell in enumerate(cells):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            compile(source, f"{path}:cell{index}", "exec")
            self.assertEqual(cell["outputs"], [])
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cells[index - 1]["cell_type"], "markdown")
            code.append(source)
        self.assertIn("RUN_FINAL_EVALUATION = False", code[0])
        self.assertIn("FINAL_SEEDS = (42, 123, 777)", code[0])
        self.assertIn("human-study-development", "\n".join(code))
        self.assertIn("human-study'", "\n".join(code))
        self.assertIn("EXPECTED_MANIFEST_SHA256", "\n".join(code))
        self.assertIn("human_paired_differences.csv", code[-1])


if __name__ == "__main__":
    unittest.main()
