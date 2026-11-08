import json
import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT = Path(__file__).resolve().parents[1]


def load_runner():
    import importlib.util

    path = PROJECT / "scripts/run_original_kladb_external.py"
    spec = importlib.util.spec_from_file_location("run_original_kladb_external", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OriginalKladbExternalNotebookTests(unittest.TestCase):
    def test_notebook_cells_compile_and_training_is_opt_in(self):
        path = PROJECT / "kaggle_original_train_kladb_external.ipynb"
        notebook = json.loads(path.read_text())
        self.assertEqual(notebook["nbformat"], 4)
        code = []
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                compile(source, f"{path}:cell{index}", "exec")
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(notebook["cells"][index - 1]["cell_type"], "markdown")
                code.append(source)
        self.assertIn("RUN_TRAINING = False", code[0])
        self.assertIn("DECLARED_SEEDS = (42, 123, 777)", code[0])
        self.assertIn("--check-only", "\n".join(code))
        self.assertIn("cuda:1", "\n".join(code))
        self.assertIn("external_predictions.csv", "\n".join(code))

    def test_external_export_is_deterministic_when_source_is_available(self):
        source = PROJECT / "outputs/kladb_human_locked/full_manifest.csv"
        if not source.exists():
            self.skipTest("Generated KlaDB source manifest is not present")
        import importlib.util

        path = PROJECT / "scripts/export_kladb_external.py"
        spec = importlib.util.spec_from_file_location("export_kladb_external", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            manifest = module.export(source, Path(temporary))
            self.assertEqual(manifest["sha256"], "0165ebe6d79e52bd74a848aa76e788e27804bf78ba2e8c98126d119cb449eae9")
            self.assertEqual(manifest["counts"], {"rows": 5095, "positives": 254, "putative_negatives": 4841, "proteins": 81})
            self.assertEqual(json.loads((Path(temporary) / "external_test_manifest.json").read_text()), manifest)

    def test_predictions_use_full_protein_position(self):
        runner = load_runner()
        split = SimpleNamespace(records=[SimpleNamespace(name="P12345:778", sequence="A" * 25 + "K" + "A" * 25, label=1)])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "predictions.csv"
            runner.save_external_predictions(output, split, [0.8])
            with output.open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["position_1based"], "778")
            self.assertEqual(row["predicted_label"], "1")


if __name__ == "__main__":
    unittest.main()
