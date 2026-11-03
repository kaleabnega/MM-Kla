import csv
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_hybridkla_overlap import audit, load_training_workbook


class HybridKlaOverlapTests(unittest.TestCase):
    def make_workbook(self, path, rows):
        header = ("UniProt_ID", "Sequence", "Label")
        xml_rows = []
        for index, values in enumerate((header, *rows), start=1):
            cells = "".join(
                f'<c r="{column}{index}" t="inlineStr"><is><t>{value}</t></is></c>'
                for column, value in zip("ABC", values)
            )
            xml_rows.append(f'<row r="{index}">{cells}</row>')
        xml = (
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<sheetData>{"".join(xml_rows)}</sheetData></worksheet>'
        )
        with ZipFile(path, "w") as workbook:
            workbook.writestr("xl/worksheets/sheet1.xml", xml)

    def make_split(self, root, name, sequence, label, site_id="P1_26"):
        with (root / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("Site_ID", "ACC_ID", "Sequence", "Label"))
            writer.writeheader()
            writer.writerow({"Site_ID": site_id, "ACC_ID": site_id.rsplit("_", 1)[0],
                             "Sequence": sequence, "Label": label})

    def test_exact_overlap_and_label_disagreement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_window = "A" * 25 + "K" + "A" * 25
            other_window = "C" * 25 + "K" + "C" * 25
            workbook = root / "train_data.xlsx"
            self.make_workbook(workbook, [("train1", source_window, "1")])
            self.make_split(root, "train", source_window, "1")
            self.make_split(root, "validation", other_window, "0")
            self.make_split(root, "internal_test", source_window, "0")
            self.make_split(root, "external_test", other_window, "0")
            result = audit(workbook, root)
            self.assertEqual(result["source"]["generic_train_ids"], 1)
            self.assertEqual(result["source"]["accession_position_ids"], 0)
            self.assertEqual(result["partitions"]["train"]["matched_same_label"], 1)
            self.assertEqual(result["partitions"]["internal_test"]["matched_opposite_label"], 1)
            self.assertEqual(result["partitions"]["external_test"]["exact_window_overlap"], 0)

    def test_accession_position_overlap_is_counted_separately(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_window = "A" * 25 + "K" + "A" * 25
            other_window = "C" * 25 + "K" + "C" * 25
            workbook = root / "train_data.xlsx"
            self.make_workbook(workbook, [("P1_26", source_window, "1")])
            for partition in ("train", "validation", "internal_test"):
                self.make_split(root, partition, other_window, "0", "P2_26")
            self.make_split(root, "external_test", other_window, "0", "P1_52")
            result = audit(workbook, root)
            self.assertEqual(result["source"]["accession_position_ids"], 1)
            external = result["partitions"]["external_test"]
            self.assertEqual(external["exact_window_overlap"], 0)
            self.assertEqual(external["exact_site_id_overlap"], 0)
            self.assertEqual(external["rows_on_matching_accessions"], 1)
            self.assertEqual(external["distinct_matching_accessions"], 1)

    def test_invalid_source_window_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train_data.xlsx"
            self.make_workbook(path, [("train1", "AAA", "1")])
            with self.assertRaisesRegex(ValueError, "Invalid sequence or label"):
                load_training_workbook(path)


if __name__ == "__main__":
    unittest.main()
