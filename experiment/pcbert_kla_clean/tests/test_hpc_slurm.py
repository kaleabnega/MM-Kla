import os
import subprocess
import tempfile
import unittest
from pathlib import Path


HPC_DIR = Path(__file__).resolve().parents[1] / "hpc"


class HpcSlurmTests(unittest.TestCase):
    def test_job_files_have_valid_bash_syntax_and_refuse_login_execution(self):
        for name in ("setup_environment.sbatch", "train_development.sbatch"):
            path = HPC_DIR / name
            with self.subTest(name=name):
                subprocess.run(["bash", "-n", str(path)], check=True)
                source = path.read_text()
                self.assertIn("#SBATCH --partition=cvlab-a5000", source)
                self.assertIn("SLURMD_NODENAME", source)
                self.assertIn('HPC_DIR="$(pwd -P)"', source)
                self.assertNotIn("BASH_SOURCE", source)
                environment = os.environ.copy()
                for key in ("SLURM_JOB_ID", "SLURMD_NODENAME", "SLURM_JOB_PARTITION"):
                    environment.pop(key, None)
                result = subprocess.run(
                    ["bash", str(path)], capture_output=True, text=True, env=environment,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("outside a Slurm allocation", result.stderr)

    def test_training_job_uses_only_development_modes(self):
        source = (HPC_DIR / "train_development.sbatch").read_text()
        self.assertIn("--run human-study-development", source)
        self.assertIn("--development-only", source)
        self.assertIn("--gres=gpu:1", source)
        self.assertNotIn("--run human-study ", source)

    def test_hybrid_archive_rejected_before_extraction(self):
        helper = HPC_DIR / "extract_hybridkla_resources.py"
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "wrong.zip"
            archive.write_bytes(b"not the official archive")
            output = Path(temporary) / "resources"
            result = subprocess.run(
                ["python3", str(helper), str(archive), str(output)],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("checksum mismatch", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
