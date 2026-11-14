import os
import subprocess
import tempfile
import unittest
from pathlib import Path


HPC_DIR = Path(__file__).resolve().parents[1] / "hpc"


class HpcSlurmTests(unittest.TestCase):
    def test_job_files_have_valid_bash_syntax_and_refuse_login_execution(self):
        for name in (
            "setup_environment.sbatch",
            "train_development.sbatch",
            "compare_baseline_hybrid.sbatch",
        ):
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

    def test_full_epoch_switch_and_paired_comparison(self):
        training = (HPC_DIR / "train_development.sbatch").read_text()
        comparison = (HPC_DIR / "compare_baseline_hybrid.sbatch").read_text()
        self.assertIn('RUN_PATIENCE="${RUN_PATIENCE:-5}"', training)
        self.assertIn('--patience "${RUN_PATIENCE}"', training)
        self.assertIn("RUN_MODEL=baseline bash", comparison)
        self.assertIn("RUN_MODEL=hybridkla_architecture_aligned bash", comparison)
        self.assertLess(
            comparison.index("RUN_MODEL=baseline bash"),
            comparison.index("RUN_MODEL=hybridkla_architecture_aligned bash"),
        )
        self.assertIn("73ddb128c441d7b213c4636e6bb6ce34501e82143c2b9bc06ec329d68a422993", comparison)

    def test_environment_checks_allocated_gpu_before_ready(self):
        source = (HPC_DIR / "setup_environment.sbatch").read_text()
        self.assertIn("#SBATCH --gres=gpu:1", source)
        self.assertIn("torch==2.7.1+cu118", source)
        self.assertIn("torch.cuda.is_available()", source)
        self.assertLess(source.index("torch.cuda.is_available()"), source.index('touch "${VENV_DIR}/.ready"'))
        training = (HPC_DIR / "train_development.sbatch").read_text()
        self.assertIn('REPO_DIR}/../.venv-hpc-cu118', source)
        self.assertIn('REPO_DIR}/../.venv-hpc-cu118', training)

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

    def test_clear_bilstm_jobs_share_protocol_and_require_compute_allocation(self):
        runner = HPC_DIR / "run_clear_development.sh"
        source = runner.read_text()
        self.assertIn("--run clear-lactyl-development", source)
        self.assertIn("--head-learning-rate 1e-4", source)
        self.assertIn("token_gated_protbert_bilstm", source)
        self.assertIn("token_gated_protbert_head_lr", source)
        subprocess.run(["bash", "-n", str(runner)], check=True)
        for filename in (
            "project_mmkla_bilstm_control.sbatch",
            "project_mmkla_bilstm_clear.sbatch",
        ):
            path = HPC_DIR / filename
            with self.subTest(job=filename):
                subprocess.run(["bash", "-n", str(path)], check=True)
                self.assertIn("#SBATCH --partition=cvlab-a5000", path.read_text())
                self.assertIn("#SBATCH --gres=gpu:1", path.read_text())
        environment = os.environ.copy()
        for key in ("SLURM_JOB_ID", "SLURMD_NODENAME", "SLURM_JOB_PARTITION"):
            environment.pop(key, None)
        result = subprocess.run(
            ["bash", str(runner)], capture_output=True, text=True, env=environment,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Refusing to train outside a Slurm compute allocation", result.stderr)

    def test_full_protbert_job_keeps_control_training_protocol(self):
        runner = (HPC_DIR / "run_clear_development.sh").read_text()
        job = HPC_DIR / "project_mmkla_full_protbert_clear.sbatch"
        subprocess.run(["bash", "-n", str(job)], check=True)
        self.assertIn("RUN_MODEL=token_gated_protbert_full30_last4", job.read_text())
        self.assertIn("#SBATCH --partition=cvlab-a5000", job.read_text())
        self.assertIn("#SBATCH --gres=gpu:1", job.read_text())
        self.assertIn("--encoder-layers 30 --train-last-encoder-layers 4", runner)
        self.assertIn("--head-learning-rate 1e-4", runner)
        self.assertIn("--epochs 30 --patience 5 --batch-size 4", runner)
        self.assertIn("--run clear-lactyl-development", runner)

    def test_cksaap_job_keeps_four_layer_protbert_protocol(self):
        runner = (HPC_DIR / "run_clear_development.sh").read_text()
        job = HPC_DIR / "project_mmkla_cksaap_clear.sbatch"
        subprocess.run(["bash", "-n", str(job)], check=True)
        self.assertIn("RUN_MODEL=token_gated_protbert_cksaap", job.read_text())
        self.assertIn("#SBATCH --partition=cvlab-a5000", job.read_text())
        self.assertIn("#SBATCH --gres=gpu:1", job.read_text())
        self.assertIn("--model-name Rostlab/prot_bert --encoder-layers 4", runner)
        self.assertIn("model_args+=(--add-cksaap)", runner)
        self.assertIn("optimizer_args+=(--head-learning-rate 1e-4)", runner)
        self.assertIn("--epochs 30 --patience 5 --batch-size 4", runner)

    def test_new_clear_jobs_keep_locked_development_protocol(self):
        runner = (HPC_DIR / "run_clear_development.sh").read_text()
        for filename, model in (
            ("project_mmkla_full_protbert_all_clear.sbatch", "token_gated_protbert_full30_all"),
            ("project_mmkla_cksaap_projection_clear.sbatch", "token_gated_protbert_cksaap_projection"),
        ):
            path = HPC_DIR / filename
            with self.subTest(job=filename):
                subprocess.run(["bash", "-n", str(path)], check=True)
                self.assertIn(f"RUN_MODEL={model}", path.read_text())
                self.assertIn("#SBATCH --partition=cvlab-a5000", path.read_text())
                self.assertIn("#SBATCH --gres=gpu:1", path.read_text())
        self.assertIn("--encoder-layers 30)", runner)
        self.assertIn("architecture=token_gated_cksaap_projection", runner)
        self.assertIn("--add-cksaap", runner)
        self.assertIn("--run clear-lactyl-development", runner)
        self.assertIn("--monitor-metric AUPRC", runner)
        self.assertIn("--head-learning-rate 1e-4", runner)
        self.assertNotIn("--run clear-lactyl ", runner)

    def test_full_esm2_job_trains_all_layers_on_clear_development(self):
        runner = (HPC_DIR / "run_clear_development.sh").read_text()
        job = HPC_DIR / "project_mmkla_esm2_150m_full30_clear.sbatch"
        subprocess.run(["bash", "-n", str(job)], check=True)
        self.assertIn("RUN_MODEL=token_gated_esm2_150m_full30", job.read_text())
        self.assertIn("#SBATCH --partition=cvlab-a5000", job.read_text())
        self.assertIn("#SBATCH --gres=gpu:1", job.read_text())
        self.assertIn("--model-name facebook/esm2_t30_150M_UR50D --encoder-layers 30", runner)
        self.assertIn("--run clear-lactyl-development", runner)
        self.assertIn("--head-learning-rate 1e-4", runner)
        self.assertIn("--monitor-metric AUPRC", runner)
        self.assertIn("--epochs 30 --patience 5 --batch-size 4", runner)


if __name__ == "__main__":
    unittest.main()
