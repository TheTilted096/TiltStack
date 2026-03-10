#!/usr/bin/env python3
"""
Unified clustering pipeline for poker equity states.

Orchestrates the full K-means clustering workflow:
  1. Build C++ expander (if needed)
  2. Generate random sample indices
  3. Compute equity vectors for sampled indices
  4. Train K-means centroids
  5. Stream all states and assign cluster labels

Usage:
    python cluster_pipeline.py
    python cluster_pipeline.py -k 30000 --sample-size 20000000 -t 16 --gpu

All parameters can be overridden via command line.

Nathaniel Potter, 03-10-2026
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


class ClusterPipeline:
    """Orchestrates the clustering pipeline."""

    def __init__(self, build_dir: str, output_dir: str,
                 expander: str, k: int, sample_size: int, niter: int,
                 seed: int, threads: int, gpu: bool, verbose: bool = True):
        self.build_dir = Path(build_dir)
        self.output_dir = Path(output_dir)
        self.expander_name = expander
        self.k = k
        self.sample_size = sample_size
        self.niter = niter
        self.seed = seed
        self.threads = threads
        self.gpu = gpu
        self.verbose = verbose

        # Ensure directories exist
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Derive paths
        self.expander_path = self.build_dir / expander
        self.indices_path = self.output_dir / "sample_indices.bin"
        self.sample_path = self.output_dir / "river_sample.npy"
        self.centroids_path = self.output_dir / "river_centroids.npy"
        self.labels_path = self.output_dir / "river_labels.bin"

    def log(self, msg: str):
        """Print timestamped log message."""
        if self.verbose:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {msg}", file=sys.stderr)

    def run_command(self, cmd: list[str], env: dict = None) -> subprocess.CompletedProcess:
        """Run a shell command with optional environment."""
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        try:
            result = subprocess.run(
                cmd,
                env=full_env,
                check=True,
            )
            return result
        except subprocess.CalledProcessError as e:
            self.log(f"ERROR: Command failed with exit code {e.returncode}")
            sys.exit(1)

    def pipe_commands(self, cmd1: list[str], cmd2: list[str], env: dict = None):
        """Run cmd1 | cmd2 with piped communication."""
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        try:
            proc1 = subprocess.Popen(
                cmd1,
                env=full_env,
                stdout=subprocess.PIPE,
                stderr=sys.stderr,
            )

            proc2 = subprocess.Popen(
                cmd2,
                stdin=proc1.stdout,
                env=full_env,
                stderr=sys.stderr,
            )

            proc1.stdout.close()
            proc1.wait()
            proc2.wait()

            if proc1.returncode != 0 or proc2.returncode != 0:
                self.log(f"ERROR: Pipeline failed (exit codes: {proc1.returncode}, {proc2.returncode})")
                sys.exit(1)
        except Exception as e:
            self.log(f"ERROR: {e}")
            sys.exit(1)

    def step_build_expander(self):
        """Step 0: Build C++ expander if not already built."""
        if self.expander_path.exists():
            self.log(f"Expander already exists, skipping build.")
            return

        self.log(f"==> Step 0: Building {self.expander_name}...")
        self.run_command(["make"])
        self.log(f"Build complete.")

    def step_generate_indices(self):
        """Step 1a: Generate random sample indices."""
        if self.indices_path.exists():
            self.log(f"Indices already exist, skipping generation.")
            return

        self.log(f"==> Step 1/5: Generating {self.sample_size:,} random indices...")
        script = Path(__file__).parent / "generate_sample_indices.py"
        self.run_command([
            sys.executable, str(script),
            "-n", str(self.sample_size),
            "--seed", str(self.seed),
            "-o", str(self.indices_path),
        ])
        self.log(f"✓ Indices generated.")

    def step_compute_sample(self):
        """Step 1b: Compute equity vectors for sampled indices."""
        if self.sample_path.exists():
            self.log(f"Sample already exists, skipping computation.")
            return

        self.log(f"==> Step 2/5: Computing equity vectors for {self.sample_size:,} samples...")

        thread_env = {
            "OMP_NUM_THREADS": str(self.threads),
            "MKL_NUM_THREADS": str(self.threads),
            "OPENBLAS_NUM_THREADS": str(self.threads),
        }

        pipe_script = Path(__file__).parent / "pipe_to_npy.py"
        expander_cmd = [str(self.expander_path), "--sample", str(self.indices_path), "-"]
        pipe_cmd = [sys.executable, str(pipe_script), "-o", str(self.sample_path)]

        self.pipe_commands(expander_cmd, pipe_cmd, env=thread_env)
        self.log(f"✓ Sample computed.")

    def step_train_centroids(self):
        """Step 2: Train K-means centroids on the sample."""
        if self.centroids_path.exists():
            self.log(f"Centroids already exist, skipping training.")
            return

        self.log(f"==> Step 3/5: Training K={self.k:,} centroids ({self.niter} iterations)...")

        thread_env = {
            "OMP_NUM_THREADS": str(self.threads),
            "MKL_NUM_THREADS": str(self.threads),
            "OPENBLAS_NUM_THREADS": str(self.threads),
        }

        script = Path(__file__).parent / "river_clusterer.py"
        cmd = [
            sys.executable, str(script),
            "--train-only", str(self.sample_path),
            "-k", str(self.k),
            "-i", str(self.niter),
            "--seed", str(self.seed),
            "-o", str(self.output_dir / "river"),
        ]
        if self.gpu:
            cmd.append("--gpu")

        self.run_command(cmd, env=thread_env)
        self.log(f"✓ Centroids trained.")

    def step_assign_labels(self):
        """Step 3: Stream all states and assign cluster labels."""
        if self.labels_path.exists():
            self.log(f"Labels already exist, skipping assignment.")
            return

        self.log(f"==> Step 4/5: Assigning labels to all states...")

        thread_env = {
            "OMP_NUM_THREADS": str(self.threads),
            "MKL_NUM_THREADS": str(self.threads),
            "OPENBLAS_NUM_THREADS": str(self.threads),
        }

        script = Path(__file__).parent / "river_clusterer.py"
        expander_cmd = [str(self.expander_path), "--quiet", "-"]
        clusterer_cmd = [
            sys.executable, str(script),
            "--assign-only", str(self.centroids_path),
            "--stdin",
            "-o", str(self.output_dir / "river"),
        ]
        if self.gpu:
            clusterer_cmd.append("--gpu")

        self.pipe_commands(expander_cmd, clusterer_cmd, env=thread_env)
        self.log(f"✓ Labels assigned.")

    def run(self):
        """Execute the full pipeline."""
        start_time = time.time()
        self.log(f"Starting clustering pipeline")
        self.log(f"Parameters: K={self.k:,}, sample_size={self.sample_size:,}, "
                 f"niter={self.niter}, threads={self.threads}, gpu={self.gpu}")

        try:
            self.step_build_expander()
            self.step_generate_indices()
            self.step_compute_sample()
            self.step_train_centroids()
            self.step_assign_labels()

            elapsed = time.time() - start_time
            self.log(f"==> Pipeline complete in {elapsed / 60:.1f} minutes")
            self.log(f"Centroids: {self.centroids_path}")
            self.log(f"Labels:    {self.labels_path}")
            self.log(f"(You can delete {self.sample_path} and {self.indices_path} to reclaim space)")
        except Exception as e:
            self.log(f"FATAL: {e}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Unified K-means clustering pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cluster_pipeline.py
  python cluster_pipeline.py -k 30000 --sample-size 20000000 -t 16
  python cluster_pipeline.py --gpu -t 8
        """,
    )

    parser.add_argument(
        "-e", "--expander", default="river_expander",
        help="Name of C++ expander executable in build/ (default: river_expander)",
    )
    parser.add_argument(
        "-k", "--clusters", type=int, default=30_000,
        help="Number of clusters (default: 30000)",
    )
    parser.add_argument(
        "-s", "--sample-size", type=int, default=20_000_000,
        help="Training sample size (default: 20000000)",
    )
    parser.add_argument(
        "-i", "--niter", type=int, default=25,
        help="K-means iterations (default: 25)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "-t", "--threads", type=int, default=16,
        help="CPU thread limit (default: 16)",
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="Use GPU for K-means training",
    )
    parser.add_argument(
        "-b", "--build-dir", default="build",
        help="Build directory for C++ binaries (default: build/)",
    )
    parser.add_argument(
        "-o", "--output-dir", default="output",
        help="Output directory for results (default: output/)",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Suppress status messages",
    )

    args = parser.parse_args()

    pipeline = ClusterPipeline(
        build_dir=args.build_dir,
        output_dir=args.output_dir,
        expander=args.expander,
        k=args.clusters,
        sample_size=args.sample_size,
        niter=args.niter,
        seed=args.seed,
        threads=args.threads,
        gpu=args.gpu,
        verbose=not args.quiet,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
