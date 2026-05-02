"""
tb_launch.py — TensorBoard launch helper shared by training scripts.

TensorBoard is launched as a subprocess with stdout/stderr redirected to
/dev/null, so all gRPC/absl C++ startup noise is isolated from the parent
terminal.  The subprocess is terminated automatically when the parent exits.
"""

import os
import signal
import sys
import subprocess
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


def launch_tb(
    log_dir: Path, port: int = 6006, reload_interval: int = 20
) -> tuple[SummaryWriter, subprocess.Popen]:
    """
    Launch TensorBoard pointed at log_dir.parent and return (writer, proc).
    The caller is responsible for writer.close() and proc.terminate().
    """
    proc = subprocess.Popen(
        [
            Path(sys.executable).parent / "tensorboard",
            "--logdir",
            str(log_dir.parent),
            "--port",
            str(port),
            "--reload_interval",
            str(reload_interval),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return SummaryWriter(log_dir=str(log_dir)), proc


def stop_tb(proc: subprocess.Popen, timeout: int = 5) -> None:
    """Kill TensorBoard and all its worker subprocesses.

    proc.terminate() only kills the top-level process; TensorBoard's gRPC
    workers are in the same process group (start_new_session=True makes the
    TB pid the group leader) so os.killpg reaches them all.
    """
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        proc.wait(timeout=timeout)
    except ProcessLookupError:
        pass  # already gone
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
