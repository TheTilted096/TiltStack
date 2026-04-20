"""
tb_launch.py — TensorBoard launch helper shared by training scripts.

TensorBoard is launched as a subprocess with stdout/stderr redirected to
/dev/null, so all gRPC/absl C++ startup noise is isolated from the parent
terminal.  The subprocess is terminated automatically when the parent exits.
"""

import sys
import subprocess
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


def launch_tb(
    log_dir: Path, port: int = 6006, reload_interval: int = 20
) -> SummaryWriter:
    """
    Launch TensorBoard pointed at log_dir.parent and return a SummaryWriter
    for log_dir.  The caller is responsible for calling writer.close() and
    for registering any cleanup handlers.
    """
    subprocess.Popen(
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
    return SummaryWriter(log_dir=str(log_dir))
