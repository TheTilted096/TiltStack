"""
Interactive training controls for DeepCFR.

The trainer owns the control object and updates its progress fields. This
module owns only terminal input/output so future distributed workers can keep
training logic separate from rank-0 user interaction.
"""

import sys
import threading

from network_training import _ts


class TrainingControl:
    """Shared state between the training loop and the menu thread."""

    def __init__(self, total_iters: int, total_epochs: int) -> None:
        self._lock = threading.Lock()
        self.stop_after_iter = total_iters
        self.policy_epochs = total_epochs
        self.current_iter = 0
        self.current_epoch = 0
        self.phase = "advantage"  # "advantage" | "policy" | "done"


def run_menu(ctrl: TrainingControl, term) -> None:
    """Daemon thread: display prompts on the terminal and apply user commands."""
    while True:
        with ctrl._lock:
            if ctrl.phase == "done":
                return
            if ctrl.phase == "advantage":
                cur = ctrl.current_iter
                stop = ctrl.stop_after_iter
                prompt = f"\nenter number of adv iters ({stop}) > "
            else:
                cur_ep = ctrl.current_epoch
                epochs = ctrl.policy_epochs
                prompt = f"\nenter number of policy epochs ({epochs}) > "

        term.write(prompt)
        term.flush()

        try:
            line = sys.stdin.readline()
        except OSError:
            return

        if not line:  # EOF
            return

        line = line.strip()
        if not line:
            continue

        try:
            val = int(line)
        except ValueError:
            term.write("  ! not a number\n")
            term.flush()
            continue

        with ctrl._lock:
            if ctrl.phase == "advantage":
                cur = ctrl.current_iter
                if val <= cur:
                    term.write(f"  ! {val} <= current iter {cur}\n")
                else:
                    ctrl.stop_after_iter = val
                    term.write(f"  -> adv iters set to {val}\n")
                    print(f"[{_ts()}]  [ctrl] adv iters set to {val}", flush=True)
            elif ctrl.phase == "policy":
                cur_ep = ctrl.current_epoch
                if val <= cur_ep:
                    term.write(f"  ! {val} <= current epoch {cur_ep}\n")
                else:
                    ctrl.policy_epochs = val
                    term.write(f"  -> policy epochs set to {val}\n")
                    print(f"[{_ts()}]  [ctrl] policy epochs set to {val}", flush=True)
        term.flush()
