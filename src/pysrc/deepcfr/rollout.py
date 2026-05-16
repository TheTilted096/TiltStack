"""Rollout generation helpers for DeepCFR training."""

from dataclasses import dataclass
import time
from typing import Optional

from network_training import run_inference_loop


@dataclass(frozen=True)
class RolloutStats:
    seconds: float
    rollouts: int
    adv_new: int
    adv_size: int
    pol_new: Optional[int] = None
    pol_size: Optional[int] = None


def run_player_rollout(
    *,
    orch,
    device,
    adv_nets,
    adv_res,
    strat_res,
    hero: bool,
    iteration: int,
    samples: int,
    collect_policy: bool,
) -> RolloutStats:
    """Run one player's DeepCFR traversal and return reservoir deltas."""
    player = int(hero)
    adv_before = adv_res[player].n_seen
    pol_before = strat_res.n_seen if collect_policy else 0

    t0 = time.perf_counter()
    orch.start_deepcfr_iteration(
        hero,
        iteration,
        samples,
        adv_res[0]._cpp,
        adv_res[1]._cpp,
        pol_res=strat_res._cpp if collect_policy else None,
    )
    run_inference_loop(orch, device, adv_nets)
    orch.wait_iteration()
    rollout_secs = time.perf_counter() - t0

    rollouts = sum(s.rollout_count() for s in orch.schedulers)
    orch.clear_buffers()

    adv_new = adv_res[player].n_seen - adv_before
    adv_size = adv_res[player].size

    if not collect_policy:
        return RolloutStats(
            seconds=rollout_secs,
            rollouts=rollouts,
            adv_new=adv_new,
            adv_size=adv_size,
        )

    return RolloutStats(
        seconds=rollout_secs,
        rollouts=rollouts,
        adv_new=adv_new,
        adv_size=adv_size,
        pol_new=strat_res.n_seen - pol_before,
        pol_size=strat_res.size,
    )
