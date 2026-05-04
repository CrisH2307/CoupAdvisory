from coup.research.bc import benchmark_policy_vs_random, train_behavior_clone
from coup.research.dataset import build_dataset_from_sim_dir, build_rows_from_events
from coup.research.markov import (
    build_action_transitions_from_events,
    build_transition_dataset_from_sim_dir,
    encode_markov_state,
)


def evaluate_policies_by_scenario(*args, **kwargs):
    from coup.research.scenario_eval import evaluate_policies_by_scenario as _impl

    return _impl(*args, **kwargs)

__all__ = [
    "benchmark_policy_vs_random",
    "build_action_transitions_from_events",
    "build_dataset_from_sim_dir",
    "build_rows_from_events",
    "build_transition_dataset_from_sim_dir",
    "evaluate_policies_by_scenario",
    "encode_markov_state",
    "train_behavior_clone",
]
