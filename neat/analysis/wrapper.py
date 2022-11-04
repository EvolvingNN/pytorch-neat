import pandas as pd

from neat.analysis.algorithms import (
    random_selection_accuracies,
    greedy_1_selection_accuracies,
    greedy_2_selection_accuracies,
    diversity_rr_selection_accuracies,
)


def run_trial_analysis(final_population_prediction_map, ensemble_evaluator):
    algorithms_to_run = {
        "random": random_selection_accuracies,
        "greedy1": greedy_1_selection_accuracies,
        "greedy2": greedy_2_selection_accuracies,
        "diversity": diversity_rr_selection_accuracies,
    }
    algorithm_results = {
        name: algo(final_population_prediction_map, ensemble_evaluator)
        for name, algo in algorithms_to_run.items()
    }
    return pd.DataFrame(algorithm_results)
