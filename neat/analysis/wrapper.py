import pandas as pd

from neat.analysis.algorithms import ALGORITHMS


def run_trial_analysis(genomes, ensemble_evaluator):

    # Older versions of our codebase used a version of this method
    # that takes in a "prediction map."
    # Check to see if the caller is using this older system.
    is_in_compat_mode = type(genomes) is dict
    if is_in_compat_mode:

        # Copy the data in old format to data in new format
        pred_map = genomes
        genomes = list(pred_map.keys())
        for genome in genomes:
            genome.predictions = pred_map[genome]

        # Create a new ensemble evaluator based on the old format
        old_ensemble_evaluator = ensemble_evaluator
        ensemble_evaluator = lambda ensemble: old_ensemble_evaluator(
            [g.predictions for g in ensemble]
        )

    algorithm_results = {
        name: algo(genomes, ensemble_evaluator) for name, algo in ALGORITHMS.items()
    }

    # Clean up the compat mode modifications we made
    if is_in_compat_mode:
        for genome in genomes:
            delattr(genome, "predictions")

    return pd.DataFrame(algorithm_results)
