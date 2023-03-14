from collections import deque
import numpy as np
from neat.utils import random_ensemble_generator, speciate

"""
A set of algorithms needed for each trial's analysis
These algorithms include:
- Random ensemble (control)
- Greedy 1
- Greedy 2
- Diversity selection (round robin by speciation)

Each algorithm here:
Consumes:
    - genomes
      - each genome *must* have the fitness member set
    - evaluate function (e.g. lambda) that returns accuracy of a given ensemble
      - given ensemble is represented as a list of genomes
Returns:
    - list of size len(genomes) that has the accuracy
      of the (i+1)-size ensemble created by the algorithm for any index i
"""


def random_selection_accuracies(genomes, train_activations_map, test_activations_map, ensemble_evaluator, ensembles_per_k=1):

    accuracies = []
    for k in range(1, len(genomes) + 1):
        k_acc = []
        for ensemble in random_ensemble_generator(
            genomes=genomes, k=k, limit=ensembles_per_k
        ):
            ensemble_activations = [test_activations_map[candidate] for candidate in ensemble]
            k_acc.append(ensemble_evaluator(ensemble_activations, use_test_target = True))
        accuracies.append(np.mean(k_acc))
    return accuracies


def greedy_1_selection_accuracies(genomes, train_activations_map, test_activations_map, ensemble_evaluator):
    # Some variables needed for the greedy algorithm
    # genomes_left is the genomes left to choose from
    # genomes_picked is the current best predicted k-wise ensemble
    genomes_left = {*genomes}
    genomes_picked = []
    accuracies = []
    test_accuracies = []

    # Remove the genome that improves ensemble the most after each round
    while genomes_left:

        # Initialize this round's variables
        best_accuracy = float("-inf")
        best_genome = None

        # Find the genome that best improves the current ensemble using accuracy on training data (genomes_picked)
        for genome in genomes_left:
            ensemble = [*genomes_picked, genome]
            # Pull the ensemble activations from the training activations map
            ensemble_activations = [train_activations_map[candidate] for candidate in ensemble]
            ensemble_accuracy = ensemble_evaluator(ensemble_activations, use_test_target = False)
            if ensemble_accuracy > best_accuracy:
                best_accuracy = ensemble_accuracy
                best_genome = genome

        # Some housekeeping to finish off the round
        genomes_left.remove(best_genome)
        genomes_picked.append(best_genome)
        accuracies.append(best_accuracy)

    # Since genomes_picked is chosen greedily, we can evaluate each subsequent ensemble iteratively on the test data
    for ensemble in [genomes_picked[:i] for i in range(1, len(genomes_picked) + 1)]:
        
        ensemble_activations = [test_activations_map[candidate] for candidate in ensemble]
        ensemble_test_accuracy = ensemble_evaluator(ensemble_activations, use_test_target = True) #Specify use_test_target = True
        test_accuracies.append(ensemble_test_accuracy)

    return test_accuracies


def greedy_2_selection_accuracies(genomes, train_activations_map, test_activations_map, ensemble_evaluator):
    genomes_in_order = list(genomes)
    genomes_in_order.sort(reverse=True, key=lambda g: g.fitness)
    return __accuracies_for_genomes_in_order(genomes_in_order, test_activations_map, ensemble_evaluator)


def diversity_rr_selection_accuracies(genomes, train_activations_map, test_activations_map, ensemble_evaluator, speciation_threshold=3.0):

    # Step 1: Divide genomes based on speciation threshold
    species = speciate(genomes, speciation_threshold)

    # Step 2: Sort genomes in each species in descending order by their fitness
    for s in species:
        s.sort(reverse=True, key=lambda g: g.fitness)

    # Step 3: Pick the genomes from each species round-robin style
    species = [deque(s) for s in species]
    genomes_in_order = []

    # For each round-robin round while we still have species left to choose from
    while species:

        # Pick best genome for each species
        for s in species:
            genomes_in_order.append(s.popleft())

        # Remove empty species
        species = [s for s in species if s]

    # Step 4: Calculate the accuracies based on the picked genomes in order
    return __accuracies_for_genomes_in_order(genomes_in_order, test_activations_map, ensemble_evaluator)


def __accuracies_for_genomes_in_order(genomes_in_order, test_activations_map, ensemble_evaluator):
    """
    Creates the accuracies for a list of genomes in their ensemble order.
    E.g. the genomes for an ensemble of size 1 would be genomes_in_order[0:1],
    and the predictions for an ensemble of size k would be genomes_in_order[0:k]
    """
    return [
        ensemble_evaluator([test_activations_map[candidate] for candidate in genomes_in_order[0:k]], use_test_target = True) for k in range(1, len(genomes_in_order) + 1)
    ]


UCI_ALGORITHMS = {
    # Random algorithm runs 100 trials per ensemble size to produce better results
    "random": lambda p, _, m, e: random_selection_accuracies(p, _, m, e, ensembles_per_k=100),
    # # Greedy algorithms do not have any additional settings
    "greedy1": greedy_1_selection_accuracies,
    "greedy2": greedy_2_selection_accuracies,
    # Diversity algorithm requires varying speciation threshold
    **{
        f"diversity_{t}_threshold": (lambda t: lambda p, _, m, e: diversity_rr_selection_accuracies(
            p, _, m, e, speciation_threshold=t
        ))(t)
        for t in np.arange(1, 6, 1)
    },  
}
