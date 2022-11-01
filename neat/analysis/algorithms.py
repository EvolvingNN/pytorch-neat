from collections import deque
import numpy as np
from neat.species import Species
from neat.utils import random_ensemble_generator

"""
A set of algoritms needed for each trial's analysis
These algorithms include:
- Random ensemble (control)
- Greedy 1
- Greedy 2
- Diversity selection (round robin by speciation)

Each algorithm here:
Consumes:
    - genomes prediction map
      - each genome key in the map *must* have the fitness member set
    - evaluate function (e.g. lambda) that returns accuracy of a given ensemble
      - given ensemble is represented as a list of the
        2d predicition arrays for the ensembled genomes
Returns:
    - list of size len(pred_map) that has the accuracy
      of the (i+1)-size ensemble created by the algorithm for any index i
"""


def random_selection_accuracies(pred_map, eval_func, ensembles_per_k=1):
    accuracies = []
    for k in range(1, len(pred_map) + 1):
        k_acc = []
        for ensemble in random_ensemble_generator(
            genomes=pred_map.keys(), k=k, limit=ensembles_per_k
        ):
            ensemble_member_results = [pred_map[x] for x in ensemble]
            k_acc.append(eval_func(ensemble_member_results))
        accuracies.append(np.mean(k_acc))
    return accuracies


def greedy_1_selection_accuracies(pred_map, eval_func):
    # Some variables needed for the greedy algorithm
    # genomes_left is the genomes left to choose from
    # genomes_picked is the current best predicted k-wise ensemble
    genomes_left = {*pred_map.keys()}
    genomes_picked = []
    accuracies = []

    # Remove the genome that improves ensemble the most after each round
    while genomes_left:

        # Initialize this round's variables
        best_accuracy = float("-inf")
        best_genome = None

        # Find the genome that best improves the current ensemble (genomes_picked)
        for genome in genomes_left:
            ensemble = [*genomes_picked, genome]
            ensemble_member_results = [pred_map[x] for x in ensemble]
            ensemble_accuracy = eval_func(ensemble_member_results)
            if ensemble_accuracy > best_accuracy:
                best_accuracy = ensemble_accuracy
                best_genome = genome

        # Some housekeeping to finish off the round
        genomes_left.remove(best_genome)
        genomes_picked.append(best_genome)
        accuracies.append(best_accuracy)
    return accuracies


def greedy_2_selection_accuracies(pred_map, eval_func):
    genomes = list(pred_map.keys())
    genomes.sort(reverse=True, key=lambda g: g.fitness)
    predictions_in_order = [pred_map[g] for g in genomes]
    return __accuracies_for_predictions_in_order(predictions_in_order, eval_func)


def diversity_rr_selection_accuracies(pred_map, eval_func, speciation_threshold=3.0):

    # Step 1: Divide genomes based on speciation threshold
    species = __speciate(pred_map.keys(), speciation_threshold)

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
    predictions_in_order = [pred_map[g] for g in genomes_in_order]
    return __accuracies_for_predictions_in_order(predictions_in_order, eval_func)


def __accuracies_for_predictions_in_order(predictions_in_order, eval_func):
    """
    Creates the accuracies for a list of networks' predictions in their ensemble order
    E.g. the predictions for an ensemble of size 1 would be predictions_in_order[0:1],
    and the predictions for an ensemble of size k would be predictions_in_order[0:k]
    """
    return [
        eval_func(predictions_in_order[0:k])
        for k in range(1, len(predictions_in_order) + 1)
    ]


def __speciate(genomes, speciation_threshold):
    """
    Copied and modified from population.py
    Creates a list of species, where each species is a list of genomes
    The speciation takes place based on genomes' genetic diversity
    """

    species = []

    def speciate(genome):
        for s in species:
            if Species.species_distance(genome, s.model_genome) <= speciation_threshold:
                s.members.append(genome)
                return

        # Did not match any current species. Create a new one
        new_species = Species(len(species), genome, 0)
        new_species.members.append(genome)
        species.append(new_species)

    for genome in genomes:
        speciate(genome)

    return [s.members for s in species]
