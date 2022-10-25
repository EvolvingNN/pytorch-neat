import logging
import copy
import math
import random

import torch
import numpy as np

from neat.phenotype.feed_forward import FeedForwardNet

logger = logging.getLogger(__name__)


def rand_uni_val():
    """
    Gets a random value from a uniform distribution on the interval [0, 1]
    :return: Float
    """
    return float(torch.rand(1))


def rand_bool():
    """
    Returns a random boolean value
    :return: Boolean
    """
    return rand_uni_val() <= 0.5


def get_best_genome(population):
    """
    Gets best genome out of a population
    :param population: List of Genome instances
    :return: Genome instance
    """
    population_copy = copy.deepcopy(population)
    population_copy.sort(key=lambda g: g.fitness, reverse=True)

    return population_copy[0]


def cache_genomes_results(genomes, dataset, config):
    genomes_to_results = {}
    for genome in genomes:
        results = []
        phenotype = FeedForwardNet(genome, config)
        phenotype.to(config.DEVICE)
        for input in dataset:
            input.to(config.DEVICE)
            prediction = phenotype(input)
            results.append(prediction.numpy())
        genomes_to_results[genome] = np.array(results)
    return genomes_to_results


def ensemble_picker(genomes, k=None, limit=None):
    '''A generator that randomly picks an ensemble from the given genomes of length k

    Parameters:
        genomes (iterable): the genomes to pick from
        k (None | int): None (for random size ensembles) or the ensemble size

    Yields:
        set of genomes to use in ensemble
    '''
    genomes = list(genomes)
    n = len(genomes)
    seen = set()

    if limit is None:
        limit = 2**n - 1 if k is None else math.comb(n, k)
        limit /= 2

    while len(seen) < limit:
        ensemble_length = random.randint(1, n) if k is None else k
        all_indices = list(range(n))
        random.shuffle(all_indices)
        ensemble_indices = frozenset(all_indices[0:ensemble_length])
        if ensemble_indices not in seen:
            seen.add(ensemble_indices)
            ensemble = {genomes[i] for i in ensemble_indices}
            yield ensemble
