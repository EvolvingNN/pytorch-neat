import logging

import gym
import torch

import numpy as np
import pandas as pd

import neat.population as pop
import neat.experiments.pole_balancing.config as c
from neat.visualize import draw_net
from neat.phenotype.feed_forward import FeedForwardNet

import time
import optuna


logger = logging.getLogger(__name__)

config = c.PoleBalanceConfig

logger.info(config.DEVICE)

def objective(trial):

    config.SCALE_ACTIVATION = trial.suggest_float('SCALE_ACTIVATION', 1 , 10)

    config.FITNESS_THRESHOLD = 100000.0

    config.POPULATION_SIZE = trial.suggest_int('POP_SIZE', 5, 200)
    config.NUMBER_OF_GENERATIONS = 100
    config.SPECIATION_THRESHOLD = trial.suggest_float('SPEC_THRESH', 1.0, 5.0)

    config.CONNECTION_MUTATION_RATE = trial.suggest_float('CONN_MUT_RATE', 0.1, 0.9)
    config.CONNECTION_PERTURBATION_RATE = trial.suggest_float('CONN_PERT_RATE', 0.1, 0.9)
    config.ADD_NODE_MUTATION_RATE = trial.suggest_float('ADD_NODE_RATE', 0.1, 0.9)
    config.ADD_CONNECTION_MUTATION_RATE = trial.suggest_float('ADD_CONN_RATE', 0.1, 0.9)

    config.CROSSOVER_REENABLE_CONNECTION_GENE_RATE = trial.suggest_float('CROSSOVER', 0.1, 0.9)
    # Top percentage of species to be saved before mating
    config.PERCENTAGE_TO_SAVE = 0.80

    ttc = []
    for _ in range(5):
        neat = pop.Population(config)
        start = time.time()
        solution, generation = neat.run()
        end = time.time()

        ttc.append(end-start)

    return np.mean(ttc)


study_name = "pole_run_baseline_study"
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(direction = "minimize", study_name=study_name, storage=storage_name, load_if_exists=True)
study.optimize(objective, n_trials=3)

trial = study.best_trial

print('Accuracy: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))

df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
df.to_pickle(f"./{study_name}.pkl")

# optuna.visualization.plot_optimization_history(study)
# optuna.visualization.plot_slice(study) 
# optuna.visualization.plot_contour(study, params=['SCALE_ACTIVATION', 'POPULATION_SIZE', 'SPECIATION_THRESHOLD', 'CONNECTION_MUTATION_RATE'])