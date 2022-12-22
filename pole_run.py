import logging

import gym
import torch

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

    config.SCALE_ACTIVATION = trial.suggest_float('SCALE_ACTIVATION', 1 , 10, log = True)

    config.FITNESS_THRESHOLD = 100000.0

    config.POPULATION_SIZE = trial.suggest_int('POPULATION_SIZE', 5, 200)
    config.NUMBER_OF_GENERATIONS = trial.suggest_int('NUMBER_OF_GENERATIONS', 10, 100)
    config.SPECIATION_THRESHOLD = trial.suggest_float('SPECIATION_THRESHOLD', 1.0, 5.0)

    config.CONNECTION_MUTATION_RATE = trial.suggest_float('CONNECTION_MUTATION_RATE', 0.1, 0.9)
    config.CONNECTION_PERTURBATION_RATE = 0.90
    config.ADD_NODE_MUTATION_RATE = 0.03
    config.ADD_CONNECTION_MUTATION_RATE = 0.5

    config.CROSSOVER_REENABLE_CONNECTION_GENE_RATE = 0.25

    # Top percentage of species to be saved before mating
    config.PERCENTAGE_TO_SAVE = 0.80

    neat = pop.Population(config)
    start = time.time()
    solution, generation = neat.run()
    end = time.time()

    # if solution is not None:
    #     logger.info('Found a Solution')
    #     draw_net(solution, view=True, filename='./images/pole-balancing-solution', show_disabled=True)

    #     # OpenAI Gym
    #     env = gym.make('LongCartPole-v0')
    #     done = False
    #     observation = env.reset()

    #     fitness = 0
    #     phenotype = FeedForwardNet(solution, c.PoleBalanceConfig)

    #     while not done:
    #         env.render()
    #         input = torch.Tensor([observation]).to(c.PoleBalanceConfig.DEVICE)

    #         pred = round(float(phenotype(input)))
    #         observation, reward, done, info = env.step(pred)

    #         fitness += reward
    #     env.close()
    return (end - start)

study = optuna.create_study(direction = "minimize")
study.optimize(objective, n_trials=3)

trial = study.best_trial

print('Accuracy: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))

optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_slice(study) 
optuna.visualization.plot_contour(study, params=['SCALE_ACTIVATION', 'POPULATION_SIZE', 'SPECIATION_THRESHOLD', 'CONNECTION_MUTATION_RATE'])