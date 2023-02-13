import logging
import wandb

import gym
import torch
import numpy as np

from neat.experiments.acrobot_balancing.kwargs import KWARGS

import neat.population as pop
import neat.experiments.acrobot_balancing.config as c
import neat.experiments.acrobot_balancing.config_test as c_test
from neat.visualize import draw_net
from neat.phenotype.feed_forward import FeedForwardNet

sweep_configuration = {
    'method': 'bayes',
    'name': 'Control',
    'metric': {
        'goal': 'maximize', 
        'name': 'Best Fitness'
		},
    'parameters': {
        'USE_BIAS': {'values': [False, True]},
        'POPULATION_SIZE': {'max': 20, 'min': 3},
        'CANDIDATE_LIMIT': {'values': [2, 3, 5, 7, 13, 25]},
        'SCALE_ACTIVATION': {'max': 7, 'min': 2},
        'SPECIATION_THRESHOLD': {'max' : 5, 'min' : 1},
        'CONNECTION_MUTATION_RATE': {'max': 1.0, 'min': 0.1},
        'CONNECTION_PERTURBATION_RATE': {'max': 1.0, 'min': 0.1},
        'ADD_NODE_MUTATION_RATE': {'max': 1.0, 'min': 0.1},
        'ADD_CONNECTION_MUTATION_RATE': {'max': 1.0, 'min': 0.1},
        'CROSSOVER_REENABLE_CONNECTION_GENE_RATE': {'max': 1.0, 'min': 0.1},
        'PERCENTAGE_TO_SAVE': {'max': 1.0, 'min': 0.1}
     }
}

# sweep_id = wandb.sweep(sweep=sweep_configuration, project="acrobot", entity="evolvingnn")
# print(sweep_id)

def test():
    KWARGS['POPULATION_SIZE'] = 5
    KWARGS['NUMBER_OF_GENERATIONS'] = 5
    KWARGS['MAX_EPISODE_STEPS'] = 100
    KWARGS['USE_CONTROL'] = True
    KWARGS['USE_ACER'] = True
    KWARGS['USE_ACER_WITH_WARMUP'] = True
    KWARGS['PERCENTAGE_TO_SAVE'] = 0.8
    KWARGS['CONNECTION_MUTATION_RATE'] = 0
    KWARGS['CONNECTION_PERTURBATION_RATE'] = 0
    KWARGS['ADD_NODE_MUTATION_RATE'] = 0
    KWARGS['ADD_CONNECTION_MUTATION_RATE'] = 0
    KWARGS['CROSSOVER_REENABLE_CONNECTION_GENE_RATE'] = 0
    
    config = c_test.AcrobotBalanceConfig(**KWARGS)
    neat = pop.Population(config)
    for solution, generation in neat.run():
        ### TODO Make sure single genome return is compatbile with frozen set iteration (maybe return as a frozen set with one element)
        # for i, genome in enumerate(solution):
        #     draw_net(genome, view=True, filename=f'./images/acrobot-ensemble-solution-{i}', show_disabled=True)

        # OpenAI Gym
        env = gym.make('Acrobot-v1')
        done = False
        observation = env.reset(seed = 0)

        voting_ensemble = [FeedForwardNet(genome, config) for genome in solution]
        #phenotype = FeedForwardNet(solution, config)

        while not done:
            env.render()
            observation = np.array([observation])
            obs = torch.Tensor(observation).cpu()
            pred = config.vote(voting_ensemble, obs)
            observation, reward, done, info = env.step(pred)

        env.close()


def train():

    wandb.init(config = KWARGS, group = 'Acrobot Control')

    kwargs = {
        'VERBOSE': wandb.config.VERBOSE,
        'NUM_INPUTS': wandb.config.NUM_INPUTS,
        'NUM_OUTPUTS': wandb.config.NUM_OUTPUTS,
        'USE_BIAS': wandb.config.USE_BIAS,
        'GENERATIONAL_ENSEMBLE_SIZE': wandb.config.GENERATIONAL_ENSEMBLE_SIZE,
        'CANDIDATE_LIMIT': wandb.config.CANDIDATE_LIMIT,
        'ACTIVATION': wandb.config.ACTIVATION,
        'SCALE_ACTIVATION': wandb.config.SCALE_ACTIVATION,
        'MAX_EPISODE_STEPS': wandb.config.MAX_EPISODE_STEPS,
        'FITNESS_THRESHOLD': wandb.config.FITNESS_THRESHOLD,
        'TOP_HEIGHT' : wandb.config.TOP_HEIGHT,
        'USE_CONTROL' : wandb.config.USE_CONTROL,
        'USE_ACER' : wandb.config.USE_ACER,
        'USE_ACER_WITH_WARMUP' : wandb.config.USE_ACER_WITH_WARMUP,
        'POPULATION_SIZE': wandb.config.POPULATION_SIZE,
        'NUMBER_OF_GENERATIONS': wandb.config.NUMBER_OF_GENERATIONS,
        'SPECIATION_THRESHOLD': wandb.config.SPECIATION_THRESHOLD,
        'CONNECTION_MUTATION_RATE': wandb.config.CONNECTION_MUTATION_RATE,
        'CONNECTION_PERTURBATION_RATE': wandb.config.CONNECTION_PERTURBATION_RATE,
        'ADD_NODE_MUTATION_RATE': wandb.config.ADD_NODE_MUTATION_RATE,
        'ADD_CONNECTION_MUTATION_RATE': wandb.config.ADD_CONNECTION_MUTATION_RATE,
        'CROSSOVER_REENABLE_CONNECTION_GENE_RATE': wandb.config.CROSSOVER_REENABLE_CONNECTION_GENE_RATE,
        'PERCENTAGE_TO_SAVE': wandb.config.PERCENTAGE_TO_SAVE,
    }
    kwargs['wandb'] = wandb

    logger = logging.getLogger(__name__)

    #logger.info(c.AcrobotBalanceConfig.DEVICE)
    neat = pop.Population(c.AcrobotBalanceConfig(**kwargs))
    for solution, generation in neat.run():
        for i, genome in enumerate(solution):
            draw_net(genome, view=True, filename=f'./images/acrobot-ensemble-solution-{i}', show_disabled=True)
        



if __name__ == '__main__':
    # for _ in range(10):
        # train()
    #train()
    #wandb.agent('2umr8r06', function=train)
    test()
