import logging
import wandb

import gym
import torch

from neat.experiments.acrobot_balancing.kwargs import KWARGS

import neat.population as pop
import neat.experiments.acrobot_balancing.config as c
from neat.visualize import draw_net
from neat.phenotype.feed_forward import FeedForwardNet

sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'goal': 'maximize', 
        'name': 'ACER'
		},
    'parameters': {
        'USE_BIAS': {'values': [False, True]},
        'POPULATION_SIZE': {'max': 100, 'min': 3},
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

sweep_id = wandb.sweep(sweep=sweep_configuration, project="acrobot", entity="evolvingnn")
print(sweep_id)

def train():

    wandb.init(config = KWARGS)

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
    solution, generation = neat.run()

    if solution is not None:
        logger.info('Found a Solution')
        draw_net(solution, view=True, filename='./images/acrobot-balancing-solution', show_disabled=True)

        # OpenAI Gym
        env = gym.make('Acrobot-v1')
        done = False
        observation = env.reset()

        fitness = 0
        phenotype = FeedForwardNet(solution, c.AcrobotBalanceConfig)

        while not done:
            env.render()
            obs = torch.Tensor([observation]).cpu()

            pred = round(float(phenotype(obs)))
            observation, reward, done, info = env.step(pred)

            fitness += reward
        env.close()

if __name__ == '__main__':
    # for _ in range(10):
        # train()
        
    wandb.agent('4xm1seek', function=train)
