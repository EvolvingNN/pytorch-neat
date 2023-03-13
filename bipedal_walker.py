import logging

import gym
import torch

import neat.population as pop
import neat.experiments.bipedal_walker.config as c
from neat.visualize import draw_net
from neat.phenotype.feed_forward import FeedForwardNet
from neat.experiments.bipedal_walker.kwargs import KWARGS
import wandb


logger = logging.getLogger(__name__)

# logger.info(c.BipedalWalkerConfig.DEVICE)

sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'goal': 'maximize', 
        'name': 'diversity'
		},
    'parameters': {
        'USE_BIAS': {'values': [False, True]},
        'GENERATIONAL_ENSEMBLE_SIZE': {'values': [2, 3, 5, 9]},
        'CANDIDATE_LIMIT': {'values': [2, 7, 25]},
        'SCALE_ACTIVATION': {'max': 7, 'min': 2},
        'USE_FITNESS_COEFFICIENT': {'values': [False, True]},
        'SPECIATION_THRESHOLD': {'values': [2.0, 3.0, 4.0, 5.0]},
        'CONNECTION_MUTATION_RATE': {'max': 1.0, 'min': 0.5},
        'CONNECTION_PERTURBATION_RATE': {'max': 1.0, 'min': 0.5},
        'ADD_NODE_MUTATION_RATE': {'max': 0.1, 'min': 0.001},
        'ADD_CONNECTION_MUTATION_RATE': {'max': 0.7, 'min': 0.1},
        'CROSSOVER_REENABLE_CONNECTION_GENE_RATE': {'max': 0.7, 'min': 0.1},
        'PERCENTAGE_TO_SAVE': {'max': 1.0, 'min': 0.5},
        'MAX_EPISODE_STEPS': {'values': [50, 100, 200, 500]}
     }
}

def train():
    wandb.init(config=KWARGS)

    kwargs = {
        'VERBOSE': wandb.config.VERBOSE,
        'NUM_INPUTS': wandb.config.NUM_INPUTS,
        'NUM_OUTPUTS': wandb.config.NUM_OUTPUTS,
        'USE_BIAS': wandb.config.USE_BIAS,
        'USE_CONV': wandb.config.USE_CONV,
        'GENERATIONAL_ENSEMBLE_SIZE': wandb.config.GENERATIONAL_ENSEMBLE_SIZE,
        'CANDIDATE_LIMIT': wandb.config.CANDIDATE_LIMIT,
        'ACTIVATION': wandb.config.ACTIVATION,
        'SCALE_ACTIVATION': wandb.config.SCALE_ACTIVATION,
        'FITNESS_THRESHOLD': wandb.config.FITNESS_THRESHOLD,
        'USE_FITNESS_COEFFICIENT': wandb.config.USE_FITNESS_COEFFICIENT,
        'INITIAL_FITNESS_COEFFICIENT': wandb.config.INITIAL_FITNESS_COEFFICIENT,
        'FINAL_FITNESS_COEFFICIENT': wandb.config.FINAL_FITNESS_COEFFICIENT,
        'POPULATION_SIZE': wandb.config.POPULATION_SIZE,
        'NUMBER_OF_GENERATIONS': wandb.config.NUMBER_OF_GENERATIONS,
        'SPECIATION_THRESHOLD': wandb.config.SPECIATION_THRESHOLD,
        'CONNECTION_MUTATION_RATE': wandb.config.CONNECTION_MUTATION_RATE,
        'CONNECTION_PERTURBATION_RATE': wandb.config.CONNECTION_PERTURBATION_RATE,
        'ADD_NODE_MUTATION_RATE': wandb.config.ADD_NODE_MUTATION_RATE,
        'ADD_CONNECTION_MUTATION_RATE': wandb.config.ADD_CONNECTION_MUTATION_RATE,
        'CROSSOVER_REENABLE_CONNECTION_GENE_RATE': wandb.config.CROSSOVER_REENABLE_CONNECTION_GENE_RATE,
        'PERCENTAGE_TO_SAVE': wandb.config.PERCENTAGE_TO_SAVE,
        'MAX_EPISODE_STEPS': wandb.config.MAX_EPISODE_STEPS
    }     

    # Print the kwargs
    for key in kwargs:
        print(f"{key}: {kwargs[key]}")

    neat = pop.Population(c.BipedalWalkerConfig(**kwargs))
    solution, generation = neat.run()

    if solution is not None:
        logger.info('Found a Solution')
        draw_net(solution, view=True, filename='./images/bipedal-walker-solution', show_disabled=True)

        # OpenAI Gym
        env = gym.make('BipedalWalker-v3')
        done = False
        observation = env.reset()

        fitness = 0
        phenotype = FeedForwardNet(solution, c.BipedalWalkerConfig)

        while not done:
            env.render()
            input = torch.Tensor([observation]).to(c.BipedalWalkerConfig.DEVICE)

            pred = round(float(phenotype(input)))
            observation, reward, done, info = env.step(pred)
            

            fitness += reward
        env.close()


        # Log generation
    wandb.log({'generation': generation})

    # Clean up memory
    del neat, kwargs

    return solution, generation

if __name__ == '__main__':
    train()

   