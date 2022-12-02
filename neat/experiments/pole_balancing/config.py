import torch
import gym
import numpy as np

from neat.phenotype.feed_forward import FeedForwardNet


class PoleBalanceConfig:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VERBOSE = True

    NUM_INPUTS = 4
    NUM_OUTPUTS = 1
    USE_BIAS = True

    ACTIVATION = 'sigmoid'
    SCALE_ACTIVATION = 4.9

    FITNESS_THRESHOLD = 100000.0

    POPULATION_SIZE = 150
    NUMBER_OF_GENERATIONS = 150
    SPECIATION_THRESHOLD = 3.0

    CONNECTION_MUTATION_RATE = 0.80
    CONNECTION_PERTURBATION_RATE = 0.90
    ADD_NODE_MUTATION_RATE = 0.03
    ADD_CONNECTION_MUTATION_RATE = 0.5

    CROSSOVER_REENABLE_CONNECTION_GENE_RATE = 0.25

    # Top percentage of species to be saved before mating
    PERCENTAGE_TO_SAVE = 0.80

    # Allow episode lengths of > than 200
    gym.envs.register(
        id='LongCartPole-v0',
        entry_point='gym.envs.classic_control:CartPoleEnv',
        max_episode_steps=100000
    )

    def __init__(self, wandb_run):
        self.wandb_run = wandb_run
        # Log the config
        wandb_run.config.update({
            "DEVICE": self.DEVICE,
            "VERBOSE": self.VERBOSE,
            "NUM_INPUTS": self.NUM_INPUTS,
            "NUM_OUTPUTS": self.NUM_OUTPUTS,
            "USE_BIAS": self.USE_BIAS,
            "ACTIVATION": self.ACTIVATION,
            "SCALE_ACTIVATION": self.SCALE_ACTIVATION,
            "FITNESS_THRESHOLD": self.FITNESS_THRESHOLD,
            "POPULATION_SIZE": self.POPULATION_SIZE,
            "NUMBER_OF_GENERATIONS": self.NUMBER_OF_GENERATIONS,
            "SPECIATION_THRESHOLD": self.SPECIATION_THRESHOLD,
            "CONNECTION_MUTATION_RATE": self.CONNECTION_MUTATION_RATE,
            "CONNECTION_PERTURBATION_RATE": self.CONNECTION_PERTURBATION_RATE,
            "ADD_NODE_MUTATION_RATE": self.ADD_NODE_MUTATION_RATE,
            "ADD_CONNECTION_MUTATION_RATE": self.ADD_CONNECTION_MUTATION_RATE,
            "CROSSOVER_REENABLE_CONNECTION_GENE_RATE": self.CROSSOVER_REENABLE_CONNECTION_GENE_RATE,
            "PERCENTAGE_TO_SAVE": self.PERCENTAGE_TO_SAVE
        })


    def fitness_fn(self, genome):
        # OpenAI Gym
        env = gym.make('LongCartPole-v0')
        done = False
        observation = env.reset()

        fitness = 0
        phenotype = FeedForwardNet(genome, self)

        while not done:
            observation = np.array([observation])
            input = torch.Tensor(observation).to(self.DEVICE)

            pred = round(float(phenotype(input)))
            observation, reward, done, info = env.step(pred)

            fitness += reward
        env.close()

        return fitness
