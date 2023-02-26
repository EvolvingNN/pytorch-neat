import torch
import torch.nn as nn
import gym
import numpy as np

from neat.phenotype.feed_forward import FeedForwardNet

from neat.utils import random_ensemble_generator_for_static_genome


class BipedalWalkerConfig:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VERBOSE = True

    NUM_INPUTS = 24
    NUM_OUTPUTS = 4 #corresponding to action space 0: apply -1 torque, 1: apply 0 torque, 2: apply 1 torque
    USE_BIAS = False

    ACTIVATION = 'sigmoid'
    SCALE_ACTIVATION = 4.9

    FITNESS_THRESHOLD = float("inf")
    MAX_EPISODE_STEPS = 500

    GENERATIONAL_ENSEMBLE_SIZE = 2
    CANDIDATE_LIMIT = 6

    POPULATION_SIZE = 5
    NUMBER_OF_GENERATIONS = 100
    SPECIATION_THRESHOLD = 3.0

    CONNECTION_MUTATION_RATE = 0.80
    CONNECTION_PERTURBATION_RATE = 0.90
    ADD_NODE_MUTATION_RATE = 0.03
    ADD_CONNECTION_MUTATION_RATE = 0.5

    CROSSOVER_REENABLE_CONNECTION_GENE_RATE = 0.25

    # Top percentage of species to be saved before mating
    PERCENTAGE_TO_SAVE = 0.80

    TOP_HEIGHT = -np.inf

    #Allow episode lengths of > than 200
    gym.envs.register(
        id='BipedalWalker-v3',
        entry_point='gym.envs.box2d:BipedalWalker',
        max_episode_steps=MAX_EPISODE_STEPS
    )

    def vote(self, voting_ensemble, input):
        softmax = nn.Softmax(dim=1)
        ensemble_activations = [phenotype(input) for phenotype in voting_ensemble]
        soft_activations = torch.sum(torch.stack(ensemble_activations, dim = 0), dim = 0)
        vote = np.argmax(softmax(soft_activations).detach().numpy()[0])
        return vote

    def eval_genomes(self, population):

        for genome in population:

            sample_ensembles = random_ensemble_generator_for_static_genome(genome, population, k = self.GENERATIONAL_ENSEMBLE_SIZE, limit = self.CANDIDATE_LIMIT)  # type: ignore

            constituent_ensemble_reward = []

            for sample_ensemble in sample_ensembles:

                voting_ensemble = [FeedForwardNet(genome, self) for genome in sample_ensemble]

                env = gym.make('BipedalWalker-v3')
                done = False
                observation = env.reset()
                fitness = 0
                while not done:
                    observation = np.array([observation])
                    input = torch.Tensor(observation).to(self.DEVICE)
                    pred = self.vote(voting_ensemble, input)
                    print(pred)
                    observation, reward, done, info = env.step(pred)
                    height = -observation[0] - (observation[0]*observation[2] - observation[1]*observation[3])
                    fitness += height
                
                constituent_ensemble_reward.append(fitness/self.MAX_EPISODE_STEPS)
            
            ACER = np.mean(np.exp(constituent_ensemble_reward))
            genome.fitness = ACER
        
        population_fitness = np.mean([genome.fitness for genome in population])
        return population_fitness


    #Fitness threshold increases as generations persist. Used for Acrobot
    # def alt_fitness_fn(self, genome):
    #     # OpenAI Gym
    #     env = gym.make('Acrobot-v1')
    #     done = False
    #     observation = env.reset()

    #     fitness = 0
    #     phenotype = FeedForwardNet(genome, self)
    #     counter = 0
    #     while not done:
    #         observation = np.array([observation])
    #         input = torch.Tensor(observation).to(self.DEVICE)
    #         pred = round(float(phenotype(input)))
    #         observation, reward, done, info = env.step(pred)
    #         #height = -np.cos(observation[0]) - np.cos(observation[1] + observation[0])
    #         #\cos(x+y)=\cos x\cos y\ +\sin x\sin y
    #         height = -observation[0] - (observation[0]*observation[2] - observation[1]*observation[3])
    #         fitness += height
    #     fitness = fitness/200
    #     print("fitness: ", fitness)
    #     env.close()

    #     return fitness
