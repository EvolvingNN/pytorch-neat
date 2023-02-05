import torch
import torch.nn as nn
import gym
import numpy as np

from neat.phenotype.feed_forward import FeedForwardNet

from neat.utils import random_ensemble_generator_for_static_genome


class AcrobotBalanceConfig:

    def __init__(self, **kwargs):
        
        #self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.DEVICE = torch.device("cpu")

        for k, v in kwargs.items():
            setattr(self, k, v)

        gym.envs.register(
        id='Acrobot-v1',
        entry_point='gym.envs.classic_control:AcrobotEnv',
        max_episode_steps=self.MAX_EPISODE_STEPS
    )
    
    def __call__(self):
        return self

    #Allow episode lengths of > than 200


    def vote(self, voting_ensemble, obs):
        softmax = nn.Softmax(dim=1)
        ensemble_activations = [phenotype(obs) for phenotype in voting_ensemble]
        soft_activations = torch.sum(torch.stack(ensemble_activations, dim = 0), dim = 0)
        vote = np.argmax(softmax(soft_activations).detach().numpy()[0])
        return vote

    def eval_genomes(self, population):

        for genome in population:

            sample_ensembles = random_ensemble_generator_for_static_genome(genome, population, k = self.GENERATIONAL_ENSEMBLE_SIZE, limit = self.CANDIDATE_LIMIT)  # type: ignore

            constituent_ensemble_reward = []

            for sample_ensemble in sample_ensembles:

                voting_ensemble = [FeedForwardNet(genome, self) for genome in sample_ensemble]

                env = gym.make('Acrobot-v1')
                done = False
                observation = env.reset()
                fitness = 0
                while not done:
                    observation = np.array([observation])
                    obs = torch.Tensor(observation).cpu()
                    pred = self.vote(voting_ensemble, obs)
                    observation, reward, done, info = env.step(pred)
                    height = -observation[0] - (observation[0]*observation[2] - observation[1]*observation[3])
                    fitness += height
                
                constituent_ensemble_reward.append(fitness/self.MAX_EPISODE_STEPS)
            
            ACER = np.mean(np.exp(constituent_ensemble_reward))
            self.wandb.log({"ACER": ACER})
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
