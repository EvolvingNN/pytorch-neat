import torch
import torch.nn as nn
import gym
import numpy as np
from operator import attrgetter, itemgetter

from neat.phenotype.feed_forward import FeedForwardNet

import neat.analysis.wrapper as wrapper

from neat.utils import random_ensemble_generator_for_static_genome

class Ensemble:
    def __init__(self, ensemble):
        #assumes ensemble is a list of phenotypes (FeedForwardNet)
        self.ensemble = ensemble
        self.fitness = None
        self.max_height = None
        self.total_height = None
        self.average_reward = None
        self.step_completed = None

    def fitness_fn(self, policy = "default"):

        if policy == "default":
            fitness = self.total_height - self.step_completed # type: ignore     
            return fitness
        else:
            raise Exception("Invalid Policy")     

    
    def vote(self, voting_ensemble, obs):
        softmax = nn.Softmax(dim=1)
        ensemble_activations = [phenotype(obs) for phenotype in voting_ensemble]
        soft_activations = torch.sum(torch.stack(ensemble_activations, dim = 0), dim = 0)
        vote = np.argmax(softmax(soft_activations).detach().numpy()[0])
        return vote

    def eval_fitness(self, config):
        
        env = gym.make('Acrobot-v1')
        done = False
        observation = env.reset(seed = 0)
        total_height = 0
        step = 0
        max_height = -3

        voting_ensemble = [FeedForwardNet(genome, config) for genome in self.ensemble]

        while not done:
            observation = np.array([observation])
            obs = torch.Tensor(observation).cpu()
            pred = self.vote(voting_ensemble, obs)
            observation, reward, done, info = env.step(pred)
            height = -observation[0] - (observation[0]*observation[2] - observation[1]*observation[3])
            if height > max_height:
                max_height = height
            total_height += height
            step += 1
        
        self.total_height = total_height
        self.max_height = max_height
        self.average_reward = total_height/step
        self.step_completed = step

        self.fitness = self.fitness_fn()
        
        return self.fitness
        


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

    def __str__(self):
        return str(dir(self))


    def vote(self, voting_ensemble, obs):
        softmax = nn.Softmax(dim=1)
        ensemble_activations = [phenotype(obs) for phenotype in voting_ensemble]
        soft_activations = torch.sum(torch.stack(ensemble_activations, dim = 0), dim = 0)
        vote = np.argmax(softmax(soft_activations).detach().numpy()[0])
        return vote
        
    def eval_ensemble(self, ensemble):

        def vote(voting_ensemble, input):
            softmax = nn.Softmax(dim=1)
            ensemble_activations = [phenotype(input) for phenotype in voting_ensemble]
            soft_activations = torch.sum(torch.stack(ensemble_activations, dim = 0), dim = 0)
            vote = np.argmax(softmax(soft_activations).detach().numpy()[0])
            return vote

        voting_ensemble = [FeedForwardNet(genome, self) for genome in ensemble]

        env = gym.make('Acrobot-v1')
        done = False
        observation = env.reset()
        fitness = 0
        while not done:
            observation = np.array([observation])
            input = torch.Tensor(observation).to(self.DEVICE)
            pred = vote(voting_ensemble, input)
            observation, reward, done, info = env.step(pred)
            #height = -observation[0] - (observation[0]observation[2] - observation[1]observation[3])
            fitness += reward

        return fitness

    def constituent_ensemble_evaluation(self, genomes):

        env = gym.make('Acrobot-v1')
        done = False
        observation = env.reset(seed = 0)

        softmax = nn.Softmax(dim=1)
        voting_ensemble = [FeedForwardNet(genome, self) for genome in genomes]

        step = 0
        total_height = 0

        while not done:
            observation = np.array([observation])
            obs = torch.Tensor(observation).cpu()
            ensemble_activations = [phenotype(obs) for phenotype in voting_ensemble]
            soft_activations = torch.sum(torch.stack(ensemble_activations, dim = 0), dim = 0)
            pred = np.argmax(softmax(soft_activations).detach().numpy()[0])
            observation, reward, done, info = env.step(pred)
            height = -observation[0] - (observation[0]*observation[2] - observation[1]*observation[3])
            total_height += height
            step += 1
        reward = total_height - step

        return reward


    def eval_genomes(self, population, **kwargs):
        
        ensemble_rewards = {}

        for genome in population:

            # POLICY: CONTROL | evaluate each genome individually
            if self.USE_CONTROL: 
                phenotype = FeedForwardNet(genome, self)
                max_height = -3

                env = gym.make('Acrobot-v1')
                done = False
                observation = env.reset(seed = 0)
                total_height = 0
                step = 0
                while not done:
                    obs = torch.Tensor([observation]).cpu()
                    pred = np.argmax(phenotype(obs).detach().numpy()[0])
                    observation, reward, done, info = env.step(pred)
                    height = -observation[0] - (observation[0]*observation[2] - observation[1]*observation[3])
                    if height > max_height:
                        max_height = height
                    total_height += height
                    step += 1

                genome.max_height = max_height
                genome.total_height = total_height
                genome.fitness = total_height - step
                genome.average_reward = total_height/step
                genome.step_completed = step

            #POLICY: ACER | use the average reward from a sample of constituent ensembles to score the genome.
            if self.USE_ACER:

                sample_ensembles = random_ensemble_generator_for_static_genome(genome, population, k = self.GENERATIONAL_ENSEMBLE_SIZE, limit = self.CANDIDATE_LIMIT)  # type: ignore

                constituent_ensemble_reward = []

                for sample_ensemble in sample_ensembles:

                    ensemble = Ensemble(sample_ensemble)
                    reward = ensemble.eval_fitness(self)

                    constituent_ensemble_reward.append(reward)

                    ensemble_rewards[ensemble] = reward
                    
                ACER = np.mean(constituent_ensemble_reward)
                genome.fitness = ACER

            #POLICY: ACER WITH WARMUP | use a weighted fitness function which intially favors genome fitness, then incremenetally favors ACER
            if self.USE_ACER_WITH_WARMUP:
                ACER_coefficient = kwargs['generation']/self.NUMBER_OF_GENERATIONS
                genome_coefficient = 1 - ACER_coefficient

                genome.fitness = genome_coefficient * (genome.total_height - genome.step_completed) + ACER_coefficient *  genome.fitness
                
        if kwargs['generation'] == self.NUMBER_OF_GENERATIONS:
        
            df_results = wrapper.run_trial_analysis(population, self.constituent_ensemble_evaluation)
            print(df_results.max(axis=0).to_dict())

        best_genome = max(population, key=attrgetter('fitness'))
        best_ensemble = max(ensemble_rewards.items(), key=itemgetter(1))[0] if ensemble_rewards else None

        df_results = wrapper.run_trial_analysis(population, self.eval_ensemble)

        # self.wandb.log({"Best Max Height" : best_genome.max_height,
        #                 "Best Fitness" : best_genome.fitness,
        #                 "Best Average Reward" : best_genome.average_reward,
        #                 "Best Step Completed" : best_genome.step_completed
        #                 })

        return best_genome, best_ensemble
        
