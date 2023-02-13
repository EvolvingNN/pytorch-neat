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

    #Allow episode lengths of > than 200


    def vote(self, voting_ensemble, obs):
        softmax = nn.Softmax(dim=1)
        ensemble_activations = [phenotype(obs) for phenotype in voting_ensemble]
        soft_activations = torch.sum(torch.stack(ensemble_activations, dim = 0), dim = 0)
        vote = np.argmax(softmax(soft_activations).detach().numpy()[0])
        return vote

    def constituent_ensemble_evaluation(self, genomes):
        
        env = gym.make('Acrobot-v1')
        done = False
        observation = env.reset()

        softmax = nn.Softmax(dim=1)
        voting_ensemble = [FeedForwardNet(genome, self) for genome in genomes]
        max_height = -1
        total_reward = 0

        while not done:
            observation = np.array([observation])
            obs = torch.Tensor(observation).cpu()
            ensemble_activations = [phenotype(obs) for phenotype in voting_ensemble]
            soft_activations = torch.sum(torch.stack(ensemble_activations, dim = 0), dim = 0)
            pred = np.argmax(softmax(soft_activations).detach().numpy()[0])
            observation, reward, done, info = env.step(pred)
            height = -observation[0] - (observation[0]*observation[2] - observation[1]*observation[3])
            if height > max_height:
                max_height = height
            total_reward += height
        
        return max_height, total_reward




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

                # self.wandb.log({f"Genome {id(genome)} Max Height" : genome.max_height,
                #                 f"Genome {id(genome)} Fitness" : genome.fitness,
                #                 f"Genome {id(genome)} Average Reward" : genome.average_reward,
                #                 f"Genome {id(genome)} Step Completed" : genome.step_completed
                #                 }, step = kwargs['generation'])

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
            df_results.to_csv('./df_results.csv')

            self.wandb.save('./df_results.csv')

            self.wandb.log(df_results.max(axis=0).to_dict())


        best_genome = max(population, key=attrgetter('fitness'))
        best_ensemble = max(ensemble_rewards.items(), key=itemgetter(1))[0] if ensemble_rewards else None

        self.wandb.log({"Best Max Height" : best_genome.max_height,
                        "Best Fitness" : best_genome.fitness,
                        "Best Average Reward" : best_genome.average_reward,
                        "Best Step Completed" : best_genome.step_completed
                        }, step = kwargs['generation'])

        return best_genome, best_ensemble


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
