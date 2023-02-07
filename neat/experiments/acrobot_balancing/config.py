import torch
import torch.nn as nn
import gym
import numpy as np

from neat.phenotype.feed_forward import FeedForwardNet

import neat.analysis.wrapper as wrapper

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

        for genome in population:

            if not self.USE_CONTROL:

                sample_ensembles = random_ensemble_generator_for_static_genome(genome, population, k = self.GENERATIONAL_ENSEMBLE_SIZE, limit = self.CANDIDATE_LIMIT)  # type: ignore

                constituent_ensemble_reward = []

                genome.max_height = -1

                for sample_ensemble in sample_ensembles:

                    voting_ensemble = [FeedForwardNet(genome, self) for genome in sample_ensemble]

                    env = gym.make('Acrobot-v1')
                    env.seed(0) #use same env
                    done = False
                    observation = env.reset()
                    fitness = 0
                    step = 0
                    while not done:
                        observation = np.array([observation])
                        obs = torch.Tensor(observation).cpu()
                        pred = self.vote(voting_ensemble, obs)
                        observation, reward, done, info = env.step(pred)
                        height = -observation[0] - (observation[0]*observation[2] - observation[1]*observation[3])
                        if height > genome.max_height:
                            genome.max_height = height
                        fitness += height
                        step += 1
                    
                    constituent_ensemble_reward.append(fitness * (self.MAX_EPISODE_STEPS/step))
                
                ACER = np.mean(np.exp(constituent_ensemble_reward))
                self.wandb.log({"ACER": ACER,
                                "GENOME MAX HEIGHT": genome.max_height })
                genome.fitness = ACER

            else:

                phenotype = FeedForwardNet(genome, self)
                max_height = -1

                env = gym.make('Acrobot-v1')
                env.seed(0) #use same env
                done = False
                observation = env.reset()
                fitness = 0
                step = 0
                while not done:
                    obs = torch.Tensor([observation]).cpu()
                    pred = np.argmax(phenotype(obs).detach().numpy()[0])
                    observation, reward, done, info = env.step(pred)
                    height = -observation[0] - (observation[0]*observation[2] - observation[1]*observation[3])
                    if height > max_height:
                        max_height = height
                    fitness += height
                    step += 1
                self.wandb.log({"Max Height" : max_height,
                                "Fitness" : fitness,
                                "Average Reward" : fitness/step,
                                "Step Completed" : step
                                })
                genome.fitness = fitness
                print(genome.fitness)

        if kwargs['generation'] == self.NUMBER_OF_GENERATIONS:
            df_results = wrapper.run_trial_analysis(population, self.constituent_ensemble_evaluation)
            df_results.to_csv('./df_results.csv')

            self.wandb.save('./df_results.csv')

            self.wandb.log(df_results.max(axis=0).to_dict())

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
