import torch
import gym
import numpy as np

from neat.phenotype.feed_forward import FeedForwardNet
from neat.visualize import draw_net
from duelingDQN.model import QNetwork
import wandb





class PoleBalanceConfig:


    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VERBOSE = True

    NUM_INPUTS = 16
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

    def __init__(self) -> None:
        self.run = wandb.init(project="Dueling DQN")
        self.artifact = self.run.use_artifact('evolvingnn/Dueling DQN/ddqn:latest', type='model')
        self.artifact_dir = self.artifact.download()

        self.model = QNetwork()
        self.model.load_state_dict(torch.load(f'{self.artifact_dir}/ddqn-policy.pth'))



    # Allow episode lengths of > than 200
    gym.envs.register(
        id='LongCartPole-v0',
        entry_point='gym.envs.classic_control:CartPoleEnv',
        max_episode_steps=100000
    )

    def fitness_fn(self, genome):
        # OpenAI Gym
        env = gym.make('LongCartPole-v0')
        done = False
        observation = env.reset()
        # observation = self.model.half_forward(torch.tensor(observation, dtype=torch.float32)).detach().numpy()
        # print(observation.shape)
        #print(f"OBS | {observation}")
        fitness = 0
        phenotype = FeedForwardNet(genome, self)

        while not done:
            # observation = dtorch.tensor(observation, dtype=torch.float32)).detach().numpy()
            observation = np.array([observation])
            #run obs through pretrained layer, Gabe
            
            input = self.model.half_forward(torch.Tensor(observation).to(self.DEVICE)).detach()
            # input = input).detach()

            pred = round(float(phenotype(input)))
            observation, reward, done, info = env.step(pred)


            fitness += reward
        env.close()

        return fitness
