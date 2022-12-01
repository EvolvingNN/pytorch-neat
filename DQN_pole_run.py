import logging

import gym
import torch

import neat.population as pop
import neat.experiments.DQN_pole_balancing.config as c
from neat.visualize import draw_net
from neat.phenotype.feed_forward import FeedForwardNet
from duelingDQN.model import QNetwork
import wandb
import numpy as np

run = wandb.init(project="Dueling DQN")
artifact = run.use_artifact('evolvingnn/Dueling DQN/ddqn:latest', type='model')
artifact_dir = artifact.download()

model = QNetwork()
model.load_state_dict(torch.load(f'{artifact_dir}/ddqn-policy.pth'))


logger = logging.getLogger(__name__)

logger.info(c.PoleBalanceConfig.DEVICE)
neat = pop.Population(c.PoleBalanceConfig)
solution, generation = neat.run()

if solution is not None:
    logger.info('Found a Solution')
    draw_net(solution, view=True, filename='./images/pole-balancing-solution', show_disabled=True)

    # OpenAI Gym
    env = gym.make('LongCartPole-v0')
    done = False
    observation = env.reset()

    fitness = 0
    phenotype = FeedForwardNet(solution, c.PoleBalanceConfig)

    while not done:
        env.render()
        observation = np.array([observation])
        observation = model.half_forward(torch.tensor(observation, dtype=torch.float32)).detach()
        # print(observation.shape)
        input = observation.to(c.PoleBalanceConfig.DEVICE)

        pred = round(float(phenotype(input)))
        observation, reward, done, info = env.step(pred)

        fitness += reward
    env.close()
