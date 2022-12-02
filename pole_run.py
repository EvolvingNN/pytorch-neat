import logging

import gym
import torch

import neat.population as pop
import neat.experiments.pole_balancing.config as c
from neat.visualize import draw_net
from neat.phenotype.feed_forward import FeedForwardNet
import wandb
import time

run = wandb.init(project="PytorchNeatPoleBalancing", entity="evolvingnn")

wandb.run.log_code(".")


logger = logging.getLogger(__name__)

logger.info(c.PoleBalanceConfig.DEVICE)
neat = pop.Population(c.PoleBalanceConfig, run)
solution, generation = neat.run()

if solution is not None:
    logger.info('Found a Solution')
    draw_net(solution, view=True, filename='./images/pole-balancing-solution', show_disabled=True)
    # Sleep for a second
    time.sleep(1)
    wandb.log({"example": wandb.Image("./images/pole-balancing-solution.png")})

    # OpenAI Gym
    # env = gym.make('LongCartPole-v0')
    # done = False
    # observation = env.reset()

    # fitness = 0
    # phenotype = FeedForwardNet(solution, c.PoleBalanceConfig)

    # while not done:
    #     env.render()
    #     input = torch.Tensor([observation]).to(c.PoleBalanceConfig.DEVICE)

    #     pred = round(float(phenotype(input)))
    #     observation, reward, done, info = env.step(pred)

    #     fitness += reward
    # env.close()
