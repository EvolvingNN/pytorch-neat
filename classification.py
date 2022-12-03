import logging

import neat.population as pop
import neat.experiments.UCI.config as c
from neat.experiments.template.default_kwargs import DEFAULT_KWARGS

from neat.visualize import draw_net
from tqdm import tqdm

import uci_dataset as uci
import numpy as np
import torch
from torch.nn.functional import one_hot

logger = logging.getLogger(__name__)



ds = [uci.load_heart_disease()]

for d in tqdm(ds):
    d = d.dropna()
    data = d.iloc[:,:-1].values
    target = d.iloc[:,-1].values
    kwargs = DEFAULT_KWARGS
    kwargs['DATA'] = torch.tensor(data)
    kwargs['TARGET'] = one_hot(torch.tensor(target))
    kwargs['NUM_INPUTS'] = kwargs['DATA'].shape[1]
    kwargs['NUM_OUTPUTS'] = kwargs['TARGET'].shape[0]

    neat = pop.Population(c.UCIConfig(**kwargs))
    solution, generation = neat.run()

