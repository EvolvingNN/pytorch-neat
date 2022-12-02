import logging

import neat.population as pop
import neat.experiments.UCI.config as c
from neat.experiments.template.default_kwargs import DEFAULT_KWARGS

from neat.visualize import draw_net
from tqdm import tqdm

import uci_dataset as uci

logger = logging.getLogger(__name__)



ds = [uci.load_heart_disease()]

for d in tqdm(ds):
    d = d.dropna()
    kwargs = DEFAULT_KWARGS
    kwargs['DATA'] = d.iloc[:,:-1].values
    kwargs['TARGET'] = d.iloc[:,-1]
    kwargs['NUM_INPUTS'] = kwargs['DATA'].shape[1]
    kwargs['NUM_OUTPUS'] = kwargs['TARGET'].unique().shape[0]

    print(kwargs['TARGET'])
    neat = pop.Population(c.UCIConfig(**kwargs))
    solution, generation = neat.run()

