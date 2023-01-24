import contextlib
import logging
import wandb

import neat.population as pop
import neat.experiments.UCI.config as c
from neat.experiments.UCI.kwargs import KWARGS

from neat.visualize import draw_net
from tqdm import tqdm

import uci_dataset as uci
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch.nn.functional import one_hot



logger = logging.getLogger(__name__)



df = uci.load_heart_disease().dropna()

features = df.iloc[:,:-1]
target = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = torch.tensor(scaler.transform(X_train))
X_test = torch.tensor(scaler.transform(X_test))

y_train = torch.squeeze(one_hot(torch.tensor(y_train.to_numpy().reshape(-1,1))))  # type: ignore
y_test = torch.squeeze(one_hot(torch.tensor(y_test.to_numpy().reshape(-1,1)))) # type: ignore

sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'goal': 'maximize', 
        'name': 'diversity'
		},
    'parameters': {
        'VERBOSE': {'values': [True, False]},
        'USE_BIAS': {'values': [False, True]},
        'GENERATIONAL_ENSEMBLE_SIZE': {'values': [2, 5, 7]},
        'CANDIDATE_LIMIT': {'values': [2, 5, 7]},
        'ACTIVATION': {'values': ['sigmoid']},
        'SCALE_ACTIVATION': {'max': 7, 'min': 2},
        'USE_FITNESS_COEFFICIENT': {'values': [False, True]},
        'INITIAL_FITNESS_COEFFICIENT': {'max': 1.0, 'min': 0.0},
        'FINAL_FITNESS_COEFFICIENT': {'max': 1, 'min': 0},
        'POPULATION_SIZE': {'values': [5, 15, 50, 100, 150]},
        'NUMBER_OF_GENERATIONS': {'values': [10, 50, 100]},
        'SPECIATION_THRESHOLD': {'values': [2.0, 3.0, 4.0, 5.0]},
        'CONNECTION_MUTATION_RATE': {'max': 1.0, 'min': 0.5},
        'CONNECTION_PERTURBATION_RATE': {'max': 1.0, 'min': 0.5},
        'ADD_NODE_MUTATION_RATE': {'max': 0.1, 'min': 0.001},
        'ADD_CONNECTION_MUTATION_RATE': {'max': 0.7, 'min': 0.1},
        'CROSSOVER_REENABLE_CONNECTION_GENE_RATE': {'max': 0.7, 'min': 0.1},
        'PERCENTAGE_TO_SAVE': {'max': 1.0, 'min': 0.5}


        # 'NUMBER_OF_GENERATIONS' : {'values': [50, 100, 150, 200, 250]}
     }
}

# sweep_id = wandb.sweep(sweep=sweep_configuration, project="Classification", entity="evolvingnn")

def train():
    wandb.init(config=KWARGS, group="greedy 100 runs", project="Classification", entity="evolvingnn")
    print(f"Type of wandb.config {type(wandb.config)}")
    kwargs = KWARGS

    
    kwargs = {
    "VERBOSE": True,
        "USE_BIAS": False,
        "ACTIVATION": "sigmoid",
        "NUM_INPUTS": 784,
        "NUM_OUTPUTS":  10,
        "CANDIDATE_LIMIT": 7,
        "POPULATION_SIZE": 150,
        "SCALE_ACTIVATION": 2,
        "PERCENTAGE_TO_SAVE":  0.637982135009669,
        "SPECIATION_THRESHOLD": 2,
        "NUMBER_OF_GENERATIONS": 100,
        "ADD_NODE_MUTATION_RATE": 0.03576783437283876,
        "USE_FITNESS_COEFFICIENT": False,
        "CONNECTION_MUTATION_RATE":  0.6493265295602876,
        "FINAL_FITNESS_COEFFICIENT":  0,
        "GENERATIONAL_ENSEMBLE_SIZE": 5,
        "INITIAL_FITNESS_COEFFICIENT": 0.8636788797573082,
        "ADD_CONNECTION_MUTATION_RATE":  0.5897706837815341,
        "CONNECTION_PERTURBATION_RATE": 0.6167161436025228,
        "CROSSOVER_REENABLE_CONNECTION_GENE_RATE": 0.276368120953109,
    }     
    kwargs['DATA'] = X_train
    kwargs['TARGET'] = y_train

    kwargs['NUM_INPUTS'] = kwargs['DATA'].shape[1]
    kwargs['NUM_OUTPUTS'] = kwargs['TARGET'].shape[1]

    kwargs['TEST_DATA'] = X_test
    kwargs['TEST_TARGET'] = y_test
    
    kwargs['wandb'] = wandb

    # Print the kwargs
    for key in kwargs:
        print(f"{key}: {kwargs[key]}")

    neat = pop.Population(c.UCIConfig(**kwargs))
    solution, generation = neat.run()

    # Log generation
    wandb.log({'generation': generation})

    return solution, generation
    

if __name__ == '__main__':
    for _ in range(10):
        train()
        
    # wandb.agent("13wk40yj", function=train)

