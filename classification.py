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



def init_sweep():
    sweep_configuration = {
        'method': 'random',
        'name': 'sweep',
        'metric': {
            'goal': 'maximize', 
            'name': 'diversity'
            },
        'parameters': {
            'USE_BIAS': {'values': [False, True]},
            'GENERATIONAL_ENSEMBLE_SIZE': {'max': 21, 'min':2},
            'CANDIDATE_LIMIT': {'max': 50, 'min': 1},
            'SCALE_ACTIVATION': {'max': 7.0, 'min': 1.0},
            'USE_FITNESS_COEFFICIENT': {'values': [False, True]},
            'GENOME_FITNESS_METRIC': {'values' : ['CE LOSS', 'ACCURACY']},
            'ENSEMBLE_FITNESS_METRIC': {'values' : ['CE LOSS', 'ACCURACY']},
            'SPECIATION_THRESHOLD': {'max': 7.0, 'min' : 1.0},
            'CONNECTION_MUTATION_RATE': {'max': 1.0, 'min': 0.5},
            'CONNECTION_PERTURBATION_RATE': {'max': 1.0, 'min': 0.5},
            'ADD_NODE_MUTATION_RATE': {'max': 0.1, 'min': 0.001},
            'ADD_CONNECTION_MUTATION_RATE': {'max': 1.0, 'min': 0.1},
            'CROSSOVER_REENABLE_CONNECTION_GENE_RATE': {'max': 1.0, 'min': 0.1},
            'PERCENTAGE_TO_SAVE': {'max': 1.0, 'min': 0.5}
        }
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="Classification-2", entity="evolvingnn")
    print(sweep_id)
    return sweep_id


def control():

    wandb.init(config=KWARGS, group="control", job_type = 'random trial 1')
    
    kwargs = {
        'VERBOSE': wandb.config.VERBOSE,
        'NUM_INPUTS': wandb.config.NUM_INPUTS,
        'NUM_OUTPUTS': wandb.config.NUM_OUTPUTS,
        'USE_BIAS': wandb.config.USE_BIAS,
        'GENERATIONAL_ENSEMBLE_SIZE': wandb.config.GENERATIONAL_ENSEMBLE_SIZE,
        'CANDIDATE_LIMIT': wandb.config.CANDIDATE_LIMIT,
        'ACTIVATION': wandb.config.ACTIVATION,
        'SCALE_ACTIVATION': wandb.config.SCALE_ACTIVATION,
        'FITNESS_THRESHOLD': wandb.config.FITNESS_THRESHOLD,
        'USE_FITNESS_COEFFICIENT': wandb.config.USE_FITNESS_COEFFICIENT,
        'INITIAL_FITNESS_COEFFICIENT': wandb.config.INITIAL_FITNESS_COEFFICIENT,
        'FINAL_FITNESS_COEFFICIENT': wandb.config.FINAL_FITNESS_COEFFICIENT,
        'USE_GENOME_FITNESS': wandb.config.USE_GENOME_FITNESS,
        'GENOME_FITNESS_METRIC': wandb.config.GENOME_FITNESS_METRIC,
        'ENSEMBLE_FITNESS_METRIC': wandb.config.ENSEMBLE_FITNESS_METRIC,
        'POPULATION_SIZE': wandb.config.POPULATION_SIZE,
        'NUMBER_OF_GENERATIONS': wandb.config.NUMBER_OF_GENERATIONS,
        'SPECIATION_THRESHOLD': wandb.config.SPECIATION_THRESHOLD,
        'CONNECTION_MUTATION_RATE': wandb.config.CONNECTION_MUTATION_RATE,
        'CONNECTION_PERTURBATION_RATE': wandb.config.CONNECTION_PERTURBATION_RATE,
        'ADD_NODE_MUTATION_RATE': wandb.config.ADD_NODE_MUTATION_RATE,
        'ADD_CONNECTION_MUTATION_RATE': wandb.config.ADD_CONNECTION_MUTATION_RATE,
        'CROSSOVER_REENABLE_CONNECTION_GENE_RATE': wandb.config.CROSSOVER_REENABLE_CONNECTION_GENE_RATE,
        'PERCENTAGE_TO_SAVE': wandb.config.PERCENTAGE_TO_SAVE,
    }     

    kwargs['DATA'] = X_train
    kwargs['TARGET'] = y_train

    kwargs['NUM_INPUTS'] = kwargs['DATA'].shape[1]
    kwargs['NUM_OUTPUTS'] = kwargs['TARGET'].shape[1]

    kwargs['TEST_DATA'] = X_test
    kwargs['TEST_TARGET'] = y_test
    
    kwargs['wandb'] = wandb

    kwargs['USE_FITNESS_COEFFICIENT'] = False
    kwargs['USE_GENOME_FITNESS'] = True

    # Print the kwargs
    for key in kwargs:
        print(f"{key}: {kwargs[key]}")

    neat = pop.Population(c.UCIConfig(**kwargs))
    solution, generation = neat.run()

    # Log generation
    wandb.log({'generation': generation})

    # Clean up memory
    del neat, kwargs

    return solution, generation    

def train():
    wandb.init(config=KWARGS)
    
    kwargs = {
        'VERBOSE': wandb.config.VERBOSE,
        'NUM_INPUTS': wandb.config.NUM_INPUTS,
        'NUM_OUTPUTS': wandb.config.NUM_OUTPUTS,
        'USE_BIAS': wandb.config.USE_BIAS,
        'GENERATIONAL_ENSEMBLE_SIZE': wandb.config.GENERATIONAL_ENSEMBLE_SIZE,
        'CANDIDATE_LIMIT': wandb.config.CANDIDATE_LIMIT,
        'ACTIVATION': wandb.config.ACTIVATION,
        'SCALE_ACTIVATION': wandb.config.SCALE_ACTIVATION,
        'FITNESS_THRESHOLD': wandb.config.FITNESS_THRESHOLD,
        'USE_FITNESS_COEFFICIENT': wandb.config.USE_FITNESS_COEFFICIENT,
        'INITIAL_FITNESS_COEFFICIENT': wandb.config.INITIAL_FITNESS_COEFFICIENT,
        'FINAL_FITNESS_COEFFICIENT': wandb.config.FINAL_FITNESS_COEFFICIENT,
        'USE_GENOME_FITNESS': wandb.config.USE_GENOME_FITNESS,
        'GENOME_FITNESS_METRIC': wandb.config.GENOME_FITNESS_METRIC,
        'ENSEMBLE_FITNESS_METRIC': wandb.config.ENSEMBLE_FITNESS_METRIC,
        'POPULATION_SIZE': wandb.config.POPULATION_SIZE,
        'NUMBER_OF_GENERATIONS': wandb.config.NUMBER_OF_GENERATIONS,
        'SPECIATION_THRESHOLD': wandb.config.SPECIATION_THRESHOLD,
        'CONNECTION_MUTATION_RATE': wandb.config.CONNECTION_MUTATION_RATE,
        'CONNECTION_PERTURBATION_RATE': wandb.config.CONNECTION_PERTURBATION_RATE,
        'ADD_NODE_MUTATION_RATE': wandb.config.ADD_NODE_MUTATION_RATE,
        'ADD_CONNECTION_MUTATION_RATE': wandb.config.ADD_CONNECTION_MUTATION_RATE,
        'CROSSOVER_REENABLE_CONNECTION_GENE_RATE': wandb.config.CROSSOVER_REENABLE_CONNECTION_GENE_RATE,
        'PERCENTAGE_TO_SAVE': wandb.config.PERCENTAGE_TO_SAVE,
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

    # Clean up memory
    del neat, kwargs

    return solution, generation
    

if __name__ == '__main__':

    #control()
    #init_sweep()
        
    wandb.agent("krlskw44", function=control)

