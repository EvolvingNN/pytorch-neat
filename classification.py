import contextlib
import logging
import wandb

import neat.population as pop
import neat.experiments.UCI.config as c
import neat.experiments.UCI.config_test as c_test
from neat.experiments.UCI.kwargs import KWARGS

from neat.visualize import draw_net
from tqdm import tqdm

import uci_dataset as uci
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch.nn.functional import one_hot



logger = logging.getLogger(__name__)



df = uci.load_heart_disease().dropna()

features = df.iloc[:,:-1]
target = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=888)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = torch.tensor(scaler.transform(X_train))
X_test = torch.tensor(scaler.transform(X_test))

y_train = torch.squeeze(one_hot(torch.tensor(y_train.to_numpy().reshape(-1,1))))  # type: ignore
y_test = torch.squeeze(one_hot(torch.tensor(y_test.to_numpy().reshape(-1,1)))) # type: ignore


def init_sweep(sweep_config = 'control'):
    control_sweep_configuration = {
        'method': 'bayes',
        'name': 'UCI Classification | Control',
        'metric': {
            'goal': 'maximize', 
            'name': 'greedy1'
            },
        'parameters': {
            'USE_BIAS': {'values': [False, True]},
            'GENOME_FITNESS_METRIC': {'values' : ['CE LOSS', 'ACCURACY']},
            'ENSEMBLE_FITNESS_METRIC': {'values' : ['CE LOSS', 'ACCURACY']},
            'SPECIATION_THRESHOLD': {'values' : [1, 3, 5]},
            'POPULATION_SIZE' : {'values' : [5, 25, 100]},
            'NUMBER_OF_GENERATIONS' : {'values' : [50, 200]},
            'CONNECTION_MUTATION_RATE': {'values' : [0.1, 0.5, 0.8]},
            'CONNECTION_PERTURBATION_RATE': {'values' : [0.1, 0.5, 0.8]},
            'ADD_NODE_MUTATION_RATE': {'values' : [0.1, 0.5, 0.8]},
            'ADD_CONNECTION_MUTATION_RATE': {'values' : [0.1, 0.5, 0.8]},
            'CROSSOVER_REENABLE_CONNECTION_GENE_RATE': {'values' : [0.1, 0.5, 0.8]},
            'PERCENTAGE_TO_SAVE': {'values' : [0.1, 0.5, 0.8]}
        }
    }

    ACE_sweep_configuration = {
        'method': 'random',
        'name': 'UCI Classification | ACE',
        'metric': {
            'goal': 'maximize', 
            'name': 'greedy1'
            },
        'parameters': {
            'USE_BIAS': {'values': [False, True]},
            'GENOME_FITNESS_METRIC': {'values' : ['CE LOSS', 'ACCURACY']},
            'ENSEMBLE_FITNESS_METRIC': {'values' : ['CE LOSS', 'ACCURACY']},
            'SPECIATION_THRESHOLD': {'values' : [1, 3, 5]},
            'POPULATION_SIZE' : {'values' : [5, 25, 100]},
            'GENERATIONAL_ENSEMBLE_FRACTION' : {'values' : [0.05, 0.10, 0.25, 0.5, 0.75, 1]},
            'CANDIDATE_LIMIT' : {'values' : [0.05, 0.10, 0.25, 0.5, 0.75, 1]},
            'NUMBER_OF_GENERATIONS' : {'values' : [50, 200]},
            'CONNECTION_MUTATION_RATE': {'values' : [0.1, 0.5, 0.8]},
            'CONNECTION_PERTURBATION_RATE': {'values' : [0.1, 0.5, 0.8]},
            'ADD_NODE_MUTATION_RATE': {'values' : [0.1, 0.5, 0.8]},
            'ADD_CONNECTION_MUTATION_RATE': {'values' : [0.1, 0.5, 0.8]},
            'CROSSOVER_REENABLE_CONNECTION_GENE_RATE': {'values' : [0.1, 0.5, 0.8]},
            'PERCENTAGE_TO_SAVE': {'values' : [0.1, 0.5, 0.8]}
        }
    }

    ACE_warmup_sweep_configuration = {
        'method': 'bayes',
        'name': 'UCI Classification | ACE_warmup',
        'metric': {
            'goal': 'maximize', 
            'name': 'random'
            },
        'parameters': {
            'SPECIATION_THRESHOLD': {'values' : [1, 3, 5]},
            'POPULATION_SIZE' : {'values' : [10, 50, 100]},
            'GENERATIONAL_ENSEMBLE_FRACTION' : {'values' : [0.05, 0.25, 0.5]},
            'CANDIDATE_LIMIT' : {'values' : [0.10, 0.25, 0.5]},
            'CONNECTION_MUTATION_RATE': {'values' : [0.1, 0.5, 0.8]},
            'CONNECTION_PERTURBATION_RATE': {'values' : [0.1, 0.5, 0.8]},
            'ADD_NODE_MUTATION_RATE': {'values' : [0.1, 0.5, 0.8]},
            'ADD_CONNECTION_MUTATION_RATE': {'values' : [0.1, 0.5, 0.8]},
            'CROSSOVER_REENABLE_CONNECTION_GENE_RATE': {'values' : [0.1, 0.5, 0.8]},
            'PERCENTAGE_TO_SAVE': {'values' : [0.1, 0.5, 0.8]}
        }
    }

    sweep = {'control' : control_sweep_configuration,
             'ACE' : ACE_sweep_configuration,
             'ACE_warmup' : ACE_warmup_sweep_configuration}

    #sweep_id = wandb.sweep(sweep=sweep_configuration, project="Classification-2", entity="evolvingnn")
    sweep_id = wandb.sweep(sweep = sweep[sweep_config], entity = "evolvingnn")
    print(sweep_id)
    return sweep_id


def control(name = None):

    KWARGS['NUM_INPUTS'] = X_train.shape[1]
    KWARGS['NUM_OUTPUTS'] = y_train.shape[1]

    KWARGS['USE_FITNESS_COEFFICIENT'] = False
    KWARGS['USE_GENOME_FITNESS'] = True

    run = wandb.init(config=KWARGS, project="Classification-5", name = name)

    wandb.define_metric("generation")

    wandb.define_metric("greedy1", step_metric="generation")
    wandb.define_metric("greedy2", step_metric="generation")
    wandb.define_metric("random", step_metric="generation")

    wandb.define_metric("greedy1_best_ensemble_size", step_metric="generation")
    wandb.define_metric("greedy2_best_ensemble_size", step_metric="generation")
    wandb.define_metric("random_best_ensemble_size", step_metric="generation")

    diversity_threshold_labels = [f"diversity_{t}_threshold" for t in np.arange(1, 6, 1)]
    for l in diversity_threshold_labels:
        wandb.define_metric(l, step_metric = "generation")
        wandb.define_metric((l+"_best_ensemble_size"), step_metric = "generation")

    kwargs = {
        'VERBOSE': wandb.config.VERBOSE,
        'NUM_INPUTS': wandb.config.NUM_INPUTS,
        'NUM_OUTPUTS': wandb.config.NUM_OUTPUTS,
        'USE_BIAS': wandb.config.USE_BIAS,
        'GENERATIONAL_ENSEMBLE_FRACTION': wandb.config.GENERATIONAL_ENSEMBLE_FRACTION,
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
        'MAX_POPULATION_SIZE': wandb.config.MAX_POPULATION_SIZE,
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

    kwargs['TEST_DATA'] = X_test
    kwargs['TEST_TARGET'] = y_test
    
    kwargs['wandb'] = wandb
    kwargs['run_id'] = run.id

    kwargs['CHECKPOINTS'] = [5,25,50,100,150,200]

    kwargs['df_results'] = pd.DataFrame(columns = ['generation', 'ensemble_size', *[f"diversity_{t}_threshold" for t in np.arange(1, 6, 1)], 'greedy1', 'greedy2', 'random'])

    neat = pop.Population(c.UCIConfig(**kwargs))
    solution, generation = neat.run()

    wandb.finish()

    # Clean up memory
    del neat, kwargs

    return solution, generation

def ACE(name = None):

    KWARGS['NUM_INPUTS'] = X_train.shape[1]
    KWARGS['NUM_OUTPUTS'] = y_train.shape[1]

    KWARGS['USE_FITNESS_COEFFICIENT'] = False
    KWARGS['USE_GENOME_FITNESS'] = False

    run = wandb.init(config=KWARGS, project="Classification-ACE", tags = ["ACE", "fixed seed 888"], name = name)

    wandb.define_metric("generation")

    wandb.define_metric("greedy1", step_metric="generation")
    wandb.define_metric("greedy2", step_metric="generation")
    wandb.define_metric("random", step_metric="generation")

    diversity_threshold_labels = [f"diversity_{t}_threshold" for t in np.arange(1, 6, 1)]
    for l in diversity_threshold_labels:
        wandb.define_metric(l, step_metric = "generation")
    
    kwargs = {
        'VERBOSE': wandb.config.VERBOSE,
        'NUM_INPUTS': wandb.config.NUM_INPUTS,
        'NUM_OUTPUTS': wandb.config.NUM_OUTPUTS,
        'USE_BIAS': wandb.config.USE_BIAS,
        'GENERATIONAL_ENSEMBLE_FRACTION': wandb.config.GENERATIONAL_ENSEMBLE_FRACTION,
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
        'MAX_POPULATION_SIZE': wandb.config.MAX_POPULATION_SIZE,
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

    kwargs['TEST_DATA'] = X_test
    kwargs['TEST_TARGET'] = y_test
    
    kwargs['wandb'] = wandb
    kwargs['run_id'] = run.id

    kwargs['CHECKPOINTS'] = [5,25,50,100,150,200]

    kwargs['df_results'] = pd.DataFrame(columns = ['generation', 'ensemble_size', *[f"diversity_{t}_threshold" for t in np.arange(1, 6, 1)], 'greedy1', 'greedy2', 'random'])
    # Print the kwargs
    # for key in kwargs:
    #     print(f"{key}: {kwargs[key]}")

    neat = pop.Population(c.UCIConfig(**kwargs))
    solution, generation = neat.run()

    wandb.finish()
    # Clean up memory
    del neat, kwargs

    return solution, generation    

def ACE_warmup(name = None):

    KWARGS['NUM_INPUTS'] = X_train.shape[1]
    KWARGS['NUM_OUTPUTS'] = y_train.shape[1]

    KWARGS['USE_FITNESS_COEFFICIENT'] = True
    KWARGS['USE_GENOME_FITNESS'] = True

    run = wandb.init(config=KWARGS, project="Classification-Warmup", tags = ["ACE-with-warmup", "fixed seed 888"], name = name)

    wandb.define_metric("generation")

    wandb.define_metric("greedy1", step_metric="generation")
    wandb.define_metric("greedy2", step_metric="generation")
    wandb.define_metric("random", step_metric="generation")

    diversity_threshold_labels = [f"diversity_{t}_threshold" for t in np.arange(1, 6, 1)]
    for l in diversity_threshold_labels:
        wandb.define_metric(l, step_metric = "generation")


    
    kwargs = {
        'VERBOSE': wandb.config.VERBOSE,
        'NUM_INPUTS': wandb.config.NUM_INPUTS,
        'NUM_OUTPUTS': wandb.config.NUM_OUTPUTS,
        'USE_BIAS': wandb.config.USE_BIAS,
        'GENERATIONAL_ENSEMBLE_FRACTION': wandb.config.GENERATIONAL_ENSEMBLE_FRACTION,
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
        'MAX_POPULATION_SIZE': wandb.config.MAX_POPULATION_SIZE,
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

    kwargs['TEST_DATA'] = X_test
    kwargs['TEST_TARGET'] = y_test
    
    kwargs['wandb'] = wandb
    kwargs['run_id'] = run.id

    kwargs['CHECKPOINTS'] = [5,25,50,100,150,200]

    kwargs['df_results'] = pd.DataFrame(columns = ['generation', 'ensemble_size', *[f"diversity_{t}_threshold" for t in np.arange(1, 6, 1)], 'greedy1', 'greedy2', 'random'])
    # Print the kwargs
    # for key in kwargs:
    #     print(f"{key}: {kwargs[key]}")

    neat = pop.Population(c.UCIConfig(**kwargs))
    solution, generation = neat.run()


    # Clean up memory
    del neat, kwargs

    wandb.finish()

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

def test():
    kwargs = KWARGS

    kwargs['DATA'] = X_train
    kwargs['TARGET'] = y_train

    kwargs['NUM_INPUTS'] = kwargs['DATA'].shape[1]
    kwargs['NUM_OUTPUTS'] = kwargs['TARGET'].shape[1]

    kwargs['TEST_DATA'] = X_test
    kwargs['TEST_TARGET'] = y_test

    kwargs['CHECKPOINTS'] = [5,25,50,100,150,200]   

    neat = pop.Population(c_test.UCIConfig_test(**kwargs))
    solution, generation = neat.run()     

if __name__ == '__main__':

    #test()
    #init_sweep(sweep_config='ACE_warmup', project = 'Classification-Warmup')
    #control()
    #ACE()
    #ACE_warmup()
        
    wandb.agent("qbi11z9t", function=ACE_warmup, project="Classification-Warmup", count = 1)

