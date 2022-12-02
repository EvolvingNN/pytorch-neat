import torch
import torch.nn as nn
from torch import autograd
from neat.phenotype.feed_forward import FeedForwardNet
#from torchvision import datasets
from tqdm import tqdm

from neat.utils import create_prediction_map, random_ensemble_generator_for_static_genome

import numpy as np


class UCIConfig:
    

    def __init__(self, **kwargs):

        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for k, v in kwargs.items(): 
            setattr(self, k, v)

        increment = (self.FINAL_FITNESS_COEFFICIENT - self.INITIAL_FITNESS_COEFFICIENT)/self.NUMBER_OF_GENERATIONS  # type: ignore

        ensemble_coefficients = np.arange(self.INITIAL_FITNESS_COEFFICIENT, self.FINAL_FITNESS_COEFFICIENT, increment)  # type: ignore
        genome_coefficients = ensemble_coefficients[::-1]
        self.genome_coefficients = iter(genome_coefficients)
        self.ensemble_coefficients = iter(ensemble_coefficients)

    def __call__(self):
        return self


    def eval_genomes(self, genomes):

        dataset = self.DATA #type: ignore
        y = [np.squeeze(np.array(y_)) for y_ in self.TARGET] #type: ignore
        self.y = y

        #GET RID OF THIS | REPLACE WITH ALG SELECTED BY KWARG
        @staticmethod
        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            return np.exp(x)/np.sum(np.exp(x),axis=0)

        @staticmethod
        def cross_entropy(y,y_pred):
            loss=-np.sum(y*np.log(y_pred))
            return loss/float(y_pred.shape[0])

        def create_activation_map(self, population):  #for expieriment wrapper eval
            return create_prediction_map(population, self.data, self)
        
        def ensemble_activations_evaluator(self, ensemble_activations): #for experiment wrapper eval
            average_ensemble_activations = np.mean(ensemble_activations, axis = 0)
            ensemble_predictions = np.array([softmax(z) for z in average_ensemble_activations])
            constituent_ensemble_loss = cross_entropy(self.y,ensemble_predictions)
            ensemble_fitness = np.exp(constituent_ensemble_loss)
            return ensemble_fitness

        activations_map = create_prediction_map(genomes, dataset, self)
        genome_fitness_coefficient = next(self.genome_coefficients)
        ensemble_fitness_coefficient = next(self.ensemble_coefficients)

        print(f"fitness = {genome_fitness_coefficient} * genome_fitness + {ensemble_fitness_coefficient} * constituent_ensemble_fitness")

        for genome in tqdm(genomes):

            genome_prediction = np.array([softmax(z) for z in np.squeeze(activations_map[genome])])
            genome_loss = cross_entropy(y, genome_prediction)
            genome_fitness = np.exp(-1 * genome_loss)

            print(f"genome_loss: {genome_loss} | genome_fitness: {genome_fitness}")

            constituent_ensemble_losses = []
            #Iterate through a sample of all possible combinations of candidate genomes to ensemble for a given size k
            sample_ensembles = random_ensemble_generator_for_static_genome(genome, genomes, k = self.GENERATIONAL_ENSEMBLE_SIZE, limit = self.CANDIDATE_LIMIT)  # type: ignore

            for sample_ensemble in tqdm(sample_ensembles):

                ensemble_activations = [np.squeeze(activations_map[genome])]

                #Append candidate genome activations to list
                for candidate in sample_ensemble:
                    ensemble_activations.append(np.squeeze(activations_map[candidate]))
                
            
                average_ensemble_activations = np.mean(ensemble_activations, axis = 0)
              
                ensemble_predictions = np.array([softmax(z) for z in average_ensemble_activations]) #TODO Replace with function specified by config kwarg

                constituent_ensemble_loss = cross_entropy(y,ensemble_predictions)

                constituent_ensemble_losses.append(constituent_ensemble_loss)
            #set the genome fitness as the average loss of the candidate ensembles TODO use kwarg switching for fitness_fn
            
            ensemble_fitness = np.exp(-1 * np.mean(constituent_ensemble_losses))

            print(f"ensemble_loss: {np.mean(constituent_ensemble_losses)} | ensemble_fitness: {ensemble_fitness}")
            
            genome.fitness = genome_fitness_coefficient * genome_fitness + ensemble_fitness_coefficient * ensemble_fitness
            
            print(f"{id(genome)} : {genome.fitness}")
        
        population_fitness = np.mean([genome.fitness for genome in genomes])
        print("population_fitness: ", population_fitness)
        #return population_fitness