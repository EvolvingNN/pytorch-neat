import torch
import torch.nn as nn
from torch import autograd
from neat.phenotype.feed_forward import FeedForwardNet
#from torchvision import datasets
from tqdm import tqdm
import pickle

from neat.utils import create_prediction_map, random_ensemble_generator_for_static_genome, speciate
import neat.analysis.wrapper as wrapper

import numpy as np
import pandas as pd

# import wandb

class UCIConfig:
    

    def __init__(self, **kwargs):

        # Set the device to use CUDA if available, otherwise use CPU
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Assign keyword arguments as attributes of the object
        for k, v in kwargs.items():
            setattr(self, k, v)

        # If the fitness coefficient is set to be used
        if self.USE_FITNESS_COEFFICIENT: #type: ignore

            # Calculate the increment for the coefficients
            increment = (self.FINAL_FITNESS_COEFFICIENT - self.INITIAL_FITNESS_COEFFICIENT)/self.NUMBER_OF_GENERATIONS  # type: ignore

            # Create arrays of coefficients for the ensemble and genome
            ensemble_coefficients = np.arange(self.INITIAL_FITNESS_COEFFICIENT, self.FINAL_FITNESS_COEFFICIENT, increment)  # type: ignore
            genome_coefficients = ensemble_coefficients[::-1]
            
        else:
            # NO WARM UP - GENOME FITNESS
            if self.USE_GENOME_FITNESS: #type: ignore
                # Create arrays of ones and zeroes for the genome and ensemble coefficients
                genome_coefficients = np.ones(self.NUMBER_OF_GENERATIONS) #type: ignore
                ensemble_coefficients = np.zeros(self.NUMBER_OF_GENERATIONS) #type: ignore
            else:
            # NO WARM UP
            # Assign the coefficient arrays as iterable attributes of the object
                genome_coefficients = np.zeros(self.NUMBER_OF_GENERATIONS) #type: ignore
                ensemble_coefficients = np.ones(self.NUMBER_OF_GENERATIONS) #type: ignore
                    # Assign the coefficient arrays as iterable attributes of the object
        self.genome_coefficients = iter(genome_coefficients)
        self.ensemble_coefficients = iter(ensemble_coefficients)
    def __call__(self):
        return self

    def create_activation_map(self, genomes, X):
        genomes_to_results = {}
        for genome in tqdm(genomes):
            results = []
            phenotype = FeedForwardNet(genome, self)
            phenotype.to(self.DEVICE)
            for input in X:
                #Adds batch dimension
                input = torch.unsqueeze(input, 0)
                input.to(self.DEVICE)
                prediction = phenotype(input).to('cpu')
                results.append(prediction)
            genomes_to_results[genome] = torch.squeeze(torch.stack(results))
        return genomes_to_results
    
    def constituent_ensemble_evaluation(self, ensemble_activations, use_test_target = False):
            
        # Define the softmax function and cross-entropy loss function
        softmax = nn.Softmax(dim=1)
        CE_loss = nn.CrossEntropyLoss()

        # Sum the activations of all ensemble members
        soft_activations = torch.sum(torch.stack(ensemble_activations, dim = 0), dim = 0)
        
        # Calculate the cross-entropy loss for the ensemble
        if use_test_target:
            constituent_ensemble_loss = CE_loss(softmax(soft_activations), self.TEST_TARGET.to(torch.float32)).item()
            predicted_classes = torch.argmax(softmax(soft_activations), dim=1)
            actual_classes = torch.argmax(self.TEST_TARGET, dim = 1)
            correct_predictions = predicted_classes == actual_classes

            ensemble_fitness = np.exp(-1 * constituent_ensemble_loss)
            ensemble_accuracy = np.mean(correct_predictions.numpy())

            self.wandb.log({"test/test_constituent_ensemble_loss": constituent_ensemble_loss})
            self.wandb.log({"test/test_constituent_ensemble_accuracy": ensemble_accuracy})

        else: #use TARGET
            constituent_ensemble_loss = CE_loss(softmax(soft_activations), self.TARGET.to(torch.float32)).item()
            predicted_classes = torch.argmax(softmax(soft_activations), dim=1)
            actual_classes = torch.argmax(self.TARGET, dim = 1)
            correct_predictions = predicted_classes == actual_classes

            ensemble_fitness = np.exp(-1 * constituent_ensemble_loss)
            ensemble_accuracy = np.mean(correct_predictions.numpy())
            
            self.wandb.log({"train/train_constituent_ensemble_loss": constituent_ensemble_loss})
            self.wandb.log({"train/train_constituent_ensemble_accuracy": ensemble_accuracy})

        # Calculate the fitness of the ensemble using the negative exponential of the loss

        return ensemble_accuracy


    def eval_genomes(self, genomes, generation):

        # Create an activation map for all the genomes using the provided data
        train_activations_map = self.create_activation_map(genomes, self.DATA) #type: ignore
        test_activations_map = self.create_activation_map(genomes, self.TEST_DATA) #type: ignore

        # Get the next coefficient for the genome and ensemble fitness
        genome_fitness_coefficient = next(self.genome_coefficients)
        ensemble_fitness_coefficient = next(self.ensemble_coefficients)

        # Evaluate the fitness of each genome
        for i, genome in enumerate(genomes):

            genome.generation = generation
            
            # Define the softmax function
            softmax = nn.Softmax(dim=1)
            # Get the prediction for the genome
            genome_prediction = softmax(train_activations_map[genome])
            # Define the cross-entropy loss function
            CE_loss = nn.CrossEntropyLoss()
            # Calculate the loss for the genome
            genome_loss = CE_loss(genome_prediction, self.TARGET.to(torch.float32)).item()
            genome.genome_loss = genome_loss
            self.wandb.log({"train/genome_loss": genome_loss}, commit = False)


            predicted_classes = torch.argmax(genome_prediction, dim = 1)
            actual_classes = torch.argmax(self.TARGET, dim = 1)
            correct_predictions = predicted_classes == actual_classes
            genome_accuracy = np.mean(correct_predictions.numpy())
            genome.genome_accuracy = genome_accuracy

            self.wandb.log({"train/genome_accruacy": genome_accuracy}, commit = False)
            # Calculate the fitness of the genome using the negative exponential of the loss
            if self.GENOME_FITNESS_METRIC == "CE LOSS":
                genome_fitness = np.exp(-1 * genome_loss)
                
            elif self.GENOME_FITNESS_METRIC == "ACCURACY":
                genome_fitness = genome_accuracy
            else:
                genome_fitness = 0

            # List to store the loss of the ensembles
            constituent_ensemble_losses = []
            constituent_ensemble_accuracies = []

            # Generate a sample of all possible combinations of candidate genomes to ensemble for a given size k
            sample_ensembles = random_ensemble_generator_for_static_genome(genome, genomes, k = self.GENERATIONAL_ENSEMBLE_SIZE, limit = self.CANDIDATE_LIMIT)  # type: ignore

            # Evaluate the fitness of each ensemble
            for sample_ensemble in sample_ensembles:

                # Create a list to store the activations of the ensemble members
                ensemble_activations = [train_activations_map[genome]]

                # Append the activations of the candidate genomes to the list
                for candidate in sample_ensemble:
                    ensemble_activations.append(train_activations_map[candidate])
                    
                # Sum the activations of all ensemble members
                soft_activations = torch.sum(torch.stack(ensemble_activations, dim = 0), dim = 0)

                predicted_classes = torch.argmax(softmax(soft_activations), dim=1)
                actual_classes = torch.argmax(self.TARGET, dim = 1)
                correct_predictions = predicted_classes == actual_classes
                constituent_ensemble_accuracies.append(correct_predictions.float().mean())   
                    
                # Calculate the cross-entropy loss for the ensemble
                constituent_ensemble_loss = CE_loss(softmax(soft_activations), self.TARGET.to(torch.float32)).item()

                # Append the loss to the list
                constituent_ensemble_losses.append(constituent_ensemble_loss)

            # Calculate the ensemble fitness as the average loss of the candidate ensembles
            genome.constituent_ensemble_losses = constituent_ensemble_losses

            mean_constituent_ensemble_loss = np.mean(constituent_ensemble_losses)
            genome.mean_constituent_ensemble_loss = mean_constituent_ensemble_loss

            genome.constituent_ensemble_accuracies = [c.item() for c in constituent_ensemble_accuracies]
            mean_constituent_ensemble_accuracy = np.mean(constituent_ensemble_accuracies)
            genome.mean_constituent_ensemble_accuracy = mean_constituent_ensemble_accuracy

            self.wandb.log({"train/mean_constituent_ensemble_loss": mean_constituent_ensemble_loss}, commit = False)
            self.wandb.log({"train/mean_constituent_ensemble_accuracy": mean_constituent_ensemble_accuracy}, commit = False)
            if self.ENSEMBLE_FITNESS_METRIC == "CE LOSS":
                ensemble_fitness = np.exp(-1 * np.mean(constituent_ensemble_losses))
            elif self.ENSEMBLE_FITNESS_METRIC == "ACCURACY":
                ensemble_fitness = np.mean(constituent_ensemble_accuracies)
            else:
                ensemble_fitness = 0
            
            # Set the genome fitness as a combination of the genome fitness coefficient, genome fitness and ensemble fitness coefficient, ensemble fitness
            genome.fitness = genome_fitness_coefficient * genome_fitness + ensemble_fitness_coefficient * ensemble_fitness
            self.wandb.log({"train/genome_fitness": genome.fitness}, commit = False)
            self.wandb.log({"train/step": i + len(genomes) * (generation - 1)})

        # Create a dataframe of the results of the trial analysis
        df_results = wrapper.run_trial_analysis_UCI(train_activations_map, test_activations_map, self.constituent_ensemble_evaluation)
        
        df_results = df_results.reset_index().rename(columns = {"index" : "ensemble_size"})
        df_results['ensemble_size'] += 1
        df_results['generation'] = generation
        df_results['run_id'] = self.run_id

        self.df_results = pd.concat([df_results, self.df_results], ignore_index=True)
        
        self.wandb.log({"df_results" : self.wandb.Table(dataframe = self.df_results)})


        df_results.to_csv('./df_results.csv')
    
        # Save the csv to wandb
        self.wandb.save('./df_results.csv')

        # Pickle and save the gnomes
        with open('genomes.pkl', 'wb') as f:
            pickle.dump(genomes, f)
            
        self.wandb.save('genomes.pkl')  

        
        self.wandb.log({'generation' : generation})
        self.wandb.log(df_results.max(axis=0).to_dict())
        
        # Calculate the average fitness of the population
        population_fitness = np.mean([genome.fitness for genome in genomes])

        columns = ['generation', 'genome_loss', 'genome_accuracy', 'constituent_ensemble_losses', 'mean_constituent_ensemble_loss', 'constituent_ensemble_accuracies', 'mean_constituent_ensemble_accuracy']
        df_genome = pd.DataFrame(columns = columns)
        for column in columns:
            col = [genome.__dict__[column] for genome in genomes]
            df_genome[column] = col

        self.df_genome = pd.concat([self.df_genome, df_genome], ignore_index=True)

        self.wandb.log({"df_genome" : self.wandb.Table(dataframe = self.df_genome)})
        
        return population_fitness
