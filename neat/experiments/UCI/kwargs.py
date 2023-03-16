KWARGS = { 

        'VERBOSE' : False,

        'DATA' : None,  #set in classification.py
        'TARGET' : None,

        'NUM_INPUTS' : 784,
        'NUM_OUTPUTS' : 10,
        'USE_BIAS' : True, 

        'GENERATIONAL_ENSEMBLE_FRACTION' : 1,
        'CANDIDATE_LIMIT' : 0.75,

        'ACTIVATION' : 'sigmoid',
        'SCALE_ACTIVATION' : 4.9,

        'FITNESS_THRESHOLD' : 0.7,
                           
        'USE_FITNESS_COEFFICIENT' : False,
        'INITIAL_FITNESS_COEFFICIENT' : 0.1,
        'FINAL_FITNESS_COEFFICIENT' : 0.9,

        'USE_GENOME_FITNESS' : False,
        'GENOME_FITNESS_METRIC' : 'ACCURACY', #'CE LOSS',
        'ENSEMBLE_FITNESS_METRIC' : 'CE LOSS', #CE LOSS

        'POPULATION_SIZE' : 5,
        'NUMBER_OF_GENERATIONS' : 75,
        'SPECIATION_THRESHOLD' : 5.0,

        'CONNECTION_MUTATION_RATE' : 0.80,
        'CONNECTION_PERTURBATION_RATE' : 0.80,
        'ADD_NODE_MUTATION_RATE' : 0.10,
        'ADD_CONNECTION_MUTATION_RATE' : 0.5,

        'CROSSOVER_REENABLE_CONNECTION_GENE_RATE' : 0.50,

        'PERCENTAGE_TO_SAVE' : 0.80     

}     
