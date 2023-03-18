KWARGS = { 

        'VERBOSE' : False,

        'DATA' : None,  #set in classification.py
        'TARGET' : None,

        'NUM_INPUTS' : 784,
        'NUM_OUTPUTS' : 10,
        'USE_BIAS' : True, 

        'GENERATIONAL_ENSEMBLE_FRACTION' : 0.25,
        'CANDIDATE_LIMIT' : 0.25,

        'ACTIVATION' : 'sigmoid',
        'SCALE_ACTIVATION' : 4.9,

        'FITNESS_THRESHOLD' : 0.7,
                           
        'USE_FITNESS_COEFFICIENT' : False,
        'INITIAL_FITNESS_COEFFICIENT' : 0.1,
        'FINAL_FITNESS_COEFFICIENT' : 0.9,

        'USE_GENOME_FITNESS' : False,
        'GENOME_FITNESS_METRIC' : 'ACCURACY', #'CE LOSS',
        'ENSEMBLE_FITNESS_METRIC' : 'ACCURACY', #CE LOSS

        'POPULATION_SIZE' : 25,
        'NUMBER_OF_GENERATIONS' : 50,
        'SPECIATION_THRESHOLD' : 3.0,

        'CONNECTION_MUTATION_RATE' : 0.8,
        'CONNECTION_PERTURBATION_RATE' : 0.5,
        'ADD_NODE_MUTATION_RATE' : 0.5,
        'ADD_CONNECTION_MUTATION_RATE' : 0.1,

        'CROSSOVER_REENABLE_CONNECTION_GENE_RATE' : 0.5,

        'PERCENTAGE_TO_SAVE' : 0.1     

}     
