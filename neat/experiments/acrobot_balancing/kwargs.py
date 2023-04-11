KWARGS = { 

        'VERBOSE' : True,

        'NUM_INPUTS' : 6,
        'NUM_OUTPUTS' : 3,
        'USE_BIAS' : False, 

        'GENERATIONAL_ENSEMBLE_SIZE' : 3,
        'CANDIDATE_LIMIT' : 10,

        'ACTIVATION' : 'sigmoid',
        'SCALE_ACTIVATION' : 4.9,

        'MAX_EPISODE_STEPS' : 500,
        'FITNESS_THRESHOLD' : float("inf"),
        'TOP_HEIGHT' : -float("inf"),

        'USE_CONTROL' : True,
        'USE_ACER' : True,
        'USE_ACER_WITH_WARMUP' : True,

        'POPULATION_SIZE' : 25,
        'NUMBER_OF_GENERATIONS' : 200,
        'SPECIATION_THRESHOLD' : 3.0,

        'CONNECTION_MUTATION_RATE' : 0.80,
        'CONNECTION_PERTURBATION_RATE' : 0.90,
        'ADD_NODE_MUTATION_RATE' : 0.03,
        'ADD_CONNECTION_MUTATION_RATE' : 0.5,

        'CROSSOVER_REENABLE_CONNECTION_GENE_RATE' : 0.25,

        'PERCENTAGE_TO_SAVE' : 0.10,
        'MAX_POPULATION_SIZE' : 100,
        'CHECKPOINTS': [5,25,50,100,150,200]

}     