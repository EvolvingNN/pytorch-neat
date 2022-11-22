import logging

import neat.population as pop
import neat.experiments.MNIST.config as c
from neat.experiments.template.default_kwargs import DEFAULT_KWARGS

from neat.visualize import draw_net
from tqdm import tqdm
from multiprocessing import Pool

logger = logging.getLogger(__name__)

# num_of_solutions = 0

# avg_num_hidden_nodes = 0
# min_hidden_nodes = 0
# max_hidden_nodes = 0
# found_minimal_solution = 0

# avg_num_generations = 0
# min_num_generations = 0

# neat = pop.Population(c.MNISTConfig(**DEFAULT_KWARGS))
# solution, generation = neat.run()
def run(q):
    DEFAULT_KWARGS = q[0]
    DEFAULT_KWARGS['BINARY_CLASS'] = q[1]
    neat = pop.Population(c.MNISTConfig(**DEFAULT_KWARGS))
    solution, generation = neat.run()
    return solution,generation

if __name__ == '__main__':

    for i in tqdm(range(1)):
        queue = [(DEFAULT_KWARGS, digit) for digit in range(10)] 
        with Pool(10) as p:
            p.map(run, queue)
        

#     if solution is not None:
#         #avg_num_generations = ((avg_num_generations * num_of_solutions) + generation) / (num_of_solutions + 1)
#         #min_num_generations = min(generation, min_num_generations)

#         num_hidden_nodes = len([n for n in solution.node_genes if n.type == 'hidden'])
#         avg_num_hidden_nodes = ((avg_num_hidden_nodes * num_of_solutions) + num_hidden_nodes) / (num_of_solutions + 1)
#         min_hidden_nodes = min(num_hidden_nodes, min_hidden_nodes)
#         max_hidden_nodes = max(num_hidden_nodes, max_hidden_nodes)
#         if num_hidden_nodes == 1:
#             found_minimal_solution += 1

#         num_of_solutions += 1
#         #draw_net(solution, view=True, filename='./images/solution-' + str(num_of_solutions), show_disabled=True)

# logger.info('Total Number of Solutions: ', num_of_solutions)
# logger.info('Average Number of Hidden Nodes in a Solution', avg_num_hidden_nodes)
# logger.info('Solution found on average in:', avg_num_generations, 'generations')
# logger.info('Minimum number of hidden nodes:', min_hidden_nodes)
# logger.info('Maximum number of hidden nodes:', max_hidden_nodes)
# logger.info('Minimum number of generations:', min_num_generations)
# logger.info('Found minimal solution:', found_minimal_solution, 'times')