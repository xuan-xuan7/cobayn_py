from itertools import combinations
from collections import defaultdict
import numpy as np

def learn_struct_K2(data, order, scoring_criterion):
    num_nodes = len(order)
    node_parents = defaultdict(list)

    # Iterate over the nodes in topological order
    for i in range(num_nodes):
        node = order[i]
        parents = order[:i]  # Potential parents for the current node

        best_score = float('-inf')
        best_parents = None

        # Iterate over all possible parent combinations
        for num_parents in range(i):
            parent_combinations = combinations(parents, num_parents)

            # Evaluate the scoring criterion for each parent combination
            for parent_set in parent_combinations:
                score = scoring_criterion(data, node, parent_set)

                # Update the best score and best parent set if necessary
                if score > best_score:
                    best_score = score
                    best_parents = parent_set

        # Add the best parent set to the node_parents dictionary
        if best_parents is not None:
            node_parents[node].extend(best_parents)

    return node_parents