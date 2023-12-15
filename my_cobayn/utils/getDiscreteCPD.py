from collections import defaultdict
import numpy as np
from itertools import product
from scipy.stats import norm

def calculate_discrete_cpd(train_data, bn_model, types, node):

    # 统计每个节点的计数和联合计数
    joint_counts = defaultdict(lambda: defaultdict(int))
    
    
    node_cardinality = set(train_data.iloc[:, node])

    cpd = defaultdict(lambda: defaultdict(float))

    try:
        parents = bn_model.get_parents(node)
    except:
        return cpd
    
    # print(node, parents)
    # 遍历训练数据，统计计数和联合计数
    for index, sample in train_data.iterrows():
        node_value = sample[node]
        parent_values = tuple(sample[parent] for parent in parents)
        joint_counts[parent_values][node_value] += 1


    continuous_indices = []
    discrete_indices = []
    for i, value in enumerate(types):
        if value:
            discrete_indices.append(i)
        else:
            continuous_indices.append(i)


    parent_cardinality = [set(train_data.iloc[:, parent]) for parent in parents]

    for parent_values in product(*[cardinality for cardinality in parent_cardinality]):
        total_count = 0
        for node_value in node_cardinality:
            total_count += joint_counts[tuple(parent_values)][node_value]

        if total_count > 0:
            for node_value in node_cardinality:
                node_count = joint_counts[parent_values][node_value]
                cpd[parent_values][node_value] = node_count / total_count


    return cpd