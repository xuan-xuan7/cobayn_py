from collections import defaultdict
from itertools import product
import numpy as np
from scipy.stats import norm

def calculate_continuous_cpds(train_data, bn_model, types, node):

    # 统计每个节点的计数和联合计数
    node_counts = defaultdict(int)
    joint_counts = defaultdict(int)

    # 遍历训练数据，统计计数和联合计数
    for index, sample in train_data.iterrows():
        node_value = sample[node]

        node_counts[node] += 1

        try:
            parents = bn_model.get_parents(node)
        except:
            continue
            
        if len(parents) > 0:
            parent_values = tuple(sample[parent] for parent in parents)
            joint_counts[parent_values + (node_value,)] += 1
                

    continuous_indices = []
    discrete_indices = []
    for i, value in enumerate(types):
        if value:
            discrete_indices.append(i)
        else:
            continuous_indices.append(i)


    # 计算连续节点的条件概率分布
    parent_cardinality = [set(train_data.iloc[:, parent]) for parent in parents]

    cpd = defaultdict(lambda: defaultdict(float))

    if all(parent in discrete_indices for parent in parents):  # 所有父节点都是离散节点的情况
        for parent_values in product(*[cardinality for cardinality in parent_cardinality]):
            parent_counts = 0
            parent_total_counts = 0
            for value in set(train_data.iloc[:, node]):
                parent_total_counts += joint_counts[tuple(parent_values) + (value,)]
            for node_value in sorted(set(train_data.iloc[:, node])):
                parent_counts += joint_counts[tuple(parent_values) + (node_value,)]

                # 计算条件概率
                if parent_total_counts > 0:
                    cpd[parent_values][node_value] = parent_counts / parent_total_counts

    else:  # 至少有一个父节点是连续节点的情况
        for parent_values in product(*[cardinality for cardinality in parent_cardinality]):
            parent_continuous_indices = [index for index, parent in enumerate(parents) if parent in continuous_indices]
            parent_continuous_values = [parent_values[i] for i in parent_continuous_indices]
            parent_discrete_indices = [index for index, parent in enumerate(parents) if parent in discrete_indices]
            parent_discrete_values = [parent_values[i] for i in parent_discrete_indices]

            parent_counts = 0
            parent_total_counts = 0

            for value in set(train_data.iloc[:, node]):
                parent_total_counts += joint_counts[tuple(parent_values) + (value,)]
            for node_value in sorted(set(train_data.iloc[:, node])):
                parent_counts += joint_counts[tuple(parent_values) + (node_value,)]

                # 计算条件概率
                if parent_total_counts > 0:
                    probability = 1
                    for index in parent_continuous_indices:
                        parents_col = parents[index]
                        filtered_rows = train_data.iloc[:, parents_col].isin(parent_cardinality[index])
                        mean = np.mean(train_data.iloc[:, node][filtered_rows])
                        std = np.std(train_data.iloc[:, node][filtered_rows])
                        probability *= norm.pdf(parent_values[index], mean, std)
                    cpd[tuple(parent_discrete_values)][node_value] = probability * parent_counts / parent_total_counts
    print(cpd)
    return cpd