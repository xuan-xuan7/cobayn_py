from collections import defaultdict
from itertools import product
from scipy.stats import norm

def calculate_continuous_cpds(train_data, bn_model, types, node):


    cpd = defaultdict(lambda: defaultdict)

    try:
        parents = bn_model.get_parents(node)
    except:
        return cpd
                

    continuous_indices = []
    discrete_indices = []
    for i, value in enumerate(types):
        if value:
            discrete_indices.append(i)
        else:
            continuous_indices.append(i)


    # 计算连续节点的条件概率分布
    parent_cardinality = [set(train_data.iloc[:, parent]) for parent in parents]


    def filter_rows(data, column_index, column_values):
        filtered_data = data
        for i, index in enumerate(column_index):
            filtered_data = filtered_data[filtered_data[index] == column_values[i]]
        return filtered_data


    for parent_values in product(*[cardinality for cardinality in parent_cardinality]):
        filter_data = filter_rows(train_data, parents, parent_values)
        mu, std = norm.fit(filter_data[node])
        distribution = norm(mu, std)
        cpd[parent_values] = distribution


    return cpd