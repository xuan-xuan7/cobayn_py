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
        print(node)
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


    # 计算连续节点的条件概率分布
    parent_cardinality = [set(train_data.iloc[:, parent]) for parent in parents]


    if any(parent in continuous_indices for parent in parents):
        pass
        # 如果父节点中包含连续节点，则将连续节点离散化处理后计算CPD
        def calculate_cpd(continuous_parent_value, discrete_parent_value, discrete_node_value):
            # 获取连续父节点取值对应的参数
            continuous_mean = continuous_parent_mean[continuous_parent_value]
            continuous_std = continuous_parent_std[continuous_parent_value]
        
            # 获取离散节点的参数
            discrete_params = discrete_node_params[discrete_parent_value]
        
            # 计算离散节点在给定父节点取值下的条件概率
            probability = discrete_params[discrete_node_value]
        
            # 根据连续父节点取值来调整条件概率
            probability *= norm.pdf(discrete_node_value, continuous_mean, continuous_std)
        
            return probability
        parent_data = train_data[:, parent_indices]
        node_data = train_data[:, node_index]

        for parent_values in product(*[sorted(set(parent_data[:, i])) for i in range(len(parents))]):
            parent_mask = np.ones(len(train_data), dtype=bool)

            for i, parent_value in enumerate(parent_values):
                parent_mask &= (parent_data[:, i] == parent_value)

            node_values = sorted(set(node_data[parent_mask]))

            for node_value in node_values:
                cpd[parent_values][node_value] = np.mean(node_data[parent_mask] == node_value)

    else:
        # 如果父节点都是离散节点，则直接按照统计计数计算CPD
        for parent_values in product(*[cardinality for cardinality in parent_cardinality]):
            total_count = 0
            for node_value in node_cardinality:
                total_count += joint_counts[tuple(parent_values)][node_value]

            if total_count > 0:
                for node_value in node_cardinality:
                    node_count = joint_counts[parent_values][node_value]
                    cpd[parent_values][node_value] = node_count / total_count


    return cpd