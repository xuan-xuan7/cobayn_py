from collections import defaultdict
import numpy as np

# 假设你的训练数据 train_data 是一个二维数组，形状为 (n_samples, n_features)
n_samples, n_features = train_data.shape

# 假设连续节点的列索引为 continuous_indices，离散节点的列索引为 discrete_indices
continuous_indices = range(9)
discrete_indices = range(9, 16)

# 统计每个节点的计数和联合计数
node_counts = defaultdict(int)
joint_counts = defaultdict(lambda: defaultdict(int))

# 遍历训练数据，统计计数和联合计数
for sample in train_data:
    for i in continuous_indices:
        node = chr(ord('A') + i)
        node_value = sample[i]

        node_counts[node] += 1

        parents = dag[node]
        parent_values = tuple(sample[ord(parent) - ord('A')] for parent in parents)

        joint_counts[node][parent_values + (node_value,)] += 1

    for i in discrete_indices:
        node = chr(ord('A') + i)
        node_value = sample[i]

        node_counts[node] += 1

        parents = dag[node]
        parent_values = tuple(sample[ord(parent) - ord('A')] for parent in parents)

        joint_counts[node][parent_values + (node_value,)] += 1

# 计算连续节点的条件概率分布
continuous_cpds = {}

for node in continuous_indices:
    node = chr(ord('A') + node)
    parents = dag[node]
    parent_cardinality = [len(set(train_data[:, ord(parent) - ord('A')])) if ord(parent) - ord('A') in discrete_indices else 1 for parent in parents]

    cpd = defaultdict(lambda: defaultdict(float))

    if all(ord(parent) - ord('A') in discrete_indices for parent in parents):  # 所有父节点都是离散节点的情况
        for parent_values in product(*[range(cardinality) for cardinality in parent_cardinality]):
            for node_value in sorted(set(train_data[:, ord(node) - ord('A')])):
                parent_counts = sum(joint_counts[node][parent_values + (node_value,)].values())
                parent_total_counts = sum(joint_counts[node][parent_values + (value,)].values() for value in set(train_data[:, ord(node) - ord('A')]))

                # 计算条件概率
                cpd[parent_values][node_value] = parent_counts / parent_total_counts

    else:  # 至少有一个父节点是连续节点的情况
        for parent_values in product(*[range(cardinality) for cardinality in parent_cardinality]):
            parent_continuous_indices = [i for i, parent in enumerate(parents) if ord(parent) - ord('A') in continuous_indices]
            parent_continuous_values = [parent_values[i] for i in parent_continuous_indices]
            parent_discrete_indices = [i for i, parent in enumerate(parents) if ord(parent) - ord('A') in discrete_indices]
            parent_discrete_values = [parent_values[i] for i in parent_discrete_indices]

            for node_value in sorted(set(train_data[:, ord(node) - ord('A')])):
                parent_counts = sum(joint_counts[node][parent_values + (node_value,)].values())
                parent_continuous_counts = sum(joint_counts[node][tuple(parent_continuous_values) + (value,)].values() for value in set(train_data[:, ord(node) - ord('A')]))
                parent_total_counts = sum(joint_counts[node][parent_values + (value,)].values() for value in set(train_data[:, ord(node) - ord('A')]))

                # 计算条件概率
                if parent_continuous_counts > 0:
                    mean = np.mean(train_data[:, ord(node) - ord('A')][train_data[:, [ord(parent) - ord('A') for parent in parents[parent_continuous_indices]]] == tuple(parent_continuous_values)])
                    std = np.std(train_data[:, ord(node) - ord('A')][train_data[:, [ord(parent) - ord('A') for parent in parents[parent_continuous_indices]]] == tuple(parent_continuous_values)])
                    probability = norm.pdf(node_value, mean, std)
                    cpd[tuple(parent_discrete_values)][node_value] = probability * parent_counts / parent_total_counts
                else:
                    cpd[tuple(parent_discrete_values)][node_value] = parent_counts / parent_total_counts

    continuous_cpds[node] = cpd

# 计算离散节点的条件概率分布
discrete_cpds = {}

for node in discrete_indices:
    node = chr(ord('A') + node)
    parents = dag[node]
    parent_cardinality = [len(set(train_data[:, ord(parent) - ord('A')])) for parent in parents]

    cpd = defaultdict(lambda: defaultdict(float))

    for parent_values in product(*[range(cardinality) for cardinality in parent_cardinality]):
        for node_value in sorted(set(train_data[:, ord(node) - ord('A')])):
            parent_counts = sum(joint_counts[node][parent_values + (node_value,)].values())
            node_count = node_counts[node]

            cpd[parent_values][node_value] = parent_counts / node_count

    discrete_cpds[node] = cpd

# 打印连续节点的条件概率分布
for node in continuous_cpds:
    print(f'CPD for Continuous Node {node}:')
    print(continuous_cpds[node])
    print()

# 打印离散节点的条件概率分布
for node in discrete_cpds:
    print(f'CPD for Discrete Node {node}:')
    print(discrete_cpds[node])
    print()