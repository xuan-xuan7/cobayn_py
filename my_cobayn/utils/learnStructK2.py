from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.factors.continuous import ContinuousFactor
from pgmpy.estimators import K2Score

# 创建空的贝叶斯网络
bnet = BayesianModel()

# 假设已知节点的拓扑排序、类型和训练数据
topological_order = ['A', 'B', 'C']  # 替换为实际的节点拓扑排序
node_types = {'A': 'discrete', 'B': 'continuous', 'C': 'discrete'}  # 替换为实际节点的类型
training_data = ...  # 替换为实际的训练数据

# 对每个节点进行处理
for node in topological_order:
    node_type = node_types[node]
    
    if node_type == 'discrete':
        # 对于离散节点，根据训练数据估计TabularCPD
        cpd = TabularCPD(node, num_states, values, evidence=[], evidence_card=[])
        # 这里的num_states是离散节点的状态数量，values是根据训练数据估计得到的概率值，可以使用合适的方法进行估计
        
        # 将CPD添加到贝叶斯网络中
        bnet.add_cpds(cpd)
        
    elif node_type == 'continuous':
        # 对于高斯节点，根据训练数据估计ContinuousFactor
        factor = ContinuousFactor([node], mean, cov)
        # 这里的mean和cov是根据训练数据估计得到的高斯分布的均值和协方差矩阵，可以使用合适的方法进行估计
        
        # 将Factor添加到贝叶斯网络中
        bnet.add_factors(factor)
    
# 打印贝叶斯网络的CPDs和Factors
print(bnet.get_cpds())
print(bnet.get_factors())