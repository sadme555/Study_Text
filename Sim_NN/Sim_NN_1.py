'''
            O
  O         O       
  O         O      O
            O

n_x = 2
n_h = 4
n_y = 1
'''
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from load_data import load_planar_dataset


def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

# 定义网络结构
# 网络输入输出以及隐藏层神经元个数
def layer_sizes(X, Y):
    n_x = X.shape[]     # 输入层神经元个数
    n_h = 4             # 隐藏层神经元个数
    n_y = Y.shape[]     # 输出层神经元个数
    return (n_x,n_h,n_y)

# 初始化模型参数
# 随机初始化权重以及偏置为0
def initialize_parameters(n_x,n_h,n_y):
    '''
    输入每层的神经元数量
    返回：隐藏层、输出层的参数
    '''
    # 1.初始化权重
    np.random.seed(2) #指定随机种子

    # 2.创建隐藏层的两个参数
    # randn：标准正态分布
    # rand：均匀分布[0,1)
    # 0.01权重缩放因子，激活值更合适，训练更稳定
    # 为什么是(n_h,n_x)？  矩阵乘法 W·X + b   X是(n_x,2)  
    W1 = np.random.randn(n_h,n_x)*0.01
    # zeros :接受一个矩阵元组
    b1 = np.zeros((n_h,1)) # 4行1列

    # 3.创建输出层的两个参数
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.randm.zeros((n_y,1))

    # 4.使用断言确保数据格式正确
    assert(W1.shape == (n_h,n_x))
    assert(b1.shape == (n_h,1))
    assert(W2.shape == (n_y,n_h))
    assert(b2.shape == (n_y,1))

    parameters = {
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
    }
    return parameters


# 循环中的第一步：前向传播
def forward_propagation(X,parameters):
    '''
    输入：(n_x,)
    '''
    # 1.取出每一层的参数
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]   
    b2 = parameters["b2"]

    # 2.计算隐藏层的激活值
    Z1 = np.dot(W1,X) + b1  
    A1 = np.tanh(Z1)      # 激活函数tanh

    # 3.计算输出层的激活值
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)     # 激活函数sigmoid

    assert(A2.shape == (1,X.shape[1]))

    cache = {
        "Z1":Z1,
        "A1":A1,  
        "Z2":Z2,
        "A2":A2
    }
    return A2,cache

# 计算损失函数
def compute_cost(A2,Y,parameters):
    m = Y.shape[1]  # 样本数量
    logpro = np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),1-Y)
    cost = (-1 / m)* np.sum(logpro)
    
    cost = np.squeeze(cost)  #确保cost是一个标量
    assert(isinstance(cost,float))
    return cost

# 循环中的第二步：反向传播
def backward_propagation(parameters,cache,X,Y):





