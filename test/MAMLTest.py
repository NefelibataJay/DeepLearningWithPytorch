import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable as V

k_spt = 1  ## support data 的个数
k_query = 15  ## query data 的个数


def samplePoints(k):
    # 随机生成50个样本点
    x = np.random.rand(k, 50)
    # 各个元素的采样概率均为0.5
    y = np.random.choice([0, 1], size=k, p=[.5, .5]).reshape([-1, 1])
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    x = x.float()
    y = y.float()
    return x, y


class Model(nn.Module):
    def __init__(self, input_dim, out_dim, n_sample):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim

        self.n_sample = n_sample
        self.W = nn.Parameter(torch.zeros(size=[input_dim, out_dim]))
        # nn.init.xavier_uniform_(self.W.data,gain=1.414)

    def forward(self):
        # 生成用于训练的样本
        X_train, Y_train = samplePoints(self.n_sample)
        # 单层网络的训练
        Y_predict = torch.matmul(X_train, self.W)
        Y_predict = Y_predict.reshape(-1, 1)
        return Y_train, Y_predict


maml = Model(50, 1, 10)
optimizer = optim.Adam(maml.parameters(), lr=0.01, weight_decay=1e-5)
loss_function = nn.MSELoss()
epochs = 1000
tasks = 10
beta = 0.0001
theta_matrix = torch.zeros(size=[tasks, 50, 1])  # shape = (task, parameter.shape)
theta_matrix = theta_matrix.float()  # 保存每个任务梯度下降之后的参数
ori_theta = torch.randn(size=[50, 1])
ori_theta = ori_theta.float()  # 初始化原始参数
meta_gradient = torch.zeros_like(ori_theta)


# 下面定义训练过程
def train(epoch):
    # 对每一个任务进行迭代(训练),保留每一个任务梯度下降之后的参数
    global ori_theta, meta_gradient
    loss_sum = 0.0

    for i in range(tasks):
        maml.W.data = ori_theta.data  # 初始化网络参数
        optimizer.zero_grad()

        # task x data
        Y_train, Y_predict = maml()
        loss_value = loss_function(Y_train, Y_predict)
        loss_sum = loss_sum + loss_value.data.item()
        loss_value.backward()
        optimizer.step()
        # print(maml.W.shape)
        theta_matrix[i, :] = maml.W
    # 对每一个任务进行迭代（测试），利用保留的梯度下降之后的参数作为训练参数，计算梯度和
    for i in range(tasks):
        maml.W.data = theta_matrix[i]
        optimizer.zero_grad()
        Y_test, Y_predict_test = maml()
        loss_value = loss_function(Y_test, Y_predict_test)
        loss_value.backward()
        optimizer.step()
        meta_gradient = meta_gradient + maml.W
    # 更新初始的ori_theta
    ori_theta = ori_theta - beta * meta_gradient / tasks
    print(f"the Epoch is {epoch}", f"the Loss is {loss_sum / tasks}")


def init_model():
    return nn.Conv2d(3, 5, 3)


def randomGetBatch(tasks, batch_size):
    batchs = {}
    for i in range(tasks):
        batchs[i] = (torch.randn(batch_size, 8, 8), torch.randn(batch_size, 1))
    return batchs


k_support = 5


def meta_train(sample_batchs, model, inter_optimizer, loss_function):
    for task, batch in sample_batchs.items():
        model.train()
        new_model = model.parameters()
        X_support, Y_support = batch[:k_support]
        for i in len(X_support):
            Y_predict = model(X_support[i])
            loss = loss_function(Y_predict, Y_support[i])
            inter_optimizer.zero_grad()
            loss.backward()
            inter_optimizer.step()

        X_query, Y_query = batch[k_support:]
        for i in len(X_query):
            Y_predict = model(X_query)
            loss = loss_function(Y_predict, Y_support[i])


def MAML_train(epochs):
    model = init_model()  # 初始化模型 加载预训练模型就是迁移学习
    tasks = 5
    batch_size = 8
    b = 0.0001
    a = 0.0001
    inter_optimizer = optim.Adam(model.parameters(), lr=a)
    optimizer = optim.Adam(model.parameters(), lr=b)
    loss_function = nn.MSELoss()

    for epoch in range(epochs):
        sample_batchs = randomGetBatch(tasks, batch_size)
        meta_train(sample_batchs, model, inter_optimizer, loss_function)


if __name__ == "__main__":
    for epoch in range(epochs):
        train(epoch)
