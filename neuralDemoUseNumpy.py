import numpy as np
class NeuralNetwork():
    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((3,1))-1

    def __sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    # sigmoid曲线的梯度
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, train_set_inputs, train_set_outputs, iters_num):
        for iter in range(iters_num):
            # 前向传播
            output = self.predict(train_set_inputs)
            #计算误差
            error = train_set_outputs - output

            adjustment = np.dot(train_set_inputs.T, error * self.__sigmoid_derivative(output))
            #调整参数
            self.synaptic_weights += adjustment

    def predict(self, inputs):
        return self.__sigmoid(np.dot(inputs, self.synaptic_weights))

if __name__ == '__main__':
    neural_net = NeuralNetwork()
    print('初始参数:\n',neural_net.synaptic_weights)

    train_set_inputs = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    train_set_outputs = np.array([[0,1,1,0]]).T

    neural_net.train(train_set_inputs, train_set_outputs, 1000)
    print("训练结束之后的参数：\n",neural_net.synaptic_weights)
    print("预测：")
    print('预测数据[1,0,0]：\n',neural_net.predict(np.array([1,0,0])))
