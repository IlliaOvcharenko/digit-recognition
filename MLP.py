"""
Попытка построить MLP(multilayer perceptron)
Обучение метод обратного распостранение ошибки
Цель: решение задачи MIST digit recognition
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json


class Network:
    def __init__(self, input_number=1, neuron_number=1, max_steps=1, step_value=1):
        self.layers = [SigmoidLayer(input_number, neuron_number)]
        self.max_steps = max_steps
        self.step_value = step_value
        self.error = 0

    def get_info(self):
        network_info = {
            'loss_function': 'mean_squared_error',
            'max_steps': self.max_steps,
            'step_value': self.step_value,
            'layers': [layer.get_info() for layer in self.layers]
        }
        return network_info

    @staticmethod
    def load(network_info):
        network = Network()
        network.max_steps = network_info['max_steps']
        network.step_value = network_info['step_value']
        network.layers = [SigmoidLayer.load(layer_info) for layer_info in network_info['layers']]
        return network

    def add_layer(self, neuron_number):
        self.layers.append(SigmoidLayer(self.layers[-1].neuron_number, neuron_number))

    def test(self, input_values):
        return np.array([self.predict(input_value) for input_value in input_values])

    def predict(self, input_value):
        result = input_value
        for layer in self.layers:
            result = layer.activate(result)
        return result

    def train(self, input_values, output_values):
        input_len = len(input_values)
        print("Start training")
        for step in range(self.max_steps):
            print("Iteration: %i" % step)
            for i in range(input_len):
                # print(f"\tStep: {i}")
                self.train_step(input_values[i], output_values[i])
            print("Error: %f" % (self.error / input_len))
            self.error = 0

    def train_step(self, input_value, output_value):
        result = np.copy(input_value)
        for layer in self.layers:
            result = layer.activate(result)

        output_layer = self.layers[-1]
        output_layer.error = output_layer.result - output_value
        self.error += np.sum(output_layer.error ** 2)

        for i, layer in enumerate(reversed(self.layers[1:])):
            self.layers[len(self.layers) - i - 2].error = np.sum(
                layer.error * layer.result * (1 - layer.result) * layer.weight[0:-1],
                axis=1
            )

        for layer in self.layers:
            input_value = np.append(input_value, 1)
            delta = layer.error * layer.result * (1 - layer.result) * input_value[:, np.newaxis]
            input_value = layer.result
            layer.weight = layer.weight - self.step_value * delta


class SigmoidLayer:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def __init__(self, input_number=1, neuron_number=1):
        self.input_number = input_number
        self.neuron_number = neuron_number

        # weight initialization
        scaling = 1 / neuron_number
        self.weight = np.random.random_sample((input_number + 1, neuron_number)) * scaling\
                      - scaling / 2

        self.result = None
        self.error = None

    def get_info(self):
        layer_info = {
            'activation_function': 'sigmoid',
            'weight': self.weight.tolist()
        }
        return layer_info

    @staticmethod
    def load(layer_info):
        layer = SigmoidLayer()
        layer.weight = np.array(layer_info['weight'])
        layer.input_number = layer.weight.shape[0] - 1
        layer.neuron_number = layer.weight.shape[1]
        return layer

    def activate(self, input_value):
        input_value = np.append(input_value, 1)
        self.result = np.array(list(map(SigmoidLayer.sigmoid, np.dot(input_value, self.weight))))
        return self.result


if __name__ == '__main__':
    # l = Layer(4, 10, sigmoid)
    # result = l.run(np.array([0.23, -0.11, -0.17, -0.4]))
    # result = l.run(np.array([11, 17, 23, 27]))
    # result = l.run(np.array([1.0, 2.0, 3.0, 4.0]))

    nn = Network(
        input_number=1,
        neuron_number=10,
        max_steps=3000,
        step_value=1
    )
    # nn.add_layer(30)
    # nn.add_layer(10)
    # nn.add_layer(10)
    nn.add_layer(5)
    nn.add_layer(1)
    with open('test_save.json', 'r') as f:
        # json.dump(nn.get_info(), f)
        info = json.load(f)
        nnn = Network.load(info)
        print(len(nnn.layers))
        for l in nnn.layers:
            print("Layer size %i %i " % (l.input_number, l.neuron_number))
            print(l.weight)
        print(nnn.predict([0.5]))

    # train_size = 100
    # train_inputs = np.array([[np.random.random() - 0.5] for _ in range(train_size)])
    # train_outputs = np.array([[np.cos(x[0] * 10) * 0.5 + 0.5] for x in train_inputs])
    # plt.plot([x[0] for x in train_inputs], [y[0] for y in train_outputs], 'go')
    # # plt.show()
    #
    # nn.train(train_inputs, train_outputs)
    #
    # test_size = 1000
    # test_inputs = np.array([[np.random.random() - 0.5] for _ in range(test_size)])
    # test_outputs = nn.test(test_inputs)
    # plt.plot([x[0] for x in test_inputs], [y[0] for y in test_outputs], 'ro')
    # plt.show()
