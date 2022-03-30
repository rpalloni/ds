import numpy as np

class NeuralNetwork:
    def __init__(self, input_vectors, targets, learning_rate):
        self.weights = np.array([np.random.randn()]*input_vectors.shape[1])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        self.input_vectors = input_vectors
        self.targets = targets
        self.error = 0

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        return prediction

    def _compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2

        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        derror_dbias = derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        derror_dweights = derror_dprediction * dprediction_dlayer1 * dlayer1_dweights

        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (derror_dweights * self.learning_rate)

    # train
    def __next__(self):

        random_data_index = np.random.randint(len(self.input_vectors))
        input_vector = self.input_vectors[random_data_index]
        target = self.targets[random_data_index]
        derror_dbias, derror_dweights = self._compute_gradients(input_vector, target)
        self._update_parameters(derror_dbias, derror_dweights)
        prediction = self.predict(input_vector)
        self.error = np.square(prediction - target)
        return self.error


y = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0])

X = np.array([(90, 178), (72, 180), (48, 161), (90, 176), (48, 164), (76, 190), (62, 175), (52, 161), (93, 190), (72, 164),
              (70, 178), (60, 167), (61, 178), (73, 180), (70, 185), (89, 178), (68, 174), (72, 173), (85, 184), (76, 168)])

nnet = NeuralNetwork(X, y, 0.001)

for i in range(100):
    next(nnet)

nnet.error
