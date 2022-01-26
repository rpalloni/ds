Docs: https://www.tensorflow.org/api_docs/python/tf/


### Neural Networks

On a lower level neural networks are simply a combination of elementry math operations and some more advanced linear algebra. Each neural network consists of a sequence of **layers** in which data passes through. These layers are made up on neurons and the neurons of one layer are connected to the next. These connections are defined by some numeric values called weights. Each layer also has a bias, an extra neuron that has no connections and holds a single numeric value. Data starts at the input layer and is transformed as it passes through subsequent layers. The data at each subsequent neuron is defined as the following.

<img src="https://latex.codecogs.com/svg.image?\mathbf{\color{DarkOrange}Y&space;=(\sum_{i=0}^n&space;w_i&space;x_i)&space;&plus;&space;b}" />

*Y*: output of the current neuron \
*w*: weight of each connection to the neuron \
*x*: value of the connected neuron from the previous value \
*b*: bias at each layer, this is a constant \
*n*: number of connections

- weighted sum at each and every neuron passes information through the network
- bias allows to shift the network up or down by a constant value (intercept)
- neuron is responsible for generating/holding/passing ONE input (e.g. input image 30x30 pixels => 900 neurons)

## Activation function
A function applyed to the equation seen above to add complexity and dimensionality to the network. 

<img src="https://latex.codecogs.com/svg.image?\mathbf{\color{DarkOrange}Y&space;=&space;F((\sum_{i=0}^n&space;w_i&space;x_i)&space;&plus;&space;b)}" />

Activation functions are higher order/degree functions that aim to add a higher dimension to data and are applied to the weighed sum of a neuron. This intorduces more complexity to the model aiming at making better/more complex predictions 'augmenting' the data.

A list of some common activation functions and their graphs:

- Relu (Rectified Linear Unit) => y > 0

![alt text](https://yashuseth.files.wordpress.com/2018/02/relu-function.png?w=309&h=274)
- Tanh (Hyperbolic Tangent) => -1 < y < 1

![alt text](http://mathworld.wolfram.com/images/interactive/TanhReal.gif)
- Sigmoid  => 0 < y < 1

![alt text](https://miro.medium.com/max/970/1*Xu7B5y9gp0iL5ooBj7LtWw.png)


## Backpropagation
The network will start with predefined activation functions (they may be different at each layer), random weights and biases. As the training goes on, the model will learn the correct weights and biases and adjust the network accordingly using a technqiue called **backpropagation** . 
This adjustment process is a loss/cost function optimization, i.e. the iterative evaluation of the difference between model outcome (predictions) and actual outcome (observations) => y - y'. \

A list of some common loss/cost functions to minimize: \
*Regression losses*:
- Mean Squared Error
- Mean Absolute Error
- Smooth Mean Absolute Error

*Classificaton losses*:
- Cross-Entropy Loss (Log Loss)
- SVM Loss

**Gradient descent** is the most commonly used optimization algorithm to minimize the function by iteratively moving towards the global minimum at a certaing learning step.

A list of common optimizer:
- Gradient Descent
- Stochastic Gradient Descent
- Adaptive Gradient (AdaGrad)
- Adaptive Moment Estimation (Adam)
- Root Mean Square Propagation (RMSprop)
