Docs: https://www.tensorflow.org/api_docs/python/tf/


### Neural Networks

On a lower level neural networks are simply a combination of elementry math operations and some more advanced linear algebra. Each neural network consists of a sequence of layers in which data passes through. These layers are made up on neurons and the neurons of one layer are connected to the next. These connections are defined by some numeric values called weights. Each layer also has something called a bias, this is simply an extra neuron that has no connections and holds a single numeric value. Data starts at the input layer and is transformed as it passes through subsequent layers. The data at each subsequent neuron is defined as the following.

$$Y =(\sum_{i=0}^n w_i x_i) + b$$

$w$ stands for the weight of each connection to the neuron

$x$ stands for the value of the connected neuron from the previous value

$b$ stands for the bias at each layer, this is a constant

$n$ is the number of connections

$Y$ is the output of the current neuron

* weighted sum at each and every neuron passes information through the network. 
* bias allows us to shift the network up or down by a constant value (intercept)

## Activation function
This is a function that we apply to the equation seen above to add complexity and dimensionality to our network. Our new equation with the addition of an activation function $F(x)$ is seen below.

$Y =F((\sum_{i=0}^n w_i x_i) + b)$

Activation functions are simply a function that is applied to the weighed sum of a neuron. They can be anything we want but are typically higher order/degree functions that aim to add a higher dimension to our data. We would want to do this to introduce more comolexity to our model. By transforming our data to a higher dimension, we can typically make better, more complex predictions.

A list of some common activation functions and their graphs can be seen below.

- Relu (Rectified Linear Unit)

![alt text](https://yashuseth.files.wordpress.com/2018/02/relu-function.png?w=309&h=274)
- Tanh (Hyperbolic Tangent)

![alt text](http://mathworld.wolfram.com/images/interactive/TanhReal.gif)
- Sigmoid 

![alt text](https://miro.medium.com/max/970/1*Xu7B5y9gp0iL5ooBj7LtWw.png)


## Backpropagation
Our network will start with predefined activation functions (they may be different at each layer), random weights and biases. As we train the network by feeding it data it will learn the correct weights and biases and adjust the network accordingly using a technqiue called **backpropagation** . 


