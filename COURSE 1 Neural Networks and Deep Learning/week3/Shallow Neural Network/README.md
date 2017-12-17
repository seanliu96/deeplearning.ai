## Neural Networks Overview

![nn](https://raw.githubusercontent.com/seanliu96/deeplearning.ai/master/COURSE%201%20Neural%20Networks%20and%20Deep%20Learning/week3/Shallow%20Neural%20Network/images/nn.png)

## Neural Network Representation

![nn representation](https://raw.githubusercontent.com/seanliu96/deeplearning.ai/master/COURSE%201%20Neural%20Networks%20and%20Deep%20Learning/week3/Shallow%20Neural%20Network/images/nn.png)

## Computing a Neural Network's Output

$$
z^{[1]} = {W^{[1]}}^Tx+b^{[1]}  = {W^{[1]}}^Ta^{[0]}+b^{[1]} \\
a^{[1]} = \sigma(z^{[1]}) \\
z^{[2]} = {W^{[2]}}^Ta^{[1]}+b^{[2]} \\
a^{[2]} = \sigma(z^{[2]}) \\
...
$$

## Vectorizing across multiple examples

$$
{a^{[2]}}^{(i)}: \text{example i, layer 2}
$$

![vectorizing nn](https://raw.githubusercontent.com/seanliu96/deeplearning.ai/master/COURSE%201%20Neural%20Networks%20and%20Deep%20Learning/week3/Shallow%20Neural%20Network/images/vectorizing%20nn.png)

## Activation functions

$$
\text{sigmoid} \\
a = \frac{1}{1 + e^{-z}},  \\
a' = a(1-a) \\
\text{tanh} \\
a = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}},  \\
a' =  1 - a^2\\
\text{ReLU} \\
a = max(0, z), \\
a' = 
	\begin{cases} 
		0 & \quad \text{if }z < 0 \\
		1 & \quad \text{if }z \geq  0
	\end{cases} \\
\text{leaky ReLU} \\
a = max(0.01z, z) . \\
a' = \begin{cases} 
		0.01 & \quad \text{if }z < 0 \\
		1 & \quad \text{if }z \geq  0
	\end{cases}
$$

![activation functions](https://raw.githubusercontent.com/seanliu96/deeplearning.ai/master/COURSE%201%20Neural%20Networks%20and%20Deep%20Learning/week3/Shallow%20Neural%20Network/images/activation%20functions.png)

## Why do you need non-linear activation functions

Suppose
$$
z^{[1]} = W^{[1]}x+b^{[1]} \\
a^{[1]} = g^{[1]}(z^{[1]}) = z^{[1]} \\
z^{[2]} = W^{[2]}a^{[1]}+b^{[2]} \\
a^{[2]} = g^{[2]}(z^{[2]}) = z^{[2]}
$$
Then
$$
a^{[1]} = z^{[1]} = W^{[1]}x+b{[1]} \\
a^{[2]} = z^{[2]} = W^{[2]}a^{[1]}+b^{[2]} \\
\\\rightarrow
a^{[2]} = W^{[2]}(W^{[1]}x+b^{[1]}) + b^{[2]} = (W^{[2]}W^{[1]})x+(W^{[2]}b^{[1]}+b^{[2]})
$$
It is similar to 
$$
a^{[2]} = W'x + b'
$$
If you were to use linear activation functions or we go to call them identity activation functions, then the new network is just outputting a linear function of the input and we'll talk about deep networks later new networks with many many layers, many many hidden layers and it turns out that if you use a linear activation function or alternatively if you don't have an activation function. Then no matter how many layers, your neural network has always doing is just computing a linear activation function.

## Gradient Descent for Neural Networks

### Backpropogation

$$
dZ^{[2]} = {g^{[2]}}'(Z^{[2]}) \\
dW^{[2]} = \frac{1}{m}dZ^{[2]}{A^{[1]}}^T \\
db^{[2]} = \frac{1}{m} np.sum(dZ^{[2]}, axis=1, keepdims=True) \\
dz^{[1]} ={W^{[2]}}^TdZ^{[2]} \circ {g^{[1]}}'(Z^{[1]}) \\
dW^{[1]} = \frac{1}{m}dZ^{[1]}X^T \\
db^{[1]} = \frac{1}{m} np.sum(dZ^{[1]}, axis=1, keepdims=True)
$$

## Random Initialization

If initializing weights to zeros, then all weights will update symmetricly. Then no matter how many nodes in one layer, your neural network has always doing is just using one node in one layer.