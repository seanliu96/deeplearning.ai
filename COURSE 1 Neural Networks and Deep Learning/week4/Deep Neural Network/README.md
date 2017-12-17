## Building Blocks of Deep Neural Networks

![blocks](https://raw.githubusercontent.com/seanliu96/deeplearning.ai/master/COURSE%201%20Neural%20Networks%20and%20Deep%20Learning/week4/Deep%20Neural%20Network/images/blocks.png)

## Propagation

### Forward Propagation for Layer l

Input
$$
a^{[l-1]}
$$
Cache
$$
z^{[l]} = W^{[l]}a^{[l-1]}+b^{[l]}
$$
Output
$$
a^{[l]} = g^{[l]}(z^{[l]})
$$

#### Vectorized

Input
$$
A^{[l-1]}
$$
Cache
$$
Z^{[l]} = W^{[l]}A^{[l-1]}+b^{[l]}
$$
Output
$$
A^{[l]} = g^{[l]}(Z^{[l]})
$$

### Backward Propagation for Layer l

Input
$$
da^{[l]}
$$
Local
$$
dz{[l]} = da^{[l]} \circ {g^{[l]}}'(z^{[l]})
$$
Output
$$
dW^{[l]} = dz^{[l]} a^{[l-1]}\\
db^{[l]} = dz^{[l]}\\
da^{[l-1]} = {W^{[l]}}^T dz^{[l]}
$$

#### Vectorized

Input
$$
dA^{[l]}
$$
Local
$$
dZ^{[l]} = dA^{[l]} \circ {g^{[l]}}'(Z^{[l]})
$$
Output
$$
dW^{[l]} = \frac{1}{m}dZ^{[l]}A^{[l-1]} \\
db^{[l]} =\frac{1}{m}np.sum(dZ^{[l]}, axis=1, keepdims=True) \\
dA^{[l-1]} = {W^{[l]}}^T dZ^{[l]}
$$

## Parameters vs Hyperparameters

### Parameters

$$
W^{[1]}, b^{[1]} \\
W^{[2]}, b^{[2]} \\
...
$$

### Hyperparameters

Hyperparameters can control W and b
$$
\text{learning rate } \alpha \\
\text{# of iterations} \\
\text{# of hidden layers } L \\
\text{# of hidden units } n^{[1]}, n^{[2]}, ... \\
\text{choice of activation function} \\
\text{momentum term} \\
\text{mini batch size} \\
\text{various forms of regularization parameters }
$$
