## Normalizing Actications in a Network

### Normalizing Inputs to Speed Up Learning

$$
\mu = \frac{1}{m} \sum_i{x^{(i)}} \\
\sigma ^2 = \frac{1}{m}\sum_i{x^{(i)}}^2 \\
x = \frac{x - \mu}{\sigma^2}
$$

### Implementing Batch Norm

Given some intermediate values in NN
$$
\mu^{[l]} = \frac{1}{m} \sum_i{{z^{[l]}}^{(i)}} \\
{\sigma^{[l]}}^2 = \frac{1}{m}\sum_i{{z^{[l]}}^{(i)}}^2 \\
{z_{norm}^{[l]}}^{(i)} = \frac{{{z^{[l]}}^{(i)}} - \mu}{{\sigma^{[l]}}^2} \\
{\tilde{z}^{[l]}}^{(i)} = \gamma {z_{norm}^{[l]}}^{(i)} + \beta \text{, where } \gamma, \beta \text{ are learnable parameters of model}
$$

## Fitting Batch Norm Into a Neural Network

![adding batch norm to a network](https://github.com/seanliu96/deeplearning.ai/raw/master/COURSE%202%20Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/week3/Batch%20Normalization/images/adding%20batch%20norm%20to%20a%20network.png)

### Working With Mini-Batches

Because Batch Norm zeroes out the mean of these Z values in the layer, there's no point having this parameter b.

![working with mini-batches](https://github.com/seanliu96/deeplearning.ai/raw/master/COURSE%202%20Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/week3/Batch%20Normalization/images/working%20with%20mini-batches.png)

### Implementing Gradient Descent

![implementing gradient descent](https://github.com/seanliu96/deeplearning.ai/raw/master/COURSE%202%20Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/week3/Batch%20Normalization/images/implementing%20gradient%20descent.png)

### Why Batch Norm Work

One intuition behind why batch norm works is, this is to take on a similar range of values that can speed up learning, but further values in your hidden units and not just for your input there.

A second reason why batch norm works, is it makes weights, later or deeper than your network, say the weight on layer 10, more robust to changes to weights in earlier layers of the neural network because batch norm overcomes covariate shift on weights in earlier layers.

### Batch Norm as Regularization

- Each mini-batch is scaled by the mean / variance computed on just that mini-batch
- This adds some noise to the values z within that mini-batch. So similar to dropout, it adds some noise to each hidden layer's activations
- This has a slight relularization effect because by adding noise to the hidden units, it's forcing the downstream hidden units not to rely too much on any one hidden unit.