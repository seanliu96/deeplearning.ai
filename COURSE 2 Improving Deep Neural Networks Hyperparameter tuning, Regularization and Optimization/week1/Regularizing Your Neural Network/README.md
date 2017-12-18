## Norm Regularization

One of the first things you should try  to solve a high variance problem probably regularization
$$
\min_{w,b}{J(w,b)} = \min_{w,b}{\frac{1}{m}\sum_{i=1}^m{\ell(\hat{y}^{(i)}, y)}} + \frac{\lambda}{2m}||w||_2^2
$$
We always omit
$$
\frac{\lambda}{2m}||b||^2
$$
because w is usually a pretty high dimensional parameter vector, especially with a high variance problem.

### Different Regularization

- L2 regularization (most often)
  $$
  ||w||_2^2  = \sum_{j=1}^{n_x}{w_j^2} = w^Tw
  $$

- L1 regularisation (more zeors and more sparse)
  $$
  ||w||_1=\sum_{j=1}^{n_x}|w_j|
  $$

- Frobenius norm regularization (the sum of square of elements of a matrix)
  $$
  ||w^{[l]}||_F^2=\sum_{i=1}^{n^{[l-1]}}\sum_{j=1}^{n^{[l]}}(w_{ij}^{[l]})^2
  $$





### Derivatives

$$
dw^{[l]} = \frac{1}{m}dZ^{[l]}{A^{[l-1]}}^T + \frac{\lambda}{m}w^{[l]}
$$

### Process

$$
w^{[l]} := w^{[l]} - \alpha dw^{[l]} = w^{[l]} - \alpha(\frac{1}{m}dZ^{[l]}{A^{[l-1]}}^T + \frac{\lambda}{m}w^{[l]})=(1-\frac{\alpha \lambda}{m})w^{[l]}-\alpha(\frac{1}{m}dZ^{[l]}{A^{[l-1]}}^T)
$$

L2 regularization is sometimes called weight decay because the coefficient of w is going to be a little bit less than 1.

## Why Regularization Reduces Overfitting

If the regularization becomes very large, the parameters W very small, so Z will be relatively small, kind of ignoring the effects of b for now, so Z will be relatively small or, really, I should say it takes on a small range of values. And so the activation function if is tanh, say, will be relatively linear. And so your whole neural network will be computing something not too far from a big linear function which is therefore pretty simple function rather than a very complex highly non-linear function. And so is also much less able to overfit. 

And you might not see a decrease monotonically on cost function.

## Dropout Regulazation

With dropout, what we're going to do is go through each of the layers of the network and set some probability of eliminating a node in neural network.

### Train

Suppose
$$
d^{[3]}=np.random.rand(a^{[3]}.shape[0], a^{[3]}.shape[1]) < keep.prob
$$
Then
$$
a^{[3]} = a^{[3]} \circ d^{[3]} \\
a^{[3]} /= keep.prob
$$
Because
$$
z^{[4]} = w^{[4]}a^{[3]}+b^{[4]}, \text{where } a^{[3]} \text{decreases by 20% randomly}
$$
So it needs to divede by the keep.prob to make z not reduced.

This is inverted dropout.

### Test

No dropout.
$$
z^{[1]} = W^{[1]}a^{[0]}+b{[1]} \\
a^{[1]}=g^{[1]}(z^{[1]}) \\
z^{[2]} = W^{[2]}a^{[1]}+b{[2]} \\
a^{[2]}=g^{[2]}(z^{[2]}) \\
...
$$

## Why Does Dropout Work

Cannot rely on any one feature, so have to spread out weights (shrink weights)

## Data Augmentation

This can be an inexpensive way to give your algorithm more data and therefore sort of regularize it and reduce over fitting. And by synthesizing examples like this what you're really telling your algorithm is that If something is a cat then flipping it horizontally is still a cat.

![data augmentation](https://raw.githubusercontent.com/seanliu96/deeplearning.ai/master/COURSE%202%20Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/week1/Regularizing%20Your%20Neural%20Network/images/data%20augmentation.png)

## Early Stopping

The main downside of early stopping is that this couples these two tasks. So you no longer can work on these two problems independently, because by stopping gradient decent early, you're sort of breaking whatever you're doing to optimize cost function J, because now you're not doing a great job reducing the cost function J.

![early stopping](https://raw.githubusercontent.com/seanliu96/deeplearning.ai/master/COURSE%202%20Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/week1/Regularizing%20Your%20Neural%20Network/images/early%20stopping.png)