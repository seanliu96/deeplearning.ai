## Softmax Regression

If we have multiple possible classes, there's a generalization of logistic regression called Softmax regression.

The number of units upper layer which is layer L is going to equal to C (the number of possible classes).

And the output labels y hat is going to be C by one dimensional vector, because it now has to output C numbers, giving you these C probabilities. 

And the upper layer's activation function is
$$
a_i^{[L]} = \frac{e^{z_i^{[L]}}}{\sum_je^{z_j^{[L]}}}
$$

### Understanding Softmax

Softmax regression generalizes logistic regression to C classes

### Loss Function

$$
\ell (\hat{y}, y) = - \sum_{j=1}^C{y_j\log{\hat{y_j}}} \text{, where } y_j =1  \text{ if  } x \text{ belongs to } j \text{ class, else } y_j = 0
$$

$$
J(W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}, ..., W^{[m]}, b^{[m]}) = \frac{1}{m} \sum_{i=1}^m \ell (\hat{y}^{(i)}, y^{(i)})
$$

### Backward Prop

$$
dZ^{[L]} = \hat{y} - y
$$



