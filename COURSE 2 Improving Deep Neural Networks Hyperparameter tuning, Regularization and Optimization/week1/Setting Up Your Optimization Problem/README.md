## Nomalizing Inputs

Substract mean:
$$
\mu = \frac{1}{m}\sum_{i=1}^{m}x^{(i)} \\
x := x - \mu
$$
Normalize variance:
$$
\sigma^{2} = \frac{1}{m}\sum_{i=1}^{m}{x^{(i)}}^2 \\
x /= \sigma^{2}
$$
And use the same parameters to normalize test set

### Why Nomalize Inputs

If you normalize the features, then your cost function will on average look more symmetric. And if you're running gradient descent on the cost function, then you might have to use a very small learning rate because if you're here that gradient descent might need a lot of steps to oscillate back and forth before it finally finds its way to the minimum. Whereas if you have a more spherical contours, then wherever you start gradient descent can pretty much go straight to the minimum.

## Vanishing / Exploding Gradients

In a very deep network, 
$$
\text{if }w^{[L]} > I, \text{ then the } w^{[l]} \text{ will  grow exponentially} \\
\text{if }w^{[L]} < I, \text{ then the } w^{[l]} \text{ will  decrease exponentially} \\
$$
If your activations or gradients increase or decrease exponentially as a function of L, then these values could get really big or really small. And this makes training difficult, especially if your gradients are exponentially smaller than L, then gradient descent will take tiny little steps. It will take a long time for gradient descent to learn anything.

## Weight Initialization Optimisation for Deep Network

$$
W^{[l]} = np.random.randn(shape) * \sqrt{\frac{2}{n^{[l-1]}}} , \text{ where } g(z^{[z]}) \text{ is the ReLU function} \\
W^{[l]} = np.random.randn(shape) * \sqrt{\frac{1}{n^{[l-1]}}} , \text{ where } g(z^{[z]}) \text{ is the tanh function} \\
W^{[l]} = np.random.randn(shape) * \sqrt{\frac{2}{n^{[l-1]+n^{[l]}}}} , \text{ where } g(z^{[z]}) \text{ is the tanh function} \\
$$

## Gradient Checking

for each i
$$
d\theta_{approx}^{[i]} = \frac{J(\theta_1,\theta_2,...,\theta_{i}+\varepsilon, \theta_{i+1}...) - J(\theta_1,\theta_2,...,\theta_{i}-\varepsilon, \theta_{i+1}...)}{2\varepsilon} \approx d\theta^{[i]}
$$
check 
$$
\frac{||d\theta_{approax}^{[i]} - d\theta||_2}{||d\theta_{approax}^{[i]}||_2+||d\theta||_2} \approx 10^{-7}, \text{ where } \varepsilon= 10^{-7}
$$

### Notes

- Don't use in training - only to debug
- If algorithm fails grad check, look at components to try to identify bug
- Remember regularization
- Doesn't work with dropout
- Run at random initialization