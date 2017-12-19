## Mini-Batch Gradient Descent

You split up your training set into smaller, little baby training sets and these baby training sets are called mini-batches.
$$
X = [x^{(1)}, x^{(2)}, ...  , x^{(i)} ..., x^{(m)}], Y = [y^{(1)}, y^{(2)}, ...  , y^{(i)} ..., y^{(m)}] \\

\text{mini-batches: }
X = [X^{\{1\}}, X^{\{2\}}, ..., X^{\{t\}}, ...], Y = [Y^{\{1\}}, Y^{\{2\}}, ..., Y^{\{t\}}, ...], \text{where } X^{\{t\}}, Y^{\{t\}} is a mini-batch
$$
![mini-batch gradient descent](https://raw.githubusercontent.com/seanliu96/deeplearning.ai/master/COURSE%202%20Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/week2/Optimization%20Algorithms/images/mini-batch%20gradient%20descent.png)

The code I have written down here is also called doing one epoch of training and epoch is a word that means a single pass through the training set. Whereas with batch gradient descent, a single pass through the training allows you to take only one gradient descent step. With mini-batch gradient descent, a single pass through the training set, that is one epoch, allows you to take 5,000 gradient descent steps.

### Understanding Mini-Batch Gradient Descent

![mini-batch gradient descent2](https://raw.githubusercontent.com/seanliu96/deeplearning.ai/master/COURSE%202%20Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/week2/Optimization%20Algorithms/images/mini-batch%20gradient%20descent2.png)

- If mini-batch size = m, then batch gradient descent.
  - too long per iteration
- If mini-batch size = 1, then stochastic gradient descent
  - lose speedup from vectorizatio
  - more noisy

If small training set, use batch gradient descent (m <= 200, typically 54, 128, 256, 512 …). And make sure all mini-batch fits in CPU/GPU memory.

## Exponentially Weighted Averages

Exponentially weighted averages are faster than gradient descent,  and then we'll use this to build up to more sophisticated optimization algorithms.
$$
v_t = \beta v_{t-1} + (1-\beta) \theta_t, \text{where } v_t \text{ is as approximately averaging} \\
$$

### Understanding Exponentially Weighted Averages

$$
v_t = (1-\beta) \theta_t + (1-\beta) \beta \theta_{t-1} + (1-\beta) \beta^2 \theta_{t-2} + ... + (1-\beta) \beta ^{t-1}\theta_{1}  \\
\text{Because } (1- \varepsilon)^\varepsilon \approx \frac{1}{e}, \text{ so } \frac{1}{1-\beta} \text{ will weight more than } \frac{2}{3}
$$

This is a very efficient way to do so both from computation and memory efficiency point of view which is why it's used in a lot of machine learning.

### Bias Correction

When t is small, v is very small because previous v is very small.

But during this initial phase of learning when you're still warming up your estimates when the bias correction can help you to obtain a better estimate 
$$
v_t = \frac{\beta v_{t-1} + (1-\beta) \theta_t}{1-\beta^t}
$$
In machine learning, for most implementations of the exponential weighted average, people don't often bother to implement bias corrections. Because most people would rather just wait that initial period and have a slightly more biased estimate and go from there. But if you are concerned about the bias during this initial phase, while your exponentially weighted moving average is still warming up. Then bias correction can help you get a better estimate early on.

## Gradient Descent With Momentum

Compute dW, db on current mini-batch

Then compute V
$$
V_{dW} = \beta V_{dW} + (1-\beta) dW \\
V_{db} = \beta V_{db} + (1-\beta) db \\
$$
Then update parameters
$$
W = W - \alpha V_{dW} \\
b = W - \alpha V_{db}
$$
What this does is smooth out steps of gradient descent.
$$
\text{The most common value for } \beta \text{ is 0.9}
$$
With a few iterations you find that the gradient descent with momentum ends up eventually just taking steps that are much smaller oscillations in the vertical direction, but are more directed to just moving quickly in the horizontal direction. And so this allows your algorithm to take a more straightforward path, or to damp out the oscillations in this path to the minimum. 

## RMSprop （Root Mean Squared prop）

Compute dW, db on current mini-batch

Then compute S
$$
S_{dW} = \beta S_{dW} + (1-\beta) dW^2 \\
S_{db} = \beta S_{db} + (1-\beta) db^2 \\
$$
Then update parameters
$$
W = W - \alpha \frac{dW}{\sqrt{S_{dW}}} \\
b = b - \alpha \frac{db}{\sqrt{S_{db}}} \\
$$
The net effect of this is that your up days in the vertical direction are divided by a much larger number, and so that helps damp out the oscillations. Whereas the updates in the horizontal direction are divided by a smaller number. 

And also to make sure that your algorithm doesn't divide by 0

RMSprop, and similar to momentum, has the effects of damping out the oscillations in gradient descent, in mini-batch gradient descent. And allowing you to maybe use a larger learning rate alpha. And certainly speeding up the learning speed of your algorithm. 

## Adam Optimization Algorithm

Adam stands for Adaptive Moment Estimation.

Compute dW, db on current mini-batch

Then compute V with momentum
$$
V_{dW} = \beta_1 V_{dW} + (1-\beta_1) dW \\
V_{db} = \beta_1 V_{db} + (1-\beta_1) db \\
$$
Then compute S with RMSprop
$$
S_{dW} = \beta_2 S_{dW} + (1-\beta_2) dW^2 \\
S_{db} = \beta_2 S_{db} + (1-\beta_2) db^2 \\
$$
Then do bias correction
$$
V_{dW}^{corrected} = \frac{V_{dW}}{1-\beta_1^t} \\
V_{db}^{corrected} = \frac{V_{db}}{1-\beta_1^t} \\
S_{dW}^{corrected} = \frac{S_{dW}}{1-\beta_2^t} \\
S_{db}^{corrected} = \frac{V_{db}}{1-\beta_2^t} \\
$$
Then update parameters
$$
W = W - \alpha \frac{V_{dW}^{corrected}} {\sqrt{S_{dW}^{corrected}} + \varepsilon} \\
b = b - \alpha \frac{V_{db}^{corrected}} {\sqrt{S_{db}^{corrected}} + \varepsilon} 
$$
So this algorithm combines the effect of gradient descient with momentum together with gradient descent with RMSprop

### Hyperparameters Choice

$$
\alpha : \text{needs to be tune} \\
\beta_1: 0.9 (dW) \\
\beta_2: 0.999 (dW^2) \\
\varepsilon: 10^-8
$$

## Learning Rate Decay

One of the things that might help speed up your learning algorithm, is to slowly reduce your learning rate over time. We call this learning rate decay. 
$$
\alpha = \frac{1}{1+decay\_rate * epoch\_num} \alpha_0, \text{ where } decay\_rate \text{ and } \alpha_0 \text{ are hyperparamters}
$$

### Other Leanring Rate Decay

$$
\alpha = 0.95^{epoch\_num} \alpha_0 \\
\alpha = \frac{k}{\sqrt{epoch\_num}} \alpha_0 \\
\alpha = \frac{\alpha_0}{[epoch\_num / 10]}
$$

