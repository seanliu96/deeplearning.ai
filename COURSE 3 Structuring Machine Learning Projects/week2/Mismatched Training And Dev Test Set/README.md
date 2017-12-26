## Training and Testing on Different Distributions

### Example

There are two sources of data used to develop the mobile app.

1. small, 10 000 pictures uploaded from the mobile application (not professionally)
2. large, from the web, you downloaded 200 000 pictures

The guideline used is that you have to choose a development set and test set to reflect data you expect to get in the future and consider important to do well.

![different distributions](https://raw.githubusercontent.com/seanliu96/deeplearning.ai/master/COURSE%203%20Structuring%20Machine%20Learning%20Projects/week2/Mismatched%20Training%20And%20Dev%20Test%20Set/images/different%20distributions.png)

The advantage of this way of splitting up is that the target is well defined.

The disadvantage is that the training distribution is different from the development and test set distributions. 

## Bias and Variance with Mismathed Data Distributions

When the training set, development and test sets distributions are different, we have a mismatch data problem.

> training-development set has the same distribution as the training set but it is not used for training the neural network

Bayes error <—(avoidable bias)—>traning set error<—(variance)—>development-training set error<—(data mismatched)—>development set error<—(degree of overfitting to the development set)—>test set error

## Addressing Data Mismatch

The guideline is

- perform manual error analysis to understand the error differences between traning, development/test sets. Development should never be done on test set to avoid overfitting
- Make training data or collect data similar to development and test sets (normal data + noise = synthesized data).