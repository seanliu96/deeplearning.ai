## Train/Dev/Test sets

- train set: train model (60% or higher)
- dev set: hold-out cross validation 
- test set: take the best model

Make sure dev set and test set come from same distribution

Not having a test set might be okey

## Bias and Variance

- high bias: underfitting
- just right
- high variance: overfitting

![bias and variance](https://raw.githubusercontent.com/seanliu96/deeplearning.ai/master/COURSE%202%20Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/week1/Setting%20Up%20Your%20Machine%20Learning%20Application/images/bias%20and%20variance.png)

When it comes high bias or high variance, we need to see the optimal error (base error).

## Basic "recipe" for machine learning

- high bias -> bigger network
- high variance -> more data

![recipe](https://raw.githubusercontent.com/seanliu96/deeplearning.ai/master/COURSE%202%20Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/week1/Setting%20Up%20Your%20Machine%20Learning%20Application/images/recipe.png)