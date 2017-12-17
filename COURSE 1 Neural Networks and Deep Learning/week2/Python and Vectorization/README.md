## Vectorization

### Logistic Regression Derivatives

![logistic regression derivatives](https://raw.githubusercontent.com/seanliu96/deeplearning.ai/master/COURSE%201%20Neural%20Networks%20and%20Deep%20Learning/week2/Python%20and%20Vectorization/images/logistic%20regression%20derivatives.png)

## Vectorizing Logistic Regression

$$
X = [x^{(1)}, x^{(2)}, ..., x^{(m)}] \\
Y = [y^{(1)}, y^{(2)}, ..., y^{(m)}] \\
Z = [z^{(1)}, z^{(2)}, ..., z^{(m)}] \\
A = [a^{(1)}, a^{(2)}, ..., a^{(m)}] = \sigma(Z) \\
$$

### Implementing Logistic Regression

![implementing logistic regression](https://raw.githubusercontent.com/seanliu96/deeplearning.ai/master/COURSE%201%20Neural%20Networks%20and%20Deep%20Learning/week2/Python%20and%20Vectorization/images/implementing%20logistic%20regression.png)

## Broadcasting in Python

### General Principle

$$
(m, n) [+-*/] (1, n) \rightarrow (m, n) [+-*/] (m, n) \\
(m, n) [+-*/] (m, 1) \rightarrow (m, n) [+-*/] (m, n)
$$

