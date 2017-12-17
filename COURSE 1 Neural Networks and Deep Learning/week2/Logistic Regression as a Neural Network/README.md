## Binary Classification

In a binary classification problem, the result is a discrete value output

### Notation

1. a training example:
   $$
   (x,y), x\in \mathbb{R}^{n_x}, y\in \{0, 1\}
   $$

2. m training examples:
   $$
   \{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), ..., (x^{(m)}, y^{(m)})\} \\
   m = m_{\text{train}} = \text{# of train examples}
   $$

3. matrix:
   $$
   X = [x^{(1)},x^{(2)}, ..., x^{(m)}] \in \mathbb{R}^{n_x \times m} \\
   Y = [y^{(1)},y^{(2)}, ..., y^{(m)}] \in \mathbb{R}^{1 \times m} \\
   $$

4. goal:
   $$
   \text{Given } x, \hat{y}=P(y=1|x), \text{where }0 \leq \hat{y}
   $$



## Logistic Regression

### parameters

1. The input features vector:

   $$
   x \in \mathbb{R}^{n_x}, \text{where } n_x \text{ is the number of features}
   $$

2. The training label:
   $$
   y \in \{0,1\}
   $$

3. The weights:
   $$
   w \in \mathbb{R}^{n_X}, \text{where } n_x \text{ is the number of features}
   $$

4. The threshold:
   $$
   b \in \mathbb{R}
   $$

5. The output:
   $$
   \hat{y} = \sigma(w^Tx+b)
   $$

6. Sigmoid function:
   $$
   s  = \sigma(w^tx+b)=\sigma(z)= \frac{1}{1 + e^{-z}}
   $$



### Loss (error)​ function:

$$
\ell(\hat{y}, y) = -(y \log(\hat{y}) + (1-y) \log(1-\hat{y}))
$$

### Cost function:

$$
J(w, b) = \frac{1}{m} \sum_{i=1}^m{\ell(\hat{y}^{(i)}, y^{(i)})}=-\frac{1}{m} \sum_{i=1}^m{(y ^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)}) \log(1-\hat{y}^{(i)}))}
$$

## Gradient Descent

Want to find w and b that minimize J(w, b)

### Process

Repeat
$$
w := w - \alpha \frac{\partial J(w, b)}{\partial{w}} \\
b := b - \alpha \frac{\partial J(b, w)}{\partial{b}}
$$

## Logistic Regression Gradient Descent

Recap
$$
z = w^Tx+b \\
\hat{y} = a = \sigma(z) \\
\ell (a, y) =  -(y \log(a) + (1-y) \log(1-a))
$$

### Gradient Descent

$$
dz = \frac{\partial{\ell}}{\partial{z}} = a-y=a(1-a) \\
dw_1 = \frac{\partial{\ell}}{\partial{w_1}} = x_1 \cdot dz \\
dw_2 = \frac{\partial{\ell}}{\partial{w_2}} = x_2 \cdot dz \\
... \\
db = \frac{\partial{\ell}}{\partial{b}} = dz
$$

### Process

$$
w_1 := w_1 - \alpha dw_1 \\
w_2 := w_2 - \alpha dw_2 \\
... \\
b := b - \alpha db
$$

### Gradient Descent on m examples

Recap
$$
J(w, b) = \frac{1}{m} \sum_{i=1}^m{\ell(a^{(i)}, y^{(i)})}=-\frac{1}{m} \sum_{i=1}^m{(y ^{(i)}\log(a^{(i)}) + (1-y^{(i)}) \log(1-a^{(i)}))} \\
a^{(i)} = y^{(i)} = \sigma(z^{(i)}) = \sigma(w^Tx+b) \\
$$
Descent
$$
dz^{(i)} = \frac{\partial{\ell}}{\partial{z^{(i)}}} = a^{(i)}-y^{(i)} \\
dw_1 = \frac{1}{m} \sum_{i=1}^m{\frac{\partial{\ell}}{\partial{w_1}} }= \frac{1}{m} \sum_{i=1}^m{{x_1 \cdot dz^{(i)}}} \\
dw_2 = \frac{1}{m} \sum_{i=1}^m{\frac{\partial{\ell}}{\partial{w_2}} }= \frac{1}{m} \sum_{i=1}^m{{x_2 \cdot dz^{(i)}}} \\
... \\
db = \frac{1}{m} \sum_{i=1}^m{\frac{\partial{\ell}}{\partial{b}}} = \frac{1}{m} \sum_{i=1}^m{dz^{(i)}}
$$

###  Pseudocode

![pseudocode](https://raw.githubusercontent.com/seanliu96/deeplearning.ai/master/COURSE%201%20Neural%20Networks%20and%20Deep%20Learning/week2/Logistic%20Regression%20as%20a%20Neural%20Network/images/pseudocode.png)