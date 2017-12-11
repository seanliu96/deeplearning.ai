### What is neural network?

It is a powerful learning algorithm inspired by how the brain works.

#### Example 1 - single neural network

Given data about the size of houses on the real estate market and you want to fit a function that will
predict their price. It is a linear regression problem because the price as a function of size is a continuous
output.

We know the prices can never be negative so we are creating a function called Rectified Linear Unit (ReLU)
which starts at zero.

![housing price prediction](housing price prediction.png)

The input is the size of the house (x)

The output is the price (y)

The "neuron" implements the function ReLU (blue line)

![single neural network](single neural network.png)

#### Example 2 – Multiple neural network

The price of a house can be affected by other features such as size, number of bedrooms, zip code and
wealth. The role of the neural network is to predicted the price and it will automatically generate the
hidden units. We only need to give the inputs x and the output y.

![housing price prediction2](housing price prediction2.png)



### Supervised learning for Neural Network

In supervised learning, we are given a data set and already know what our correct output should look like,
having the idea that there is a relationship between the input and the output.
Supervised learning problems are categorized into "regression" and "classification" problems. In a
regression problem, we are trying to predict results within a continuous output, meaning that we are
trying to map input variables to some continuous function. In a classification problem, we are instead
trying to predict results in a discrete output. In other words, we are trying to map input variables into
discrete categories.

There are different types of neural network, for example Convolution Neural Network (CNN) used often
for image application and Recurrent Neural Network (RNN) used for one-dimensional sequence data
such as translating English to Chinses or a temporal component such as text transcript. As for the
autonomous driving, it is a hybrid neural network architecture.

### Neural Network examples

![neural network examples](nn examples.png)

### Structured vs unstructured data

Structured data refers to things that has a defined meaning such as price, age whereas unstructured
data refers to thing like pixel, raw audio, text.

![structured vs unstructured data](structured data vs unstructured data.png)

### Why is deep learning taking off?

Deep learning is taking off due to a large amount of **data** available through the digitization of the society, faster **computation** and innovation in the development of neural network **algorithm**.

![scale drives deep learning progress](scale drives deep learning progress.png)

Two things have to be considered to get to the high level of performance:

1. Being able to train a big enough neural network
2. Huge amount of labeled data


The process of training a neural network is iterative.

![idea code experiment](idea code experiment.png)

It could take a good amount of time to train a neural network, which affects your productivity. Faster computation helps to iterate and improve new algorithm.




