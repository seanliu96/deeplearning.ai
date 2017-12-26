## Evaluation Metric

To choose a classifier, a well-defined development set and an evaluation metric speed up the iteration process.  There are different metrics to evaluate the performance of a classifier, they are called evaluation matrices. It is important to note that these evaluation matrices must be evaluated on a training set, a development set or on the test set.

### Single Number Evaluation Metric

| Predicate \| Actual | 1              | 0              |
| ------------------- | -------------- | -------------- |
| 1                   | True Positive  | False Positive |
| 0                   | False Negative | True Negative  |

- Precision
  $$
  \text{Precision(%) }=\frac{\text{True Positive}}{\text{True Positive + False Positive}} \times 100 %
  $$

- Recall
  $$
  <Empty \space Math \space Block>
  $$

  $$
  \text{Recall(%) }=\frac{\text{True Positive}}{\text{True Positive + False Negative}}  \times 100 %
  $$


The problem with using precision/recall as the evaluation metric is that you are not sure which one is better since in this case, both of them have a good precision et recall. 

- F1-score, a harmonic mean, combine both precision and recall
  $$
  \text{F1-score(%) }=\frac{2}{\frac{1}{p} + \frac{1}{r}}
  $$


## Satisficing and Optimizating Metrics

The general rule is

> - 1 optimizing metric (you want to do as well as possible)
> - N - 1 satisficing metrics (you want to be satisfice)

## Train / Dev / Test Distributions

Setting up the training, development and test sets have a huge impact on productivity. It is important to choose the development and test sets from the same distribution and it must be taken randomly from all the data.

The guideline is

>  Choose a development set and test set (from the same distribution) to reflect data you expect to get in the future and consider important to do well.

## Size of Dev and Test Sets

### Old Way of Splitting Data

- training set (70%) + test set (30%)
- traning set (60%) + dev set (20%) + test set (20%)

### Modern Era - Big Data

- traning set (98%) + dev set (1%) + test set (1%)

The guideline is 

> - Set up the size of the test set to give a high confidence in the overall performance of the system
> - Test set helps evaluate the performance of the final classifier which could be less 30% of the whole data set
> - The dev set has to be big enough to valuate different ideas

## When to Change Dev / Test Sets and Metrics

The guideline is 

> 1. Define correctly an evaluation metric that helps better rank order classifiers
> 2. Optimize the evaluation metric

